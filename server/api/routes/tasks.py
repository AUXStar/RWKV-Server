from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
import time, uuid, json, copy
from typing import AsyncGenerator

from ...scheduler import BaseScheduler
from ...task import Task, Status
from ...utils import finish_callback, stream_callback
from ..dependencies import get_scheduler
from ..schemas import (
    SingleTaskModel,
    TaskResponseModel,
    NormalTaskCreate,
    NormalTaskUpdate,
    DataFrame,
)

router = APIRouter(prefix="/tasks")
tasks: dict[str, Task] = {}


@router.post("/single", response_model=TaskResponseModel)
async def single(
    data: SingleTaskModel, scheduler: BaseScheduler = Depends(get_scheduler)
):
    future, callback = finish_callback()
    prefill_time = time.time()
    task = scheduler.new_task(
        prompt=data.prompt,
        max_tokens=data.max_tokens,
        presence_penalty=data.presence_penalty,
        repetition_penalty=data.repetition_penalty,
        penalty_decay=data.penalty_decay,
        temperature=data.temperature,
        top_k=data.top_k,
        top_p=data.top_p,
        seed=data.seed,
        finish_callback=callback,
    )
    prefill_time = time.time() - prefill_time

    gen_time = time.time()
    result = await future
    gen_time = time.time() - gen_time
    speed = len(result[0]) / gen_time if gen_time > 0 else 0
    result = task.model_loader.decode(result[0])
    return TaskResponseModel(
        result=result,
        prefill_time=prefill_time,
        gen_time=gen_time,
        speed=speed,
    )


async def _stream_sse(
    task: Task, task_id: str, future, prefill_time: float
) -> AsyncGenerator[bytes, None]:

    yield f"data: {json.dumps({'task_id': task_id, 'prefill_time': prefill_time})}\n\n".encode()

    gen_time = time.time()
    async for chunk in future:
        chunk = task.model_loader.decode(chunk)
        df = DataFrame(
            data=chunk,
            task_id=task_id,
            gen_time=time.time() - gen_time,
            speed=(
                len(chunk) / (time.time() - gen_time)
                if time.time() - gen_time > 0
                else 0
            ),
        )
        yield f"data: {df.model_dump_json()}\n\n".encode()
        gen_time = time.time()

    yield b"data: [DONE]\n\n"


@router.post("/create")
async def create(
    data: NormalTaskCreate, scheduler: BaseScheduler = Depends(get_scheduler)
):
    future, collect, finish = stream_callback()

    prefill_time = time.time()
    task = scheduler.new_task(
        prompt=data.prompt,
        max_tokens=data.max_tokens,
        presence_penalty=data.presence_penalty,
        repetition_penalty=data.repetition_penalty,
        penalty_decay=data.penalty_decay,
        temperature=data.temperature,
        top_k=data.top_k,
        top_p=data.top_p,
        seed=data.seed,
        collect_callback=collect,
        finish_callback=finish,
    )
    prefill_time = time.time() - prefill_time

    task_id = f"TASK_{uuid.uuid4().hex}"
    tasks[task_id] = task

    if not data.stream:
        return TaskResponseModel(
            task_id=task_id,
            result="",
            prefill_time=prefill_time,
            gen_time=0,
            speed=0,
            finished=False,
        )

    return StreamingResponse(
        _stream_sse(task, task_id, future, prefill_time),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/{task_id}/get_result", response_model=TaskResponseModel)
async def get_result(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    length = len(task.generated_tokens)
    if length == 0:
        return TaskResponseModel(
            task_id=task_id,
            result="",
            prefill_time=0,
            gen_time=0,
            speed=0,
            finished=(task.status == Status.FINISHED),
        )

    toks = task.generated_tokens[:length]
    task.generated_tokens = task.generated_tokens[length:]

    result = task.model_loader.decode(toks)
    return TaskResponseModel(
        task_id=task_id,
        result=result,
        prefill_time=0,
        gen_time=0,
        speed=0,
        finished=(task.status == Status.FINISHED),
    )


@router.post("/{task_id}/fork", response_model=TaskResponseModel)
async def fork(
    task_id: str,
    data: NormalTaskUpdate,
    scheduler: BaseScheduler = Depends(get_scheduler),
):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    task_id = f"TASK_{uuid.uuid4().hex}"
    tasks[task_id] = copy.deepcopy(task)

    return await continue_task(task_id, data, scheduler)


@router.post("/{task_id}/continue")
async def continue_task(
    task_id: str,
    data: NormalTaskUpdate,
    scheduler: BaseScheduler = Depends(get_scheduler),
):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    prefill_time = time.time()
    if data.prompt is not None:
        task._sync_prefill(data.prompt)
    prefill_time = time.time() - prefill_time

    for field in [
        "max_tokens",
        "temperature",
        "top_k",
        "top_p",
        "presence_penalty",
        "repetition_penalty",
        "penalty_decay",
    ]:
        if (value := getattr(data, field)) is not None:
            setattr(task, field, value)

    future, collect, finish = stream_callback()
    task.collect_callback = collect
    task.finish_callback = finish
    scheduler.add_task(task)

    if not data.stream:
        return TaskResponseModel(
            task_id=task_id,
            result="",
            prefill_time=prefill_time,
            gen_time=0,
            speed=0,
            finished=False,
        )

    return StreamingResponse(
        _stream_sse(task, task_id, future, prefill_time),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/{task_id}/stop")
async def stop(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.status != Status.FINISHED:
        task.stop()
    
    return {"stopped": task.status == Status.FINISHED}

@router.post("/{task_id}/delete")
async def delete(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.status != Status.FINISHED:
        task.stop()
    del tasks[task_id]
    return {"deleted": True}