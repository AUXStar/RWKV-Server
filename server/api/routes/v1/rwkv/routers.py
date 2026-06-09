from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
import time
import uuid
import json
from typing import AsyncGenerator

from .....scheduler import BaseScheduler
from .....task.manager import get_task_manager
from .....task.task import Task, Status
from .....utils import finish_callback, stream_callback
from ....dependencies import get_scheduler
from .schemas import (
    TaskResponseModel,
    TaskCreate,
    TaskUpdate,
    DataFrame,
)

router = APIRouter(prefix="/tasks", tags=["rwkv"])
task_manager = get_task_manager()


def gen_id(tmp=False):
    if tmp:
        return f"TMP_{uuid.uuid4().hex}"
    return f"TASK_{uuid.uuid4().hex}"


@router.post("/tmp", response_model=TaskResponseModel)
async def tmp_task(data: TaskCreate, scheduler: BaseScheduler = Depends(get_scheduler)):
    task_id = gen_id(True)
    return await create(task_id, data, scheduler)


async def _stream_sse(
    task: Task, task_id: str, future, prefill_time: float
) -> AsyncGenerator[bytes, None]:
    yield f"data: {json.dumps({'task_id': task_id, 'prefill_time': prefill_time})}\n\n".encode()

    gen_time = time.time()
    async for chunk_toks in future:
        chunk = task.model_loader.decode(chunk_toks)
        df = DataFrame(
            data=chunk,
            task_id=task_id,
            gen_time=time.time() - gen_time,
            speed=(
                len(chunk_toks) / (time.time() - gen_time)
                if time.time() - gen_time > 0
                else 0
            ),
        )
        yield f"data: {df.model_dump_json()}\n\n".encode()
        gen_time = time.time()

    yield b"data: [DONE]\n\n"


@router.post("/create")
async def create_task(
    data: TaskCreate, scheduler: BaseScheduler = Depends(get_scheduler)
):

    task_id = gen_id(False)
    return await create(task_id, data, scheduler)


async def create(task_id:str, data: TaskCreate, scheduler: BaseScheduler):
    future, collect, finish = stream_callback()
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

    task_manager.put_task(task_id, task)

    if not data.stream:
        return TaskResponseModel(
            task_id=task_id,
            result="",
            prefill_time=task.prefill_time,
            gen_time=0,
            speed=0,
            finished=False,
        )

    return StreamingResponse(
        _stream_sse(task, task_id, future, task.prefill_time),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{task_id}/get_result", response_model=TaskResponseModel)
async def get_result(
    task_id: str,
    scheduler: BaseScheduler = Depends(get_scheduler),
):
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    toks = task.pop_tokens()
    result = task.model_loader.decode(toks)
    return TaskResponseModel(
        task_id=task_id,
        result=result,
        prefill_time=task.prefill_time,
        gen_time=max(0, task.finish_time - task.run_time),
        speed=scheduler.per_speed,
        finished=(task.status == Status.FINISHED),
    )


@router.post("/{task_id}/fork", response_model=TaskResponseModel)
async def fork(
    task_id: str,
    data: TaskUpdate,
    tmp: bool = False,
    scheduler: BaseScheduler = Depends(get_scheduler),
):
    source_task = task_manager.get_task(task_id)
    if not source_task:
        raise HTTPException(status_code=404, detail="Source task not found")

    new_task_id = task_manager.fork_template(task_id, gen_id(tmp))
    return await continue_task(new_task_id, data, scheduler)


@router.post("/{task_id}/continue")
async def continue_task(
    task_id: str,
    data: TaskUpdate,
    scheduler: BaseScheduler = Depends(get_scheduler),
):
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if data.prompt is not None:
        task.prefill(data.prompt)

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
            prefill_time=task.prefill_time,
            gen_time=0,
            speed=0,
            finished=False,
        )

    return StreamingResponse(
        _stream_sse(task, task_id, future, task.prefill_time),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/{task_id}/stop")
async def stop(task_id: str):
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.status != Status.FINISHED:
        task.stop()
    return {"stopped": task.status == Status.FINISHED}


@router.post("/{task_id}/as_template", response_model=TaskResponseModel)
async def convert_to_template(task_id: str):
    source_task = task_manager.get_task(task_id)
    if not source_task:
        raise HTTPException(status_code=404, detail="Task not found")
    if source_task.task_id.startswith("_"):
        raise HTTPException(status_code=400, detail="Task is already a template")

    new_template_id = f"_TMPL_{uuid.uuid4().hex}"

    task_manager.fork_template(task_id, new_task_id=new_template_id)
    return TaskResponseModel(
        task_id=new_template_id,
        result="",
        prefill_time=0,
        gen_time=0,
        speed=0,
        finished=False,
    )


@router.post("/{task_id}/delete")
async def delete(task_id: str, force: bool = False):
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if not force and task.task_id.startswith("_"):
        raise HTTPException(
            status_code=403, detail="Cannot delete template without force=true"
        )
    if task.status != Status.FINISHED:
        task.stop()
    task_manager.delete_task_from_any_level(task_id, force=force)
    return {"deleted": True}


@router.get("/list")
async def list_tasks():
    status = task_manager.list_all_tasks()

    return {
        "cpu_cache_count": len(status["cpu_cache"]),
        "database_count": len(status["database"]),
        "total_count": status["total_count"],
        "cpu_tasks": status["cpu_cache"],
        "db_tasks": status["database"][:100],
    }
