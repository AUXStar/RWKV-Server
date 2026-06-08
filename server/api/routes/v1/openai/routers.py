from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
import time
import uuid
import asyncio
import copy
from typing import AsyncGenerator, List, Optional, Any, Tuple
from loguru import logger

from .....scheduler import BaseScheduler
from .....task.task import Task, Status
from ....dependencies import get_scheduler
from .....utils import finish_callback, stream_callback

from .schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    ModelsResponse,
    Model,
    Choice,
    ChunkChoice,
    Delta,
    CompletionChoice,
    CompletionChunk,
    CompletionChunkChoice,
    CompletionRequest,
    CompletionResponse,
)
from .utils import (
    generate_chat_id,
    generate_completion_id,
    messages_to_prompt,
    calculate_usage,
    calculate_completion_usage,
    normalize_stop_sequences,
    parse_tool_call,
)

log = logger.bind(module="api.openai")

router = APIRouter(tags=["openai"])

MODEL_NAME = "rwkv-7-g1g-2.9b"
MODEL_CREATED_TIME = int(time.time())
DEFAULT_PENALTY_DECAY = 0.994
DEFAULT_TOP_K = 20


class TaskStreamHandler:
    """管理单个流式任务的 token 处理、stop 序列检测和工具调用解析"""

    def __init__(self, task: Task, stop_sequences: List[str], index: int, output_queue: asyncio.Queue):
        self.task = task
        self.stop_seqs = stop_sequences
        self.index = index
        self.output_queue = output_queue
        self.buffer = ""               # 累积的文本（用于检测标签）
        self.tool_state = 0            # 0=正常, 1=检测到'<', 2=在<tool_call>内部
        self.generated_tokens = []     # 已生成的所有 token
        self.loop = asyncio.get_running_loop()

    def on_tokens(self, tokens: List[int]) -> None:
        """由 task.collect_callback 调用，处理每个 token 块"""
        self.generated_tokens.extend(tokens)
        # 立即解码新增的文本
        delta_text = self.task.model_loader.decode(tokens)
        self._process_text(delta_text)

    def _process_text(self, delta_text: str) -> None:
        """处理新增的文本片段，包含 stop 检测和 tool call 状态机"""
        # 合并缓冲区和新文本
        combined = self.buffer + delta_text

        # 1. 优先检查 stop 序列
        for stop in self.stop_seqs:
            if stop in combined:
                # 截断到 stop 之前
                before_stop = combined.split(stop)[0]
                if before_stop:
                    # 发送 stop 之前剩余的文本
                    self._emit_text(before_stop)
                # 发送结束标记
                self._emit_finish("stop")
                self.task.stop()
                return

        # 2. 状态机处理 tool call
        self.buffer += delta_text

        if self.tool_state == 0:                     # 正常模式
            if "<" in self.buffer:
                self.tool_state = 1                  # 可能开始标签，暂不输出
                # 不发送任何内容，继续累积
            else:
                # 没有特殊标记，直接输出
                self._emit_text(self.buffer)
                self.buffer = ""

        elif self.tool_state == 1:                   # 已看到 '<'，等待 <tool_call>
            if "<tool_call>" in self.buffer:
                # 确认是有效标签，进入标签内模式，丢弃已累积内容
                self.tool_state = 2
                self.buffer = ""                     # 丢弃整个标签本身
            elif "<" in self.buffer and "<tool_call>" not in self.buffer:
                # 假阳性：比如 "<?xml" 但不是 <tool_call>
                self._emit_text(self.buffer)          # 把缓存全部吐出
                self.buffer = ""
                self.tool_state = 0
            # 否则继续等待

        elif self.tool_state == 2:                   # 在 <tool_call> ... </tool_call> 内部
            if "</tool_call>" in self.buffer:
                # 结束标签出现，解析整个工具调用
                content, tool_calls = parse_tool_call(self.buffer)
                if content:
                    self._emit_text(content)
                for tc in tool_calls:
                    self._emit_tool_call(tc)
                self.buffer = ""
                self.tool_state = 0
                self.task.stop()
                self._emit_finish("tool_calls")
            # 否则继续累积，不输出任何文本

    def _emit_text(self, text: str) -> None:
        """发送普通文本事件"""
        if not text:
            return
        self.output_queue.put_nowait(("text", self.index, text))

    def _emit_tool_call(self, tool_call) -> None:
        """发送工具调用事件"""
        chunk = {
            "id": tool_call.id,
            "type": "function",
            "function": {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
            }
        }
        log.info(chunk)
        # 注意：OpenAI 要求 tool_calls 是一个列表，每个元素包含 index 字段
        self.output_queue.put_nowait(("tool_call", self.index, [{"index": 0, **chunk}]))

    def _emit_finish(self, reason: str) -> None:
        """发送任务结束事件"""
        self.output_queue.put_nowait(("finish", self.index, reason))

    def on_finish(self) -> None:
        """任务完成时调用，清理可能遗留的缓冲区"""
        if self.buffer and self.tool_state != 2:
            self._emit_text(self.buffer)
        # 如果还没有发送 finish，补一个 stop
        self._emit_finish("stop")


async def _merge_chat_tasks(
    tasks: List[Task],
    stop_sequences: List[str],
    scheduler: BaseScheduler,
) -> AsyncGenerator[Tuple[str, int, Any], None]:
    """
    合并多个任务的输出，产出统一事件流。
    事件类型：
      - ("text", index, text)
      - ("tool_call", index, tool_call_chunk_list)
      - ("finish", index, finish_reason)
    """
    output_queue = asyncio.Queue()
    handlers = []

    # 启动所有任务
    for idx, task in enumerate(tasks):
        handler = TaskStreamHandler(task, stop_sequences, idx, output_queue)
        handlers.append(handler)

        # 包装回调，使用 lambda 绑定当前 handler 和原生的 finish_callback
        def make_collect(h):
            return lambda tokens: h.on_tokens(tokens)

        def make_finish(h):
            return lambda _: h.on_finish()

        task.collect_callback = make_collect(handler)
        task.finish_callback = make_finish(handler)
        scheduler.add_task(task)

    # 生产者已就绪，现在作为消费者读取 output_queue
    finished = [False] * len(tasks)
    finished_count = 0
    while finished_count < len(tasks):
        typ, idx, data = await output_queue.get()
        if typ == "finish":
            if not finished[idx]:
                finished[idx] = True
                finished_count += 1
        yield (typ, idx, data)


async def _handle_chat_non_streaming(
    tasks: List[Task],
    prompt: str,
    stop_sequences: List[str],
    scheduler: BaseScheduler,
) -> ChatCompletionResponse:
    """非流式聊天：收集所有事件后构建完整响应"""
    chat_id = generate_chat_id()
    created_time = int(time.time())

    # 暂存每个 choice 的内容和 finish_reason
    contents = [""] * len(tasks)
    tool_calls_list = [[] for _ in range(len(tasks))]
    finish_reasons = ["stop"] * len(tasks)

    async for typ, idx, data in _merge_chat_tasks(tasks, stop_sequences, scheduler):
        if typ == "text":
            contents[idx] += data
        elif typ == "tool_call":
            tool_calls_list[idx].extend(data)   # data 是列表
        elif typ == "finish":
            finish_reasons[idx] = data

    # 构建 choices
    total_usage = None
    choices = []
    for idx, task in enumerate(tasks):
        content = contents[idx]
        tool_calls = tool_calls_list[idx] if tool_calls_list[idx] else None
        finish_reason = finish_reasons[idx]

        # 计算 usage
        usage = calculate_usage(task, prompt)
        if total_usage is None:
            total_usage = usage
        else:
            total_usage.completion_tokens += usage.completion_tokens
            total_usage.total_tokens += usage.completion_tokens

        choices.append(
            Choice(
                index=idx,
                message={"role": "assistant", "content": content, "tool_calls": tool_calls},
                finish_reason=finish_reason,
            )
        )

    return ChatCompletionResponse(
        id=chat_id,
        created=created_time,
        model=MODEL_NAME,
        choices=choices,
        usage=total_usage,
    )


async def _handle_chat_streaming(
    request: Request,
    tasks: List[Task],
    prompt: str,
    stop_sequences: List[str],
    scheduler: BaseScheduler,
    stream_options: Optional[dict] = None,
) -> StreamingResponse:
    """流式聊天：将事件流转换为 SSE 格式"""
    chat_id = generate_chat_id()
    created_time = int(time.time())

    async def sse_generator() -> AsyncGenerator[bytes, None]:
        # 1. 先发送每个 choice 的 role 前缀
        for idx in range(len(tasks)):
            chunk = ChatCompletionChunk(
                id=chat_id,
                created=created_time,
                model=MODEL_NAME,
                choices=[ChunkChoice(index=idx, delta=Delta(role="assistant"), finish_reason=None)],
            )
            yield f"data: {chunk.model_dump_json()}\n\n".encode()

        # 2. 处理实际事件
        finished = [False] * len(tasks)
        finished_count = 0
        async for typ, idx, data in _merge_chat_tasks(tasks, stop_sequences, scheduler):
            if typ == "text":
                chunk = ChatCompletionChunk(
                    id=chat_id,
                    created=created_time,
                    model=MODEL_NAME,
                    choices=[ChunkChoice(index=idx, delta=Delta(content=data), finish_reason=None)],
                )
                yield f"data: {chunk.model_dump_json()}\n\n".encode()
            elif typ == "tool_call":
                chunk = ChatCompletionChunk(
                    id=chat_id,
                    created=created_time,
                    model=MODEL_NAME,
                    choices=[ChunkChoice(index=idx, delta=Delta(tool_calls=data), finish_reason="tool_calls")],
                )
                yield f"data: {chunk.model_dump_json()}\n\n".encode()
            elif typ == "finish":
                if not finished[idx]:
                    finished[idx] = True
                    finished_count += 1
                # 可忽略 finish 事件，最后统一发送 stop chunk
            # 检查客户端断开
            if await request.is_disconnected():
                break

        # 3. 发送每个 choice 的 stop chunk
        for idx in range(len(tasks)):
            chunk = ChatCompletionChunk(
                id=chat_id,
                created=created_time,
                model=MODEL_NAME,
                choices=[ChunkChoice(index=idx, delta=Delta(), finish_reason="stop")],
            )
            yield f"data: {chunk.model_dump_json()}\n\n".encode()

        # 4. 如果需要 usage，计算并发送
        if stream_options and stream_options.get("include_usage"):
            total_usage = None
            for task in tasks:
                usage = calculate_usage(task, prompt)
                if total_usage is None:
                    total_usage = usage
                else:
                    total_usage.completion_tokens += usage.completion_tokens
                    total_usage.total_tokens += usage.completion_tokens
            usage_chunk = ChatCompletionChunk(
                id=chat_id,
                created=created_time,
                model=MODEL_NAME,
                choices=[],
                usage=total_usage,
            )
            yield f"data: {usage_chunk.model_dump_json()}\n\n".encode()

        yield b"data: [DONE]\n\n"

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _apply_stop_sequences(text: str, stop_sequences: List[str]) -> Tuple[str, str]:
    """在文本中查找 stop 序列，返回 (截断后的文本, finish_reason)"""
    for stop in stop_sequences:
        if stop in text:
            return text.split(stop)[0], "stop"
    return text, "stop"


async def _handle_completion_non_streaming(
    task_items: List[Tuple[str, Task]],
    stop_sequences: List[str],
    scheduler: BaseScheduler,
    echo: bool,
) -> CompletionResponse:
    """非流式文本补全"""
    completion_id = generate_completion_id()
    created_time = int(time.time())

    futures = []
    for prompt, task in task_items:
        future, callback = finish_callback()
        task.finish_callback = callback
        scheduler.add_task(task)
        futures.append((prompt, task, future))

    choices = []
    total_usage = None

    for idx, (prompt, task, future) in enumerate(futures):
        await future
        result_text = task.model_loader.decode(task.generated_tokens)
        result_text, finish_reason = _apply_stop_sequences(result_text, stop_sequences)

        if echo:
            result_text = prompt + result_text

        usage = calculate_completion_usage(task, prompt, echo)
        if total_usage is None:
            total_usage = usage
        else:
            total_usage.completion_tokens += usage.completion_tokens
            total_usage.total_tokens += usage.completion_tokens

        choices.append(CompletionChoice(index=idx, text=result_text, finish_reason=finish_reason))

    return CompletionResponse(
        id=completion_id,
        created=created_time,
        model=MODEL_NAME,
        choices=choices,
        usage=total_usage,
    )


async def _handle_completion_streaming(
    request: Request,
    task_items: List[Tuple[str, Task]],
    stop_sequences: List[str],
    scheduler: BaseScheduler,
    echo: bool,
    stream_options: Optional[dict],
) -> StreamingResponse:
    """流式文本补全"""
    completion_id = generate_completion_id()
    created_time = int(time.time())
    output_queue = asyncio.Queue()
    all_tasks = []
    prompt_map = {}

    async def consumer(generator, idx: int):
        try:
            async for item in generator:
                if item is None:
                    break
                await output_queue.put((idx, item))
        finally:
            await output_queue.put((idx, None))

    for idx, (prompt, task) in enumerate(task_items):
        generator, collect, finish = stream_callback()
        original_collect = collect

        def wrapped_collect(tokens, task=task, idx=idx, orig=original_collect):
            # 累积 tokens 并检测 stop
            if not hasattr(task, "_stream_buffer"):
                task._stream_buffer = []
            task._stream_buffer.extend(tokens)
            full_text = task.model_loader.decode(task._stream_buffer)
            for stop in stop_sequences:
                if stop in full_text:
                    stop_pos = full_text.find(stop)
                    before = full_text[:stop_pos]
                    # 计算需要发送的增量
                    prev_text = task.model_loader.decode(task._stream_buffer[:-len(tokens)]) if len(task._stream_buffer) > len(tokens) else ""
                    delta = before[len(prev_text):]
                    if delta:
                        if hasattr(task.model_loader, "tokenize"):
                            delta_tokens = task.model_loader.tokenize(delta)
                            orig(delta_tokens)
                        else:
                            orig(tokens)
                    task.stop()
                    return
            orig(tokens)

        def wrapped_finish(_):
            finish(None)

        task.collect_callback = wrapped_collect
        task.finish_callback = wrapped_finish
        all_tasks.append(task)
        prompt_map[idx] = prompt
        scheduler.add_task(task)
        asyncio.create_task(consumer(generator, idx))

    async def sse_generator():
        # 如果 echo=True，先发送每个 prompt
        if echo:
            for idx, prompt in prompt_map.items():
                chunk = CompletionChunk(
                    id=completion_id,
                    created=created_time,
                    model=MODEL_NAME,
                    choices=[CompletionChunkChoice(index=idx, text=prompt, finish_reason=None)],
                )
                yield f"data: {chunk.model_dump_json()}\n\n".encode()

        finished_count = 0
        while finished_count < len(task_items):
            if await request.is_disconnected():
                break
            try:
                idx, tokens = await asyncio.wait_for(output_queue.get(), timeout=1.0)
                if tokens is None:
                    finished_count += 1
                    continue
                text = all_tasks[idx].model_loader.decode(tokens)
                chunk = CompletionChunk(
                    id=completion_id,
                    created=created_time,
                    model=MODEL_NAME,
                    choices=[CompletionChunkChoice(index=idx, text=text, finish_reason=None)],
                )
                yield f"data: {chunk.model_dump_json()}\n\n".encode()
            except asyncio.TimeoutError:
                continue

        # 发送结束
        for idx in range(len(task_items)):
            chunk = CompletionChunk(
                id=completion_id,
                created=created_time,
                model=MODEL_NAME,
                choices=[CompletionChunkChoice(index=idx, text="", finish_reason="stop")],
            )
            yield f"data: {chunk.model_dump_json()}\n\n".encode()

        if stream_options and stream_options.get("include_usage"):
            total_usage = None
            for idx, (prompt, task) in enumerate(task_items):
                usage = calculate_completion_usage(task, prompt, echo)
                if total_usage is None:
                    total_usage = usage
                else:
                    total_usage.completion_tokens += usage.completion_tokens
                    total_usage.total_tokens += usage.completion_tokens
            usage_chunk = CompletionChunk(
                id=completion_id,
                created=created_time,
                model=MODEL_NAME,
                choices=[],
                usage=total_usage,
            )
            yield f"data: {usage_chunk.model_dump_json()}\n\n".encode()

        yield b"data: [DONE]\n\n"

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/models", response_model=ModelsResponse)
async def list_models():
    return ModelsResponse(data=[Model(id=MODEL_NAME, created=MODEL_CREATED_TIME)])


@router.post("/chat/completions")
async def chat_completions(
    request: Request,
    data: ChatCompletionRequest,
    scheduler: BaseScheduler = Depends(get_scheduler),
):
    prompt = messages_to_prompt(data.messages)
    stop_sequences = normalize_stop_sequences(data.stop)
    repetition_penalty = data.frequency_penalty if data.frequency_penalty != 0 else 0

    try:
        base_task = Task(
            prompt=prompt,
            model_loader=scheduler.model_loader,
            batch_sampler=scheduler.sampler,
            max_tokens=data.max_tokens,
            presence_penalty=data.presence_penalty,
            repetition_penalty=repetition_penalty,
            penalty_decay=DEFAULT_PENALTY_DECAY,
            temperature=data.temperature,
            top_k=DEFAULT_TOP_K,
            top_p=data.top_p,
            seed=data.seed,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create task: {str(e)}")

    tasks = [base_task]
    for i in range(1, data.n):
        task = copy.copy(base_task)
        task.task_id = f"TASK_{uuid.uuid4().hex}"
        task.rand_state += i
        tasks.append(task)

    if not data.stream:
        return await _handle_chat_non_streaming(tasks, prompt, stop_sequences, scheduler)
    else:
        return await _handle_chat_streaming(
            request, tasks, prompt, stop_sequences, scheduler, data.stream_options
        )


@router.post("/completions")
async def completions(
    request: Request,
    data: CompletionRequest,
    scheduler: BaseScheduler = Depends(get_scheduler),
):
    stop_sequences = normalize_stop_sequences(data.stop)
    repetition_penalty = data.frequency_penalty if data.frequency_penalty != 0 else 0
    prompts = data.prompt if isinstance(data.prompt, list) else [data.prompt]

    all_items = []
    for prompt in prompts:
        try:
            base_task = Task(
                prompt=prompt,
                model_loader=scheduler.model_loader,
                batch_sampler=scheduler.sampler,
                max_tokens=data.max_tokens,
                presence_penalty=data.presence_penalty,
                repetition_penalty=repetition_penalty,
                penalty_decay=DEFAULT_PENALTY_DECAY,
                temperature=data.temperature,
                top_k=DEFAULT_TOP_K,
                top_p=data.top_p,
                seed=data.seed,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create task: {str(e)}")

        for i in range(data.n):
            if i == 0:
                all_items.append((prompt, base_task))
            else:
                task = copy.copy(base_task)
                task.task_id = f"TASK_{uuid.uuid4().hex}"
                task.rand_state += i
                all_items.append((prompt, task))

    if not data.stream:
        return await _handle_completion_non_streaming(all_items, stop_sequences, scheduler, data.echo)
    else:
        return await _handle_completion_streaming(
            request, all_items, stop_sequences, scheduler, data.echo, data.stream_options
        )