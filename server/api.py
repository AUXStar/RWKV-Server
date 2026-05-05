from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union, Iterator
import asyncio
import threading
import time
import json
import uuid
import queue

from . import setup_logging
from .scheduler import RWKV070ModelLoader, DynamicScheduler

# ---------- 初始化模型与调度器 ----------
model = RWKV070ModelLoader(
    "/home/njzy/rwkv_agent/server/model/rwkv7-g1f-2.9b-20260420-ctx8192"
)
manager = DynamicScheduler(model, 35, 15)
manager.start_daemon()
setup_logging()

app = FastAPI(title="RWKV OpenAI-Compatible API")

# ---------- OpenAI 标准请求/响应模型 ----------
class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str = "rwkv-7b"  # 可自定义
    messages: List[ChatMessage]
    max_tokens: int = Field(50, ge=1, le=512)
    temperature: float = Field(0.3, ge=0.0, le=2.0)
    top_p: float = Field(0.1, ge=0.0, le=1.0)
    n: int = Field(1, ge=1, le=5)
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0)  # 映射到 repetition_penalty
    logit_bias: Optional[Dict[int, float]] = None
    user: Optional[str] = None
    seed: Optional[int] = None

class CompletionRequest(BaseModel):
    model: str = "rwkv-7b"
    prompt: str
    max_tokens: int = 2048
    temperature: float = 0.3
    top_p: float = 0.1
    n: int = 1
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: Optional[int] = None

class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Dict[str, int]

class CompletionChoice(BaseModel):
    text: str
    index: int
    finish_reason: Optional[str] = None

class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Dict[str, int]

# ---------- 辅助函数 ----------
def convert_messages_to_prompt(messages: List[ChatMessage]) -> str:
    """
    将 OpenAI 对话消息转换为 RWKV 的 prompt 字符串。
    简单格式：<|system|>\n...\n<|user|>\n...\n<|assistant|>\n...
    """
    prompt_parts = []
    for msg in messages:
        role = msg.role
        content = msg.content.strip()
        if role == "system":
            prompt_parts.append(f"System: \n{content}")
        elif role == "user":
            prompt_parts.append(f"User: \n{content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: \n{content}")
        else:
            prompt_parts.append(f"{role}: \n{content}")
    # 末尾添加 assistant 开始标记，让模型继续生成
    prompt_parts.append("Assistant: \n")
    return "\n".join(prompt_parts)

def estimate_tokens(text: str) -> int:
    """简单的 token 估算：按空格分词，实际可用 tokenizer 替换"""
    return len(text.split())

def generate_response_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"

# ---------- 全局 Future 管理 ----------
request_futures: Dict[str, asyncio.Future] = {}
future_lock = threading.Lock()
async def generate_completion(
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    presence_penalty: float,
    penalty_decay: float,
    seed: int,
    stream: bool = False,
) -> Union[str, Iterator[str]]:
    """
    内部统一生成函数，返回完整文本或异步生成器（用于流式）。
    """
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    req_id = generate_response_id()

    with future_lock:
        request_futures[req_id] = future

    if stream:
        # 流式：使用队列收集解码后的 token 字符串
        token_queue = queue.Queue()
        # 用来累积完整文本（如果需要 finish_callback 也需要解码结果）
        accumulated_tokens = []  # 存储解码后的文本

        def collect_callback(token_id: int):
            """
            假设回调传递的是单个 token ID (int)
            """
            # 调用 model.decode 将 token ID 转为字符串
            # 需要注意：decode 通常需要列表，并可能返回带空格的字符串
            # 这里假设 model.decode([token_id]) 返回一个字符串
            decoded = model.decode(token_id)  # 使用全局 model 实例
            token_queue.put(decoded)
            accumulated_tokens.append(decoded)

        def finish_callback(full_text: str):
            """
            注意：根据你之前的 Task 行为，finish_callback 可能没有参数或参数是完整文本。
            为了兼容，我们同样将完整文本放入队列作为结束信号。
            如果 finish_callback 不提供完整文本，可以用 ''.join(accumulated_tokens) 构建。
            """
            # 如果 finish_callback 接收到了完整文本，直接使用；否则自己拼接
            loop.call_soon_threadsafe(token_queue.put, None)

        try:
            manager.new_task(
                prompt=prompt,
                max_tokens=max_tokens,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                penalty_decay=penalty_decay,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed,
                collect_callback=collect_callback,   # 假设回调接收 token 整数 ID
                finish_callback=finish_callback,
            )
        except Exception as e:
            with future_lock:
                request_futures.pop(req_id, None)
            raise HTTPException(500, f"Task creation failed: {e}")

        async def stream_generator():
            while True:
                try:
                    token_str = await loop.run_in_executor(None, token_queue.get, True, 0.1)
                except queue.Empty:
                    continue
                if token_str is None:
                    break
                yield token_str

        return stream_generator()
    else:
        # 非流式：直接等待完整结果
        def finish_callback(result_text: str):
            loop.call_soon_threadsafe(future.set_result, result_text)

        try:
            manager.new_task(
                prompt=prompt,
                max_tokens=max_tokens,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                penalty_decay=penalty_decay,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed,
                finish_callback=finish_callback,
            )
        except Exception as e:
            with future_lock:
                request_futures.pop(req_id, None)
            raise HTTPException(500, f"Task creation failed: {e}")

        try:
            result = await future
        except Exception as e:
            raise HTTPException(500, f"Generation failed: {e}")
        finally:
            with future_lock:
                request_futures.pop(req_id, None)
        return result
    
# ---------- OpenAI Chat Completions 端点 ----------
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    prompt = convert_messages_to_prompt(request.messages)
    seed = request.seed if request.seed is not None else 42
    # 将 frequency_penalty 映射为 repetition_penalty
    repetition_penalty = request.frequency_penalty

    if request.stream:
        # 流式响应
        async def stream_events():
            gen = await generate_completion(
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=20,  # 可配置
                repetition_penalty=repetition_penalty,
                presence_penalty=request.presence_penalty,
                penalty_decay=0.0,
                seed=seed,
                stream=True,
            )
            chunk_id = generate_response_id()
            created = int(time.time())
            # 发送第一个包含角色的 chunk
            first_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
            }
            yield f"data: {json.dumps(first_chunk, ensure_ascii=False)}\n\n"
            # 逐 token 发送
            async for token in gen:
                chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            # 发送结束标记
            yield "data: [DONE]\n\n"
        return StreamingResponse(stream_events(), media_type="text/event-stream")
    else:
        # 非流式
        full_text = await generate_completion(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=20,
            repetition_penalty=repetition_penalty,
            presence_penalty=request.presence_penalty,
            penalty_decay=0.0,
            seed=seed,
            stream=False,
        )
        # 构建响应
        response = ChatCompletionResponse(
            id=generate_response_id(),
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=full_text.strip()),
                    finish_reason="stop"
                )
            ],
            usage={
                "prompt_tokens": estimate_tokens(prompt),
                "completion_tokens": estimate_tokens(full_text),
                "total_tokens": estimate_tokens(prompt + full_text)
            }
        )
        return response

# ---------- OpenAI Completions 端点（可选） ----------
@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    seed = request.seed if request.seed is not None else 42
    repetition_penalty = request.frequency_penalty

    if request.stream:
        async def stream_events():
            gen = await generate_completion(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=20,
                repetition_penalty=repetition_penalty,
                presence_penalty=request.presence_penalty,
                penalty_decay=0.0,
                seed=seed,
                stream=True,
            )
            chunk_id = generate_response_id()
            created = int(time.time())
            async for token in gen:
                chunk = {
                    "id": chunk_id,
                    "object": "text_completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [{"text": token, "index": 0, "finish_reason": None}]
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(stream_events(), media_type="text/event-stream")
    else:
        full_text = await generate_completion(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=20,
            repetition_penalty=repetition_penalty,
            presence_penalty=request.presence_penalty,
            penalty_decay=0.0,
            seed=seed,
            stream=False,
        )
        response = CompletionResponse(
            id=generate_response_id(),
            created=int(time.time()),
            model=request.model,
            choices=[
                CompletionChoice(
                    text=full_text.strip(),
                    index=0,
                    finish_reason="stop"
                )
            ],
            usage={
                "prompt_tokens": estimate_tokens(request.prompt),
                "completion_tokens": estimate_tokens(full_text),
                "total_tokens": estimate_tokens(request.prompt + full_text)
            }
        )
        return response

# ---------- 健康检查 ----------
@app.get("/health")
async def health():
    return {"status": "ok", "tasks_pending": len(manager.tasks) if manager.tasks else 0}

# ---------- 优雅关闭 ----------
@app.on_event("shutdown")
def shutdown_event():
    manager.shutdown()
    if manager.backthr and manager.backthr.is_alive():
        manager.backthr.join(timeout=2.0)

# ---------- 原有自定义生成端点（可选保留） ----------
# 如果需要保留原来的 /generate 和 /generate_stream，也可以加上