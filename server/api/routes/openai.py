import time
import uuid
import json
import asyncio
import loguru
import enum
from typing import AsyncGenerator, List, Optional, Union, Dict
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator

from ...scheduler import BaseScheduler
from ...utils import finish_callback, stream_callback
from ..dependencies import get_scheduler
from ...config import settings

log = loguru.logger.bind(module="api.openai")
router = APIRouter(prefix="/v1")

# 全局配置（单模型专用）
MODEL_NAME = getattr(settings, "model_name", "rwkv-7b-v0.7")
DEFAULT_TEMPLATE = getattr(settings, "default_chat_template", "rwkv-v0.7")
REQUEST_TIMEOUT = getattr(settings, "api_request_timeout", 120)
MAX_MAX_TOKENS = getattr(settings, "max_max_tokens", 8192000)
DEFAULT_PENALTY_DECAY = getattr(settings, "default_penalty_decay", 0.994)
DISCONNECT_CHECK_INTERVAL = 0.1

# ---------- 对话模板系统（已修复RWKV v0.7官方模板） ----------
CHAT_TEMPLATES = {
    # RWKV v0.7 官方标准模板 - 这是最关键的修复！
    "rwkv-v0.7": lambda msgs: (
        "以下是用户和AI助手之间的对话。AI助手乐于助人、知识渊博、诚实可靠，并且总是用\n\n"
        + "".join([
            f"User: {msg['content']}\n\nAssistant:" if msg['role'] == 'user' 
            else f"{msg['content']}\n\n" 
            for msg in msgs
        ])
    ),
    "chatml": lambda msgs: "\n".join([
        f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>" 
        for msg in msgs
    ]) + "\n<|im_start|>assistant\n",
    "qwen": lambda msgs: "\n".join([
        f"<|{msg['role']}|>\n{msg['content']}" 
        for msg in msgs
    ]) + "\n<|assistant|>\n",
}

# ---------- OpenAI 标准请求模型 ----------
class Role(str, enum.Enum):
    system = "system"
    user = "user"
    assistant = "assistant"

class ChatMessageContentPart(BaseModel):
    type: str
    text: Optional[str] = None

class ChatMessage(BaseModel):
    role: Role
    content: Union[str, List[ChatMessageContentPart]]
    name: Optional[str] = None

    @validator("content", pre=True)
    def validate_content(cls, v):
        if isinstance(v, list):
            return [p for p in v if p.get("type") == "text" and "text" in p]
        return v

    def get_text(self) -> str:
        if isinstance(self.content, str):
            return self.content.strip()
        return "\n".join([p.text for p in self.content if p.text]).strip()

class ChatCompletionRequest(BaseModel):
    model: str  # 忽略，单模型专用
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(settings.default_temperature, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(settings.default_top_p, ge=0.0, le=1.0)
    n: Optional[int] = Field(1, ge=1, le=1)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = Field(settings.default_max_tokens, ge=1, le=MAX_MAX_TOKENS)
    presence_penalty: Optional[float] = Field(settings.default_presence_penalty, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    user: Optional[str] = None
    # 扩展参数
    repetition_penalty: Optional[float] = Field(settings.default_repetition_penalty, ge=1.0)
    top_k: Optional[int] = Field(settings.default_top_k, ge=0)
    seed: Optional[int] = None
    penalty_decay: Optional[float] = Field(DEFAULT_PENALTY_DECAY, ge=0.0, le=1.0)
    template: Optional[str] = None

class CompletionRequest(BaseModel):
    model: str  # 忽略，单模型专用
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = Field(16, ge=1, le=MAX_MAX_TOKENS)
    temperature: Optional[float] = Field(settings.default_temperature, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(settings.default_top_p, ge=0.0, le=1.0)
    n: Optional[int] = Field(1, ge=1, le=1)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = Field(settings.default_presence_penalty, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    user: Optional[str] = None
    # 扩展参数
    repetition_penalty: Optional[float] = Field(settings.default_repetition_penalty, ge=1.0)
    top_k: Optional[int] = Field(settings.default_top_k, ge=0)
    seed: Optional[int] = None
    penalty_decay: Optional[float] = Field(DEFAULT_PENALTY_DECAY, ge=0.0, le=1.0)

# ---------- 核心辅助函数 ----------
def build_prompt(messages: List[ChatMessage], template: str = DEFAULT_TEMPLATE) -> str:
    """构建符合指定模板的prompt"""
    plain_msgs = [
        {"role": msg.role.value, "content": msg.get_text()}
        for msg in messages
        if msg.get_text()
    ]
    
    if not plain_msgs:
        raise ValueError("No valid text content in messages")
    
    return CHAT_TEMPLATES.get(template.lower(), CHAT_TEMPLATES[DEFAULT_TEMPLATE])(plain_msgs)

def map_params(req: Union[ChatCompletionRequest, CompletionRequest]) -> dict:
    """将OpenAI参数映射到底层推理参数"""
    params = {
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "top_k": req.top_k,
        "presence_penalty": req.presence_penalty,
        "repetition_penalty": req.repetition_penalty,
        "penalty_decay": req.penalty_decay,
    }
    
    if req.seed is not None:
        params["seed"] = req.seed
        
    return params

def process_stop(stop: Optional[Union[str, List[str]]]) -> List[str]:
    """处理stop参数，返回最多4个停止序列"""
    if not stop:
        # 默认添加RWKV常用的停止序列，防止模型继续生成用户对话
        return ["\n\nUser:", "\nUser:"]
    if isinstance(stop, str):
        return [stop, "\n\nUser:", "\nUser:"]
    return stop[:4] + ["\n\nUser:", "\nUser:"]

def check_and_truncate(text: str, stop_sequences: List[str]) -> tuple[str, bool]:
    """检查文本是否包含停止序列，返回截断后的文本和是否触发停止"""
    for seq in stop_sequences:
        if seq in text:
            return text.split(seq, 1)[0], True
    return text, False

# ---------- 流式生成器 ----------
async def stream_response(
    task,
    request_id: str,
    created: int,
    generator: AsyncGenerator,
    request: Request,
    stop_sequences: List[str],
    is_chat: bool = True,
):
    """通用流式响应生成器"""
    try:
        if is_chat:
            yield f"data: {json.dumps({
                'id': request_id,
                'object': 'chat.completion.chunk',
                'created': created,
                'model': MODEL_NAME,
                'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]
            }, ensure_ascii=False)}\n\n".encode("utf-8")

        full_text = ""
        stop_triggered = False
        last_check = time.time()

        async for tokens in generator:
            if stop_triggered:
                break

            # 检查客户端断开
            current_time = time.time()
            if current_time - last_check > DISCONNECT_CHECK_INTERVAL:
                if await request.is_disconnected():
                    log.info(f"[{request_id}] Client disconnected, stopping task")
                    task.stop()
                    break
                last_check = current_time

            # 解码和停止检测
            chunk_text = task.model_loader.decode(tokens)
            if not chunk_text:
                continue

            full_text += chunk_text
            chunk_text, stop_triggered = check_and_truncate(full_text, stop_sequences)

            if chunk_text:
                if is_chat:
                    payload = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": MODEL_NAME,
                        "choices": [{"index": 0, "delta": {"content": chunk_text}, "finish_reason": None}]
                    }
                else:
                    payload = {
                        "id": request_id,
                        "object": "text_completion.chunk",
                        "created": created,
                        "model": MODEL_NAME,
                        "choices": [{"text": chunk_text, "index": 0, "finish_reason": None}]
                    }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")

            if stop_triggered:
                task.stop()
                break

        # 发送结束chunk
        if not await request.is_disconnected():
            if is_chat:
                yield f"data: {json.dumps({
                    'id': request_id,
                    'object': 'chat.completion.chunk',
                    'created': created,
                    'model': MODEL_NAME,
                    'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]
                }, ensure_ascii=False)}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"

    except Exception as e:
        log.error(f"[{request_id}] Stream error: {str(e)}", exc_info=True)
        if not await request.is_disconnected():
            error_payload = {
                "id": request_id,
                "object": "chat.completion.chunk" if is_chat else "text_completion.chunk",
                "created": created,
                "model": MODEL_NAME,
                "choices": [{"index": 0, "delta": {} if is_chat else {"text": ""}, "finish_reason": "error"}],
                "error": {"message": str(e), "type": "internal_error"}
            }
            yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"

# ---------- 端点实现 ----------
@router.post("/chat/completions")
async def chat_completions(
    request: Request,
    req: ChatCompletionRequest,
    scheduler: BaseScheduler = Depends(get_scheduler),
):
    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    log.info(f"[{request_id}] Chat request: stream={req.stream}")

    try:
        stop_sequences = process_stop(req.stop)
        prompt = build_prompt(req.messages, req.template or DEFAULT_TEMPLATE)
        params = map_params(req)

        if req.stream:
            gen, collect_cb, finish_cb = stream_callback()
            task = scheduler.new_task(
                prompt=prompt,
                collect_callback=collect_cb,
                finish_callback=finish_cb,
                **params
            )
            created = int(time.time())

            return StreamingResponse(
                stream_response(task, request_id, created, gen, request, stop_sequences, is_chat=True),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "X-Request-ID": request_id,
                },
            )

        # 非流式处理
        future, callback = finish_callback()
        task = scheduler.new_task(
            prompt=prompt,
            finish_callback=callback,
            **params
        )

        # 监控断开
        async def monitor():
            while True:
                if await request.is_disconnected():
                    log.info(f"[{request_id}] Client disconnected, stopping task")
                    task.stop()
                    break
                await asyncio.sleep(DISCONNECT_CHECK_INTERVAL)

        monitor_task = asyncio.create_task(monitor())

        try:
            result = await asyncio.wait_for(future, timeout=REQUEST_TIMEOUT)
            generated_tokens = result[0]
        except asyncio.TimeoutError:
            task.stop()
            raise HTTPException(status_code=504, detail="Request timed out")
        finally:
            monitor_task.cancel()

        generated_text = task.model_loader.decode(generated_tokens)
        generated_text, _ = check_and_truncate(generated_text, stop_sequences)

        # 计算token
        prompt_tokens = len(task.tokenize(prompt))
        completion_tokens = len(generated_tokens)

        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_NAME,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": generated_text},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.error(f"[{request_id}] Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@router.post("/completions")
async def completions(
    request: Request,
    req: CompletionRequest,
    scheduler: BaseScheduler = Depends(get_scheduler),
):
    request_id = f"cmpl-{uuid.uuid4().hex}"
    log.info(f"[{request_id}] Completion request: stream={req.stream}")

    try:
        stop_sequences = process_stop(req.stop)
        
        if isinstance(req.prompt, list):
            if len(req.prompt) > 1:
                raise HTTPException(status_code=400, detail="Only single prompt supported")
            prompt = req.prompt[0]
        else:
            prompt = req.prompt

        if not prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        params = map_params(req)

        if req.stream:
            gen, collect_cb, finish_cb = stream_callback()
            task = scheduler.new_task(
                prompt=prompt,
                collect_callback=collect_cb,
                finish_callback=finish_cb,
                **params
            )
            created = int(time.time())

            return StreamingResponse(
                stream_response(task, request_id, created, gen, request, stop_sequences, is_chat=False),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "X-Request-ID": request_id,
                },
            )

        # 非流式
        future, callback = finish_callback()
        task = scheduler.new_task(
            prompt=prompt,
            finish_callback=callback,
            **params
        )

        async def monitor():
            while True:
                if await request.is_disconnected():
                    log.info(f"[{request_id}] Client disconnected, stopping task")
                    task.stop()
                    break
                await asyncio.sleep(DISCONNECT_CHECK_INTERVAL)

        monitor_task = asyncio.create_task(monitor())

        try:
            result = await asyncio.wait_for(future, timeout=REQUEST_TIMEOUT)
            generated_tokens = result[0]
        except asyncio.TimeoutError:
            task.stop()
            raise HTTPException(status_code=504, detail="Request timed out")
        finally:
            monitor_task.cancel()

        generated_text = task.model_loader.decode(generated_tokens)
        generated_text, _ = check_and_truncate(generated_text, stop_sequences)

        prompt_tokens = len(task.tokenize(prompt))
        completion_tokens = len(generated_tokens)

        return {
            "id": request_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": MODEL_NAME,
            "choices": [{
                "text": generated_text,
                "index": 0,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"[{request_id}] Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@router.get("/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": MODEL_NAME,
            "object": "model",
            "created": int(time.time()) - 86400,
            "owned_by": "self",
            "permission": [],
            "root": MODEL_NAME,
            "parent": None
        }]
    }

@router.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": int(time.time())}