from pydantic import BaseModel, ConfigDict, Field, model_validator
from enum import Enum
from typing import Any, Optional
import time

from .....config import settings


class TmpTaskModel(BaseModel):
    prompt: str | list[int] = Field(description="生成的提示词，可以是字符串或整数列表")
    max_tokens: int = Field(
        default=settings.default_max_tokens,
        ge=0,
        le=3000,
        description="生成的最大 token 数 3000 以内",
    )
    presence_penalty: float = Field(
        default=settings.default_presence_penalty, description="存在惩罚"
    )
    repetition_penalty: float = Field(
        default=settings.default_repetition_penalty, description="重复惩罚"
    )
    penalty_decay: float = Field(
        default=settings.default_penalty_decay, description="惩罚衰减"
    )
    temperature: float = Field(default=settings.default_temperature, description="温度")
    top_k: int = Field(default=settings.default_top_k, description="Top-K 采样")
    top_p: float = Field(default=settings.default_top_p, description="Top-P 采样")
    seed: int = Field(default_factory=time.time_ns, description="随机种子")


class TaskResponseModel(BaseModel):
    task_id: str = ""
    result: str
    prefill_time: float
    gen_time: float
    speed: float
    finished: bool = True


class TaskCreate(BaseModel):
    prompt: str | list[int]
    max_tokens: int = Field(default=settings.default_max_tokens, ge=0, le=4096000)
    temperature: float | list[float] = Field(
        default=settings.default_temperature, ge=0.0, le=2.0
    )
    top_k: int | list[int] = Field(default=settings.default_top_k, ge=0, le=100)
    top_p: float | list[float] = Field(default=settings.default_top_p, ge=0.0, le=1.0)
    presence_penalty: float | list[float] = Field(
        default=settings.default_presence_penalty, ge=0.0, le=10.0
    )
    repetition_penalty: float | list[float] = Field(
        default=settings.default_repetition_penalty, ge=0.0, le=10.0
    )
    penalty_decay: float | list[float] = Field(
        default=settings.default_penalty_decay, ge=0.0, le=1.0
    )
    stream: bool = False
    seed: int | None = None

    model_config = ConfigDict(extra="forbid", strict=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _auto_seed(self):
        if self.seed is None:
            self.seed = time.time_ns() % (2**32)
        return self


class TaskUpdate(BaseModel):
    prompt: str | list[int] | None = None
    max_tokens: int | None = Field(default=None, ge=0, le=4096)
    temperature: float | list[float] | None = Field(default=None, ge=0.0, le=2.0)
    top_k: int | list[int] | None = Field(default=None, ge=0, le=100)
    top_p: float | list[float] | None = Field(default=None, ge=0.0, le=1.0)
    presence_penalty: float | list[float] | None = Field(default=None, ge=0.0, le=10.0)
    repetition_penalty: float | list[float] | None = Field(
        default=None, ge=0.0, le=10.0
    )
    penalty_decay: float | list[float] | None = Field(default=None, ge=0.0, le=1.0)
    stream: bool | None = None

    model_config = ConfigDict(extra="forbid", strict=True, arbitrary_types_allowed=True)


class DataFrame(BaseModel):
    data: str
    task_id: str
    gen_time: float
    speed: float

class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"


class ChatMessage(BaseModel):
    role: Role
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str  # 实际会忽略或映射到某个默认模型
    messages: list[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1  # 你当前只支持单条生成，可忽略 >1
    stream: Optional[bool] = False
    stop: Optional[str|list[str]] = None
    max_tokens: Optional[int] = 16
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0  # 映射到 repetition_penalty
    logit_bias: Optional[dict[str, float]] = None
    user: Optional[str] = None
    repetition_penalty: Optional[float] = 1.0
    top_k: Optional[int] = 0
    seed: Optional[int] = None


class CompletionRequest(BaseModel):
    model: str
    prompt: str | list[str]
    suffix: Optional[str] = None
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    stop: Optional[str|list[str]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    best_of: Optional[int] = 1
    logit_bias: Optional[dict[str, float]] = None
    user: Optional[str] = None
    # 扩展
    repetition_penalty: Optional[float] = 1.0
    top_k: Optional[int] = 0
    seed: Optional[int] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None  # "stop", "length", etc.


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: dict[str, int]  # {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


class CompletionResponseChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = None


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionResponseChoice]
    usage: dict[str, int]