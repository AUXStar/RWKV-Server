import time

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .....task import Status
from .....config import settings

class TaskResponseModel(BaseModel):
    """任务响应模型"""

    task_id: str = Field(default="", description="任务 ID")
    result: str = Field(description="生成结果")
    prefill_time: float = Field(description="预处理时间（秒）")
    gen_time: float = Field(description="生成时间（秒）")
    speed: float = Field(description="生成速度（token/秒）")
    finished: bool = Field(default=True, description="是否已完成")

    model_config = ConfigDict(extra="forbid", strict=True)


class TaskCreate(BaseModel):
    """创建任务请求模型"""

    prompt: str | list[int] = Field(description="提示词，可以是字符串或 token ID 列表")
    max_tokens: int = Field(
        default=settings.default_max_tokens,
        ge=0,
        le=40960,
        description="最大生成 token 数",
    )
    temperature: float = Field(
        default=settings.default_temperature,
        ge=0.0,
        le=2.0,
        description="温度",
    )
    top_k: int = Field(
        default=settings.default_top_k,
        ge=-1,
        le=100,
        description="Top-K 采样",
    )
    top_p: float = Field(
        default=settings.default_top_p,
        ge=0.0,
        le=1.0,
        description="Top-P 采样",
    )
    presence_penalty: float = Field(
        default=settings.default_presence_penalty,
        ge=0.0,
        le=10.0,
        description="存在惩罚",
    )
    repetition_penalty: float = Field(
        default=settings.default_repetition_penalty,
        ge=0.0,
        le=10.0,
        description="重复惩罚",
    )
    penalty_decay: float = Field(
        default=settings.default_penalty_decay,
        ge=0.0,
        le=1.0,
        description="惩罚衰减",
    )
    stream: bool = Field(default=False, description="是否流式输出")
    seed: int | None = Field(default=None, description="随机种子，为 None 时自动生成")

    model_config = ConfigDict(extra="forbid", strict=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _auto_seed(self) -> "TaskCreate":
        """如果 seed 为 None，则自动生成一个 32 位种子"""
        if self.seed is None:
            self.seed = time.time_ns() % (2**32)
        return self


class TaskUpdate(BaseModel):
    """更新任务请求模型"""

    prompt: str | list[int] | None = Field(
        default=None, description="提示词，可以是字符串或 token ID 列表"
    )
    max_tokens: int | None = Field(
        default=None,
        ge=0,
        le=40960,
        description="最大生成 token 数",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="温度参数",
    )
    top_k: int | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Top-K 采样参数",
    )
    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Top-P 采样参数",
    )
    presence_penalty: float | None = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description="存在惩罚",
    )
    repetition_penalty: float | None = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description="重复惩罚",
    )
    penalty_decay: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="惩罚衰减",
    )
    stream: bool = Field(default=False, description="是否流式输出")

    model_config = ConfigDict(extra="forbid", strict=True, arbitrary_types_allowed=True)


class FIMRequest(BaseModel):
    """Fill In Middle 请求模型

    给定 prefix 和 suffix，模型生成中间填充内容。
    prompt 构造格式：✿prefix✿✿suffix✿{suffix}✿middle✿{prefix}
    """

    prefix: str = Field(description="FIM 前缀文本（光标前的内容）")
    suffix: str = Field(default="", description="FIM 后缀文本（光标后的内容）")
    max_tokens: int = Field(
        default=settings.default_max_tokens,
        ge=1,
        le=40960,
        description="最大生成 token 数",
    )
    temperature: float = Field(
        default=settings.default_temperature,
        ge=0.0,
        le=2.0,
        description="温度",
    )
    top_k: int = Field(
        default=settings.default_top_k,
        ge=-1,
        le=100,
        description="Top-K 采样",
    )
    top_p: float = Field(
        default=settings.default_top_p,
        ge=0.0,
        le=1.0,
        description="Top-P 采样",
    )
    presence_penalty: float = Field(
        default=settings.default_presence_penalty,
        ge=0.0,
        le=10.0,
        description="存在惩罚",
    )
    repetition_penalty: float = Field(
        default=settings.default_repetition_penalty,
        ge=0.0,
        le=10.0,
        description="重复惩罚",
    )
    penalty_decay: float = Field(
        default=settings.default_penalty_decay,
        ge=0.0,
        le=1.0,
        description="惩罚衰减",
    )
    stream: bool = Field(default=False, description="是否流式输出")
    seed: int | None = Field(default=None, description="随机种子，为 None 时自动生成")

    model_config = ConfigDict(extra="forbid", strict=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _auto_seed(self) -> "FIMRequest":
        """如果 seed 为 None，则自动生成一个 32 位种子"""
        if self.seed is None:
            self.seed = time.time_ns() % (2**32)
        return self


class DataFrame(BaseModel):
    """数据帧模型"""

    data: str = Field(description="数据内容")
    task_id: str = Field(description="任务 ID")
    gen_time: float = Field(description="生成时间（秒）")
    speed: float = Field(description="生成速度（token/秒）")

    model_config = ConfigDict(extra="forbid", strict=True)

class TaskInfo(BaseModel):
    task_id: str
    generated_buf:int
    status:Status