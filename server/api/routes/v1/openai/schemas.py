from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from time import time_ns

class TextContentPart(BaseModel):
    type: str = "text"
    text: str


ContentPart = Union[TextContentPart, Dict[str, Any]]


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: FunctionCall


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[ContentPart], None]
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(2048, ge=1, le=8192, description="生成的最大token数")
    temperature: Optional[float] = Field(1.0, ge=0.0, le=2.0, description="采样温度，值越高越随机")
    top_p: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="核采样概率")
    n: Optional[int] = Field(1, ge=1, le=5, description="生成的结果数量")
    stream: Optional[bool] = Field(False, description="是否流式返回结果")
    stop: Optional[Union[str, List[str]]] = Field(None, description="停止序列")
    presence_penalty: Optional[float] = Field(2.0, ge=-2.0, le=2.0, description="存在惩罚")
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0, description="频率惩罚（使用重复惩罚模拟）")
    seed: int = Field(default_factory=time_ns, ge=0, description="随机种子，用于保证结果可复现")
    stream_options: Optional[Dict[str, Any]] = Field(None, description="流式响应选项")
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(None, description="工具调用选择")
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="可用工具列表")


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


class Delta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class ChunkChoice(BaseModel):
    index: int
    delta: Delta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChunkChoice]
    usage: Optional[Usage] = None


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]] = Field(..., description="用于补全的提示文本")
    max_tokens: Optional[int] = Field(2048, ge=1, le=8192, description="生成的最大token数")
    temperature: Optional[float] = Field(1.0, ge=0.0, le=2.0, description="采样温度，值越高越随机")
    top_p: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="核采样概率")
    n: Optional[int] = Field(1, ge=1, le=5, description="生成的结果数量")
    stream: Optional[bool] = Field(False, description="是否流式返回结果")
    stop: Optional[Union[str, List[str]]] = Field(None, description="停止序列")
    presence_penalty: Optional[float] = Field(2.0, ge=-2.0, le=2.0, description="存在惩罚")
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0, description="频率惩罚（使用重复惩罚模拟）")
    seed: int = Field(default_factory=time_ns, ge=0, description="随机种子，用于保证结果可复现")
    stream_options: Optional[Dict[str, Any]] = Field(None, description="流式响应选项")
    echo: Optional[bool] = Field(False, description="是否在响应中回显提示文本")


class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: Optional[str] = None


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Usage


class CompletionChunkChoice(BaseModel):
    index: int
    text: str
    finish_reason: Optional[str] = None


class CompletionChunk(BaseModel):
    id: str
    object: str = "text_completion.chunk"
    created: int
    model: str
    choices: List[CompletionChunkChoice]
    usage: Optional[Usage] = None


class Model(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "rwkv"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[Model]


class ErrorResponse(BaseModel):
    error: Dict[str, Any]