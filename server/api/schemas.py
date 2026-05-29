from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional

class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    FINISHED = "finished"
    STOPPED = "stopped"
    FAILED = "failed"

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=200, ge=1, le=4096)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.5, ge=0.0, le=1.0)
    top_k: int = Field(default=-1, ge=-1)
    presence_penalty: float = Field(default=2.0, ge=0.0, le=10.0)
    repetition_penalty: float = Field(default=0.0, ge=0.0, le=10.0)
    penalty_decay: float = Field(default=1.0, ge=0.0, le=1.0)
    seed: Optional[int] = None

class TaskResponse(BaseModel):
    task_id: str
    status: TaskStatus

class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    generated_tokens: int
    max_tokens: int

class TaskResultResponse(BaseModel):
    task_id: str
    content: str
    usage: dict
    finish_reason: str