from fastapi import APIRouter, Depends
import time
from pydantic import BaseModel, Field

from ...scheduler import BaseScheduler
from ...task import Task
from ..dependencies import get_scheduler

router = APIRouter(prefix="/tasks")


class NewTask(BaseModel):
    prompt: str | list[int] = Field(description="生成的提示词，可以是字符串或整数列表")
    max_tokens: int = Field(
        default=200, ge=1, le=10240, description="生成的最大 token 数"
    )
    presence_penalty: float = Field(default=2, description="存在惩罚")
    repetition_penalty: float = Field(default=0, description="重复惩罚")
    penalty_decay: float = Field(default=0.994, description="惩罚衰减")
    temperature: float = Field(default=1, description="温度")
    top_k: int = Field(default=20, description="Top-K 采样")
    top_p: float = Field(default=0.5, description="Top-P 采样")
    seed: int = Field(default_factory=time.time_ns, description="随机种子")


@router.post("/new", response_model=str)
async def create_task(data: NewTask, scheduler: BaseScheduler = Depends(get_scheduler)):
    task_id = scheduler.new_task(
        prompt=data.prompt,
        max_tokens=data.max_tokens,
        presence_penalty=data.presence_penalty,
        repetition_penalty=data.repetition_penalty,
        penalty_decay=data.penalty_decay,
        temperature=data.temperature,
        top_k=data.top_k,
        top_p=data.top_p,
        seed=data.seed,
    )
    return task_id
