import time
from pydantic_settings import BaseSettings
from pydantic import model_validator


class Settings(BaseSettings):
    model_path: str = "model/rwkv7-g1g-2.9b-20260526-ctx8192.pth"

    max_batch_size: int = 256
    buffer_size: int = 32

    default_max_tokens: int = 4096
    default_temperature: float = 1
    default_top_p: float = 0.3
    default_top_k: int = 50
    default_presence_penalty: float = 2.0
    default_repetition_penalty: float = 0
    default_penalty_decay: float = 1.0

    verbose: bool = True

    vocab_path: str | None = "server/eof_v20230424.txt"

    task_default_cpu_capacity: int = 256 # 应比推理槽位更多
    task_db_max_size: int = 10000 # 防止爆db
    task_async_queue_size: int = 200 # sql写入线程
    task_db_path: str = "rwkv_tasks.db"

    @property
    def default_seed(self):
        return time.time_ns()
    
    
    @model_validator(mode='after')
    def check_cpu_capacity(self):
        if self.task_default_cpu_capacity < self.max_batch_size:
            raise ValueError(
                f"task_default_cpu_capacity ({self.task_default_cpu_capacity}) "
                f"must be >= max_batch_size ({self.max_batch_size})"
            )
        return self

    class Config:
        env_prefix = "RWKV_"


settings = Settings()
