import time
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_path: str = "server/model/rwkv7-g1g-2.9b-20260526-ctx8192.pth"

    max_batch_size: int = 256
    buffer_size: int = 32

    default_max_tokens: int = 2000
    default_temperature: float = 1
    default_top_p: float = 0.3
    default_top_k: int = 50
    default_presence_penalty: float = 2.0
    default_repetition_penalty: float = 0
    default_penalty_decay: float = 1.0

    verbose: bool = True

    vocab_path: str | None = "server/eof_v20230424.txt"

    task_default_cpu_capacity: int = 300
    task_db_max_size: int = 10000
    task_async_queue_size: int = 200
    task_db_path: str = "rwkv_tasks.db"

    @property
    def default_seed(self):
        return time.time_ns()

    class Config:
        env_prefix = "RWKV_"


settings = Settings()
