import time
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_path: str = "/home/njzy/rwkv_agent/server/model/rwkv7-g1f-2.9b-20260420-ctx8192"
    max_batch_size: int = 270
    buffer_size: int = 35
    default_max_tokens: int = 200
    default_temperature: float = 0.8
    default_top_p: float = 0.5
    default_top_k: int = -1
    default_presence_penalty: float = 2.0
    default_repetition_penalty: float = 0.0
    default_penalty_decay: float = 1.0

    @property
    def default_seed(self):
        return time.time_ns()

    class Config:
        env_prefix = "RWKV_"

settings = Settings()