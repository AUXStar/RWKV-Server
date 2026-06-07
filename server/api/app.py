from contextlib import asynccontextmanager
from fastapi import FastAPI
import time
from ..scheduler import RWKV070ModelLoader
from ..scheduler import DynamicScheduler
from ..config import settings
from .routes import tasks, openai


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动
    model_loader = RWKV070ModelLoader(settings.model_path, settings.vocab_path)
    scheduler = DynamicScheduler(
        model_loader,
        max_batch_size=settings.max_batch_size,
        buffer_size=settings.buffer_size,
    )
    scheduler.start_daemon()
    app.state.scheduler = scheduler
    yield
    # 关闭
    scheduler.shutdown()
    time.sleep(0.5)


def create_app():
    app = FastAPI(lifespan=lifespan)
    app.include_router(tasks.router)
    app.include_router(openai.router)
    return app
