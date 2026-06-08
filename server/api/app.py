from contextlib import asynccontextmanager
from fastapi import FastAPI
import time

from ..scheduler import RWKV070ModelLoader, DynamicScheduler
from ..task.manager import get_task_manager, shutdown_task_manager
from ..config import settings
from .routes.v1 import router


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
    get_task_manager().set_dependencies(scheduler.model_loader,scheduler.sampler)
    yield
    # 关闭
    scheduler.shutdown()
    shutdown_task_manager()
    time.sleep(0.5)


def create_app():
    app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    return app
