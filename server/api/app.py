from contextlib import asynccontextmanager
from fastapi import FastAPI
from ..scheduler import RWKV070ModelLoader
from ..scheduler import DynamicScheduler
from ..config import settings
from .routes import generate, tasks, stream

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动
    model_loader = RWKV070ModelLoader(settings.model_path)
    scheduler = DynamicScheduler(
        model_loader,
        max_batch_size=settings.max_batch_size,
        buffer_size=settings.buffer_size
    )
    scheduler.start_daemon()
    app.state.scheduler = scheduler
    yield
    # 关闭
    scheduler.shutdown()
    import time
    time.sleep(0.5)

def create_app():
    app = FastAPI(lifespan=lifespan)
    app.include_router(generate.router)
    app.include_router(tasks.router)
    app.include_router(stream.router)
    return app