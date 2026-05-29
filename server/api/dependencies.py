from fastapi import Request
from ..scheduler.scheduler_base import BaseScheduler

def get_scheduler(request: Request) -> BaseScheduler:
    return request.app.state.scheduler