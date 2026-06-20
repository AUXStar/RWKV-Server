from fastapi import Request,HTTPException
from ..scheduler.scheduler_base import BaseScheduler
from ..task import Task, get_task_manager

task_manager = get_task_manager()

def get_scheduler(request: Request) -> BaseScheduler:
    return request.app.state.scheduler

def get_task(task_id:str)->Task:
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task