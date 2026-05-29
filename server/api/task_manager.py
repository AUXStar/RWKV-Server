from ..task import Task

class TaskManager:
    def __init__(self):
        self._tasks = {}

    def create_task(self, scheduler, prompt, max_tokens) -> str:
        task_id = str(uuid.uuid4())
        task = scheduler.new_task(..., 
            collect_callback=self._make_collect_callback(task_id),
            finish_callback=self._make_finish_callback(task_id))
        self._tasks[task_id] = TaskRecord(task_id, task)
        return task_id

    def get_result(self, task_id):
        record = self._tasks.get(task_id)
        if record and record.task.status == Status.FINISHED:
            return {"content": record.result_text, "usage": record.usage, ...}
        return None