import sqlite3
import pickle
import threading
import time
import uuid
import copy
from collections import OrderedDict
from typing import Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import queue
from loguru import logger
from ..config import settings

log = logger.bind(module="task.manager")


class BoundedThreadPoolExecutor(ThreadPoolExecutor):
    def __init__(
        self,
        max_workers=1,
        queue_size=settings.task_async_queue_size,
        thread_name_prefix="",
    ):
        self._work_queue = queue.Queue(maxsize=queue_size)
        super().__init__(max_workers, thread_name_prefix=thread_name_prefix)

    def submit(self, fn, *args, **kwargs):
        self._work_queue.put(None, block=True, timeout=None)
        future = super().submit(fn, *args, **kwargs)

        def _done(_):
            try:
                self._work_queue.get_nowait()
            except queue.Empty:
                pass

        future.add_done_callback(_done)
        return future


class TaskManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, cpu_capacity: int = settings.task_default_cpu_capacity):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, cpu_capacity: int = settings.task_default_cpu_capacity):
        if self._initialized:
            return

        self.cpu_capacity = cpu_capacity
        self.l1_cache: OrderedDict[str, Any] = OrderedDict()
        self.cache_lock = threading.Lock()

        # SQLite with WAL
        self.db_conn = sqlite3.connect(
            settings.task_db_path, check_same_thread=False, timeout=10.0
        )
        self.db_conn.execute("PRAGMA journal_mode=WAL")
        self.db_conn.execute("PRAGMA synchronous=NORMAL")
        self.db_cursor = self.db_conn.cursor()
        self.db_lock = threading.Lock()
        self._init_db()

        self.io_executor = BoundedThreadPoolExecutor(
            max_workers=1, thread_name_prefix="task_writer"
        )
        self._initialized = True
        log.info(
            f"TaskManager initialized: CPU capacity={self.cpu_capacity}, "
            f"async queue size={settings.task_async_queue_size}, db={settings.task_db_path}"
        )

    def _init_db(self):
        with self.db_lock:
            self.db_cursor.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    data_blob BLOB,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_template INTEGER DEFAULT 0
                )
            """)
            self.db_cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_last_updated ON tasks(last_updated)"
            )
            self.db_cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_is_template ON tasks(is_template)"
            )
            self.db_conn.commit()
        log.debug("Database table and indexes ensured")

    def _serialize(self, obj: Any) -> bytes:
        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

    def _deserialize(self, blob: bytes) -> Any:
        return pickle.loads(blob)

    def is_temporary(self, task_id: str) -> bool:
        return task_id.startswith("TMP_")

    def is_readonly(self, task_id: str) -> bool:
        return task_id.startswith("_")

    def _persist_task(self, task_id: str, obj: Any):
        if self.is_temporary(task_id):
            log.debug(f"Skip persisting temporary task {task_id}")
            return

        is_template = 1 if self.is_readonly(task_id) else 0
        try:
            blob = self._serialize(obj)
            with self.db_lock:
                self.db_cursor.execute(
                    "INSERT OR REPLACE INTO tasks (task_id, data_blob, last_updated, is_template) VALUES (?, ?, ?, ?)",
                    (task_id, blob, time.time(), is_template),
                )
                self.db_conn.commit()
            with self.db_lock:
                self.db_cursor.execute(
                    "SELECT COUNT(*) FROM tasks WHERE is_template = 0"
                )
                count = self.db_cursor.fetchone()[0]
                if count > settings.task_db_max_size:
                    delete_cnt = count - settings.task_db_max_size
                    self.db_cursor.execute(
                        "DELETE FROM tasks WHERE is_template = 0 AND task_id IN "
                        "(SELECT task_id FROM tasks WHERE is_template = 0 ORDER BY last_updated ASC LIMIT ?)",
                        (delete_cnt,),
                    )
                    self.db_conn.commit()
                    log.info(f"Pruned {delete_cnt} non-template tasks from DB (limit {settings.task_db_max_size})")
        except Exception as e:
            log.error(f"Failed to persist task {task_id}: {e}")

    def put_task(self, task_id: str, obj: Any):
        if task_id is None or obj is None:
            return

        if self.is_readonly(task_id) and not self.is_temporary(task_id):
            exists = task_id in self.l1_cache
            if not exists:
                with self.db_lock:
                    self.db_cursor.execute(
                        "SELECT 1 FROM tasks WHERE task_id = ?", (task_id,)
                    )
                    exists = self.db_cursor.fetchone() is not None
            if exists:
                raise ValueError(f"Cannot modify read-only template: {task_id}")

        with self.cache_lock:
            if task_id in self.l1_cache:
                del self.l1_cache[task_id]
            self.l1_cache[task_id] = obj
            # 淘汰
            evicted = []
            while len(self.l1_cache) > self.cpu_capacity:
                evicted_id, evicted_obj = self.l1_cache.popitem(last=False)
                if self.is_readonly(evicted_id):
                    continue
                evicted.append(evicted_id)
                if not self.is_temporary(evicted_id):
                    self.io_executor.submit(self._persist_task, evicted_id, evicted_obj)
            if evicted:
                log.debug(f"Evicted tasks due to capacity limit: {evicted}")

    def get_task(self, task_id: str) -> Optional[Any]:
        if task_id is None:
            return None

        with self.cache_lock:
            if task_id in self.l1_cache:
                obj = self.l1_cache[task_id]
                self.l1_cache.move_to_end(task_id)
                return obj

        if self.is_temporary(task_id):
            return None

        if self.is_readonly(task_id):
            return None

        return self._fetch_from_db(task_id)

    def _fetch_from_db(self, task_id: str) -> Optional[Any]:
        with self.db_lock:
            self.db_cursor.execute(
                "SELECT data_blob FROM tasks WHERE task_id = ?", (task_id,)
            )
            row = self.db_cursor.fetchone()
            if not row:
                return None

        try:
            obj = self._deserialize(row[0])
            obj.model_loader = self.model_loader
            obj.setup_rand = self.batch_sampler.setup_rand
            self.put_task(task_id, obj)
            log.debug(f"Loaded task {task_id} from DB into cache")
            return obj
        except Exception as e:
            log.error(f"Failed to deserialize task {task_id} from DB: {e}")
            return None

    def set_dependencies(self, model_loader, batch_sampler):
        self.model_loader = model_loader
        self.batch_sampler = batch_sampler

    def fork_template(self, template_id: str, new_task_id: Optional[str] = None) -> str:
        src_obj = self.get_task(template_id)
        if src_obj is None:
            raise KeyError(f"Source task {template_id} not found")

        if new_task_id is None:
            new_task_id = f"TASK_{uuid.uuid4().hex}"
        else:
            if self.is_readonly(new_task_id) and not self.is_temporary(new_task_id):
                raise ValueError("Forked task ID cannot be read-only")

        new_obj = copy.copy(src_obj)
        self.put_task(new_task_id, new_obj)
        log.info(f"Forked task {template_id} -> {new_task_id}")
        return new_task_id

    def close_task(self, task_id: str):
        obj_to_save = None
        with self.cache_lock:
            if task_id in self.l1_cache:
                obj = self.l1_cache.pop(task_id)
                if not self.is_readonly(task_id) and not self.is_temporary(task_id):
                    obj_to_save = obj
        if obj_to_save is not None:
            self.io_executor.submit(self._persist_task, task_id, obj_to_save)
            log.debug(f"Closed task {task_id}, scheduled for persistence")
        elif self.is_temporary(task_id):
            log.debug(f"Closed temporary task {task_id}, no persistence")

    def delete_task_from_any_level(self, task_id: str, force: bool = False) -> bool:
        if self.is_readonly(task_id) and not force:
            log.warning(f"Refused to delete readonly template {task_id} without force=True")
            return False

        deleted = False
        with self.cache_lock:
            if task_id in self.l1_cache:
                del self.l1_cache[task_id]
                deleted = True

        if not self.is_temporary(task_id):
            with self.db_lock:
                self.db_cursor.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
                self.db_conn.commit()
                if self.db_cursor.rowcount > 0:
                    deleted = True
        else:
            log.debug(f"Temporary task {task_id} has no DB record, cache deletion only")

        if deleted:
            log.info(f"Deleted task {task_id} (force={force})")
        else:
            log.debug(f"Task {task_id} not found for deletion")
        return deleted

    def flush_all(self):
        log.info("Flushing all tasks to disk...")
        self.io_executor.shutdown(wait=True)
        tasks_to_save = []
        with self.cache_lock:
            for tid, obj in list(self.l1_cache.items()):
                if not self.is_readonly(tid) and not self.is_temporary(tid):
                    tasks_to_save.append((tid, obj))
            self.l1_cache.clear()

        if tasks_to_save:
            with self.db_lock:
                try:
                    self.db_conn.execute("BEGIN TRANSACTION")
                    for tid, obj in tasks_to_save:
                        blob = self._serialize(obj)
                        self.db_conn.execute(
                            "INSERT OR REPLACE INTO tasks (task_id, data_blob, last_updated, is_template) VALUES (?, ?, ?, ?)",
                            (tid, blob, time.time(), 0),
                        )
                    self.db_conn.commit()
                    log.info(f"Flushed {len(tasks_to_save)} tasks to DB")
                except Exception as e:
                    log.error(f"Flush error: {e}")
                    self.db_conn.rollback()
        self.db_conn.close()
        log.info("TaskManager flushed and closed")

    def list_tasks_in_db(self) -> List[Tuple[str, float]]:
        with self.db_lock:
            self.db_cursor.execute(
                "SELECT task_id, last_updated FROM tasks ORDER BY last_updated DESC"
            )
            return self.db_cursor.fetchall()

    def list_all_tasks(self) -> dict:
        with self.cache_lock:
            cpu_tasks = list(self.l1_cache.keys())
        db_tasks = self.list_tasks_in_db()
        db_task_ids = [tid for tid, _ in db_tasks if not self.is_temporary(tid)]
        return {
            "cpu_cache": cpu_tasks,
            "database": db_task_ids,
            "total_count": len(cpu_tasks) + len(db_task_ids),
        }

    def print_all_tasks_status(self):
        status = self.list_all_tasks()
        log.info(f"TaskManager status: total={status['total_count']}, "
                 f"CPU cache={len(status['cpu_cache'])}/{self.cpu_capacity}, "
                 f"DB={len(status['database'])}")


def get_task_manager(
    cpu_capacity: int = settings.task_default_cpu_capacity,
) -> TaskManager:
    return TaskManager(cpu_capacity)


def shutdown_task_manager():
    get_task_manager().flush_all()


def show_all_tasks_status():
    get_task_manager().print_all_tasks_status()


def remove_task_from_any_level(task_id: str) -> bool:
    return get_task_manager().delete_task_from_any_level(task_id)