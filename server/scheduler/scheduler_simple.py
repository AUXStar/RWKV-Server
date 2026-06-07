import torch
import time
from loguru import logger

from .loader import RWKV070ModelLoader
from .scheduler_base import BaseScheduler
from ..task import Status

# 模块日志绑定
log = logger.bind(module="scheduler.simple")


class SimpleScheduler(BaseScheduler):
    """简化版推理调度器，可基于此类自定义调度器"""

    def __init__(
        self,
        model_loader: RWKV070ModelLoader,
        max_batch_size: int = 35,
        buffer_size: int = 50,
    ):
        super().__init__(model_loader, max_batch_size, buffer_size)

        log.info(f"SimpleScheduler initialized | batch_size={max_batch_size}")

    def run(self):
        it = 0
        while self.tasks:
            it += 1
            # 分配新任务到批次
            self.update_batch(self.worker["mask"])

            # 无活跃任务则退出循环
            active_cnt = self._get_active_count()
            if active_cnt == 0:
                break

            # 执行批量推理，计时
            st = time.time()
            self.engine.generate(self.worker, self.worker["mask"], self.worker["stop_flags"])
            pulse_time = time.time() - st

            # 收集生成结果
            self._collect()

            # 打印推理速度日志
            if active_cnt and pulse_time > 0:
                speed = active_cnt * self.buffer_size / pulse_time
                log.info(
                    f"Iter {it} fixed_cap={self.current_capacity} active={active_cnt} speed={speed:.2f} tok/s per_task={speed/active_cnt:.2f} tok/s"
                )
