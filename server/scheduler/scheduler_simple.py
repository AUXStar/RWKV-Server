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
