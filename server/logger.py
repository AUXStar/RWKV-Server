import sys
from loguru import logger

LOG_FORMAT = (
    "<green>{time:HH:mm:ss.SS}</green> | "
    "<cyan>{extra[module]:<18}</cyan> | "
    "<level>{level:7}</level> | "
    "<level>{message}</level>"
)

def setup_logging(verbose: bool = True):
    """
    配置 loguru 全局日志。
    
    Args:
        verbose: True -> INFO 级别，False -> WARNING 级别。
    """
    logger.remove()  # 清除默认 handler

    # 控制台 handler（异步彩色）
    logger.add(
        sys.stdout,
        format=LOG_FORMAT,
        level="INFO" if verbose else "WARNING",
        enqueue=True,       # 异步、线程安全
        colorize=True,
        backtrace=False,    # 生产环境通常不需要复杂回溯
        diagnose=False,
    )

    return logger