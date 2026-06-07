import sys
import logging
import inspect
from loguru import logger

from .config import settings

is_setup = False


class InterceptHandler(logging.Handler):
    """劫持标准 logging 输出，转发至 loguru"""

    def emit(self, record: logging.LogRecord) -> None:
        inter_logger = logger.bind(module=record.name)
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        inter_logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


LOG_FORMAT = (
    "<green>{time:HH:mm:ss.SS}</green> | "
    "<level>{level:7}</level> | "
    "<cyan>{extra[module]:<19}</cyan> | "
    "<level>{message}</level>"
)


LOGGERS_TO_INTERCEPT = [
    "uvicorn",
    "uvicorn.access",
    "uvicorn.error",
    "fastapi",
    "starlette",
    "starlette.routing",
    "gunicorn",
    "gunicorn.error",
]

def setup_logging(verbose: bool = True):
    """
    全局日志初始化
    Args:
        verbose: True=INFO级别, False=WARNING级别
    """
    global is_setup
    if is_setup:
        return logger

    logger.remove()

    logger.add(
        sys.stdout,
        format=LOG_FORMAT,
        level="INFO" if verbose else "WARNING",
        enqueue=True,
        colorize=True,
        backtrace=False,
        diagnose=False,
    )

    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    for logger_name in LOGGERS_TO_INTERCEPT:
        log = logging.getLogger(logger_name)
        log.handlers = [InterceptHandler()]
        log.propagate = False

    is_setup = True
    root_logger = logger.bind(module="main")
    return root_logger


setup_logging(settings.verbose)
