import os
import loguru
from functools import partial
from chatchat.settings.settings import Settings
from memoization import cached, CachingAlgorithmFlag

def _filter_logs(record: dict) -> bool:
    """
    日志过滤: 隐藏DEBUG日志, 清除ERROR日志的异常, 只保留错误本身, 减少日志冗余 \n
    日志级别: DEBUG(10) < INFO(20) < WARNING(30) < ERROR(40) < CRITICAL(50)
    """
    if record["level"].no <= 10 and not Settings.basic_settings.log_verbose:
        return False
    if record["level"].no == 40 and not Settings.basic_settings.log_verbose:
        record["exception"] = None
    return True

@cached(max_size=100, algorithm=CachingAlgorithmFlag.LRU)
def build_logger(log_file: str = "chatchat"):
    """
    构建一个带有彩色输出和日志文件的日志记录器 \n
    logger = build_logger("api") \n
    logger.info("<green>some message</green>")
    """
    loguru.logger._core.handlers[0]._filter = _filter_logs
    logger = loguru.logger.opt(colors=True) # 启用颜色输出
    logger.opt = partial(loguru.logger.opt, colors=True)
    logger.warn = logger.warning
    if log_file:
        if not log_file.endswith(".log"):
            log_file = f"{log_file}.log"
        if not os.path.isabs(log_file):
            log_file = str((Settings.basic_settings.LOG_PATH / log_file).resolve())
        logger.add(log_file, colorize=False, filter=_filter_logs)
    return logger
