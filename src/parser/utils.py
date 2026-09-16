"""
Вспомогательные функции для парсера (логирование, повторные попытки)
"""

import logging
import sys
from functools import wraps
from time import sleep


def setup_logger(level=logging.INFO):
    """Настройка базового логгера."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def retry_on_exception(retries=3, delay=1):
    """
    Декоратор для повторного выполнения функции при исключениях.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == retries - 1:
                        raise
                    sleep(delay)
            return None
        return wrapper
    return decorator