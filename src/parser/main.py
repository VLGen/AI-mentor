"""
Точка входа для запуска парсера.
"""

import sys
import os

# Добавляем корень проекта в path, если запускаем напрямую
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CATEGORIES, OUTPUT_DIR, MAX_DEPTH, DELAY
from src.parser.parser import collect_wiki_pages
from src.parser.utils import setup_logger


def main():
    logger = setup_logger()
    logger.info("Запуск парсера Википедии")
    collect_wiki_pages(CATEGORIES, OUTPUT_DIR, MAX_DEPTH, DELAY)
    logger.info("Парсинг завершён")


if __name__ == '__main__':
    main()