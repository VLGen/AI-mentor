"""
Точка входа для запуска чанкинга и индексации
"""

import logging
from pathlib import Path

from src.config import CHUNKS_DIR, RAW_DATA_DIR
from .chunking import process_all_pages
from .indexer import index_chunks
from src.parser.utils import setup_logger


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = setup_logger()
    if not any(Path(CHUNKS_DIR).iterdir()):
        logger.info("Начало подготовки чанков")
        process_all_pages(RAW_DATA_DIR, CHUNKS_DIR)
        logger.info("Начало индексации чанков")
    index_chunks()


if __name__ == "__main__":
    main()