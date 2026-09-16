"""
Разделение на чанки
"""

import os
import re
import json
import logging
from typing import Any, List, Dict

from llama_index.core.node_parser import SentenceSplitter
from src.config import (
    RAW_DATA_DIR,
    CHUNKS_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_TEXT_LENGTH,
)
from src.parser.storage import load_index, load_page

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    # Удаление конструкций displaystyle
    text = re.sub(r'\\{\\displaystyle[^}]*\\}', '', text, flags=re.DOTALL)
    # Удаление множественных переносов
    text = re.sub(r'\n\s*\n', '\n\n', text)
    # Схлопывание пробелов
    text = re.sub(r' +', ' ', text)
    return text.strip()

def process_page(page_id: int, raw_dir: str) -> List[Dict[str, Any]]:
    """
    Загружает страницу, очищает текст и разбивает на чанки.
    @return: Список словарей с чанками и метаданными.
    """
    data = load_page(page_id, raw_dir)
    if not data:
        return []
    # Получаем текст страницы
    text = data.get('extract', '')
    if len(text) <= MIN_TEXT_LENGTH:
        logger.debug(f"Страница {page_id} слишком короткая ({len(text)} символов), пропускаем.")
        return []
    # Очистка полученного текста
    text = clean_text(text)
    # Разбиение на чанки
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separator=" ",
        paragraph_separator="\n\n",
        secondary_chunking_regex="[^,.;]+[,.;]?",
    )
    chunks = splitter.split_text(text)
    # Формирование выходных данных
    result = []
    for idx, chunk in enumerate(chunks):
        result.append({
            'pageid': page_id,
            'title': data.get('title', ''),
            'url': data.get('url', ''),
            'categories': data.get('categories', []),
            'chunk_index': idx,
            'total_chunks': len(chunks),
            'text': chunk,
            'metadata': {
                'source': data.get('url', ''),
                'pageid': page_id
            }
        })
    return result

def process_all_pages(raw_dir: str, chunks_dir: str) -> None:
    """
    Основная функция для обработки всех страниц и сохранения чанков
    """
    os.makedirs(chunks_dir, exist_ok=True)
    page_ids = load_index(raw_dir)
    logger.info(f"Начало обработки {len(page_ids)} страниц.")

    total_chunks = 0        # Счетчик общего количества чанков
    for page_id in page_ids:
        # Берем все чанки для одной страницы
        chunks = process_page(page_id, raw_dir)
        if chunks:
            # Сохранение чанков
            chunks_file = os.path.join(chunks_dir, f"{page_id}_chunks.jsonl")
            with open(chunks_file, 'w', encoding='utf-8') as f:
                for chunk in chunks:
                    # Записываем каждый чанк и его метаданные в json-файл
                    f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
            total_chunks += len(chunks)
            if total_chunks % 100 == 0:
                logger.info(f"Обработано {len(page_ids)} страниц / {total_chunks} чанков")

    logger.info(f"Создано {total_chunks} чанков из {len(page_ids)} страниц.")
