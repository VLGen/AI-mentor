"""
Функции для работы с файловым хранилищем
"""

import os
import json
import logging
from typing import Set

logger = logging.getLogger(__name__)

def get_existing_pageid(output_dir: str) -> Set[str]:
    """Возвращает множество идентификаторов страниц, которые уже существуют в файловом хранилище."""
    return load_index(output_dir)

def save_page(data: dict, output_dir: str) -> None:
    """Сохраняет данные страницы в JSON-файл с именем, соответствующим идентификатору страницы."""
    page_id = data.get('pageid')
    if not page_id:
        logger.warning("Попытка сохранить страницу без pageid, пропускаем.")
        return
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{page_id}.json")
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Ошибка сохранения страницы {page_id} в файл {filepath}: {e}")

def load_page(page_id: int, output_dir: str) -> dict:
    """Загружает данные страницы из JSON-файла по идентификатору страницы."""
    filepath = os.path.join(output_dir, f"{page_id}.json")
    if not os.path.exists(filepath):
        logger.warning(f"Файл {filepath} не найден.")
        return {}
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_index(pageids: set[int], output_dir: str) -> None:
    """Сохраняет множество pageid в файл index.json для быстрой проверки."""
    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, 'index.json')
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(list(pageids), f, ensure_ascii=False, indent=2)

def load_index(output_dir: str) -> set[int]:
    """Загружает индекс из файла, если он существует."""    
    index_path = os.path.join(output_dir, 'index.json')
    if not os.path.exists(index_path):
        return set()
    with open(index_path, 'r', encoding='utf-8') as f:
        return set(json.load(f))
    