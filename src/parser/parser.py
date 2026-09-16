"""
Основной модуль парсера. Содержит функции для парсинга данных из Википедии и сохранения их в формате JSON.
"""

import time
import logging
from typing import Optional

from parser.utils import retry_on_exception
import wikipediaapi
from src.config import (
    API_LANGUAGE, USER_AGENT, DELAY, MAX_DEPTH, CATEGORIES, OUTPUT_DIR
)
from .storage import get_existing_pageid, save_page, save_index

logger = logging.getLogger(__name__)

@retry_on_exception(retries=3, delay=2)
def get_category_members(category_title: str, max_depth: int=MAX_DEPTH) -> dict:
    """
    Рекурсивно собирает заголовки всех статей из категорий и подкатегорий до указанной глубины.
    @param category_title: Заголовок категории
    @param max_depth: Максимальная глубина рекурсии
    @return: Словарь {название статьи: идентификатор страницы}
    """
    wiki = wikipediaapi.Wikipedia(user_agent=USER_AGENT, language=API_LANGUAGE)      # Создаем объект Wikipedia с указанным языком и user-agent
    category = wiki.page(category_title)        # Получаем объект категории по заголовку
    if not category.exists():
        logger.warning(f"Категория {category_title} не найдена.")
        return {}
    
    page_map = {}                           # Словарь для хранения заголовков статей и идентификатора страницы
    categories_to_process = [category]      # Список категорий для обработки на текущем уровне
    processed_categories = set()            # Множество для хранения уже обработанных категорий
    current_depth = 0                       # Текущая глубина рекурсии

    # Рекурсивно проходимся по всем категориям
    while categories_to_process and current_depth < max_depth:  
        next_level_categories = []
        for cat in categories_to_process:
            if cat.title in processed_categories:
                continue
            processed_categories.add(cat.title)
            for member in cat.categorymembers.values():
                if member.ns == 0:  # ns=0 означает, что это статья
                    page_map[member.title] = member.pageid
                elif member.ns == 14 and current_depth < max_depth:  # ns=14 означает, что это подкатегория
                    next_level_categories.append(member)
        categories_to_process = next_level_categories
        current_depth += 1

    return page_map

@retry_on_exception(retries=3, delay=1)
def get_page_content(page_title: str) -> Optional[dict]:
    """
    Загружает данные страницы по ее заголовку

    Возвращает словрь с полями:
        - title: заголовок
        - pageid: идентификатор
        - extract: полный текст (включая все разделы)
        - summary: краткое описание (первый абзац)
        - categories: список категорий
        - url: полный URL страницы
        - sections: заголовки разделов
    Если страница не существует, возвращает None.
    """
    wiki = wikipediaapi.Wikipedia(user_agent=USER_AGENT, language=API_LANGUAGE)
    page = wiki.page(page_title)
    if not page.exists():
        logger.debug(f"Страница {page_title} не существует.")
        return None

    return {
        'title':page.title,
        'pageid':page.pageid,
        'extract':page.text,
        'summary':page.summary,
        'categories':[cat.title for cat in page.categories.values()],
        'url':page.fullurl,
        'sections':[s.title for s in page.sections]
    }

def collect_wiki_pages(
        categories: list[str]=CATEGORIES,
        output_dir: str=OUTPUT_DIR,
        max_depth: int=MAX_DEPTH,
        delay: float=DELAY
) -> None:
    """Основная функция для сбора данных из Википедии по указанным категориям и сохранения их в JSON-файлы.
    @param categories: Список категорий для парсинга
    @param output_dir: Директория для сохранения данных
    @param max_depth: Максимальная длина обхода подкатегорий
    @param delay: Задержка между запросами
    """
    logger.info("Запуск сбора данных из Википедии...")

    # Сбор всех заголовков из всех категорий
    all_pages = {}
    for cat in categories:
        logger.info(f"Сбор страниц из категории: {cat}")
        pages = get_category_members(cat, max_depth)
        all_pages.update(pages)      # Обновляем словарь, добавляя элементы из словаря pages
        logger.info(f"Найдено страниц: {len(pages)}")

    logger.info(f"Уникальных страниц для загрузки: {len(all_pages)}")

    # Проверка загруженных страниц
    existing_pageids = get_existing_pageid(output_dir)
    logger.info(f"Уже загружено {len(existing_pageids)} страниц.")

    # Загрузка только новых страниц
    to_download = {
        title: pageid
        for title, pageid in all_pages.items()
        if pageid not in existing_pageids
    }
    logger.info(f"{len(to_download)} новых страниц к загрузке.")

    # Загрузка и сохранение
    downloaded = 0
    for idx, (title, pageid) in enumerate(to_download.items()):
        try:
            data = get_page_content(title)
            if data and data.get("extract"):
                save_page(data, output_dir)
                downloaded += 1
            if (idx + 1) % 10 == 0:
                logger.info(f"Обработано {idx+1}/{len(to_download)} страниц, загружено {downloaded}")
            time.sleep(delay)       # Пауза между запросами
        except Exception as e:
            logger.error(f"Ошибка при обработке страницы '{title}': {e}")

    save_index(existing_pageids | set(to_download.values()), output_dir)
    logger.info(f"Сбор данных завершен. Загружено {downloaded} новых страниц.")