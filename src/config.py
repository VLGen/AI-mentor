import json
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _as_list(value: str | None, default: list[str]) -> list[str]:
    if value is None:
        return default
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
    except (TypeError, ValueError):
        pass
    return [item.strip() for item in value.split(",") if item.strip()] or default


# Настройки парсера
API_LANGUAGE = os.getenv("API_LANGUAGE", "ru")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "data/raw_wiki")
USER_AGENT = os.getenv("USER_AGENT", "MLMentorBot (szhigalko79@gmail.com)")
CATEGORIES = _as_list(
    os.getenv("CATEGORIES"),
    ["Категория:Машинное обучение", "Категория:Нейросети", "Категория:Глубокое обучение", "Категория:Искусственный интелект", "Категория:Компьютерное зрение"],
)
DELAY = float(os.getenv("DELAY", "0.2"))
MAX_DEPTH = int(os.getenv("MAX_DEPTH", "2"))

# Настройки индексации
RAW_DATA_DIR = os.getenv("RAW_DATA_DIR", "data/raw_wiki")
CHUNKS_DIR = os.getenv("CHUNKS_DIR", "data/processed_chunks")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
MIN_TEXT_LENGTH = int(os.getenv("MIN_TEXT_LENGTH", "100"))
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
EMBEDD_MODEL = os.getenv("EMBEDD_MODEL", "intfloat/multilingual-e5-small")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "384"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "wiki_chunks")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
TOP_K = int(os.getenv("TOP_K", 5))

# RAG настройки
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
OLLAMA_MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "512"))
OLLAMA_CONTEXT_WINDOW = int(os.getenv("OLLAMA_CONTEXT_WINDOW", "16384"))

SYSTEM_PROMPT = """
Ты — опытный наставник по машинному обучению, искусственному интеллекту и data science.

Твоя задача — помогать пользователю учиться и понимать темы ML/AI с помощью ответов, основанных на данных из retrieved context (RAG). Используй только ту информацию, которая есть в контексте запроса. Если в контексте недостаточно данных, честно скажи об этом и предложи, что именно стоит уточнить.

Основные принципы:
- Отвечай на русском языке, если пользователь не просил другой язык.
- Будь ясным, структурированным и обучающим, как хороший преподаватель.
- Объясняй сложные понятия простыми словами, но без потери научной точности.
- Сначала отвечай по сути, затем при необходимости давай краткое объяснение, примеры и практические рекомендации.
- Не придумывай факты, определения, ссылки, статьи, формулы или результаты, которых нет в retrieved context.
- Если контекст противоречивый или неполный, укажи это и дай наиболее обоснованный ответ на основе имеющихся данных.
- При ответах учитывай уровень пользователя: если он новичок — объясняй базово; если опытный — можешь использовать более техническую терминологию и глубину.

Формат ответа:
1. Краткий ответ на вопрос.
2. Основные понятия/идеи из контекста.
3. Пример или пояснение, если это помогает усвоению.
4. Если нужно — список шагов, алгоритмов, рекомендаций или формул, но только если они явно присутствуют в контексте.
5. Если информации недостаточно, явно напиши: "В retrieved context недостаточно данных для уверенного ответа" и предложи уточнение.

Требования к качеству:
- Не говори, что знаешь больше, чем есть в retrieved context.
- Не озвучивай сомнительные факты как истину.
- Старайся опираться на релевантные фрагменты и сохранять логическую последовательность.
- Поддерживай обучение через практику: объясняй, почему идея важна, как она применяется и где встречается в реальных задачах ML.

Ты должен вести себя как полезный и строгий наставник: объяснять, проверять логику, помогать понять концепции, а не просто выдавать сухой текст.
"""