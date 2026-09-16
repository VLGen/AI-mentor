from __future__ import annotations

import json
import logging
from typing import Any

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from src.config import COLLECTION_NAME, EMBEDD_MODEL, QDRANT_HOST, QDRANT_PORT, TOP_K

logger = logging.getLogger(__name__)


def _build_client(host: str = QDRANT_HOST, port: int = QDRANT_PORT) -> QdrantClient:
    """Создает клиент Qdrant."""
    logger.debug("Инициализация Qdrant клиента на %s:%s...", host, port)
    return QdrantClient(host=host, port=port)


def _build_query_filter(filter_by: dict[str, Any] | None) -> models.Filter | None:
    """Преобразует словарь с фильтрами в Qdrant filter."""
    if not filter_by:
        return None

    must = []
    for key, value in filter_by.items():
        if isinstance(value, list):
            must.append(models.FieldCondition(key=key, match=models.MatchAny(any=value)))
        elif isinstance(value, (str, int, float, bool)):
            must.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
        else:
            continue

    if not must:
        return None
    logger.debug("Создан фильтр запроса для полей %s.", list(filter_by.keys()))
    return models.Filter(must=must)


def _embed_query(query: str, model_name: str = EMBEDD_MODEL) -> list[float]:
    """Генерирует вектор эмбеддинга для текста запроса."""
    logger.debug("Проведение эмбеддинга запроса с %s.", model_name)
    model = SentenceTransformer(model_name)
    embedding = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
    return embedding.astype(float).tolist()


def _extract_text_from_payload(payload: dict[str, Any]) -> str:
    """Извлекает текст чанка из различных форматов payload, которые использует LlamaIndex/Qdrant."""
    for key in ("text", "content", "page_content"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    node_content = payload.get("_node_content")
    if isinstance(node_content, str):
        try:
            parsed = json.loads(node_content)
        except (TypeError, ValueError):
            return node_content.strip()
        if isinstance(parsed, dict):
            for key in ("text", "content", "page_content"):
                value = parsed.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            nested_text = parsed.get("metadata", {}).get("text")
            if isinstance(nested_text, str) and nested_text.strip():
                return nested_text.strip()
            return ""
        if isinstance(parsed, str) and parsed.strip():
            return parsed.strip()

    if isinstance(node_content, dict):
        for key in ("text", "content", "page_content"):
            value = node_content.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return ""


def search_chunks(
    query: str,
    top_k: int = TOP_K,
    filter_by: dict[str, Any] | None = None,
    collection_name: str = COLLECTION_NAME,
    host: str = QDRANT_HOST,
    port: int = QDRANT_PORT,
) -> list[dict[str, Any]]:
    """
    Выполняет поиск по запросу в Qdrant и возвращает найденные чанки вместе с метаданными.

    Returns:
        [{
            'id': ..., 'score': ..., 'text': ..., 'metadata': {...}, 'pageid': ..., 'title': ..., 'url': ...
        }, ...]
    """
    if not query or not query.strip():
        raise ValueError("Запрос не может быть пустым.")

    logger.info(
        "Поиск чанков (top_k=%d, collection=%s, filter=%s)...",
        top_k,
        collection_name,
        filter_by,
    )
    client = _build_client(host=host, port=port)
    query_filter = _build_query_filter(filter_by)
    vector = _embed_query(query)

    results = client.query_points(
        collection_name=collection_name,
        query=vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    ).points

    chunks = []
    for item in results:
        payload = item.payload or {}
        text = _extract_text_from_payload(payload)
        chunk = {
            "id": item.id,
            "score": float(item.score),
            "text": text,
            "metadata": payload,
            "pageid": payload.get("pageid"),
            "title": payload.get("title", ""),
            "url": payload.get("url", ""),
            "categories": payload.get("categories", []),
            "chunk_index": payload.get("chunk_index"),
            "total_chunks": payload.get("total_chunks"),
        }
        chunks.append(chunk)

    logger.debug("Поиск вернул %d чанков.", len(chunks))
    return chunks


def retrieve_context(
    query: str,
    top_k: int = TOP_K,
    filter_by: dict[str, Any] | None = None,
    collection_name: str = COLLECTION_NAME,
    host: str = QDRANT_HOST,
    port: int = QDRANT_PORT,
) -> str:
    """Возвращает только текст найденных чанков, объединенный в один контекст."""
    logger.debug("Извлечение контекста...")
    chunks = search_chunks(
        query=query,
        top_k=top_k,
        filter_by=filter_by,
        collection_name=collection_name,
        host=host,
        port=port,
    )
    return "\n\n---\n\n".join(chunk["text"] for chunk in chunks if chunk.get("text"))


__all__ = ["search_chunks", "retrieve_context", "_embed_query"]
