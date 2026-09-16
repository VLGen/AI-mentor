"""
Построение индексов
"""
import json
import logging
from pathlib import Path
from typing import Any, Iterator, List

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from src.config import (
	BATCH_SIZE,
	CHUNKS_DIR,
	COLLECTION_NAME,
	EMBEDD_MODEL,
	QDRANT_HOST,
	QDRANT_PORT,
	VECTOR_SIZE,
)

logger = logging.getLogger(__name__)

def iter_chunks(chunks_dir: str) -> Iterator[dict[str, Any]]:
	"""Читает подготовленные чанки из JSONL-файлов в стабильном порядке."""
	for chunks_file in sorted(Path(chunks_dir).glob("*_chunks.jsonl")):
		with chunks_file.open("r", encoding="utf-8") as file:
			for line_number, line in enumerate(file, start=1):
				if not line.strip():
					continue
				try:
					chunk = json.loads(line)
				except json.JSONDecodeError as error:
					logger.warning("Пропущена строка %s:%s: %s", chunks_file, line_number, error)
					continue
				if not isinstance(chunk, dict) or not chunk.get("text"):
					logger.warning("Пропущен некорректный чанк %s:%s", chunks_file, line_number)
					continue
				yield chunk

def create_collection(client: QdrantClient) -> None:
	"""Создаёт чистую коллекцию с параметрами из config.py."""
	if client.collection_exists(COLLECTION_NAME):
		client.delete_collection(COLLECTION_NAME)
	client.create_collection(
		collection_name=COLLECTION_NAME,
		vectors_config=models.VectorParams(
			size=VECTOR_SIZE,
			distance=models.Distance.COSINE,
		),
	)

def build_documents(chunks: List[dict[str, Any]]) -> List[Document]:
	"""Преобразует подготовленные чанки в документы LlamaIndex."""
	documents = []
	for chunk in chunks:
		metadata = {
			"pageid": chunk.get("pageid"),
			"title": chunk.get("title", ""),
			"url": chunk.get("url", ""),
			"categories": chunk.get("categories", []),
			"chunk_index": chunk.get("chunk_index"),
			"total_chunks": chunk.get("total_chunks"),
		}
		metadata_keys = list(metadata)
		documents.append(
			Document(
				text=chunk["text"],
				metadata=metadata,
				excluded_embed_metadata_keys=metadata_keys,			# Исключаю видимость метаданных для модели эмбеддингов
				excluded_llm_metadata_keys=metadata_keys,			# Исключаю видимость метаданных для LLM
				id_=f"{chunk.get('pageid')}:{chunk.get('chunk_index')}",
			)
		)
	return documents

def index_chunks() -> int:
	"""Индексирует все чанки через LlamaIndex в Qdrant."""
    # Создание векторной бд
	client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
	create_collection(client)

    # Загрузка модели эмбеддингов из hugging face
	embed_model = HuggingFaceEmbedding(model_name=EMBEDD_MODEL)
	embedding_size = len(embed_model.get_text_embedding("Проверка размерности эмбеддинга"))
	if embedding_size != VECTOR_SIZE:
		raise ValueError(
			f"Модель вернула размерность {embedding_size}, ожидалась {VECTOR_SIZE}."
		)
	Settings.embed_model = embed_model

	vector_store = QdrantVectorStore(
		client=client,
		collection_name=COLLECTION_NAME,
	)
	storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Создание векторных индексов 
	indexed = 0
	batch: List[dict[str, Any]] = []
	for chunk in iter_chunks(CHUNKS_DIR):
		batch.append(chunk)
		if len(batch) >= BATCH_SIZE:
			VectorStoreIndex.from_documents(
				build_documents(batch),
				storage_context=storage_context,
				embed_model=embed_model,
			)
			indexed += len(batch)
			batch.clear()
			logger.info("Проиндексировано чанков: %s", indexed)

    # Добавление последнего батча, если он не полный
	if batch:
		VectorStoreIndex.from_documents(
			build_documents(batch),
			storage_context=storage_context,
			embed_model=embed_model,
		)
		indexed += len(batch)

	logger.info("Индексация завершена, всего чанков: %s.", indexed)
	return indexed
