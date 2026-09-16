import logging
from typing import Any, Optional

from src.config import SYSTEM_PROMPT, TOP_K
from src.rag.llm import LocalLLM
from src.rag.retriever import retrieve_context

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(
        self,
        llm: Optional[LocalLLM] = None,
        top_k: int = TOP_K,
        system_prompt: Optional[str] = None,
    ):
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.llm = llm or LocalLLM(system_prompt=self.system_prompt)
        self.top_k = top_k

    def build_context(
        self,
        query: str,
        filter_by: Optional[dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> str:
        if not query or not str(query).strip():
            raise ValueError("Запрос не может быть пустым.")

        active_top_k = self.top_k if top_k is None else top_k
        logger.info("Получение контекста для запроса: %s (top_k=%s)", query, active_top_k)
        return retrieve_context(query=query, top_k=active_top_k, filter_by=filter_by)

    def answer(
        self,
        query: str,
        filter_by: Optional[dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> str:
        if not query or not str(query).strip():
            raise ValueError("Запрос не может быть пустым.")

        context = self.build_context(query=query, filter_by=filter_by, top_k=top_k)
        if not context.strip():
            logger.warning("Пустой контекст для запроса: %s", query)
            raise ValueError("В retrieved context недостаточно данных для уверенного ответа.")

        return self.llm.generate_response(
            query=query,
            context=context,
            system_prompt=self.system_prompt,
        )

    def __call__(
        self,
        query: str,
        filter_by: Optional[dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> str:
        return self.answer(
            query=query,
            filter_by=filter_by,
            top_k=top_k,
        )


__all__ = ["RAGPipeline"]
