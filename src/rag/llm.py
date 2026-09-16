import logging
from typing import Optional

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.llms.ollama import Ollama

from src.config import (
    OLLAMA_BASE_URL,
    OLLAMA_CONTEXT_WINDOW,
    OLLAMA_MAX_TOKENS,
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
    SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


class LocalLLM:
    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        max_tokens: int = OLLAMA_MAX_TOKENS,
        context_window: int = OLLAMA_CONTEXT_WINDOW,
        temperature: float = OLLAMA_TEMPERATURE,
        system_prompt: Optional[str] = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.context_window = context_window
        self.temperature = temperature
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.llm: Optional[Ollama] = None
        self.initialize_model()

    def initialize_model(self) -> "LocalLLM":
        if self.llm is not None:
            return self

        try:
            self.llm = Ollama(
                model=self.model,
                base_url=self.base_url,
                temperature=self.temperature,
                context_window=self.context_window,
                request_timeout=60,
            )
            logger.info("Local LLM initialized: model=%s, base_url=%s", self.model, self.base_url)
            return self
        except Exception as exc:
            logger.exception("Не удалось инициализировать локальную LLM")
            raise RuntimeError(f"Не удалось инициализировать локальную LLM: {exc}") from exc

    def check_model_availability(self) -> bool:
        try:
            self.initialize_model()
            if self.llm is None:
                return False

            response = self.llm.complete("ping")
            text = getattr(response, "text", None)
            return bool(text and str(text).strip())
        except Exception as exc:
            logger.warning("LLM модель не доступна: %s", exc)
            return False

    def generate_response(
        self,
        query: str,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        if not query or not str(query).strip():
            raise ValueError("Запрос не может быть пустым.")

        self.initialize_model()

        if not self.check_model_availability():
            raise RuntimeError(f"Локальная модель '{self.model}' недоступна по адресу {self.base_url}.")

        active_system_prompt = system_prompt or self.system_prompt
        prompt_text = query.strip()

        if context:
            prompt_text = f"Контекст:\n{context}\n\nВопрос пользователя:\n{query}"

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=active_system_prompt),
            ChatMessage(role=MessageRole.USER, content=prompt_text),
        ]

        try:
            response = self.llm.chat(messages=messages, max_tokens=self.max_tokens)
            if hasattr(response, "message") and response.message is not None:
                content = response.message.content
                if isinstance(content, list):
                    return "".join(str(part) for part in content).strip()
                return str(content).strip()

            if hasattr(response, "text"):
                return str(response.text).strip()

            return str(response).strip()
        except Exception as exc:
            logger.exception("Ошибка генерации ответа через локальную модель")
            raise RuntimeError(f"Не удалось сгенерировать ответ через локальную LLM: {exc}") from exc
