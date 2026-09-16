import argparse

from src.rag.pipeline import RAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG-пайплайн для локальной LLM и поиска по Qdrant.")
    parser.add_argument("query", nargs="?", help="Вопрос пользователя для RAG-системы.")
    parser.add_argument("--top-k", type=int, default=None, help="Количество чанков для retrieval.")
    args = parser.parse_args()

    query = args.query or input("Введите ваш вопрос: ").strip()
    if not query:
        raise ValueError("Пустой запрос.")

    pipeline = RAGPipeline()
    answer = pipeline.answer(query=query, top_k=args.top_k)
    print("\nОтвет:\n")
    print(answer)


if __name__ == "__main__":
    main()
