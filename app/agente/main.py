import os
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

from .rag_service import RAGService


def ensure_chroma_path():
    """
    Guarantees that CHROMA_PATH points to the local vector store under data/.
    This avoids relying on the current working directory when running the script.
    """
    project_root = Path(__file__).resolve().parents[2]
    default_vector_db = project_root / "data" / "vector_db"
    os.environ.setdefault("CHROMA_PATH", str(default_vector_db))


def main():
    load_dotenv()
    ensure_chroma_path()

    rag = RAGService()
    session_id = str(uuid4())

    print("Chat RAG interactivo. Escribe 'salir' para terminar.")

    while True:
        try:
            question = input("\nTú: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not question:
            continue

        if question.lower() in {"salir", "exit", "quit"}:
            break

        response = rag.process_query(input=question, session_id=session_id)
        print(f"\nAgente: {response['content']}")

    print("\nSesión finalizada.")


if __name__ == "__main__":
    main()
