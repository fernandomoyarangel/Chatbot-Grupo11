import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStore

from langchain.tools import tool
from langchain_core.runnables import RunnableConfig


load_dotenv()


def get_embeddings():
    """
    Construye el backend de embeddings segun variables de entorno.
    Devuelve: instancia de embeddings compatible con LangChain.
    Nota: por defecto usa HuggingFace; con EMBEDDINGS_BACKEND=ollama usa Ollama.
    """
    backend = os.getenv("EMBEDDINGS_BACKEND", "huggingface").lower()
    if backend == "ollama":
        model_name = os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma:latest")
        return OllamaEmbeddings(model=model_name)
    model_name = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(model_name=model_name)


@lru_cache(maxsize=None)
def get_vector_store() -> VectorStore:
    """
    Obtiene la conexion con la BD de embeddings usando ChromaDB.
    Recibe: nada (lee CHROMA_PATH).
    Devuelve: instancia VectorStore lista para busquedas.
    """
    embeddings = get_embeddings()

    persist_directory = os.getenv(
        "CHROMA_PATH",
        "./data/vector_db"
    )

    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    return vector_store


def retrieve_context_data(query: str, k: int = 20):
    """
    Recupera documentos relevantes y serializa el contexto para el prompt.
    Recibe: query (la query del usuario), k (numero de documentos a recuperar).
    Devuelve: (serialized, retrieved_docs).
    """
    vector_store = get_vector_store()
    retrieved_docs = vector_store.max_marginal_relevance_search(
        query, 
        k=k, 
        fetch_k=50, 
        lambda_mult=0.5 
    )

    type_map = {
        "specs": "TECHNICAL SPECS",
        "plot": "PLOT FRAGMENT"
    }

    serialized = "\n\n".join(
        (
            f"CONTENT BLOCK:\n"
            f"TYPE: {type_map.get(doc.metadata.get('doc_type', ''), 'GENERAL INFO')}\n"
            f"SOURCE_ID: {doc.metadata.get('source', 'Unknown')}\n"
            f"CONTENT:\n"
            f"{doc.page_content}\n"
            f"----------------"
        )
        for doc in retrieved_docs
    )

    return serialized, retrieved_docs


def get_rag_tools():
    """
    Define y devuelve las herramientas de RAG para el agente.
    Recibe: nada.
    Devuelve: lista de herramientas LangChain.
    """

    @tool(description="Recuperacion de contexto", response_format="content_and_artifact")
    def retrieve_context(query: str, config: RunnableConfig = None):
        """
        Herramienta LangChain que expone la recuperacion de contexto.
        Recibe: query (str), config (RunnableConfig opcional).
        Devuelve: (serialized, retrieved_docs).
        """
        return retrieve_context_data(query=query, k=10)

    return [retrieve_context]