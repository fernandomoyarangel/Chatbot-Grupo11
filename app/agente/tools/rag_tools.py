import os
from functools import lru_cache

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStore

from langchain.tools import tool
from langchain_core.runnables import RunnableConfig

@lru_cache(maxsize=None)
def get_vector_store() -> VectorStore:
    """
    Obtiene la conexión con la BD de embeddings usando ChromaDB.
    Persiste automáticamente en disco.
    """

    embeddings = OllamaEmbeddings(
        model="embeddinggemma:latest"
        # Para qwen normalmente se usa:
        # model="nomic-embed-text"
    )

    persist_directory = os.getenv(
        "CHROMA_PATH",
        "./data/vector_db"
    )

    vector_store = Chroma(
        collection_name="movie_rag",
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    return vector_store



def get_rag_tools(local=True, titulo=None):
    """
    Función para obtener las herramientas que puede emplear el agente

    Parámetros:  
    - local (bool): indica si se emplearán modelos locales o Gemini  
    - titulo (string): indica el título del libro sobre el que trabajará el RAG

    Returns:  
        - list[BaseTool]: lista con la herramienta de recuperación de contexto
    """



    @tool(description="Recuperación de contexto", response_format="content_and_artifact")
    def retrieve_context(query: str, config: RunnableConfig = None):
        """ 
        Función empleada como herramienta de recuperación de información relevante de un libro.

        Parámetros:  
        - query (string): consulta a realizar sobre el libro
        
        Returns:  
        - serialized (string): texto original recuperado a partir de la información de los documentos  
        - retrieved_docs (list[Document]): lista con los documentos recuperados por la consulta
        """
        vector_store = get_vector_store()

        filtro = {"book": titulo} 

        k=10

        retrieved_docs = vector_store.similarity_search(query, k=k, filter=filtro)

        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )

        return serialized, retrieved_docs


    return [retrieve_context]



