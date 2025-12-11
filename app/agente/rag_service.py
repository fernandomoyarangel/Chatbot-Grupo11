from functools import lru_cache
from typing import Optional

from .tools.rag_tools import retrieve_context_data
from ..core.config import Settings
from .uc3m_llm import UC3MChatModel


@lru_cache(maxsize=None)
def get_llm_model():
    return UC3MChatModel(
        model="llama3.1:8b"   # o "qwen3:8b" segun lo que os indiquen
    )


class RAGService:
    """
    Servicio RAG directo: recupera contexto desde Chroma y consulta el LLM UC3M
    con el prompt definido en Settings. No usa tool calling.
    """

    def __init__(self, idioma: str = "espanol", k: int = 10):
        self.llm = get_llm_model()
        self.prompt_template = Settings.get_prompt()
        self.k = k

    
    def _build_context(self, query: str):
        serialized, docs = retrieve_context_data(query=query, k=self.k)
        return serialized, docs

    def process_query(self, input: str, session_id: str):
        serialized, docs = self._build_context(input)
        prompt = self.prompt_template.format(context=serialized, question=input)

        answer_text = self.llm.invoke(prompt)

        # extrae fuentes legibles desde metadata
        sources = []
        for doc in docs:
            meta = doc.metadata or {}
            src = meta.get("source") or meta  # usa “source” si lo guardaste en ingest
            sources.append({"source": src, "content": doc.page_content})

        return {
            "role": "assistant",
            "content": answer_text,
            "sources": sources,  # docs usados
        }
