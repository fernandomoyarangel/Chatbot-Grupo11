from functools import lru_cache
from typing import Optional
from langchain_core.prompts import PromptTemplate

from .tools.rag_tools import retrieve_context_data
from ..core.config import Settings
from .uc3m_llm import UC3MChatModel


@lru_cache(maxsize=None)
def get_llm_model():
    return UC3MChatModel(
        model="llama3.1:8b"   # o "qwen3:8b" segun lo que os indiquen
    )


class RAGService:
    def __init__(self, idioma: str = "english", k: int = 10):
        self.llm = get_llm_model()
        self.k = k
        self.prompt_template = Settings.get_prompt()
        

    def _build_context(self, query: str):
        return retrieve_context_data(query=query, k=self.k)

    def process_query(self, input: str, session_id: str):
        serialized, docs = self._build_context(input)
        
        # --- CORTAFUEGOS EN INGLÉS ---
        if not docs:
            return {
                "role": "assistant",
                "content": "I am sorry, I could not find any information about that in the movie database.",
                "sources": []
            }
        # -----------------------------

        prompt_text = self.prompt_template.format(context=serialized, question=input)
        
        # Invocación al LLM
        response_obj = self.llm.invoke(prompt_text)

        # Limpieza de respuesta
        if hasattr(response_obj, 'content'):
            answer_text = response_obj.content
        else:
            answer_text = str(response_obj)

        sources = []
        for doc in docs:
            meta = doc.metadata or {}
            src = meta.get("source") or meta.get("name") or "Unknown"
            sources.append({"source": src, "content": doc.page_content})

        return {
            "role": "assistant",
            "content": answer_text,
            "sources": sources,
        }
