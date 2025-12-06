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

    def __init__(self, idioma: str = "espanol", k: int = 5):
        self.llm = get_llm_model()
        self.prompt_template = Settings.get_prompt()
        self.k = k

    def _build_context(self, query: str) -> str:
        serialized, _docs = retrieve_context_data(query=query, k=self.k)
        return serialized

    def process_query(self, input: str, session_id: str):
        """
        Metodo empleado para que el agente procese una peticion.

        Parametros:
        - input (string): peticion del usuario
        - session_id (string): identificador de sesion del usuario

        Returns:
        - JSON: la respuesta del agente en formato JSON
        """
        context = self._build_context(input)
        prompt = self.prompt_template.format(context=context, question=input)

        answer_text = self.llm.invoke(prompt)

        return {
            "role": "assistant",
            "content": answer_text,
        }
