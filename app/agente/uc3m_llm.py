import os
import re
import requests
from typing import List, Optional, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from dotenv import load_dotenv
from app.core.config import settings

from functools import lru_cache



UC3M_URL = settings.UC3M_URL
DEFAULT_MODEL = settings.DEFAULT_MODEL

load_dotenv()

def _messages_to_prompt(messages: List[BaseMessage]) -> str:
    """
    Convierte una lista de mensajes de LangChain en un prompt de texto plano.

    Args:
        messages (List[BaseMessage]): Mensajes con roles y contenido.

    Returns:
        str: Prompt formateado como "role: content".
    """
    parts = []
    for m in messages:
        role = getattr(m, "type", "user")
        parts.append(f"{role}: {m.content}")
    return "\n".join(parts)


class UC3MChatModel(BaseChatModel):
    """
    Implementación de un modelo de chat basado en la API UC3M.

    Adapta el formato de mensajes de LangChain a un prompt plano y
    devuelve respuestas compatibles con ChatResult.
    """
    model: str = DEFAULT_MODEL

    temperature: float = 0.5

    @property
    def _llm_type(self) -> str:
        """
        Identificador interno del tipo de LLM.

        Returns:
            str: Nombre del tipo de modelo.
        """
        return "uc3m_chat_api"

    @property
    def _identifying_params(self):
        """
        Parámetros que identifican de forma única al modelo.

        Returns:
            dict: Modelo y temperatura configurados.
        """
        return {"model": self.model, "temperature": self.temperature}

    def _generate(self, messages: List[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        """
        Genera una respuesta a partir de una lista de mensajes.

        Convierte los mensajes en un prompt, llama a la API UC3M y limpia
        etiquetas internas del modelo antes de devolver la respuesta.

        Args:
            messages (List[BaseMessage]): Mensajes de entrada.
            stop: Tokens de parada (no usado).
            run_manager: Gestor de ejecución (no usado).

        Returns:
            ChatResult: Resultado con el mensaje generado por el modelo.
        """
        prompt = _messages_to_prompt(messages)
        payload = {"model": self.model, "prompt": prompt, "stream": False, "temperature": self.temperature}
        headers = {
            "Content-Type": "application/json",
            "X-API-KEY": os.getenv("UC3M_API_KEY"),
        }
        resp = requests.post(UC3M_URL, headers=headers, json=payload, verify=False, timeout=120)
        resp.raise_for_status()
        text = resp.json().get("response") or resp.json().get("output") or ""
        clean_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=clean_text))])

@lru_cache(maxsize=None)
def get_llm_model():
    """
    Devuelve una instancia cacheada del modelo UC3MChatModel.

    Returns:
        UC3MChatModel: Modelo de chat configurado.
    """
    return UC3MChatModel(
        model=settings.DEFAULT_MODEL,
        temperature=0.5 
    )
