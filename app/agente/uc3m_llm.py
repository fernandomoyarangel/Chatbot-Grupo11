import os
import requests
from typing import List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from dotenv import load_dotenv
from app.core.config import settings



UC3M_URL = settings.UC3M_URL
DEFAULT_MODEL = settings.DEFAULT_MODEL

load_dotenv()

def _messages_to_prompt(messages: List[BaseMessage]) -> str:
    """
    Convierte la lista de mensajes en un prompt plano role:content.
    Ajusta este formato si la API cambiara.
    """
    parts = []
    for m in messages:
        role = getattr(m, "type", "user")
        parts.append(f"{role}: {m.content}")
    return "\n".join(parts)


class UC3MChatModel(BaseChatModel):
    model: str = DEFAULT_MODEL

    @property
    def _llm_type(self) -> str:
        return "uc3m_chat_api"

    @property
    def _identifying_params(self):
        return {"model": self.model}

    def _generate(self, messages: List[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        prompt = _messages_to_prompt(messages)
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        headers = {
            "Content-Type": "application/json",
            "X-API-KEY": os.getenv("UC3M_API_KEY"),
        }
        resp = requests.post(UC3M_URL, headers=headers, json=payload, verify=False, timeout=120)
        resp.raise_for_status()
        text = resp.json().get("response") or resp.json().get("output") or ""
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])
