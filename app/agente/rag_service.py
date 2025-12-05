from functools import lru_cache
from pydantic import BaseModel

from langchain.agents import create_agent
from langchain_community.llms import Ollama

from .tools.rag_tools import get_rag_tools
from ..core.config import Settings
from .uc3m_llm import UC3MLLM

class Response(BaseModel):
    content:str


@lru_cache(maxsize=None)
def get_llm_model():
    return UC3MLLM(
        model="llama3.1:8b"   # o "qwen3:8b" según lo que os indiquen
    )

class RAGService:

    def __init__(self, idioma: str = 'español'):

        model = get_llm_model()
        tools = get_rag_tools()

        prompt = Settings.get_prompt()

        self.agent = create_agent(
            model=model,
            tools=tools,
            system_prompt=prompt
        )

    def process_query(self, input: str, session_id: str):
        """
        Método empleado para que el agente procese una petición.

        Parámetros:  
        - input (string): petición del usuario  
        - session_id (string): identificador de sesión del usuario

        Returns:  
        - JSON: la respuesta del agente en formato JSON
        """
        config = {
            "configurable": {
                "thread_id": session_id,
            },
            "tags": ["chatbot"],
            "metadata": {
                "session_id": session_id,
            }
        }
        
        response = self.agent.invoke(
            {"messages": {"role": "user", "content": input}},
            config=config)

        last_message = response["messages"][-1]

        # Extraemos SOLO el texto producido por el UC3M LLM
        answer_text = last_message.content

        return {
            "role": "assistant",
            "content": answer_text
        }