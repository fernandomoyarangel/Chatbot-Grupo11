import os
import requests
from typing import Optional, List, Any

from langchain_core.language_models.llms import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from dotenv import load_dotenv

load_dotenv()


UC3M_URL = "https://yiyuan.tsc.uc3m.es/api/generate"
DEFAULT_MODEL = "qwen3:8b" 


class UC3MLLM(LLM):

    model: str = DEFAULT_MODEL

    @property
    def _llm_type(self) -> str:
        return "uc3m_api"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }

        headers = {
            "Content-Type": "application/json",
            "X-API-KEY": os.getenv("UC3M_API_KEY"),
        }

        response = requests.post(
            UC3M_URL,
            headers=headers,
            json=payload,
            verify=False,
            timeout=120,
        )

        response.raise_for_status()

        data = response.json()

        return data.get("response") or data.get("output") or ""
