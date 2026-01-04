from functools import lru_cache
from typing import Optional
from langchain_core.prompts import PromptTemplate

from .tools.rag_tools import retrieve_context_data
from ..core.config import Settings
from .uc3m_llm import UC3MChatModel
from .translation_service import TranslationService


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
        self.translation_service = TranslationService()
        

    def _build_context(self, query: str):
        return retrieve_context_data(query=query, k=self.k)

    def process_query(self, input: str, session_id: str, language: str = "english"):
       
        # Step 1: Translate question if in Spanish mode
        original_input = input
        if language == "spanish":
            input = self.translation_service.translate_es_to_en(input)
            # Log translation for debugging
            print(f"\n{'='*70}")
            print(f"TRANSLATION DEBUG")
            print(f"{'='*70}")
            print(f"Original (ES): {original_input}")
            print(f"Translated (EN): {input}")
            print(f"{'='*70}\n")
        
        # Step 2: Build context using (possibly translated) query
        serialized, docs = self._build_context(input)
        
        # Step 3: Fallback message if no documents found
        if not docs:
            no_info_message = "I am sorry, I could not find any information about that in the movie database."
            
            # Translate fallback message if in Spanish mode
            if language == "spanish":
                no_info_message = self.translation_service.translate_en_to_es(no_info_message)
            
            return {
                "role": "assistant",
                "content": no_info_message,
                "sources": []
            }

        # Step 4: Generate prompt and invoke LLM (always in English)
        prompt_text = self.prompt_template.format(context=serialized, question=input)
        response_obj = self.llm.invoke(prompt_text)

        # Step 5: Extract answer text
        if hasattr(response_obj, 'content'):
            answer_text = response_obj.content
        else:
            answer_text = str(response_obj)
        
        # Step 6: Translate answer back to Spanish if needed
        if language == "spanish":
            answer_text = self.translation_service.translate_en_to_es(answer_text)

        # Step 7: Build sources list
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
