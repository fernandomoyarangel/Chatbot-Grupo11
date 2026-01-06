from functools import lru_cache
from typing import Optional
from langchain_core.prompts import PromptTemplate

from .tools.rag_tools import retrieve_context_data
from ..core.config import Settings
from .uc3m_llm import UC3MChatModel
from .translation_service import TranslationService
from app.core.utils import build_doc_key, load_topic_maps
from pathlib import Path

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
        doc_topics, topics_info = load_topic_maps()
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
                "sources": [],
                "suggestions": []
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

            doc_key = build_doc_key(meta, doc.page_content)
            topic_id = doc_topics.get(doc_key, -1)
            topic_words = topics_info.get(str(topic_id), {}).get("top_words", [])

            sources.append({
                "source": src,
                "content": doc.page_content,
                "topic_id": topic_id,
                "topic_words": topic_words,
            })
        negative_markers = [
            "I am sorry, I cannot find",
            "Lo siento, no puedo encontrar",
            "no he encontrado información",
            "cannot find that information"
        ]

        is_negative_answer = any(marker in answer_text for marker in negative_markers)
        suggestions=[]
        if is_negative_answer:
            sources = []
        else:
            suggestions = self._generate_suggestions(answer_text, serialized, input, language)
        return {
            "role": "assistant",
            "content": answer_text,
            "sources": sources,
            "suggestions": suggestions
        }

    def summarize_document(self, filename: str, language: str = "english") -> str:
        """
        Lee el archivo completo de data/source_docs y genera un resumen.
        """
        try:
            # 1. Construir la ruta al archivo
            # Asumimos que rag_service.py está en app/agente/, así que subimos 2 niveles para llegar a la raíz
            project_root = Path(__file__).resolve().parents[2]
            file_path = project_root / "data" / "source_docs" / filename

            # 2. Verificar existencia
            if not file_path.exists():
                return f"Error: No se encuentra el archivo fuente original ({filename})."

            # 3. Leer contenido
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                full_text = f.read()

            # 4. TRUNCADO DE SEGURIDAD (IMPORTANTE)
                max_chars = 25000
            if len(full_text) > max_chars:
                full_text = full_text[:max_chars] + "\n... [Texto truncado por longitud] ..."

            # 5. Crear Prompt
            if language.lower() == "spanish":
                prompt = (
                    "Eres un experto en cine. A continuación se presenta el contenido de un documento sobre una película. "
                    "Genera un resumen completo y estructurado en Español que capture la trama principal y los detalles clave.\n\n"
                    f"DOCUMENTO:\n{full_text}\n\nRESUMEN:"
                )
            else:
                prompt = (
                    "You are a movie expert. Below is the content of a document about a movie. "
                    "Generate a complete and structured summary in English capturing the main plot and key details.\n\n"
                    f"DOCUMENT:\n{full_text}\n\nSUMMARY:"
                )

            # 6. Invocar LLM
            response_obj = self.llm.invoke(prompt)

            if hasattr(response_obj, 'content'):
                return response_obj.content
            return str(response_obj)

        except Exception as e:
            return f"Error al procesar el archivo: {str(e)}"

    def _generate_suggestions(self, answer_text: str, context_text: str, question_text: str, language: str) -> list:
        """Genera 2 preguntas cortas de seguimiento basadas en el contexto y la pregunta original."""
        try:
            safe_context = context_text[:3000]

            if language == "spanish":
                prompt = (
                    f"Tu tarea es sugerir preguntas futuras para el usuario basadas en el siguiente CONTEXTO.\n"
                    f"PREGUNTA ORIGINAL DEL USUARIO: '{question_text}'\n"
                    f"CONTEXTO DISPONIBLE:\n{safe_context}\n\n"
                    f"Genera exactamente 2 preguntas breves, sobre el tema global del contexto.\n\n"
                    f"REGLAS:\n"
                    f"- Las preguntas deben tener respuesta en el CONTEXTO.\n"
                    f"- Sé conciso.\n\n"
                    f"Formato: Solo las 2 preguntas separadas por saltos de línea.\n"
                    f"PREGUNTAS:"
                )
            else:
                prompt = (
                    f"Your task is to suggest follow-up questions based on the provided CONTEXT.\n"
                    f"ORIGINAL USER QUESTION: '{question_text}'\n"
                    f"AVAILABLE CONTEXT:\n{safe_context}\n\n"
                    f"- Questions MUST be answerable using the CONTEXT.\n"
                    f"Generate exactly 2 brief questions, about the main theme of the context.\n\n"
                    f"RULES:\n"
                    f"- Be concise.\n\n"
                    f"Format: Only the 2 questions separated by newlines.\n"
                    f"QUESTIONS:"
                )

            response = self.llm.invoke(prompt)
            text = response.content if hasattr(response, 'content') else str(response)

            # Limpiar y filtrar
            suggestions = [line.strip() for line in text.split('\n') if line.strip()]

            clean_suggestions = []
            for s in suggestions[:2]:  # Máximo 2
                clean_s = s.lstrip("1234567890.-• ").strip()
                if clean_s:
                    clean_suggestions.append(clean_s)

            return clean_suggestions
        except Exception as e:
            print(f"Error generando sugerencias: {e}")
            return []