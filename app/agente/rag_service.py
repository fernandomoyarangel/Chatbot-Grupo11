import time
from functools import lru_cache
from typing import Optional
from langchain_core.prompts import PromptTemplate

from .tools.rag_tools import retrieve_context_data
from ..core.evaluation_logger import log_automated_metric
from ..core.config import Settings
# from .uc3m_llm import UC3MChatModel
from .translation_service import TranslationService
from app.core.utils import build_doc_key, load_topic_maps
from app.agente.uc3m_llm import get_llm_model
from pathlib import Path
import random

# @lru_cache(maxsize=None)
# def get_llm_model():
#     return UC3MChatModel(
#         model="llama3.1:8b"   # o "qwen3:8b" segun lo que os indiquen
#     )


class RAGService:
    def __init__(self, idioma: str = "english", k: int = 20):
        self.llm = get_llm_model()
        self.k = k
        self.prompt_template = Settings.get_prompt()
        self.surprise_prompt = Settings.get_surprise_prompt()
        self.translation_service = TranslationService()
        

    def _build_context(self, query: str):
        # Recuperamos los documentos
        serialized_full, docs = retrieve_context_data(query=query, k=self.k)
        
        return serialized_full, docs

    def process_query(self, input: str, session_id: str, language: str = "english"):
        start_time = time.time()
        doc_topics, topics_info = load_topic_maps()
        
        # Step 1: Translate question if in Spanish mode
        original_input = input
        if language == "spanish":
            input = self.translation_service.translate_es_to_en(input)
            print(f"\n{'='*70}\nTRANSLATION DEBUG\nOriginal: {original_input}\nTranslated: {input}\n{'='*70}\n")
        
        # Step 2: Build context
        serialized, docs = self._build_context(input)
        
        # Step 3: Fallback if no docs
        if not docs:
            no_info_message = "I am sorry, I could not find any information about that in the movie database."
            if language == "spanish":
                no_info_message = self.translation_service.translate_en_to_es(no_info_message)
            return {
                "role": "assistant",
                "content": no_info_message,
                "sources": [],
                "suggestions": []
            }

        # Step 4: Generate prompt (Direct generation in target language)
        if language == "spanish":
            lang_instruction = (
                "OUTPUT INSTRUCTION: The user is asking in Spanish. "
                "Answer ONLY in Spanish. Translate the information from the context naturally. "
                "Maintain Markdown formatting (lists, bolding) strictly."
            )
        else:
            lang_instruction = "Answer in English."

        prompt_text = self.prompt_template.format(
            context=serialized, 
            question=input, 
            language_instruction=lang_instruction
        )
        
        # Al invocar esto, uc3m_llm.py ya limpia el <think> internamente
        response_obj = self.llm.invoke(prompt_text)

        # Step 5: Extract answer text (Ya viene limpio)
        if hasattr(response_obj, 'content'):
            answer_text = response_obj.content
        else:
            answer_text = str(response_obj)
        

        # --- FILTRADO INTELIGENTE DE FUENTES ---
        final_docs = []
        answer_lower = answer_text.lower()
        query_lower = original_input.lower()
        
        for doc in docs:
            meta = doc.metadata or {}
            movie_title = str(meta.get("name", "")).lower()
            
            # REGLA A: Título en respuesta
            if movie_title and movie_title in answer_lower:
                final_docs.append(doc)
                continue
            # REGLA B: Título en pregunta
            if movie_title and movie_title in query_lower:
                final_docs.append(doc)
                continue
        
        if not final_docs and docs:
            final_docs = docs[:3]

        # 7. CONSTRUCCIÓN DE SOURCES
        sources = []
        seen_sources = set()

        for doc in final_docs:
            meta = doc.metadata or {}
            source_name = meta.get("source", "Unknown")
            doc_key = build_doc_key(meta, doc.page_content)
            
            if source_name not in seen_sources:
                sources.append({
                    "source": source_name,
                    "content": doc.page_content,
                    "topic": str(doc_topics.get(doc_key, "Unknown")),
                    "box_office": meta.get("box_office", "N/A"),
                    "year": meta.get("year", "N/A")
                })
                seen_sources.add(source_name)

        # 8. DETECCIÓN NEGATIVA
        negative_markers = [
            "I am sorry, I cannot find", "Lo siento, no puedo encontrar",
            "no he encontrado información", "cannot find that information",
            "I don't see any information", "don't have access to",
            "Unfortunately, I don't see", "does not include information"
        ]
        is_negative_answer = any(marker in answer_text for marker in negative_markers)
        
        if is_negative_answer:
            sources = []
            suggestions = []
        else:
            suggestions = self._generate_suggestions(answer_text, serialized, input, language)

        # 9. LOGS
        end_time = time.time()
        log_automated_metric(
            question=original_input,
            answer=answer_text,
            time_taken=end_time - start_time,
            is_negative_answer=is_negative_answer,
            source_count=len(sources)
        )

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

    def get_curiosity(self, language: str = "english") -> str:
        """Genera un dato curioso aleatorio buscando en la base de datos."""
        try:
            # 1. Palabras clave (igual que antes)
            search_terms = ["plot twist", "ending", "character", "death", "wedding", "war", "love", "secret", "family"]
            random_term = random.choice(search_terms)

            # 2. Contexto (igual que antes)
            serialized_context, docs = retrieve_context_data(query=random_term, k=3)

            if not docs:
                return "No he encontrado cintas en la filmoteca hoy."

            # 3. Invocar LLM
            prompt_text = self.surprise_prompt.format(context=serialized_context)
            response_obj = self.llm.invoke(prompt_text)

            fact_text = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)

            # --- LIMPIEZA DE RESPUESTA (NUEVO) ---
            # Quitamos comillas extra, espacios y posibles introducciones que se hayan colado
            fact_text = fact_text.strip().strip('"').strip("'")

            # Si el modelo sigue diciendo "Here is a detail:", lo cortamos a la fuerza
            if ":" in fact_text[:20]:
                fact_text = fact_text.split(":", 1)[1].strip()

            # 4. Traducción y Formato Final
            if language == "spanish":
                fact_text = self.translation_service.translate_en_to_es(fact_text)

                # Normalizamos el inicio para que quede perfecto
                # Quitamos variantes para unificar
                lower_fact = fact_text.lower()
                if lower_fact.startswith("sabías que"):
                    fact_text = "¿Sabías que" + fact_text[10:]  # Reconstruimos con interrogación si falta
                elif lower_fact.startswith("¿sabías que"):
                    pass  # Ya está bien
                else:
                    fact_text = "¿Sabías que... " + fact_text

            return fact_text

        except Exception as e:
            return f"Error generando curiosidad: {str(e)}"