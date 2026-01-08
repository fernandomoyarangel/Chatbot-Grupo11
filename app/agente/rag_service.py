import time
from functools import lru_cache
from typing import Optional
from langchain_core.prompts import PromptTemplate

from .tools.rag_tools import retrieve_context_data
from ..core.evaluation_logger import log_automated_metric
from ..core.config import Settings
from .translation_service import TranslationService
from app.core.utils import build_doc_key, load_topic_maps
from app.agente.uc3m_llm import get_llm_model
from pathlib import Path
import random


class RAGService:
    """
    Servicio principal de Retrieval-Augmented Generation (RAG) para consultas sobre las películas

    Este servicio:
    - Recupera contexto relevante desde una base documental.
    - Genera respuestas mediante un LLM.
    - Soporta consultas en inglés y español.
    - Incluye fuentes, sugerencias de seguimiento y métricas automáticas.
    """

    def __init__(self, idioma: str = "english", k: int = 20):
        """
        Inicializa el servicio RAG.

        Args:
            idioma (str): Idioma por defecto del servicio ("english" o "spanish").
            k (int): Número máximo de documentos a recuperar del sistema de búsqueda.
        """
        self.llm = get_llm_model()
        self.k = k
        self.prompt_template = Settings.get_prompt()
        self.surprise_prompt = Settings.get_surprise_prompt()
        self.translation_service = TranslationService()
        

    def _build_context(self, query: str):
        """
        Recupera y serializa el contexto relevante para una consulta dada.

        Args:
            query (str): Consulta del usuario ya normalizada (normalmente en inglés).

        Returns:
            Tuple[str, List[Document]]:
                - Texto serializado del contexto recuperado.
                - Lista de documentos originales recuperados.
        """
        serialized_full, docs = retrieve_context_data(query=query, k=self.k)
        
        return serialized_full, docs

    def process_query(self, input: str, session_id: str, language: str = "english"):
        """
        Procesa una consulta del usuario utilizando un flujo RAG completo.

        El flujo incluye:
        - Traducción automática si la consulta está en español.
        - Recuperación de documentos relevantes.
        - Generación de respuesta mediante LLM.
        - Selección y filtrado de fuentes.
        - Generación de sugerencias de seguimiento.
        - Registro de métricas automáticas.

        Args:
            input (str): Consulta original del usuario.
            session_id (str): Identificador de sesión para trazabilidad.
            language (str): Idioma de la consulta ("english" o "spanish").

        Returns:
            dict: Diccionario con la respuesta estructurada:
                - role (str): Rol del mensaje ("assistant").
                - content (str): Respuesta generada por el modelo.
                - sources (list): Lista de fuentes documentales utilizadas.
                - suggestions (list): Preguntas sugeridas de seguimiento.
        """
        start_time = time.time()
        doc_topics, topics_info = load_topic_maps()
        

        original_input = input
        if language == "spanish":
            input = self.translation_service.translate_es_to_en(input)
            print(f"\n{'='*70}\nTRANSLATION DEBUG\nOriginal: {original_input}\nTranslated: {input}\n{'='*70}\n")
        

        serialized, docs = self._build_context(input)
        

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
        
 
        response_obj = self.llm.invoke(prompt_text)


        if hasattr(response_obj, 'content'):
            answer_text = response_obj.content
        else:
            answer_text = str(response_obj)
        

        final_docs = []
        answer_lower = answer_text.lower()
        query_lower = original_input.lower()
        
        for doc in docs:
            meta = doc.metadata or {}
            movie_title = str(meta.get("name", "")).lower()
            

            if movie_title and movie_title in answer_lower:
                final_docs.append(doc)
                continue

            if movie_title and movie_title in query_lower:
                final_docs.append(doc)
                continue
        
        if not final_docs and docs:
            final_docs = docs[:3]


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
        Genera un resumen estructurado de un documento completo de la base de datos.

        El documento se lee desde la carpeta `data/source_docs` y se trunca
        automáticamente si excede la longitud máxima permitida por el modelo.

        Args:
            filename (str): Nombre del archivo fuente a resumir.
            language (str): Idioma del resumen ("english" o "spanish").

        Returns:
            str: Resumen generado del documento o mensaje de error en caso de fallo.
        """
        try:

            project_root = Path(__file__).resolve().parents[2]
            file_path = project_root / "data" / "source_docs" / filename


            if not file_path.exists():
                return f"Error: No se encuentra el archivo fuente original ({filename})."


            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                full_text = f.read()


                max_chars = 25000
            if len(full_text) > max_chars:
                full_text = full_text[:max_chars] + "\n... [Texto truncado por longitud] ..."

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

            response_obj = self.llm.invoke(prompt)

            if hasattr(response_obj, 'content'):
                return response_obj.content
            return str(response_obj)

        except Exception as e:
            return f"Error al procesar el archivo: {str(e)}"

    def _generate_suggestions(self, answer_text: str, context_text: str, question_text: str, language: str) -> list:
        """
        Genera preguntas de seguimiento basadas en el contexto recuperado.

        Las preguntas:
        - Están relacionadas con el tema global del contexto.
        - Son respondibles usando exclusivamente el contexto disponible.
        - Se limitan a dos preguntas breves.

        Args:
            answer_text (str): Respuesta generada por el LLM.
            context_text (str): Contexto textual utilizado en la generación.
            question_text (str): Pregunta original del usuario.
            language (str): Idioma de salida ("english" o "spanish").

        Returns:
            list: Lista de exactamente 2 preguntas sugeridas (si es posible).
        """
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

            suggestions = [line.strip() for line in text.split('\n') if line.strip()]

            clean_suggestions = []
            for s in suggestions[:2]: 
                clean_s = s.lstrip("1234567890.-• ").strip()
                if clean_s:
                    clean_suggestions.append(clean_s)

            return clean_suggestions
        except Exception as e:
            print(f"Error generando sugerencias: {e}")
            return []

    def get_curiosity(self, language: str = "english") -> str:
        """
        Genera un dato curioso aleatorio sobre películas utilizando el sistema RAG.

        El método:
        - Selecciona un término de búsqueda aleatorio.
        - Recupera contexto relevante.
        - Genera un dato curioso mediante un prompt específico.
        - Traduce y adapta el resultado al idioma solicitado.

        Args:
            language (str): Idioma del dato curioso ("english" o "spanish").

        Returns:
            str: Dato curioso generado o mensaje de error si no hay resultados.
        """
        try:

            search_terms = ["plot twist", "ending", "character", "death", "wedding", "war", "love", "secret", "family"]
            random_term = random.choice(search_terms)


            serialized_context, docs = retrieve_context_data(query=random_term, k=3)

            if not docs:
                return "No he encontrado cintas en la filmoteca hoy."


            prompt_text = self.surprise_prompt.format(context=serialized_context)
            response_obj = self.llm.invoke(prompt_text)

            fact_text = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)


            fact_text = fact_text.strip().strip('"').strip("'")


            if ":" in fact_text[:20]:
                fact_text = fact_text.split(":", 1)[1].strip()

            if language == "spanish":
                fact_text = self.translation_service.translate_en_to_es(fact_text)


                lower_fact = fact_text.lower()
                if lower_fact.startswith("sabías que"):
                    fact_text = "¿Sabías que" + fact_text[10:] 
                elif lower_fact.startswith("¿sabías que"):
                    pass  
                else:
                    fact_text = "¿Sabías que... " + fact_text

            return fact_text

        except Exception as e:
            return f"Error generando curiosidad: {str(e)}"