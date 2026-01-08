
from fastapi import FastAPI
from pydantic import BaseModel
from uuid import uuid4

from app.agente.rag_service import RAGService
from app.topic_modeling.service import run_topic_modeling,build_topic_visualization

from fastapi import APIRouter,  HTTPException


router = APIRouter()

class ChatRequest(BaseModel):
    """
    Petición de entrada para el endpoint de chat.
    """
    question: str
    language: str = "english"  # "english" or "spanish"


class ChatResponse(BaseModel):
    """
    Respuesta del endpoint de chat.
    """
    session_id: str
    answer: dict  





rag = RAGService()

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Procesa una consulta de chat mediante el servicio RAG.

    Args:
        req (ChatRequest): Pregunta del usuario e idioma.

    Returns:
        ChatResponse: Respuesta generada y session_id asociado.
    """

    session_id = str(uuid4())

    response = rag.process_query(
        input=req.question,
        session_id=session_id,
        language=req.language
    )

    return ChatResponse(
        session_id=session_id,
        answer=response  
    )


class TopicModelRequest(BaseModel):
    """
    Petición para construir el modelo de tópicos.
    """
    language: str = "english"

class TopicModelResponse(BaseModel):
    """
    Resultado de la construcción del modelo de tópicos.
    """
    output_dir: str
    model_dir: str
    topic_count: int
    document_count: int

@router.post("/topics", response_model=TopicModelResponse)
def build_topics(req: TopicModelRequest):
    """
    Ejecuta el modelado de tópicos sobre el corpus.

    Args:
        req (TopicModelRequest): Idioma del corpus.

    Returns:
        TopicModelResponse: Información del modelo generado.
    """
    try:
        result = run_topic_modeling(language=req.language)
        return TopicModelResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    
class TopicVisualizationRequest(BaseModel):
    """
    Petición para generar la visualización de tópicos.
    """
    top_n_topics: int = 10
    top_n_words: int = 10
    rebuild_if_missing: bool = False
    language: str = "english"

class TopicVisualizationResponse(BaseModel):
    """
    Respuesta con la visualización de tópicos.
    """
    plot_html: str
    topics: list

@router.post("/topics/visualize", response_model=TopicVisualizationResponse)
def visualize_topics(req: TopicVisualizationRequest):
    """
    Genera una visualización interactiva del modelo de tópicos.

    Args:
        req (TopicVisualizationRequest): Parámetros de visualización.

    Returns:
        TopicVisualizationResponse: HTML del gráfico y tópicos extraídos.
    """
    try:
        payload = build_topic_visualization(
            top_n_topics=req.top_n_topics,
            top_n_words=req.top_n_words,
        )
        return TopicVisualizationResponse(**payload)
    except FileNotFoundError:
        if req.rebuild_if_missing:
            run_topic_modeling(language=req.language)
            payload = build_topic_visualization(
                top_n_topics=req.top_n_topics,
                top_n_words=req.top_n_words,
            )
            return TopicVisualizationResponse(**payload)
        raise HTTPException(status_code=404, detail="Topic model not found. Run /topics first.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

class SummaryRequest(BaseModel):
    """
    Petición para resumir un documento.
    """
    filename: str
    language: str = "english"

class SummaryResponse(BaseModel):
    """
    Respuesta con el resumen del documento.
    """
    summary: str

@router.post("/summary", response_model=SummaryResponse)
def get_summary(req: SummaryRequest):
    """
    Genera un resumen de un documento almacenado.

    Args:
        req (SummaryRequest): Nombre del archivo e idioma.

    Returns:
        SummaryResponse: Resumen generado por el modelo.
    """
    try:
        summary_text = rag.summarize_document(filename=req.filename, language=req.language)
        return SummaryResponse(summary=summary_text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@router.get("/health")
def health():
    """
    Endpoint de comprobación de estado del servicio.

    Returns:
        dict: Estado del servicio.
    """
    return {"status": "ok"}

class SurpriseRequest(BaseModel):
    """
    Petición para obtener un dato curioso.
    """
    language: str = "english"

class SurpriseResponse(BaseModel):
    """
    Respuesta con el dato curioso generado.
    """
    curiosity: str

@router.post("/surprise", response_model=SurpriseResponse)
def get_surprise(req: SurpriseRequest):
    """
    Devuelve un dato curioso aleatorio sobre películas.

    Args:
        req (SurpriseRequest): Idioma de la respuesta.

    Returns:
        SurpriseResponse: Dato curioso generado.
    """
    try:
        text = rag.get_curiosity(language=req.language)
        return SurpriseResponse(curiosity=text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))