
from fastapi import FastAPI
from pydantic import BaseModel
from uuid import uuid4

from app.agente.rag_service import RAGService
from app.topic_modeling.service import run_topic_modeling,build_topic_visualization

from fastapi import APIRouter,  HTTPException


router = APIRouter()

class ChatRequest(BaseModel):
    question: str
    language: str = "english"  # "english" or "spanish"


class ChatResponse(BaseModel):
    session_id: str
    answer: dict  # contiene content, role, y sources





rag = RAGService()

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):

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
    language: str = "english"

class TopicModelResponse(BaseModel):
    output_dir: str
    model_dir: str
    topic_count: int
    document_count: int

@router.post("/topics", response_model=TopicModelResponse)
def build_topics(req: TopicModelRequest):
    try:
        result = run_topic_modeling(language=req.language)
        return TopicModelResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    
class TopicVisualizationRequest(BaseModel):
    top_n_topics: int = 10
    top_n_words: int = 10
    rebuild_if_missing: bool = False
    language: str = "english"

class TopicVisualizationResponse(BaseModel):
    plot_html: str
    topics: list

@router.post("/topics/visualize", response_model=TopicVisualizationResponse)
def visualize_topics(req: TopicVisualizationRequest):
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
    filename: str
    language: str = "english"

class SummaryResponse(BaseModel):
    summary: str

@router.post("/summary", response_model=SummaryResponse)
def get_summary(req: SummaryRequest):
    try:
        # Usamos la variable 'rag' que ya tienes definida arriba en este archivo
        summary_text = rag.summarize_document(filename=req.filename, language=req.language)
        return SummaryResponse(summary=summary_text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
# -------------------
@router.get("/health")
def health():
    return {"status": "ok"}

class SurpriseRequest(BaseModel):
    language: str = "english"

class SurpriseResponse(BaseModel):
    curiosity: str

@router.post("/surprise", response_model=SurpriseResponse)
def get_surprise(req: SurpriseRequest):
    try:
        # Usamos la instancia 'rag' que ya tienes creada en este archivo
        text = rag.get_curiosity(language=req.language)
        return SurpriseResponse(curiosity=text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))