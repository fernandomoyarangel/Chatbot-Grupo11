
from fastapi import FastAPI
from pydantic import BaseModel
from uuid import uuid4

from agente.rag_service import RAGService
from fastapi import APIRouter


router = APIRouter()

class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    session_id: str
    answer: dict


rag = RAGService()

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):

    session_id = str(uuid4())

    response = rag.process_query(
        input=req.question,
        session_id=session_id
    )

    return ChatResponse(
        session_id=session_id,
        answer=response
    )


@router.get("/health")
def health():
    return {"status": "ok"}