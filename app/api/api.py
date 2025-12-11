from fastapi import FastAPI
from pydantic import BaseModel
from uuid import uuid4

from app.agente.rag_service import RAGService
from fastapi.middleware.cors import CORSMiddleware
from app.api import endpoints
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Movie RAG Chatbot API",
    description="API REST para chatbot RAG con Qwen3 + ChromaDB",
    version="1.0.0"
)

origins = [
    "http://localhost:8080",
    "http://localhost:8501",
    "http://0.0.0.0:8501",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(endpoints.router)