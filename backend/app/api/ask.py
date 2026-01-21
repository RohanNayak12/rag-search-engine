from fastapi import APIRouter
from nltk.data import retrieve
from pydantic import BaseModel
from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parents[3]
INDEX_PATH=PROJECT_ROOT / "data" / "faiss" / "index.faiss"
METADATA_PATH=PROJECT_ROOT / "data" / "embeddings" / "metadata.jsonl"
CHUNKS_PATH=PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
PROMPT_PATH=PROJECT_ROOT / "backend" / "app" / "prompts" / "rag_prompt.txt"

from backend.app.services.retrieval import Retrievar
from backend.app.services.rag import RAGService

router=APIRouter()
retriever=Retrievar(
    INDEX_PATH,
    METADATA_PATH,
    CHUNKS_PATH
)
rag=RAGService(retriever,PROMPT_PATH)

class AskRequest(BaseModel):
    question: str
    top_k: int=5

@router.post("/ask")
def ask(request: AskRequest):
    ans=rag.answer(request.question,top_k=request.top_k)
    return {
        "question": request.question,
        "answer": ans
    }