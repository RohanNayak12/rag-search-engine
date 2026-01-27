from fastapi import APIRouter
from pathlib import Path

from backend.app.api.ask import retriever
from backend.app.services.incremental_indexing import index_new_pdfs

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[3]

@router.post("/index-new")
def index_new():
    msg = index_new_pdfs(
        raw_pdf_dir=PROJECT_ROOT / "data" / "raw_pdfs",
        index_path=PROJECT_ROOT / "data" / "faiss" / "index.faiss",
        metadata_path=PROJECT_ROOT / "data" / "embeddings" / "metadata.jsonl",
        indexed_files_path=PROJECT_ROOT / "data" / "indexed_files.json",
    )
    retriever.reload()
    return {"status": msg}
