from fastapi import APIRouter

from backend.app.api.ask import retriever

router = APIRouter()
@router.get("/stats")
def stats():
    return {
        "total_vectors":retriever.idx.ntotal,
        "total_chunks":len(retriever.metadata)
    }