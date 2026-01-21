from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from backend.app.services.retrieval import Retrievar
from backend.app.services.rag import RAGService

INDEX_PATH = PROJECT_ROOT / "data" / "faiss" / "index.faiss"
METADATA_PATH = PROJECT_ROOT / "data" / "embeddings" / "metadata.jsonl"
PROMPT_PATH = PROJECT_ROOT / "backend" / "app" / "prompts" / "rag_prompt.txt"
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"

def main():
    retriever=Retrievar(INDEX_PATH, METADATA_PATH,CHUNKS_PATH)
    rag=RAGService(retriever, PROMPT_PATH)

    question = "What is transformer architecture?"
    answer=rag.answer(question)
    print("Question: ",question)
    print("Answer: ",answer)

if __name__ == "__main__":
    main()