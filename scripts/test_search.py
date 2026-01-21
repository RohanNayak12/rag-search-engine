from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parents[1]

from backend.app.services.retrieval import Retrievar

INDEX_PATH=PROJECT_ROOT/"data"/"faiss"/"index.faiss"
METADATA_PATH = PROJECT_ROOT / "data" / "embeddings" / "metadata.jsonl"
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"

def main():
    retriever=Retrievar(INDEX_PATH,METADATA_PATH,CHUNKS_PATH)
    query="What is Embeddings?"
    res=retriever.search(query)
    for r in res:
        print(r)

if __name__=="__main__":
    main()