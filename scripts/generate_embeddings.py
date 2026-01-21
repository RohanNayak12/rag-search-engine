import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from backend.app.services.embeddings import EmbeddingsService

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
EMBEDDING_DIR = PROJECT_ROOT / "data" / "embeddings"
VECTORS_PATH = EMBEDDING_DIR / "vectors.npy"
METADATA_PATH = EMBEDDING_DIR / "metadata.jsonl"

def main():
    print(os.getcwd())
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
    texts=[]
    metadata=[]

    with CHUNKS_PATH.open("r",encoding="utf-8") as f:
        for line in f:
            record=json.loads(line)
            texts.append(record["text"])
            metadata.append({
                "doc_id":record["doc_id"],
                "page":record["page"],
                "chunk_id":record["chunk_id"],
            })
        print(f"Loaded {len(texts)} chunks")

        embedder=EmbeddingsService()
        embeddings=embedder.embed_text(texts)

        print("Embedding shape: ",embeddings.shape)
        np.save(VECTORS_PATH,embeddings)

        with METADATA_PATH.open("w",encoding="utf-8") as f:
            for m in metadata:
                f.write(json.dumps(m)+"\n")
        print("Embeddings+Metadata saved")

if __name__=="__main__":
    main()