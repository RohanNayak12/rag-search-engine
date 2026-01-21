import numpy as np
from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parents[1]

from backend.app.services.index import FaissIndexService

VECTORS_PATH = PROJECT_ROOT / "data" / "embeddings" / "vectors.npy"
FAISS_DIR = PROJECT_ROOT / "data" / "faiss"
INDEX_PATH = FAISS_DIR / "index.faiss"

def main():
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    vectors=np.load(VECTORS_PATH)
    print("Vectors Loaded ",vectors.shape)

    dim=vectors.shape[1]
    index=FaissIndexService(dim=dim)
    index.add(vectors)
    index.save(INDEX_PATH)
    print("Index built and saved")

if __name__=="__main__":
    main()