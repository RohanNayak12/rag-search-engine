import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class Retrievar:
    def __init__(self, idx_path: Path, metadata_path: Path, chunks_path: Path):
        self.idx_path = idx_path
        self.metadata_path = metadata_path
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        if not idx_path.exists() or not metadata_path.exists():
            print("⚠️ No FAISS index found yet. Retriever disabled.")
            self.idx = None
            self.metadata = []
            return

        self.idx = faiss.read_index(str(idx_path))
        self.metadata = []

        with metadata_path.open(mode="r", encoding="utf-8") as f:
            for line in f:
                self.metadata.append(json.loads(line))

        print("FAISS vectors:", self.idx.ntotal)
        print("Metadata records:", len(self.metadata))

        assert self.idx.ntotal == len(self.metadata), (
            f"FAISS index ({self.idx.ntotal}) "
            f"!= metadata records ({len(self.metadata)})"
        )

    def reload(self):
        print("Reloading retriever...")
        self.idx = faiss.read_index(str(self.idx_path))
        self.metadata = []

        with self.metadata_path.open(mode="r", encoding="utf-8") as f:
            for line in f:
                self.metadata.append(json.loads(line))

        assert self.idx.ntotal == len(self.metadata), (
            f"FAISS index ({self.idx.ntotal}) "
            f"!= metadata records ({len(self.metadata)})"
        )
        print(f"✅ Loaded {self.idx.ntotal} vectors")

    def cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def search(self, query: str, top_k: int = 5, use_mmr: bool = True):
        if self.idx is None or not self.metadata:
            return []

        query_vec = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        if use_mmr:
            # Get more candidates for MMR reranking
            candidate_k = max(top_k * 4, 20)
        else:
            candidate_k = top_k

        scores, indices = self.idx.search(
            query_vec.reshape(1, -1),
            candidate_k
        )

        valid_indices = [i for i in indices[0] if i >= 0]

        if not valid_indices:
            return []

        if use_mmr:
            # Extract embeddings from FAISS index (not from metadata!)
            candidate_embeddings = np.zeros((len(valid_indices), self.idx.d), dtype=np.float32)
            for idx, faiss_idx in enumerate(valid_indices):
                candidate_embeddings[idx] = self.idx.reconstruct(int(faiss_idx))

            selected = mmr(
                query_vec,
                candidate_embeddings,
                lambda_param=0.7,
                top_k=top_k
            )
            final_indices = [valid_indices[i] for i in selected]
        else:
            final_indices = valid_indices[:top_k]

        res = []
        for i in final_indices:
            rec = self.metadata[i]
            res.append({
                "text": rec["text"],
                "doc_id": rec["doc_id"],
                "page": rec["page"],
                "chunk_id": rec["chunk_id"]
            })

        return res


def mmr(
        query_embedding: np.ndarray,
        doc_embeddings: np.ndarray,
        lambda_param: float = 0.7,
        top_k: int = 5
):

    selected = []
    candidates = list(range(len(doc_embeddings)))

    relevance_scores = doc_embeddings @ query_embedding

    first = int(np.argmax(relevance_scores))
    selected.append(first)
    candidates.remove(first)

    while len(selected) < top_k and candidates:
        mmr_scores = []
        for c in candidates:
            relevance = relevance_scores[c]

            diversity = max(
                doc_embeddings[c] @ doc_embeddings[s]
                for s in selected
            )

            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity
            mmr_scores.append((mmr_score, c))

        _, best = max(mmr_scores, key=lambda x: x[0])
        selected.append(best)
        candidates.remove(best)

    return selected
