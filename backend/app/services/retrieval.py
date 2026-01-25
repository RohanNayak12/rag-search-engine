import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


class Retrievar:
    def __init__(self,idx_path:Path,metadata_path:Path,chunks_path: Path):

        self.model=SentenceTransformer("all-MiniLM-L6-v2")

        if not idx_path.exists() or not metadata_path.exists():
            print("⚠️ No FAISS index found yet. Retriever disabled.")
            self.idx = None
            self.metadata = []
            return
        self.idx = faiss.read_index(str(idx_path))

        self.metadata=[]

        with metadata_path.open(mode="r",encoding="utf-8") as f:
            for line in f:
                self.metadata.append(json.loads(line))

        print("FAISS vectors:", self.idx.ntotal)
        print("Metadata records:", len(self.metadata))

        assert self.idx.ntotal == len(self.metadata), (
            f"FAISS index ({self.idx.ntotal}) "
            f"!= metadata records ({len(self.metadata)})"
        )


    def search(self,query:str,top_k:int=5):

        if self.idx is None or not self.metadata:
            return []

        query_vec=self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        scores,indices=self.idx.search(query_vec,top_k)
        res=[]
        for score,i in zip(scores[0],indices[0]):
            if i<0:
                continue
            meta=self.metadata[i]
            res.append({
                "score":float(score),
                "text": meta["text"],
                "doc_id":meta["doc_id"],
                "page":meta["page"],
                "chunk_id":meta["chunk_id"]
            })
        print("Top indices:", indices)
        print("Top scores:", scores)

        return res
