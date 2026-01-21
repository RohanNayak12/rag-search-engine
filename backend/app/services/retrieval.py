import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


class Retrievar:
    def __init__(self,idx_path:Path,metadata_path:Path,chunks_path: Path):
        self.idx=faiss.read_index(str(idx_path))
        self.model=SentenceTransformer("all-MiniLM-L6-v2")
        self.metadata=[]

        with metadata_path.open(mode="r",encoding="utf-8") as f:
            for line in f:
                self.metadata.append(json.loads(line))

        self.chunks = []
        with chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                self.chunks.append(record["text"])

    def search(self,query:str,top_k:int=5):
        query_vec=self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        scores,indices=self.idx.search(query_vec,top_k)
        res=[]
        for score,i in zip(scores[0],indices[0]):
            meta=self.metadata[i]
            res.append({
                "score":float(score),
                "text": self.chunks[i],
                **meta
            })
        return res
