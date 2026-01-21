from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingsService:
    def __init__(self,model_name:str="all-MiniLM-L6-v2"):
        self.model=SentenceTransformer(model_name)

    def embed_text(self,text:List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            text,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings