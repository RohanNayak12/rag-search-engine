import faiss
import numpy as np
from pathlib import Path

class FaissIndexService:
    def __init__(self,dim:int):
        self.index=faiss.IndexFlatIP(dim)

    def add(self,vector:np.ndarray):
        self.index.add(vector)

    def save(self,path:Path):
        faiss.write_index(self.index,str(path))

    @staticmethod
    def load(path:Path):
        return faiss.read_index(str(path))