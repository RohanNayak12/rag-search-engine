import json
from pathlib import Path
from typing import List

import faiss
import numpy as np

from backend.app.services.ingestion import extract_from_pdf
from backend.app.utils.chunking import chunk_txt
from backend.app.services.embeddings import EmbeddingsService

def index_new_pdfs(
    raw_pdf_dir: Path,
    index_path: Path,
    metadata_path: Path,
    indexed_files_path: Path,
):
    indexed_files = set()
    if indexed_files_path.exists():
        indexed_files = set(json.loads(indexed_files_path.read_text()))

    new_pdfs = [pdf for pdf in raw_pdf_dir.glob("*.pdf") if pdf.name not in indexed_files]
    if not new_pdfs:
        return "No new documents to index."

    new_chunks: List[str] = []
    new_metadata: List[dict] = []

    for pdf in new_pdfs:
        pages = extract_from_pdf(pdf)
        chunk_id = 0
        for page in pages:
            for chunk in chunk_txt(page["text"]):
                new_chunks.append(chunk)
                new_metadata.append({
                    "doc_id": pdf.name,
                    "page": page["page"],
                    "chunk_id": chunk_id,
                    "text": chunk,
                })
                chunk_id += 1

    embedder = EmbeddingsService()
    new_vectors = embedder.embed_text(new_chunks).astype("float32")
    faiss.normalize_L2(new_vectors)

    if index_path.exists():
        index = faiss.read_index(str(index_path))
        if index.d != new_vectors.shape[1]:
            raise RuntimeError("Embedding dimension mismatch")
        index.add(new_vectors)
    else:
        index = faiss.IndexFlatIP(new_vectors.shape[1])
        index.add(new_vectors)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    # 🔹 Write metadata ONCE (with embeddings)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("a", encoding="utf-8") as f:
        for record, emb in zip(new_metadata, new_vectors):
            record["embedding"] = emb.tolist()
            f.write(json.dumps(record) + "\n")

    indexed_files.update(pdf.name for pdf in new_pdfs)
    indexed_files_path.write_text(json.dumps(sorted(indexed_files), indent=2))

    assert index.ntotal == sum(1 for _ in metadata_path.open()), \
        "FAISS and metadata out of sync"

    return f"Indexed {len(new_pdfs)} new document(s). Total chunks: {index.ntotal}"
