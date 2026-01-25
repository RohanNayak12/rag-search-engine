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

    all_chunks: List[str] = []
    all_metadata: List[dict] = []
    newly_indexed_files: List[str] = []

    for pdf in raw_pdf_dir.glob("*.pdf"):
        pages = extract_from_pdf(pdf)

        chunk_id = 0
        for page in pages:
            chunks = chunk_txt(page["text"])
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadata.append({
                    "doc_id": pdf.name,
                    "page": page["page"],
                    "chunk_id": chunk_id,
                    "text": chunk,
                })
                chunk_id += 1

        if pdf.name not in indexed_files:
            newly_indexed_files.append(pdf.name)

    if not all_chunks:
        return "No documents found to index."

    embedder = EmbeddingsService()
    vectors = embedder.embed_text(all_chunks).astype("float32")
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as f:
        for record in all_metadata:
            f.write(json.dumps(record) + "\n")

    indexed_files.update(newly_indexed_files)
    indexed_files_path.write_text(json.dumps(sorted(indexed_files), indent=2))

    assert index.ntotal == len(all_metadata), (
        f"FAISS index ({index.ntotal}) "
        f"!= metadata records ({len(all_metadata)})"
    )

    return (
        f"Indexed {len(newly_indexed_files)} new document(s). "
        f"Total chunks: {len(all_metadata)}"
    )
