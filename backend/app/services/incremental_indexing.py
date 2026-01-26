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

    new_pdfs=[pdf for pdf in raw_pdf_dir.glob("*.pdf") if pdf.name not in indexed_files]
    if not new_pdfs:
        return "No new documents to index."


    new_chunks:List[str] = []
    new_metadata:List[dict] = []

    for pdf in new_pdfs:
        try:
            pages = extract_from_pdf(pdf)
            chunk_id = 0
            for page in pages:
                chunks = chunk_txt(page["text"])
                for chunk in chunks:
                    new_chunks.append(chunk)
                    new_metadata.append({
                        "doc_id": pdf.name,
                        "page": page["page"],
                        "chunk_id": chunk_id,
                        "text": chunk,
                    })
                    chunk_id += 1

        except Exception as e:
            print(f"Error processing {pdf.name}: {e}")
            continue

    if not new_chunks:
        return "No valid content found in the new documents."

    embedder = EmbeddingsService()
    new_vectors = embedder.embed_text(new_chunks).astype("float32")
    faiss.normalize_L2(new_vectors)

    if index_path.exists() and metadata_path.exists():
        print(f"Loading existing index from {index_path}")
        index = faiss.read_index(str(index_path))
        if index.d != new_vectors.shape[1]:
            raise RuntimeError(
                f"Inconsistent dimensions between existing index ({index.d}) "
                f"and new embeddings ({new_vectors.shape[1]})"
            )
        old_count=index.ntotal
        index.add(new_vectors)
        print(f"Added {new_vectors.shape[0]} new vectors to existing index (total: {index.ntotal-old_count})")
        with metadata_path.open("a",encoding="utf-8") as f:
            for record in new_metadata:
                f.write(json.dumps(record) + "\n")

    else:
        print("Creating new index from scratch")
        index = faiss.IndexFlatIP(new_vectors.shape[1])
        index.add(new_vectors)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", encoding="utf-8") as f:
            for record in new_metadata:
                f.write(json.dumps(record) + "\n")


    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    indexed_files.update([pdf.name for pdf in new_pdfs])
    indexed_files_path.parent.mkdir(parents=True, exist_ok=True)
    indexed_files_path.write_text(json.dumps(sorted(indexed_files),indent=2))

    total_metadata=0
    with metadata_path.open(mode="r",encoding="utf-8") as f:
        total_metadata=sum(1 for _ in f)

    assert index.ntotal == total_metadata, (
        f"FAISS index ({index.ntotal}) "
        f"!= metadata records ({total_metadata})"
    )

    return (
        f"Indexed {len(new_pdfs)} new document(s). "
        f"Total chunks: {index.ntotal}"
    )
