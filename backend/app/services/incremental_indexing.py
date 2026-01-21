import json
from pathlib import Path
import numpy as np
import faiss

from backend.app.services.ingestion import extract_from_pdf
from backend.app.utils.chunking import chunk_txt
from backend.app.services.embeddings import EmbeddingsService


def index_new_pdfs(
    raw_pdf_dir: Path,
    index_path: Path,
    vectors_path: Path,
    metadata_path: Path,
    indexed_files_path: Path,
):
    indexed_files = set()

    if indexed_files_path.exists():
        indexed_files = set(json.loads(indexed_files_path.read_text()))

    new_chunks = []
    new_metadata = []
    newly_indexed_files = []

    for pdf in raw_pdf_dir.glob("*.pdf"):
        if pdf.name in indexed_files:
            continue

        pages = extract_from_pdf(pdf)
        for page in pages:
            for idx, chunk in enumerate(chunk_txt(page["text"])):
                new_chunks.append(chunk)
                new_metadata.append({
                    "doc_id": pdf.name,
                    "page": page["page"],
                    "chunk_id": idx,
                })

        newly_indexed_files.append(pdf.name)

    if not new_chunks:
        return "No new documents to index"

    embedder = EmbeddingsService()
    new_vectors = embedder.embed_text(new_chunks).astype("float32")
    faiss.normalize_L2(new_vectors)

    if index_path.exists():
        index = faiss.read_index(str(index_path))
    else:
        index = faiss.IndexFlatIP(new_vectors.shape[1])

    index.add(new_vectors)
    faiss.write_index(index, str(index_path))

    with metadata_path.open("a", encoding="utf-8") as f:
        for m, c in zip(new_metadata, new_chunks):
            f.write(json.dumps({**m, "text": c}) + "\n")

    indexed_files.update(newly_indexed_files)
    indexed_files_path.write_text(json.dumps(list(indexed_files), indent=2))

    return f"Indexed {len(newly_indexed_files)} new document(s)"
