import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from pathlib import Path
from backend.app.services.ingestion import extract_from_pdf
from backend.app.utils.chunking import chunk_txt

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_PDF_DIR = PROJECT_ROOT / "data" / "raw_pdfs"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"

def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUT_PATH.exists():
        if OUT_PATH.is_file():
            OUT_PATH.unlink()
        else:
            raise RuntimeError(
                f"{OUT_PATH} exists but is a DIRECTORY. Delete it manually."
            )

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for pdf_path in RAW_PDF_DIR.glob("*.pdf"):
            pages = extract_from_pdf(pdf_path)

            for page in pages:
                chunks = chunk_txt(page["text"])

                for idx, chunk in enumerate(chunks):
                    record = {
                        "doc_id": pdf_path.name,
                        "page": page["page"],
                        "chunk_id": idx,
                        "text": chunk
                    }
                    f.write(json.dumps(record) + "\n")

    print("Ingestion Complete")

if __name__ == "__main__":
    main()
