from pathlib import Path
from pypdf import PdfReader

def extract_from_pdf(pdf_path:Path)->list[dict]:
    reader = PdfReader(str(pdf_path))
    pages=[]

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({
                "page":i+1,
                "text":text.strip()
            })
    return pages