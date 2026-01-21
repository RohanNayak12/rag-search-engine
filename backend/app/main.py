from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from backend.app.api.upload import router as upload_router
from backend.app.api.ask import router as ask_router
from backend.app.api.health import router as health_router
from backend.app.api.index_new import router as index_new

app = FastAPI(title="RAG Search Engine")

BASE_DIR=Path(__file__).resolve().parent
app.mount("/static",StaticFiles(directory=BASE_DIR/"static"),name="static")

app.include_router(health_router)
app.include_router(index_new)
app.include_router(ask_router)
app.include_router(upload_router)

@app.get("/",response_class=HTMLResponse)
def home():
    html_path=BASE_DIR/"template"/ "index.html"
    return html_path.read_text(encoding="utf-8")