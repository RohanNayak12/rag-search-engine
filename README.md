# RAG Search Engine

A **Retrieval-Augmented Generation (RAG)** search engine that lets you upload PDF documents and ask natural-language questions about them. It retrieves the most relevant passages using a FAISS vector index and generates grounded, cited answers via **Google Gemini**.

---

## Features

- **PDF Ingestion** — Upload PDFs through the API or drop them into `data/raw_pdfs/`
- **Semantic Search** — Dense vector retrieval using `sentence-transformers` (`all-MiniLM-L6-v2`) and a FAISS index
- **MMR Re-ranking** — Maximal Marginal Relevance ensures diverse, non-redundant context chunks
- **AI-Powered Answers** — Google Gemini (`gemini-2.5-flash`) generates answers grounded strictly in retrieved context, with source citations
- **Incremental Indexing** — New PDFs are indexed without re-processing already-indexed documents
- **REST API** — FastAPI backend with endpoints for uploading, indexing, and querying
- **Web UI** — Simple HTML frontend served at the root URL

---

## Project Structure

```
rag-search-engine/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── ask.py            # POST /ask — question answering endpoint
│   │   │   ├── upload.py         # POST /upload — PDF upload endpoint
│   │   │   ├── index_new.py      # POST /index — trigger incremental indexing
│   │   │   ├── health.py         # GET /health — health check
│   │   │   └── stats.py          # Index statistics
│   │   ├── services/
│   │   │   ├── rag.py            # RAGService — builds prompt & calls Gemini
│   │   │   ├── retrieval.py      # Retrievar — FAISS search + MMR re-ranking
│   │   │   ├── embeddings.py     # EmbeddingsService — sentence-transformer wrapper
│   │   │   ├── ingestion.py      # PDF text extraction (pypdf)
│   │   │   └── incremental_indexing.py  # Index only new PDFs
│   │   ├── utils/
│   │   │   └── chunking.py       # Text chunking utilities
│   │   ├── prompts/
│   │   │   └── rag_prompt.txt    # System prompt template
│   │   ├── template/
│   │   │   └── index.html        # Web UI
│   │   ├── static/               # Static assets
│   │   ├── config.py             # App settings (pydantic-settings)
│   │   └── main.py               # FastAPI app entry point
│   └── requirements.txt
├── data/
│   ├── raw_pdfs/                 # Place PDF files here
│   ├── embeddings/               # metadata.jsonl (chunks + embeddings)
│   └── faiss/                    # index.faiss (FAISS vector index)
└── scripts/
    ├── injest_pdfs.py            # Extract & chunk all PDFs → chunks.jsonl
    ├── generate_embeddings.py    # Embed chunks → vectors.npy + metadata.jsonl
    ├── build_faiss_index.py      # Build FAISS index from vectors
    ├── test_rag.py               # End-to-end RAG test
    └── test_search.py            # Retrieval-only test
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/rag-search-engine.git
cd rag-search-engine
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r backend/requirements.txt
```

### 4. Set Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

> Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

---

## Indexing Documents

### Option A — Using the API (Recommended)

1. **Start the server** (see below).
2. **Upload a PDF:**
   ```bash
   curl -X POST http://localhost:8000/upload \
     -F "file=@your_document.pdf"
   ```
3. **Trigger indexing:**
   ```bash
   curl -X POST http://localhost:8000/index
   ```

### Option B — Using Scripts (Batch / Offline)

Run the three pipeline scripts in order from the project root:

```bash
# Step 1: Extract text and chunk all PDFs in data/raw_pdfs/
python scripts/injest_pdfs.py

# Step 2: Generate embeddings for all chunks
python scripts/generate_embeddings.py

# Step 3: Build the FAISS index
python scripts/build_faiss_index.py
```

---

## Running the Server

```bash
uvicorn backend.app.main:app --reload
```

The server starts at **http://localhost:8000**.

| URL | Description |
|-----|-------------|
| `http://localhost:8000/` | Web UI |
| `http://localhost:8000/docs` | Interactive API docs (Swagger) |
| `http://localhost:8000/health` | Health check |

---

## API Reference

### `POST /upload`
Upload a PDF file to be indexed.

**Request:** `multipart/form-data` with field `file` (PDF only)

**Response:**
```json
{ "filename": "document.pdf", "status": "uploaded" }
```

---

### `POST /index`
Index any newly uploaded PDFs that haven't been processed yet.

**Response:**
```json
{ "message": "Indexed 2 new document(s). Total chunks: 348" }
```

---

### `POST /ask`
Ask a question against the indexed documents.

**Request body:**
```json
{
  "question": "What is the main conclusion of the paper?",
  "top_k": 5
}
```

**Response:**
```json
{
  "question": "What is the main conclusion of the paper?",
  "answer": "According to [Source: paper.pdf | page 12], the main conclusion is..."
}
```

---

## How It Works

```
PDF Upload
    │
    ▼
Text Extraction (pypdf)
    │
    ▼
Text Chunking (token-aware)
    │
    ▼
Embedding Generation (all-MiniLM-L6-v2)
    │
    ▼
FAISS Index (IndexFlatIP, cosine similarity)
    │
    ▼
Query → Embed Query → FAISS Search → MMR Re-rank
    │
    ▼
Build Context (top-k chunks with source citations)
    │
    ▼
Gemini LLM → Grounded Answer
```

**MMR (Maximal Marginal Relevance):** When retrieving chunks, the engine fetches 4× the requested candidates and applies MMR re-ranking (`λ=0.7`) to balance relevance with diversity, reducing redundant context sent to the LLM.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API Framework | [FastAPI](https://fastapi.tiangolo.com/) |
| Embedding Model | `all-MiniLM-L6-v2` via [sentence-transformers](https://www.sbert.net/) |
| Vector Store | [FAISS](https://github.com/facebookresearch/faiss) (`faiss-cpu`) |
| LLM | [Google Gemini](https://ai.google.dev/) (`gemini-2.5-flash`) |
| PDF Parsing | [pypdf](https://pypdf.readthedocs.io/) |
| Settings | [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) |

---

## Requirements

- Python 3.10+
- A valid **Google Gemini API key**

---

## License

This project is open-source. Feel free to use and modify it for your own purposes.
