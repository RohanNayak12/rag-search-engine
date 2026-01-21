from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "RAG Search Engine"
    DATA_DIR: str = "data"
    RAW_PDF_DIR: str = "data/raw_pdfs"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

settings = Settings()
