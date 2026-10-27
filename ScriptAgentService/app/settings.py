import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

SERVICE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVICE_DIR.parent

load_dotenv(REPO_ROOT / ".env")


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    service_api_key: str
    llm_model: str
    # Llama 3.3 70B on Groq. The retrieval budget is far smaller than this on purpose.
    llm_context_tokens: int
    qdrant_url: str
    qdrant_collection: str
    embedding_model: str
    rag_context_tokens: int
    rag_min_score: float
    chunk_tokens: int
    crew_max_rpm: int
    crew_verbose: bool


def load_settings() -> Settings:
    return Settings(
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        service_api_key=os.getenv("SERVICE_API_KEY", ""),
        llm_model=os.getenv("LLM_MODEL", "groq/llama-3.3-70b-versatile"),
        llm_context_tokens=int(os.getenv("LLM_CONTEXT_TOKENS", "128000")),
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "scripts"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        rag_context_tokens=int(os.getenv("RAG_CONTEXT_TOKENS", "1200")),
        rag_min_score=float(os.getenv("RAG_MIN_SCORE", "0.55")),
        chunk_tokens=int(os.getenv("CHUNK_TOKENS", "350")),
        crew_max_rpm=int(os.getenv("CREW_MAX_RPM", "20")),
        crew_verbose=_bool("CREW_VERBOSE", False),
    )


settings = load_settings()
