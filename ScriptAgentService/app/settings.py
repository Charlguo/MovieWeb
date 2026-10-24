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
    genre_model_dir: Path
    max_genre_revisions: int
    crew_max_rpm: int
    crew_verbose: bool


def load_settings() -> Settings:
    model_dir = Path(os.getenv("GENRE_MODEL_DIR", "models/genre-classifier"))
    if not model_dir.is_absolute():
        model_dir = SERVICE_DIR / model_dir

    return Settings(
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        service_api_key=os.getenv("SERVICE_API_KEY", ""),
        llm_model=os.getenv("LLM_MODEL", "groq/llama-3.3-70b-versatile"),
        genre_model_dir=model_dir,
        max_genre_revisions=int(os.getenv("MAX_GENRE_REVISIONS", "1")),
        crew_max_rpm=int(os.getenv("CREW_MAX_RPM", "20")),
        crew_verbose=_bool("CREW_VERBOSE", False),
    )


settings = load_settings()
