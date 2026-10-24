import logging
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from .crew import generate_script
from .genre_classifier import get_classifier
from .settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(_: FastAPI):
    await run_in_threadpool(get_classifier)
    yield


app = FastAPI(title="Script Agent Service", version="1.0.0", lifespan=lifespan)


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = settings.service_api_key
    if not expected or expected.startswith("your-"):
        raise HTTPException(500, "SERVICE_API_KEY is not set in .env")
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(401, "Invalid or missing X-Api-Key header")


class GenerateRequest(BaseModel):
    title: str = Field(default="", max_length=200)
    genre: str | None = Field(default=None, max_length=100)
    description: str | None = Field(default=None, max_length=2000)
    prompt: str = Field(min_length=1, max_length=2000)


class GenreCheckResponse(BaseModel):
    requested: list[str]
    predicted: str
    confidence: float
    matches: bool | None


class GenerateResponse(BaseModel):
    script: str
    revisions: int
    genre_check: GenreCheckResponse | None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "genre_classifier_loaded": get_classifier() is not None}


@app.post("/generate", response_model=GenerateResponse, dependencies=[Depends(verify_api_key)])
async def generate(req: GenerateRequest) -> GenerateResponse:
    try:
        result = await run_in_threadpool(
            generate_script, req.title, req.genre, req.description, req.prompt
        )
    except Exception as exc:
        logger.exception("Script generation failed")
        raise HTTPException(502, f"Script generation failed: {exc}") from exc

    check = result.genre_check
    return GenerateResponse(
        script=result.script,
        revisions=result.revisions,
        genre_check=GenreCheckResponse(
            requested=check.requested,
            predicted=check.top.genre,
            confidence=round(check.top.confidence, 4),
            matches=check.matches,
        )
        if check
        else None,
    )
