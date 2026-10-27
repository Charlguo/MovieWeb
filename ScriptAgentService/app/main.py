import logging
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from .crew import generate_script
from .embeddings import get_embedder
from .settings import settings
from .store import ScriptStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

store = ScriptStore()


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Download and load the embedding model once, before the first request.
    await run_in_threadpool(get_embedder)
    if store.ping():
        await run_in_threadpool(store.ensure_collection)
    yield


app = FastAPI(title="Script Agent Service", version="2.0.0", lifespan=lifespan)


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


class SourceResponse(BaseModel):
    movie_id: int
    title: str
    genre: str
    score: float


class GenerateResponse(BaseModel):
    script: str
    sources: list[SourceResponse]
    context_tokens: int


class IndexRequest(BaseModel):
    id: int
    title: str = Field(min_length=1, max_length=200)
    genre: str | None = Field(default=None, max_length=100)
    description: str | None = Field(default=None, max_length=2000)
    script: str | None = Field(default=None, max_length=20000)


class IndexResponse(BaseModel):
    movie_id: int
    chunks: int


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "vector_store": store.ping(),
        "embedding_model": settings.embedding_model,
        "llm": settings.llm_model,
        "context_window_tokens": settings.llm_context_tokens,
        "retrieval_budget_tokens": settings.rag_context_tokens,
    }


@app.post("/index", response_model=IndexResponse, dependencies=[Depends(verify_api_key)])
async def index_movie(req: IndexRequest) -> IndexResponse:
    try:
        chunks = await run_in_threadpool(
            store.index_movie, req.id, req.title, req.genre, req.description, req.script
        )
    except Exception as exc:
        logger.exception("Indexing failed")
        raise HTTPException(503, f"Could not index the script: {exc}") from exc
    return IndexResponse(movie_id=req.id, chunks=chunks)


@app.delete("/index/{movie_id}", dependencies=[Depends(verify_api_key)])
async def delete_movie(movie_id: int) -> dict:
    try:
        await run_in_threadpool(store.delete_movie, movie_id)
    except Exception as exc:
        logger.exception("Delete from vector store failed")
        raise HTTPException(503, f"Could not remove the script: {exc}") from exc
    return {"movie_id": movie_id, "deleted": True}


@app.post("/generate", response_model=GenerateResponse, dependencies=[Depends(verify_api_key)])
async def generate(req: GenerateRequest) -> GenerateResponse:
    try:
        result = await run_in_threadpool(
            generate_script, req.title, req.genre, req.description, req.prompt, store
        )
    except Exception as exc:
        logger.exception("Script generation failed")
        raise HTTPException(502, f"Script generation failed: {exc}") from exc

    return GenerateResponse(
        script=result.script,
        sources=[SourceResponse(**source) for source in result.sources],
        context_tokens=result.context_tokens,
    )
