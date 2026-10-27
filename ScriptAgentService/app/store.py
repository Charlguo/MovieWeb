"""Qdrant vector store for saved screenplays.

Each chunk is stored with the movie it came from. Re-indexing a movie deletes
its old chunks first, so edits replace the previous vectors instead of
duplicating them.
"""

import logging
import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

from .embeddings import Embedder, get_embedder
from .settings import settings
from .tokens import count_tokens, split_paragraphs

logger = logging.getLogger(__name__)

_NAMESPACE = uuid.UUID("8f3c2a1e-6b4d-4e7a-9c1f-2d5b7a0e4c31")


@dataclass
class RetrievedChunk:
    movie_id: int
    title: str
    genre: str
    text: str
    score: float


class ScriptStore:
    def __init__(self, client: QdrantClient | None = None, embedder: Embedder | None = None):
        self.client = client or QdrantClient(url=settings.qdrant_url)
        self.embedder = embedder or get_embedder()
        self.collection = settings.qdrant_collection
        self._ready = False

    def ensure_collection(self) -> None:
        if self._ready:
            return
        names = {item.name for item in self.client.get_collections().collections}
        if self.collection not in names:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.embedder.dimensions, distance=Distance.COSINE),
            )
        self._ready = True

    def ping(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            logger.warning("Qdrant is not reachable at %s", settings.qdrant_url)
            return False

    def index_movie(self, movie_id: int, title: str, genre: str | None, description: str | None, script: str | None) -> int:
        """Replace this movie's vectors. Returns how many chunks were stored."""
        self.ensure_collection()
        self.delete_movie(movie_id)

        header = f"Title: {title}\nGenre: {genre or 'unspecified'}\nLogline: {description or 'none'}"
        body_chunks = split_paragraphs(script or "")
        documents = [f"{header}\n\n{chunk}" for chunk in body_chunks] or [header]
        vectors = self.embedder.embed(documents)

        points = [
            PointStruct(
                id=str(uuid.uuid5(_NAMESPACE, f"{movie_id}:{index}")),
                vector=vector,
                payload={
                    "movie_id": movie_id,
                    "title": title,
                    "genre": genre or "",
                    "text": document,
                },
            )
            for index, (document, vector) in enumerate(zip(documents, vectors))
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        return len(points)

    def delete_movie(self, movie_id: int) -> None:
        self.ensure_collection()
        self.client.delete(
            collection_name=self.collection,
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="movie_id", match=MatchValue(value=movie_id))])
            ),
        )

    def search(self, query: str, limit: int = 6) -> list[RetrievedChunk]:
        self.ensure_collection()
        vector = self.embedder.embed([query])[0]
        found = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=limit,
            score_threshold=settings.rag_min_score,
        ).points
        return [
            RetrievedChunk(
                movie_id=int(point.payload["movie_id"]),
                title=point.payload.get("title") or "Untitled",
                genre=point.payload.get("genre") or "",
                text=point.payload.get("text") or "",
                score=point.score or 0.0,
            )
            for point in found
        ]


def build_context(chunks: list[RetrievedChunk]) -> tuple[str, list[dict], int]:
    """Fit the best chunks into the retrieval token budget, highest score first."""
    blocks: list[str] = []
    sources: list[dict] = []
    seen: set[int] = set()
    used_tokens = 0

    for chunk in chunks:
        genre = f", {chunk.genre}" if chunk.genre else ""
        block = f"Source: {chunk.title}{genre} (similarity {chunk.score:.2f})\n{chunk.text}"
        cost = count_tokens(block)
        if used_tokens + cost > settings.rag_context_tokens:
            continue
        blocks.append(block)
        used_tokens += cost
        if chunk.movie_id not in seen:
            seen.add(chunk.movie_id)
            sources.append({"movie_id": chunk.movie_id, "title": chunk.title, "genre": chunk.genre, "score": round(chunk.score, 4)})

    context = "\n\n".join(blocks)
    return context, sources, used_tokens
