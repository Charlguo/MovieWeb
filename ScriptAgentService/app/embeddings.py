"""Turn text into vectors with a small ONNX embedding model.

An embedding is a list of numbers that places text with similar meaning near
each other. The language model never sees these numbers. Retrieval uses them
to find relevant scripts, and the scripts themselves go into the prompt.

FastEmbed runs BAAI/bge-small-en-v1.5 through ONNX Runtime, so this service
does not depend on PyTorch.
"""

from functools import lru_cache

from fastembed import TextEmbedding

from .settings import settings


class Embedder:
    def __init__(self, model_name: str):
        self.model = TextEmbedding(model_name=model_name)
        probe = next(self.model.embed(["probe"]))
        self.dimensions = len(probe)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [vector.tolist() for vector in self.model.embed(texts)]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder(settings.embedding_model)
