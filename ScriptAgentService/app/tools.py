from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from .genre_classifier import GenreClassifier


class DraftStore:
    """Holds the latest screenplay draft for one request.

    The writer task's callback saves its output here, so the genre tool can
    classify the draft without the LLM having to paste the full script into
    a tool call (which would burn tokens and risk truncation).
    """

    def __init__(self) -> None:
        self.draft: str = ""

    def save(self, output) -> None:
        self.draft = output.raw


class GenreCheckInput(BaseModel):
    requested_genre: str = Field(description="The genre the producer asked for, e.g. 'horror'.")


class GenreCheckTool(BaseTool):
    name: str = "genre_classifier"
    description: str = (
        "Runs a fine-tuned PyTorch genre classifier on the current screenplay draft and "
        "reports the predicted genres with confidence scores, and whether they match the "
        "requested genre. Use it before writing your notes."
    )
    args_schema: Type[BaseModel] = GenreCheckInput

    _classifier: GenreClassifier = PrivateAttr()
    _store: DraftStore = PrivateAttr()

    def __init__(self, classifier: GenreClassifier, store: DraftStore, **kwargs):
        super().__init__(**kwargs)
        self._classifier = classifier
        self._store = store

    def _run(self, requested_genre: str) -> str:
        if not self._store.draft:
            return "No draft is available yet."
        return self._classifier.check(self._store.draft, requested_genre).summary()
