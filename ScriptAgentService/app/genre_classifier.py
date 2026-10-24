import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .genres import parse_genres
from .settings import settings

logger = logging.getLogger(__name__)


@dataclass
class GenrePrediction:
    genre: str
    confidence: float


@dataclass
class GenreCheck:
    requested: list[str]
    predictions: list[GenrePrediction]
    matches: bool | None

    @property
    def top(self) -> GenrePrediction:
        return self.predictions[0]

    def summary(self) -> str:
        preds = ", ".join(f"{p.genre} ({p.confidence:.0%})" for p in self.predictions)
        if self.matches is None:
            return f"Classifier reads this script as: {preds}."
        verdict = "MATCHES" if self.matches else "DOES NOT MATCH"
        return (
            f"Classifier reads this script as: {preds}. "
            f"Requested genre ({', '.join(self.requested)}) {verdict}."
        )


class GenreClassifier:
    """Fine-tuned transformer (see training/train_genre_classifier.py) served with PyTorch."""

    def __init__(self, model_dir: Path, max_length: int = 512):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()
        self.max_length = max_length

    @torch.inference_mode()
    def predict(self, text: str, top_k: int = 3) -> list[GenrePrediction]:
        inputs = self.tokenizer(
            text, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self.device)
        probs = torch.softmax(self.model(**inputs).logits[0], dim=-1)
        values, indices = probs.topk(min(top_k, probs.numel()))
        id2label = self.model.config.id2label
        return [
            GenrePrediction(genre=id2label[i.item()], confidence=v.item())
            for v, i in zip(values, indices)
        ]

    def check(self, script: str, requested_genre: str | None, match_top_k: int = 2) -> GenreCheck:
        predictions = self.predict(script)
        requested = parse_genres(requested_genre)
        matches = None
        if requested:
            matches = any(p.genre in requested for p in predictions[:match_top_k])
        return GenreCheck(requested=requested, predictions=predictions, matches=matches)


@lru_cache(maxsize=1)
def get_classifier() -> GenreClassifier | None:
    """Load the classifier once; return None if it hasn't been trained yet."""
    model_dir = settings.genre_model_dir
    if not (model_dir / "config.json").exists():
        logger.warning(
            "Genre classifier not found at %s; genre checks are disabled. "
            "Run training/train_genre_classifier.py to create it.",
            model_dir,
        )
        return None
    logger.info("Loading genre classifier from %s", model_dir)
    return GenreClassifier(model_dir)
