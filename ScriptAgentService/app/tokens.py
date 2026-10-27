"""Token budgets for the Llama 3 context window.

A token is the unit the model actually reads, roughly a word or a piece of a
word. Llama 3.3 70B accepts about 128,000 tokens in one request (the context
window). This service spends only a small slice of that window on retrieved
scripts, so the user's pitch stays the main instruction and the request stays
inside Groq's tokens-per-minute limit.

Counts use the cl100k tokenizer. Llama 3 has its own tokenizer, so a count
here is a budget estimate, not the exact number Groq will bill.
"""

from functools import lru_cache

import tiktoken

from .settings import settings


@lru_cache(maxsize=1)
def _encoder():
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoder().encode(text or ""))


def split_paragraphs(text: str, chunk_tokens: int | None = None) -> list[str]:
    """Pack paragraphs into chunks that each fit the token budget."""
    limit = chunk_tokens or settings.chunk_tokens
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for paragraph in paragraphs:
        cost = count_tokens(paragraph)
        if current and current_tokens + cost > limit:
            chunks.append("\n\n".join(current))
            current, current_tokens = [], 0
        current.append(paragraph)
        current_tokens += cost
    if current:
        chunks.append("\n\n".join(current))
    return chunks
