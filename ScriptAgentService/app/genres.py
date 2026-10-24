"""Canonical genre vocabulary shared by the training script and the API.

Free-text genres ("Sci-Fi Thriller", "romantic comedy", "comedy-drama") are
mapped onto a fixed label set so the classifier's predictions can be compared
with whatever the user typed into the form.
"""

import re

GENRE_ALIASES: dict[str, list[str]] = {
    "action": ["action", "martial arts", "superhero", "disaster"],
    "adventure": ["adventure", "swashbuckler"],
    "animation": ["animation", "animated", "anime", "cartoon"],
    "comedy": ["comedy", "comedic", "satire", "parody", "spoof", "slapstick"],
    "crime": ["crime", "gangster", "heist", "film noir", "noir"],
    "drama": ["drama", "melodrama", "biographical", "biopic"],
    "fantasy": ["fantasy", "fairy tale", "sword and sorcery"],
    "horror": ["horror", "slasher", "zombie", "supernatural", "monster"],
    "musical": ["musical", "music"],
    "mystery": ["mystery", "detective", "whodunit"],
    "romance": ["romance", "romantic", "love story"],
    "science fiction": ["science fiction", "sci-fi", "sci fi", "scifi", "space opera", "cyberpunk"],
    "thriller": ["thriller", "suspense", "spy", "espionage"],
    "war": ["war", "world war"],
    "western": ["western"],
}

GENRES: list[str] = sorted(GENRE_ALIASES)

_PATTERNS = [
    (genre, re.compile(rf"\b{re.escape(alias)}\b"))
    for genre, aliases in GENRE_ALIASES.items()
    for alias in sorted(aliases, key=len, reverse=True)
]


def parse_genres(text: str | None) -> list[str]:
    """Return canonical genres found in `text`, ordered by where they appear."""
    if not text:
        return []
    lowered = text.lower()
    first_seen: dict[str, int] = {}
    for genre, pattern in _PATTERNS:
        match = pattern.search(lowered)
        if match and (genre not in first_seen or match.start() < first_seen[genre]):
            first_seen[genre] = match.start()
    return sorted(first_seen, key=first_seen.get)
