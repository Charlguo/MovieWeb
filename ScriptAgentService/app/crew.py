"""Multi-agent screenwriting pipeline.

Story Architect, Screenwriter, Script Doctor, and Final Draft Editor run in
order through CrewAI. Before they start, retrieval adds similar scenes from
the catalog to the prompt. The agents are told to learn structure and tone
from those scenes and not to copy them.

Temperature is set per role. It controls how much the model varies the next
token: lower is more repeatable, higher is more inventive.
"""

import logging
import re
from dataclasses import dataclass, field

from crewai import LLM, Agent, Crew, Process, Task

from .settings import settings
from .store import ScriptStore, build_context

logger = logging.getLogger(__name__)

# The critic should stay consistent. The writer is allowed to vary.
TEMPERATURE = {
    "architect": 0.6,
    "writer": 0.8,
    "doctor": 0.3,
    "editor": 0.5,
}

SCREENPLAY_RULES = """Screenplay formatting rules:
- Start with a SLUGLINE (e.g. INT. LOCATION - DAY).
- Character names in CAPS above their dialogue.
- Clear, concise action lines in present tense.
- Realistic, subtext-driven dialogue.
- No commentary, headings, markdown or notes outside the screenplay itself."""

REFERENCE_RULES = """Reference rules:
- The catalog excerpts below are examples of structure and tone only.
- Do not copy their plot, characters, or dialogue.
- If no excerpts are listed, write from the pitch alone."""


@dataclass
class ScriptResult:
    script: str
    sources: list[dict] = field(default_factory=list)
    context_tokens: int = 0


def _llm(role: str) -> LLM:
    return LLM(model=settings.llm_model, api_key=settings.groq_api_key, temperature=TEMPERATURE[role])


def _clean(text: str) -> str:
    text = text.strip()
    fenced = re.match(r"^```[^\n]*\n(.*?)\n?```$", text, re.DOTALL)
    return fenced.group(1).strip() if fenced else text


def _agents() -> dict[str, Agent]:
    common = dict(verbose=settings.crew_verbose, allow_delegation=False)
    return {
        "architect": Agent(
            role="Story Architect",
            goal="Turn a producer's pitch into a tight, genre-true outline for a single scene.",
            backstory=(
                "A veteran story editor who has broken story for studio features. You think in "
                "beats, stakes and reversals, and you know the conventions of every genre."
            ),
            llm=_llm("architect"),
            **common,
        ),
        "writer": Agent(
            role="Screenwriter",
            goal="Write a vivid, producible scene that follows the outline beat for beat.",
            backstory=(
                "A WGA screenwriter known for sharp dialogue and economical action lines. "
                "You write in industry-standard screenplay format."
            ),
            llm=_llm("writer"),
            **common,
        ),
        "doctor": Agent(
            role="Story Critic",
            goal="Find the specific weaknesses in a draft and give actionable notes to fix them.",
            backstory=(
                "A seasoned story analyst. You check pacing, character voice, genre tone and "
                "formatting, and you flag any passage that copies a reference scene."
            ),
            llm=_llm("doctor"),
            **common,
        ),
        "editor": Agent(
            role="Final Draft Editor",
            goal="Apply the script doctor's notes and deliver a clean, correctly formatted final scene.",
            backstory=(
                "A meticulous showrunner's assistant who turns notes into a polished final draft "
                "and never lets a formatting error through."
            ),
            llm=_llm("editor"),
            **common,
        ),
    }


def _retrieve(store: ScriptStore, title: str, genre: str, description: str, prompt: str) -> tuple[str, list[dict], int]:
    query = f"Title: {title}\nGenre: {genre}\nLogline: {description}\nScene: {prompt}"
    try:
        chunks = store.search(query)
    except Exception:
        logger.exception("Retrieval failed; generating without catalog context")
        return "", [], 0
    context, sources, used = build_context(chunks)
    logger.info("Retrieved %s catalog chunks, %s tokens of context", len(chunks), used)
    return context, sources, used


def generate_script(
    title: str,
    genre: str | None,
    description: str | None,
    prompt: str,
    store: ScriptStore | None = None,
) -> ScriptResult:
    if not settings.groq_api_key or settings.groq_api_key.startswith("your-"):
        raise RuntimeError("GROQ_API_KEY is not set in .env")

    store = store or ScriptStore()
    safe_title = title or "Untitled"
    safe_genre = genre or "unspecified genre"
    safe_description = description or "not provided"
    context, sources, context_tokens = _retrieve(store, safe_title, safe_genre, safe_description, prompt)
    references = context or "No earlier scripts are in the catalog yet."

    agents = _agents()
    outline = Task(
        description=(
            "Movie title: {title}\nGenre: {genre}\nLogline: {description}\n"
            "Scene request from the producer: {prompt}\n\n"
            + REFERENCE_RULES
            + "\n\nCatalog excerpts:\n{references}\n\n"
            "Break this scene into 5-8 beats. Identify the location and time of day, the "
            "characters present, what each character wants, and the turn at the end of the scene."
        ),
        expected_output="A numbered beat outline, plus the slugline and a short character list.",
        agent=agents["architect"],
    )
    draft = Task(
        description=(
            "Write the full scene for '{title}' ({genre}) from the outline.\n\n"
            + SCREENPLAY_RULES
            + "\n\n"
            + REFERENCE_RULES
        ),
        expected_output="The complete scene in screenplay format, about two pages long.",
        agent=agents["writer"],
        context=[outline],
    )
    review = Task(
        description=(
            "Review the draft of '{title}', a {genre} scene. Give 3-6 specific notes covering "
            "genre tone, pacing, dialogue and formatting. Call out anything copied from the "
            "catalog excerpts."
        ),
        expected_output="A numbered list of concrete, actionable notes.",
        agent=agents["doctor"],
        context=[draft],
    )
    final = Task(
        description=(
            "Rewrite the draft of '{title}', applying every note from the script doctor.\n\n"
            + SCREENPLAY_RULES
        ),
        expected_output="Only the final screenplay text, with nothing before or after it.",
        agent=agents["editor"],
        context=[draft, review],
    )
    crew = Crew(
        agents=list(agents.values()),
        tasks=[outline, draft, review, final],
        process=Process.sequential,
        max_rpm=settings.crew_max_rpm,
        verbose=settings.crew_verbose,
    )
    script = _clean(crew.kickoff(inputs={
        "title": safe_title,
        "genre": safe_genre,
        "description": safe_description,
        "prompt": prompt,
        "references": references,
    }).raw)
    return ScriptResult(script=script, sources=sources, context_tokens=context_tokens)
