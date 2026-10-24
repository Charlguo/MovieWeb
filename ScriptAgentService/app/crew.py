"""Multi-agent screenwriting pipeline.

Story Architect -> Screenwriter -> Script Doctor -> Final Draft Editor, run
sequentially by CrewAI. After the crew finishes, the PyTorch genre classifier
acts as a guardrail: if the final script doesn't read as the requested genre,
the Screenwriter is sent back with the classifier's verdict as feedback.
"""

import logging
import re
from dataclasses import dataclass

from crewai import LLM, Agent, Crew, Process, Task

from .genre_classifier import GenreCheck, get_classifier
from .settings import settings
from .tools import DraftStore, GenreCheckTool

logger = logging.getLogger(__name__)

SCREENPLAY_RULES = """Screenplay formatting rules:
- Start with a SLUGLINE (e.g. INT. LOCATION - DAY).
- Character names in CAPS above their dialogue.
- Clear, concise action lines in present tense.
- Realistic, subtext-driven dialogue.
- No commentary, headings, markdown or notes outside the screenplay itself."""


@dataclass
class ScriptResult:
    script: str
    revisions: int
    genre_check: GenreCheck | None


def _llm(temperature: float) -> LLM:
    return LLM(model=settings.llm_model, api_key=settings.groq_api_key, temperature=temperature)


def _clean(text: str) -> str:
    text = text.strip()
    fenced = re.match(r"^```[^\n]*\n(.*?)\n?```$", text, re.DOTALL)
    return fenced.group(1).strip() if fenced else text


def _agents(genre_tool: GenreCheckTool | None) -> dict[str, Agent]:
    common = dict(verbose=settings.crew_verbose, allow_delegation=False)
    return {
        "architect": Agent(
            role="Story Architect",
            goal="Turn a producer's pitch into a tight, genre-true outline for a single scene.",
            backstory=(
                "A veteran story editor who has broken story for studio features. You think in "
                "beats, stakes and reversals, and you know the conventions of every genre."
            ),
            llm=_llm(0.6),
            **common,
        ),
        "writer": Agent(
            role="Screenwriter",
            goal="Write a vivid, producible scene that follows the outline beat for beat.",
            backstory=(
                "A WGA screenwriter known for sharp dialogue and economical action lines. "
                "You write in industry-standard screenplay format."
            ),
            llm=_llm(0.8),
            **common,
        ),
        "doctor": Agent(
            role="Script Doctor",
            goal="Find the specific weaknesses in a draft and give actionable notes to fix them.",
            backstory=(
                "A sought-after script doctor. You check pacing, character voice, genre tone and "
                "formatting, and you back up genre judgements with data when you can."
            ),
            tools=[genre_tool] if genre_tool else [],
            llm=_llm(0.3),
            **common,
        ),
        "editor": Agent(
            role="Final Draft Editor",
            goal="Apply the script doctor's notes and deliver a clean, correctly formatted final scene.",
            backstory=(
                "A meticulous showrunner's assistant who turns notes into a polished final draft "
                "and never lets a formatting error through."
            ),
            llm=_llm(0.5),
            **common,
        ),
    }


def _run_crew(inputs: dict[str, str], classifier) -> str:
    store = DraftStore()
    genre_tool = GenreCheckTool(classifier, store) if classifier else None
    agents = _agents(genre_tool)

    outline = Task(
        description=(
            "Movie title: {title}\nGenre: {genre}\nLogline: {description}\n"
            "Scene request from the producer: {prompt}\n\n"
            "Break this scene into 5-8 beats. Identify the location and time of day, the "
            "characters present, what each character wants, and the turn at the end of the scene. "
            "Lean into the conventions of the genre."
        ),
        expected_output="A numbered beat outline, plus the slugline and a short character list.",
        agent=agents["architect"],
    )
    draft = Task(
        description=(
            "Write the full scene for '{title}' ({genre}) from the outline.\n\n" + SCREENPLAY_RULES
        ),
        expected_output="The complete scene in screenplay format, about two pages long.",
        agent=agents["writer"],
        context=[outline],
        callback=store.save,
    )
    review_steps = (
        "First call the genre_classifier tool with requested_genre='{genre}' and factor its "
        "verdict into your notes. "
        if genre_tool
        else ""
    )
    review = Task(
        description=(
            "Review the draft of '{title}', a {genre} scene. "
            + review_steps
            + "Then give 3-6 specific notes covering genre tone, pacing, dialogue and formatting."
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
    return _clean(crew.kickoff(inputs=inputs).raw)


def _revise_for_genre(inputs: dict[str, str], script: str, check: GenreCheck) -> str:
    writer = _agents(None)["writer"]
    task = Task(
        description=(
            "This scene for '{title}' is supposed to be {genre}, but a genre classifier says:\n"
            "{verdict}\n\nRewrite it so it unmistakably reads as {genre}: adjust tone, stakes, "
            "imagery and dialogue while keeping the story beats.\n\n" + SCREENPLAY_RULES
            + "\n\nCurrent scene:\n{script}"
        ),
        expected_output="Only the revised screenplay text, with nothing before or after it.",
        agent=writer,
    )
    crew = Crew(agents=[writer], tasks=[task], max_rpm=settings.crew_max_rpm,
                verbose=settings.crew_verbose)
    return _clean(crew.kickoff(inputs={**inputs, "script": script, "verdict": check.summary()}).raw)


def generate_script(title: str, genre: str | None, description: str | None, prompt: str) -> ScriptResult:
    if not settings.groq_api_key or settings.groq_api_key.startswith("your-"):
        raise RuntimeError("GROQ_API_KEY is not set in .env")

    inputs = {
        "title": title or "Untitled",
        "genre": genre or "unspecified genre",
        "description": description or "not provided",
        "prompt": prompt,
    }
    classifier = get_classifier()

    script = _run_crew(inputs, classifier)
    check = classifier.check(script, genre) if classifier else None

    revisions = 0
    while check and check.matches is False and revisions < settings.max_genre_revisions:
        logger.info("Genre mismatch (%s); revising.", check.summary())
        script = _revise_for_genre(inputs, script, check)
        check = classifier.check(script, genre)
        revisions += 1

    return ScriptResult(script=script, revisions=revisions, genre_check=check)
