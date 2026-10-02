# RAG pipeline for plant biology papers using Neo4j + Gemini/GPT + LangChain
# Requires env: GOOGLE_API_KEY, OPENAI_API_KEY, NEO4J_URI, NEO4J_USERNAME,
# NEO4J_PASSWORD. Importing this module (what `main.py` does at startup)
# raises if any of those are missing or blank.

import os
import argparse
import asyncio
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator, List, Dict, Optional, Tuple, Any, Union
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph, Neo4jVector
import requests
import re
import json
import hashlib
import warnings
import logging
import time
from enum import Enum
from pydantic import BaseModel, Field
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from threading import local
from dotenv import load_dotenv
from rich.logging import RichHandler

from getPrompt import getPrompt
from literature_graph_expansion import expand_literature_graph
from literature_candidate_ranking import literature_candidate_sort_key, score_literature_candidate_texts, score_literature_candidate_vectors
from taxon_filter import metadata_matches_taxon, resolve_taxon_filter, taxon_regex, text_is_clearly_other_taxon

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# Checked at import so `uvicorn main:app` (and the `Query.py` CLI) fail
# before any client is constructed. `USE_CACHE` and `ACCESSION_API_URL`
# stay optional: cache defaults off, and accession lookups already report
# a missing URL when that path is used.
REQUIRED_ENV_VARS = (
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "NEO4J_URI",
    "NEO4J_USERNAME",
    "NEO4J_PASSWORD",
)


def _require_env_vars() -> None:
    missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name, "").strip()]
    if not missing:
        return
    raise RuntimeError(
        "Missing required environment variables: "
        + ", ".join(missing)
        + ". Set them in the repository root .env file or the process environment."
    )


_require_env_vars()

useCache = os.getenv("USE_CACHE") or False

warnings.simplefilter("ignore", DeprecationWarning)

# Plain one-line-per-record log file next to this module (append mode), alongside
# the coloured console output. Rich formatting would wrap lines and drop
# timestamps in a file, so the file gets the original plain format.
query_log_handler = logging.FileHandler(
    Path(__file__).resolve().parent / "query.log", mode="a", encoding="utf-8"
)
query_log_handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s %(message)s")
)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    # markup=False so bracketed labels like "[original]" aren't parsed as Rich markup.
    handlers=[
        RichHandler(rich_tracebacks=True, markup=False, show_path=False),
        query_log_handler,
    ],
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("google_genai.models").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("neo4j").setLevel(logging.WARNING)
# Server-side query notifications (e.g. deprecation notices) are logged by the
# driver via this dedicated logger regardless of the "neo4j" logger's level.
logging.getLogger("neo4j.notifications").setLevel(
    logging.ERROR
)  # comment out in future if fixed upstream
logger = logging.getLogger(__name__)
# Set LOG_LEVEL=DEBUG to see per-stage diagnostics from this module only.
# Accepts a level name (any case) or number; anything else falls back to INFO.
_log_level_setting = (os.getenv("LOG_LEVEL") or "INFO").strip()
_log_level = (
    int(_log_level_setting) if _log_level_setting.isdigit()
    else logging.getLevelNamesMapping().get(_log_level_setting.upper())
)
logger.setLevel(_log_level if _log_level is not None else logging.INFO)
if _log_level is None:
    logger.warning("Unknown LOG_LEVEL %r; using INFO.", _log_level_setting)


@contextmanager
def log_step(
    name: str,
    number: Optional[Union[int, str]] = None,
    token_tally: Optional[list] = None,
):
    """Log '<n>. <name> Started' / '<name> Ended: X sec.' around a pipeline step.

    Set step["model"] and step["usage"] (a usage_metadata dict) inside the block
    to also log the tokens used; they are appended to `token_tally` when given.
    """
    step: dict[str, Any] = {"model": None, "usage": None}
    logger.info("%s%s Started", f"{number}. " if number is not None else "", name)
    started = time.perf_counter()
    status = "Ended"
    try:
        yield step
    except Exception:
        status = "Failed"
        raise
    except BaseException:
        status = "Cancelled"
        raise
    finally:
        logger.info("%s %s: %.1f sec.", name, status, time.perf_counter() - started)
        if step["usage"]:
            logger.info(
                "Tokens used (%s): %s",
                step["model"] or "unknown model",
                step["usage"],
            )
            if token_tally is not None:
                token_tally.append(
                    _token_tally_entry(name, step["model"], step["usage"])
                )


def _token_tally_entry(step_name: str, model: Optional[str], usage: dict) -> dict:
    """One per-step row of RunState.token_usage["by_step"]."""
    reasoning = (usage.get("output_token_details") or {}).get("reasoning")
    entry = {
        "step": step_name,
        "model": model,
        "input_tokens": usage.get("input_tokens") or 0,
        "output_tokens": usage.get("output_tokens") or 0,
        "total_tokens": usage.get("total_tokens") or 0,
    }
    if reasoning:
        entry["output_token_details"] = {"reasoning": reasoning}
    return entry


def summarise_token_tally(
    token_tally: list, elapsed_seconds: Optional[float] = None
) -> dict:
    """Sum per-step token usage into RunState.token_usage and log the run total."""

    def sum_usage(entries: list) -> dict:
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        reasoning = 0
        for entry in entries:
            for key in usage:
                usage[key] += entry.get(key) or 0
            reasoning += (entry.get("output_token_details") or {}).get("reasoning") or 0
        if reasoning:
            usage["output_token_details"] = {"reasoning": reasoning}
        return usage

    total = sum_usage(token_tally)
    entries_by_model: dict[str, list] = {}
    for entry in token_tally:
        entries_by_model.setdefault(entry["model"] or "unknown model", []).append(entry)
    by_model = {model: sum_usage(entries) for model, entries in entries_by_model.items()}

    logger.info("─" * 60)
    logger.info(
        "End Query - total usage%s: %s",
        f" ({elapsed_seconds:.1f} sec.)" if elapsed_seconds is not None else "",
        total,
    )
    for model, usage in by_model.items():
        logger.info("    %s: %s", model, usage)
    # Per-step lines were already logged by log_step; repeat them only at DEBUG.
    for entry in token_tally:
        logger.debug(
            "    %s (%s): %s",
            entry["step"],
            entry["model"] or "unknown model",
            {k: v for k, v in entry.items() if k not in ("step", "model")},
        )
    return {"total": total, "by_model": by_model, "by_step": list(token_tally)}


GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"

# Answer-generation models and reasoning levels selectable from the
# frontend (`frontend/components/model-selector.tsx`). Edit
# `frontend/config/models.json` — this module only loads and checks it.
# `GraphRAG/main.py` exposes the same values via `GET /options`.
# `PlantBioRAG.query()` falls back to `defaultModel` /
# `defaultReasoningLevel` only when no selection was made at all (e.g. an
# older frontend build); an explicit-but-unrecognised value fails the run
# instead of silently substituting a different model - see
# `_resolve_model_name`/`_resolve_reasoning_level`.
#
# Spans two providers: ids starting with "gpt-" go through `ChatOpenAI`,
# everything else through `ChatGoogleGenerativeAI` (see `_model_provider`).
# Both providers use the same reasoning-level vocabulary, so
# `reasoningLevels` is shared across every model.
#
# `defaultModel` is also the fixed model for internal helper calls
# (question expansion, accession extraction). Those calls always use
# `ChatGoogleGenerativeAI`, so the default must be a Gemini id.
# `thinkingBudgets` is the token budget for Gemini models that predate
# `thinking_level` (e.g. gemini-2.5-flash, which rejects `thinking_level`
# with an API 400). GPT models ignore it and take `reasoning_level`
# directly as OpenAI's `reasoning_effort`.
_MODEL_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "frontend" / "config" / "models.json"
)


def _string_list(config: dict, key: str, path: Path) -> List[str]:
    value = config.get(key)
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item.strip() for item in value)
    ):
        raise RuntimeError(f"{path}: {key} must be a non-empty list of strings")
    if len(value) != len(set(value)):
        raise RuntimeError(f"{path}: {key} contains duplicates")
    return value


def _load_model_config(path: Path) -> dict:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise RuntimeError(f"Model config not found at {path}") from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Model config at {path} is not valid JSON: {e}") from e
    if not isinstance(config, dict):
        raise RuntimeError(f"{path} must be a JSON object")

    models = _string_list(config, "models", path)
    reasoning_levels = _string_list(config, "reasoningLevels", path)
    default_model = config.get("defaultModel")
    default_reasoning = config.get("defaultReasoningLevel")
    if default_model not in models:
        raise RuntimeError(f"{path}: defaultModel {default_model!r} is not in models")
    if not str(default_model).startswith("gemini-"):
        raise RuntimeError(
            f"{path}: defaultModel is also the internal helper model and must "
            "be a Gemini model id"
        )
    if default_reasoning not in reasoning_levels:
        raise RuntimeError(
            f"{path}: defaultReasoningLevel {default_reasoning!r} is not in "
            "reasoningLevels"
        )

    budgets = config.get("thinkingBudgets")
    if not isinstance(budgets, dict):
        raise RuntimeError(f"{path}: thinkingBudgets must be an object")
    parsed_budgets: Dict[str, int] = {}
    for level in reasoning_levels:
        budget = budgets.get(level)
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
            raise RuntimeError(
                f"{path}: thinkingBudgets.{level} must be a non-negative integer"
            )
        parsed_budgets[level] = budget

    return {
        "models": models,
        "defaultModel": default_model,
        "reasoningLevels": reasoning_levels,
        "defaultReasoningLevel": default_reasoning,
        "thinkingBudgets": parsed_budgets,
    }


_MODEL_CONFIG = _load_model_config(_MODEL_CONFIG_PATH)
GEMINI_MODEL = _MODEL_CONFIG["defaultModel"]
AVAILABLE_MODELS = _MODEL_CONFIG["models"]
AVAILABLE_REASONING_LEVELS = _MODEL_CONFIG["reasoningLevels"]
ANSWER_THINKING_LEVEL = _MODEL_CONFIG["defaultReasoningLevel"]
REASONING_LEVEL_TO_THINKING_BUDGET = _MODEL_CONFIG["thinkingBudgets"]


def _model_provider(model_name: str) -> str:
    """ "openai" for GPT models (routed through `ChatOpenAI`/the Responses
    API), "google" for everything else (Gemini, via
    `ChatGoogleGenerativeAI`). Add new GPT models to `frontend/config/models.json`
    freely - anything named "gpt-*" is picked up automatically."""
    return "openai" if model_name.startswith("gpt-") else "google"


def _model_supports_thinking_level(model_name: str) -> bool:
    """Only Gemini 3+ models accept `thinking_level`/`reasoning_effort`."""
    return model_name.startswith("gemini-3")


def _thinking_kwargs(model_name: str, reasoning_level: str) -> Dict[str, Any]:
    """Translates a resolved reasoning level into whichever thinking
    parameter `model_name` actually accepts, for use both when constructing
    a `ChatGoogleGenerativeAI` and on each `astream`/`invoke` call. Only
    called for `_model_provider(model_name) == "google"` - GPT models take
    their reasoning effort at construction time instead (see
    `_get_answer_llm`)."""
    if _model_supports_thinking_level(model_name):
        return {"thinking_level": reasoning_level}
    return {"thinking_budget": REASONING_LEVEL_TO_THINKING_BUDGET[reasoning_level]}


class UnavailableModelSelectionError(ValueError):
    """Raised by `_resolve_model_name`/`_resolve_reasoning_level` when the
    caller explicitly asked for a model/reasoning level that isn't in
    `AVAILABLE_MODELS`/`AVAILABLE_REASONING_LEVELS`. Raised (not
    substituted) deliberately: silently answering with a different model
    than the one requested would be misleading, so `query()` lets this
    propagate and fail the run - `main.py`'s `_run_agui_events` turns it
    into a `RunErrorEvent`, and the CLI in this file's `main()` just lets
    it crash (its `--model`/`--reasoning-level` choices are already
    restricted by argparse, so it shouldn't be reachable from the CLI)."""


def _resolve_model_name(model_name: Optional[str]) -> str:
    """No selection at all (`None`/empty - e.g. an older frontend build)
    falls back to the default model. An explicit-but-unrecognised
    selection is a hard failure instead - see
    `UnavailableModelSelectionError`."""
    if not model_name:
        return GEMINI_MODEL
    if model_name in AVAILABLE_MODELS:
        return model_name
    raise UnavailableModelSelectionError(
        f"Unknown model {model_name!r} requested; available models are "
        f"{AVAILABLE_MODELS}"
    )


def _resolve_reasoning_level(reasoning_level: Optional[str]) -> str:
    """Same fallback-vs-fail policy as `_resolve_model_name`, for the
    reasoning level."""
    if not reasoning_level:
        return ANSWER_THINKING_LEVEL
    if reasoning_level in AVAILABLE_REASONING_LEVELS:
        return reasoning_level
    raise UnavailableModelSelectionError(
        f"Unknown reasoning level {reasoning_level!r} requested; available "
        f"levels are {AVAILABLE_REASONING_LEVELS}"
    )


def _resolve_max_context_chars(max_context_chars: Optional[int]) -> int:
    """Same fallback-vs-fail policy as `_resolve_model_name`, for the
    literature context character budget selected by the frontend's
    literature context slider (`frontend/components/literature-context-
    selector.tsx`). Must be an integer within
    `[MIN_LITERATURE_CONTEXT_CHARS, MAX_LITERATURE_CONTEXT_CHARS]`."""
    if max_context_chars is None:
        return MAX_CHARACTERS
    if (
        isinstance(max_context_chars, bool)
        or not isinstance(max_context_chars, int)
        or not (
            MIN_LITERATURE_CONTEXT_CHARS
            <= max_context_chars
            <= MAX_LITERATURE_CONTEXT_CHARS
        )
    ):
        raise ValueError(
            f"Invalid max literature context {max_context_chars!r} requested; "
            f"must be an integer between {MIN_LITERATURE_CONTEXT_CHARS} and "
            f"{MAX_LITERATURE_CONTEXT_CHARS} characters"
        )
    return max_context_chars


# CONTEXT OPTIMISATION
# Metadata retrieval: filter taxon, then fetch vector/full-text candidates.
SPECIES_FILTER_ENABLED = True
METADATA_FILTER_OVERFETCH_MULTIPLIER = 5  # Extra candidates before species filtering.
METADATA_VECTOR_K = 40  # Matching vector candidates retained per expanded query.
METADATA_FULLTEXT_K = 40  # Matching full-text candidates retained per expanded query.

# Shared ranking: RRF is used by literature, metadata, and Pretzel retrieval.
RRF_RANK_CONSTANT = 60  # Standard reciprocal-rank denominator offset.

# Metadata retrieval: elbow cutoff, neighbor expansion, then context limits.
METADATA_RRF_MIN_SEEDS = 5  # Minimum retained before a score-drop cutoff is allowed.
METADATA_RRF_SCAN_LIMIT = 80  # Ranked candidates inspected to find an elbow.
METADATA_RRF_MIN_RELATIVE_DROP = 0.35  # Minimum fractional drop between adjacent scores.
METADATA_RRF_MIN_ABSOLUTE_DROP = 0.005  # Minimum absolute drop as well as relative drop.
METADATA_MAX_RESULTS_AFTER_RRF = 40  # Hard cap after applying the elbow cutoff.
METADATA_NEIGHBORS_PER_SEED = 2  # Maximum adjacent metadata nodes per selected seed.
METADATA_MAX_TRIPLES = 50  # Maximum relationship descriptions included in metadata context.
METADATA_MAX_CONTEXT_CHARS = 20000  # Character budget for metadata context.

# Literature/Pretzel retrieval defaults; vector and full-text limits are per query.
QUERY_VECTOR_MAX_CHUNKS = 40
QUERY_FULL_TEXT_MAX_CHUNKS = 40
QUERY_MAX_CHUNKS = 80  # Maximum RRF-ranked literature chunks per expanded query.
# Bounds/step for the literature context budget the frontend can select
# per run (see `_resolve_max_context_chars`); exposed via `GET /options`.
MIN_LITERATURE_CONTEXT_CHARS = 10000
MAX_LITERATURE_CONTEXT_CHARS = 500000
LITERATURE_CONTEXT_CHARS_STEP = 10000
MAX_TRIPLES = 50
LITERATURE_TRIPLE_MAX_CHARACTERS = 5000  # Separate budget for literature relationships.
RERANK_MAX_TEXT_CHARS = 1500  # Passage excerpt sent to the semantic evidence judge.
RERANK_CANDIDATES_PER_QUERY = 3  # Preserve candidates from each generated query.
RERANK_BATCH_SIZE = 16  # Batch size for the extra semantic-ranking calls.
RERANK_MAX_CONCURRENT_BATCHES = 3  # Independent judge calls in flight; lower if API rate-limited.
RERANK_MIN_SCORE = 2.0  # Exclude topic-only mentions; allow indirect and direct evidence.
RERANK_MAX_METADATA_CHARS = 3000  # Metadata scope shown to the evidence judge.
RERANK_MODEL = "gemini-2.5-flash"  # Lower-cost model used only for literature relevance scoring.

# USER ADJUSTABLE VARIABLES 
LITERATURE_CHUNKS_TO_RERANK = 80  # Maximum literature chunks sent to relevance ranking per request.
MAX_CHARACTERS = 50000  # The final size of the literature context sent to the LLM.

# Accession API config
ACCESSION_API_URL = os.getenv("ACCESSION_API_URL") or ""
ACCESSION_API_TOKEN = "research_accessions"
ACCESSION_API_TIMEOUT = 120

METADATA_MAX_CHARACTERS = 50000

SEMANTIC_CACHE_INDEX = "semantic_cache_vector"
SEMANTIC_CACHE_THRESHOLD = 0.92
SEMANTIC_CACHE_TOP_K = 1
SEMANTIC_CACHE_TTL_DAYS = 14


class Stage(str, Enum):
    """Ordered stages of a `PlantBioRAG.query()` run, named after the
    existing `logger.info(...)` milestones."""

    EXPANDING_QUESTION = "expanding_question"  # "Analysis of question"
    RETRIEVING_CONTEXT = "retrieving_context"  # "Context retrieval"
    JUDGING_RELEVANCE = "judging_relevance"  # "Literature relevance judge"
    GENERATING_ANSWER = "generating_answer"  # "Call LLM to answer"
    CHECKING_AGG_ACCESSIONS = (
        "checking_agg_accessions"  # "Extract accessions" / "Call accession API"
    )
    PRESENTING_ACCESSIONS = (
        "presenting_accessions"  # "Summarise and present accession results"
    )


class RunState(BaseModel):
    """Minimal, JSON-serializable snapshot of where a `query()` run is at."""

    stage: Stage
    expanded_question: Optional[str] = None
    # Queries actually used for retrieval: the original question first, then
    # the deduplicated expansions. Set on entering RETRIEVING_CONTEXT, so it
    # stays empty for cache hits and direct AGG lookups.
    retrieval_queries: List[str] = Field(default_factory=list)
    # Answer-generation model/reasoning level actually used for this run,
    # after `_resolve_model_name`/`_resolve_reasoning_level` have applied
    # their fallbacks - lets the frontend confirm the selection took effect.
    model_name: str = GEMINI_MODEL
    reasoning_level: str = ANSWER_THINKING_LEVEL
    # Literature context character budget actually used for this run, after
    # `_resolve_max_context_chars` has applied its fallback.
    max_context_chars: int = MAX_CHARACTERS
    species: str = ""
    is_agg_accession_query: bool = False
    needs_clarification: bool = False
    accessions: List[str] = Field(default_factory=list)
    usage_metadata: dict = Field(default_factory=dict)
    # Run-wide tally of every LLM call: {"total": usage, "by_step": [{step, model, ...usage}]}.
    token_usage: dict = Field(default_factory=dict)
    # Raw context strings retrieved from Neo4j and injected into the
    # answer-generation prompt by `_build_answer_prompt` (see there for the
    # exact `### [Source: ...]` framing each is wrapped in). Exposed here so
    # callers (e.g. the frontend's pipeline status panel) can inspect exactly
    # what was retrieved, independent of the final cited answer text.
    literature_context: Optional[str] = None
    metadata_context: Optional[str] = None
    pretzel_context: Optional[str] = None
    retrieval_diagnostics: dict = Field(default_factory=dict)
    error: Optional[str] = None


# Internal, protocol-agnostic events yielded by `PlantBioRAG.run()`.
# The wiring layer (main.py) maps these to `ag_ui.core` events.
@dataclass
class StageChangeEvent:
    state: RunState


@dataclass
class TextEvent:
    text: str


@dataclass
class ReasoningEvent:
    """A chunk of the model's thinking/reasoning trace, emitted separately
    from the final answer text (see `include_thoughts` on `self.llm`)."""

    text: str


@dataclass
class ResultEvent:
    state: RunState


@dataclass
class ErrorEvent:
    state: RunState


RunEvent = Union[StageChangeEvent, TextEvent, ReasoningEvent, ResultEvent, ErrorEvent]

global_instruction_and_information = getPrompt("global_instruction_and_information")


class PlantBioRAG:
    def __init__(self):
        self.emb = GoogleGenerativeAIEmbeddings(model=GEMINI_EMBEDDING_MODEL)
        # Fixed model for internal helper calls (question/query expansion,
        # accession extraction/presentation) - these aren't exposed to the
        # frontend's model/reasoning selectors, only the final answer is.
        self.llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
        self._reranker_llm = ChatGoogleGenerativeAI(
            model=RERANK_MODEL, temperature=0, thinking_budget=0
        )
        self._chunk_reranker_usage = local()
        self._chunk_reranker_usage.value = {
            "model": RERANK_MODEL,
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "available": True,
        }
        # Answer-generation clients, one per (model_name, reasoning_level)
        # combo actually requested so far, built lazily by `_get_answer_llm`.
        # Separate from `self.llm` so only the final-answer call requests
        # thought text - other `_llm_invoke` calls (JSON extraction,
        # accession presentation) would otherwise pay for unused thinking
        # tokens.
        self._answer_llm_cache: Dict[Tuple[str, str], ChatGoogleGenerativeAI] = {}
        self.graph = Neo4jGraph()
        self.vs = Neo4jVector(
            embedding=self.emb,
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            node_label="Chunk",
            text_node_property="text",
            embedding_node_property="embedding",
            index_name="vector",
        )
        if useCache:
            self._ensure_semantic_cache_vector_index()
            self._clear_expired_semantic_cache()

    def _ensure_semantic_cache_vector_index(self) -> None:
        try:
            self.graph.query(
                """
                CREATE VECTOR INDEX semantic_cache_vector IF NOT EXISTS
                FOR (c:SemanticCache)
                ON (c.embedding)
                OPTIONS {
                  indexConfig: {
                    `vector.dimensions`: 3072,
                    `vector.similarity_function`: 'cosine'
                  }
                }
                """
            )
            logger.info("Semantic cache vector index ensured: %s", SEMANTIC_CACHE_INDEX)
        except Exception as e:
            logger.warning("Failed to ensure semantic cache vector index: %s", e)

    def _clear_expired_semantic_cache(self) -> None:
        try:
            res = self.graph.query(
                """
                MATCH (c:SemanticCache)
                WHERE c.created_at IS NOT NULL
                  AND c.created_at < datetime() - duration({days: $ttl_days})
                WITH collect(c) AS expired, count(c) AS deleted_count
                FOREACH (n IN expired | DETACH DELETE n)
                RETURN deleted_count
                """,
                params={"ttl_days": SEMANTIC_CACHE_TTL_DAYS},
            )
            deleted_count = res[0]["deleted_count"] if res else 0
            logger.info("Cleared expired semantic cache entries: %s", deleted_count)
        except Exception as e:
            logger.warning("Failed to clear expired semantic cache entries: %s", e)

    # 1. Literature Graph RAG
    # 1. Vector indexing
    def _vector_chunks(
        self, q: str, k: int = QUERY_VECTOR_MAX_CHUNKS
    ) -> Dict[str, float]:
        out = {}
        # 1 means the vectors are identical (most similar). 0 means the vectors are diametrically opposite (most dissimilar).
        for doc, score in self.vs.similarity_search_with_score(q, k=k):
            # if score < 0.25:
            #     continue
            cid = doc.metadata.get("chunk_id")
            if cid:
                out[cid] = max(out.get(cid, 0), score)
        return out

    def escape_lucene_plain_text(self, q: str) -> str:
        _LUCENE_SPECIAL_CHARS = re.compile(r'([+\-!(){}\[\]^"~*?:\\\/&|])')
        _LUCENE_BOOLEAN_WORDS = re.compile(r"\b(AND|OR|NOT)\b")
        # Treat user input as plain text for a Lucene-backed Neo4j fulltext query: Escapes Lucene query parser metacharacters; Lowercases uppercase Boolean operators so they are searched as words; Normalises whitespace.
        if not q:
            return ""
        q = re.sub(r"\s+", " ", q).strip()
        q = _LUCENE_BOOLEAN_WORDS.sub(lambda m: m.group(1).lower(), q)
        q = _LUCENE_SPECIAL_CHARS.sub(r"\\\1", q)
        return q

    # Run vector + full-text retrieval concurrently
    def _hybrid_scores_concurrent(
        self,
        q: str,
        vector_fn,
        fulltext_fn,
        k: int,
        rrf_k: int = RRF_RANK_CONSTANT,
        diagnostics: Optional[dict] = None,
    ) -> Dict[str, float]:
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as executor:
            vector_future = executor.submit(vector_fn, q, k)
            fulltext_future = executor.submit(fulltext_fn, q, k)
            vector_scores = vector_future.result()
            fulltext_scores = fulltext_future.result()
        fused = self._rrf_fusion(vector_scores, fulltext_scores, rrf_k)
        if diagnostics is not None:
            diagnostics.update({
                "query": q, "escaped_fulltext_query": self.escape_lucene_plain_text(q),
                "effective_vector_k": k, "effective_fulltext_k": k,
                "rrf_constant": rrf_k, "elapsed_seconds": time.perf_counter() - started,
                "return_order_note": "Insertion order of unique chunk IDs in the existing score maps, after deduplication and missing-ID removal.",
                "ranking": {},
            })
            for label, scores in (("vector", vector_scores), ("fulltext", fulltext_scores)):
                return_order = {cid: index for index, cid in enumerate(scores, start=1)}
                diagnostics["ranking"][label] = [
                    {"chunk_id": cid, "score": float(score), "rank": rank,
                     "return_order": return_order[cid], "rrf_contribution": 1.0 / (rrf_k + rank)}
                    for rank, (cid, score) in enumerate(
                        sorted(scores.items(), key=lambda item: item[1], reverse=True), start=1
                    )
                ]
            diagnostics["ranking"]["rrf"] = [
                {"chunk_id": cid, "score": fused[cid], "rank": rank}
                for rank, cid in enumerate(sorted(fused, key=fused.get, reverse=True), start=1)
            ]
            diagnostics["ties"] = {}
            for label, ranking in diagnostics["ranking"].items():
                groups = {}
                for item in ranking:
                    groups.setdefault(item["score"], []).append(item["chunk_id"])
                diagnostics["ties"][label] = [
                    {"score": score, "chunk_ids_in_rank_order": ids}
                    for score, ids in groups.items() if len(ids) > 1
                ]
        return fused

    # Run expanded-query searches concurrently
    def _multi_query_hybrid_scores_concurrent(
        self,
        expanded_queries: list[str],
        vector_fn,
        fulltext_fn,
        k: int,
        rrf_k: int = RRF_RANK_CONSTANT,
    ) -> Dict[str, float]:
        all_fused: Dict[str, float] = {}
        with ThreadPoolExecutor(
            max_workers=min(8, max(1, len(expanded_queries)))
        ) as executor:
            future_to_query = {
                executor.submit(
                    self._hybrid_scores_concurrent,
                    eq,
                    vector_fn,
                    fulltext_fn,
                    k,
                    rrf_k,
                ): eq
                for eq in expanded_queries
            }
            for future in as_completed(future_to_query):
                fused = future.result()
                for key, score in fused.items():
                    all_fused[key] = all_fused.get(key, 0.0) + score
        return all_fused

    # One literature search branch for one expanded query
    def _search_literature_one_query(
        self,
        expanded_query: str,
        k: int,
        original_question: str,
        taxon_filter: Optional[dict[str, Any]] = None,
        query_label: str = "query",
    ) -> dict:
        query_started = time.perf_counter()
        search_trace = {}
        fused = self._hybrid_scores_concurrent(
            expanded_query, self._vector_chunks, self._fulltext_chunks, k,
            diagnostics=search_trace,
        )
        ranked_cids = sorted(fused, key=lambda cid: (-fused[cid], cid))
        candidate_rows = self.graph.query(
            """
            MATCH (c:Chunk) WHERE c.chunk_id IN $cids
            RETURN c.chunk_id AS chunk_id, c.source_path AS source_path,
                   c.text AS text
            """,
            params={"cids": ranked_cids},
        ) if ranked_cids else []
        candidate_by_id = {row["chunk_id"]: row for row in candidate_rows}
        taxonomy_rejected_ids = {
            cid for cid in ranked_cids
            if cid in candidate_by_id
            and text_is_clearly_other_taxon(candidate_by_id[cid].get("text"), taxon_filter)
        }
        eligible_ranked_cids = [cid for cid in ranked_cids if cid not in taxonomy_rejected_ids]
        seed_cids = eligible_ranked_cids[:k]
        outside_top_k_ids = eligible_ranked_cids[k:]
        hybrid_search_seconds = time.perf_counter() - query_started
        seeded_chunks, expanded_chunks, triples, expansion_timings = expand_literature_graph(
            self.graph, seed_cids, [candidate_by_id[cid] for cid in seed_cids if cid in candidate_by_id]
        )
        logger.debug(
            'Literature query [%s] "%s": %.2fs = hybrid search %.2fs (%d seeds) + graph neighbours %.2fs (%d neighbour chunks, %d triples)',
            query_label, expanded_query,
            hybrid_search_seconds + expansion_timings["total_seconds"],
            hybrid_search_seconds, expansion_timings.get("seed_count", 0),
            expansion_timings["total_seconds"], expansion_timings.get("neighbor_count", 0),
            expansion_timings.get("triple_count", 0),
        )
        search_trace["graph_expansion"] = expansion_timings
        postprocess_started = time.perf_counter()
        all_chunks = self._dedupe_chunks(seeded_chunks + expanded_chunks)
        expanded_chunk_scores: dict[str, float] = {}
        seed_ids_by_expanded_chunk: dict[Any, set[str]] = {}
        for triple in triples:
            seed_id = triple.get("seed_chunk_id")
            for expanded_id in triple.get("expanded_chunk_ids", [triple.get("expanded_chunk_id")]):
                if seed_id:
                    seed_ids_by_expanded_chunk.setdefault(expanded_id, set()).add(seed_id)
                if expanded_id and seed_id:
                    expanded_chunk_scores[expanded_id] = max(expanded_chunk_scores.get(expanded_id, 0.0), fused.get(seed_id, 0.0))
        provenance_index_seconds = time.perf_counter() - postprocess_started

        candidates = []
        expanded_chunk_ids = {chunk.get("chunk_id") for chunk in expanded_chunks}
        for cid in taxonomy_rejected_ids:
            row = candidate_by_id[cid]
            candidates.append({
                "chunk_id": cid, "source_path": row.get("source_path", ""),
                "text": row.get("text", ""), "rrf_score": fused[cid],
                "included": False,
                "rejection_reason": "Clearly identified as a different taxonomy",
            })
        for cid in outside_top_k_ids:
            if cid in expanded_chunk_ids:
                continue
            row = candidate_by_id.get(cid)
            if row and row.get("text"):
                candidates.append({
                    "chunk_id": cid, "source_path": row.get("source_path", ""),
                    "text": row["text"], "rrf_score": fused[cid],
                    "included": False, "rejection_reason": "Outside RRF top-K",
                })
        for chunk in all_chunks:
            text = chunk.get("text", "")
            if not text:
                continue
            chunk_id = chunk.get("chunk_id")
            candidates.append({
                "chunk_id": chunk_id, "source_path": chunk.get("source_path", ""),
                "text": text,
                "rrf_score": fused.get(chunk_id, expanded_chunk_scores.get(chunk_id)),
                "is_direct_search_match": chunk_id in fused,
                "rejection_reason": (
                    "Clearly identified as a different taxonomy"
                    if text_is_clearly_other_taxon(text, taxon_filter) else None
                ),
            })
        excluded_seed_ids = {
            item["chunk_id"] for item in candidates
            if item["rejection_reason"] and item["chunk_id"] in seed_cids
        }
        for item in candidates:
            item["included"] = False
            cid = item.get("chunk_id")
            item["retrieval_queries"] = [{
                "query": expanded_query,
                "per_query_rrf_score": item.get("rrf_score"),
                "first_query_rejection_reason": item.get("rejection_reason"),
                "is_seed": cid in seed_cids,
                "is_expanded_chunk": cid in expanded_chunk_ids,
                "expanded_from_seed_ids": sorted(seed_ids_by_expanded_chunk.get(cid, ())),
            }]
        search_trace.update({
            "seed_limit": k, "seed_chunk_ids": seed_cids,
            "taxonomy_rejected_ids": sorted(taxonomy_rejected_ids),
            "outside_rrf_top_k_ids": outside_top_k_ids,
            "expanded_chunk_ids": sorted(cid for cid in expanded_chunk_ids if cid),
            "missing_graph_chunk_ids": [cid for cid in ranked_cids if cid not in candidate_by_id],
            "fetched_chunk_metadata": [
                {"chunk_id": cid, "source_path": row.get("source_path"),
                 "text_chars": len(row.get("text") or ""),
                 "text_sha256": hashlib.sha256((row.get("text") or "").encode("utf-8")).hexdigest()}
                for cid, row in candidate_by_id.items()
            ],
        })

        relationship_dedup_started = time.perf_counter()
        triple_by_text: dict[str, dict[str, Any]] = {}
        for triple in triples:
            text = triple["text"]
            seed_id = triple.get("seed_chunk_id")
            score = fused.get(seed_id, 0.0)
            taxonomy_rejected = seed_id in excluded_seed_ids
            current = triple_by_text.get(text)
            if current is None or (
                (current["rejection_reason"] and not taxonomy_rejected)
                or (bool(current["rejection_reason"]) == taxonomy_rejected
                    and score > current["rrf_score"])
            ):
                relevance_score = current["relevance_score"] if current is not None else self._triple_relevance_score(
                    original_question, re.sub(r"^\[Source: .*?\]\s*", "", text)
                )
                triple_by_text[text] = {
                    "text": text, "rrf_score": score,
                    "relevance_score": relevance_score,
                    "included": False,
                    "rejection_reason": (
                        "Clearly identified as a different taxonomy" if taxonomy_rejected else None
                    ),
                }
        postprocess_timings = {
            "provenance_index_seconds": provenance_index_seconds,
            "candidate_build_seconds": relationship_dedup_started - postprocess_started - provenance_index_seconds,
            "relationship_dedup_seconds": time.perf_counter() - relationship_dedup_started,
            "total_seconds": time.perf_counter() - postprocess_started,
        }
        search_trace["post_expansion_processing"] = postprocess_timings
        logger.debug(
            'Literature post-expansion [%s] "%s": %.2fs (provenance index %.2fs, candidates %.2fs, relationship dedup %.2fs)',
            query_label, expanded_query, postprocess_timings["total_seconds"],
            postprocess_timings["provenance_index_seconds"],
            postprocess_timings["candidate_build_seconds"],
            postprocess_timings["relationship_dedup_seconds"],
        )
        return {
            "expanded_query": expanded_query,
            "search_trace": search_trace,
            "diagnostics": {"chunks": candidates, "triples": list(triple_by_text.values())},
        }

    @staticmethod
    def _triple_relevance_score(question: str, relationship: str) -> float:
        """Favor question entities and biological edges over graph boilerplate."""
        question_lower = question.casefold()
        question_words = set(re.findall(r"[a-z0-9]+", question_lower))
        stop_words = {
            "which", "what", "where", "when", "who", "does", "do", "any",
            "the", "of", "in", "on", "to", "for", "a", "an", "is", "are",
            "have", "has", "carry", "carries", "carrying", "genome", "genomes",
        }
        question_terms = {
            token for token in re.findall(r"[a-z0-9]+", question_lower)
            if token not in stop_words and not token.isdigit()
        }
        endpoints = re.sub(r"-\[[A-Z_]+\]->", " ", relationship)
        endpoint_terms = set(re.findall(r"[a-z0-9]+", endpoints.casefold()))
        overlap = len(question_terms & endpoint_terms)
        question_entities = set(re.findall(
            r"\b(?:lr|yr)\s*\d+[a-z0-9]*\b", question_lower
        ))
        matched_entities = sum(
            bool(re.search(rf"\b{re.escape(entity)}\b", endpoints.casefold()))
            for entity in question_entities
        )
        score = matched_entities * 10.0 + min(overlap, 8) * 2.5
        relation_match = re.search(r"-\[([A-Z_]+)\]->", relationship)
        normalized_type = relation_match.group(1).replace("_", "") if relation_match else ""
        if normalized_type in {"CARRIES", "CARRIESGENE", "HASGENE"} and question_words & {
            "carry", "carries", "carrying", "have", "has", "possess",
            "possesses", "contain", "contains",
        }:
            score += 15.0
        if normalized_type in {
            "PLEIOTROPICTO", "CONFERSRESISTANCETO", "CARRIES", "CARRIESGENE",
            "HASGENE", "LOCATEDON", "ASSOCIATEDWITH", "ASSOCIATEDWITHTRAIT",
        }:
            score += 1.0
        if normalized_type in {"CITEDIN", "HASDOI", "BELONGSTOSPECIES"}:
            score -= 5.0
        if normalized_type in {"HASPROJECT", "HASASSEMBLY"}:
            score -= 2.0
        return score

    def _semantic_rerank_chunks(
        self,
        question: str,
        expanded_queries: list[str],
        metadata_context: str,
        candidates: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Judge answer evidence in small batches with the dedicated reranker LLM."""
        results: dict[str, dict[str, Any]] = {}
        self._chunk_reranker_usage.batches = []
        self._chunk_reranker_usage.value = {
            "model": RERANK_MODEL,
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "available": True,
        }
        query_hints = [query for query in expanded_queries if query != question][:12]
        metadata_scope = metadata_context[:RERANK_MAX_METADATA_CHARS]
        prepared_batches = []
        for start in range(0, len(candidates), RERANK_BATCH_SIZE):
            batch = candidates[start : start + RERANK_BATCH_SIZE]
            batch_number = start // RERANK_BATCH_SIZE + 1
            for index, candidate in enumerate(batch):
                candidate["sent_to_relevance_ranker"] = True
                candidate["reranker_batch"] = batch_number
                candidate["reranker_passage_id"] = index
                candidate["reranker_text_chars"] = len(candidate.get("text", "")[:RERANK_MAX_TEXT_CHARS])
                candidate["reranker_text_truncated"] = len(candidate.get("text", "")) > RERANK_MAX_TEXT_CHARS
            payload = [
                {
                    "id": index,
                    "source": item.get("source_path", ""),
                    "text": item.get("text", "")[:RERANK_MAX_TEXT_CHARS],
                }
                for index, item in enumerate(batch)
            ]
            prompt = getPrompt("literature_relevance_judge").rstrip() + "\n" + json.dumps(
                {
                    "question": question,
                    "generated_queries": query_hints,
                    "metadata_scope": metadata_scope,
                    "passages": payload,
                },
                ensure_ascii=False,
            )
            batch_trace = {
                "batch": batch_number, "model": RERANK_MODEL,
                "candidate_ids_in_input_order": [item.get("diagnostic_id") or item.get("chunk_id") for item in batch],
                "prompt": prompt, "prompt_chars": len(prompt),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "status": "started",
            }
            self._chunk_reranker_usage.batches.append(batch_trace)
            prepared_batches.append((batch, prompt, batch_trace))
        with ThreadPoolExecutor(max_workers=min(RERANK_MAX_CONCURRENT_BATCHES, len(prepared_batches) or 1)) as executor:
            submitted = [(batch, trace, executor.submit(self._reranker_llm.invoke, prompt), time.perf_counter())
                         for batch, prompt, trace in prepared_batches]
            for batch, batch_trace, future, started in submitted:
                try:
                    response = future.result()
                except Exception as error:
                    batch_trace.update({"status": "call_failed", "error": str(error),
                                        "elapsed_seconds": time.perf_counter() - started})
                    raise
                batch_trace.update({
                    "status": "response_received", "elapsed_seconds": time.perf_counter() - started,
                    "response_id": getattr(response, "id", None),
                    "response_metadata": getattr(response, "response_metadata", {}),
                })
                usage_totals = self._chunk_reranker_usage.value
                usage_totals["calls"] += 1
                usage = getattr(response, "usage_metadata", None)
                if not isinstance(usage, dict):
                    usage = {}
                batch_trace["usage_metadata"] = usage
                logger.debug(
                    "Relevance judge batch %d/%d: %.2fs, %d passages, %s in / %s out tokens",
                    batch_trace["batch"], len(prepared_batches), batch_trace["elapsed_seconds"],
                    len(batch), usage.get("input_tokens", "?"), usage.get("output_tokens", "?"),
                )
                for token_key in ("input_tokens", "output_tokens", "total_tokens"):
                    token_count = usage.get(token_key)
                    if isinstance(token_count, int) and not isinstance(token_count, bool):
                        usage_totals[token_key] += token_count
                    else:
                        usage_totals["available"] = False
                raw = self._message_text(response).strip()
                batch_trace["raw_response"] = raw
                clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I)
                try:
                    parsed = json.loads(clean)
                except Exception as error:
                    batch_trace.update({"status": "parse_failed", "error": str(error)})
                    raise
                batch_trace["parsed_response"] = parsed
                batch_trace["status"] = "parsed"
                for item in parsed.get("items", []):
                    try:
                        candidate = batch[int(item["id"])]
                        score = max(0.0, min(4.0, float(item["score"])))
                    except (KeyError, TypeError, ValueError, IndexError):
                        continue
                    entities = item.get("entities", [])
                    if not isinstance(entities, list):
                        entities = []
                    reason = item.get("reason", "")
                    results[str(candidate.get("chunk_id") or id(candidate))] = {
                        "semantic_score": score,
                        "semantic_reason": reason.strip() if isinstance(reason, str) else "",
                        "evidence_entities": sorted({
                            " ".join(entity.split())
                            for entity in entities
                            if isinstance(entity, str) and entity.strip()
                        }),
                    }
        return results

    @staticmethod
    def _boost_connected_triple_relevance(
        question: str, triples: list[dict[str, Any]]
    ) -> None:
        """Promote one-hop biological evidence linked to a question-matched triple."""
        stop_words = {
            "which", "what", "where", "when", "who", "does", "do", "any",
            "the", "of", "in", "on", "to", "for", "a", "an", "is", "are",
            "have", "has", "carry", "carries", "carrying", "genome", "genomes",
        }
        question_terms = {
            token for token in re.findall(r"[a-z0-9]+", question.casefold())
            if token not in stop_words and not token.isdigit()
        }
        def matches_question(endpoint: str) -> bool:
            terms = set(re.findall(r"[a-z0-9]+", endpoint.casefold()))
            return bool(terms & question_terms)

        parsed = []
        bridge_entities = set()
        for item in triples:
            match = re.match(
                r"^\s*(.*?)\s*-\[([A-Z_]+)\]->\s*(.*?)\s*$",
                item["relationship"],
            )
            if not match:
                continue
            source, relation_type, target = match.groups()
            normalized_type = relation_type.replace("_", "")
            source_norm, target_norm = source.casefold(), target.casefold()
            source_question = matches_question(source)
            target_question = matches_question(target)
            parsed.append((item, source_norm, target_norm, normalized_type))
            if normalized_type not in {
                "PLEIOTROPICTO", "CONFERSRESISTANCETO", "CARRIES", "CARRIESGENE",
                "HASGENE", "LOCATEDON", "ASSOCIATEDWITH", "ASSOCIATEDWITHTRAIT",
            }:
                continue
            if source_question and not target_question:
                bridge_entities.add(target_norm)
            elif target_question and not source_question:
                bridge_entities.add(source_norm)

        for item, source, target, normalized_type in parsed:
            if normalized_type in {
                "PLEIOTROPICTO", "CONFERSRESISTANCETO", "CARRIES", "CARRIESGENE",
                "HASGENE", "LOCATEDON", "ASSOCIATEDWITH", "ASSOCIATEDWITHTRAIT",
            } and (source in bridge_entities or target in bridge_entities):
                item["relevance_score"] += 10.0

    # Stage: RETRIEVING_CONTEXT. Run all expanded literature searches
    # concurrently, merge them, and rank the candidates by RRF with the
    # lexical/vector tie-breaks. Judging which candidates reach the answer
    # prompt happens later, in `_judge_literature_candidates`.
    def _get_literature_context_concurrent(
        self, expanded_queries: list[str], k: int,
        taxon_filter: Optional[dict[str, Any]] = None,
        original_question: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        if not expanded_queries:
            return None
        literature_started = time.perf_counter()
        relevance_question = original_question or " ".join(expanded_queries)
        chunks_by_id: dict[Any, dict[str, Any]] = {}
        deduplicated: dict[str, dict[str, Any]] = {}
        searches = {}
        completion_order = []
        with ThreadPoolExecutor(max_workers=min(8, len(expanded_queries))) as executor:
            futures = {
                executor.submit(
                    self._search_literature_one_query,
                    query,
                    k,
                    relevance_question,
                    taxon_filter,
                    query_label=(
                        "original" if index == 0 and original_question
                        and " ".join(query.split()).casefold() == " ".join(original_question.split()).casefold()
                        else f"expanded {index}/{len(expanded_queries) - 1}"
                    ),
                ): query for index, query in enumerate(expanded_queries)
            }
            for future in as_completed(futures):
                query = futures[future]
                result = future.result()
                completion_order.append(query)
                searches[query] = result["search_trace"]
                for item in result["diagnostics"]["chunks"]:
                    chunk_id = item.get("chunk_id")
                    key = chunk_id or (item.get("source_path", ""), hash(item.get("text", "")))
                    current = chunks_by_id.get(key)
                    score = item.get("rrf_score") or 0.0
                    if current is None:
                        is_eligible = item.get("rejection_reason") is None
                        current = {
                            **item,
                            "included": False,
                            "rejection_reason": item.get("rejection_reason"),
                            "_eligible": is_eligible,
                            "_query_scores": {query: score} if is_eligible else {},
                            "_rejected_score": score if not is_eligible else 0.0,
                        }
                        current["rrf_score"] = score
                        chunks_by_id[key] = current
                    else:
                        current["retrieval_queries"].extend(item.get("retrieval_queries", []))
                        current["is_direct_search_match"] = bool(current.get("is_direct_search_match") or item.get("is_direct_search_match"))
                        if item.get("rejection_reason") is None:
                            current["_eligible"] = True
                            current["rejection_reason"] = None
                            current["_query_scores"][query] = max(
                                current["_query_scores"].get(query, 0.0), score
                            )
                            current["rrf_score"] = sum(current["_query_scores"].values())
                        else:
                            current["_rejected_score"] = max(
                                current["_rejected_score"], score
                            )
                            if not current["_eligible"]:
                                current["rrf_score"] = current["_rejected_score"]
                for item in result["diagnostics"]["triples"]:
                    match = re.match(r"^\[Source: (.*?)\]\s*(.*)$", item["text"])
                    sources = set(match.group(1).split("; ")) if match else set()
                    relationship = match.group(2) if match else item["text"]
                    key = " ".join(relationship.casefold().split())
                    current = deduplicated.get(key)
                    if current is None:
                        current = {
                            "relationship": relationship, "sources": set(),
                            "rrf_score": item["rrf_score"],
                            "relevance_score": item["relevance_score"],
                            "included": False, "rejection_reason": item["rejection_reason"],
                        }
                        deduplicated[key] = current
                    current["sources"].update(sources)
                    current["rrf_score"] = max(current["rrf_score"], item["rrf_score"])
                    current["relevance_score"] = max(
                        current["relevance_score"], item["relevance_score"]
                    )
                    if item["rejection_reason"] is None:
                        current["rejection_reason"] = None
                result["diagnostics"]["triples"] = []

        search_seconds = time.perf_counter() - literature_started
        logger.info(
            "Literature retrieval, %d %s in parallel: %.2fs wall",
            len(expanded_queries), "query" if len(expanded_queries) == 1 else "queries", search_seconds,
        )
        all_chunks = list(chunks_by_id.values())
        lexical_started = time.perf_counter()
        score_literature_candidate_texts(all_chunks, relevance_question, expanded_queries)
        lexical_seconds = time.perf_counter() - lexical_started
        vector_timings = {}
        vector_tie_break_error = None
        try:
            vector_scored_count = score_literature_candidate_vectors(
                all_chunks, relevance_question, self.graph, self.emb, vector_timings
            )
        except Exception as error:
            vector_scored_count = 0
            vector_tie_break_error = str(error)
            logger.warning("Literature candidate vector tie-break unavailable: %s", error)
        logger.debug(
            "Literature candidate scoring: lexical %.2fs, question embedding %.2fs, Neo4j cosine %.2fs (%d chunks scored)",
            lexical_seconds, vector_timings.get("question_embedding", 0.0),
            vector_timings.get("neo4j_cosine", 0.0), vector_scored_count,
        )
        ranking_started = time.perf_counter()
        rrf_ranked = sorted(
            all_chunks,
            key=lambda item: literature_candidate_sort_key(item, item.get("rrf_score")),
        )
        for rank, item in enumerate(rrf_ranked, start=1):
            item["rrf_rank"] = rank
            item["rrf_eligible_query_contributions"] = dict(item.get("_query_scores", {}))
            item["text_chars"] = len(item.get("text", ""))
            item["text_sha256"] = hashlib.sha256(item.get("text", "").encode("utf-8")).hexdigest()
            item["diagnostic_id"] = item.get("chunk_id") or item["text_sha256"]
            item["sent_to_relevance_ranker"] = False
        combined_rrf_groups = {}
        for item in rrf_ranked:
            combined_rrf_groups.setdefault(item.get("rrf_score") or 0.0, []).append(item.get("chunk_id"))
        ranking_seconds = time.perf_counter() - ranking_started
        rejection_counts = Counter(item.get("rejection_reason") for item in all_chunks)
        logger.debug(
            "Literature candidates: %d chunks (%d eligible, %d taxonomy-rejected, %d outside top-K), %d triples",
            len(all_chunks), rejection_counts[None],
            rejection_counts["Clearly identified as a different taxonomy"],
            rejection_counts["Outside RRF top-K"], len(deduplicated),
        )
        return {
            "relevance_question": relevance_question,
            "expanded_queries": expanded_queries,
            "chunks": all_chunks,
            "rrf_ranked": rrf_ranked,
            "triples": list(deduplicated.values()),
            "run_trace": {
                "question_used_for_relevance": relevance_question,
                "expanded_queries_in_input_order": expanded_queries,
                "query_completion_order": completion_order,
                "combined_rrf_ids_in_rank_order": [item.get("chunk_id") for item in rrf_ranked],
                "combined_rrf_ties": [
                    {"score": score, "chunk_ids_in_rank_order": ids}
                    for score, ids in combined_rrf_groups.items() if len(ids) > 1
                ],
                "searches": [searches[query] for query in expanded_queries],
                "rrf_aggregation_rule": "Sum per-query RRF only for eligible branches; otherwise retain maximum rejected-branch score. Expanded chunks inherit the maximum originating seed score per query.",
                "rrf_tie_rule": "Equal RRF scores use each chunk's vector similarity to the question, then question-term overlap, direct-search status, and chunk ID.",
                "vector_tie_break_scored_count": vector_scored_count,
                "vector_tie_break_error": vector_tie_break_error,
                "timings_seconds": {
                    "search_and_graph_expansion": search_seconds,
                    "lexical_scoring": lexical_seconds,
                    "question_embedding": vector_timings.get("question_embedding", 0.0),
                    "neo4j_cosine": vector_timings.get("neo4j_cosine", 0.0),
                    "candidate_ranking": ranking_seconds,
                    "total_retrieval": time.perf_counter() - literature_started,
                },
            },
        }

    # Stage: JUDGING_RELEVANCE. Pick the reranker pool from the ranked
    # literature candidates, judge it with the reranker LLM (falling back to
    # RRF order if that fails), fill the literature context budget, then add
    # the question-ranked relationships.
    def _judge_literature_candidates(
        self, candidates: dict[str, Any], max_context_chars: int, metadata_context: str,
    ) -> tuple[str, dict[str, Any]]:
        judging_started = time.perf_counter()
        relevance_question = candidates["relevance_question"]
        expanded_queries = candidates["expanded_queries"]
        all_chunks = candidates["chunks"]
        rrf_ranked = candidates["rrf_ranked"]
        pool_started = time.perf_counter()
        eligible_chunks = [item for item in all_chunks if item.get("rejection_reason") is None]
        rerank_pool: list[dict[str, Any]] = []
        pool_ids = set()
        for query in expanded_queries:
            query_ranked = sorted(
                (item for item in eligible_chunks if query in item.get("_query_scores", {})),
                key=lambda item: literature_candidate_sort_key(item, item["_query_scores"][query]),
            )
            for item in query_ranked[:RERANK_CANDIDATES_PER_QUERY]:
                identity = item.get("chunk_id") or id(item)
                if identity not in pool_ids and len(rerank_pool) < LITERATURE_CHUNKS_TO_RERANK:
                    pool_ids.add(identity)
                    rerank_pool.append(item)
        for item in rrf_ranked:
            if len(rerank_pool) >= LITERATURE_CHUNKS_TO_RERANK:
                break
            identity = item.get("chunk_id") or id(item)
            if item in eligible_chunks and identity not in pool_ids:
                pool_ids.add(identity)
                rerank_pool.append(item)
        reranker_error = None
        pool_ids_in_order = [item["diagnostic_id"] for item in rerank_pool]
        pool_seconds = time.perf_counter() - pool_started
        logger.debug("Literature candidate pool selection: %.2fs (%d chunks)", pool_seconds, len(rerank_pool))
        reranker_started = time.perf_counter()
        try:
            rerank_results = self._semantic_rerank_chunks(
                relevance_question, expanded_queries, metadata_context, rerank_pool
            )
            if len(rerank_results) != len(rerank_pool):
                raise ValueError("Semantic reranker returned an incomplete result set")
            for item in rerank_pool:
                result = rerank_results[str(item.get("chunk_id") or id(item))]
                item.update(result)
        except Exception as error:
            logger.warning("Literature semantic reranking failed; using RRF order: %s", error)
            reranker_error = str(error)
            rerank_pool = eligible_chunks
            for item in eligible_chunks:
                item["semantic_score"] = None
                item["evidence_entities"] = []
        reranker_seconds = time.perf_counter() - reranker_started

        for item in eligible_chunks:
            if item.get("semantic_score") is None and rerank_pool is not eligible_chunks:
                item["rejection_reason"] = "Outside semantic reranker candidate pool"
        semantic_available = bool(rerank_pool) and rerank_pool[0].get("semantic_score") is not None
        ranked_pool = sorted(
            rerank_pool,
            key=lambda item: (
                item.get("semantic_score") if semantic_available else 0.0,
                item.get("rrf_score") or 0.0,
                -len(item.get("text", "")),
            ),
            reverse=True,
        )
        scored_count = sum(item.get("semantic_score") is not None for item in ranked_pool)
        logger.debug(
            "Relevance judge: %.2fs (%d scored, fallback=%s)",
            reranker_seconds, scored_count, reranker_error,
        )
        for rank, item in enumerate(ranked_pool, start=1):
            item["rank"] = rank
            if semantic_available and item["semantic_score"] < RERANK_MIN_SCORE:
                item["rejection_reason"] = "Below semantic relevance threshold"

        selection_started = time.perf_counter()
        ranked_chunks = sorted(
            all_chunks,
            key=lambda item: (
                item.get("rank") is None,
                item.get("rank") or item.get("rrf_rank", 0),
            ),
        )
        grouped_chunks = {query: [] for query in expanded_queries}
        included_chars = 0
        nonempty_groups = 0
        selection_pool = [
            item for item in ranked_pool
            if item.get("rejection_reason") is None
        ]
        selected_entities = set()
        selection_rank = 0
        while selection_pool:
            options = []
            for item in selection_pool:
                query_scores = item.get("_query_scores", {})
                best_query = (
                    max(query_scores, key=query_scores.get)
                    if query_scores else expanded_queries[0]
                )
                starts_group = not grouped_chunks.setdefault(best_query, [])
                framing_chars = (
                    len(f"\nFor sub-question\n{best_query}\n\n### Context Chunks are:\n")
                    + (1 if nonempty_groups else 0)
                    if starts_group else len(os.linesep)
                )
                rendered_chunk = f"[Source: {item.get('source_path', '')}] {item.get('text', '')}\n"
                cost = len(rendered_chunk) + framing_chars
                entity_labels = {
                    " ".join(entity.split()).casefold(): " ".join(entity.split())
                    for entity in item.get("evidence_entities", [])
                }
                novel_entity_keys = set(entity_labels) - selected_entities
                novel_entities = [entity_labels[key] for key in novel_entity_keys]
                utility = (item.get("semantic_score") or 0.0) + min(len(novel_entities), 4) * 0.25
                options.append((utility / max(cost, 1), item, best_query, starts_group, rendered_chunk, novel_entities, novel_entity_keys, cost))
            utility_per_char, item, best_query, starts_group, rendered_chunk, novel_entities, novel_entity_keys, cost = max(
                options, key=lambda option: option[0]
            )
            selection_pool.remove(item)
            item["budget_decision"] = {
                "assigned_query": best_query, "rendered_cost_chars": cost,
                "context_chars_before_decision": included_chars, "context_limit_chars": max_context_chars,
                "utility_per_character": utility_per_char,
                "new_evidence_entities_at_decision": sorted(novel_entities),
            }
            item.pop("_eligible", None)
            item.pop("_rejected_score", None)
            if included_chars + cost > max_context_chars:
                item["rejection_reason"] = "Out of context size bound"
                continue
            item["included"] = True
            selection_rank += 1
            item["selection_rank"] = selection_rank
            item["new_evidence_entities"] = sorted(novel_entities)
            selected_entities.update(novel_entity_keys)
            included_chars += cost
            if starts_group:
                nonempty_groups += 1
            grouped_chunks[best_query].append(rendered_chunk)

        logger.info(
            "Literature selection: %d of %d %s chunks included (%s / %s chars)",
            selection_rank, len(ranked_pool),
            "judged" if semantic_available else "RRF-ranked (judge unavailable)",
            f"{included_chars:,}", f"{max_context_chars:,}",
        )
        logger.debug(
            "Literature rejections: %s",
            dict(Counter(
                item["rejection_reason"] for item in all_chunks if item.get("rejection_reason")
            )),
        )
        rejection_stages = {
            "Clearly identified as a different taxonomy": "taxonomy_filter",
            "Outside RRF top-K": "per_query_rrf_limit",
            "Outside semantic reranker candidate pool": "reranker_candidate_limit",
            "Below semantic relevance threshold": "semantic_score_threshold",
            "Out of context size bound": "final_context_budget",
        }
        for item in ranked_chunks:
            item["first_rejection_stage"] = rejection_stages.get(item.get("rejection_reason"))
            item["semantic_rank"] = item.get("rank")
        diagnostics = {
            "run_trace": {
                **candidates["run_trace"],
                "rerank_candidate_ids_in_input_order": pool_ids_in_order,
                "reranker_batches": getattr(self._chunk_reranker_usage, "batches", []),
                "metadata_scope_used_by_judge": metadata_context[:RERANK_MAX_METADATA_CHARS],
                "metadata_scope_truncated": len(metadata_context) > RERANK_MAX_METADATA_CHARS,
                "budget_tie_rule": "Equal utility-per-character choices retain ranked-pool order.",
            },
            "chunks": [
                {key: value for key, value in item.items() if not key.startswith("_")}
                for item in ranked_chunks
            ],
            "triples": [],
            "chunk_reranker": {
                "method": "LLM evidence relevance + marginal entity coverage per character",
                "model": RERANK_MODEL,
                "candidate_count": len(eligible_chunks),
                "scored_count": scored_count,
                "candidate_limit": LITERATURE_CHUNKS_TO_RERANK,
                "minimum_score": RERANK_MIN_SCORE,
                "token_usage": dict(self._chunk_reranker_usage.value),
                "fallback_error": reranker_error,
            },
        }

        parts = [
            f"\nFor sub-question\n{query}\n\n### Context Chunks are:\n"
            + os.linesep.join(grouped_chunks.get(query, []))
            for query in expanded_queries
            if grouped_chunks.get(query)
        ]

        triple_candidates = candidates["triples"]
        self._boost_connected_triple_relevance(relevance_question, triple_candidates)
        triple_candidates.sort(
            key=lambda item: (
                item["relevance_score"],
                item["rrf_score"],
                item["relationship"].casefold(),
            ),
            reverse=True,
        )
        selected_triples, triple_chars = [], 0
        relationship_header = "### Entity Relationships (deduplicated and question-ranked):\n"
        for rank, item in enumerate(triple_candidates, start=1):
            item["rank"] = rank
            item["text"] = (
                f"[Source: {'; '.join(sorted(item['sources']))}] "
                f"{item['relationship']}"
            )
            if item["rejection_reason"]:
                continue
            if len(selected_triples) >= MAX_TRIPLES:
                item["rejection_reason"] = "Outside relationship count limit"
            elif triple_chars + len(item["text"]) > LITERATURE_TRIPLE_MAX_CHARACTERS:
                item["rejection_reason"] = "Out of relationship context size bound"
            else:
                item["included"] = True
                selected_triples.append(item["text"])
                triple_chars += len(item["text"])
        logger.debug(
            "Relationships: %d of %d included, %s chars",
            len(selected_triples), len(triple_candidates), f"{triple_chars:,}",
        )
        diagnostics["triples"] = [
            {key: item[key] for key in (
                "text", "rrf_score", "relevance_score", "rank", "included", "rejection_reason"
            )}
            for item in triple_candidates
        ]
        if selected_triples:
            parts.append(relationship_header + "\n".join(selected_triples))
        context = "\n".join(parts)
        diagnostics["run_trace"]["timings_seconds"] = {
            **candidates["run_trace"]["timings_seconds"],
            "candidate_pool": pool_seconds,
            "relevance_judge": reranker_seconds,
            "selection_and_assembly": time.perf_counter() - selection_started,
            "total_judging": time.perf_counter() - judging_started,
        }
        return context, diagnostics

    # 2. Full-text indexing
    def _fulltext_chunks(
        self, q: str, k: int = QUERY_FULL_TEXT_MAX_CHUNKS
    ) -> Dict[str, float]:
        cleaned_q = self.escape_lucene_plain_text(q)
        if not cleaned_q:
            return {}
        res = self.graph.query(
            """
            CALL db.index.fulltext.queryNodes('idx_chunk_text', $q) YIELD node, score
            RETURN node.chunk_id AS cid, score ORDER BY score DESC LIMIT $k
        """,
            params={"q": cleaned_q, "k": k},
        )
        return {r["cid"]: r["score"] for r in res if r.get("cid")}

    # Use Reciprocal Rank Fusion (RRF) instead of min-max normalized weights
    def _rrf_fusion(
        self,
        vector_scores: Dict[str, float],
        ft_scores: Dict[str, float],
        k_penalty=RRF_RANK_CONSTANT,
    ) -> Dict[str, float]:
        rrf_scores = {}
        for rankings in [vector_scores, ft_scores]:
            # Sort by score descending to get rank
            sorted_items = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
            for rank, (cid, _) in enumerate(sorted_items):
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (
                    1.0 / (k_penalty + rank + 1)
                )
        return rrf_scores

    # Deduplicate chunks before reranking and prompt assembly.
    def _dedupe_chunks(self, chunks: List[dict]) -> List[dict]:
        seen = set()
        deduped = []
        for c in chunks:
            key = c.get("chunk_id") or (
                c.get("source_path", ""),
                c.get("text", "")[:200],
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(c)
        return deduped

    # Helper for embedding rerank
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
        norm = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(np.dot(va, vb) / norm) if norm else 0.0

    # Extracts plain text from an LLM response/chunk. `.content` is usually
    # a plain string, but some providers (e.g. Gemini) can return a list of
    # content blocks instead, so that case is flattened here too.
    @staticmethod
    def _message_text(resp: Any) -> str:
        text = getattr(resp, "text", None)
        if isinstance(text, str) and text:
            return text
        content = getattr(resp, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            pieces: List[str] = []
            for part in content:
                if isinstance(part, str):
                    pieces.append(part)
                elif isinstance(part, dict) and part.get("type") in (None, "text"):
                    text_part = part.get("text") or ""
                    if text_part:
                        pieces.append(text_part)
            return "".join(pieces)
        return str(content) if content else str(resp)

    def _llm_invoke(self, prompt: Any, diagnostics: Optional[dict] = None) -> str:
        resp = self.llm.invoke(prompt)
        if diagnostics is not None:
            diagnostics["usage_metadata"] = getattr(resp, "usage_metadata", None) or {}
        return self._message_text(resp).strip()

    # Extracts the model's thinking/reasoning-trace text from a response or
    # streamed chunk. Only populated when `include_thoughts=True` is passed
    # to the call (see `_generate_answer_stream`); other LLM calls in this
    # class don't request thoughts, so this returns "" for them.
    #
    # LangChain's Google GenAI adapter stores thoughts as v0
    # `{type: "thinking", thinking: ...}` blocks on `.content`, and as v1
    # `{type: "reasoning", reasoning: ...}` blocks on `.content_blocks`.
    # `ChatOpenAI` (with `output_version="responses/v1"`, set in
    # `_get_answer_llm`) instead puts GPT's reasoning summary in a
    # `{type: "reasoning", summary: [{type: "summary_text", text: ...}]}`
    # block directly on `.content` - handled by the `summary` branch below.
    @staticmethod
    def _thinking_from_parts(parts: Any) -> str:
        if not isinstance(parts, list):
            return ""
        pieces: List[str] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            kind = part.get("type")
            if kind == "thinking":
                text = part.get("thinking") or part.get("text") or ""
                if isinstance(text, str) and text:
                    pieces.append(text)
            elif kind == "reasoning":
                text = part.get("reasoning") or part.get("text") or ""
                if isinstance(text, str) and text:
                    pieces.append(text)
                for summary_part in part.get("summary") or []:
                    if not isinstance(summary_part, dict):
                        continue
                    summary_text = summary_part.get("text") or ""
                    if isinstance(summary_text, str) and summary_text:
                        pieces.append(summary_text)
        return "".join(pieces)

    @staticmethod
    def _message_thinking(resp: Any) -> str:
        thinking = PlantBioRAG._thinking_from_parts(getattr(resp, "content", None))
        if thinking:
            return thinking
        return PlantBioRAG._thinking_from_parts(getattr(resp, "content_blocks", None))

    # Question analysis and retrieval query expansion
    def expand_question_and_queries(
        self, q: str, diagnostics: Optional[dict] = None,
    ) -> tuple[str, list[str], bool, bool, list[str], str, str]:
        prompt = (
            getPrompt("expand_question_and_queries")
            + f"""
        @@@@
        {q}
        @@@@
        """
        )
        if diagnostics is not None:
            diagnostics.update({"model": GEMINI_MODEL, "temperature": 0,
                                "prompt": prompt, "prompt_chars": len(prompt),
                                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()})
        logger.debug("Question analysis prompt: %s chars", f"{len(prompt):,}")
        resp = self._llm_invoke(prompt, diagnostics)
        if diagnostics is not None:
            diagnostics["raw_response"] = resp
        logger.debug("Question analysis raw response: %s", resp)
        clean_json = resp.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)
        if diagnostics is not None:
            diagnostics["parsed_response"] = data
        expanded_question = str(data.get("expanded_question", q))
        expanded_queries = data.get("expanded_queries", [])
        expanded_question = (
            expanded_question.replace("/", " ").replace(":", " ").replace("\n", " ")
        )
        expanded_queries = [
            str(s).replace("/", " ").replace(":", " ").replace("\n", " ")
            for s in expanded_queries
            if s
        ]
        llm_query_count = len(expanded_queries)
        # Always include original user question first for exact-match retrieval
        expanded_queries.insert(0, q)
        # De-duplicate while preserving order (case/whitespace/trailing-punctuation-insensitive)
        seen_query_keys = set()
        deduplicated_queries = []
        for s in expanded_queries:
            key = " ".join(s.split()).casefold().rstrip("?.! ")
            if key not in seen_query_keys:
                seen_query_keys.add(key)
                deduplicated_queries.append(s)
        expanded_queries = deduplicated_queries
        logger.debug(
            "Query expansion: %d from LLM, %d removed as duplicates of earlier queries",
            llm_query_count, llm_query_count + 1 - len(expanded_queries),
        )
        is_agg_accession_query = bool(data.get("is_agg_accession_query", False))
        is_direct_agg_lookup = bool(data.get("is_direct_agg_lookup", False))
        raw_direct_accessions = data.get("direct_agg_accessions", [])
        direct_agg_accessions = (
            [
                " ".join(name.split())
                for name in raw_direct_accessions
                if isinstance(name, str) and name.strip()
            ]
            if isinstance(raw_direct_accessions, list)
            else []
        )
        direct_agg_accessions = list(dict.fromkeys(direct_agg_accessions))
        # Never take the direct route unless all three signals agree.
        if is_direct_agg_lookup and not (is_agg_accession_query and direct_agg_accessions):
            logger.debug(
                "Direct AGG lookup requested but not used: %s",
                " and ".join(
                    reason for reason, missing in (
                        ("not classified as an AGG query", not is_agg_accession_query),
                        ("no accessions extracted", not direct_agg_accessions),
                    ) if missing
                ),
            )
        is_direct_agg_lookup = bool(
            is_agg_accession_query and is_direct_agg_lookup and direct_agg_accessions
        )
        accession_question = str(data.get("accession_question", "")).strip()
        species = str(data.get("species", "")).strip()
        return (
            expanded_question,
            expanded_queries,
            is_agg_accession_query,
            is_direct_agg_lookup,
            direct_agg_accessions,
            accession_question,
            species,
        )

    def _extract_accessions(
        self, question: str, answer: str, species: str,
        diagnostics: Optional[dict] = None,
    ) -> List[str]:
        """Extract only relevant accessions in one LLM call."""
        payload = json.dumps(
            {"question": question, "answer": answer, "species": species},
            ensure_ascii=False,
        )
        prompt = (
            getPrompt("extract_accessions")
            + """
Input JSON:
"""
            + payload
        )
        raw = self._llm_invoke(prompt, diagnostics).strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        try:
            result = json.loads(raw)
            if not isinstance(result, list) or any(
                not isinstance(name, str) for name in result
            ):
                raise ValueError("Expected a JSON array of accession names")
            names = []
            for raw_name in result:
                name = " ".join(raw_name.split())
                if not name:
                    continue
                if name not in answer:
                    logger.warning(
                        "Ignoring extracted accession not present in answer: %s", name
                    )
                    continue
                names.append(self._normalise_accession_name(name, species))
            return list(dict.fromkeys(names))
        except (ValueError, TypeError):
            logger.warning("Invalid relevant-accession extraction; skipping AGG lookup")
            return []

    @staticmethod
    def _normalise_accession_name(name: str, species: str) -> str:
        """Format AGG IDs without letting the model invent accession identifiers."""
        match = re.fullmatch(
            r"AGG\s*(\d+)(?:\s*(BARL|WHEA|CHIC|PEAS|LENS|LUPN))?",
            name,
            flags=re.IGNORECASE,
        )
        if not match:
            return name
        suffixes = {
            "wheat": "WHEA",
            "barley": "BARL",
            "chickpea": "CHIC",
            "chick pea": "CHIC",
            "field pea": "PEAS",
            "lentil": "LENS",
            "lupin": "LUPN",
        }
        suffix = (
            match.group(2) or suffixes.get(species.strip().casefold(), "")
        ).upper()
        return f"AGG {match.group(1)} {suffix}".strip()

    # Call the accession API with extracted accession names
    def _call_accession_api(
        self, question: str, accessions: List[str]
    ) -> Optional[dict]:
        if not ACCESSION_API_URL:
            logger.error(
                "Accession API is not configured. Set ACCESSION_API_URL in the root .env file."
            )
            return None
        payload = {
            "token": ACCESSION_API_TOKEN,
            "question": question,
            "accessions": accessions,
        }
        try:
            response = requests.post(
                ACCESSION_API_URL,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=ACCESSION_API_TIMEOUT,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logger.error("Accession API Error: The request timed out.")
        except requests.exceptions.ConnectionError:
            logger.error("Accession API Error: Failed to connect to the server.")
        except requests.exceptions.HTTPError as err:
            logger.error("Accession API HTTP Error: %s", err)
        except Exception as e:
            logger.exception("Accession API unexpected error: %s", e)
        return None

    # Use LLM to present accession API response clearly to the user
    def _present_accession_results(
        self, original_question: str, api_response: dict,
        diagnostics: Optional[dict] = None,
    ) -> str:
        prompt = (
            getPrompt("present_accession_results")
            + f"""


        A user asked: "{original_question}". 

        The AGG accession API returned the following results:
        {api_response}
        """
        )
        resp = self._llm_invoke(prompt, diagnostics)
        return resp

    # 2. Metadata Graph RAG
    def _filter_metadata_scores(
        self,
        scores: Dict[str, float],
        k: int,
        taxon_filter: Optional[dict[str, Any]],
    ) -> Dict[str, float]:
        """Keep the highest-ranked matching metadata nodes, preserving order."""
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if not taxon_filter:
            return dict(ranked[:k])
        candidate_ids = [nid for nid, _ in ranked]
        rows = self.graph.query(
            "UNWIND $nids AS nid MATCH (n:MetadataGraph) "
            "WHERE elementId(n) = nid "
            "RETURN nid, n.crop AS crop, n.species AS species",
            params={"nids": candidate_ids},
        )
        properties_by_id = {
            row["nid"]: {"crop": row.get("crop"), "species": row.get("species")}
            for row in rows
        }
        filtered = [
            (nid, score)
            for nid, score in ranked
            if nid in properties_by_id
            and metadata_matches_taxon(
                properties_by_id[nid], taxon_filter, allow_unclassified=True
            )
        ]
        return dict(filtered[:k])

    def _vector_chunks_metadata(
        self,
        q: str,
        k: int = METADATA_VECTOR_K,
        taxon_filter: Optional[dict[str, Any]] = None,
    ) -> Dict[str, float]:
        # Vector search in metadata_graph via metadata_vector_index using Gemini embeddings.
        q_emb = self.emb.embed_query(q)
        search_k = k * METADATA_FILTER_OVERFETCH_MULTIPLIER if taxon_filter else k
        res = self.graph.query(
            "CYPHER 25 MATCH (n:MetadataGraph) "
            "SEARCH n IN (VECTOR INDEX metadata_vector_index FOR $emb LIMIT $k) SCORE AS score "
            "RETURN elementId(n) AS nid, score ORDER BY score DESC",
            params={"k": search_k, "emb": q_emb},
        )
        scores = {r["nid"]: r["score"] for r in res}
        return self._filter_metadata_scores(scores, k, taxon_filter)

    def _fulltext_chunks_metadata(
        self,
        q: str,
        k: int = METADATA_FULLTEXT_K,
        taxon_filter: Optional[dict[str, Any]] = None,
    ) -> Dict[str, float]:
        # Fulltext search in metadata_graph via metadata_fulltext_index.
        cleaned_q = self.escape_lucene_plain_text(q)
        if not cleaned_q:
            return {}
        res = self.graph.query(
            "CALL db.index.fulltext.queryNodes('metadata_fulltext_index', $q) YIELD node, score "
            "RETURN elementId(node) AS nid, score ORDER BY score DESC LIMIT $k",
            params={
                "q": cleaned_q,
                "k": k * METADATA_FILTER_OVERFETCH_MULTIPLIER if taxon_filter else k,
            },
        )
        scores = {r["nid"]: r["score"] for r in res}
        return self._filter_metadata_scores(scores, k, taxon_filter)

    def _expand_one_hop(
        self,
        nids: List[str],
        taxon_filter: Optional[dict[str, Any]] = None,
        neighbor_limit_per_seed: int = METADATA_NEIGHBORS_PER_SEED,
    ):
        # Expand MetadataGraph seed nodes by one hop in either direction.
        # Returns nodes with all properties plus chunk_id for dedupe/fetch.
        taxon_clause = ""
        params = {"nids": nids, "neighbor_limit": neighbor_limit_per_seed}
        if taxon_filter:
            params["taxon_regex"] = taxon_regex(taxon_filter)
            # Unclassified neighbors stay eligible; explicitly classified
            # neighbors must match the question's crop/species.
            taxon_clause = """
            WHERE n2 IS NULL
               OR (n2.crop IS NULL AND n2.species IS NULL)
               OR coalesce(toString(n2.crop), '') =~ $taxon_regex
               OR coalesce(toString(n2.species), '') =~ $taxon_regex
            """
        query = f"""
        MATCH (n1:MetadataGraph) WHERE elementId(n1) IN $nids
        CALL {{
            WITH n1
            OPTIONAL MATCH (n1)-[r]-(n2)
            {taxon_clause}
            WITH n1, r, n2 ORDER BY elementId(n2)
            LIMIT $neighbor_limit
            RETURN
                collect(DISTINCT CASE WHEN n2 IS NULL THEN NULL ELSE n2 {{
                    .*, chunk_id: elementId(n2),
                    source_path: 'Metadata Graph', labels: labels(n2)
                }} END) AS neighbors,
                collect(DISTINCT CASE WHEN r IS NULL OR n2 IS NULL THEN NULL ELSE
                    '[Source: Metadata Graph] ' +
                    coalesce(n1.displayName, n1.shortName, n1.id, n1.projectName,
                            n1.accessionName, n1.curatorName, elementId(n1)) +
                    ' -[' + type(r) + ']- ' +
                    coalesce(n2.displayName, n2.shortName, n2.id, n2.projectName,
                            n2.accessionName, n2.curatorName, elementId(n2))
                END) AS triples
        }}
        RETURN
            collect(DISTINCT n1 {{
                .*, chunk_id: elementId(n1),
                source_path: 'Metadata Graph', labels: labels(n1)
            }}) AS seedchunks,
            reduce(acc = [], items IN collect(neighbors) | acc + items) AS expandedchunks,
            reduce(acc = [], items IN collect(triples) | acc + items) AS triples
        """
        res = self.graph.query(query, params=params)
        if not res or not res[0]["seedchunks"]:
            return [], [], []
        seedchunks = [c for c in res[0]["seedchunks"] if c and c.get("chunk_id")]
        expandedchunks = [
            c for c in res[0]["expandedchunks"] if c and c.get("chunk_id")
        ]
        triples = [t for t in res[0]["triples"] if t is not None]
        return seedchunks, expandedchunks, triples

    def _search_metadata_hybrid(
        self,
        expanded_queries: list[str],
        max_chars: int = METADATA_MAX_CONTEXT_CHARS,
        taxon_filter: Optional[dict[str, Any]] = None,
    ) -> str:
        # Hybrid search for metadata_graph.
        # Run expanded-query metadata hybrid searches concurrently.
        # Each query also runs vector + full-text concurrently.
        vector_fn = lambda query, _k: self._vector_chunks_metadata(
            query, METADATA_VECTOR_K, taxon_filter
        )
        fulltext_fn = lambda query, _k: self._fulltext_chunks_metadata(
            query, METADATA_FULLTEXT_K, taxon_filter
        )
        all_fused = self._multi_query_hybrid_scores_concurrent(
            expanded_queries,
            vector_fn,
            fulltext_fn,
            max(METADATA_VECTOR_K, METADATA_FULLTEXT_K),
            RRF_RANK_CONSTANT,
        )
        ranked = sorted(all_fused.items(), key=lambda item: item[1], reverse=True)
        # Search deeper than the final result cap so an elbow below the first
        # few ranks can be detected. The result cap is applied afterwards.
        scan_count = min(METADATA_RRF_SCAN_LIMIT, len(ranked))
        elbow_seed_count = scan_count
        best_drop = 0.0
        min_seeds = min(METADATA_RRF_MIN_SEEDS, scan_count)
        for index in range(max(0, min_seeds - 1), scan_count - 1):
            current_score = ranked[index][1]
            next_score = ranked[index + 1][1]
            absolute_drop = current_score - next_score
            relative_drop = absolute_drop / current_score if current_score else 0.0
            if (
                absolute_drop >= METADATA_RRF_MIN_ABSOLUTE_DROP
                and relative_drop >= METADATA_RRF_MIN_RELATIVE_DROP
                and relative_drop > best_drop
            ):
                best_drop = relative_drop
                elbow_seed_count = index + 1
        selected_count = min(elbow_seed_count, METADATA_MAX_RESULTS_AFTER_RRF)
        top_nids = [nid for nid, _ in ranked[:selected_count]]
        if not top_nids:
            return ""
        seeded_chunks, expanded_chunks, triples = self._expand_one_hop(
            top_nids, taxon_filter, METADATA_NEIGHBORS_PER_SEED
        )
        all_chunks_deduplicated = self._dedupe_chunks(seeded_chunks + expanded_chunks)
        deduped_nids = [
            c.get("chunk_id") for c in all_chunks_deduplicated if c.get("chunk_id")
        ]
        fetch_nids = deduped_nids or top_nids
        res = self.graph.query(
            """
            UNWIND range(0, size($nids) - 1) AS i
            WITH $nids[i] AS nid, i
            MATCH (n:MetadataGraph)
            WHERE elementId(n) = nid
            RETURN i, n {.*} AS props
            ORDER BY i
            """,
            params={"nids": fetch_nids},
        )
        # Limit by max_chars.
        context_parts, total_chars = [], 0
        for t in triples[:METADATA_MAX_TRIPLES]:
            text = json.dumps({"relationship": t}, ensure_ascii=False)
            if total_chars + len(text) > max_chars:
                return "\n".join(context_parts)
            context_parts.append(text)
            total_chars += len(text)
        for r in res:
            props = {
                k: v
                for k, v in r["props"].items()
                if k != "embedding" and v is not None
            }
            text = json.dumps(props)
            if total_chars + len(text) > max_chars:
                break
            context_parts.append(text)
            total_chars += len(text)
        return "\n".join(context_parts)

    def _get_metadata_context(
        self,
        query: str,
        expanded_queries: list[str],
        taxon_filter: Optional[dict[str, Any]] = None,
    ) -> str:
        started = time.perf_counter()
        context_parts = []
        result = self._search_metadata_hybrid(
            expanded_queries, METADATA_MAX_CONTEXT_CHARS, taxon_filter
        )
        if result:
            context_parts.append(f"### Metadata Graph (Hybrid Search):\n{result}")
        context = "\n\n".join(context_parts)
        logger.info(
            "Metadata context: %.2fs, %s chars", time.perf_counter() - started, f"{len(context):,}"
        )
        return context

    def _vector_chunks_pretzel(
        self, q: str, k: int = QUERY_VECTOR_MAX_CHUNKS
    ) -> Dict[str, float]:
        # Vector search in pretzel_graph using Gemini embeddings.
        q_emb = self.emb.embed_query(q)
        res = self.graph.query(
            "CYPHER 25 MATCH (n:PretzelFunction) "
            "SEARCH n IN (VECTOR INDEX pretzel_functions_vector FOR $emb LIMIT $k) SCORE AS score "
            "RETURN elementId(n) AS nid, score",
            params={"k": k, "emb": q_emb},
        )
        return {r["nid"]: r["score"] for r in res}

    def _fulltext_chunks_pretzel(
        self, q: str, k: int = QUERY_FULL_TEXT_MAX_CHUNKS
    ) -> Dict[str, float]:
        # Fulltext search in pretzel_graph.
        cleaned_q = self.escape_lucene_plain_text(q)
        if not cleaned_q:
            return {}
        res = self.graph.query(
            "CALL db.index.fulltext.queryNodes('idx_pretzel_function_text', $q) YIELD node, score "
            "RETURN elementId(node) AS nid, score ORDER BY score DESC LIMIT $k",
            params={"q": cleaned_q, "k": k},
        )
        return {r["nid"]: r["score"] for r in res}

    def _get_pretzel_context(
        self, expanded_queries: list[str], max_chars: int = METADATA_MAX_CHARACTERS
    ) -> str:
        # Hybrid search for pretzel_graph.
        # Run expanded-query Pretzel hybrid searches concurrently
        # Each query also runs vector + full-text concurrently
        started = time.perf_counter()
        all_fused = self._multi_query_hybrid_scores_concurrent(
            expanded_queries,
            self._vector_chunks_pretzel,
            self._fulltext_chunks_pretzel,
            QUERY_VECTOR_MAX_CHUNKS,
        )
        # Sort by fused score descending
        top_nids = sorted(all_fused, key=lambda x: all_fused[x], reverse=True)[:50]
        if not top_nids:
            logger.info("Pretzel context: %.2fs, 0 chars", time.perf_counter() - started)
            return ""
        # Fetch full node properties
        res = self.graph.query(
            """UNWIND $nids AS nid
            MATCH (p:PretzelFunction) WHERE elementId(p) = nid 
            RETURN p {.*} AS props""",
            params={"nids": top_nids},
        )
        # Limit by total_characters = 10000
        context_parts, total_chars = [], 0
        exclude_keys = {"embedding", "id", "chunk_id"}
        for r in res:
            props = {
                k: v
                for k, v in r["props"].items()
                if k not in exclude_keys and v is not None
            }
            text = json.dumps(props)
            if total_chars + len(text) > max_chars:
                break
            context_parts.append(text)
            total_chars += len(text)
        context = "\n".join(context_parts)
        logger.info(
            "Pretzel context: %.2fs, %s chars", time.perf_counter() - started, f"{len(context):,}"
        )
        return context

    def _semantic_cache_lookup(self, q_emb) -> Optional[tuple[str, str, dict]]:
        res = self.graph.query(
            """
        CALL db.index.vector.queryNodes($index, $k, $emb)
        YIELD node, score
        WHERE score >= $threshold
          AND node.created_at IS NOT NULL
          AND node.created_at >= datetime() - duration({days: $ttl_days})
        RETURN node.question AS question,
               node.expanded_question AS expanded_question,
               node.answer AS answer,
               node.usage_metadata AS usage_metadata,
               score
        ORDER BY score DESC
        LIMIT 1
        """,
            params={
                "index": SEMANTIC_CACHE_INDEX,
                "k": SEMANTIC_CACHE_TOP_K,
                "emb": q_emb,
                "threshold": SEMANTIC_CACHE_THRESHOLD,
                "ttl_days": SEMANTIC_CACHE_TTL_DAYS,
            },
        )
        if not res:
            return None
        r = res[0]
        logger.info(
            "Semantic cache hit: score=%s question=%s", r["score"], r["question"]
        )
        usage = {}
        try:
            usage = json.loads(r.get("usage_metadata") or "{}")
        except Exception:
            pass
        return r["expanded_question"], r["answer"], usage

    def _semantic_cache_store(
        self, q: str, expanded_question: str, answer: str, usage_metadata: dict, q_emb
    ) -> None:
        self.graph.query(
            """
        CREATE (c:SemanticCache {
            question: $question,
            expanded_question: $expanded_question,
            answer: $answer,
            usage_metadata: $usage_metadata,
            embedding: $embedding,
            created_at: datetime()
        })
        """,
            params={
                "question": q,
                "expanded_question": expanded_question,
                "answer": answer,
                "usage_metadata": json.dumps(usage_metadata),
                "embedding": q_emb,
            },
        )

    # --- Step helpers for query() -------------------------------------
    # Each helper below does the "real work" for one stage of the pipeline
    # and either returns plain data or raises. `query()` itself stays
    # responsible for all `yield`ing (stage changes / text / result), and
    # decides - per stage - whether a failure here should be fatal (abort
    # the whole run) or recoverable (degrade gracefully and keep going).

    # Stage: RETRIEVING_CONTEXT. Non-fatal: each of the three sources is
    # caught independently, so e.g. a Neo4j hiccup on the metadata graph
    # doesn't prevent literature context (or the whole answer) from coming
    # back. A failed metadata/Pretzel source contributes an empty string; a
    # failed literature source returns no candidates (None), so the
    # JUDGING_RELEVANCE stage is skipped. Failures are recorded by label in
    # the returned source-errors dict.
    def _retrieve_context(
        self,
        q: str,
        expanded_queries: List[str],
        k: int,
        taxon_filter: Optional[dict[str, Any]] = None,
    ) -> Tuple[Optional[dict[str, Any]], str, str, dict[str, Any]]:
        source_errors = {}
        def safe_call(fn, label, *args):
            try:
                return fn(*args)
            except Exception as e:
                source_errors[label] = {"type": type(e).__name__, "message": str(e)}
                logger.warning("%s context retrieval failed: %s", label, e)
                return ""

        with ThreadPoolExecutor(max_workers=3) as executor:
            metadata_future = executor.submit(
                safe_call,
                self._get_metadata_context,
                "Metadata",
                q,
                expanded_queries,
                taxon_filter,
            )
            literature_future = executor.submit(
                safe_call,
                self._get_literature_context_concurrent,
                "Literature",
                expanded_queries,
                k,
                taxon_filter,
                q,
            )
            pretzel_future = None
            if "pretzel" in q.lower():
                pretzel_future = executor.submit(
                    safe_call, self._get_pretzel_context, "Pretzel", expanded_queries
                )
            literature_candidates = literature_future.result() or None
            metadata_context = metadata_future.result()
            pretzel_context = pretzel_future.result() if pretzel_future else ""
        return literature_candidates, metadata_context, pretzel_context, source_errors

    # Pure string assembly - no I/O, so nothing to catch here.
    def _build_answer_prompt(
        self,
        expanded_question: str,
        q: str,
        literature_context: str,
        metadata_context: str,
        pretzel_context: str,
    ) -> str:
        prompt = (
            global_instruction_and_information
            + getPrompt("build_answer_prompt")
            + f"""
        """
        )
        if literature_context:
            prompt += literature_context
        if metadata_context:
            prompt += f"""\n\n\n
        ### [Source: Metadata Graph]:
        {metadata_context}"""
        if pretzel_context:
            prompt += f"""\n\n\n
        ### [Source: Pretzel Documentation]:
        {pretzel_context}"""
        prompt += f"""



            Analysis of user question: 
            {expanded_question}

            User Question:
            @@@@
            {q}
            @@@@
            Answer:"""
        return prompt

    # Prompt used on a semantic-cache hit: instead of re-running retrieval,
    # ask the LLM to answer the new question using only the cached answer
    # to a very similar prior question as context.
    def _build_cached_answer_prompt(self, q: str, cached_answer: str) -> str:
        return (
            global_instruction_and_information
            + getPrompt("build_cached_answer_prompt")
            + f"""

            User Question:
            @@@@
            {q}
            @@@@

            Previous answer to a similar question: 
            ####
            {cached_answer}
            ####

            Answer:"""
        )

    # Returns (building and caching, if necessary) the answer-generation
    # client for one (model_name, reasoning_level) combo. Called with
    # already-validated values from `_resolve_model_name`/
    # `_resolve_reasoning_level`, so every distinct combo a user actually
    # selects in the frontend gets its own client, reused across requests.
    #
    # Dispatches on `_model_provider`: GPT models go through `ChatOpenAI`
    # with `reasoning_effort` set directly to `reasoning_level` (OpenAI
    # uses the same "minimal"/"low"/"medium"/"high" vocabulary as
    # `AVAILABLE_REASONING_LEVELS`), plus `output_version="responses/v1"`
    # so the reasoning summary shows up as a `{"type": "reasoning", ...}`
    # content block for `_message_thinking` to read - the OpenAI analogue
    # of Gemini's `include_thoughts=True`. Reasoning-model temperature
    # constraints are handled by `ChatOpenAI` itself (it silently drops an
    # unsupported `temperature`), so none is passed here.
    def _get_answer_llm(
        self, model_name: str, reasoning_level: str
    ) -> Union[ChatGoogleGenerativeAI, ChatOpenAI]:
        key = (model_name, reasoning_level)
        llm = self._answer_llm_cache.get(key)
        if llm is None:
            if _model_provider(model_name) == "openai":
                llm = ChatOpenAI(
                    model=model_name,
                    reasoning_effort=reasoning_level,
                    output_version="responses/v1",
                )
            else:
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=0,
                    include_thoughts=True,
                    **_thinking_kwargs(model_name, reasoning_level),
                )
            self._answer_llm_cache[key] = llm
        return llm

    # Stage: GENERATING_ANSWER. Fatal by design: with no answer, there is
    # nothing useful left to yield, so `query()` lets this propagate up to
    # its outer `except` and end the run with an `ErrorEvent`. Streams the
    # response via `llm.astream` so `query()` can yield text chunks as they
    # arrive instead of blocking for the full answer.
    async def _generate_answer_stream(
        self, prompt: Any, model_name: str, reasoning_level: str
    ) -> AsyncGenerator[Any, None]:
        llm = self._get_answer_llm(model_name, reasoning_level)
        # Only the Gemini branch needs its thinking kwargs repeated on
        # every call - `thinking_level`/`thinking_budget` can be
        # overridden per `astream` call, unlike GPT's `reasoning_effort`,
        # which `_get_answer_llm` already bakes in at construction time.
        call_kwargs: Dict[str, Any] = {}
        if _model_provider(model_name) == "google":
            call_kwargs = {
                "include_thoughts": True,
                **_thinking_kwargs(model_name, reasoning_level),
            }
        async for chunk in llm.astream(prompt, **call_kwargs):
            yield chunk

    # Stages: CHECKING_AGG_ACCESSIONS. Extraction can itself fail (it calls
    # the LLM); `query()` catches that at the call site and treats it the
    # same as "no accessions found", since the main answer has already been
    # produced and shouldn't be thrown away over an optional side lookup.
    # `_call_accession_api` already degrades to `None` internally on
    # network/HTTP errors, so it's safe to call directly here.
    def _lookup_agg_accessions(
        self, q: str, answer: str, species: str, accession_question: str,
        token_tally: Optional[list] = None,
    ) -> Tuple[List[str], Optional[dict]]:
        logger.info(
            "[AGG Accession Query Detected] Extracting accessions from RAG answer..."
        )
        extraction_diagnostics: dict = {}
        with log_step("Extract relevant accessions", 6, token_tally) as step:
            accessions = self._extract_accessions(q, answer, species, extraction_diagnostics)
            step["model"] = GEMINI_MODEL
            step["usage"] = extraction_diagnostics.get("usage_metadata")
        logger.info("[Relevant Accessions]: %s", accessions)

        api_response = None
        if accessions:
            with log_step("Call accession API", 7):
                api_response = self._call_accession_api(accession_question, accessions)
            logger.info(f"api_response: {api_response}")
        return accessions, api_response

    # Main query
    # Single-turn only: `q` is the latest user message. No prior conversation
    # history is accepted or used to drive expansion/retrieval/generation.
    #
    # Async generator: yields protocol-agnostic internal events (stage
    # changes, text output, final result/error) as the run progresses,
    # instead of computing everything and returning once. Preserves all
    # existing retrieval/generation logic unchanged; only the control flow
    # differs. The final-answer LLM call is streamed via `llm.astream`, and
    # the remaining blocking calls (question/query expansion, concurrent
    # retrieval, accession lookup/presentation - all still synchronous
    # under the hood) are offloaded via `asyncio.to_thread(...)` so they
    # don't block the event loop.
    #
    # Error handling policy, by stage:
    # - EXPANDING_QUESTION: recoverable. Falls back to the raw question
    #   and "not an accession query" so the run can still proceed.
    # - RETRIEVING_CONTEXT: recoverable per-source (see _retrieve_context).
    # - GENERATING_ANSWER: fatal. No answer means nothing left to give the
    #   caller, so this is allowed to propagate to the outer except below.
    # - CHECKING_AGG_ACCESSIONS / PRESENTING_ACCESSIONS: recoverable. These
    #   run only after the main answer has already been yielded, so a
    #   failure here is reported inline as a TextEvent and the run still
    #   ends with a successful ResultEvent rather than an ErrorEvent.
    #
    # Species clarification: if it's an accession query with no species,
    # there's no `input()` call here (no blocking on stdin under an async
    # server). Instead the run sets `needs_clarification=True`, asks for
    # the species as a TextEvent, and ends via ResultEvent without calling
    # the accession API. The next user message is expected to supply the
    # species in plain text, re-derived by `expand_question_and_queries`.
    async def query(
        self,
        q: str,
        k: int = QUERY_MAX_CHUNKS,
        # Literature context character budget. `None` falls back to
        # `MAX_CHARACTERS`; an explicit out-of-range value raises
        # `ValueError` (see `_resolve_max_context_chars`), with the same
        # propagate-and-fail behaviour as the model selection below.
        max_context_chars: Optional[int] = None,
        # Raw values as forwarded by the frontend's model/reasoning
        # selectors (see `frontend/hooks/use-model-config.ts` and
        # `GraphRAG/main.py`'s reading of `forwardedProps`). Resolved
        # against `AVAILABLE_MODELS`/`AVAILABLE_REASONING_LEVELS` below
        # before use: unset (`None`) falls back to the existing defaults,
        # but an explicit-but-unrecognised value raises
        # `UnavailableModelSelectionError` here - deliberately *not*
        # caught below, so it propagates straight out of this generator
        # and fails the run rather than silently answering with a
        # different model than the one requested.
        model_name: Optional[str] = None,
        reasoning_level: Optional[str] = None,
    ) -> AsyncGenerator[RunEvent, None]:
        logger.info(f"Start query.")
        resolved_model_name = _resolve_model_name(model_name)
        resolved_reasoning_level = _resolve_reasoning_level(reasoning_level)
        max_context_chars = _resolve_max_context_chars(max_context_chars)
        logger.info(
            "Using model=%s reasoning_level=%s max_context_chars=%d",
            resolved_model_name,
            resolved_reasoning_level,
            max_context_chars,
        )
        state = RunState(
            stage=Stage.EXPANDING_QUESTION,
            model_name=resolved_model_name,
            reasoning_level=resolved_reasoning_level,
            max_context_chars=max_context_chars,
        )
        # Every LLM call's usage for this run; summarised into state.token_usage at the end.
        token_tally: list[dict] = []
        query_started = time.perf_counter()
        try:
            # Classify before cache/retrieval so a direct AGG lookup can skip
            # GraphRAG when no suitable cached response exists.
            expansion_diagnostics: dict = {}
            with log_step("Analysis of question", 1, token_tally) as step:
                try:
                    (
                        expanded_question,
                        expanded_queries,
                        is_agg_accession_query,
                        is_direct_agg_lookup,
                        direct_agg_accessions,
                        accession_question,
                        species,
                    ) = await asyncio.to_thread(
                        self.expand_question_and_queries, q, expansion_diagnostics
                    )
                except Exception as e:
                    raw_response = expansion_diagnostics.get("raw_response")
                    if raw_response is None:
                        logger.warning("Question and query expansion failed: %s", e)
                    else:
                        logger.warning(
                            "Question and query expansion failed: %s; raw response: %s",
                            e, raw_response[:500],
                        )
                    expanded_question = q
                    expanded_queries = [q]
                    is_agg_accession_query = False
                    is_direct_agg_lookup = False
                    direct_agg_accessions = []
                    accession_question = ""
                    species = ""
                step["model"] = expansion_diagnostics.get("model")
                step["usage"] = expansion_diagnostics.get("usage_metadata")
                logger.info(
                    "Generated %d retrieval %s:",
                    len(expanded_queries), "query" if len(expanded_queries) == 1 else "queries",
                )
                for index, expanded_query in enumerate(expanded_queries):
                    logger.info(
                        "  [%s] %s",
                        "original" if index == 0 else f"expanded {index}/{len(expanded_queries) - 1}",
                        expanded_query,
                    )

            state = state.model_copy(
                update={
                    "expanded_question": expanded_question,
                    "species": species,
                    "is_agg_accession_query": is_agg_accession_query,
                }
            )
            yield StageChangeEvent(state=state)

            cached = None
            q_emb = None
            if useCache:
                with log_step("Semantic cache lookup", 0):
                    q_emb = await asyncio.to_thread(self.emb.embed_query, q)
                    cached = await asyncio.to_thread(self._semantic_cache_lookup, q_emb)

            if cached:
                cached_expanded_question, cached_answer, _cached_usage = cached
                state = state.model_copy(
                    update={
                        "stage": Stage.GENERATING_ANSWER,
                        "expanded_question": cached_expanded_question,
                    }
                )
                yield StageChangeEvent(state=state)

                prompt = self._build_cached_answer_prompt(q, cached_answer)

                with log_step("Use LLM and previous cached answer to answer", 4, token_tally) as step:
                    answer_parts = []
                    full_chunk = None
                    async for chunk in self._generate_answer_stream(
                        prompt, resolved_model_name, resolved_reasoning_level
                    ):
                        full_chunk = chunk if full_chunk is None else full_chunk + chunk
                        thinking = self._message_thinking(chunk)
                        if thinking:
                            yield ReasoningEvent(text=thinking)
                        text = self._message_text(chunk)
                        if text:
                            answer_parts.append(text)
                            yield TextEvent(text=text)
                    usage_metadata = getattr(full_chunk, "usage_metadata", {}) or {}
                    step["model"] = resolved_model_name
                    step["usage"] = usage_metadata
                state = state.model_copy(update={"usage_metadata": usage_metadata})
                yield ResultEvent(state=state.model_copy(update={"token_usage": summarise_token_tally(token_tally, time.perf_counter() - query_started)}))
                return

            if is_direct_agg_lookup:
                logger.info(
                    "[LLM-classified direct AGG query] Skipping GraphRAG; checking %s",
                    direct_agg_accessions,
                )
                state = state.model_copy(
                    update={
                        "stage": Stage.CHECKING_AGG_ACCESSIONS,
                        "accessions": direct_agg_accessions,
                    }
                )
                yield StageChangeEvent(state=state)

                api_response = await asyncio.to_thread(
                    self._call_accession_api,
                    accession_question or q,
                    direct_agg_accessions,
                )
                if api_response is None:
                    if not ACCESSION_API_URL:
                        direct_answer = (
                            "AGG lookup is not configured. Set ACCESSION_API_URL "
                            "in the repository-root .env file and try again."
                        )
                    else:
                        direct_answer = (
                            "The AGG accession service was unavailable or returned "
                            "no usable response. Please try again."
                        )
                else:
                    state = state.model_copy(
                        update={"stage": Stage.PRESENTING_ACCESSIONS}
                    )
                    yield StageChangeEvent(state=state)
                    presentation_diagnostics: dict = {}
                    try:
                        direct_answer = await asyncio.to_thread(
                            self._present_accession_results, q, api_response,
                            presentation_diagnostics,
                        )
                        if presentation_diagnostics.get("usage_metadata"):
                            token_tally.append(_token_tally_entry(
                                "Summarise and present accession results", GEMINI_MODEL,
                                presentation_diagnostics["usage_metadata"],
                            ))
                    except Exception as e:
                        logger.exception("Presenting direct AGG results failed: %s", e)
                        direct_answer = (
                            "The AGG service returned a result, but it could not be "
                            f"summarised. Raw response: {api_response}"
                        )

                    if useCache:
                        await asyncio.to_thread(
                            self._semantic_cache_store,
                            q,
                            expanded_question,
                            direct_answer,
                            {},
                            q_emb,
                        )

                yield TextEvent(text=direct_answer)
                yield ResultEvent(state=state.model_copy(update={"token_usage": summarise_token_tally(token_tally, time.perf_counter() - query_started)}))
                return

            state = state.model_copy(
                update={
                    "stage": Stage.RETRIEVING_CONTEXT,
                    "retrieval_queries": expanded_queries,
                }
            )
            yield StageChangeEvent(state=state)

            # Run literature, metadata, and Pretzel context retrieval concurrently.
            with log_step("Context retrieval", 2):
                taxon_filter = (
                    resolve_taxon_filter(q, species) if SPECIES_FILTER_ENABLED else None
                )
                if taxon_filter:
                    logger.info(
                        "Applying conservative crop/species filtering: %s",
                        taxon_filter["canonical_crops"],
                    )
                (
                    literature_candidates,
                    metadata_context,
                    pretzel_context,
                    source_errors,
                ) = (
                    await asyncio.to_thread(
                        self._retrieve_context,
                        q,
                        expanded_queries,
                        k,
                        taxon_filter,
                    )
                )

            # Judge the literature candidates and assemble the literature
            # context. Non-fatal, like retrieval: a failure here leaves the
            # answer to be built from the metadata/Pretzel context alone.
            literature_context = ""
            literature_diagnostics: dict[str, Any] = {"chunks": [], "triples": []}
            if not literature_candidates or not literature_candidates["chunks"]:
                logger.info("Literature relevance judge skipped: no literature candidates")
            else:
                state = state.model_copy(update={"stage": Stage.JUDGING_RELEVANCE})
                yield StageChangeEvent(state=state)
                try:
                    with log_step("Literature relevance judge", 3, token_tally) as step:
                        literature_context, literature_diagnostics = await asyncio.to_thread(
                            self._judge_literature_candidates,
                            literature_candidates,
                            max_context_chars,
                            metadata_context,
                        )
                        reranker_usage = literature_diagnostics["chunk_reranker"]["token_usage"]
                        if reranker_usage.get("calls"):
                            step["model"] = reranker_usage.get("model")
                            step["usage"] = reranker_usage
                except Exception as e:
                    source_errors["Literature"] = {"type": type(e).__name__, "message": str(e)}
                    logger.warning("Literature relevance judging failed: %s", e)
            literature_diagnostics.pop("run_trace", None)
            retrieval_diagnostics = {
                "literature": literature_diagnostics, "source_errors": source_errors,
            }
            # Expose exactly what was retrieved from Neo4j and will be added
            # to the answer-generation prompt, so it can be inspected without
            # having to parse the prompt/answer itself.
            state = state.model_copy(
                update={
                    "literature_context": literature_context or None,
                    "metadata_context": metadata_context or None,
                    "pretzel_context": pretzel_context or None,
                    "retrieval_diagnostics": retrieval_diagnostics,
                }
            )

            prompt = self._build_answer_prompt(
                expanded_question,
                q,
                literature_context,
                metadata_context,
                pretzel_context,
            )
            context_char_count = sum(len(context or "") for context in (
                literature_context, metadata_context, pretzel_context
            ))
            logger.info("Retrieved context: %s chars; answer prompt: %s chars",
                        f"{context_char_count:,}", f"{len(prompt):,}")


            state = state.model_copy(update={"stage": Stage.GENERATING_ANSWER})
            yield StageChangeEvent(state=state)

            with log_step("Call LLM to answer", 4, token_tally) as step:
                answer_parts = []
                full_chunk = None
                async for chunk in self._generate_answer_stream(
                    prompt, resolved_model_name, resolved_reasoning_level
                ):
                    full_chunk = chunk if full_chunk is None else full_chunk + chunk
                    thinking = self._message_thinking(chunk)
                    if thinking:
                        yield ReasoningEvent(text=thinking)
                    text = self._message_text(chunk)
                    if text:
                        answer_parts.append(text)
                        yield TextEvent(text=text)
                usage_metadata = getattr(full_chunk, "usage_metadata", {}) or {}
                step["model"] = resolved_model_name
                step["usage"] = usage_metadata

            answer = "".join(answer_parts).strip()
            state = state.model_copy(update={"usage_metadata": usage_metadata})
            full_answer = answer

            if is_agg_accession_query and not species:
                state = state.model_copy(update={"needs_clarification": True})
                yield TextEvent(
                    text="\n\nTo look up these accessions in the Australian Grains "
                    "Genebank (AGG), please specify the species (e.g. wheat, "
                    "barley, oat, chickpea)."
                )
                # Leaving this in for the moment until we add something filter out unrelated queries
                if useCache:
                    await asyncio.to_thread(
                        self._semantic_cache_store,
                        q,
                        expanded_question,
                        full_answer,
                        usage_metadata,
                        q_emb,
                    )
                yield ResultEvent(state=state.model_copy(update={"token_usage": summarise_token_tally(token_tally, time.perf_counter() - query_started)}))
                return

            if is_agg_accession_query:
                state = state.model_copy(
                    update={"stage": Stage.CHECKING_AGG_ACCESSIONS}
                )
                yield StageChangeEvent(state=state)

                try:
                    accessions, api_response = await asyncio.to_thread(
                        self._lookup_agg_accessions,
                        q,
                        answer,
                        species,
                        accession_question,
                        token_tally,
                    )
                except Exception as e:
                    logger.exception("AGG accession lookup failed: %s", e)
                    accessions, api_response = [], None

                state = state.model_copy(update={"accessions": accessions})
                if accessions:
                    if api_response:
                        logger.info(
                            "[Accession API response received] Presenting to user via LLM..."
                        )
                        logger.debug("Accession API response: %s", api_response)

                        state = state.model_copy(
                            update={"stage": Stage.PRESENTING_ACCESSIONS}
                        )
                        yield StageChangeEvent(state=state)

                        presentation_diagnostics: dict = {}
                        with log_step("Summarise and present accession results", 8, token_tally) as step:
                            step["model"] = GEMINI_MODEL
                            try:
                                accession_summary = await asyncio.to_thread(
                                    self._present_accession_results, q, api_response,
                                    presentation_diagnostics,
                                )
                                step["usage"] = presentation_diagnostics.get("usage_metadata")
                            except Exception as e:
                                logger.exception(
                                    "Presenting accession results failed: %s", e
                                )
                                accession_summary = f"(Could not summarise results; raw API response: {api_response})"
                        appended_text = (
                            "\n\n---\n\n**Australian Grains Genebank (AGG) Accession Lookup:**\n"
                            + accession_summary
                        )
                    else:
                        appended_text = "\n\n[AGG accession lookup failed — API unavailable or returned no data.]"
                else:
                    appended_text = "\n\n[No accessions could be confidently selected as direct answers to your question for AGG lookup.]"
                full_answer += appended_text
                yield TextEvent(text=appended_text)

            if useCache:
                await asyncio.to_thread(
                    self._semantic_cache_store,
                    q,
                    expanded_question,
                    full_answer,
                    usage_metadata,
                    q_emb,
                )
            yield ResultEvent(state=state.model_copy(update={"token_usage": summarise_token_tally(token_tally, time.perf_counter() - query_started)}))
        except Exception as e:
            logger.exception("query() failed: %s", e)
            yield ErrorEvent(state=state.model_copy(update={
                "error": str(e),
                "token_usage": summarise_token_tally(token_tally, time.perf_counter() - query_started),
            }))


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Plant Biology RAG Pipeline")
    parser.add_argument(
        "query", type=str, help="The question you want to ask the RAG pipeline"
    )
    parser.add_argument(
        "--model",
        choices=AVAILABLE_MODELS,
        default=None,
        help=f"Answer-generation model. Defaults to {GEMINI_MODEL}.",
    )
    parser.add_argument(
        "--reasoning-level",
        choices=AVAILABLE_REASONING_LEVELS,
        default=None,
        help=f"Answer-generation thinking level. Defaults to {ANSWER_THINKING_LEVEL}.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=None,
        help=(
            "Literature context character budget "
            f"({MIN_LITERATURE_CONTEXT_CHARS}-{MAX_LITERATURE_CONTEXT_CHARS}). "
            f"Defaults to {MAX_CHARACTERS}."
        ),
    )
    args = parser.parse_args()

    rag = PlantBioRAG()

    async def _run():
        answer_parts = []
        final_state = None
        start_time = time.perf_counter()
        async for event in rag.query(
            args.query,
            max_context_chars=args.max_context_chars,
            model_name=args.model,
            reasoning_level=args.reasoning_level,
        ):
            if isinstance(event, TextEvent):
                elapsed = time.perf_counter() - start_time
                print(f"[{elapsed:6.2f}s] {event.text!r}")
                answer_parts.append(event.text)
            elif isinstance(event, (ResultEvent, ErrorEvent)):
                final_state = event.state
        return "".join(answer_parts), final_state

    answer, final_state = asyncio.run(_run())
    if final_state is not None:
        if final_state.error:
            logger.error("Run error: %s", final_state.error)
        logger.info(
            "Model: %s, Reasoning level: %s, Max literature context: %d chars",
            final_state.model_name,
            final_state.reasoning_level,
            final_state.max_context_chars,
        )
        for label, ctx in (
            ("Literature", final_state.literature_context),
            ("Metadata Graph", final_state.metadata_context),
            ("Pretzel Documentation", final_state.pretzel_context),
        ):
            logger.info(
                "Retrieved context [%s]: %d chars", label, len(ctx) if ctx else 0
            )


if __name__ == "__main__":
    main()
