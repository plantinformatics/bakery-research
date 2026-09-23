# RAG pipeline for plant biology papers using Neo4j + Gemini/GPT + LangChain
# Requires env: GOOGLE_API_KEY, OPENAI_API_KEY, NEO4J_URI, NEO4J_USERNAME,
# NEO4J_PASSWORD. Importing this module (what `main.py` does at startup)
# raises if any of those are missing or blank.

import os
import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator, List, Dict, Optional, Tuple, Any, Union
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph, Neo4jVector
import requests
import re
import json
import warnings
import logging
import time
from enum import Enum
from pydantic import BaseModel, Field
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

import prompts

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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s"
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

GEMINI_MODEL = "gemini-3.8-flash"
# GPT models selectable alongside the Gemini models below. Routed through
# `ChatOpenAI` (OpenAI's Responses API) instead of `ChatGoogleGenerativeAI`
# - see `_model_provider`/`_get_answer_llm`.
GPT_MODEL = "gpt-5.6-luna"
GPT_SOL_MODEL = "gpt-5.6-sol"
GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"
# How hard the model reasons before producing the final answer (Gemini 3+
# models only; replaces the older token-based `thinking_budget`).
# One of "minimal", "low", "medium", "high" - higher levels reason more
# deeply at the cost of latency/tokens. Unset defaults to "high".
ANSWER_THINKING_LEVEL = "medium"

# Answer-generation models selectable from the frontend's model selector
# (see `frontend/components/model-selector.tsx`), and the thinking/reasoning
# levels selectable from its reasoning selector. `GraphRAG/main.py` exposes
# these via `GET /options` so the frontend has a single source of truth to
# populate the two dropdowns from. `PlantBioRAG.query()` falls back to the
# first/`ANSWER_THINKING_LEVEL` defaults only when no selection was made at
# all (e.g. an older frontend build); an explicit-but-unrecognised value
# (a stale selection, a bad actor) fails the run instead of silently
# substituting a different model than the one requested - see
# `_resolve_model_name`/`_resolve_reasoning_level`.
#
# Spans two providers: Gemini models go through `ChatGoogleGenerativeAI`,
# `GPT_MODEL`/`GPT_SOL_MODEL` go through `ChatOpenAI` (see `_model_provider`).
# Both providers happen to use the same "minimal"/"low"/"medium"/"high"
# reasoning-effort vocabulary, so `AVAILABLE_REASONING_LEVELS` is shared
# across all of `AVAILABLE_MODELS` without per-model overrides.
AVAILABLE_MODELS = [GEMINI_MODEL, "gemini-2.5-flash", GPT_MODEL, GPT_SOL_MODEL]
AVAILABLE_REASONING_LEVELS = ["minimal", "low", "medium", "high"]

# `thinking_budget` values (in tokens) used for the reasoning selector on
# Gemini models that predate `thinking_level` (e.g. gemini-2.5-flash, which
# rejects `thinking_level` outright with an API 400 - "Thinking level is
# not supported for this model"). Kept in the same "minimal" -> "high"
# vocabulary as `AVAILABLE_REASONING_LEVELS` so the selector behaves
# consistently across model families, even though the underlying Gemini
# API parameter differs. Not used for GPT models, which take
# `reasoning_level` directly as OpenAI's `reasoning_effort`.
REASONING_LEVEL_TO_THINKING_BUDGET = {
    "minimal": 0,
    "low": 1024,
    "medium": 8192,
    "high": 24576,
}


def _model_provider(model_name: str) -> str:
    """"openai" for GPT models (routed through `ChatOpenAI`/the Responses
    API), "google" for everything else (Gemini, via
    `ChatGoogleGenerativeAI`). Add new GPT models to `AVAILABLE_MODELS`
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


MAX_CHARACTERS = 600000
MAX_TRIPLES = 50
QUERY_VECTOR_MAX_CHUNKS = 40
QUERY_FULL_TEXT_MAX_CHUNKS = 40
QUERY_MAX_CHUNKS = 80
MAX_METADATA_CHUNKS = 80
RERANK_MAX_TEXT_CHARS = 2000

# Accession API config
ACCESSION_API_URL = os.getenv("ACCESSION_API_URL") or ""
ACCESSION_API_TOKEN = "research_accessions"
ACCESSION_API_TIMEOUT = 120

METADATA_MAX_CHARACTERS = 300000

SEMANTIC_CACHE_INDEX = "semantic_cache_vector"
SEMANTIC_CACHE_THRESHOLD = 0.92
SEMANTIC_CACHE_TOP_K = 1
SEMANTIC_CACHE_TTL_DAYS = 14


class Stage(str, Enum):
    """Ordered stages of a `PlantBioRAG.query()` run, named after the
    existing `logger.info(...)` milestones."""

    EXPANDING_QUESTION = "expanding_question"  # "Analysis of question"
    RETRIEVING_CONTEXT = "retrieving_context"  # "Concurrent retrieval"
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
    # Answer-generation model/reasoning level actually used for this run,
    # after `_resolve_model_name`/`_resolve_reasoning_level` have applied
    # their fallbacks - lets the frontend confirm the selection took effect.
    model_name: str = GEMINI_MODEL
    reasoning_level: str = ANSWER_THINKING_LEVEL
    species: str = ""
    is_agg_accession_query: bool = False
    needs_clarification: bool = False
    accessions: List[str] = Field(default_factory=list)
    usage_metadata: dict = Field(default_factory=dict)
    # Raw context strings retrieved from Neo4j and injected into the
    # answer-generation prompt by `prompts.build_answer_prompt` (see there
    # for the exact `### [Source: ...]` framing each is wrapped in). Exposed here so
    # callers (e.g. the frontend's pipeline status panel) can inspect exactly
    # what was retrieved, independent of the final cited answer text.
    literature_context: Optional[str] = None
    metadata_context: Optional[str] = None
    pretzel_context: Optional[str] = None
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


class PlantBioRAG:
    def __init__(self):
        self.emb = GoogleGenerativeAIEmbeddings(model=GEMINI_EMBEDDING_MODEL)
        # Fixed model for internal helper calls (question/query expansion,
        # accession extraction/presentation) - these aren't exposed to the
        # frontend's model/reasoning selectors, only the final answer is.
        self.llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
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
        self, q: str, vector_fn, fulltext_fn, k: int
    ) -> Dict[str, float]:
        with ThreadPoolExecutor(max_workers=2) as executor:
            vector_future = executor.submit(vector_fn, q, k)
            fulltext_future = executor.submit(fulltext_fn, q, k)
            vector_scores = vector_future.result()
            fulltext_scores = fulltext_future.result()
        return self._rrf_fusion(vector_scores, fulltext_scores)

    # Run expanded-query searches concurrently
    def _multi_query_hybrid_scores_concurrent(
        self, expanded_queries: list[str], vector_fn, fulltext_fn, k: int
    ) -> Dict[str, float]:
        all_fused: Dict[str, float] = {}
        with ThreadPoolExecutor(
            max_workers=min(8, max(1, len(expanded_queries)))
        ) as executor:
            future_to_query = {
                executor.submit(
                    self._hybrid_scores_concurrent, eq, vector_fn, fulltext_fn, k
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
        self, expanded_query: str, k: int, max_chars_for_query: int
    ) -> dict:
        fused = self._hybrid_scores_concurrent(
            expanded_query, self._vector_chunks, self._fulltext_chunks, k
        )
        seed_cids = sorted(fused, key=lambda x: fused[x], reverse=True)[:k]
        seeded_chunks, expanded_chunks, triples = self._expand_hops(seed_cids)
        all_chunks_deduplicated = self._dedupe_chunks(seeded_chunks + expanded_chunks)
        context_chunks, total_characters = [], 0
        for chunk in all_chunks_deduplicated:
            text = chunk.get("text", "")
            if not text:
                continue
            if total_characters + len(text) > max_chars_for_query:
                break
            context_chunks.append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "source_path": chunk.get("source_path", ""),
                    "text": text,
                }
            )
            total_characters += len(text)
        return {
            "expanded_query": expanded_query,
            "context_chunks": context_chunks,
            "triples": triples[:MAX_TRIPLES],
        }

    # Run all expanded literature searches concurrently
    def _get_literature_context_concurrent(
        self, expanded_queries: list[str], k: int, max_context_chars: int
    ) -> str:
        if not expanded_queries:
            return ""
        max_chars_for_query = int(max_context_chars / max(1, len(expanded_queries)))
        results_by_query: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=min(8, len(expanded_queries))) as executor:
            future_to_query = {
                executor.submit(
                    self._search_literature_one_query, eq, k, max_chars_for_query
                ): eq
                for eq in expanded_queries
            }
            for future in as_completed(future_to_query):
                eq = future_to_query[future]
                results_by_query[eq] = future.result()
        # Preserve expanded_queries order in final prompt
        added_chunk_keys = set()
        parts = []
        for eq in expanded_queries:
            result = results_by_query.get(eq)
            if not result:
                continue
            context_chunks = []
            for chunk in result["context_chunks"]:
                cid = chunk.get("chunk_id")
                source_path = chunk.get("source_path", "")
                text = chunk.get("text", "")
                key = cid if cid else (source_path, hash(text))
                if key in added_chunk_keys:
                    continue
                added_chunk_keys.add(key)
                context_chunks.append(f"[Source: {source_path}] {text}\n")
            triple_summ = "\n".join(result["triples"])
            parts.append(
                f"""
                For sub-question
                {eq}

                ### Context Chunks are:
                {os.linesep.join(context_chunks)}

                ### Entity Relationships are:
                {triple_summ}
                """
            )
        return "\n".join(parts)

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
        self, vector_scores: Dict[str, float], ft_scores: Dict[str, float], k_penalty=60
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

    # 4. Expand from seed chunks to nodes and chunks.
    def _expand_hops(self, cids: List[str]):
        # Fetch actual Triples (Node-Rel-Node)
        query = """
        MATCH (c1:Chunk) WHERE c1.chunk_id IN $cids
        OPTIONAL MATCH (c1)-[:MENTIONS]->(n)
        OPTIONAL MATCH (n)-[r]-(m) WHERE type(r) <> 'MENTIONS' 
        AND n.id IS NOT NULL AND m.id IS NOT NULL  
        OPTIONAL MATCH (m)<-[:MENTIONS]-(c2:Chunk) WHERE NOT c2.chunk_id IN $cids
        RETURN
            collect(DISTINCT c1 {.chunk_id, .source_path, .text}) AS seedchunks,
            collect(DISTINCT c2 {.chunk_id, .source_path, .text}) AS expandedchunks,
            collect(DISTINCT
            CASE
                WHEN r IS NULL THEN NULL
                WHEN c2.source_path IS NULL OR c2.source_path = c1.source_path THEN
                    '[Source: ' + coalesce(c1.source_path, '') + '] ' + n.id + ' -[' + type(r) + ']-> ' + m.id
                ELSE
                    '[Source: ' + coalesce(c1.source_path, '') + '; ' + coalesce(c2.source_path, '') + '] ' + n.id + ' -[' + type(r) + ']-> ' + m.id
            END
        ) AS triples
        """
        res = self.graph.query(query, params={"cids": cids})
        if not res or not res[0]["seedchunks"]:
            return [], [], []
        # Filter out "None -[None]-> None" strings
        triples = [t for t in res[0]["triples"] if t is not None]
        return res[0]["seedchunks"], res[0]["expandedchunks"], triples

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

    def _llm_invoke(self, prompt: Any) -> str:
        resp = self.llm.invoke(prompt)
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
        self, q: str
    ) -> tuple[str, list[str], bool, bool, list[str], str, str]:
        prompt = prompts.build_expand_question_prompt(q)
        resp = self._llm_invoke(prompt)
        clean_json = resp.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)
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
        # Always include original user question first for exact-match retrieval
        expanded_queries.insert(0, q)
        # De-duplicate while preserving order
        expanded_queries = list(dict.fromkeys(expanded_queries))
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
        self, question: str, answer: str, species: str
    ) -> List[str]:
        """Extract only relevant accessions in one LLM call."""
        payload = json.dumps(
            {"question": question, "answer": answer, "species": species},
            ensure_ascii=False,
        )
        prompt = prompts.build_extract_accessions_prompt(payload)
        raw = self._llm_invoke(prompt).strip()
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
        self, original_question: str, api_response: dict
    ) -> str:
        prompt = prompts.build_present_accession_results_prompt(
            original_question, api_response
        )
        resp = self._llm_invoke(prompt)
        return resp

    # 2. Metadata Graph RAG
    def _vector_chunks_metadata(
        self, q: str, k: int = QUERY_VECTOR_MAX_CHUNKS
    ) -> Dict[str, float]:
        # Vector search in metadata_graph via metadata_vector_index using Gemini embeddings.
        q_emb = self.emb.embed_query(q)
        res = self.graph.query(
            "CYPHER 25 MATCH (n:MetadataGraph) "
            "SEARCH n IN (VECTOR INDEX metadata_vector_index FOR $emb LIMIT $k) SCORE AS score "
            "RETURN elementId(n) AS nid, score",
            params={"k": k, "emb": q_emb},
        )
        return {r["nid"]: r["score"] for r in res}

    def _fulltext_chunks_metadata(
        self, q: str, k: int = QUERY_FULL_TEXT_MAX_CHUNKS
    ) -> Dict[str, float]:
        # Fulltext search in metadata_graph via metadata_fulltext_index.
        cleaned_q = self.escape_lucene_plain_text(q)
        if not cleaned_q:
            return {}
        res = self.graph.query(
            "CALL db.index.fulltext.queryNodes('metadata_fulltext_index', $q) YIELD node, score "
            "RETURN elementId(node) AS nid, score ORDER BY score DESC LIMIT $k",
            params={"q": cleaned_q, "k": k},
        )
        return {r["nid"]: r["score"] for r in res}

    def _expand_one_hop(self, nids: List[str]):
        # Expand MetadataGraph seed nodes by 1 hop following any relationship, both directions.
        # Returns nodes with all properties plus chunk_id for dedupe/fetch.
        query = """
        MATCH (n1:MetadataGraph) WHERE elementId(n1) IN $nids
        OPTIONAL MATCH (n1)-[r]-(n2)
        RETURN
            collect(DISTINCT n1 {
                .*,
                chunk_id: elementId(n1),
                source_path: 'Metadata Graph',
                labels: labels(n1)
            }) AS seedchunks,

            collect(DISTINCT n2 {
                .*,
                chunk_id: elementId(n2),
                source_path: 'Metadata Graph',
                labels: labels(n2)
            }) AS expandedchunks,

            collect(DISTINCT
                CASE
                    WHEN r IS NULL OR n2 IS NULL THEN NULL
                    ELSE
                        '[Source: Metadata Graph] ' +
                        coalesce(n1.displayName, n1.shortName, n1.id, n1.projectName,
                                n1.accessionName, n1.curatorName, elementId(n1)) +
                        ' -[' + type(r) + ']- ' +
                        coalesce(n2.displayName, n2.shortName, n2.id, n2.projectName,
                                n2.accessionName, n2.curatorName, elementId(n2))
                END
            ) AS triples
        """
        res = self.graph.query(query, params={"nids": nids})
        if not res or not res[0]["seedchunks"]:
            return [], [], []
        seedchunks = [c for c in res[0]["seedchunks"] if c and c.get("chunk_id")]
        expandedchunks = [
            c for c in res[0]["expandedchunks"] if c and c.get("chunk_id")
        ]
        triples = [t for t in res[0]["triples"] if t is not None]
        return seedchunks, expandedchunks, triples

    def _search_metadata_hybrid(
        self, expanded_queries: list[str], max_chars: int = METADATA_MAX_CHARACTERS
    ) -> str:
        # Hybrid search for metadata_graph.
        # Run expanded-query metadata hybrid searches concurrently.
        # Each query also runs vector + full-text concurrently.
        all_fused = self._multi_query_hybrid_scores_concurrent(
            expanded_queries,
            self._vector_chunks_metadata,
            self._fulltext_chunks_metadata,
            QUERY_VECTOR_MAX_CHUNKS,
        )
        # Sort by fused score descending
        top_nids = sorted(all_fused, key=lambda x: all_fused[x], reverse=True)[
            :MAX_METADATA_CHUNKS
        ]
        if not top_nids:
            return ""
        seeded_chunks, expanded_chunks, triples = self._expand_one_hop(top_nids)
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
        for t in triples[:MAX_TRIPLES]:
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

    def _get_metadata_context(self, query: str, expanded_queries: list[str]) -> str:
        context_parts = []
        result = self._search_metadata_hybrid(expanded_queries)
        if result:
            context_parts.append(f"### Metadata Graph (Hybrid Search):\n{result}")
        return "\n\n".join(context_parts)

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
        all_fused = self._multi_query_hybrid_scores_concurrent(
            expanded_queries,
            self._vector_chunks_pretzel,
            self._fulltext_chunks_pretzel,
            QUERY_VECTOR_MAX_CHUNKS,
        )
        # Sort by fused score descending
        top_nids = sorted(all_fused, key=lambda x: all_fused[x], reverse=True)[:50]
        if not top_nids:
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
        return "\n".join(context_parts)

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
    # back. A failed source just contributes an empty string.
    def _retrieve_context(
        self, q: str, expanded_queries: List[str], k: int, max_context_chars: int
    ) -> Tuple[str, str, str]:
        def safe_call(fn, label, *args):
            try:
                return fn(*args)
            except Exception as e:
                logger.warning("%s context retrieval failed: %s", label, e)
                return ""

        with ThreadPoolExecutor(max_workers=3) as executor:
            literature_future = executor.submit(
                safe_call,
                self._get_literature_context_concurrent,
                "Literature",
                expanded_queries,
                k,
                max_context_chars,
            )
            metadata_future = executor.submit(
                safe_call, self._get_metadata_context, "Metadata", q, expanded_queries
            )
            pretzel_future = None
            if "pretzel" in q.lower():
                pretzel_future = executor.submit(
                    safe_call, self._get_pretzel_context, "Pretzel", expanded_queries
                )
            literature_context = literature_future.result()
            metadata_context = metadata_future.result()
            pretzel_context = pretzel_future.result() if pretzel_future else ""
        return literature_context, metadata_context, pretzel_context

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
        self, q: str, answer: str, species: str, accession_question: str
    ) -> Tuple[List[str], Optional[dict]]:
        logger.info(
            "[AGG Accession Query Detected] Extracting accessions from RAG answer..."
        )
        start_time = time.perf_counter()
        accessions = self._extract_accessions(q, answer, species)
        end_time = time.perf_counter()
        logger.info("6. Extract relevant accessions: %.1f sec.", end_time - start_time)
        logger.info("[Relevant Accessions]: %s", accessions)

        api_response = None
        if accessions:
            start_time = time.perf_counter()
            api_response = self._call_accession_api(accession_question, accessions)
            end_time = time.perf_counter()
            logger.info(f"7. Call accession API: {end_time - start_time:0.1f} sec.")
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
        max_context_chars: int = MAX_CHARACTERS,
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
        logger.info(
            "Using model=%s reasoning_level=%s",
            resolved_model_name,
            resolved_reasoning_level,
        )
        state = RunState(
            stage=Stage.EXPANDING_QUESTION,
            model_name=resolved_model_name,
            reasoning_level=resolved_reasoning_level,
        )
        try:
            # Classify before cache/retrieval so a direct AGG lookup can skip
            # GraphRAG when no suitable cached response exists.
            start_time = time.perf_counter()
            try:
                (
                    expanded_question,
                    expanded_queries,
                    is_agg_accession_query,
                    is_direct_agg_lookup,
                    direct_agg_accessions,
                    accession_question,
                    species,
                ) = await asyncio.to_thread(self.expand_question_and_queries, q)
            except Exception as e:
                logger.warning("Question and query expansion failed: %s", e)
                expanded_question = q
                expanded_queries = [q]
                is_agg_accession_query = False
                is_direct_agg_lookup = False
                direct_agg_accessions = []
                accession_question = ""
                species = ""
            end_time = time.perf_counter()
            logger.info(f"1. Analysis of question: {end_time - start_time:0.1f} sec.")

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
                start_time = time.perf_counter()
                q_emb = await asyncio.to_thread(self.emb.embed_query, q)
                cached = await asyncio.to_thread(self._semantic_cache_lookup, q_emb)
                end_time = time.perf_counter()
                logger.info(
                    f"0. Semantic cache lookup: {end_time - start_time:0.1f} sec."
                )

            if cached:
                cached_expanded_question, cached_answer, _cached_usage = cached
                state = state.model_copy(
                    update={
                        "stage": Stage.GENERATING_ANSWER,
                        "expanded_question": cached_expanded_question,
                    }
                )
                yield StageChangeEvent(state=state)

                prompt = prompts.build_cached_answer_prompt(q, cached_answer)

                start_time = time.perf_counter()
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
                end_time = time.perf_counter()
                logger.info(
                    "4. Use LLM and previous cached answer to answer: %.1f sec.",
                    end_time - start_time,
                )

                usage_metadata = getattr(full_chunk, "usage_metadata", {}) or {}
                state = state.model_copy(update={"usage_metadata": usage_metadata})
                yield ResultEvent(state=state)
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
                    try:
                        direct_answer = await asyncio.to_thread(
                            self._present_accession_results, q, api_response
                        )
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
                yield ResultEvent(state=state)
                return

            state = state.model_copy(update={"stage": Stage.RETRIEVING_CONTEXT})
            yield StageChangeEvent(state=state)

            # Run literature, metadata, and Pretzel context retrieval concurrently.
            start_time = time.perf_counter()
            literature_context, metadata_context, pretzel_context = (
                await asyncio.to_thread(
                    self._retrieve_context, q, expanded_queries, k, max_context_chars
                )
            )
            end_time = time.perf_counter()
            logger.info(
                "2. Concurrent retrieval: literature + metadata + Pretzel context: %.1f sec.",
                end_time - start_time,
            )

            # Expose exactly what was retrieved from Neo4j and will be added
            # to the answer-generation prompt, so it can be inspected without
            # having to parse the prompt/answer itself.
            state = state.model_copy(
                update={
                    "literature_context": literature_context or None,
                    "metadata_context": metadata_context or None,
                    "pretzel_context": pretzel_context or None,
                }
            )

            prompt = prompts.build_answer_prompt(
                expanded_question,
                q,
                literature_context,
                metadata_context,
                pretzel_context,
            )

            state = state.model_copy(update={"stage": Stage.GENERATING_ANSWER})
            yield StageChangeEvent(state=state)

            start_time = time.perf_counter()
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
            end_time = time.perf_counter()
            logger.info(f"4. Call LLM to answer: {end_time - start_time:0.1f} sec.")

            answer = "".join(answer_parts).strip()
            usage_metadata = getattr(full_chunk, "usage_metadata", {}) or {}
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
                yield ResultEvent(state=state)
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

                        start_time = time.perf_counter()
                        try:
                            accession_summary = await asyncio.to_thread(
                                self._present_accession_results, q, api_response
                            )
                        except Exception as e:
                            logger.exception(
                                "Presenting accession results failed: %s", e
                            )
                            accession_summary = f"(Could not summarise results; raw API response: {api_response})"
                        end_time = time.perf_counter()
                        logger.info(
                            f"8. Summarise and present accession results: {end_time - start_time:0.1f} sec."
                        )
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
            yield ResultEvent(state=state)
        except Exception as e:
            logger.exception("query() failed: %s", e)
            yield ErrorEvent(state=state.model_copy(update={"error": str(e)}))


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
    args = parser.parse_args()

    rag = PlantBioRAG()

    async def _run():
        answer_parts = []
        final_state = None
        start_time = time.perf_counter()
        async for event in rag.query(
            args.query, model_name=args.model, reasoning_level=args.reasoning_level
        ):
            if isinstance(event, TextEvent):
                elapsed = time.perf_counter() - start_time
                print(f"[{elapsed:6.2f}s] {event.text!r}")
                answer_parts.append(event.text)
            elif isinstance(event, (ResultEvent, ErrorEvent)):
                final_state = event.state
        return "".join(answer_parts), final_state

    answer, final_state = asyncio.run(_run())
    logger.info("Final Answer:\n%s", answer)
    if final_state is not None:
        if final_state.error:
            logger.error("Run error: %s", final_state.error)
        logger.info(
            "Model: %s, Reasoning level: %s",
            final_state.model_name,
            final_state.reasoning_level,
        )
        logger.info("Token Usage: %s", str(final_state.usage_metadata))
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
