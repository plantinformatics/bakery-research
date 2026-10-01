"""Deterministic candidate tie breaking before literature LLM reranking.

RRF remains the primary retrieval score. Graph-expanded chunks often inherit
identical seed scores, so use their own vector similarity to order those ties.
"""

from collections import Counter
import math
import re
import time


_STOPWORDS = {
    "a", "an", "and", "any", "are", "as", "at", "be", "between", "by",
    "do", "does", "for", "from", "have", "how", "in", "is", "it", "of",
    "on", "or", "that", "the", "their", "these", "this", "to", "what",
    "which", "with",
}


def score_literature_candidate_texts(candidates, question, expanded_queries):
    """Set a corpus-aware lexical tie-break score on candidate dictionaries.

    This is not an RRF score and never substitutes for the LLM relevance judge.
    Rare matching question terms matter more than corpus-common terms.
    """
    query_text = " ".join([question] + list(expanded_queries))
    query_terms = Counter(
        term for term in re.findall(r"[\w-]+", query_text.casefold())
        if len(term) > 2 and term not in _STOPWORDS
    )
    document_terms = []
    document_frequency = Counter()
    for candidate in candidates:
        words = re.findall(r"[\w-]+", (candidate.get("text") or "").casefold())
        counts = Counter(term for term in words if term in query_terms)
        document_terms.append((counts, len(words)))
        document_frequency.update(counts.keys())

    total = len(candidates)
    for candidate, (counts, length) in zip(candidates, document_terms):
        score = sum(
            min(query_count, 3)
            * math.log1p((total - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
            * (count / (count + 1.0))
            for term, query_count in query_terms.items()
            if (count := counts.get(term, 0))
        )
        candidate["retrieval_lexical_score"] = score / (1.0 + length / 400.0)


def literature_candidate_sort_key(candidate, rrf_score):
    """Higher RRF, then own vector similarity; ID settles exact ties."""
    return (
        -(rrf_score or 0.0),
        -candidate.get("retrieval_vector_similarity", -1.0),
        -candidate.get("retrieval_lexical_score", 0.0),
        -int(bool(candidate.get("is_direct_search_match"))),
        str(candidate.get("chunk_id") or ""),
    )


def score_literature_candidate_vectors(candidates, question, graph, embedding_model, timings=None):
    """Score expanded chunks by their *own* embeddings, not only seed RRF.

    Neo4j computes cosine server-side, so the large vectors are not transferred
    to Python. A failed call leaves the lexical tie-break available.
    """
    ids = [item["chunk_id"] for item in candidates if item.get("chunk_id") and item.get("rejection_reason") is None]
    if not ids:
        return 0
    started = time.perf_counter()
    try:
        query_embedding = embedding_model.embed_query(question)
    finally:
        if timings is not None:
            timings["question_embedding"] = time.perf_counter() - started
    started = time.perf_counter()
    try:
        rows = graph.query(
            """
            MATCH (c:Chunk) WHERE c.chunk_id IN $cids AND c.embedding IS NOT NULL
            RETURN c.chunk_id AS chunk_id,
                   vector.similarity.cosine(c.embedding, $query_embedding) AS similarity
            """,
            params={"cids": ids, "query_embedding": query_embedding},
        )
    finally:
        if timings is not None:
            timings["neo4j_cosine"] = time.perf_counter() - started
    by_id = {row["chunk_id"]: row["similarity"] for row in rows if row.get("similarity") is not None}
    for item in candidates:
        if item.get("chunk_id") in by_id:
            item["retrieval_vector_similarity"] = by_id[item["chunk_id"]]
    return len(by_id)
