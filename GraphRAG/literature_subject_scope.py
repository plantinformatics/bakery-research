"""Subject-scoped literature judging, relationship selection, and diagnostics."""
import math
import re
from collections import Counter
from metadata_query_grounding import matching_subjects

# Relation types scope_relationships may admit without a subject endpoint.
SCOPE_BRIDGE_RELATIONS = frozenset({"PLEIOTROPICTO", "LINKEDTO", "ALIASOF", "SYNONYMOF", "SAMEAS"})
# Relation types read by Query._boost_connected_triple_relevance, including edges
# that scope_relationships later rejects.
RELEVANCE_BOOST_RELATIONS = frozenset({
    "PLEIOTROPICTO", "CONFERSRESISTANCETO", "CARRIES", "CARRIESGENE",
    "HASGENE", "LOCATEDON", "ASSOCIATEDWITH", "ASSOCIATEDWITHTRAIT",
})


def relationship_scope_filter(subjects):
    """Return a predicate for graph edges that can affect subject-scoped output.

    With resolved subjects, scope_relationships rejects every edge that neither
    names a subject nor has a bridge type, and the relevance boost reads only
    boost types. Other edges cannot change the selected relationships or their
    scores, so graph expansion can skip building their text. Neighbour chunks are
    unaffected. Returns None when no subjects are resolved.
    """
    names = [subject.strip() for subject in subjects or []
             if isinstance(subject, str) and subject.strip()]
    if not names:
        return None
    # Same boundary rule as matching_subjects, compiled once for all subjects.
    pattern = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(name) for name in names) + r")(?!\w)", re.I)
    kept_types = SCOPE_BRIDGE_RELATIONS | RELEVANCE_BOOST_RELATIONS

    def keep(edge, relation_type):
        return (str(relation_type).upper().replace("_", "") in kept_types
                or pattern.search(edge) is not None)
    return keep


def derive_subject_assessments(score, entities, questions):
    """Fallback per-subject scores when the judge omits or garbles them.

    The judge's top-level score still judges the passage against the question.
    Subjects named in its evidence entities get that score; others get 0.
    """
    names = " | ".join(entity for entity in entities if isinstance(entity, str)) if isinstance(entities, list) else ""
    named = set(matching_subjects(names, list(questions)))
    return {subject: {"score": score if subject in named else 0.0,
                      "reason": ("Derived from the judge's overall score; per-subject scores were missing"
                                 if subject in named else
                                 "Not among the judge's evidence entities; per-subject scores were missing")}
            for subject in questions}


def subject_questions(subjects, queries, question):
    return {subject: next((query for query in queries if matching_subjects(query, [subject])),
                          f"For {subject} only: {question}") for subject in subjects}


def parse_subject_assessments(item, questions):
    """Require every subject score; malformed/missing results use existing retries."""
    if not questions:
        return {}
    rows = item.get("subject_assessments")
    if not isinstance(rows, list):
        raise ValueError("Missing subject assessments")
    results = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("subject") not in questions:
            raise ValueError("Unknown subject assessment")
        subject = row["subject"]
        score = row.get("score")
        if (subject in results or isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or not 0 <= score <= 4
                or not isinstance(row.get("reason"), str)):
            raise ValueError("Invalid subject assessment")
        results[subject] = {"score": score, "reason": row["reason"]}
    if set(results) != set(questions):
        raise ValueError("Incomplete subject assessments")
    return results


def scope_relationships(triples, subjects, question):
    """Keep subject edges and short gene bridges to the requested entity.

    Paths are undirected because retrieval emits undirected graph traversals.
    Scope selection is not a biological claim or a gene-equivalence inference.
    """
    if not subjects:
        return
    question = question.split("Metadata-resolved search subjects:")[0]
    bridges = SCOPE_BRIDGE_RELATIONS
    parsed, adjacency = [], {}
    for index, item in enumerate(triples):
        match = re.match(r"^\s*(.*?)\s*-\[([^]]+)\]->\s*(.*?)\s*$", item["relationship"])
        if not match or item.get("rejection_reason"):
            continue
        left, relation, right = match.groups()
        left, right = left.strip(), right.strip()
        matches = matching_subjects(left, subjects) + matching_subjects(right, subjects)
        parsed.append((index, left, right, matches))
        if relation.upper().replace("_", "") in bridges and not matches:
            adjacency.setdefault(left.casefold(), []).append((right, index))
            adjacency.setdefault(right.casefold(), []).append((left, index))
    allowed = {}
    for index, left, right, matches in parsed:
        if not matches:
            continue
        allowed.setdefault(index, set()).update(matches)
        other = right if matching_subjects(left, subjects) else left
        paths = [(other, [], {other.casefold()})]
        for node, path, visited in paths:
            if matching_subjects(question, [node]) and not matching_subjects(node, subjects):
                for edge in [index] + path:
                    allowed.setdefault(edge, set()).update(matches)
            if len(path) < 2:
                for neighbor, edge in adjacency.get(node.casefold(), []):
                    if neighbor.casefold() not in visited:
                        paths.append((neighbor, path + [edge], visited | {neighbor.casefold()}))
    for index, item in enumerate(triples):
        item["scope_subjects"] = sorted(allowed.get(index, []))
        if index not in allowed and not item.get("rejection_reason"):
            item["rejection_reason"] = "Outside resolved subjects and question-linked gene bridges"


def subject_coverage(subjects, queries, question, chunks, triples):
    coverage = {}
    for subject, query in subject_questions(subjects, queries, question).items():
        retrieved = [item for item in chunks if any(
            path.get("query") == query for path in item.get("retrieval_queries", []))]
        judged = [item for item in chunks if subject in item.get("subject_assessments", {})]
        accepted = [item for item in judged if item["subject_assessments"][subject]["score"] >= 2]
        selected = [item for item in accepted if item.get("included")]
        relations = [item for item in triples if subject in item.get("scope_subjects", []) and item.get("included")]
        coverage[subject] = {
            "question": query, "retrieved": len(retrieved),
            "mentioning_subject": sum(bool(matching_subjects(item.get("text", ""), [subject])) for item in chunks),
            "judged": len(judged), "accepted": len(accepted), "selected": len(selected),
            "selected_relationships": len(relations),
            "rejections": dict(Counter(item.get("rejection_reason") for item in retrieved if item.get("rejection_reason"))),
            "assessments": [{"chunk_id": item.get("chunk_id"), **item["subject_assessments"][subject],
                             "included": bool(item.get("included")), "rejection_reason": item.get("rejection_reason")}
                            for item in judged],
        }
    return coverage
