"""Fetch literature graph neighbors without multiplying full chunk texts by paths."""

import time


def expand_literature_graph(graph, chunk_ids, seed_rows=None):
    """Return seed chunks, neighboring chunks, supported triples, and timings.

    Fetch compact relationships and neighbor links separately from chunk text. The
    original all-in-one OPTIONAL MATCH multiplied seed/relationship/neighbor
    rows and aggregated large text maps with DISTINCT inside Neo4j.
    """
    started = time.perf_counter()
    timings = {}
    if not chunk_ids:
        return [], [], [], {"total_seconds": 0.0, "seed_count": 0, "relation_count": 0,
                            "target_count": 0, "neighbor_link_count": 0,
                            "neighbor_count": 0, "triple_count": 0}

    stage_started = time.perf_counter()
    seed_rows = seed_rows if seed_rows is not None else graph.query(
        """
        MATCH (c:Chunk) WHERE c.chunk_id IN $cids
        RETURN c.chunk_id AS chunk_id, c.source_path AS source_path, c.text AS text
        """,
        params={"cids": chunk_ids},
    )
    timings["seed_fetch_seconds"] = time.perf_counter() - stage_started
    seed_chunks = list({(row.get("chunk_id"), row.get("source_path"), row.get("text")): row for row in seed_rows}.values())
    if not seed_chunks:
        timings.update({"total_seconds": time.perf_counter() - started, "seed_count": 0,
                        "relation_count": 0, "target_count": 0, "neighbor_link_count": 0,
                        "neighbor_count": 0, "triple_count": 0})
        return [], [], [], timings

    stage_started = time.perf_counter()
    relation_rows = graph.query(
        """
        MATCH (c:Chunk) WHERE c.chunk_id IN $cids
        MATCH (c)-[:MENTIONS]->(n)-[r]-(m)
        WHERE type(r) <> 'MENTIONS' AND n.id IS NOT NULL AND m.id IS NOT NULL
        RETURN DISTINCT c.chunk_id AS seed_chunk_id, c.source_path AS seed_source_path,
                        n.id AS source_id, type(r) AS relation_type, m.id AS target_id,
                        elementId(m) AS target_node_id
        """,
        params={"cids": chunk_ids},
    )
    timings["relation_fetch_seconds"] = time.perf_counter() - stage_started
    target_ids = sorted({row["target_node_id"] for row in relation_rows})

    stage_started = time.perf_counter()
    neighbor_links = graph.query(
        """
        UNWIND $target_node_ids AS target_node_id
        MATCH (m) WHERE elementId(m) = target_node_id
        MATCH (m)<-[:MENTIONS]-(c:Chunk)
        WHERE NOT c.chunk_id IN $cids
        RETURN DISTINCT target_node_id, c.chunk_id AS chunk_id
        """,
        params={"target_node_ids": target_ids, "cids": chunk_ids},
    ) if target_ids else []
    timings["neighbor_link_fetch_seconds"] = time.perf_counter() - stage_started

    neighbor_ids = sorted({row["chunk_id"] for row in neighbor_links if row.get("chunk_id")})
    stage_started = time.perf_counter()
    neighbor_chunks = graph.query(
        """
        MATCH (c:Chunk) WHERE c.chunk_id IN $neighbor_chunk_ids
        RETURN c.chunk_id AS chunk_id, c.source_path AS source_path, c.text AS text
        """,
        params={"neighbor_chunk_ids": neighbor_ids},
    ) if neighbor_ids else []
    timings["neighbor_text_fetch_seconds"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    chunks_by_id = {row["chunk_id"]: row for row in neighbor_chunks}
    by_target = {}
    expanded = {}
    for row in neighbor_links:
        chunk = chunks_by_id.get(row["chunk_id"])
        if chunk is None:
            continue
        by_target.setdefault(row["target_node_id"], []).append(chunk)
        expanded[(chunk["chunk_id"], chunk["source_path"], chunk["text"])] = chunk

    triples = {}
    for relation in relation_rows:
        seed_source = relation.get("seed_source_path") or ""
        for neighbor in by_target.get(relation["target_node_id"]) or [None]:
            neighbor_source = neighbor.get("source_path") if neighbor else None
            sources = seed_source
            if neighbor_source is not None and neighbor_source != relation.get("seed_source_path"):
                sources += "; " + neighbor_source
            text = f"[Source: {sources}] {relation['source_id']} -[{relation['relation_type']}]-> {relation['target_id']}"
            key = (text, relation["seed_chunk_id"])
            triple = triples.setdefault(key, {
                "text": text, "seed_chunk_id": relation["seed_chunk_id"], "expanded_chunk_ids": {},
            })
            triple["expanded_chunk_ids"][neighbor.get("chunk_id") if neighbor else None] = None
    triple_count = sum(len(triple["expanded_chunk_ids"]) for triple in triples.values())
    for triple in triples.values():
        triple["expanded_chunk_ids"] = list(triple["expanded_chunk_ids"])
    timings.update({
        "triple_assembly_seconds": time.perf_counter() - stage_started,
        "total_seconds": time.perf_counter() - started,
        "seed_count": len(seed_chunks), "relation_count": len(relation_rows),
        "target_count": len(target_ids), "neighbor_link_count": len(neighbor_links),
        "neighbor_count": len(expanded),
        "triple_count": triple_count, "grouped_triple_count": len(triples),
    })
    return seed_chunks, list(expanded.values()), list(triples.values()), timings
