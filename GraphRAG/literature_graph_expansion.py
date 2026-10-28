"""Fetch literature graph neighbors without multiplying full chunk texts by paths."""

import threading
import time
from literature_read_cache import LiteratureReadCache


def prepare_literature_graph(graph, chunk_ids, read_cache, keep_relationship=None):
    """Fetch and index the graph neighbourhood of chunk_ids into read_cache.

    Relations, neighbour links, neighbour texts, target groups and per-seed edge
    indexes are stored once per key, so later calls for any subset of these
    seeds only read the cache. Returns the rows and indexes for chunk_ids.
    """
    cache = read_cache
    timings = {}
    stage_started = time.perf_counter()
    relation_rows = cache.read_many("relations", chunk_ids, lambda missing: graph.query(
        """
        MATCH (c:Chunk) WHERE c.chunk_id IN $cids
        MATCH (c)-[:MENTIONS]->(n)-[r]-(m)
        WHERE type(r) <> 'MENTIONS' AND n.id IS NOT NULL AND m.id IS NOT NULL
        RETURN DISTINCT c.chunk_id AS seed_chunk_id, c.source_path AS seed_source_path,
                        n.id AS source_id, type(r) AS relation_type, m.id AS target_id,
                        elementId(m) AS target_node_id
        """,
        params={"cids": missing},
    ), "seed_chunk_id")
    timings["relation_fetch_seconds"] = time.perf_counter() - stage_started
    target_ids = sorted({row["target_node_id"] for row in relation_rows})

    stage_started = time.perf_counter()
    neighbor_links = cache.read_many("links", target_ids, lambda missing: graph.query(
        """
        UNWIND $target_node_ids AS target_node_id
        MATCH (m) WHERE elementId(m) = target_node_id
        MATCH (m)<-[:MENTIONS]-(c:Chunk)
        RETURN DISTINCT target_node_id, c.chunk_id AS chunk_id
        """,
        params={"target_node_ids": missing},
    ), "target_node_id") if target_ids else []
    timings["neighbor_link_fetch_seconds"] = time.perf_counter() - stage_started

    neighbor_ids = sorted({row["chunk_id"] for row in neighbor_links if row.get("chunk_id")})
    stage_started = time.perf_counter()
    neighbor_chunks = cache.read_many("texts", neighbor_ids, lambda missing: graph.query(
        """
        MATCH (c:Chunk) WHERE c.chunk_id IN $neighbor_chunk_ids
        RETURN c.chunk_id AS chunk_id, c.source_path AS source_path, c.text AS text
        """,
        params={"neighbor_chunk_ids": missing},
    ), "chunk_id") if neighbor_ids else []
    timings["neighbor_text_fetch_seconds"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    chunks_by_id = {row["chunk_id"]: row for row in neighbor_chunks}
    relations_by_seed = {}
    for row in relation_rows:
        relations_by_seed.setdefault(row["seed_chunk_id"], []).append(row)
    expansion_namespace = "seed_expansion" if keep_relationship is None else "seed_expansion_scoped"
    seeds = list(dict.fromkeys(chunk_ids))
    if all(cache.contains(expansion_namespace, seed) for seed in seeds):
        # Every seed is already indexed (e.g. by SharedSeedExpansion); skip regrouping targets.
        seed_edges = {seed: cache.compute_once(expansion_namespace, seed, None) for seed in seeds}
        timings["seed_index_seconds"] = time.perf_counter() - stage_started
        return {
            "relation_rows": relation_rows, "target_ids": target_ids,
            "neighbor_links": neighbor_links, "chunks_by_id": chunks_by_id,
            "relations_by_seed": relations_by_seed, "seed_edges": seed_edges,
            "timings": timings,
        }
    links_by_target = {}
    for row in neighbor_links:
        links_by_target.setdefault(row["target_node_id"], []).append(row["chunk_id"])

    def build_target(target):
        groups = {}
        for cid in links_by_target.get(target, ()):
            chunk = chunks_by_id.get(cid)
            if chunk is not None:
                groups.setdefault(chunk.get("source_path"), set()).add(cid)
        return tuple((source, frozenset(ids)) for source, ids in groups.items())

    # Target grouping is independent of a query's seed exclusions. All seed
    # chunks are fetched above so the shared value is complete for every branch.
    target_groups = {
        target: cache.compute_once("target_groups", target, lambda target=target: build_target(target))
        for target in target_ids
    }

    def build_seed(seed):
        edges, reachable = [], set()
        for relation in relations_by_seed.get(seed, ()):
            seed_source = relation.get("seed_source_path") or ""
            edge = f"{relation['source_id']} -[{relation['relation_type']}]-> {relation['target_id']}"
            if keep_relationship is not None and not keep_relationship(edge, relation["relation_type"]):
                # Keep the neighbour chunks reachable; skip only the relationship text.
                for _source, ids in target_groups[relation["target_node_id"]]:
                    reachable.update(ids)
                continue
            groups = []
            for source, ids in target_groups[relation["target_node_id"]]:
                sources = seed_source
                if source is not None and source != relation.get("seed_source_path"):
                    sources += "; " + source
                groups.append((f"[Source: {sources}] {edge}", ids))
                reachable.update(ids)
            edges.append((f"[Source: {seed_source}] {edge}", tuple(groups)))
        return tuple(edges), frozenset(reachable)

    seed_edges = {
        seed: cache.compute_once(expansion_namespace, seed, lambda seed=seed: build_seed(seed))
        for seed in seeds
    }
    timings["seed_index_seconds"] = time.perf_counter() - stage_started
    return {
        "relation_rows": relation_rows, "target_ids": target_ids,
        "neighbor_links": neighbor_links, "chunks_by_id": chunks_by_id,
        "relations_by_seed": relations_by_seed, "seed_edges": seed_edges,
        "timings": timings,
    }


def _seed_triple_texts(edges, union_seed_ids):
    """Triple texts one seed yields for any query whose seeds are within union_seed_ids.

    A group is skipped only when all its neighbour chunks are the query's own
    seeds. If no group lies entirely within the union, no query can skip one, so
    the texts are the same for every query. Returns None when they could differ.
    """
    texts = {}
    for fallback, groups in edges:
        for text, ids in groups:
            if ids <= union_seed_ids:
                return None
            texts.setdefault(text, None)
        if not groups:
            texts.setdefault(fallback, None)
    return tuple(texts)


class SharedSeedExpansion:
    """Expand the union of a request's seeds once, before each query's own view.

    Each query calls arrive(seed_ids) after choosing its seeds. The last arrival
    prepares the union in the shared read cache; the others wait, then build
    their views from cached entries. A query that fails before arriving would
    leave the rest waiting, so waits time out and fall back to per-query reads,
    which give the same result.
    """

    def __init__(self, graph, query_count, read_cache, keep_relationship=None, timeout_seconds=30.0):
        self._graph = graph
        self._read_cache = read_cache
        self._keep_relationship = keep_relationship
        self._timeout_seconds = timeout_seconds
        self._condition = threading.Condition()
        self._remaining = query_count
        self._seeds = {}
        self._ready = False
        self.seed_uses = 0
        self.timings = None

    def arrive(self, seed_ids):
        with self._condition:
            self._seeds.update(dict.fromkeys(seed_ids))
            self.seed_uses += len(seed_ids)
            self._remaining -= 1
            is_last = self._remaining == 0
            if not is_last:
                self._condition.wait_for(lambda: self._ready, timeout=self._timeout_seconds)
                return
        started = time.perf_counter()
        try:
            prepared = prepare_literature_graph(
                self._graph, list(self._seeds), self._read_cache, self._keep_relationship
            )
            union = frozenset(self._seeds)
            namespace = "seed_view" if self._keep_relationship is None else "seed_view_scoped"
            for seed, (edges, _reachable) in prepared["seed_edges"].items():
                self._read_cache.compute_once(
                    namespace, seed, lambda seed=seed, edges=edges: (union, _seed_triple_texts(edges, union))
                )
            self.timings = {**prepared["timings"], "total_seconds": time.perf_counter() - started,
                            "unique_seed_count": len(self._seeds), "seed_uses": self.seed_uses}
        finally:
            with self._condition:
                self._ready = True
                self._condition.notify_all()


def expand_literature_graph(graph, chunk_ids, seed_rows=None, read_cache=None, provenance=None,
                            keep_relationship=None):
    """Return seed chunks, neighboring chunks, supported triples, and timings.

    Fetch compact relationships and neighbor links separately from chunk text. The
    original all-in-one OPTIONAL MATCH multiplied seed/relationship/neighbor
    rows and aggregated large text maps with DISTINCT inside Neo4j.

    keep_relationship(edge, relation_type), when given, limits which relationship
    texts are built. Every relation still contributes its neighbour chunks.

    When provenance is given, triples carry no expanded_chunk_ids (callers use the
    provenance index instead), which skips building per-triple id sets.
    """
    cache = read_cache if read_cache is not None else LiteratureReadCache()
    started = time.perf_counter()
    timings = {}
    if not chunk_ids:
        return [], [], [], {"total_seconds": 0.0, "seed_count": 0, "relation_count": 0,
                            "target_count": 0, "neighbor_link_count": 0,
                            "neighbor_count": 0, "triple_count": 0, "grouped_triple_count": 0}

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
                        "neighbor_count": 0, "triple_count": 0, "grouped_triple_count": 0})
        return [], [], [], timings

    prepared = prepare_literature_graph(graph, chunk_ids, cache, keep_relationship)
    timings.update(prepared["timings"])
    relations_by_seed, seed_edges = prepared["relations_by_seed"], prepared["seed_edges"]
    chunks_by_id = prepared["chunks_by_id"]
    seed_ids = set(chunk_ids)

    stage_started = time.perf_counter()
    view_namespace = "seed_view" if keep_relationship is None else "seed_view_scoped"
    triples, expanded_ids = [], set()
    provenance_links = 0
    relationship_edges_total = relationship_edges_kept = 0
    for seed in dict.fromkeys(chunk_ids):
        edges, reachable = seed_edges[seed]
        relationship_edges_total += len(relations_by_seed.get(seed, ()))
        relationship_edges_kept += len(edges)
        # Rebuild only query-dependent views; never mutate the shared index.
        eligible = reachable - seed_ids
        expanded_ids.update(eligible)
        provenance_links += len(eligible)
        if provenance is not None:
            for cid in eligible:
                provenance.setdefault(cid, set()).add(seed)
            # Texts shared across the request's queries (SharedSeedExpansion), when valid here.
            view = cache.compute_once(view_namespace, seed, None) if cache.contains(view_namespace, seed) else None
            if view is not None and view[1] is not None and seed_ids <= view[0]:
                triples.extend({"text": text, "seed_chunk_id": seed} for text in view[1])
                continue
        seed_triples = {}
        for fallback, groups in edges:
            found = False
            for text, ids in groups:
                if ids <= seed_ids:  # every neighbour chunk is one of this query's seeds
                    continue
                found = True
                triple = seed_triples.setdefault(text, {"text": text, "seed_chunk_id": seed,
                                                        "expanded_chunk_ids": set()})
                if provenance is None:
                    triple["expanded_chunk_ids"].update(ids - seed_ids)
            if not found:
                triple = seed_triples.setdefault(fallback, {"text": fallback, "seed_chunk_id": seed,
                                                            "expanded_chunk_ids": set()})
                triple["expanded_chunk_ids"].add(None)
        triples.extend(seed_triples.values())
    triple_count = sum(len(triple["expanded_chunk_ids"]) for triple in triples) if provenance is None else None
    for triple in triples:
        # Query.py uses the compact provenance index and does not need to walk
        # these repeated IDs. Keep the public result available to other callers.
        if provenance is None:
            triple["expanded_chunk_ids"] = sorted(triple["expanded_chunk_ids"], key=lambda cid: (cid is not None, cid or ""))
        else:
            triple.pop("expanded_chunk_ids", None)
    expanded = [chunks_by_id[cid] for cid in sorted(expanded_ids)]
    timings.update({
        "triple_assembly_seconds": time.perf_counter() - stage_started,
        "total_seconds": time.perf_counter() - started,
        "seed_count": len(seed_chunks), "relation_count": len(prepared["relation_rows"]),
        "target_count": len(prepared["target_ids"]),
        "neighbor_link_count": sum(row["chunk_id"] not in seed_ids for row in prepared["neighbor_links"]),
        "neighbor_count": len(expanded),
        "triple_count": triple_count, "grouped_triple_count": len(triples),
        "unique_seed_neighbor_links": provenance_links,
        "relationship_edges_total": relationship_edges_total,
        "relationship_edges_kept": relationship_edges_kept,
        "shared_processing": True,
    })
    return seed_chunks, expanded, triples, timings
