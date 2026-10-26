"""Resolve metadata-dependent literature queries without hardcoded cohorts."""

import json
import re


def one_hop_relationship_excerpt(metadata, max_chars=5000):
    """Return only one-hop relationship records within a strict char budget."""
    relationships = []
    used_chars = 0
    for line in metadata.splitlines():
        try:
            record = json.loads(line)
        except (TypeError, json.JSONDecodeError):
            continue
        structured = False
        if isinstance(record, dict):
            relationship = record.get("relationship")
            if isinstance(relationship, str) and relationship.strip():
                excerpt = relationship.strip()
            elif isinstance(record.get("source"), dict) and isinstance(record.get("target"), dict):
                # Preserve endpoint labels and edge type so the extractor can
                # distinguish collection membership from unrelated nodes.
                excerpt = line.strip()
                structured = True
            else:
                continue
        else:
            continue
        separator = "\n" if relationships else ""
        remaining = max_chars - used_chars - len(separator)
        if remaining <= 0:
            break
        if structured and len(excerpt) > remaining:
            break
        excerpt = excerpt[:remaining]
        relationships.append(excerpt)
        used_chars += len(separator) + len(excerpt)
    return "\n".join(relationships)


def matching_subjects(text, subjects):
    """Return exact, case-insensitive subject names present in a passage."""
    return [
        subject for subject in subjects
        if isinstance(subject, str) and subject.strip()
        and re.search(r"(?<!\w)" + re.escape(subject.strip()) + r"(?!\w)", text, re.I)
    ]


def compact_metadata_context(metadata, max_chars=5000):
    """Pack complete deduplicated identity records in retrieval order."""
    records, seen = [], set()
    fields = {"id", "genomeid", "accessionname", "displayname", "name", "type",
              "categories", "category", "species", "collection", "collectionname",
              "description", "datasetname", "label", "cultivar", "assemblyname"}
    for line in metadata.splitlines():
        try:
            record = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(record, dict) or "relationship" in record:
            continue
        compact = {key: value for key, value in record.items()
                   if key.lower() in fields and value not in (None, "", [])}
        if not compact:
            continue
        # Keep dataset labels attached to their original record.
        encoded = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
        if encoded not in seen:
            records.append(encoded)
            seen.add(encoded)
    # Replace repeated alignment identities with their genome record only when
    # that genome independently carries the same collection labels.
    decoded = [json.loads(record) for record in records]
    genomes = {row.get("id"): row for row in decoded
               if str(row.get("type", "")).casefold() == "genome" and row.get("id")}
    ordered, used_identities = [], set()
    for row in decoded:
        genome = genomes.get(row.get("genomeId"))
        if genome and row.get("categories") == genome.get("categories"):
            row = genome
        elif genome:
            # Preserve collection labels on the dataset that actually owns them.
            # Joining exact IDs makes the accession visible beside that evidence
            # without assigning the dataset's categories to a reference genome.
            identity = {key: genome[key] for key in ("id", "accessionName", "displayName", "type")
                        if key in genome}
            row = {**row, "linked_genome": identity}
        encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        if encoded not in used_identities:
            ordered.append(encoded)
            used_identities.add(encoded)
    records = ordered
    packed, used = [], 0
    for record in records:
        cost = len(record) + bool(packed)
        if used + cost <= max_chars:
            packed.append(record)
            used += cost
    return "\n".join(packed), {"identity_records_available": len(records),
                                "identity_records_in_excerpt": len(packed),
                                "identity_records_omitted": len(records)-len(packed),
                                "linked_identity_records_in_excerpt": sum('"linked_genome":' in row for row in packed)}


def ground_literature_queries(question, queries, metadata, plan, extraction_prompt,
                              invoke, diagnostics, logger, expanded_question=None):
    """Extract subjects from the ordered metadata context, capped at 5,000 chars."""
    metadata_excerpt, compact_stats = compact_metadata_context(metadata)
    diagnostics.update(compact_stats)
    logger.info("Metadata extraction: %d exact dataset-to-genome identity joins in packed context",
                compact_stats["linked_identity_records_in_excerpt"])
    logger.info("Metadata identity compaction: %d records available, %d packed, %d omitted (5,000-char limit)",
                compact_stats["identity_records_available"], compact_stats["identity_records_in_excerpt"],
                compact_stats["identity_records_omitted"])
    diagnostics.update({
        "metadata_chars_scanned": len(metadata_excerpt),
        "metadata_excerpt": metadata_excerpt,
        "metadata_extraction_source": "compacted_ordered_metadata_context",
        "entities": [],
    })
    if not metadata_excerpt.strip():
        logger.info("Metadata extracted entities: [] (metadata context unavailable)")
        return list(queries), question
    logger.info(
        "Compacted ordered metadata context supplied to entity extraction (%d chars):\n%s",
        len(metadata_excerpt), metadata_excerpt,
    )
    prompt = extraction_prompt.rstrip() + "\n" + json.dumps(
        {"question": question, "dependency": plan,
         "metadata_excerpt": metadata_excerpt}, ensure_ascii=False
    )
    raw = invoke(prompt, diagnostics)
    data = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip()))
    rows = data.get("entities", [])
    if not isinstance(rows, list):
        raise ValueError("Metadata entity extraction did not return a list")
    names, seen = [], set()
    specific_queries = []
    rejected = 0
    for row in rows:
        if not isinstance(row, dict):
            rejected += 1
            continue
        name, evidence = row.get("name"), row.get("evidence")
        if not isinstance(name, str) or not isinstance(evidence, str):
            rejected += 1
            continue
        name, evidence = name.strip(), evidence.strip()
        if (not name or not evidence or name not in evidence or evidence not in metadata_excerpt
                or name.casefold() in seen):
            rejected += 1
            continue
        seen.add(name.casefold())
        names.append(name)
        direct_query = row.get("literature_query")
        if not isinstance(direct_query, str) or not matching_subjects(direct_query, [name]):
            direct_query = f"For {name} only: {question} Evaluate the requested property for this subject."
        specific_queries.append(direct_query.strip())
    diagnostics["entities"] = names
    diagnostics["entity_extraction_returned_count"] = len(rows)
    diagnostics["entity_extraction_rejected_count"] = rejected
    logger.info("Metadata entity extraction: %d proposals, %d rejected by evidence validation; "
                "%d ordered metadata chars; accepted=%s",
                len(rows), rejected, len(metadata_excerpt),
                json.dumps(names, ensure_ascii=False))
    if not names:
        return list(queries), question
    scope = (f"{expanded_question or question}\n"
             f"Metadata-resolved search subjects: {', '.join(names)}. "
             "Prioritize evidence about these subjects. Metadata membership does not "
             "establish the requested biological property. This list may be incomplete.")
    # Each subject gets a direct search; the retrieval executor runs these concurrently.
    grounded_queries = list(dict.fromkeys(specific_queries))
    diagnostics["grounded_question"] = scope
    diagnostics["retrieval_queries"] = grounded_queries
    for query in specific_queries:
        logger.info("Metadata-grounded literature query: %s", query)
    return grounded_queries, scope
