"""Select actual metadata category values before fetching member nodes."""
import json
import re
from metadata_query_grounding import compact_metadata_context


def category_metadata_context(graph, question, plan, invoke, prompt, diagnostics, logger):
    inventory = graph.query(
        "MATCH (n:MetadataGraph) WHERE n.categories IS NOT NULL "
        "RETURN DISTINCT n.categories AS categories"
    )
    values = {}
    for row in inventory:
        raw = row['categories']
        parts = raw if isinstance(raw, list) else [raw]
        for part in parts:
            if not isinstance(part, str):
                continue
            for label in part.split(';'):
                label = label.strip()
                if label and label != '.':
                    values.setdefault(label, []).append(raw)
    labels = sorted(values, key=str.casefold)
    diagnostics['available_categories'] = labels
    if not labels:
        diagnostics['category_status'] = 'no_categories_available'
        return None
    selection_trace = {}
    diagnostics['category_selection'] = selection_trace
    raw_response = invoke(prompt.rstrip() + '\n' + json.dumps({
        'question': question, 'metadata_dependency': plan, 'available_categories': labels
    }, ensure_ascii=False), selection_trace)
    parsed = json.loads(re.sub(r'^```(?:json)?\s*|\s*```$', '', raw_response.strip()))
    selected = parsed.get('categories')
    if not isinstance(selected, list) or any(not isinstance(label, str) or label not in values for label in selected):
        raise ValueError('Category selector must return exact available category names')
    selected = list(dict.fromkeys(selected))
    diagnostics['selected_categories'] = selected
    diagnostics['category_reason'] = parsed.get('reason', '')
    logger.info('Metadata category selection: %s; reason: %s', selected, diagnostics['category_reason'])
    if not selected:
        diagnostics['category_status'] = 'no_matching_category'
        return None
    raw_values = []
    for label in selected:
        for raw in values[label]:
            if raw not in raw_values:
                raw_values.append(raw)
    rows = graph.query(
        "MATCH (n:MetadataGraph) WHERE n.categories IN $category_values "
        "RETURN n {.*, embedding: null} AS props "
        "ORDER BY coalesce(n.id, ''), coalesce(n.displayName, ''), elementId(n)",
        params={'category_values': raw_values},
    )
    records = [{key: value for key, value in row['props'].items()
                if key != 'embedding' and value is not None} for row in rows]
    diagnostics['category_node_count'] = len(records)
    diagnostics['category_status'] = 'selected'
    # Fetch every explicitly tagged node before compaction, without hybrid top-K
    # or neighbor expansion. Keep a compact representation for the answer too.
    serialized = '\n'.join(json.dumps(record, ensure_ascii=False) for record in records)
    context, stats = compact_metadata_context(serialized, max_chars=max(len(serialized), 1))
    diagnostics['category_identity_records'] = stats['identity_records_available']
    logger.info('Metadata category fetch: %d explicitly tagged nodes, %d compact chars', len(records), len(context))
    return '### Metadata Graph (explicit category membership):\n' + context
