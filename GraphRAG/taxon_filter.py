"""Small, conservative crop/species matching helpers for GraphRAG retrieval."""

from __future__ import annotations

import re
from typing import Any


TAXON_ALIASES = {
    "wheat": (
        "wheat",
        "bread wheat",
        "durum wheat",
        "triticum aestivum",
        "triticum durum",
        "triticum turgidum",
    ),
    "barley": ("barley", "hordeum vulgare"),
    "oat": ("oat", "oats", "avena sativa"),
    "maize": ("maize", "corn", "zea mays"),
    "chickpea": ("chickpea", "chick pea", "cicer arietinum"),
    "lentil": ("lentil", "lentils", "lens culinaris"),
    "field pea": ("field pea", "field peas", "pea", "peas", "pisum sativum"),
    "faba bean": ("faba bean", "faba beans", "vicia faba"),
    "canola": ("canola", "rapeseed", "oilseed rape", "brassica napus"),
    "rye": ("rye", "secale cereale"),
    "sorghum": ("sorghum", "sorghum bicolor"),
    "soybean": ("soybean", "soybeans", "soy", "glycine max"),
    "mungbean": ("mungbean", "mung bean", "vigna radiata"),
}


def _normalise(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


def _contains_alias(text: str, alias: str) -> bool:
    normalised_text = f" {_normalise(text)} "
    normalised_alias = f" {_normalise(alias)} "
    return normalised_alias in normalised_text


def _canonical_taxa(value: Any) -> set[str]:
    if value is None:
        return set()
    return {
        canonical
        for canonical, aliases in TAXON_ALIASES.items()
        if any(_contains_alias(value, alias) for alias in aliases)
        or (canonical == "field pea" and _normalise(value) == "fieldpea")
    }


def resolve_taxon_filter(
    question: str,
    detected_species: str = "",
    crop_overrides: list[str] | None = None,
) -> dict[str, Any] | None:
    """Resolve recognized crop names from the question/classifier or overrides."""
    if crop_overrides:
        targets = {
            canonical
            for value in crop_overrides
            for canonical in _canonical_taxa(value)
        }
        # Permit canonical override values even if their spelling is not an alias.
        targets.update(
            value.casefold()
            for value in crop_overrides
            if value.casefold() in TAXON_ALIASES
        )
    else:
        targets = _canonical_taxa(question) | _canonical_taxa(detected_species)
    if not targets:
        return None
    aliases = sorted(
        {alias for target in targets for alias in TAXON_ALIASES[target]}
    )
    return {"canonical_crops": sorted(targets), "aliases": aliases}


def metadata_matches_taxon(
    properties: dict[str, Any],
    taxon_filter: dict[str, Any],
    *,
    allow_unclassified: bool = False,
) -> bool:
    """Match known metadata by species first, then crop; reject known conflicts."""
    targets = set(taxon_filter["canonical_crops"])
    crop_taxa = _canonical_taxa(properties.get("crop"))
    species_taxa = _canonical_taxa(properties.get("species"))
    if (crop_taxa | species_taxa) & targets:
        return True
    if crop_taxa or species_taxa:
        return False
    return allow_unclassified


def text_is_clearly_other_taxon(
    text: Any, taxon_filter: dict[str, Any] | None
) -> bool:
    """Reject text only when it names other known taxa but none of the targets.

    Mixed-species passages and passages with no recognizable taxonomy remain
    eligible, since a mere mention of another crop is not enough to classify
    the chunk as irrelevant.
    """
    if not taxon_filter or not text:
        return False
    mentioned = _canonical_taxa(text)
    targets = set(taxon_filter.get("canonical_crops", []))
    return bool(mentioned) and not bool(mentioned & targets)


def taxon_regex(taxon_filter: dict[str, Any]) -> str:
    """Create a word-bounded Java regex for Cypher neighbor filtering."""
    alternatives = []
    for alias in taxon_filter["aliases"]:
        phrase = r"\s+".join(re.escape(word) for word in _normalise(alias).split())
        alternatives.append(phrase)
    return r"(?i)(^|[^a-z0-9])(" + "|".join(alternatives) + r")($|[^a-z0-9])"
