#!/usr/bin/env python3
"""Collate one-line field/value input into type-specific CSV files."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


TYPE_HEADERS: dict[str, list[str]] = {
    "Alignment": [
        "_id", "shortName", "displayName", "type", "tags", "Crop", "Species",
        "Data Source", "Licensing of original data", "Curator", "Publication",
        "Comments", "Categories", "Genome _id", "Alignment type", "Marker type", "",
    ],
    "Genetic Map": [
        "_id", "shortName", "displayName", "type", "tags", "Crop", "species",
        "Data Source", "Licensing of original data", "Curator", "Publication",
        "Comments", "Categories", "Parent names", "Marker type", "Population type",
    ],
    "Genome": [
        "_id", "shortName", "displayName", "type", "tags", "Crop", "species",
        "Data Source", "Licensing of original data", "Curator", "Publication",
        "Comments", "Categories", "Accession name", "Project ID", "EBI-ENA ID",
        "PanBARLEX name",
    ],
    "QTL": [
        "_id", "shortName", "displayName", "type", "tags", "Crop", "Species",
        "Data Source", "Licensing of original data", "Curator", "Publication",
        "Comments", "Categories", "Defined in", "", "",
    ],
    "VCF": [
        "_id", "shortName", "displayName", "type", "tags", "Crop", "species",
        "Data Source", "Licensing of original data", "Curator", "Publication",
        "Comments", "Categories", "platform", "Marker type", "Project ID",
    ],
}

ALWAYS_EMPTY_FILES = (
    "Ontology table.csv",
    "DatasetAccession.csv",
    "Project.csv",
    "Curator.csv",
)

class InputFormatError(ValueError):
    """Raised when the input does not follow the expected field:value format."""


def parse_relations(text: str) -> dict[str, str]:
    """Parse one ``field_name: value`` relation per non-empty line."""
    fields: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue

        if ":" not in line:
            raise InputFormatError(
                f"line {line_number} does not contain ':'"
            )

        field_name, value = (part.strip() for part in line.split(":", 1))

        if not field_name:
            raise InputFormatError(f"empty field name on line {line_number}")
        if field_name in fields:
            raise InputFormatError(
                f"duplicate field {field_name!r} on line {line_number}"
            )
        fields[field_name] = value

    return fields


def field_value(fields: dict[str, str], header: str) -> str:
    """Find an input field exactly, then fall back to case-insensitive matching."""
    if not header:
        return ""
    if header in fields:
        return fields[header]

    folded_header = header.casefold()
    matches = [value for name, value in fields.items() if name.casefold() == folded_header]
    if len(matches) > 1:
        raise InputFormatError(
            f"multiple input fields match output column {header!r} ignoring case"
        )
    return matches[0] if matches else ""


def write_outputs(fields: dict[str, str], output_dir: Path) -> None:
    input_type = field_value(fields, "type")
    if input_type not in TYPE_HEADERS:
        choices = ", ".join(TYPE_HEADERS)
        raise InputFormatError(
            f"unknown or missing type {input_type!r}; expected one of: {choices}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    for type_name, headers in TYPE_HEADERS.items():
        output_path = output_dir / f"{type_name}.csv"
        with output_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(headers)
            if type_name == input_type:
                writer.writerow([field_value(fields, header) for header in headers])

    for filename in ALWAYS_EMPTY_FILES:
        (output_dir / filename).write_bytes(b"")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read one-line field:value relations and create type-specific CSV files."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="input text file (reads standard input when omitted)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("."),
        help="directory for generated CSV files (default: current directory)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.input is None:
            text = sys.stdin.read()
        else:
            text = args.input.read_text(encoding="utf-8-sig")
        fields = parse_relations(text)
        write_outputs(fields, args.output_dir)
    except (OSError, InputFormatError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
