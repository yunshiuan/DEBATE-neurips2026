#!/usr/bin/env python3
"""Update the dataset's Croissant metadata from a verified human-view export.

Run from the source checkout after ``debate export-human``::

    python scripts/update_human_view_metadata.py \
        --views-dir /path/to/DEBATE_LLM/human_views \
        --croissant /path/to/DEBATE_LLM/croissant-rai.json

This only updates local metadata. It does not upload files or change the
reproduction archives, the code's pinned release, or existing RAI statements.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile


VIEW_FILES = (
    "human_events.csv",
    "paper_human_evaluation.csv",
    "human_opinion_pairs.csv",
)
PUBLIC_BASE = "https://huggingface.co/datasets/seantw/DEBATE_LLM/resolve/main/human_views"
DATA_TYPES = {
    "string": "sc:Text",
    "integer": "cr:Int64",
    "number": "cr:Float64",
    "boolean": "sc:Boolean",
}
LEGACY_DESCRIPTIONS = {
    "text": "Recorded human event text or survey response.",
    "sliderValue": (
        "For initial_opinion and post_opinion records, the participant's recorded "
        "1–6 slider rating, from Certainly disagree to Certainly agree. This is "
        "distinct from a stance annotation of the participant's text. The human "
        "views name this field self_reported_opinion and include submission flags."
    ),
    "agreement_level": (
        "Legacy model-generated agreement text copied into this export; this is "
        "not a human self-report or an annotation of the human text. The human "
        "views omit it and derive self_reported_opinion_label from the recorded "
        "human slider rating instead."
    ),
    "validity": (
        "Legacy automatic validity label. The paper_human_evaluation view exposes "
        "human_validity using the paper's stage-specific validity rules."
    ),
}


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def human_view_entries(views_dir: Path) -> tuple[list[dict], list[dict]]:
    """Validate the export and generate its FileObjects and RecordSets."""
    schema = read_json(views_dir / "schema.json")
    manifest = read_json(views_dir / "manifest.json")
    if schema.get("schema_version") != 1 or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported human-view schema or manifest version")
    files = {entry["path"]: entry for entry in manifest["files"]}
    if len(files) != len(manifest["files"]):
        raise ValueError("Duplicate file paths in the human-view manifest")
    schema_info = files["schema.json"]
    schema_path = views_dir / "schema.json"
    if schema_info["size_bytes"] != schema_path.stat().st_size or schema_info["sha256"] != sha256(schema_path):
        raise ValueError("Schema does not match the manifest: schema.json")
    if set(schema["views"]) != set(VIEW_FILES):
        raise ValueError(f"Expected exactly these view schemas: {VIEW_FILES}")

    distributions = []
    record_sets = []
    for filename in VIEW_FILES:
        path = views_dir / filename
        view = schema["views"][filename]
        file_info = files[filename]
        if file_info["size_bytes"] != path.stat().st_size:
            raise ValueError(f"Size does not match the manifest: {filename}")
        if file_info["sha256"] != sha256(path):
            raise ValueError(f"SHA-256 does not match the manifest: {filename}")
        columns = [field["name"] for field in view["fields"]]
        if not columns or len(columns) != len(set(columns)):
            raise ValueError(f"Empty or repeated schema field names: {filename}")
        with path.open(encoding="utf-8", newline="") as handle:
            if next(csv.reader(handle), None) != columns:
                raise ValueError(f"CSV header does not match the schema: {filename}")
        primary_key = view["primary_key"]
        if not primary_key or not set(primary_key).issubset(columns):
            raise ValueError(f"Invalid primary key: {filename}")

        file_id = f"human_views/{filename}"
        record_set_id = Path(filename).stem
        distributions.append({
            "@type": "cr:FileObject",
            "@id": file_id,
            "name": filename,
            "description": view["description"],
            "contentUrl": f"{PUBLIC_BASE}/{filename}",
            "encodingFormat": "text/csv",
            "contentSize": str(file_info["size_bytes"]),
            "sha256": file_info["sha256"],
        })
        fields = []
        for field in view["fields"]:
            name = field["name"]
            field_type = field["type"]
            description = field["description"]
            if field_type not in DATA_TYPES:
                raise ValueError(f"Unsupported type {field_type!r}: {filename}/{name}")
            if not isinstance(field["nullable"], bool):
                raise ValueError(f"Nullable must be boolean: {filename}/{name}")
            if not isinstance(description, str) or not description.strip():
                raise ValueError(f"Missing field description: {filename}/{name}")
            if field["nullable"]:
                description += " An empty CSV cell denotes a missing value."
            fields.append({
                "@type": "cr:Field",
                "@id": f"{record_set_id}/{name}",
                "name": name,
                "description": description,
                "dataType": DATA_TYPES[field_type],
                "source": {
                    "fileObject": {"@id": file_id},
                    "extract": {"column": name},
                },
            })
        record_sets.append({
            "@type": "cr:RecordSet",
            "@id": record_set_id,
            "name": record_set_id,
            "description": view["description"],
            "key": [{"@id": f"{record_set_id}/{name}"} for name in primary_key],
            "field": fields,
        })
    return distributions, record_sets


def update_metadata(metadata: dict, views_dir: Path) -> dict:
    """Replace only the generated human-view entries and clarify legacy fields."""
    distributions, record_sets = human_view_entries(views_dir)
    updated = copy.deepcopy(metadata)
    # Without this alias, readers interpret the declared v1.1 metadata as v0.8
    # and silently miss the RecordSets. v1.1 also defines FileObjects/FileSets
    # and the includes property in the Croissant namespace.
    updated["@context"]["conformsTo"] = "dct:conformsTo"
    updated["@context"]["includes"] = "cr:includes"
    for entry in updated["distribution"]:
        if entry.get("@type") in ("https://schema.org/FileObject", "sc:FileObject"):
            entry["@type"] = "cr:FileObject"
        elif entry.get("@type") in ("https://schema.org/FileSet", "sc:FileSet"):
            entry["@type"] = "cr:FileSet"
    distribution_ids = {entry["@id"] for entry in distributions}
    record_set_ids = {entry["@id"] for entry in record_sets}
    updated["distribution"] = [
        entry for entry in updated["distribution"]
        if entry.get("@id") not in distribution_ids
    ] + distributions
    updated["recordSet"] = [
        entry for entry in updated["recordSet"]
        if entry.get("@id") not in record_set_ids
    ] + record_sets
    for record_set in updated["recordSet"]:
        if record_set.get("@id") != "default":
            continue
        for field in record_set["field"]:
            column = field.get("source", {}).get("extract", {}).get("column")
            if column in LEGACY_DESCRIPTIONS:
                field["description"] = LEGACY_DESCRIPTIONS[column]
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views-dir", required=True, type=Path)
    parser.add_argument("--croissant", required=True, type=Path)
    parser.add_argument("--check", action="store_true", help="Validate without writing; fail if an update is needed")
    args = parser.parse_args()
    try:
        original = read_json(args.croissant)
        updated = update_metadata(original, args.views_dir)
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Cannot update human-view metadata: {error}\n")
    if args.check:
        if original != updated:
            parser.exit(1, "Croissant human-view metadata needs updating.\n")
        print("Human-view files, schema, and Croissant metadata are consistent.")
        return
    if original == updated:
        print("Croissant human-view metadata is already current.")
        return
    # Replace atomically only after all source files and schema entries validate.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.croissant.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(updated, indent=2, ensure_ascii=False) + "\n")
    try:
        temporary.replace(args.croissant)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Updated {args.croissant} with {len(VIEW_FILES)} human-data views.")


if __name__ == "__main__":
    main()
