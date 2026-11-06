"""Small, dependency-free utilities shared by the runnable pipelines."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

PROMPTS = Path(__file__).resolve().parent / "prompts"
VERSION_NAMES = {"full": "v0", "next": "v1_5", "v0": "v0", "v1_5": "v1_5"}


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def write_csv(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def select_csvs(input_dir: Path, mask: Path | None = None, limit: int = 0):
    paths = sorted(input_dir.glob("*.csv"))
    if mask:
        prefixes = {line.strip() for line in mask.read_text().splitlines() if line.strip()}
        paths = [path for path in paths if path.stem in prefixes]
        missing = prefixes - {path.stem for path in paths}
        if missing:
            raise ValueError(f"{len(missing)} mask sessions have no CSV in {input_dir}")
    if limit:
        paths = paths[:limit]
    if not paths:
        raise ValueError(f"No input CSVs selected in {input_dir}")
    return paths
