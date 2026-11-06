#!/usr/bin/env python3

from __future__ import annotations

import csv
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterable

from neurips_2026.sean_reproduce.base_model_v3_style import CANONICAL_V3_HUMAN_SOURCE_MODEL

from generate_v3_degroot_abm_summary_common import (
    build_degroot_proxy_rows,
    load_human_rows_by_prefix,
)
from generate_v3_degroot_abm_report import build_degroot_start_states


@dataclass(frozen=True)
class ProxyEvalModelSpec:
    model_dir: str
    alpha_public: float
    alpha_private: float
    fit_scope: str = ""


def _stringify_row(row: dict[str, object]) -> dict[str, str]:
    return {
        str(key): "" if value is None else str(value)
        for key, value in row.items()
    }


def _materialize_validity_columns(row: dict[str, str], *, target_kind: str) -> dict[str, str]:
    materialized = dict(row)
    effective_validity = str(materialized.get("effective_validity") or "").strip().upper()
    if not effective_validity:
        return materialized

    candidates = ("human_validity", "validity") if target_kind == "human" else ("llm_validity", "validity")
    for candidate in candidates:
        if candidate in materialized:
            materialized[candidate] = effective_validity
            break
    else:
        materialized[candidates[0]] = effective_validity
    return materialized


def prepare_materialized_rows(rows: list[dict[str, object]], *, target_kind: str) -> list[dict[str, str]]:
    return [
        _materialize_validity_columns(_stringify_row(row), target_kind=target_kind)
        for row in rows
    ]


def write_materialized_csv(path: Path, rows: list[dict[str, object]], *, target_kind: str) -> None:
    materialized_rows = prepare_materialized_rows(rows, target_kind=target_kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in materialized_rows:
        for key in row:
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["event_type"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized_rows)


def resolve_existing_model_csv(
    *,
    roots: list[Path],
    data_prefix: str,
    model_dir: str,
    filename: str,
) -> Path | None:
    for root in roots:
        candidate = root / data_prefix / model_dir / filename
        if candidate.exists():
            return candidate
    return None


def _normalize_prefix_model_specs(
    prefixes: Iterable[str],
    *,
    model_specs: list[ProxyEvalModelSpec] | None,
    prefix_model_specs: dict[str, list[ProxyEvalModelSpec]] | None,
) -> dict[str, list[ProxyEvalModelSpec]]:
    if prefix_model_specs is not None:
        return {
            str(data_prefix): list(prefix_model_specs.get(str(data_prefix), []))
            for data_prefix in prefixes
        }
    shared_specs = list(model_specs or [])
    return {
        str(data_prefix): list(shared_specs)
        for data_prefix in prefixes
    }


def filter_valid_degroot_rows_by_prefix(
    rows_by_prefix: dict[str, list[dict[str, str]]],
    *,
    likert_col: str,
) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, str]]]:
    valid_rows: dict[str, list[dict[str, str]]] = {}
    skipped_rows: list[dict[str, str]] = []
    for data_prefix, human_rows in rows_by_prefix.items():
        try:
            build_degroot_start_states(
                human_rows,
                data_prefix=data_prefix,
                likert_col=likert_col,
            )
        except ValueError as exc:
            skipped_rows.append(
                {
                    "data_prefix": data_prefix,
                    "reason": str(exc),
                }
            )
            continue
        valid_rows[data_prefix] = human_rows
    return valid_rows, skipped_rows


def load_valid_human_rows_by_prefix(
    *,
    roots: list[Path],
    prefixes: list[str],
    human_filename: str,
    likert_col: str,
    initial_post_sidecar_roots: list[Path],
    tweet_sidecar_roots: list[Path],
) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, str]]]:
    rows_by_prefix = load_human_rows_by_prefix(
        roots=roots,
        prefixes=prefixes,
        human_filename=human_filename,
        likert_col=likert_col,
        initial_post_sidecar_roots=initial_post_sidecar_roots,
        tweet_sidecar_roots=tweet_sidecar_roots,
    )
    return filter_valid_degroot_rows_by_prefix(rows_by_prefix, likert_col=likert_col)


def _resolve_worker_count(workers: int | None, prefix_count: int) -> int:
    if prefix_count <= 1:
        return 1
    if workers is None:
        return max(1, min(prefix_count, os.cpu_count() or 1, 8))
    return max(1, min(prefix_count, workers))


def _materialize_prefix(
    data_prefix: str,
    *,
    rows_by_prefix: dict[str, list[dict[str, str]]],
    roots: list[Path],
    human_filename: str,
    llm_filename: str,
    likert_col: str,
    output_root: Path,
    prefix_specs: dict[str, list[ProxyEvalModelSpec]],
    passthrough_model_dirs: list[str] | None,
    human_model_dir: str,
) -> tuple[list[dict[str, str]], int, int]:
    human_rows = rows_by_prefix.get(data_prefix)
    if human_rows is None:
        return [], 0, 0

    human_path = output_root / data_prefix / human_model_dir / human_filename
    write_materialized_csv(human_path, human_rows, target_kind="human")

    manifest_rows: list[dict[str, str]] = []
    generated_model_row_count = 0
    copied_passthrough_count = 0

    for spec in prefix_specs.get(data_prefix, []):
        proxy_rows = build_degroot_proxy_rows(
            human_rows,
            data_prefix=data_prefix,
            likert_col=likert_col,
            alpha_public=spec.alpha_public,
            alpha_private=spec.alpha_private,
        )
        proxy_path = output_root / data_prefix / spec.model_dir / llm_filename
        write_materialized_csv(proxy_path, proxy_rows, target_kind="llm")
        generated_model_row_count += 1
        manifest_rows.append(
            {
                "data_prefix": data_prefix,
                "model_dir": spec.model_dir,
                "alpha_public": f"{spec.alpha_public:.12f}",
                "alpha_private": f"{spec.alpha_private:.12f}",
                "fit_scope": spec.fit_scope,
                "human_csv": str(human_path),
                "llm_csv": str(proxy_path),
            }
        )

    for model_dir in passthrough_model_dirs or []:
        source_path = resolve_existing_model_csv(
            roots=roots,
            data_prefix=data_prefix,
            model_dir=model_dir,
            filename=llm_filename,
        )
        if source_path is None:
            continue
        target_path = output_root / data_prefix / model_dir / llm_filename
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target_path)
        copied_passthrough_count += 1
        manifest_rows.append(
            {
                "data_prefix": data_prefix,
                "model_dir": model_dir,
                "alpha_public": "",
                "alpha_private": "",
                "fit_scope": "passthrough",
                "human_csv": str(human_path),
                "llm_csv": str(target_path),
            }
        )

    return manifest_rows, generated_model_row_count, copied_passthrough_count


def materialize_proxy_eval_tree(
    *,
    roots: list[Path],
    prefixes: list[str],
    human_filename: str,
    llm_filename: str,
    likert_col: str,
    initial_post_sidecar_roots: list[Path],
    tweet_sidecar_roots: list[Path],
    output_root: Path,
    model_specs: list[ProxyEvalModelSpec] | None = None,
    prefix_model_specs: dict[str, list[ProxyEvalModelSpec]] | None = None,
    passthrough_model_dirs: list[str] | None = None,
    human_model_dir: str = CANONICAL_V3_HUMAN_SOURCE_MODEL,
    write_manifest: bool = True,
    workers: int | None = None,
) -> dict[str, Path | int]:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    rows_by_prefix = load_human_rows_by_prefix(
        roots=roots,
        prefixes=prefixes,
        human_filename=human_filename,
        likert_col=likert_col,
        initial_post_sidecar_roots=initial_post_sidecar_roots,
        tweet_sidecar_roots=tweet_sidecar_roots,
    )
    rows_by_prefix, skipped_rows = filter_valid_degroot_rows_by_prefix(rows_by_prefix, likert_col=likert_col)
    prefix_specs = _normalize_prefix_model_specs(
        prefixes,
        model_specs=model_specs,
        prefix_model_specs=prefix_model_specs,
    )

    manifest_rows: list[dict[str, str]] = []
    generated_prefix_count = 0
    generated_model_row_count = 0
    copied_passthrough_count = 0
    available_prefixes = [data_prefix for data_prefix in prefixes if rows_by_prefix.get(data_prefix) is not None]
    generated_prefix_count = len(available_prefixes)
    worker_count = _resolve_worker_count(workers, len(available_prefixes))
    materialize_prefix = partial(
        _materialize_prefix,
        rows_by_prefix=rows_by_prefix,
        roots=roots,
        human_filename=human_filename,
        llm_filename=llm_filename,
        likert_col=likert_col,
        output_root=output_root,
        prefix_specs=prefix_specs,
        passthrough_model_dirs=passthrough_model_dirs,
        human_model_dir=human_model_dir,
    )

    if worker_count == 1:
        prefix_results = map(materialize_prefix, available_prefixes)
    else:
        executor = ThreadPoolExecutor(max_workers=worker_count)
        prefix_results = executor.map(materialize_prefix, available_prefixes)

    try:
        for prefix_manifest_rows, prefix_generated_model_row_count, prefix_copied_passthrough_count in prefix_results:
            manifest_rows.extend(prefix_manifest_rows)
            generated_model_row_count += prefix_generated_model_row_count
            copied_passthrough_count += prefix_copied_passthrough_count
    finally:
        if worker_count != 1:
            executor.shutdown(wait=True)

    manifest_path = output_root / "degroot_proxy_manifest.csv"
    skipped_path = output_root / "degroot_proxy_skipped_prefixes.csv"
    if write_manifest:
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "data_prefix",
                    "model_dir",
                    "alpha_public",
                    "alpha_private",
                    "fit_scope",
                    "human_csv",
                    "llm_csv",
                ],
            )
            writer.writeheader()
            writer.writerows(manifest_rows)
        with skipped_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["data_prefix", "reason"])
            writer.writeheader()
            writer.writerows(skipped_rows)

    return {
        "output_root": output_root,
        "manifest_path": manifest_path,
        "skipped_path": skipped_path,
        "generated_prefix_count": generated_prefix_count,
        "generated_model_row_count": generated_model_row_count,
        "copied_passthrough_count": copied_passthrough_count,
    }


__all__ = [
    "ProxyEvalModelSpec",
    "filter_valid_degroot_rows_by_prefix",
    "load_valid_human_rows_by_prefix",
    "materialize_proxy_eval_tree",
    "prepare_materialized_rows",
    "resolve_existing_model_csv",
    "write_materialized_csv",
]