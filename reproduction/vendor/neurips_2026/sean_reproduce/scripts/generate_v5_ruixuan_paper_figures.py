#!/usr/bin/env python3
"""Rebuild the paper's analytical figure family from the 2026-07-16 v5 BD run.

The run contains two split-specific SFT models and both v0 (full dynamics) and
v1_5 (next-message) simulations. Per the SFT-only-update authorization, it
reuses the canonical eval-v3 base-model series while keeping v5 SFT series,
masks, and validity labels distinct and explicitly documented.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import spearmanr


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
RADAR_DIR = REPO_ROOT / "neurips_2026" / "chengtao" / "src" / "radar"
SCRIPT_DIR = SCRIPT_PATH.parent
for import_path in (REPO_ROOT, RADAR_DIR, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from generate_plots import (  # noqa: E402
    DEFAULT_UTTERANCE_EVENT_TYPES,
    add_v2_bootstrap_cis_to_raw_rows,
    build_model_aggregates,
    build_profile,
    build_raw_rows_from_aggregates,
    compute_prefix_model,
    compute_prefix_model_from_rows,
    merge_prefix_result_into_aggregate,
    parse_round,
    participant_id,
    read_csv_rows,
    safe_mean,
    safe_pstdev,
)
from generate_plots import FINE_TUNED_DISPLAY_MAP, MODEL_COLOR_MAP  # noqa: E402
from group_opinion_convergence import (  # noqa: E402
    compute_consensus_floor_lists,
    compute_half_sd_reduction_lists,
    strict_private_participants,
    strict_public_participants,
)
from generate_v3_degroot_abm_report import fit_best_alphas_from_human_rows  # noqa: E402
from generate_v3_degroot_abm_summary_common import build_degroot_proxy_rows  # noqa: E402
from neurips_2026.sean_reproduce.base_model_v3_style import (  # noqa: E402
    BASE_MODEL_V3_COLOR_MAP,
    BASE_MODEL_V3_DISPLAY_LABELS,
    BASE_MODEL_V4_LLM_ORDER,
)
import create_v3_report as histogram_style  # noqa: E402
import create_v4_abm_transition_publication_panel as transition_style  # noqa: E402
import create_v4_beyond_stance_exports as beyond_stance_style  # noqa: E402
import create_v4_scatter_post_train_publication_exports as scatter_style  # noqa: E402
import create_v4_scatter_post_train_spearman_rho_exports as spearman_ci_style  # noqa: E402
import create_v4_scatter_post_train_spearman_rho_publication_exports as spearman_style  # noqa: E402
import correlate_abm_summary_metrics as correlation_style  # noqa: E402
import create_v4_vs_v4b_abm_summary_post_training_publication_figures as bar_style  # noqa: E402
import generate_v3_depth_base_line_plots as line_style  # noqa: E402
from create_v4_vs_v4b_summary_base_model_report import build_page_figure  # noqa: E402


DEFAULT_RESULT_ROOT = Path("data/imports/result_ruixuan_20260716/eval_v5/human_llm")
DEFAULT_BASE_RESULT_ROOT = REPO_ROOT / "neurips_2026" / "chengtao" / "result" / "eval_v3" / "human_llm"
DEFAULT_DATA_ROOT = Path("data/imports/v5_bd_data_ruixuan_20260716")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "neurips_2026" / "sean_reproduce" / "v5_ruixuan"

PARTITION_OUTPUT_DIRS = {
    "group_train": "group_train_set",
    "group_test": "group_test_set",
    "topic_train": "topic_train_set",
    "topic_test": "topic_test_set",
}

GROUP_MODEL = "ft:Llama-3.1-8B-Instruct-SFT-20260608-bd-real:bd-group-5epochs"
TOPIC_MODEL = "ft:Llama-3.1-8B-Instruct-SFT-20260608-bd-real:bd-topic-5epochs"
BASE_MODELS = list(BASE_MODEL_V4_LLM_ORDER)
FOCUSED_BASE_MODEL = "Llama-3.1-8B-Instruct"
FOCUSED_OTHER_BASE_MODEL = "gpt-5-mini"
DEGROOT_MODEL = "DeGroot fit"
HUMAN_MODEL = "human"
LIKERT_COL = "likert_pred"

VERSION_SPECS = {
    "v0": {
        "label": "Full Dynamics Simulation",
        "human_filename": "opinion_human_memory_gpt-5-mini_v0.csv",
        "llm_filename": "opinion_llm_memory_gpt-5-mini_v0.csv",
        "color": "#1F77B4",
        "hatch": None,
        "marker": "o",
    },
    "v1_5": {
        "label": "Next Message Prediction",
        "human_filename": "opinion_human_memory_gpt-5-mini_v1_5.csv",
        "llm_filename": "opinion_llm_memory_gpt-5-mini_v1_5.csv",
        "color": "#D62728",
        "hatch": "///",
        "marker": "s",
    },
}

PARTITION_SPECS = {
    "group_train": {"mask": "group_train_prefixes.txt", "model": GROUP_MODEL, "fit_partition": "group_train"},
    "group_test": {"mask": "group_test_prefixes.txt", "model": GROUP_MODEL, "fit_partition": "group_train"},
    "topic_train": {"mask": "topic_train_prefixes.txt", "model": TOPIC_MODEL, "fit_partition": "topic_train"},
    "topic_test": {"mask": "topic_test_prefixes.txt", "model": TOPIC_MODEL, "fit_partition": "topic_train"},
}

MODEL_COLORS = {
    **BASE_MODEL_V3_COLOR_MAP,
    GROUP_MODEL: "#35689A",
    TOPIC_MODEL: "#35689A",
    DEGROOT_MODEL: "#7F7F7F",
    HUMAN_MODEL: "#222222",
}

# The v5 SFT identifiers are new, but the paper renderer's visual grammar is
# deliberately the same as the canonical v4 reproduction suite.  Register the
# new identifiers with the shared plotting helpers so they receive the Llama
# SFT highlight color and short publication label rather than a generic gray.
for _model, _color in MODEL_COLORS.items():
    MODEL_COLOR_MAP[_model] = _color
FINE_TUNED_DISPLAY_MAP[GROUP_MODEL] = "Llama SFT"
FINE_TUNED_DISPLAY_MAP[TOPIC_MODEL] = "Llama SFT"
PROFILE_CACHE: dict[tuple[str, str, str, str, str, float, float], dict[str, list[dict[str, str]]]] = {}
METRIC_GROUPS = (
    ("Single-Opinion Prediction Error", "public_stance_alignment_abs", "private_stance_alignment_abs"),
    ("Individual Opinion-Update Prediction Error", "public_stance_change_error_abs", "private_stance_change_error_abs"),
    ("Group Opinion Convergence Prediction Error", "public_opinion_convergence_error", "private_opinion_convergence_error"),
)


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    rows_list = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in rows_list:
            for field in row:
                if field not in fieldnames:
                    fieldnames.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or [])
        writer.writeheader()
        writer.writerows(rows_list)


def finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def sem(values: list[float]) -> float:
    if len(values) < 2:
        return math.nan
    mean_value = safe_mean(values)
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def model_label(model: str) -> str:
    if model == GROUP_MODEL:
        return "Llama SFT (group)"
    if model == TOPIC_MODEL:
        return "Llama SFT (topic)"
    if model == DEGROOT_MODEL:
        return "DeGroot fit"
    if model == HUMAN_MODEL:
        return "Human"
    if model in BASE_MODEL_V3_DISPLAY_LABELS:
        return BASE_MODEL_V3_DISPLAY_LABELS[model]
    return model


def publication_model_label(model: str) -> str:
    """Short labels used by the canonical final-paper figure layouts."""
    if model in {GROUP_MODEL, TOPIC_MODEL}:
        return "Llama SFT"
    if model == DEGROOT_MODEL:
        return "DeGroot fit"
    if model == HUMAN_MODEL:
        return "Human"
    return {
        "gpt-5-mini": "gpt-5-mini",
        "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
        "gemini-2.5-flash-lite": "gemini-2.5-flash",
        "Mistral-7B-Instruct-v0.3": "Mistral-7B",
        "Qwen2.5-32B-Instruct": "Qwen2.5-32B",
        "Llama-3.1-70B-Instruct": "Llama-3.1-70B",
        "Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    }.get(model, model_label(model))


def comparison_model_order(partition: str) -> list[str]:
    """Paper-style order: reusable baselines, the split-specific v5 SFT, then DeGroot."""
    return [*BASE_MODELS, str(PARTITION_SPECS[partition]["model"]), DEGROOT_MODEL]


def focused_model_order(partition: str, *, include_degroot: bool = True) -> list[str]:
    order = [HUMAN_MODEL, FOCUSED_BASE_MODEL, str(PARTITION_SPECS[partition]["model"])]
    if include_degroot:
        order.append(FOCUSED_OTHER_BASE_MODEL)
        order.append(DEGROOT_MODEL)
    return order


def short_partition(partition: str) -> str:
    split, state = partition.split("_", 1)
    return f"{split.title()} split — {state}"


def partition_output_dir(output_dir: Path, partition: str) -> Path:
    """Return the persistent figure directory for one v5 split/mask."""
    return output_dir / PARTITION_OUTPUT_DIRS[partition]


def read_prefixes(data_root: Path, partition: str) -> list[str]:
    mask_path = data_root / "data" / "sft_data" / "masks" / str(PARTITION_SPECS[partition]["mask"])
    with mask_path.open(encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def read_eval_rows(path: Path, *, result_root: Path) -> list[dict[str, str]]:
    return read_csv_rows(
        path,
        respect_validity_labels=True,
        likert_col=LIKERT_COL,
        data_roots=[result_root],
        initial_post_sidecar_roots=[],
        tweet_sidecar_roots=[],
        apply_v3_additional_group_masks=False,
    )


def eval_path(result_root: Path, prefix: str, model: str, filename: str) -> Path:
    return result_root / prefix / model / filename


def first_human_path(result_root: Path, prefix: str, version: str, preferred_model: str) -> Path | None:
    filename = str(VERSION_SPECS[version]["human_filename"])
    for model in (preferred_model, GROUP_MODEL, TOPIC_MODEL):
        candidate = eval_path(result_root, prefix, model, filename)
        if candidate.exists():
            return candidate
    return None


def fit_alphas(
    *, result_root: Path, data_root: Path, version: str, partition: str,
) -> dict[str, float]:
    prefixes = read_prefixes(data_root, partition)
    preferred_model = str(PARTITION_SPECS[partition]["model"])
    rows_by_prefix: dict[str, list[dict[str, str]]] = {}
    missing_human_prefixes = 0
    for prefix in prefixes:
        human_path = first_human_path(result_root, prefix, version, preferred_model)
        if human_path is None:
            missing_human_prefixes += 1
            continue
        rows_by_prefix[prefix] = read_eval_rows(human_path, result_root=result_root)
    result = fit_best_alphas_from_human_rows(
        rows_by_prefix=rows_by_prefix,
        likert_col=LIKERT_COL,
    )
    # These counts make the source-resolution contract auditable in the
    # diagnostics CSV.  The train fit is deliberately reused for its paired
    # test partition, exactly as in the v4 paper workflow.
    result["fit_prefix_count"] = float(len(rows_by_prefix))
    result["fit_missing_human_prefix_count"] = float(missing_human_prefixes)
    return result


def compute_raw_rows(
    *, sft_result_root: Path, base_result_root: Path, prefixes: list[str], sft_model: str, version: str, alpha_spec: dict[str, float], bootstrap_reps: int, models: list[str] | None = None, include_degroot: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return reusable base, v5-SFT, and DeGroot v2-style metric rows."""
    human_filename = str(VERSION_SPECS[version]["human_filename"])
    llm_filename = str(VERSION_SPECS[version]["llm_filename"])
    selected_models = list(models) if models is not None else [*BASE_MODELS, sft_model]
    model_aggregates = build_model_aggregates(selected_models)
    degroot_aggregates = build_model_aggregates([DEGROOT_MODEL])
    model_results: dict[str, dict[str, Any]] = {}
    degroot_results: dict[str, dict[str, Any]] = {}
    statuses: list[dict[str, Any]] = []

    for prefix_index, prefix in enumerate(prefixes, start=1):
        if prefix_index == 1 or prefix_index % 50 == 0 or prefix_index == len(prefixes):
            print(f"  {version}: {len(selected_models)} model series — prefix {prefix_index}/{len(prefixes)}", flush=True)
        model_results[prefix] = {}
        for model in selected_models:
            model_root = sft_result_root if model == sft_model else base_result_root
            source = "SFT v5" if model == sft_model else "Reused base"
            model_result = compute_prefix_model(
                data_prefix=prefix,
                model=model,
                roots=[model_root],
                human_filename=human_filename,
                llm_filename=llm_filename,
                utterance_event_types=set(DEFAULT_UTTERANCE_EVENT_TYPES),
                likert_col=LIKERT_COL,
                human_text_col="text",
                llm_text_col="llm_text",
                min_key_match_rate=0.7,
                include_conditioned_alignment=False,
                respect_validity_labels=True,
                initial_post_sidecar_roots=[],
                tweet_sidecar_roots=[],
                apply_v3_additional_group_masks=False,
            )
            merge_prefix_result_into_aggregate(model_aggregates[model], model_result)
            model_results[prefix][model] = model_result
            statuses.append(
                {
                    "data_prefix": prefix,
                    "version": version,
                    "model": model,
                    "source": source,
                    "status": model_result.status,
                    "message": model_result.message,
                    "human_path": model_result.human_path,
                    "llm_path": model_result.llm_path,
                }
            )

        if not include_degroot:
            continue

        human_path = first_human_path(sft_result_root, prefix, version, sft_model)
        if human_path is None:
            statuses.append({"data_prefix": prefix, "version": version, "model": DEGROOT_MODEL, "source": "DeGroot", "status": "missing_human", "message": "Human CSV missing.", "human_path": "", "llm_path": ""})
            continue
        try:
            human_rows = read_eval_rows(human_path, result_root=sft_result_root)
            proxy_rows = build_degroot_proxy_rows(
                human_rows,
                data_prefix=prefix,
                likert_col=LIKERT_COL,
                alpha_public=float(alpha_spec["alpha_public"]),
                alpha_private=float(alpha_spec["alpha_private"]),
            )
            proxy_result = compute_prefix_model_from_rows(
                data_prefix=prefix,
                model=DEGROOT_MODEL,
                human_rows=human_rows,
                llm_rows=proxy_rows,
                utterance_event_types=set(DEFAULT_UTTERANCE_EVENT_TYPES),
                likert_col=LIKERT_COL,
                min_key_match_rate=0.7,
                include_conditioned_alignment=False,
                human_path=str(human_path),
                llm_path=f"degroot(alpha_public={alpha_spec['alpha_public']:.6f},alpha_private={alpha_spec['alpha_private']:.6f})",
            )
        except Exception as exc:  # keep a damaged prefix visible instead of failing the entire regeneration
            statuses.append({"data_prefix": prefix, "version": version, "model": DEGROOT_MODEL, "source": "DeGroot", "status": "error", "message": str(exc), "human_path": str(human_path), "llm_path": ""})
            continue
        merge_prefix_result_into_aggregate(degroot_aggregates[DEGROOT_MODEL], proxy_result)
        degroot_results[prefix] = {DEGROOT_MODEL: proxy_result}
        statuses.append({"data_prefix": prefix, "version": version, "model": DEGROOT_MODEL, "source": "DeGroot", "status": proxy_result.status, "message": proxy_result.message, "human_path": proxy_result.human_path, "llm_path": proxy_result.llm_path})

    raw_rows: list[dict[str, Any]] = []
    if selected_models:
        raw_rows = build_raw_rows_from_aggregates(model_aggregates, selected_models, ub_mode="gaussian_likert")
        raw_rows = add_v2_bootstrap_cis_to_raw_rows(raw_rows, models=selected_models, prefixes=prefixes, prefix_results_by_prefix=model_results, bootstrap_reps=bootstrap_reps, seed=42)
    proxy_rows: list[dict[str, Any]] = []
    if include_degroot:
        proxy_rows = build_raw_rows_from_aggregates(degroot_aggregates, [DEGROOT_MODEL], ub_mode="gaussian_likert")
        proxy_rows = add_v2_bootstrap_cis_to_raw_rows(proxy_rows, models=[DEGROOT_MODEL], prefixes=prefixes, prefix_results_by_prefix=degroot_results, bootstrap_reps=bootstrap_reps, seed=42)
    return [*raw_rows, *proxy_rows], statuses


def compute_all_summary_rows(*, result_root: Path, base_result_root: Path, data_root: Path, bootstrap_reps: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str], dict[str, float]]]:
    alpha_specs: dict[tuple[str, str], dict[str, float]] = {}
    print("Fitting DeGroot alpha diagnostics from the v5 human surface...", flush=True)
    for partition, spec in PARTITION_SPECS.items():
        for version in VERSION_SPECS:
            alpha_specs[(partition, version)] = fit_alphas(
                result_root=result_root,
                data_root=data_root,
                version=version,
                partition=str(spec["fit_partition"]),
            )

    all_rows: list[dict[str, Any]] = []
    all_statuses: list[dict[str, Any]] = []
    for partition, spec in PARTITION_SPECS.items():
        prefixes = read_prefixes(data_root, partition)
        for version in VERSION_SPECS:
            print(f"Computing {partition} / {version} metrics for {len(prefixes)} prefixes...", flush=True)
            rows, statuses = compute_raw_rows(
                sft_result_root=result_root,
                base_result_root=base_result_root,
                prefixes=prefixes,
                sft_model=str(spec["model"]),
                version=version,
                alpha_spec=alpha_specs[(partition, version)],
                bootstrap_reps=bootstrap_reps,
            )
            for row in rows:
                row["partition"] = partition
                row["simulation_version"] = version
                row["model_label"] = model_label(str(row.get("model", "")))
                row["fit_alpha_public"] = alpha_specs[(partition, version)]["alpha_public"]
                row["fit_alpha_private"] = alpha_specs[(partition, version)]["alpha_private"]
                row["evaluation_source"] = (
                    "v5 SFT result root" if str(row.get("model")) == str(spec["model"])
                    else "reused eval_v3 base result root" if str(row.get("model")) in BASE_MODELS
                    else "v5 human-surface DeGroot fit"
                )
            for status in statuses:
                status["partition"] = partition
            all_rows.extend(rows)
            all_statuses.extend(statuses)
    return all_rows, all_statuses, alpha_specs


def compute_alpha_specs(
    *,
    result_root: Path,
    data_root: Path,
    partitions: Iterable[str] | None = None,
    versions: Iterable[str] | None = None,
) -> dict[tuple[str, str], dict[str, float]]:
    """Fit the v5 DeGroot diagnostic, sharing each train fit with its test split."""
    selected_partitions = list(PARTITION_SPECS) if partitions is None else list(partitions)
    selected_versions = list(VERSION_SPECS) if versions is None else list(versions)
    alpha_specs: dict[tuple[str, str], dict[str, float]] = {}
    fit_cache: dict[tuple[str, str], dict[str, float]] = {}
    print("Fitting DeGroot alpha diagnostics from the v5 human surface...", flush=True)
    for partition in selected_partitions:
        spec = PARTITION_SPECS[partition]
        for version in selected_versions:
            fit_key = (str(spec["fit_partition"]), version)
            if fit_key not in fit_cache:
                fit_cache[fit_key] = fit_alphas(
                    result_root=result_root,
                    data_root=data_root,
                    version=version,
                    partition=fit_key[0],
                )
            alpha_specs[(partition, version)] = dict(fit_cache[fit_key])
    return alpha_specs


def add_summary_provenance(
    rows: list[dict[str, Any]], *, partition: str, version: str, sft_model: str, alpha_spec: dict[str, float],
) -> None:
    for row in rows:
        row["partition"] = partition
        row["simulation_version"] = version
        row["model_label"] = model_label(str(row.get("model", "")))
        row["fit_alpha_public"] = alpha_spec["alpha_public"]
        row["fit_alpha_private"] = alpha_spec["alpha_private"]
        row["evaluation_source"] = (
            "v5 SFT result root" if str(row.get("model")) == sft_model
            else "reused eval_v3 base result root" if str(row.get("model")) in BASE_MODELS
            else "v5 human-surface DeGroot fit"
        )


def model_cache_stem(model: str) -> str:
    if model == DEGROOT_MODEL:
        return "degroot_reference"
    if model == GROUP_MODEL:
        return "sft_group"
    if model == TOPIC_MODEL:
        return "sft_topic"
    return f"base_{BASE_MODELS.index(model):02d}"


def partial_cache_paths(*, output_dir: Path, partition: str, version: str, model: str) -> tuple[Path, Path]:
    root = output_dir / "csv" / "intermediate"
    stem = f"{partition}__{version}__{model_cache_stem(model)}"
    return root / f"{stem}.csv", root / f"{stem}_status.csv"


def write_alpha_specs(
    *,
    output_dir: Path,
    alpha_specs: dict[tuple[str, str], dict[str, float]],
    merge_existing: bool = False,
) -> None:
    output_path = output_dir / "csv" / "degroot_fit_alphas.csv"
    if merge_existing and output_path.exists():
        existing = read_alpha_specs(output_path)
        existing.update(alpha_specs)
        alpha_specs = existing
    alpha_rows = [
        {"partition": partition, "simulation_version": version, "fit_partition": PARTITION_SPECS[partition]["fit_partition"], **values}
        for (partition, version), values in sorted(alpha_specs.items())
    ]
    write_csv(output_path, alpha_rows)


def read_alpha_specs(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    alpha_specs: dict[tuple[str, str], dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            alpha_specs[(str(row["partition"]), str(row["simulation_version"]))] = {
                field: finite(value)
                for field, value in row.items()
                if field not in {"partition", "simulation_version", "fit_partition"}
            }
    return alpha_specs


def prepare_model_cache(*, model: str, result_root: Path, base_result_root: Path, data_root: Path, output_dir: Path, bootstrap_reps: int, partition_filter: str | None = None, version_filter: str | None = None, write_alpha: bool = True) -> None:
    """Compute one model's metric rows in short, resumable local-terminal jobs."""
    selected_partitions = [partition_filter] if partition_filter is not None else None
    selected_versions = [version_filter] if version_filter is not None else None
    alpha_specs = compute_alpha_specs(
        result_root=result_root,
        data_root=data_root,
        partitions=selected_partitions,
        versions=selected_versions,
    )
    if write_alpha:
        write_alpha_specs(output_dir=output_dir, alpha_specs=alpha_specs, merge_existing=True)
    for partition, spec in PARTITION_SPECS.items():
        if partition_filter is not None and partition != partition_filter:
            continue
        sft_model = str(spec["model"])
        if model in {GROUP_MODEL, TOPIC_MODEL} and model != sft_model:
            continue
        prefixes = read_prefixes(data_root, partition)
        for version in VERSION_SPECS:
            if version_filter is not None and version != version_filter:
                continue
            print(f"Preparing {model_label(model)} for {partition} / {version} ({len(prefixes)} prefixes)...", flush=True)
            raw_rows, statuses = compute_raw_rows(
                sft_result_root=result_root,
                base_result_root=base_result_root,
                prefixes=prefixes,
                sft_model=sft_model,
                version=version,
                alpha_spec=alpha_specs[(partition, version)],
                bootstrap_reps=bootstrap_reps,
                models=[] if model == DEGROOT_MODEL else [model],
                include_degroot=model == DEGROOT_MODEL,
            )
            selected_rows = [row for row in raw_rows if str(row.get("model")) == model]
            add_summary_provenance(selected_rows, partition=partition, version=version, sft_model=sft_model, alpha_spec=alpha_specs[(partition, version)])
            selected_statuses = [row for row in statuses if str(row.get("model")) == model]
            for status in selected_statuses:
                status["partition"] = partition
            row_path, status_path = partial_cache_paths(output_dir=output_dir, partition=partition, version=version, model=model)
            write_csv(row_path, selected_rows)
            write_csv(status_path, selected_statuses)
    print(f"Prepared reusable cache for {model_label(model)}.", flush=True)


def assemble_cached_summary(*, output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str], dict[str, float]]]:
    alpha_path = output_dir / "csv" / "degroot_fit_alphas.csv"
    if not alpha_path.exists():
        raise SystemExit("Missing csv/degroot_fit_alphas.csv. Run --prepare-model for every model first.")
    rows: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    missing_paths: list[str] = []
    for partition, spec in PARTITION_SPECS.items():
        expected_models = [*BASE_MODELS, str(spec["model"]), DEGROOT_MODEL]
        for version in VERSION_SPECS:
            for model in expected_models:
                row_path, status_path = partial_cache_paths(output_dir=output_dir, partition=partition, version=version, model=model)
                if not row_path.exists() or not status_path.exists():
                    missing_paths.extend(str(path.relative_to(output_dir)) for path in (row_path, status_path) if not path.exists())
                    continue
                with row_path.open(encoding="utf-8", newline="") as handle:
                    rows.extend(csv.DictReader(handle))
                with status_path.open(encoding="utf-8", newline="") as handle:
                    statuses.extend(csv.DictReader(handle))
    if missing_paths:
        raise SystemExit("Missing reusable metric caches:\n" + "\n".join(sorted(missing_paths)))
    return rows, statuses, read_alpha_specs(alpha_path)


def filtered_rows(rows: list[dict[str, Any]], *, partition: str, version: str | None = None) -> list[dict[str, Any]]:
    selected = [row for row in rows if str(row.get("partition")) == partition]
    if version is not None:
        selected = [row for row in selected if str(row.get("simulation_version")) == version]
    return selected


def comparison_rows_by_version(rows: list[dict[str, Any]], *, partition: str) -> dict[str, list[dict[str, Any]]]:
    """Translate the v5 summary cache into the canonical v4/v4_b plot contract."""
    version_map = {"v0": "v4", "v1_5": "v4_b"}
    output: dict[str, list[dict[str, Any]]] = {"v4": [], "v4_b": []}
    for row in filtered_rows(rows, partition=partition):
        canonical_version = version_map.get(str(row.get("simulation_version", "")))
        if canonical_version is None:
            continue
        output[canonical_version].append({
            **row,
            "comparison_version": canonical_version,
        })
    return output


def render_alignment_bars(*, rows: list[dict[str, Any]], partition: str, output_path: Path) -> None:
    """Render the canonical M1/A3 publication layout with v5 values.

    The canonical renderer is intentionally reused here rather than mimicked:
    it controls the bar spacing, title treatment, legend, typography, and
    public/private grouping used in the byte-validated v4 suite.
    """
    rows_by_version = comparison_rows_by_version(rows, partition=partition)
    model_order = comparison_model_order(partition)
    figure = build_page_figure(
        page_title=f"v5 combined BD: {short_partition(partition)}",
        rows_by_version=rows_by_version,
        hide_count_annotations=True,
        model_order=model_order,
        shared_models=set(),
    )
    bar_style._clean_publication_figure(figure, model_order=model_order)
    bar_style._tighten_y_limits(
        figure,
        rows_by_version=rows_by_version,
        model_order=model_order,
        has_value_labels=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    write_csv(output_path.with_suffix(".csv"), [row for version_rows in rows_by_version.values() for row in version_rows])


def render_correlations(*, rows: list[dict[str, Any]], partition: str, output_path: Path) -> None:
    """Use the canonical M2/A4 1x4 correlation renderer for both modes."""
    comparison_rows = [row for version_rows in comparison_rows_by_version(rows, partition=partition).values() for row in version_rows]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_input = output_path.with_name(f"{output_path.name}_canonical_input.csv")
    write_csv(canonical_input, comparison_rows)
    metric_rows = correlation_style._prepare_rows(canonical_input, include_degroot=False)
    correlation_rows = correlation_style._correlation_rows(metric_rows)
    write_csv(output_path.with_suffix(".csv"), correlation_rows)
    write_csv(
        output_path.with_name(f"{output_path.name}_metric_rows.csv"),
        metric_rows.to_dict(orient="records"),
    )
    for version_key, suffix in (("v4", "full_dynamics"), ("v4_b", "next_message")):
        for publication_variant in ("raw", "pub"):
            if version_key == "v4" and publication_variant == "pub":
                figure_path = output_path.with_suffix(".png")
            else:
                figure_path = output_path.with_name(f"{output_path.name}_{suffix}_{publication_variant}.png")
            correlation_style._render_version_scatter_panels(
                metric_rows,
                correlation_rows,
                version_key=version_key,
                publication_variant=publication_variant,
                output_path=figure_path,
            )


def profile_rows_for_partition(*, result_root: Path, base_result_root: Path, data_root: Path, partition: str, version: str, alpha_spec: dict[str, float]) -> dict[str, list[dict[str, str]]]:
    cache_key = (
        str(result_root),
        str(base_result_root),
        str(data_root),
        partition,
        version,
        float(alpha_spec["alpha_public"]),
        float(alpha_spec["alpha_private"]),
    )
    cached = PROFILE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    sft_model = str(PARTITION_SPECS[partition]["model"])
    rows_by_series: dict[str, list[dict[str, str]]] = {
        HUMAN_MODEL: [],
        **{model: [] for model in BASE_MODELS},
        sft_model: [],
        DEGROOT_MODEL: [],
    }
    for prefix in read_prefixes(data_root, partition):
        human_path = first_human_path(result_root, prefix, version, sft_model)
        if human_path is None:
            continue
        try:
            human_rows = read_eval_rows(human_path, result_root=result_root)
            rows_by_series[HUMAN_MODEL].extend(human_rows)
            proxy_rows = build_degroot_proxy_rows(human_rows, data_prefix=prefix, likert_col=LIKERT_COL, alpha_public=float(alpha_spec["alpha_public"]), alpha_private=float(alpha_spec["alpha_private"]))
            rows_by_series[DEGROOT_MODEL].extend(proxy_rows)
            for model in [*BASE_MODELS, sft_model]:
                model_root = result_root if model == sft_model else base_result_root
                llm_path = eval_path(model_root, prefix, model, str(VERSION_SPECS[version]["llm_filename"]))
                if llm_path.exists():
                    rows_by_series[model].extend(read_eval_rows(llm_path, result_root=model_root))
        except Exception:
            continue
    PROFILE_CACHE[cache_key] = rows_by_series
    return rows_by_series


def per_group_sd_trajectory(rows: list[dict[str, str]], *, kind: str) -> tuple[list[float], list[float]]:
    profile = build_profile(rows, default_group="v5", likert_col=LIKERT_COL)
    first: list[float] = []
    second: list[float] = []
    if kind == "public":
        middle: list[float] = []
        for gid in sorted(profile.tweet_round):
            participants = strict_public_participants(profile, gid)
            if not participants:
                continue
            rounds = profile.tweet_round[gid]
            first.append(safe_pstdev([rounds[p][1] for p in participants]))
            middle.append(safe_pstdev([rounds[p][2] for p in participants]))
            second.append(safe_pstdev([rounds[p][3] for p in participants]))
        return [safe_mean(first), safe_mean(middle), safe_mean(second)], [sem(first), sem(middle), sem(second)]
    for gid in sorted(set(profile.initial_private) | set(profile.final_private)):
        participants = strict_private_participants(profile, gid)
        if not participants:
            continue
        first.append(safe_pstdev([profile.initial_private[gid][p] for p in participants]))
        second.append(safe_pstdev([profile.final_private[gid][p] for p in participants]))
    return [safe_mean(first), safe_mean(second)], [sem(first), sem(second)]


def convergence_reference_deltas(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    """Return the canonical two human-derived convergence references.

    The v4 composite shows the expected one-step-toward-first-partner change
    and a 50% SD-reduction target as horizontal guides in its bar panels.  The
    same definitions are retained here, but are recomputed from each v5 split
    rather than borrowed from the submission-era v4 report.
    """
    one_step_public, one_step_private = compute_consensus_floor_lists(
        rows,
        data_prefix="v5",
        likert_col=LIKERT_COL,
    )
    half_public, half_private = compute_half_sd_reduction_lists(
        rows,
        data_prefix="v5",
        likert_col=LIKERT_COL,
    )
    return {
        "public": {
            "One-Step Toward First Partner": safe_mean(one_step_public),
            "50% SD Reduction": safe_mean(half_public),
        },
        "private": {
            "One-Step Toward First Partner": safe_mean(one_step_private),
            "50% SD Reduction": safe_mean(half_private),
        },
    }


def convergence_line_style(model: str) -> dict[str, float | str]:
    """Use the submission renderer's hierarchy for trajectory series."""
    return line_style.series_line_style(
        model,
        model,
        publication_model_label(model),
        is_human=model == HUMAN_MODEL,
    )


def convergence_legend_priority(label: str) -> tuple[int, int]:
    if label == "Human":
        return (0, 0)
    if "Llama-3.1-8B" in label:
        return (0, 1)
    if label == "Llama SFT":
        return (0, 2)
    if label == "One-Step Toward First Partner":
        return (2, 0)
    if label == "50% SD Reduction":
        return (2, 1)
    return (1, 0)


def anchor_trajectories_to_human(
    raw_trajectories: dict[str, dict[str, tuple[list[float], list[float]]]],
) -> dict[str, dict[str, tuple[list[float], list[float]]]]:
    """Anchor conditioned-model trajectories at the shared human initial state.

    Models receive the human initial private opinion and first public tweet as
    conditioning context.  As in the canonical v4 line renderer, the plotted
    initial point and its SE therefore use the human aggregate for every
    model, even when model-specific source availability would yield a slightly
    different raw subset aggregate.
    """
    anchored: dict[str, dict[str, tuple[list[float], list[float]]]] = {}
    for model, kind_trajectories in raw_trajectories.items():
        anchored[model] = {}
        for kind, (raw_means, raw_ses) in kind_trajectories.items():
            means = list(raw_means)
            ses = list(raw_ses)
            if model != HUMAN_MODEL:
                human_means, human_ses = raw_trajectories[HUMAN_MODEL][kind]
                if means and human_means:
                    means[0] = human_means[0]
                if ses and human_ses:
                    ses[0] = human_ses[0]
            anchored[model][kind] = (means, ses)
    return anchored


def render_convergence_composite(*, result_root: Path, base_result_root: Path, data_root: Path, partition: str, alpha_spec: dict[str, float], output_path: Path) -> None:
    series = profile_rows_for_partition(result_root=result_root, base_result_root=base_result_root, data_root=data_root, partition=partition, version="v0", alpha_spec=alpha_spec)
    target_model = str(PARTITION_SPECS[partition]["model"])
    order = [HUMAN_MODEL, *BASE_MODELS, target_model, DEGROOT_MODEL]
    raw_trajectories = {model: {kind: per_group_sd_trajectory(series[model], kind=kind) for kind in ("public", "private")} for model in order}
    trajectories = anchor_trajectories_to_human(raw_trajectories)
    reference_deltas = convergence_reference_deltas(series[HUMAN_MODEL])
    reference_targets = line_style.compute_sd_baseline_targets(series[HUMAN_MODEL], "v5")
    # Match the v4 publication composite: two convergence bars, followed by
    # public/private SD trajectories, with a shared legend beneath the line
    # panels rather than a custom figure title.
    fig, axes = plt.subplots(1, 4, figsize=(26.5, 6.8))
    output_rows: list[dict[str, Any]] = []
    bar_limits: list[float] = []
    for kind in ("public", "private"):
        for model in order:
            means, _ = trajectories[model][kind]
            if all(not math.isnan(value) for value in (means[0], means[-1])):
                bar_limits.append(means[-1] - means[0])
        bar_limits.extend(value for value in reference_deltas[kind].values() if not math.isnan(value))
    if bar_limits:
        lower, upper = min(bar_limits), max(bar_limits)
        spread = upper - lower
        bar_padding = max(0.08, spread * 0.2 if spread > 0 else 0.12)
        bar_ymin, bar_ymax = lower - bar_padding, upper + bar_padding
    else:
        bar_ymin, bar_ymax = -0.1, 0.1
    for index, kind in enumerate(("public", "private")):
        values = []
        errors = []
        for model in order:
            means, ses = trajectories[model][kind]
            raw_means, _raw_ses = raw_trajectories[model][kind]
            delta = means[-1] - means[0] if all(not math.isnan(v) for v in (means[0], means[-1])) else math.nan
            raw_delta = raw_means[-1] - raw_means[0] if all(not math.isnan(v) for v in (raw_means[0], raw_means[-1])) else math.nan
            errors.append(math.sqrt(sum(value ** 2 for value in (ses[0], ses[-1]) if not math.isnan(value))))
            values.append(delta)
            output_rows.append({"panel": f"{kind}_convergence", "model": model, "model_label": model_label(model), "mean_delta_sd": delta, "raw_mean_delta_sd": raw_delta, "bootstrap_or_delta_se": errors[-1], "anchored_to_human": model != HUMAN_MODEL, "partition": partition, "simulation_version": "v0"})
        axes[index].bar(np.arange(len(order)), values, color=[MODEL_COLORS[model] for model in order], edgecolor="white", linewidth=0.6, width=0.8)
        axes[index].errorbar(np.arange(len(order)), values, yerr=errors, fmt="none", ecolor="#2F2F2F", elinewidth=1.4, capsize=4.5, capthick=1.6, zorder=5)
        axes[index].axhline(0, color="#666666", linestyle="--", linewidth=1.0, alpha=0.9)
        for reference_label, reference_color in line_style.REFERENCE_LINE_SPECS:
            reference_value = reference_deltas[kind][reference_label]
            if not math.isnan(reference_value):
                axes[index].axhline(reference_value, color=reference_color, linewidth=1.8, linestyle=":", alpha=0.95)
            output_rows.append({"panel": f"{kind}_convergence_reference", "model": reference_label, "model_label": reference_label, "mean_delta_sd": reference_value, "bootstrap_or_delta_se": math.nan, "partition": partition, "simulation_version": "v0"})
        axes[index].set_xticks(np.arange(len(order)))
        axes[index].set_xticklabels([publication_model_label(model) for model in order], rotation=90, ha="center", va="top")
        axes[index].set_title(f"{kind.title()} Opinion Convergence", loc="left", fontweight="bold", fontsize=18)
        axes[index].set_ylabel("Opinion Convergence\n(Δ Within-Group Opinion SD)" if index == 0 else "", fontsize=16)
        axes[index].set_ylim(bar_ymin, bar_ymax)
        axes[index].grid(False)
        axes[index].tick_params(axis="x", labelsize=14, pad=1)
        axes[index].tick_params(axis="y", labelsize=12)
    for index, kind in enumerate(("public", "private"), start=2):
        labels = ["Tweet r1", "Tweet r2", "Tweet r3"] if kind == "public" else ["Initial", "Post"]
        for model in order:
            means, ses = trajectories[model][kind]
            raw_means, raw_ses = raw_trajectories[model][kind]
            style = convergence_line_style(model)
            axes[index].errorbar(
                range(len(labels)),
                means,
                yerr=ses,
                marker="o",
                linestyle=str(style["linestyle"]),
                linewidth=float(style["linewidth"]),
                markersize=float(style["markersize"]),
                elinewidth=float(style["elinewidth"]),
                color=MODEL_COLORS[model],
                label=publication_model_label(model),
                capsize=2.5 if model == HUMAN_MODEL else 2.0,
                alpha=float(style["alpha"]),
                zorder=float(style["zorder"]),
            )
            for step, (mean_value, se_value) in enumerate(zip(means, ses)):
                output_rows.append({"panel": f"{kind}_trajectory", "model": model, "model_label": model_label(model), "step": labels[step], "mean_sd": mean_value, "se": se_value, "raw_mean_sd": raw_means[step], "raw_se": raw_ses[step], "anchored_to_human": model != HUMAN_MODEL and step == 0, "partition": partition, "simulation_version": "v0"})
        target_kind = "tweet" if kind == "public" else "opinion"
        for reference_label, reference_color in line_style.REFERENCE_LINE_SPECS:
            target_value = finite(reference_targets.get(target_kind, {}).get(reference_label))
            if not math.isnan(target_value):
                marker_line = axes[index].plot(
                    [float(len(labels) - 1) + line_style.REFERENCE_MARKER_X_OFFSET],
                    [target_value],
                    color=reference_color,
                    linestyle="None",
                    marker=line_style.REFERENCE_MARKER,
                    markersize=line_style.REFERENCE_MARKER_SIZE,
                    markeredgecolor="white",
                    markeredgewidth=line_style.REFERENCE_MARKER_EDGE_WIDTH,
                    alpha=line_style.REFERENCE_LINE_ALPHA,
                    zorder=1,
                )[0]
                marker_line.set_path_effects(line_style.REFERENCE_LINE_PATH_EFFECTS)
            output_rows.append({"panel": f"{kind}_trajectory_reference", "model": reference_label, "model_label": reference_label, "step": labels[-1], "mean_sd": target_value, "se": math.nan, "partition": partition, "simulation_version": "v0"})
        axes[index].set_xticks(range(len(labels)))
        axes[index].set_xticklabels(labels)
        axes[index].set_xlim(-0.2, float(len(labels) - 1) + line_style.REFERENCE_MARKER_X_OFFSET + 0.10)
        axes[index].set_title(f"{kind.title()} Opinion SD Trajectory", loc="left", fontweight="bold", fontsize=18)
        axes[index].set_ylabel("Within-group Opinion SD" if index == 2 else "", fontsize=16)
        axes[index].grid(False)
        axes[index].tick_params(axis="x", labelsize=16)
        axes[index].tick_params(axis="y", labelsize=12)
    legend_handles: list[Line2D] = []
    for model in order:
        style = convergence_line_style(model)
        legend_handles.append(Line2D(
            [0], [0],
            color=MODEL_COLORS[model],
            marker="o",
            linestyle=str(style["linestyle"]),
            linewidth=float(style["linewidth"]),
            markersize=float(style["markersize"]),
            alpha=float(style["alpha"]),
            label=publication_model_label(model),
        ))
    for reference_label, reference_color in line_style.REFERENCE_LINE_SPECS:
        legend_handles.append(Line2D(
            [0], [0],
            color=reference_color,
            linewidth=0,
            marker=line_style.REFERENCE_MARKER,
            markersize=line_style.REFERENCE_MARKER_SIZE,
            markeredgecolor="white",
            markeredgewidth=line_style.REFERENCE_MARKER_EDGE_WIDTH,
            alpha=line_style.REFERENCE_LINE_ALPHA,
            label=reference_label,
        ))
    legend_handles = [
        handle for _index, handle in sorted(
            enumerate(legend_handles),
            key=lambda item: (convergence_legend_priority(item[1].get_label()), item[0]),
        )
    ]
    fig.legend(legend_handles, [handle.get_label() for handle in legend_handles], loc="lower center", bbox_to_anchor=(0.74, 0.135), ncol=4, frameon=False, fontsize=15, columnspacing=0.9, handlelength=2.2)
    fig.tight_layout(rect=(0, 0.13, 1, 0.94), w_pad=2.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    write_csv(output_path.with_suffix(".csv"), output_rows)


def transition_matrix(rows: list[dict[str, str]], *, kind: str) -> np.ndarray:
    profile = build_profile(rows, default_group="v5", likert_col=LIKERT_COL)
    matrix = np.zeros((6, 6), dtype=int)
    if kind == "public":
        for gid in profile.tweet_round:
            for participant in strict_public_participants(profile, gid):
                rounds = profile.tweet_round[gid][participant]
                start, end = rounds[1], rounds[3]
                matrix[max(1, min(6, int(round(start)))) - 1, max(1, min(6, int(round(end)))) - 1] += 1
    else:
        for gid in set(profile.initial_private) | set(profile.final_private):
            for participant in strict_private_participants(profile, gid):
                start = profile.initial_private[gid][participant]
                end = profile.final_private[gid][participant]
                matrix[max(1, min(6, int(round(start)))) - 1, max(1, min(6, int(round(end)))) - 1] += 1
    return matrix


def transition_panel_specs(partition: str) -> list[tuple[str, str, str]]:
    """Canonical M4/A8 five-column panel order and title-strip labels."""
    sft_model = str(PARTITION_SPECS[partition]["model"])
    return [
        (HUMAN_MODEL, "human", "Human"),
        (FOCUSED_BASE_MODEL, "base: Llama-3.1-8B-Instruct", "Llama-3.1-8B\nInstruct"),
        (sft_model, f"sft: {sft_model}", "LLaMA SFT"),
        (FOCUSED_OTHER_BASE_MODEL, FOCUSED_OTHER_BASE_MODEL, "gpt-5-mini"),
        (DEGROOT_MODEL, "DeGroot fit", "DeGroot fit"),
    ]


def normalized_frobenius_diff(model_matrix: np.ndarray, human_matrix: np.ndarray) -> float | None:
    if model_matrix.shape != human_matrix.shape:
        return None
    human_norm = float(np.linalg.norm(human_matrix, ord="fro"))
    diff_norm = float(np.linalg.norm(model_matrix - human_matrix, ord="fro"))
    return diff_norm if human_norm == 0 else diff_norm / human_norm


def render_transition_panel(*, result_root: Path, base_result_root: Path, data_root: Path, partition: str, alpha_spec: dict[str, float], output_path: Path) -> None:
    series = profile_rows_for_partition(result_root=result_root, base_result_root=base_result_root, data_root=data_root, partition=partition, version="v0", alpha_spec=alpha_spec)
    panel_specs = transition_panel_specs(partition)
    matrices = {
        (kind, model): transition_matrix(series[model], kind=kind)
        for kind in ("public", "private")
        for model, _style_label, _display_label in panel_specs
    }
    percentages = [
        (matrix / int(matrix.sum())).max()
        for matrix in matrices.values()
        if int(matrix.sum())
    ]
    vmax = max(percentages, default=1.0)
    fig = plt.figure(figsize=(transition_style.FIGURE_WIDTH, transition_style.FIGURE_HEIGHT))
    grid = fig.add_gridspec(2, 5, left=0.055, right=0.905, top=0.88, bottom=0.13, wspace=0.38, hspace=0.42)
    fig.text(0.018, 0.69, "Public opinion", rotation=90, va="center", ha="center", fontsize=transition_style.ROW_LABEL_FONTSIZE)
    fig.text(0.018, 0.275, "Private opinion", rotation=90, va="center", ha="center", fontsize=transition_style.ROW_LABEL_FONTSIZE)
    output_rows: list[dict[str, Any]] = []
    image = None
    for row_index, kind in enumerate(("public", "private")):
        human_matrix = matrices[(kind, HUMAN_MODEL)]
        for col_index, (model, style_label, display_label) in enumerate(panel_specs):
            matrix = matrices[(kind, model)]
            l2_norm = None if model == HUMAN_MODEL else normalized_frobenius_diff(matrix, human_matrix)
            image = transition_style.draw_publication_heatmap(
                fig.add_subplot(grid[row_index, col_index]),
                matrix,
                style_label,
                display_label_override=display_label,
                vmax=vmax,
                value_type="likert_pred" if kind == "public" else "slider_value",
                variant="pub",
                panel_info_text=None if l2_norm is None else f"L2 vs human = {l2_norm:.4f}",
            )
            total = int(matrix.sum())
            for start in range(6):
                for end in range(6):
                    count = int(matrix[start, end])
                    output_rows.append({
                        "partition": partition,
                        "simulation_version": "v0",
                        "publication_variant": "pub",
                        "opinion_type": kind,
                        "model": model,
                        "display_label": display_label.replace("\n", " "),
                        "l2_norm_vs_human": "" if l2_norm is None else l2_norm,
                        "start_likert": start + 1,
                        "end_likert": end + 1,
                        "count": count,
                        "share": (count / total) if total else 0.0,
                        "panel_total": total,
                    })
    if image is not None:
        colorbar = fig.colorbar(image, ax=fig.axes, fraction=0.025, pad=0.02, shrink=0.88)
        colorbar.set_label("Share of transitions (%)", fontsize=transition_style.COLORBAR_LABEL_FONTSIZE)
        colorbar.ax.yaxis.set_major_formatter(transition_style.PercentFormatter(xmax=1.0, decimals=0))
        colorbar.ax.tick_params(labelsize=transition_style.COLORBAR_TICK_FONTSIZE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    write_csv(output_path.with_suffix(".csv"), output_rows)


def partner_map(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    aliases: dict[str, dict[str, str]] = defaultdict(dict)
    partners: dict[str, dict[str, str]] = defaultdict(dict)
    for row in rows:
        participant = participant_id(row)
        if not participant:
            continue
        group = str(row.get("group_id") or "v5").strip() or "v5"
        for key in ("empirica_id", "worker_id", "sender_id"):
            value = str(row.get(key) or "").strip()
            if value:
                aliases[group].setdefault(value, participant)
        if str(row.get("event_type") or "").strip() == "tweet" and parse_round(row.get("chat_round_order")) == 1:
            recipient = str(row.get("recipient_id") or "").strip()
            if recipient:
                partners[group].setdefault(participant, recipient)
    return {group: {person: aliases[group].get(recipient, recipient) for person, recipient in group_partners.items()} for group, group_partners in partners.items()}


def partner_gap_points(rows: list[dict[str, str]], *, kind: str) -> list[tuple[float, float]]:
    profile = build_profile(rows, default_group="v5", likert_col=LIKERT_COL)
    partners = partner_map(rows)
    points: list[tuple[float, float]] = []
    for gid, group_partners in partners.items():
        if kind == "public":
            values = profile.tweet_round.get(gid, {})
            for person, partner in group_partners.items():
                if person not in values or partner not in values or not all(round_idx in values[person] and round_idx in values[partner] for round_idx in (1, 3)):
                    continue
                # Match the paper's M5/A9 definition exactly: partner minus
                # self, not an absolute gap.  Its signed x-axis is what makes
                # the canonical fitted trend lines and Spearman panels useful.
                points.append((values[partner][1] - values[person][1], values[person][3] - values[person][1]))
        else:
            initial, final = profile.initial_private.get(gid, {}), profile.final_private.get(gid, {})
            for person, partner in group_partners.items():
                if person not in initial or partner not in initial or person not in final:
                    continue
                points.append((initial[partner] - initial[person], final[person] - initial[person]))
    return points


def render_partner_gap_composite(*, result_root: Path, base_result_root: Path, data_root: Path, partition: str, alpha_spec: dict[str, float], output_path: Path, bootstrap_reps: int) -> None:
    series = profile_rows_for_partition(result_root=result_root, base_result_root=base_result_root, data_root=data_root, partition=partition, version="v0", alpha_spec=alpha_spec)
    order = [HUMAN_MODEL, *BASE_MODELS, str(PARTITION_SPECS[partition]["model"]), DEGROOT_MODEL]
    points = {(kind, model): partner_gap_points(series[model], kind=kind) for kind in ("public", "private") for model in order}
    # Reuse the canonical M5/A9 visual structure: two square scatter panels
    # followed by two Spearman-bar panels, with a compact shared legend.
    fig = plt.figure(figsize=(27.0, 7.1))
    outer_grid = fig.add_gridspec(1, 2, width_ratios=(1.05, 1.15), wspace=0.14, left=0.05, right=0.99, top=0.92, bottom=0.24)
    scatter_grid = outer_grid[0].subgridspec(1, 2, wspace=0.06)
    spearman_grid = outer_grid[1].subgridspec(1, 2, wspace=0.20)
    axes = [
        fig.add_subplot(scatter_grid[0, 0]),
        fig.add_subplot(scatter_grid[0, 1]),
        fig.add_subplot(spearman_grid[0, 0]),
        fig.add_subplot(spearman_grid[0, 1]),
    ]
    output_rows: list[dict[str, Any]] = []
    legend_order = [model for model in order if model != HUMAN_MODEL] + [HUMAN_MODEL]
    scatter_handles: list[Any] = []
    for index, kind in enumerate(("public", "private")):
        ax = axes[index]
        entity_points = {model: points[(kind, model)] for model in order if points[(kind, model)]}
        spec = scatter_style.PANEL_KEY_TO_SPEC[kind]
        x_lim, y_lim = scatter_style.square_scatter_limits(
            *scatter_style.scatter_mod.compute_scatter_limits(spec, [entity_points])
        )
        scatter_handles.extend(
            scatter_style.render_publication_axis(
                ax,
                spec=spec,
                entity_points=entity_points,
                legend_order=[model for model in legend_order if entity_points.get(model)],
                x_lim=x_lim,
                y_lim=y_lim,
                page_index=1,
                panel_index=index,
                panel_title="",
                show_grid=False,
                footnote_text=None,
            )
        )
        ax.set_title(f"{kind.title()} Opinion", fontsize=18, fontweight="bold")
        ax.xaxis.label.set_fontsize(15)
        ax.yaxis.label.set_fontsize(15)
        ax.tick_params(axis="both", labelsize=12)
        for model in order:
            for x_value, y_value in points[(kind, model)]:
                output_rows.append({
                    "panel": f"{kind}_scatter",
                    "partition": partition,
                    "simulation_version": "v0",
                    "model": model,
                    "model_label": publication_model_label(model),
                    "partner_minus_self_initial_gap": x_value,
                    "opinion_change": y_value,
                })
    for index, kind in enumerate(("public", "private"), start=2):
        ax = axes[index]
        panel_rows: list[dict[str, str]] = []
        for model_index, model in enumerate(order):
            series_points = points[(kind, model)]
            rho = float(spearmanr([point[0] for point in series_points], [point[1] for point in series_points]).statistic) if len(series_points) >= 2 else math.nan
            ci_low, ci_high = spearman_ci_style.bootstrap_spearman_ci(
                series_points,
                bootstrap_reps=bootstrap_reps,
                # Match the canonical v4 exporter's reproducible per-panel
                # seeds: model position, offset by 1,000 for private opinion.
                seed=1 + model_index + (1000 if kind == "private" else 0),
            )
            panel_rows.append({
                "row_type": "correlation_summary",
                "panel_key": kind,
                "entity": model,
                "entity_display": publication_model_label(model),
                "spearman_rho": "" if math.isnan(rho) else str(rho),
                "spearman_rho_ci_low": "" if math.isnan(ci_low) else str(ci_low),
                "spearman_rho_ci_high": "" if math.isnan(ci_high) else str(ci_high),
                "n_points": str(len(series_points)),
            })
            output_rows.append({
                "panel": f"{kind}_spearman",
                "partition": partition,
                "simulation_version": "v0",
                "model": model,
                "model_label": publication_model_label(model),
                "spearman_rho": rho,
                "spearman_rho_ci_low": ci_low,
                "spearman_rho_ci_high": ci_high,
                "n": len(series_points),
            })
        spearman_style.render_panel(ax, rows=panel_rows, panel_title="", publication=True)
        ax.set_title(f"{kind.title()} Opinion", fontsize=18, fontweight="bold")
        if index == 3:
            ax.set_ylabel("")
        ax.xaxis.label.set_fontsize(15)
        ax.yaxis.label.set_fontsize(15)
        ax.set_xticklabels([publication_model_label(model) for model in order], rotation=90, ha="center", va="top")
        ax.tick_params(axis="x", labelsize=13, pad=1)
        ax.tick_params(axis="y", labelsize=12)
    handle_map = {handle.get_label(): handle for handle in scatter_handles}
    ordered_handles = [handle_map[model] for model in legend_order if model in handle_map]
    for handle in ordered_handles:
        handle.set_label(publication_model_label(handle.get_label()))
    fig.legend(
        handles=ordered_handles,
        loc="lower center",
        bbox_to_anchor=(0.262, 0.042),
        ncol=4,
        frameon=False,
        fontsize=11.5,
        columnspacing=0.9,
        handlelength=2.2,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    write_csv(output_path.with_suffix(".csv"), output_rows)


def stage_values(rows: list[dict[str, str]], *, kind: str) -> tuple[list[float], list[float]]:
    profile = build_profile(rows, default_group="v5", likert_col=LIKERT_COL)
    if kind == "public":
        first, final = [], []
        for gid in profile.tweet_round:
            for participant in strict_public_participants(profile, gid):
                first.append(profile.tweet_round[gid][participant][1])
                final.append(profile.tweet_round[gid][participant][3])
        return first, final
    first, final = [], []
    for gid in set(profile.initial_private) | set(profile.final_private):
        for participant in strict_private_participants(profile, gid):
            first.append(profile.initial_private[gid][participant])
            final.append(profile.final_private[gid][participant])
    return first, final


def render_histogram(*, result_root: Path, base_result_root: Path, data_root: Path, partition: str, alpha_spec: dict[str, float], kind: str, output_path: Path) -> None:
    series = profile_rows_for_partition(result_root=result_root, base_result_root=base_result_root, data_root=data_root, partition=partition, version="v0", alpha_spec=alpha_spec)
    sft_model = str(PARTITION_SPECS[partition]["model"])
    panels = [
        (HUMAN_MODEL, "first"),
        (HUMAN_MODEL, "final"),
        (FOCUSED_BASE_MODEL, "final"),
        (sft_model, "final"),
    ]
    # The focused A6/A7 export is a 2x2 publication page in the canonical
    # reproduction suite, rather than the earlier v5 one-row diagnostic.
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 10.4))
    axes_flat = list(np.asarray(axes).reshape(-1))
    output_rows: list[dict[str, Any]] = []
    for index, (ax, (model, stage)) in enumerate(zip(axes_flat, panels)):
        first, final = stage_values(series[model], kind=kind)
        values = first if stage == "first" else final
        counts = [sum(1 for value in values if int(round(value)) == score) for score in range(1, 7)]
        total = len(values)
        summary_row = {
            "n": str(total),
            "mean_likert": str(safe_mean(values)),
            **{f"count_{score}": str(count) for score, count in zip(range(1, 7), counts)},
        }
        if model == HUMAN_MODEL:
            title = "Human anchor" if stage == "first" else "Human final"
        elif model == FOCUSED_BASE_MODEL:
            title = "Llama 8B"
        elif model in {GROUP_MODEL, TOPIC_MODEL}:
            title = "Llama SFT"
        else:
            title = publication_model_label(model)
        histogram_style.add_histogram_subplot(
            ax,
            row=summary_row,
            title=title,
            color=MODEL_COLORS[model],
            show_ylabel=index % 2 == 0,
        )
        mean_value = histogram_style.mean_from_summary_row(summary_row)
        std_value = histogram_style.std_from_summary_row(summary_row)
        ax.set_title(f"{title}\n(mean={mean_value:.2f}, sd={std_value:.2f})", fontsize=15, pad=8)
        ax.grid(False)
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        if index % 2 == 0:
            ax.set_ylabel("Share (%)", fontsize=13)
        ax.set_xlabel("Likert", fontsize=13)
        for score, count in zip(range(1, 7), counts):
            output_rows.append({"partition": partition, "simulation_version": "v0", "opinion_type": kind, "model": model, "stage": stage, "likert": score, "count": count, "n": total, "percent": 100 * count / total if total else math.nan})
    opinion_label = "Public" if kind == "public" else "Private"
    fig.suptitle(f"Histogram of {opinion_label} Opinion Stance", fontsize=26, y=0.985, fontweight="bold")
    fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    write_csv(output_path.with_suffix(".csv"), output_rows)


def load_aggregate_metrics(*, result_root: Path, partition: str) -> list[dict[str, Any]]:
    split = "group" if partition.startswith("group") else "topic"
    state = "test" if partition.endswith("test") else "train"
    model = str(PARTITION_SPECS[partition]["model"])
    desired = (
        ("Semantic similarity — tweet", f"semantic_similarity_tweet_round_{state}_{split}"),
        ("Semantic similarity — post opinion", f"semantic_similarity_post_opinion_round_{state}_{split}"),
        ("ROUGE-L — tweet", f"rougeL_tweet_round_{state}_{split}"),
        ("ROUGE-L — post opinion", f"rougeL_post_opinion_round_{state}_{split}"),
        ("Mean length difference — tweet", f"n_word_diff_tweet_round_{state}_{split}"),
        ("Mean length difference — post opinion", f"n_word_diff_post_opinion_round_{state}_{split}"),
    )
    rows: list[dict[str, Any]] = []
    for version in VERSION_SPECS:
        path = result_root / f"llm_report_{version}_bd.csv"
        values: dict[str, float] = {}
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("model")) == model:
                    values[str(row.get("type"))] = finite(row.get("score"))
        for label, metric in desired:
            rows.append({"partition": partition, "simulation_version": version, "model": model, "metric": metric, "metric_label": label, "score": values.get(metric, math.nan)})
    return rows


BEYOND_STANCE_FIELD_BY_PREFIX = {
    "semantic_similarity_tweet_round": "semantic_similarity_tweet",
    "semantic_similarity_post_opinion_round": "semantic_similarity_post_opinion",
    "rougeL_tweet_round": "rouge_l_tweet",
    "rougeL_post_opinion_round": "rouge_l_post_opinion",
    "n_word_diff_tweet_round": "n_word_diff_tweet",
    "n_word_diff_post_opinion_round": "n_word_diff_post_opinion",
}


def canonical_beyond_stance_base_rows() -> list[dict[str, Any]]:
    """Load the submission's reusable breadth base-model aggregate roster.

    These aggregates are the base-model counterpart to the v5 SFT-only
    `llm_report_*.csv` files.  The user explicitly approved reusing base
    results; keeping the original table here makes A1/A2 follow the same
    public/private paper layout as their canonical v4 counterparts.
    """
    rows = beyond_stance_style.parse_maintable_rows(
        beyond_stance_style.MAINTABLE_FILES["breadth"],
        "breadth",
    )
    return beyond_stance_style._base_breadth_plot_rows(rows)


def v5_beyond_stance_sft_rows(rows: list[dict[str, Any]], *, partition: str) -> list[dict[str, Any]]:
    state = "test" if partition.endswith("test") else "train"
    split = "group" if partition.startswith("group") else "topic"
    target_model = str(PARTITION_SPECS[partition]["model"])
    records: list[dict[str, Any]] = []
    for version in VERSION_SPECS:
        values = {
            str(row["metric"]): finite(row["score"])
            for row in rows
            if str(row["simulation_version"]) == version
        }
        record: dict[str, Any] = {
            "source_table": "v5_sft",
            "simulation_mode": (
                "Simulation Mode 3: Full Conversation Simulation (v0)"
                if version == "v0"
                else "Simulation Mode 2: Tweet-guided Conversation Simulation (v1b)"
            ),
            "simulation_mode_short": "full_conversation" if version == "v0" else "tweet_guided",
            "model": target_model,
        }
        for metric_prefix, field_name in BEYOND_STANCE_FIELD_BY_PREFIX.items():
            metric = f"{metric_prefix}_{state}_{split}"
            record[field_name] = values.get(metric, math.nan)
        records.append(record)
    return records


def render_beyond_stance(*, result_root: Path, partition: str, output_path: Path) -> None:
    aggregate_rows = load_aggregate_metrics(result_root=result_root, partition=partition)
    base_rows = canonical_beyond_stance_base_rows()
    sft_rows = v5_beyond_stance_sft_rows(aggregate_rows, partition=partition)
    target_model = str(PARTITION_SPECS[partition]["model"])
    model_order = beyond_stance_style._insert_post_sft_model(
        beyond_stance_style._ordered_model_names(base_rows),
        target_model,
    )
    public_centers = list(range(len(model_order)))
    private_centers = [position + len(model_order) + beyond_stance_style.PANEL_SECTION_GAP for position in public_centers]
    all_centers = public_centers + private_centers
    lookup_by_mode: dict[str, dict[str, dict[str, Any]]] = {}
    for simulation_mode in beyond_stance_style.SIMULATION_MODE_PLOT_SPECS:
        mode_lookup = {
            str(row["model"]): row
            for row in base_rows
            if str(row["simulation_mode"]) == simulation_mode
        }
        mode_lookup.update({
            str(row["model"]): row
            for row in sft_rows
            if str(row["simulation_mode"]) == simulation_mode
        })
        lookup_by_mode[simulation_mode] = mode_lookup

    from matplotlib.patches import Patch

    style = beyond_stance_style.PLOT_VARIANTS["pub"]
    row_specs = beyond_stance_style.FIGURE_ROW_GROUP_SPECS[0][1]
    fig, axes = plt.subplots(
        len(row_specs),
        1,
        figsize=beyond_stance_style.figure_size(style=style, family="llama", row_count=len(row_specs)),
        sharex=False,
    )
    axes = list(np.atleast_1d(axes).flatten())
    beyond_stance_style._add_figure_panel_labels(fig, variant="pub", single_row=False)
    legend_handles = [
        Patch(
            facecolor="#BFBFBF",
            edgecolor="#1F1F1F",
            label=str(beyond_stance_style.SIMULATION_MODE_PLOT_SPECS["Simulation Mode 3: Full Conversation Simulation (v0)"]["label"]),
        ),
        Patch(
            facecolor="#BFBFBF",
            edgecolor="#1F1F1F",
            hatch="///",
            alpha=0.75,
            label=str(beyond_stance_style.SIMULATION_MODE_PLOT_SPECS["Simulation Mode 2: Tweet-guided Conversation Simulation (v1b)"]["label"]),
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc=beyond_stance_style._legend_loc(variant="pub", single_row=False),
        ncol=2,
        frameon=False,
        bbox_to_anchor=beyond_stance_style._legend_bbox(variant="pub", single_row=False),
        fontsize=float(style["legend_fontsize"]),
    )
    source_rows = [*base_rows, *sft_rows]
    for row_index, (axis, (row_title, public_field, private_field, fixed_limits)) in enumerate(zip(axes, row_specs)):
        row_limits = fixed_limits if fixed_limits is not None else beyond_stance_style._row_limits_for_metric(source_rows, public_field, private_field)
        beyond_stance_style._render_metric_row(
            ax=axis,
            row_title=row_title,
            public_centers=public_centers,
            private_centers=private_centers,
            model_order=model_order,
            lookup_by_mode=lookup_by_mode,
            public_field=public_field,
            private_field=private_field,
            row_limits=row_limits,
            row_index=row_index,
            variant="pub",
            single_row=False,
        )
    labels = [beyond_stance_style.plot_tick_label(model_name, variant="pub") for model_name in model_order]
    axes[-1].set_xticks(all_centers)
    axes[-1].set_xticklabels(labels + labels, rotation=float(style["llama_tick_rotation"]), ha="center")
    axes[-1].tick_params(axis="x", labelbottom=True, pad=8, labelsize=float(style["tick_fontsize"]))
    for axis in axes[:-1]:
        axis.set_xticklabels([])
    fig.subplots_adjust(
        left=float(style["left"]),
        right=float(style["right"]),
        bottom=float(style["bottom"]),
        top=float(style["top"]),
        hspace=float(style["hspace"]),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    output_rows: list[dict[str, Any]] = []
    for row in base_rows:
        output_rows.append({"provenance": "reused canonical v4 breadth base aggregate", **row})
    for row in sft_rows:
        output_rows.append({"provenance": "supplied v5 SFT aggregate", **row})
    write_csv(output_path.with_suffix(".csv"), output_rows)


def write_readme(*, output_dir: Path, result_root: Path, base_result_root: Path, data_root: Path, statuses: list[dict[str, Any]], alpha_specs: dict[tuple[str, str], dict[str, float]], bootstrap_reps: int) -> None:
    status_counts: dict[tuple[str, str, str, str], int] = defaultdict(int)
    for status in statuses:
        status_counts[(
            str(status.get("partition", "")),
            str(status.get("version", "")),
            str(status.get("source", "")),
            str(status.get("status", "")),
        )] += 1
    sft_coverage: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for (partition, version, source, status), count in status_counts.items():
        if source == "SFT v5":
            sft_coverage[(partition, version)][status] += count
    fits_have_observations = all(
        finite(result.get("public_count")) > 0 and finite(result.get("private_count")) > 0
        for result in alpha_specs.values()
    )
    if fits_have_observations:
        degroot_diagnostic = (
            "The v5 DeGroot series is a fitted baseline. It uses the same v4 paper protocol: "
            "fit public self-retention first on pooled tweet rounds 2 and 3, then fit private "
            "self-retention on final private opinions while holding the public alpha fixed. Each "
            "test partition reuses its corresponding training-split fit."
        )
    else:
        degroot_diagnostic = (
            "One or more alpha objectives have no usable observations; inspect "
            "`csv/degroot_fit_alphas.csv` before treating the associated DeGroot series as fitted."
        )
    lines = [
        "# v5 Ruixuan combined breadth/depth paper-figure regeneration",
        "",
        "This directory rebuilds the M1–M5 and A1–A9 analytical figure family from the supplied 2026-07-16 v5 BD simulation package. It is intentionally isolated from the canonical v4 submission artifacts.",
        "",
        "## Inputs and contract",
        "",
        f"- Evaluation result root: `{result_root}`",
        f"- Reused base-model result root: `{base_result_root}`",
        f"- Split-data/mask root: `{data_root}`",
        f"- Models: `{TOPIC_MODEL}` (topic split) and `{GROUP_MODEL}` (group split).",
        f"- Reused base models: {', '.join(f'`{model}`' for model in BASE_MODELS)}.",
        "- Simulation modes: `v0` = Full Dynamics Simulation; `v1_5` = Next Message Prediction.",
        "- The v5 bundle uses true combined breadth/depth group/topic splits and its own validity labels. It does **not** apply the v4 golden group or individual masks.",
        "- The v5 change is SFT-only. All seven paper base-model series are reused from the canonical `eval_v3/human_llm` source; only the split-specific SFT series come from the supplied v5 result root. This is an intentional mixed-source comparison, recorded per prefix in the status CSV.",
        "- A1/A2 use the same approved base-model reuse through `v4/raw_result_tables/Mini-Twitter_ Reproducing Result Tables - Leo_V4_Maintable_Breadth.csv`, merged with the supplied v5 `llm_report_{v0,v1_5}_bd.csv` SFT aggregates. Their CSVs label the provenance of every row explicitly.",
        "- M3/A5 trajectory plots use the canonical v4 conditioning rule: every model’s plotted public Tweet r1 and private Initial SD (including its SE) is anchored to the split’s human aggregate. The CSV preserves each model’s raw subset-specific initial value in `raw_mean_sd`/`raw_se` and marks the plotted anchor with `anchored_to_human = True`.",
        f"- M5/A9 Spearman bars show paired point-level 95% percentile-bootstrap confidence intervals ({bootstrap_reps:,} replicates), using the canonical v4 panel-specific deterministic seeds. The exported CSVs include `spearman_rho_ci_low` and `spearman_rho_ci_high`.",
        "",
        "## Paper mapping",
        "",
        "| Figure | Output directory | v5 source split |",
        "|---|---|---|",
        "| M1–M5 | `group_train_set/` | group train / group SFT |",
        "| A1 | `group_test_set/A1_group_test_beyond_stance` | group test / group SFT |",
        "| A2 | `topic_test_set/A2_topic_test_beyond_stance` | topic test / topic SFT |",
        "| A3–A5, A7–A9 | `topic_train_set/` | topic train / topic SFT |",
        "| A6 | `group_train_set/` | group train / group SFT / v0 |",
        "| Group-test counterparts | `group_test_set/` | M1–M5 and A6 |",
        "| Topic-test counterparts | `topic_test_set/` | A3–A5 and A7–A9 |",
        "",
        "The paper's historical placed-asset names may say `heldout`; this table names the actual v5 source masks explicitly.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "export PYTHON=python",
        "export PYTHONPATH=\"$PWD/src\"",
        "export MPLCONFIGDIR=outputs/tmp/codex-mpl-v5-ruixuan",
        f"\"$PYTHON\" neurips_2026/sean_reproduce/scripts/generate_v5_ruixuan_paper_figures.py --result-root \"{result_root}\" --base-result-root \"{base_result_root}\" --data-root \"{data_root}\" --output-dir \"$PWD/neurips_2026/sean_reproduce/v5_ruixuan\" --bootstrap-reps {bootstrap_reps}",
        "```",
        "",
        "### Focused partner-gap rerender",
        "",
        "To rerender only the M5/A9 partner-gap composites from an existing summary cache, use `--reuse-summary --render-only partner-gap`. Add `--partner-gap-partition {group_train,group_test,topic_train,topic_test}` to run one split at a time.",
        "",
        "### Terminal-friendly resume",
        "",
        "For a terminal with a short foreground-job limit, first prepare each required model/partition/version cache, then assemble. For example:",
        "```bash",
        f"\"$PYTHON\" neurips_2026/sean_reproduce/scripts/generate_v5_ruixuan_paper_figures.py --result-root \"{result_root}\" --base-result-root \"{base_result_root}\" --data-root \"{data_root}\" --output-dir \"$PWD/neurips_2026/sean_reproduce/v5_ruixuan\" --bootstrap-reps {bootstrap_reps} --prepare-model Llama-3.1-8B-Instruct --prepare-partition group_train --prepare-version v0",
        "# Repeat for every needed base/SFT/DeGroot model and partition/version; caches are written under csv/intermediate/.",
        f"\"$PYTHON\" neurips_2026/sean_reproduce/scripts/generate_v5_ruixuan_paper_figures.py --result-root \"{result_root}\" --base-result-root \"{base_result_root}\" --data-root \"{data_root}\" --output-dir \"$PWD/neurips_2026/sean_reproduce/v5_ruixuan\" --bootstrap-reps {bootstrap_reps} --assemble-from-cache",
        "```",
        "",
        "### Test-set analytical counterparts",
        "",
        "The canonical paper family uses group-test counterparts for the main-text panels and topic-test counterparts for the appendix panels. They are rendered directly under `group_test_set/` and `topic_test_set/`; A1 and A2 live in those same split directories.",
        "",
        "```bash",
        f"\"$PYTHON\" neurips_2026/sean_reproduce/scripts/generate_v5_ruixuan_paper_figures.py --result-root \"{result_root}\" --base-result-root \"{base_result_root}\" --data-root \"{data_root}\" --output-dir \"$PWD/neurips_2026/sean_reproduce/v5_ruixuan\" --reuse-summary --render-only test-main",
        f"\"$PYTHON\" neurips_2026/sean_reproduce/scripts/generate_v5_ruixuan_paper_figures.py --result-root \"{result_root}\" --base-result-root \"{base_result_root}\" --data-root \"{data_root}\" --output-dir \"$PWD/neurips_2026/sean_reproduce/v5_ruixuan\" --reuse-summary --render-only test-appendix",
        "```",
        "",
        "## DeGroot alpha diagnostic",
        "",
        degroot_diagnostic,
        "",
        "| Evaluation partition | Simulation | alpha_public | alpha_private | fit scope | public obs. | private obs. | combined MSE | human prefixes |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for (partition, version), result in sorted(alpha_specs.items()):
        lines.append(
            f"| {partition} | {version} | {float(result['alpha_public']):.6f} | "
            f"{float(result['alpha_private']):.6f} | {PARTITION_SPECS[partition]['fit_partition']} | "
            f"{finite(result.get('public_count')):.0f} | {finite(result.get('private_count')):.0f} | "
            f"{finite(result.get('combined_mse')):.6f} | {finite(result.get('fit_prefix_count')):.0f} |"
        )
    lines.extend([
        "",
        "## Coverage",
        "",
        "- Full per-prefix diagnostics: `csv/per_prefix_status.csv`; compact counts: `csv/coverage_summary.csv`.",
        "- Every analytical aggregate uses only prefixes with the required source CSVs. Missing human/LLM evaluations occur in both train and test masks; do not interpret the figures as complete-mask results. Base-series coverage is reported separately as `source = Reused base` in the compact CSV.",
        "",
        "| SFT evaluation partition | Simulation | expected prefixes | usable | missing human | missing LLM |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for (partition, version), counts in sorted(sft_coverage.items()):
        expected = sum(counts.values())
        lines.append(
            f"| {partition} | {version} | {expected} | {counts.get('ok', 0)} | {counts.get('missing_human', 0)} | {counts.get('missing_llm', 0)} |"
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_partition_readmes(output_dir: Path) -> None:
    """Document the four split directories without duplicating shared diagnostics."""
    descriptions = {
        "group_train": "Paper train panels using the group-train mask and group SFT model.",
        "group_test": "Group-test counterparts, plus A1 beyond-stance metrics, using the group-test mask and group SFT model.",
        "topic_train": "Paper appendix train panels using the topic-train mask and topic SFT model.",
        "topic_test": "Topic-test counterparts, plus A2 beyond-stance metrics, using the topic-test mask and topic SFT model.",
    }
    for partition, description in descriptions.items():
        directory = partition_output_dir(output_dir, partition)
        directory.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# v5 Ruixuan {partition.replace('_', ' ')} figures",
            "",
            description,
            "",
            "All figures retain the approved reused base-model series and v5 SFT provenance described in `../README.md`. Shared summary, coverage, alpha diagnostics, and resumable intermediate caches remain under `../csv/`.",
        ]
        if partition in {"group_test", "topic_test"}:
            lines.append("The beyond-stance panel in this directory combines canonical reusable breadth base aggregates with supplied v5 SFT aggregate values; its CSV records row-level provenance.")
        (directory / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_coverage(statuses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create a compact, machine-readable index for the per-prefix diagnostics."""
    counts: dict[tuple[str, str, str, str], int] = defaultdict(int)
    for row in statuses:
        counts[(
            str(row.get("partition", "")),
            str(row.get("version", "")),
            str(row.get("source", "")),
            str(row.get("status", "")),
        )] += 1
    return [
        {
            "partition": partition,
            "simulation_version": version,
            "source": source,
            "status": status,
            "prefix_count": count,
        }
        for (partition, version, source, status), count in sorted(counts.items())
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild v5 Ruixuan paper analytical figures from supplied BD simulations.")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--base-result-root", type=Path, default=DEFAULT_BASE_RESULT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-reps", type=int, default=1000, help="Prefix bootstrap repetitions for metric confidence intervals.")
    parser.add_argument("--reuse-summary", action="store_true", help="Reuse csv/summary_metrics.csv and fitted alpha/status CSVs already present in output-dir.")
    parser.add_argument("--prepare-model", choices=[*BASE_MODELS, GROUP_MODEL, TOPIC_MODEL, DEGROOT_MODEL], help="Prepare one model's resumable metric cache instead of rendering figures.")
    parser.add_argument("--prepare-partition", choices=list(PARTITION_SPECS), help="With --prepare-model, prepare only one partition to keep a terminal job short.")
    parser.add_argument("--prepare-version", choices=list(VERSION_SPECS), help="With --prepare-model, prepare only one simulation version to keep a terminal job short.")
    parser.add_argument("--skip-alpha-write", action="store_true", help="With --prepare-model, avoid concurrently rewriting the shared alpha diagnostic CSV.")
    parser.add_argument("--assemble-from-cache", action="store_true", help="Assemble all prepared model caches, then render the complete figure family.")
    parser.add_argument("--render-only", choices=("all", "main", "appendix", "appendix-hist", "appendix-end", "partner-gap", "test-all", "test-main", "test-appendix", "none"), default="all", help="Render the paper train family, a focused subset, or test counterparts into their matching {group,topic}_{train,test}_set directories.")
    parser.add_argument("--partner-gap-partition", choices=list(PARTITION_SPECS), help="With --render-only partner-gap, limit regeneration to one v5 mask partition.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result_root, base_result_root, data_root, output_dir = args.result_root.resolve(), args.base_result_root.resolve(), args.data_root.resolve(), args.output_dir.resolve()
    if not result_root.exists() or not base_result_root.exists() or not data_root.exists():
        raise SystemExit("The supplied SFT-result root, base-result root, or data root does not exist.")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.prepare_model and args.assemble_from_cache:
        raise SystemExit("Use either --prepare-model or --assemble-from-cache, not both.")
    if (args.prepare_partition or args.prepare_version) and not args.prepare_model:
        raise SystemExit("--prepare-partition and --prepare-version require --prepare-model.")
    if args.partner_gap_partition and args.render_only != "partner-gap":
        raise SystemExit("--partner-gap-partition requires --render-only partner-gap.")
    if args.prepare_model:
        prepare_model_cache(
            model=str(args.prepare_model),
            result_root=result_root,
            base_result_root=base_result_root,
            data_root=data_root,
            output_dir=output_dir,
            bootstrap_reps=max(0, int(args.bootstrap_reps)),
            partition_filter=args.prepare_partition,
            version_filter=args.prepare_version,
            write_alpha=not args.skip_alpha_write,
        )
        return 0
    summary_path = output_dir / "csv" / "summary_metrics.csv"
    status_path = output_dir / "csv" / "per_prefix_status.csv"
    coverage_path = output_dir / "csv" / "coverage_summary.csv"
    alpha_path = output_dir / "csv" / "degroot_fit_alphas.csv"
    if args.assemble_from_cache:
        rows, statuses, alpha_specs = assemble_cached_summary(output_dir=output_dir)
        write_csv(summary_path, rows)
        write_csv(status_path, statuses)
    elif args.reuse_summary:
        if not all(path.exists() for path in (summary_path, status_path, alpha_path)):
            raise SystemExit("--reuse-summary requires existing summary_metrics.csv, per_prefix_status.csv, and degroot_fit_alphas.csv.")
        with summary_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        with status_path.open(encoding="utf-8", newline="") as handle:
            statuses = list(csv.DictReader(handle))
        alpha_specs = read_alpha_specs(alpha_path)
        if not any(str(row.get("model")) in BASE_MODELS for row in rows):
            raise SystemExit("--reuse-summary cache predates reusable base-model rows; rerun once without --reuse-summary.")
    else:
        rows, statuses, alpha_specs = compute_all_summary_rows(result_root=result_root, base_result_root=base_result_root, data_root=data_root, bootstrap_reps=max(0, int(args.bootstrap_reps)))
        write_csv(summary_path, rows)
        write_csv(status_path, statuses)
        write_alpha_specs(output_dir=output_dir, alpha_specs=alpha_specs)

    write_csv(coverage_path, summarize_coverage(statuses))

    if args.render_only in {"all", "main"}:
        group_train_output_dir = partition_output_dir(output_dir, "group_train")
        print("Rendering M1–M5...", flush=True)
        print("  M1: model-comparison bars", flush=True)
        render_alignment_bars(rows=rows, partition="group_train", output_path=group_train_output_dir / "M1_group_train_alignment_bars")
        print("  M2: error correlations", flush=True)
        render_correlations(rows=rows, partition="group_train", output_path=group_train_output_dir / "M2_group_train_error_correlations")
        print("  M3: convergence", flush=True)
        render_convergence_composite(result_root=result_root, base_result_root=base_result_root, data_root=data_root, partition="group_train", alpha_spec=alpha_specs[("group_train", "v0")], output_path=group_train_output_dir / "M3_group_train_convergence_sd")
        print("  M4: transition matrices", flush=True)
        render_transition_panel(result_root=result_root, base_result_root=base_result_root, data_root=data_root, partition="group_train", alpha_spec=alpha_specs[("group_train", "v0")], output_path=group_train_output_dir / "M4_group_train_transition_matrices")
        print("  M5: partner-gap panels", flush=True)
        render_partner_gap_composite(result_root=result_root, base_result_root=base_result_root, data_root=data_root, partition="group_train", alpha_spec=alpha_specs[("group_train", "v0")], output_path=group_train_output_dir / "M5_group_train_partner_gap", bootstrap_reps=max(0, int(args.bootstrap_reps)))
    if args.render_only in {"all", "appendix"}:
        group_test_output_dir = partition_output_dir(output_dir, "group_test")
        topic_test_output_dir = partition_output_dir(output_dir, "topic_test")
        topic_train_output_dir = partition_output_dir(output_dir, "topic_train")
        print("Rendering A1–A9...", flush=True)
        print("  A1/A2: beyond-stance metrics", flush=True)
        render_beyond_stance(result_root=result_root, partition="group_test", output_path=group_test_output_dir / "A1_group_test_beyond_stance")
        render_beyond_stance(result_root=result_root, partition="topic_test", output_path=topic_test_output_dir / "A2_topic_test_beyond_stance")
        print("  A3/A4: topic comparison and correlations", flush=True)
        render_alignment_bars(rows=rows, partition="topic_train", output_path=topic_train_output_dir / "A3_topic_train_alignment_bars")
        render_correlations(rows=rows, partition="topic_train", output_path=topic_train_output_dir / "A4_topic_train_error_correlations")
        print("  A5: topic convergence", flush=True)
        render_convergence_composite(result_root=result_root, base_result_root=base_result_root, data_root=data_root, partition="topic_train", alpha_spec=alpha_specs[("topic_train", "v0")], output_path=topic_train_output_dir / "A5_topic_train_convergence_sd")
    if args.render_only in {"all", "appendix", "appendix-hist"}:
        print("  A6/A7: focused opinion histograms", flush=True)
        for partition, figure_id in (("group_train", "A6"), ("topic_train", "A7")):
            for kind in ("public", "private"):
                render_histogram(result_root=result_root, base_result_root=base_result_root, data_root=data_root, partition=partition, alpha_spec=alpha_specs[(partition, "v0")], kind=kind, output_path=partition_output_dir(output_dir, partition) / f"{figure_id}_{partition}_{kind}_histograms")
    if args.render_only in {"all", "appendix", "appendix-end"}:
        topic_train_output_dir = partition_output_dir(output_dir, "topic_train")
        print("  A8/A9: topic transitions and partner gaps", flush=True)
        render_transition_panel(result_root=result_root, base_result_root=base_result_root, data_root=data_root, partition="topic_train", alpha_spec=alpha_specs[("topic_train", "v0")], output_path=topic_train_output_dir / "A8_topic_train_transition_matrices")
        render_partner_gap_composite(result_root=result_root, base_result_root=base_result_root, data_root=data_root, partition="topic_train", alpha_spec=alpha_specs[("topic_train", "v0")], output_path=topic_train_output_dir / "A9_topic_train_partner_gap", bootstrap_reps=max(0, int(args.bootstrap_reps)))
    if args.render_only in {"test-all", "test-main"}:
        group_test_output_dir = partition_output_dir(output_dir, "group_test")
        print("Rendering test-set M1–M5 (group test)...", flush=True)
        render_alignment_bars(rows=rows, partition="group_test", output_path=group_test_output_dir / "M1_group_test_alignment_bars")
        render_correlations(rows=rows, partition="group_test", output_path=group_test_output_dir / "M2_group_test_error_correlations")
        render_convergence_composite(result_root=result_root, base_result_root=base_result_root, data_root=data_root, partition="group_test", alpha_spec=alpha_specs[("group_test", "v0")], output_path=group_test_output_dir / "M3_group_test_convergence_sd")
        render_transition_panel(result_root=result_root, base_result_root=base_result_root, data_root=data_root, partition="group_test", alpha_spec=alpha_specs[("group_test", "v0")], output_path=group_test_output_dir / "M4_group_test_transition_matrices")
        render_partner_gap_composite(result_root=result_root, base_result_root=base_result_root, data_root=data_root, partition="group_test", alpha_spec=alpha_specs[("group_test", "v0")], output_path=group_test_output_dir / "M5_group_test_partner_gap", bootstrap_reps=max(0, int(args.bootstrap_reps)))
    if args.render_only in {"test-all", "test-appendix"}:
        group_test_output_dir = partition_output_dir(output_dir, "group_test")
        topic_test_output_dir = partition_output_dir(output_dir, "topic_test")
        print("Rendering test-set appendix counterparts (topic test; group-test A6)...", flush=True)
        render_alignment_bars(rows=rows, partition="topic_test", output_path=topic_test_output_dir / "A3_topic_test_alignment_bars")
        render_correlations(rows=rows, partition="topic_test", output_path=topic_test_output_dir / "A4_topic_test_error_correlations")
        render_convergence_composite(result_root=result_root, base_result_root=base_result_root, data_root=data_root, partition="topic_test", alpha_spec=alpha_specs[("topic_test", "v0")], output_path=topic_test_output_dir / "A5_topic_test_convergence_sd")
        render_histogram(result_root=result_root, base_result_root=base_result_root, data_root=data_root, partition="group_test", alpha_spec=alpha_specs[("group_test", "v0")], kind="public", output_path=group_test_output_dir / "A6_group_test_public_histograms")
        render_histogram(result_root=result_root, base_result_root=base_result_root, data_root=data_root, partition="group_test", alpha_spec=alpha_specs[("group_test", "v0")], kind="private", output_path=group_test_output_dir / "A6_group_test_private_histograms")
        for kind in ("public", "private"):
            render_histogram(result_root=result_root, base_result_root=base_result_root, data_root=data_root, partition="topic_test", alpha_spec=alpha_specs[("topic_test", "v0")], kind=kind, output_path=topic_test_output_dir / f"A7_topic_test_{kind}_histograms")
        render_transition_panel(result_root=result_root, base_result_root=base_result_root, data_root=data_root, partition="topic_test", alpha_spec=alpha_specs[("topic_test", "v0")], output_path=topic_test_output_dir / "A8_topic_test_transition_matrices")
        render_partner_gap_composite(result_root=result_root, base_result_root=base_result_root, data_root=data_root, partition="topic_test", alpha_spec=alpha_specs[("topic_test", "v0")], output_path=topic_test_output_dir / "A9_topic_test_partner_gap", bootstrap_reps=max(0, int(args.bootstrap_reps)))
    if args.render_only == "partner-gap":
        all_partner_gap_panels = (
            ("group_train", "M5_group_train_partner_gap"),
            ("group_test", "M5_group_test_partner_gap"),
            ("topic_train", "A9_topic_train_partner_gap"),
            ("topic_test", "A9_topic_test_partner_gap"),
        )
        selected_partner_gap_panels = [
            (partition, output_stem)
            for partition, output_stem in all_partner_gap_panels
            if args.partner_gap_partition in {None, partition}
        ]
        print(f"Rendering partner-gap panels: {', '.join(partition for partition, _ in selected_partner_gap_panels)}", flush=True)
        for partition, output_stem in selected_partner_gap_panels:
            render_partner_gap_composite(
                result_root=result_root,
                base_result_root=base_result_root,
                data_root=data_root,
                partition=partition,
                alpha_spec=alpha_specs[(partition, "v0")],
                output_path=partition_output_dir(output_dir, partition) / output_stem,
                bootstrap_reps=max(0, int(args.bootstrap_reps)),
            )
    write_partition_readmes(output_dir)
    write_readme(output_dir=output_dir, result_root=result_root, base_result_root=base_result_root, data_root=data_root, statuses=statuses, alpha_specs=alpha_specs, bootstrap_reps=max(0, int(args.bootstrap_reps)))
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
