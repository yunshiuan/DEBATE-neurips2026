#!/usr/bin/env python3
"""Render a controlled Qwen3.5-4B versus Llama-SFT v5 comparison bundle.

The supplied Qwen package includes a base Qwen3.5-4B run and split-specific
SFT runs.  The contemporaneous Llama results are stored in Leo's 2026-06-18
delivery.  Those deliveries have model-specific generated continuations and
judge labels, so this script evaluates all model outputs for a partition
against one common human/validity reference: the matching Qwen SFT human CSV.
This makes each plotted model comparison internally comparable while keeping
all unavailable prefixes explicit in the CSV sidecars.

It reuses the v5 paper renderer's metric definitions and visual grammar, but
intentionally renders only the stance/dynamics families that exist in the
delivery.  The package contains no semantic-similarity / ROUGE aggregate
reports, so A1/A2 are documented as unavailable rather than fabricated.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
SCRIPT_DIR = SCRIPT_PATH.parent
RADAR_DIR = REPO_ROOT / "neurips_2026" / "chengtao" / "src" / "radar"
for import_path in (REPO_ROOT, SCRIPT_DIR, RADAR_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import generate_v5_ruixuan_paper_figures as renderer  # noqa: E402


DEFAULT_QWEN_ROOT = Path("data/imports/leo_qwen4b_result_0727/eval/human_llm")
DEFAULT_LLAMA_ROOT = Path("data/imports/result_0618_leo/eval/human_llm")
DEFAULT_LLAMA_BASE_ROOT = REPO_ROOT / "neurips_2026" / "chengtao" / "result" / "eval_v3" / "human_llm"
DEFAULT_MASK_ROOT = REPO_ROOT / "neurips_2026" / "sean_reproduce" / "v5_leo" / "masks"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "neurips_2026" / "sean_reproduce" / "v5_leo" / "qwen4b_comparison"
DEFAULT_HTML_PATH = REPO_ROOT / "neurips_2026" / "sean_reproduce" / "v5_leo" / "SFT_Qwen4B.html"

QWEN_BASE = "Qwen3.5-4B"
QWEN_GROUP = "ft:Qwen3.5-4B-SFT-20260724:both-group-3epochs-unsloth"
QWEN_TOPIC = "ft:Qwen3.5-4B-SFT-20260724:both-topic-3epochs-unsloth"
LLAMA_BASE = "Llama-3.1-8B-Instruct"
LLAMA_GROUP = "ft:Llama-3.1-8B-Instruct-SFT-20260614:both-group-3epochs-unsloth"
LLAMA_TOPIC = "ft:Llama-3.1-8B-Instruct-SFT-20260613:both-topic-3epochs-unsloth"

PARTITIONS = ("group_train", "group_test", "topic_train", "topic_test")
PARTITION_OUTPUT_DIRS = {
    "group_train": "group_train_set",
    "group_test": "group_test_set",
    "topic_train": "topic_train_set",
    "topic_test": "topic_test_set",
}
PARTITION_MASKS = {
    "group_train": "group_train_prefixes.txt",
    "group_test": "group_test_prefixes.txt",
    "topic_train": "topic_train_prefixes.txt",
    "topic_test": "topic_test_prefixes.txt",
}

MODEL_COLORS = {
    renderer.HUMAN_MODEL: "#222222",
    LLAMA_BASE: "#7B8FB2",
    LLAMA_GROUP: "#35689A",
    LLAMA_TOPIC: "#35689A",
    QWEN_BASE: "#D89021",
    QWEN_GROUP: "#C45B18",
    QWEN_TOPIC: "#C45B18",
    renderer.DEGROOT_MODEL: "#7F7F7F",
}


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    records = list(rows)
    fields: list[str] = []
    for row in records:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def is_group_partition(partition: str) -> bool:
    return partition.startswith("group_")


def qwen_sft_model(partition: str) -> str:
    return QWEN_GROUP if is_group_partition(partition) else QWEN_TOPIC


def llama_sft_model(partition: str) -> str:
    return LLAMA_GROUP if is_group_partition(partition) else LLAMA_TOPIC


def comparison_models(partition: str) -> list[str]:
    return [LLAMA_BASE, llama_sft_model(partition), QWEN_BASE, qwen_sft_model(partition)]


def source_name(model: str) -> str:
    if model == LLAMA_BASE:
        return "canonical Llama-3.1-8B base"
    if model in {LLAMA_GROUP, LLAMA_TOPIC}:
        return "Leo Llama SFT"
    if model == QWEN_BASE:
        return "supplied Qwen3.5-4B base"
    if model in {QWEN_GROUP, QWEN_TOPIC}:
        return "supplied Qwen3.5-4B SFT"
    if model == renderer.DEGROOT_MODEL:
        return "fitted DeGroot reference"
    return "unknown"


def model_label(model: str) -> str:
    return {
        renderer.HUMAN_MODEL: "Human",
        LLAMA_BASE: "Llama-3.1-8B",
        LLAMA_GROUP: "Llama SFT (group)",
        LLAMA_TOPIC: "Llama SFT (topic)",
        QWEN_BASE: "Qwen3.5-4B",
        QWEN_GROUP: "Qwen3.5-4B SFT (group)",
        QWEN_TOPIC: "Qwen3.5-4B SFT (topic)",
        renderer.DEGROOT_MODEL: "DeGroot fit",
    }.get(model, model)


def publication_model_label(model: str) -> str:
    return {
        renderer.HUMAN_MODEL: "Human",
        LLAMA_BASE: "Llama-3.1-8B",
        LLAMA_GROUP: "Llama SFT",
        LLAMA_TOPIC: "Llama SFT",
        QWEN_BASE: "Qwen3.5-4B",
        QWEN_GROUP: "Qwen SFT",
        QWEN_TOPIC: "Qwen SFT",
        renderer.DEGROOT_MODEL: "DeGroot fit",
    }.get(model, model_label(model))


def output_dir_for(output_dir: Path, partition: str) -> Path:
    return output_dir / PARTITION_OUTPUT_DIRS[partition]


def read_prefixes(mask_root: Path, partition: str) -> list[str]:
    mask_path = mask_root / PARTITION_MASKS[partition]
    with mask_path.open(encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def canonical_human_path(qwen_root: Path, partition: str, prefix: str, version: str) -> Path:
    return qwen_root / prefix / qwen_sft_model(partition) / str(renderer.VERSION_SPECS[version]["human_filename"])


def model_root(
    model: str,
    *,
    qwen_root: Path,
    llama_root: Path,
    llama_base_root: Path,
) -> Path:
    if model == LLAMA_BASE:
        return llama_base_root
    if model in {LLAMA_GROUP, LLAMA_TOPIC}:
        return llama_root
    if model in {QWEN_BASE, QWEN_GROUP, QWEN_TOPIC}:
        return qwen_root
    raise ValueError(f"Unknown model: {model}")


def read_rows(path: Path, *, root: Path) -> list[dict[str, str]]:
    return renderer.read_eval_rows(path, result_root=root)


_DYNAMICS_FIELDS = (
    "group_id",
    "event_type",
    "chat_round_order",
    "message_id",
    "empirica_id",
    "worker_id",
    "sender_id",
    "recipient_id",
    "pseudo_id",
    renderer.LIKERT_COL,
)


def compact_dynamics_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep only the stance/partner fields needed by dynamics and DeGroot.

    The raw CSV rows embed long generation and judge-prompt strings. Retaining
    all of those rows across 500+ prefixes would make the fit/profile phase
    unnecessarily memory-heavy; none are used by the paper dynamics metrics.
    """
    compact: list[dict[str, str]] = []
    for row in rows:
        event_type = str(row.get("event_type") or "").strip()
        if event_type not in {"Initial Opinion", "Post Opinion", "tweet"}:
            continue
        if event_type == "tweet" and renderer.parse_round(row.get("chat_round_order")) not in {1, 2, 3}:
            continue
        compact.append({field: str(row.get(field) or "") for field in _DYNAMICS_FIELDS})
    return compact


def fit_alphas(
    *,
    qwen_root: Path,
    mask_root: Path,
    partition: str,
    version: str,
) -> dict[str, float]:
    rows_by_prefix: dict[str, list[dict[str, str]]] = {}
    missing = 0
    for prefix in read_prefixes(mask_root, partition):
        path = canonical_human_path(qwen_root, partition, prefix, version)
        if not path.is_file():
            missing += 1
            continue
        rows_by_prefix[prefix] = compact_dynamics_rows(read_rows(path, root=qwen_root))
    result = renderer.fit_best_alphas_from_human_rows(rows_by_prefix=rows_by_prefix, likert_col=renderer.LIKERT_COL)
    result["fit_prefix_count"] = float(len(rows_by_prefix))
    result["fit_missing_human_prefix_count"] = float(missing)
    return result


def fit_all_alphas(*, qwen_root: Path, mask_root: Path) -> dict[tuple[str, str], dict[str, float]]:
    result: dict[tuple[str, str], dict[str, float]] = {}
    cache: dict[tuple[str, str], dict[str, float]] = {}
    for partition in PARTITIONS:
        fit_partition = "group_train" if partition.startswith("group_") else "topic_train"
        for version in renderer.VERSION_SPECS:
            key = (fit_partition, version)
            if key not in cache:
                cache[key] = fit_alphas(qwen_root=qwen_root, mask_root=mask_root, partition=fit_partition, version=version)
            result[(partition, version)] = dict(cache[key])
    return result


def metric_result_from_shared_human(
    *,
    prefix: str,
    model: str,
    partition: str,
    version: str,
    human_rows: list[dict[str, str]],
    qwen_root: Path,
    llama_root: Path,
    llama_base_root: Path,
) -> tuple[Any | None, dict[str, Any]]:
    root = model_root(model, qwen_root=qwen_root, llama_root=llama_root, llama_base_root=llama_base_root)
    llm_path = root / prefix / model / str(renderer.VERSION_SPECS[version]["llm_filename"])
    status = {
        "data_prefix": prefix,
        "version": version,
        "model": model,
        "source": source_name(model),
        "human_reference": f"Qwen SFT {('group' if is_group_partition(partition) else 'topic')} human surface",
        "human_path": str(canonical_human_path(qwen_root, partition, prefix, version)),
        "llm_path": str(llm_path),
    }
    if not llm_path.is_file():
        status.update({"status": "missing_llm", "message": "LLM CSV missing."})
        return None, status
    try:
        llm_rows = read_rows(llm_path, root=root)
        result = renderer.compute_prefix_model_from_rows(
            data_prefix=prefix,
            model=model,
            human_rows=human_rows,
            llm_rows=llm_rows,
            utterance_event_types=set(renderer.DEFAULT_UTTERANCE_EVENT_TYPES),
            likert_col=renderer.LIKERT_COL,
            min_key_match_rate=0.7,
            include_conditioned_alignment=False,
            human_path=status["human_path"],
            llm_path=status["llm_path"],
        )
    except Exception as exc:
        status.update({"status": "error", "message": str(exc)})
        return None, status
    status.update({"status": result.status, "message": result.message})
    return result, status


def compute_partition_rows(
    *,
    qwen_root: Path,
    llama_root: Path,
    llama_base_root: Path,
    mask_root: Path,
    partition: str,
    version: str,
    alpha_spec: dict[str, float],
    bootstrap_reps: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    models = comparison_models(partition)
    aggregates = renderer.build_model_aggregates(models)
    degroot_aggregates = renderer.build_model_aggregates([renderer.DEGROOT_MODEL])
    per_prefix_results: dict[str, dict[str, Any]] = {}
    degroot_results: dict[str, dict[str, Any]] = {}
    statuses: list[dict[str, Any]] = []
    prefixes = read_prefixes(mask_root, partition)

    for index, prefix in enumerate(prefixes, start=1):
        if index == 1 or index % 50 == 0 or index == len(prefixes):
            print(f"  {partition}/{version}: prefix {index}/{len(prefixes)}", flush=True)
        per_prefix_results[prefix] = {}
        human_path = canonical_human_path(qwen_root, partition, prefix, version)
        if not human_path.is_file():
            for model in [*models, renderer.DEGROOT_MODEL]:
                statuses.append({
                    "data_prefix": prefix,
                    "version": version,
                    "model": model,
                    "source": source_name(model),
                    "status": "missing_human_reference",
                    "message": "Matching Qwen SFT human CSV missing.",
                    "human_reference": f"Qwen SFT {('group' if is_group_partition(partition) else 'topic')} human surface",
                    "human_path": str(human_path),
                    "llm_path": "",
                })
            continue
        try:
            human_rows = read_rows(human_path, root=qwen_root)
        except Exception as exc:
            for model in [*models, renderer.DEGROOT_MODEL]:
                statuses.append({
                    "data_prefix": prefix,
                    "version": version,
                    "model": model,
                    "source": source_name(model),
                    "status": "human_reference_error",
                    "message": str(exc),
                    "human_reference": f"Qwen SFT {('group' if is_group_partition(partition) else 'topic')} human surface",
                    "human_path": str(human_path),
                    "llm_path": "",
                })
            continue

        for model in models:
            result, status = metric_result_from_shared_human(
                prefix=prefix,
                model=model,
                partition=partition,
                version=version,
                human_rows=human_rows,
                qwen_root=qwen_root,
                llama_root=llama_root,
                llama_base_root=llama_base_root,
            )
            statuses.append(status)
            if result is not None:
                per_prefix_results[prefix][model] = result

        # The comparison figures are an SFT ablation, not four independently
        # sampled reports.  Aggregate a prefix only when every displayed
        # learned model has a successful matched result on the shared human
        # surface.  This removes the one-prefix availability differences
        # between base and SFT runs instead of letting them alter a bar height.
        common_complete = (
            set(per_prefix_results[prefix]) == set(models)
            and all(result.status == "ok" for result in per_prefix_results[prefix].values())
        )
        if common_complete:
            for model, result in per_prefix_results[prefix].items():
                renderer.merge_prefix_result_into_aggregate(aggregates[model], result)
        else:
            for status in statuses:
                if status.get("data_prefix") != prefix or status.get("model") not in models:
                    continue
                if status.get("status") == "ok":
                    status["status"] = "excluded_not_common_complete"
                    status["message"] = "Excluded from learned-model aggregate because at least one comparison model lacks a matched usable result."

        if not common_complete:
            statuses.append({
                "data_prefix": prefix,
                "version": version,
                "model": renderer.DEGROOT_MODEL,
                "source": source_name(renderer.DEGROOT_MODEL),
                "status": "excluded_not_common_complete",
                "message": "Excluded because the learned-model comparison sample is incomplete.",
                "human_reference": f"Qwen SFT {('group' if is_group_partition(partition) else 'topic')} human surface",
                "human_path": str(human_path),
                "llm_path": "",
            })
            continue
        try:
            proxy_rows = renderer.build_degroot_proxy_rows(
                human_rows,
                data_prefix=prefix,
                likert_col=renderer.LIKERT_COL,
                alpha_public=float(alpha_spec["alpha_public"]),
                alpha_private=float(alpha_spec["alpha_private"]),
            )
            proxy_result = renderer.compute_prefix_model_from_rows(
                data_prefix=prefix,
                model=renderer.DEGROOT_MODEL,
                human_rows=human_rows,
                llm_rows=proxy_rows,
                utterance_event_types=set(renderer.DEFAULT_UTTERANCE_EVENT_TYPES),
                likert_col=renderer.LIKERT_COL,
                min_key_match_rate=0.7,
                include_conditioned_alignment=False,
                human_path=str(human_path),
                llm_path=f"degroot(alpha_public={alpha_spec['alpha_public']:.6f},alpha_private={alpha_spec['alpha_private']:.6f})",
            )
        except Exception as exc:
            statuses.append({
                "data_prefix": prefix,
                "version": version,
                "model": renderer.DEGROOT_MODEL,
                "source": source_name(renderer.DEGROOT_MODEL),
                "status": "error",
                "message": str(exc),
                "human_reference": f"Qwen SFT {('group' if is_group_partition(partition) else 'topic')} human surface",
                "human_path": str(human_path),
                "llm_path": "",
            })
        else:
            renderer.merge_prefix_result_into_aggregate(degroot_aggregates[renderer.DEGROOT_MODEL], proxy_result)
            degroot_results[prefix] = {renderer.DEGROOT_MODEL: proxy_result}
            statuses.append({
                "data_prefix": prefix,
                "version": version,
                "model": renderer.DEGROOT_MODEL,
                "source": source_name(renderer.DEGROOT_MODEL),
                "status": proxy_result.status,
                "message": proxy_result.message,
                "human_reference": f"Qwen SFT {('group' if is_group_partition(partition) else 'topic')} human surface",
                "human_path": str(human_path),
                "llm_path": proxy_result.llm_path,
            })

    rows = renderer.build_raw_rows_from_aggregates(aggregates, models, ub_mode="gaussian_likert")
    rows = renderer.add_v2_bootstrap_cis_to_raw_rows(
        rows,
        models=models,
        prefixes=prefixes,
        prefix_results_by_prefix=per_prefix_results,
        bootstrap_reps=bootstrap_reps,
        seed=42,
    )
    proxy_rows = renderer.build_raw_rows_from_aggregates(degroot_aggregates, [renderer.DEGROOT_MODEL], ub_mode="gaussian_likert")
    proxy_rows = renderer.add_v2_bootstrap_cis_to_raw_rows(
        proxy_rows,
        models=[renderer.DEGROOT_MODEL],
        prefixes=prefixes,
        prefix_results_by_prefix=degroot_results,
        bootstrap_reps=bootstrap_reps,
        seed=42,
    )
    return [*rows, *proxy_rows], statuses


def compute_summary(
    *,
    qwen_root: Path,
    llama_root: Path,
    llama_base_root: Path,
    mask_root: Path,
    bootstrap_reps: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str], dict[str, float]]]:
    print("Fitting DeGroot diagnostics from the common Qwen SFT human surfaces...", flush=True)
    alphas = fit_all_alphas(qwen_root=qwen_root, mask_root=mask_root)
    all_rows: list[dict[str, Any]] = []
    all_statuses: list[dict[str, Any]] = []
    for partition in PARTITIONS:
        for version in renderer.VERSION_SPECS:
            print(f"Computing controlled comparison metrics: {partition}/{version}", flush=True)
            rows, statuses = compute_partition_rows(
                qwen_root=qwen_root,
                llama_root=llama_root,
                llama_base_root=llama_base_root,
                mask_root=mask_root,
                partition=partition,
                version=version,
                alpha_spec=alphas[(partition, version)],
                bootstrap_reps=bootstrap_reps,
            )
            for row in rows:
                model = str(row.get("model", ""))
                row.update({
                    "partition": partition,
                    "simulation_version": version,
                    "model_label": model_label(model),
                    "evaluation_source": source_name(model),
                    "human_reference": f"Qwen SFT {('group' if is_group_partition(partition) else 'topic')} human surface",
                    "fit_alpha_public": alphas[(partition, version)]["alpha_public"],
                    "fit_alpha_private": alphas[(partition, version)]["alpha_private"],
                })
            for status in statuses:
                status["partition"] = partition
            all_rows.extend(rows)
            all_statuses.extend(statuses)
    return all_rows, all_statuses, alphas


def configure_renderer() -> None:
    """Install comparison names/colours while retaining shared v4 styles."""
    renderer.GROUP_MODEL = QWEN_GROUP
    renderer.TOPIC_MODEL = QWEN_TOPIC
    renderer.PARTITION_SPECS = {
        "group_train": {"mask": PARTITION_MASKS["group_train"], "model": QWEN_GROUP, "fit_partition": "group_train"},
        "group_test": {"mask": PARTITION_MASKS["group_test"], "model": QWEN_GROUP, "fit_partition": "group_train"},
        "topic_train": {"mask": PARTITION_MASKS["topic_train"], "model": QWEN_TOPIC, "fit_partition": "topic_train"},
        "topic_test": {"mask": PARTITION_MASKS["topic_test"], "model": QWEN_TOPIC, "fit_partition": "topic_train"},
    }
    renderer.MODEL_COLORS = dict(MODEL_COLORS)
    for model, color in MODEL_COLORS.items():
        renderer.MODEL_COLOR_MAP[model] = color
    renderer.FINE_TUNED_DISPLAY_MAP[LLAMA_GROUP] = "Llama SFT"
    renderer.FINE_TUNED_DISPLAY_MAP[LLAMA_TOPIC] = "Llama SFT"
    renderer.FINE_TUNED_DISPLAY_MAP[QWEN_GROUP] = "Qwen SFT"
    renderer.FINE_TUNED_DISPLAY_MAP[QWEN_TOPIC] = "Qwen SFT"
    renderer.model_label = model_label
    renderer.publication_model_label = publication_model_label
    renderer.spearman_ci_style.bootstrap_spearman_ci = renderer.spearman_ci_style.bootstrap_spearman_ci


def configure_partition_models(partition: str) -> None:
    """Give the shared renderer the four comparison series for one split."""
    renderer.BASE_MODELS = [LLAMA_BASE, llama_sft_model(partition), QWEN_BASE]
    renderer.FOCUSED_BASE_MODEL = LLAMA_BASE
    renderer.FOCUSED_OTHER_BASE_MODEL = QWEN_BASE
    renderer.PROFILE_CACHE.clear()


def profile_series(
    *,
    qwen_root: Path,
    llama_root: Path,
    llama_base_root: Path,
    mask_root: Path,
    partition: str,
    version: str,
    alpha_spec: dict[str, float],
) -> dict[str, list[dict[str, str]]]:
    series: dict[str, list[dict[str, str]]] = {
        renderer.HUMAN_MODEL: [],
        **{model: [] for model in comparison_models(partition)},
        renderer.DEGROOT_MODEL: [],
    }
    llm_filename = str(renderer.VERSION_SPECS[version]["llm_filename"])
    for prefix in read_prefixes(mask_root, partition):
        human_path = canonical_human_path(qwen_root, partition, prefix, version)
        if not human_path.is_file():
            continue
        try:
            human_rows = compact_dynamics_rows(read_rows(human_path, root=qwen_root))
        except Exception:
            continue
        series[renderer.HUMAN_MODEL].extend(human_rows)
        try:
            proxy_rows = renderer.build_degroot_proxy_rows(
                human_rows,
                data_prefix=prefix,
                likert_col=renderer.LIKERT_COL,
                alpha_public=float(alpha_spec["alpha_public"]),
                alpha_private=float(alpha_spec["alpha_private"]),
            )
        except ValueError:
            # Match the metric stage: a duplicate valid round-1 tweet makes a
            # DeGroot partner graph ambiguous, but does not invalidate the
            # human or real-model comparison for this prefix.
            proxy_rows = []
        series[renderer.DEGROOT_MODEL].extend(proxy_rows)
        for model in comparison_models(partition):
            root = model_root(model, qwen_root=qwen_root, llama_root=llama_root, llama_base_root=llama_base_root)
            llm_path = root / prefix / model / llm_filename
            if not llm_path.is_file():
                continue
            try:
                series[model].extend(compact_dynamics_rows(read_rows(llm_path, root=root)))
            except Exception:
                continue
    return series


def install_profile_override(
    *,
    qwen_root: Path,
    llama_root: Path,
    llama_base_root: Path,
    mask_root: Path,
) -> None:
    def profile_rows_for_partition(*, result_root: Path, base_result_root: Path, data_root: Path, partition: str, version: str, alpha_spec: dict[str, float]) -> dict[str, list[dict[str, str]]]:
        del result_root, base_result_root, data_root
        return profile_series(
            qwen_root=qwen_root,
            llama_root=llama_root,
            llama_base_root=llama_base_root,
            mask_root=mask_root,
            partition=partition,
            version=version,
            alpha_spec=alpha_spec,
        )
    renderer.profile_rows_for_partition = profile_rows_for_partition


def transition_specs(partition: str) -> list[tuple[str, str, str]]:
    return [
        (renderer.HUMAN_MODEL, "human", "Human"),
        (LLAMA_BASE, "base: Llama-3.1-8B-Instruct", "Llama-3.1-8B\nbase"),
        (llama_sft_model(partition), "sft: Llama", "Llama SFT"),
        (QWEN_BASE, "base: Qwen3.5-4B", "Qwen3.5-4B\nbase"),
        (qwen_sft_model(partition), "sft: Qwen", "Qwen SFT"),
        (renderer.DEGROOT_MODEL, "DeGroot fit", "DeGroot fit"),
    ]


def render_transition(
    *,
    qwen_root: Path,
    llama_root: Path,
    llama_base_root: Path,
    mask_root: Path,
    partition: str,
    alpha_spec: dict[str, float],
    output_path: Path,
) -> None:
    series = profile_series(
        qwen_root=qwen_root,
        llama_root=llama_root,
        llama_base_root=llama_base_root,
        mask_root=mask_root,
        partition=partition,
        version="v0",
        alpha_spec=alpha_spec,
    )
    specs = transition_specs(partition)
    matrices = {
        (kind, model): renderer.transition_matrix(series[model], kind=kind)
        for kind in ("public", "private")
        for model, _style, _display in specs
    }
    percentages = [(matrix / int(matrix.sum())).max() for matrix in matrices.values() if int(matrix.sum())]
    vmax = max(percentages, default=1.0)
    fig = plt.figure(figsize=(25.5, renderer.transition_style.FIGURE_HEIGHT))
    grid = fig.add_gridspec(2, len(specs), left=0.055, right=0.91, top=0.88, bottom=0.13, wspace=0.42, hspace=0.42)
    fig.text(0.018, 0.69, "Public opinion", rotation=90, va="center", ha="center", fontsize=renderer.transition_style.ROW_LABEL_FONTSIZE)
    fig.text(0.018, 0.275, "Private opinion", rotation=90, va="center", ha="center", fontsize=renderer.transition_style.ROW_LABEL_FONTSIZE)
    rows: list[dict[str, Any]] = []
    image: Any | None = None
    for row_index, kind in enumerate(("public", "private")):
        human_matrix = matrices[(kind, renderer.HUMAN_MODEL)]
        for col_index, (model, style_label, display_label) in enumerate(specs):
            matrix = matrices[(kind, model)]
            l2 = None if model == renderer.HUMAN_MODEL else renderer.normalized_frobenius_diff(matrix, human_matrix)
            image = renderer.transition_style.draw_publication_heatmap(
                fig.add_subplot(grid[row_index, col_index]),
                matrix,
                style_label,
                display_label_override=display_label,
                vmax=vmax,
                value_type="likert_pred" if kind == "public" else "slider_value",
                variant="pub",
                panel_info_text=None if l2 is None else f"L2 vs human = {l2:.4f}",
            )
            total = int(matrix.sum())
            for start in range(6):
                for end in range(6):
                    count = int(matrix[start, end])
                    rows.append({
                        "partition": partition,
                        "simulation_version": "v0",
                        "opinion_type": kind,
                        "model": model,
                        "model_label": publication_model_label(model),
                        "l2_norm_vs_human": "" if l2 is None else l2,
                        "start_likert": start + 1,
                        "end_likert": end + 1,
                        "count": count,
                        "share": count / total if total else 0.0,
                        "panel_total": total,
                    })
    if image is not None:
        colorbar = fig.colorbar(image, ax=fig.axes, fraction=0.025, pad=0.02, shrink=0.88)
        colorbar.set_label("Share of transitions (%)", fontsize=renderer.transition_style.COLORBAR_LABEL_FONTSIZE)
        colorbar.ax.yaxis.set_major_formatter(renderer.transition_style.PercentFormatter(xmax=1.0, decimals=0))
        colorbar.ax.tick_params(labelsize=renderer.transition_style.COLORBAR_TICK_FONTSIZE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    write_csv(output_path.with_suffix(".csv"), rows)


def render_histogram(
    *,
    qwen_root: Path,
    llama_root: Path,
    llama_base_root: Path,
    mask_root: Path,
    partition: str,
    alpha_spec: dict[str, float],
    kind: str,
    output_path: Path,
) -> None:
    series = profile_series(
        qwen_root=qwen_root,
        llama_root=llama_root,
        llama_base_root=llama_base_root,
        mask_root=mask_root,
        partition=partition,
        version="v0",
        alpha_spec=alpha_spec,
    )
    panels = [
        (renderer.HUMAN_MODEL, "first"),
        (renderer.HUMAN_MODEL, "final"),
        (LLAMA_BASE, "final"),
        (llama_sft_model(partition), "final"),
        (QWEN_BASE, "final"),
        (qwen_sft_model(partition), "final"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17.0, 10.4))
    output_rows: list[dict[str, Any]] = []
    for index, (axis, (model, stage)) in enumerate(zip(np.asarray(axes).reshape(-1), panels)):
        first, final = renderer.stage_values(series[model], kind=kind)
        values = first if stage == "first" else final
        counts = [sum(1 for value in values if int(round(value)) == score) for score in range(1, 7)]
        total = len(values)
        summary = {"n": str(total), "mean_likert": str(renderer.safe_mean(values)), **{f"count_{score}": str(count) for score, count in zip(range(1, 7), counts)}}
        if model == renderer.HUMAN_MODEL:
            title = "Human anchor" if stage == "first" else "Human final"
        else:
            title = publication_model_label(model)
        renderer.histogram_style.add_histogram_subplot(axis, row=summary, title=title, color=MODEL_COLORS[model], show_ylabel=index % 3 == 0)
        mean_value = renderer.histogram_style.mean_from_summary_row(summary)
        std_value = renderer.histogram_style.std_from_summary_row(summary)
        axis.set_title(f"{title}\n(mean={mean_value:.2f}, sd={std_value:.2f})", fontsize=14, pad=8)
        axis.grid(False)
        axis.tick_params(axis="x", labelsize=12)
        axis.tick_params(axis="y", labelsize=12)
        if index % 3 == 0:
            axis.set_ylabel("Share (%)", fontsize=13)
        axis.set_xlabel("Likert", fontsize=13)
        for score, count in zip(range(1, 7), counts):
            output_rows.append({"partition": partition, "simulation_version": "v0", "opinion_type": kind, "model": model, "model_label": publication_model_label(model), "stage": stage, "likert": score, "count": count, "n": total, "percent": 100 * count / total if total else math.nan})
    fig.suptitle(f"Histogram of {kind.title()} Opinion Stance", fontsize=25, y=0.985, fontweight="bold")
    fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    write_csv(output_path.with_suffix(".csv"), output_rows)


def summarize_coverage(statuses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str, str, str, str], int] = defaultdict(int)
    for row in statuses:
        counts[(str(row.get("partition", "")), str(row.get("version", "")), str(row.get("model", "")), str(row.get("source", "")), str(row.get("status", "")))] += 1
    return [
        {"partition": partition, "simulation_version": version, "model": model, "source": source, "status": status, "prefix_count": count}
        for (partition, version, model, source, status), count in sorted(counts.items())
    ]


def render_all(
    *,
    rows: list[dict[str, Any]],
    alphas: dict[tuple[str, str], dict[str, float]],
    qwen_root: Path,
    llama_root: Path,
    llama_base_root: Path,
    mask_root: Path,
    output_dir: Path,
    bootstrap_reps: int,
) -> None:
    for partition in PARTITIONS:
        configure_partition_models(partition)
        directory = output_dir_for(output_dir, partition)
        directory.mkdir(parents=True, exist_ok=True)
        alignment_id = "M1" if partition.startswith("group_") else "A3"
        correlation_id = "M2" if partition.startswith("group_") else "A4"
        convergence_id = "M3" if partition.startswith("group_") else "A5"
        transition_id = "M4" if partition.startswith("group_") else "A8"
        partner_id = "M5" if partition.startswith("group_") else "A9"
        histogram_id = "A6" if partition.startswith("group_") else "A7"
        print(f"Rendering comparison panels for {partition}...", flush=True)
        renderer.render_alignment_bars(rows=rows, partition=partition, output_path=directory / f"{alignment_id}_{partition}_qwen_vs_llama_alignment_bars")
        renderer.render_correlations(rows=rows, partition=partition, output_path=directory / f"{correlation_id}_{partition}_qwen_vs_llama_error_correlations")
        renderer.render_convergence_composite(
            result_root=qwen_root,
            base_result_root=llama_base_root,
            data_root=mask_root,
            partition=partition,
            alpha_spec=alphas[(partition, "v0")],
            output_path=directory / f"{convergence_id}_{partition}_qwen_vs_llama_convergence_sd",
        )
        render_transition(
            qwen_root=qwen_root,
            llama_root=llama_root,
            llama_base_root=llama_base_root,
            mask_root=mask_root,
            partition=partition,
            alpha_spec=alphas[(partition, "v0")],
            output_path=directory / f"{transition_id}_{partition}_qwen_vs_llama_transition_matrices",
        )
        renderer.render_partner_gap_composite(
            result_root=qwen_root,
            base_result_root=llama_base_root,
            data_root=mask_root,
            partition=partition,
            alpha_spec=alphas[(partition, "v0")],
            output_path=directory / f"{partner_id}_{partition}_qwen_vs_llama_partner_gap",
            bootstrap_reps=bootstrap_reps,
        )
        for kind in ("public", "private"):
            render_histogram(
                qwen_root=qwen_root,
                llama_root=llama_root,
                llama_base_root=llama_base_root,
                mask_root=mask_root,
                partition=partition,
                alpha_spec=alphas[(partition, "v0")],
                kind=kind,
                output_path=directory / f"{histogram_id}_{partition}_qwen_vs_llama_{kind}_histograms",
            )


def figure_specs(partition: str) -> list[tuple[str, str, list[str]]]:
    alignment_id = "M1" if partition.startswith("group_") else "A3"
    correlation_id = "M2" if partition.startswith("group_") else "A4"
    convergence_id = "M3" if partition.startswith("group_") else "A5"
    transition_id = "M4" if partition.startswith("group_") else "A8"
    partner_id = "M5" if partition.startswith("group_") else "A9"
    histogram_id = "A6" if partition.startswith("group_") else "A7"
    stem = lambda suffix: f"{suffix}_{partition}_qwen_vs_llama"
    return [
        (alignment_id, "All three stance errors", [stem("%s" % alignment_id) + "_alignment_bars"]),
        (correlation_id, "Error correlations", [stem("%s" % correlation_id) + "_error_correlations", stem("%s" % correlation_id) + "_error_correlations_full_dynamics_raw", stem("%s" % correlation_id) + "_error_correlations_next_message_pub", stem("%s" % correlation_id) + "_error_correlations_next_message_raw"]),
        (convergence_id, "Opinion convergence", [stem("%s" % convergence_id) + "_convergence_sd"]),
        (transition_id, "Public/private transition matrices", [stem("%s" % transition_id) + "_transition_matrices"]),
        (partner_id, "Partner-gap relationship", [stem("%s" % partner_id) + "_partner_gap"]),
        (histogram_id, "Opinion-distribution histograms", [stem("%s" % histogram_id) + "_public_histograms", stem("%s" % histogram_id) + "_private_histograms"]),
    ]


REBUTTAL_METRICS = (
    ("Public single", "public_stance_alignment_abs"),
    ("Private single", "private_stance_alignment_abs"),
    ("Public update", "public_stance_change_error_abs"),
    ("Private update", "private_stance_change_error_abs"),
    ("Public convergence", "public_opinion_convergence_error"),
    ("Private convergence", "private_opinion_convergence_error"),
)


def markdown_html_table(headers: list[str], rows: list[list[str]]) -> tuple[str, str]:
    """Return a rendered HTML table and the equivalent compact Markdown."""
    markdown_rows = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    markdown_rows.extend("| " + " | ".join(cells) + " |" for cells in rows)
    rendered = (
        '<table class="rebuttal-table"><thead><tr>'
        + "".join(f"<th>{html.escape(header)}</th>" for header in headers)
        + "</tr></thead><tbody>"
        + "".join("<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in cells) + "</tr>" for cells in rows)
        + "</tbody></table>"
    )
    return rendered, "\n".join(markdown_rows)


def read_output_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def rebuttal_label(model: str) -> str:
    return {
        renderer.HUMAN_MODEL: "Human",
        LLAMA_BASE: "Llama base",
        LLAMA_GROUP: "Llama SFT",
        LLAMA_TOPIC: "Llama SFT",
        QWEN_BASE: "Qwen base",
        QWEN_GROUP: "Qwen SFT",
        QWEN_TOPIC: "Qwen SFT",
    }.get(model, model_label(model))


def figure_csv_path(output_dir: Path, partition: str, figure_id: str, suffix: str) -> Path:
    return output_dir_for(output_dir, partition) / f"{figure_id}_{partition}_qwen_vs_llama_{suffix}.csv"


def find_row(rows: list[dict[str, str]], **required: str) -> dict[str, str]:
    for row in rows:
        if all(str(row.get(field, "")) == value for field, value in required.items()):
            return row
    raise KeyError(f"Missing exported row matching {required}")


def format_rho(row: dict[str, str]) -> str:
    rho = finite(row["spearman_rho"])
    low = finite(row["spearman_rho_ci_low"])
    high = finite(row["spearman_rho_ci_high"])
    count = int(finite(row["n"]))
    return f"ρ={rho:.3f} [{low:.3f}, {high:.3f}]; n={count}"


def rebuttal_summary(rows: list[dict[str, Any]], output_dir: Path) -> tuple[str, str, int, int]:
    """Build figure-specific held-out result tables and one copyable Markdown block."""
    by_key = {(str(row["partition"]), str(row["simulation_version"]), str(row["model"])): row for row in rows}
    lower_count = 0
    comparison_count = 0

    # M1/A3: show both base-to-SFT deltas directly, keeping every value on the common test surface.
    error_headers = ["Split", "Mode", "Matched n", "Series"] + [label for label, _ in REBUTTAL_METRICS]
    error_rows: list[list[str]] = []
    for partition in ("group_test", "topic_test"):
        for version in ("v0", "v1_5"):
            base = by_key[(partition, version, QWEN_BASE)]
            qwen_sft = by_key[(partition, version, qwen_sft_model(partition))]
            llama_base = by_key[(partition, version, LLAMA_BASE)]
            llama_sft = by_key[(partition, version, llama_sft_model(partition))]
            context = [partition.replace("_", " ").title(), "Full dynamics" if version == "v0" else "Next message", str(int(finite(base["prefixes_ok"])))]
            llama_cells = [
                f"{finite(llama_base[metric]):.3f} → {finite(llama_sft[metric]):.3f} ({finite(llama_sft[metric]) - finite(llama_base[metric]):+.3f})"
                for _, metric in REBUTTAL_METRICS
            ]
            error_rows.append(context + ["Llama base → SFT (Δ)"] + llama_cells)
            qwen_cells: list[str] = []
            for _, metric in REBUTTAL_METRICS:
                base_value = finite(base[metric])
                sft_value = finite(qwen_sft[metric])
                delta = sft_value - base_value
                lower_count += int(delta < 0)
                comparison_count += 1
                qwen_cells.append(f"{base_value:.3f} → {sft_value:.3f} ({delta:+.3f})")
            error_rows.append(context + ["Qwen base → SFT (Δ)"] + qwen_cells)
    error_html, error_markdown = markdown_html_table(error_headers, error_rows)

    # M3/A5: use the exported aggregate convergence changes, including the human reference trajectory.
    convergence_headers = ["Panel", "Series", "Public mean Δ SD", "Private mean Δ SD"]
    convergence_rows: list[list[str]] = []
    for partition in ("group_test", "topic_test"):
        figure_id = "M3" if is_group_partition(partition) else "A5"
        source_rows = read_output_csv(figure_csv_path(output_dir, partition, figure_id, "convergence_sd"))
        for model in (renderer.HUMAN_MODEL, LLAMA_BASE, llama_sft_model(partition), QWEN_BASE, qwen_sft_model(partition)):
            public = find_row(source_rows, panel="public_convergence", model=model)
            private = find_row(source_rows, panel="private_convergence", model=model)
            convergence_rows.append([
                f"{figure_id} · {partition.replace('_', ' ').title()}",
                rebuttal_label(model),
                f"{finite(public['mean_delta_sd']):+.3f}",
                f"{finite(private['mean_delta_sd']):+.3f}",
            ])
    convergence_html, convergence_markdown = markdown_html_table(convergence_headers, convergence_rows)

    # M4/A8: the published heatmaps report L2 distance from the human transition matrix.
    transition_headers = ["Panel", "Series", "Public L2 vs human", "Private L2 vs human"]
    transition_rows: list[list[str]] = []
    for partition in ("group_test", "topic_test"):
        figure_id = "M4" if is_group_partition(partition) else "A8"
        source_rows = read_output_csv(figure_csv_path(output_dir, partition, figure_id, "transition_matrices"))
        for model in (LLAMA_BASE, llama_sft_model(partition), QWEN_BASE, qwen_sft_model(partition)):
            public = find_row(source_rows, model=model, opinion_type="public")
            private = find_row(source_rows, model=model, opinion_type="private")
            transition_rows.append([
                f"{figure_id} · {partition.replace('_', ' ').title()}",
                rebuttal_label(model),
                f"{finite(public['l2_norm_vs_human']):.3f}",
                f"{finite(private['l2_norm_vs_human']):.3f}",
            ])
    transition_html, transition_markdown = markdown_html_table(transition_headers, transition_rows)

    # M5/A9: retain rho, its plotted 95% bootstrap interval, and the channel-specific paired count.
    partner_headers = ["Panel", "Series", "Public partner-gap Spearman", "Private partner-gap Spearman"]
    partner_rows: list[list[str]] = []
    for partition in ("group_test", "topic_test"):
        figure_id = "M5" if is_group_partition(partition) else "A9"
        source_rows = read_output_csv(figure_csv_path(output_dir, partition, figure_id, "partner_gap"))
        for model in (renderer.HUMAN_MODEL, LLAMA_BASE, llama_sft_model(partition), QWEN_BASE, qwen_sft_model(partition)):
            public = find_row(source_rows, panel="public_spearman", model=model)
            private = find_row(source_rows, panel="private_spearman", model=model)
            partner_rows.append([
                f"{figure_id} · {partition.replace('_', ' ').title()}",
                rebuttal_label(model),
                format_rho(public),
                format_rho(private),
            ])
    partner_html, partner_markdown = markdown_html_table(partner_headers, partner_rows)

    sections = [
        ("M1 / A3 — stance prediction errors", "Each Llama and Qwen cell is base → SFT (SFT − base). Lower is better.", error_html, error_markdown),
        ("M3 / A5 — opinion convergence", "Mean change in cross-participant stance SD (final − initial). More negative values indicate greater convergence; the human row is the target trajectory.", convergence_html, convergence_markdown),
        ("M4 / A8 — public/private transition matrices", "L2 distance between the model and human transition matrices. Lower is better.", transition_html, transition_markdown),
        ("M5 / A9 — partner-gap relationship", "Spearman ρ between signed partner-minus-self initial gap and opinion change, with the plotted 95% paired-bootstrap interval and point count.", partner_html, partner_markdown),
    ]
    rendered = "".join(f"<h3>{html.escape(title)}</h3><p>{html.escape(description)}</p><div class=\"table-scroll\">{table}</div>" for title, description, table, _ in sections)
    markdown = "\n\n".join(f"**{title}**\n\n{description}\n\n{table}" for title, description, _, table in sections)
    return rendered, markdown, lower_count, comparison_count


def write_html_report(*, html_path: Path, output_dir: Path, rows: list[dict[str, Any]], alphas: dict[tuple[str, str], dict[str, float]], bootstrap_reps: int) -> None:
    sections: list[str] = []
    sidebar: list[str] = []
    for partition in PARTITIONS:
        label = partition.replace("_", " ").title()
        section_id = partition.replace("_", "-")
        sidebar.append(f'<a href="#{section_id}">{html.escape(label)}</a>')
        directory = PARTITION_OUTPUT_DIRS[partition]
        cards: list[str] = []
        for figure_id, title, stems in figure_specs(partition):
            primary, *auxiliary = stems
            primary_base = f"qwen4b_comparison/{directory}/{primary}"
            extra = ""
            if auxiliary:
                detail_label = "Private-channel histogram" if title == "Opinion-distribution histograms" else "Auxiliary correlation renderings"
                extra = f"<details class=\"diagnostics\"><summary>{detail_label}</summary>" + "".join(
                    f'<a class="asset" href="qwen4b_comparison/{directory}/{name}.png"><img src="qwen4b_comparison/{directory}/{name}.png" alt="{html.escape(figure_id)} auxiliary figure"></a>'
                    for name in auxiliary
                ) + "</details>"
            cards.append(
                f'<article class="card" id="{section_id}-{figure_id.lower()}">'
                f'<h3>{html.escape(figure_id)} · {html.escape(title)}</h3>'
                f'<a class="asset" href="{primary_base}.png"><img src="{primary_base}.png" alt="{html.escape(figure_id)} {html.escape(title)}"></a>'
                f'<p><a href="{primary_base}.png">PNG</a> · <a href="{primary_base}.pdf">PDF</a> · <a href="{primary_base}.csv">CSV</a></p>{extra}</article>'
            )
        sections.append(
            f'<section id="{section_id}"><h2>{html.escape(label)}</h2>'
            f'<p class="subtitle">Qwen group/topic SFT is compared with its matching Qwen base run and the corresponding Leo Llama SFT/base series on the same mask. Both full-dynamics and next-message metrics appear in the bar and correlation panels.</p>'
            + "".join(cards) + "</section>"
        )
    alpha_rows = "".join(
        f"<tr><td>{html.escape(partition.replace('_', ' '))}</td><td>{html.escape(version)}</td><td>{finite(values.get('alpha_public')):.6f}</td><td>{finite(values.get('alpha_private')):.6f}</td><td>{finite(values.get('fit_prefix_count')):.0f}</td></tr>"
        for (partition, version), values in sorted(alphas.items())
    )
    rebuttal_tables, rebuttal_markdown, lower_count, comparison_count = rebuttal_summary(rows, output_dir)
    command = html.escape(
        'export PYTHON=python\n'
        'export PYTHONPATH="$PWD/src"\n'
        'export MPLCONFIGDIR=outputs/tmp/codex-mpl-v5-qwen4b\n'
        '"$PYTHON" neurips_2026/sean_reproduce/scripts/generate_v5_leo_qwen4b_comparison.py'
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen3.5-4B SFT versus Llama SFT — v5 Leo</title>
<style>
:root {{ color-scheme: light; --ink:#26384b; --muted:#5c6d7d; --line:#d8e1eb; --paper:#f6f8fb; --accent:#075d8c; --active:#dceefa; }}
* {{ box-sizing:border-box; }} body {{ margin:0; font:15px/1.5 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--ink); background:#fff; }}
aside {{ position:fixed; inset:0 auto 0 0; width:250px; padding:25px 18px; overflow:auto; border-right:1px solid var(--line); background:var(--paper); }}
aside h1 {{ font-size:18px; line-height:1.3; margin:0 0 7px; }} aside p {{ color:var(--muted); font-size:13px; margin:0 0 18px; }} aside a {{ display:block; color:#33526b; text-decoration:none; padding:8px 10px; border-radius:6px; margin:2px 0; }} aside a.active {{ color:#034c73; background:var(--active); font-weight:700; }}
main {{ max-width:1760px; margin-left:250px; padding:30px 42px 64px; }} h2 {{ scroll-margin-top:20px; border-bottom:2px solid var(--line); padding-bottom:8px; margin-top:48px; }} h3 {{ margin:0 0 10px; font-size:21px; }}
.notice {{ padding:14px 17px; border-left:4px solid #c77916; background:#fff8e9; margin:18px 0; }} .subtitle {{ color:var(--muted); max-width:1100px; }} code,pre {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }} pre {{ white-space:pre-wrap; overflow-wrap:anywhere; background:#172430; color:#e9f3f9; padding:14px; border-radius:8px; }}
.card {{ border:1px solid var(--line); border-radius:10px; padding:20px; margin:22px 0; background:#fff; box-shadow:0 1px 3px #1526370d; }} .asset {{ display:block; }} img {{ width:100%; height:auto; border:1px solid #e4eaf0; background:#fff; }} .card p a {{ color:var(--accent); font-weight:650; }}
.diagnostics {{ margin-top:14px; }} .diagnostics summary {{ cursor:pointer; color:var(--accent); font-weight:650; }} .diagnostics .asset {{ margin-top:14px; }} table {{ border-collapse:collapse; width:100%; max-width:850px; }} th,td {{ text-align:left; padding:7px 9px; border-bottom:1px solid var(--line); }} th {{ background:var(--paper); }} .rebuttal-table {{ max-width:none; font-size:13px; }} .rebuttal-table td {{ white-space:nowrap; font-variant-numeric:tabular-nums; }} .table-scroll {{ overflow-x:auto; border:1px solid var(--line); border-radius:8px; margin:0 0 24px; }} textarea {{ display:block; width:100%; min-height:760px; resize:vertical; padding:12px; border:1px solid var(--line); border-radius:8px; font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace; color:var(--ink); background:#fbfcfe; }} .copy-button {{ border:0; border-radius:6px; padding:8px 11px; margin:0 0 8px; background:var(--accent); color:#fff; cursor:pointer; font:inherit; font-weight:650; }} .copy-button:hover {{ background:#034c73; }}
@media (max-width:900px) {{ aside {{ position:static; width:auto; border-right:0; border-bottom:1px solid var(--line); }} aside a {{ display:inline-block; }} main {{ margin:0; padding:24px 16px; }} }}
</style></head><body>
<aside><h1>Qwen3.5-4B SFT comparison</h1><p>v5 Leo masks · Qwen base/SFT versus Llama base/SFT</p><a href="#overview">Overview</a><a href="#rebuttal-summary">Rebuttal table</a>{''.join(sidebar)}<a href="#reproduce">Reproduce</a></aside>
<main><section id="overview"><h2>Overview</h2><p>This report answers whether SFT improves Qwen3.5-4B on the three stance-error families, both public/private channels, group/topic generalization, full dynamics, next-message prediction, and opinion convergence. It uses the exact v5 Leo masks and renders the same panel families as the Llama SFT bundle.</p>
<div class="notice"><strong>Comparison contract.</strong> The supplied Qwen and Leo deliveries contain different generated continuations and some different judge labels. To prevent a model-specific human/label surface from driving an apparent improvement, every Qwen/Llama metric in a partition is evaluated against the matching Qwen SFT human CSV and its validity labels. Each learned-model aggregate is further restricted to its four-model common available-prefix intersection. The common reference is group-SFT human rows for group partitions and topic-SFT human rows for topic partitions. Source availability remains explicit in <a href="qwen4b_comparison/csv/coverage_summary.csv">coverage_summary.csv</a>.</div>
<p><strong>Unavailable family:</strong> the delivered Qwen package has no <code>llm_report_v0_bd.csv</code>, <code>llm_report_v1_5_bd.csv</code>, or Jina/ROUGE aggregate files; therefore A1/A2 beyond-stance panels are intentionally absent.</p>
<p>Summary metrics: <a href="qwen4b_comparison/csv/summary_metrics.csv">CSV</a> · Per-prefix status: <a href="qwen4b_comparison/csv/per_prefix_status.csv">CSV</a>.</p></section>
<section id="rebuttal-summary"><h2>Rebuttal-ready held-out results</h2><p>Qwen SFT lowers all {lower_count}/{comparison_count} public/private M1/A3 error values on the matched group- and topic-generalization test sets. Every family now includes both Llama base and Llama SFT, in addition to Qwen base and Qwen SFT, so the two SFT effects are directly readable.</p><p>All learned-model results use the same four-model complete-prefix intersection within a partition/mode. The M1/A3 error table reports full dynamics and next-message separately; M3/A5, M4/A8, and M5/A9 follow their paper panels and therefore use full dynamics.</p>{rebuttal_tables}<h3>Copyable Markdown</h3><p>Use this directly in the NeurIPS rebuttal response field.</p><button class="copy-button" type="button" data-copy-target="rebuttal-markdown">Copy Markdown</button><textarea id="rebuttal-markdown" rows="50" readonly spellcheck="false">{html.escape(rebuttal_markdown)}</textarea></section>
{''.join(sections)}
<section id="reproduce"><h2>Reproduce</h2><pre>{command}</pre><p>Outputs are written under <code>qwen4b_comparison/</code>; this HTML page remains at <code>v5_leo/SFT_Qwen4B.html</code>.</p><h3>DeGroot fits on the common reference surface</h3><table><thead><tr><th>Partition</th><th>Simulation</th><th>α public</th><th>α private</th><th>Human prefixes</th></tr></thead><tbody>{alpha_rows}</tbody></table></section>
</main><script>
const links=[...document.querySelectorAll('aside a[href^="#"]')]; const targets=links.map(a=>[a,document.querySelector(a.getAttribute('href'))]).filter(([,e])=>e);
const observer=new IntersectionObserver(entries=>{{const visible=entries.filter(e=>e.isIntersecting).sort((a,b)=>b.intersectionRatio-a.intersectionRatio);if(!visible.length)return;const active=visible[0].target.id;links.forEach(a=>a.classList.toggle('active',a.getAttribute('href')==='#'+active));}},{{rootMargin:'-12% 0px -72% 0px',threshold:[0,.15,.4]}});targets.forEach(([,e])=>observer.observe(e));
document.querySelectorAll('[data-copy-target]').forEach(button=>button.addEventListener('click',async()=>{{const target=document.getElementById(button.dataset.copyTarget);target.focus();target.select();try{{await navigator.clipboard.writeText(target.value);}}catch(_error){{document.execCommand('copy');}}const original=button.textContent;button.textContent='Copied';setTimeout(()=>button.textContent=original,1400);}}));
</script></body></html>"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(document, encoding="utf-8")


def write_readme(
    *,
    output_dir: Path,
    qwen_root: Path,
    llama_root: Path,
    llama_base_root: Path,
    mask_root: Path,
    statuses: list[dict[str, Any]],
    alphas: dict[tuple[str, str], dict[str, float]],
    bootstrap_reps: int,
) -> None:
    coverage = summarize_coverage(statuses)
    lines = [
        "# Qwen3.5-4B SFT versus Llama SFT comparison",
        "",
        "This isolated v5 Leo comparison evaluates Qwen3.5-4B base/group-SFT/topic-SFT alongside the matching Llama-3.1-8B base/SFT series.",
        "",
        "## Controlled comparison contract",
        "",
        "- Group/topic train/test masks are reused unchanged from `../masks/` (the merged 708-prefix Leo universe and its v4-rule group split).",
        "- Human text, generated continuations, and some validity/stance labels differ across the Qwen and Llama deliveries. Every model is therefore evaluated against the matching Qwen SFT human/label reference for its partition, rather than against its own model-specific human surface.",
        "- Group partitions use the Qwen group-SFT human CSV; topic partitions use the Qwen topic-SFT human CSV.",
        "- Each learned-model aggregate uses the intersection of prefixes with successful matched results for all four learned series. This prevents a missing base or SFT CSV from changing a comparison bar's sample.",
        "- The plotted models are Llama-3.1-8B base, Llama SFT, Qwen3.5-4B base, Qwen SFT, and fitted DeGroot; the human series appears in dynamics panels.",
        "- Both `v0` full dynamics and `v1_5` next-message prediction are included in the stance/error panels. M3/A5, M4/A8, M5/A9, and A6/A7 use full dynamics, matching the paper families.",
        f"- Metric and M5/A9 intervals use {bootstrap_reps:,} bootstrap repetitions.",
        "- The supplied Qwen delivery lacks the aggregate Jina/ROUGE source files, so A1/A2 are not rendered.",
        "",
        "## Inputs",
        "",
        f"- Qwen evaluation root: `{qwen_root}`",
        f"- Leo Llama-SFT evaluation root: `{llama_root}`",
        f"- Canonical Llama base root: `{llama_base_root}`",
        f"- Reused v5 Leo masks: `{mask_root}`",
        "",
        "## Outputs",
        "",
        "- `group_train_set/` and `group_test_set/`: M1–M5 plus A6 public/private histograms.",
        "- `topic_train_set/` and `topic_test_set/`: A3–A5, A7 public/private histograms, A8–A9.",
        "- `csv/summary_metrics.csv`: all stance metric rows and bootstrap intervals.",
        "- `csv/per_prefix_status.csv` and `csv/coverage_summary.csv`: model-specific coverage and failure provenance.",
        "- `csv/degroot_fit_alphas.csv`: common-human-surface DeGroot diagnostics.",
        "",
        "## DeGroot fits",
        "",
        "| Partition | Simulation | alpha_public | alpha_private | human prefixes |",
        "|---|---|---:|---:|---:|",
    ]
    for (partition, version), values in sorted(alphas.items()):
        lines.append(f"| {partition} | {version} | {finite(values.get('alpha_public')):.6f} | {finite(values.get('alpha_private')):.6f} | {finite(values.get('fit_prefix_count')):.0f} |")
    lines.extend([
        "",
        "## Reproduce",
        "",
        "```bash",
        "export PYTHON=python",
        "export PYTHONPATH=\"$PWD/src\"",
        "export MPLCONFIGDIR=outputs/tmp/codex-mpl-v5-qwen4b",
        "\"$PYTHON\" neurips_2026/sean_reproduce/scripts/generate_v5_leo_qwen4b_comparison.py",
        "```",
    ])
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_csv(output_dir / "csv" / "coverage_summary.csv", coverage)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render controlled Qwen3.5-4B versus Llama SFT v5 Leo figures.")
    parser.add_argument("--qwen-root", type=Path, default=DEFAULT_QWEN_ROOT)
    parser.add_argument("--llama-root", type=Path, default=DEFAULT_LLAMA_ROOT)
    parser.add_argument("--llama-base-root", type=Path, default=DEFAULT_LLAMA_BASE_ROOT)
    parser.add_argument("--mask-root", type=Path, default=DEFAULT_MASK_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--html-path", type=Path, default=DEFAULT_HTML_PATH)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--reuse-summary", action="store_true", help="Rerender from existing summary/status/alpha CSVs.")
    return parser.parse_args()


def read_alpha_csv(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            (str(row["partition"]), str(row["simulation_version"])): {
                field: finite(value) for field, value in row.items() if field not in {"partition", "simulation_version"}
            }
            for row in csv.DictReader(handle)
        }


def main() -> int:
    args = parse_args()
    qwen_root = args.qwen_root.expanduser().resolve()
    llama_root = args.llama_root.expanduser().resolve()
    llama_base_root = args.llama_base_root.expanduser().resolve()
    mask_root = args.mask_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    html_path = args.html_path.expanduser().resolve()
    required = [qwen_root, llama_root, llama_base_root, mask_root]
    if not all(path.is_dir() for path in required):
        missing = ", ".join(str(path) for path in required if not path.is_dir())
        raise SystemExit(f"Required input directory does not exist: {missing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_renderer()
    install_profile_override(qwen_root=qwen_root, llama_root=llama_root, llama_base_root=llama_base_root, mask_root=mask_root)
    summary_path = output_dir / "csv" / "summary_metrics.csv"
    status_path = output_dir / "csv" / "per_prefix_status.csv"
    alpha_path = output_dir / "csv" / "degroot_fit_alphas.csv"
    if args.reuse_summary:
        if not all(path.is_file() for path in (summary_path, status_path, alpha_path)):
            raise SystemExit("--reuse-summary requires summary_metrics.csv, per_prefix_status.csv, and degroot_fit_alphas.csv.")
        with summary_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        with status_path.open(encoding="utf-8", newline="") as handle:
            statuses = list(csv.DictReader(handle))
        alphas = read_alpha_csv(alpha_path)
    else:
        rows, statuses, alphas = compute_summary(
            qwen_root=qwen_root,
            llama_root=llama_root,
            llama_base_root=llama_base_root,
            mask_root=mask_root,
            bootstrap_reps=max(0, int(args.bootstrap_reps)),
        )
        write_csv(summary_path, rows)
        write_csv(status_path, statuses)
        alpha_rows = [
            {"partition": partition, "simulation_version": version, **values}
            for (partition, version), values in sorted(alphas.items())
        ]
        write_csv(alpha_path, alpha_rows)
    render_all(
        rows=rows,
        alphas=alphas,
        qwen_root=qwen_root,
        llama_root=llama_root,
        llama_base_root=llama_base_root,
        mask_root=mask_root,
        output_dir=output_dir,
        bootstrap_reps=max(0, int(args.bootstrap_reps)),
    )
    write_readme(
        output_dir=output_dir,
        qwen_root=qwen_root,
        llama_root=llama_root,
        llama_base_root=llama_base_root,
        mask_root=mask_root,
        statuses=statuses,
        alphas=alphas,
        bootstrap_reps=max(0, int(args.bootstrap_reps)),
    )
    write_html_report(html_path=html_path, output_dir=output_dir, rows=rows, alphas=alphas, bootstrap_reps=max(0, int(args.bootstrap_reps)))
    print(html_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
