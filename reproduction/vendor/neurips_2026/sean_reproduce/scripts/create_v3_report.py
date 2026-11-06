#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pypdf import PdfReader, PdfWriter, Transformation


SEAN_REPRODUCE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SEAN_REPRODUCE_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CHENGTAO_RADAR_DIR = REPO_ROOT / "neurips_2026" / "chengtao" / "src" / "radar"
if str(CHENGTAO_RADAR_DIR) not in sys.path:
    sys.path.insert(0, str(CHENGTAO_RADAR_DIR))

from neurips_2026.sean_reproduce.v3.masks import (
    configured_private_mask_root,
    configured_public_mask_root,
    repo_relative_display_path,
)
from neurips_2026.sean_reproduce.base_model_v3_style import ordered_base_model_llms_for_report
from neurips_2026.sean_reproduce.base_model_v3_style import BASE_MODEL_V3_COLOR_MAP
from neurips_2026.sean_reproduce.base_model_v3_style import BASE_MODEL_V3_DISPLAY_LABELS
from neurips_2026.sean_reproduce.result_version_config import v4_family_sft_model
from neurips_2026.chengtao.src.radar.plot_opinion_histograms import StageSpec, load_stage_values_from_csv


DEFAULT_V3_DIR = SEAN_REPRODUCE_ROOT / "v3" / "histograms"

FILTERED_SPLIT_NAMES_BY_COLLECTION = {
    "breadth": ("breadth_topic_train", "breadth_topic_test", "breadth_group_train", "breadth_group_test"),
    "depth": ("depth_topic_train", "depth_topic_test", "depth_group_train", "depth_group_test"),
}
UNFILTERED_SPLIT_NAMES_BY_COLLECTION = {
    "breadth": ("breadth_topic_overall", "breadth_group_overall"),
    "depth": ("depth_topic_overall", "depth_group_overall"),
}
PRIVATE_STAGES = ("initial", "final")
PUBLIC_STAGES = ("tweet_1", "tweet_3")
ENTITIES = ("human", "llm", "sft", "dpo")
ENTITY_LABELS = {"human": "human", "llm": "base model", "sft": "sft", "dpo": "dpo"}
BASE_MODELS = (
    "Llama-3.1-70B-Instruct",
    "Llama-3.1-8B-Instruct",
    "Llama-3.1-Tulu-3-8B-SFT",
    "Mistral-7B-Instruct-v0.3",
    "Qwen2.5-32B-Instruct",
    "gemini-2.5-flash-lite",
    "gpt-4o-mini-2024-07-18",
    "gpt-5-mini",
)
LANDSCAPE_WIDTH = 14 * 72
LANDSCAPE_HEIGHT = 8.5 * 72
LIKERT_BUCKETS = [1, 2, 3, 4, 5, 6]
SFT_OVERVIEW_COLOR = "#2F5D8A"
DEGROOT_OVERVIEW_COLOR = "#7A7A7A"
PUBLIC_TO_PRIVATE_STAGE = {"tweet_1": "initial", "tweet_3": "final"}


def report_version_family(report_version: str) -> str:
    return "v4" if str(report_version).strip().startswith("v4") else "v3"


def report_version_namespace(report_version: str) -> str:
    version = str(report_version).strip()
    if version.startswith("v4_b"):
        return "v4_b"
    if version.startswith("v4"):
        return "v4"
    if version.startswith("v3_b"):
        return "v3_b"
    return "v3"


def default_output_path(collection: str, *, report_version: str, unfiltered: bool) -> Path:
    suffix = "_unfiltered" if unfiltered else ""
    report_dir = SEAN_REPRODUCE_ROOT / report_version_namespace(report_version) / "histograms"
    return report_dir / f"report_public_private_opinions_{collection}_{report_version}{suffix}.pdf"


def default_input_path(*, report_version: str) -> Path:
    return SEAN_REPRODUCE_ROOT / report_version_namespace(report_version) / "histograms"


def v3_mask_setup_lines(*, use_split_masks: bool) -> list[str]:
    lines = [
        "Mask setup:",
        (
            "• Base page masks: golden_group_v0 aggregate / collection / split txt masks, depending on the page."
            if use_split_masks
            else "• Base page masks: golden_group_v0 aggregate / collection txt masks only; no additional split masks."
        ),
        f"• Public/tweet additional mask root: {repo_relative_display_path(configured_public_mask_root())}",
        f"• Private/opinion additional mask root: {repo_relative_display_path(configured_private_mask_root())}",
    ]
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a standalone histogram PDF report for public and private opinions.")
    parser.add_argument("--v3-dir", type=Path, default=None)
    parser.add_argument("--collection", choices=tuple(FILTERED_SPLIT_NAMES_BY_COLLECTION), default="breadth")
    parser.add_argument("--unfiltered", action="store_true", help="Build the collection-mask-only variant without train/test split masks.")
    parser.add_argument("--report-version", default="v3")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def active_entities(report_version: str) -> tuple[str, ...]:
    if report_version_family(report_version) == "v4":
        return ("human", "llm", "sft")
    return ENTITIES


def active_entity_labels(report_version: str) -> dict[str, str]:
    entities = active_entities(report_version)
    return {entity: ENTITY_LABELS[entity] for entity in entities}


def active_base_models(report_version: str) -> tuple[str, ...]:
    return tuple(ordered_base_model_llms_for_report(report_version, list(BASE_MODELS)))


def split_display_label(split_name: str) -> str:
    return str(split_name).replace("_", " ").title()


def format_share(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.1%}"


def format_mean(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.3f}"


def format_rate_cell(rate_tuple: tuple[int, int, float]) -> str:
    valid, total, share = rate_tuple
    if total == 0:
        return f"{valid}/{total} (n/a)"
    return f"{valid}/{total} ({format_share(share)})"


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_summary_map(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    rows = read_summary(path)
    return {(row["analysis_group"], row["entity"], row["stage"]): row for row in rows}


def read_status_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_summary_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_stage_counts(message: str, stages: tuple[str, ...]) -> dict[str, int]:
    counts = {stage: 0 for stage in stages}
    if not message:
        return counts
    for chunk in (part.strip() for part in message.split(",")):
        if "=" not in chunk:
            continue
        stage, value = chunk.split("=", 1)
        stage = stage.strip()
        if stage in counts:
            counts[stage] = int(value.strip())
    return counts


def normalize_phase(phase: str) -> str:
    for prefix in ("private_valid_only__", "public_valid_only__"):
        if phase.startswith(prefix):
            return phase[len(prefix):]
    return phase


def group_entity_label(entity: str) -> str:
    if entity == "human":
        return "human"
    entity_lower = entity.lower()
    if "dpo" in entity_lower or "orpo" in entity_lower or "ppo" in entity_lower:
        return "dpo"
    if "sft" in entity_lower or entity.startswith("ft:"):
        return "sft"
    return "llm"


def resolve_source_csv(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    candidates = [
        (CHENGTAO_RADAR_DIR / path).resolve(),
        (REPO_ROOT / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def mean_from_summary_row(row: dict[str, str]) -> float:
    value = str(row.get("mean_likert") or "").strip()
    return float(value) if value else math.nan


def std_from_summary_row(row: dict[str, str]) -> float:
    total = int(row.get("n") or 0)
    if total <= 0:
        return math.nan
    mean = mean_from_summary_row(row)
    variance = 0.0
    for bucket in LIKERT_BUCKETS:
        count = int(row.get(f"count_{bucket}") or 0)
        variance += count * ((bucket - mean) ** 2)
    return math.sqrt(variance / total)


def row_from_values(values: list[float]) -> dict[str, str]:
    counts = {bucket: 0 for bucket in LIKERT_BUCKETS}
    for value in values:
        rounded = int(round(value))
        if rounded in counts:
            counts[rounded] += 1
    total = len(values)
    mean = (sum(values) / total) if total else math.nan
    row: dict[str, str] = {
        "n": str(total),
        "mean_likert": str(mean),
    }
    for bucket in LIKERT_BUCKETS:
        row[f"count_{bucket}"] = str(counts[bucket])
    return row


def stage_specs_for_prefix(prefix: str) -> tuple[StageSpec, StageSpec]:
    if prefix == "public_valid_only":
        return (
            StageSpec(key="tweet_1", title="Tweet 1", event_type="tweet", chat_round_order=1.0),
            StageSpec(key="tweet_3", title="Tweet 3", event_type="tweet", chat_round_order=3.0),
        )
    return (
        StageSpec(key="initial", title="Initial Opinion", event_type="Initial Opinion"),
        StageSpec(key="final", title="Post Opinion", event_type="Post Opinion"),
    )


def split_phase_member_prefixes(status_rows: list[dict[str, str]], *, phase: str) -> set[str]:
    return {
        str(row.get("data_prefix") or "").strip()
        for row in status_rows
        if str(row.get("phase") or "").strip() == phase and str(row.get("status") or "").strip() == "ok"
    }


def aggregate_split_entity_row(
    status_rows: list[dict[str, str]],
    *,
    phase: str,
    entity: str,
    stage: str,
    prefix: str,
) -> dict[str, str]:
    stage_specs = stage_specs_for_prefix(prefix)
    target_stage = next(spec for spec in stage_specs if spec.key == stage)
    split_members = split_phase_member_prefixes(status_rows, phase=phase)
    candidate_rows = [
        row
        for row in status_rows
        if str(row.get("phase") or "").strip() == phase
        and str(row.get("entity") or "").strip() == entity
        and str(row.get("status") or "").strip() == "ok"
    ]
    if not candidate_rows and entity != "human" and not entity.startswith("ft:"):
        base_phase = f"base_model_{entity}"
        candidate_rows = [
            row
            for row in status_rows
            if str(row.get("phase") or "").strip() == base_phase
            and str(row.get("entity") or "").strip() == entity
            and str(row.get("status") or "").strip() == "ok"
            and str(row.get("data_prefix") or "").strip() in split_members
        ]
    values: list[float] = []
    for row in candidate_rows:
        csv_path = resolve_source_csv(str(row.get("csv_path") or ""))
        stage_values = load_stage_values_from_csv(
            csv_path,
            likert_col="likert_pred",
            stage_specs=(target_stage, target_stage),
            filter_malformed_text=True,
            validity_field_candidates=("llm_validity", "validity") if entity != "human" else ("human_validity", "validity"),
            roots=[],
            sidecar_roots=[],
        )
        values.extend(stage_values.get(stage, []))
    return row_from_values(values)


def preferred_v4_family_sft_for_phase(phase: str) -> str | None:
    phase_suffix = str(phase).split("__", 1)[-1]
    for collection_type in ("breadth", "depth"):
        for split_type in ("topic", "group"):
            if phase_suffix.startswith(f"{collection_type}_{split_type}_"):
                return v4_family_sft_model(collection_type, split_type)
    return None


def find_sft_model_for_split(status_rows: list[dict[str, str]], *, phase: str) -> str:
    candidate_entities: list[str] = []
    for row in status_rows:
        if str(row.get("phase") or "").strip() != phase:
            continue
        if str(row.get("status") or "").strip() != "ok":
            continue
        entity = str(row.get("entity") or "").strip()
        if entity.startswith("ft:") and "SFT" in entity:
            candidate_entities.append(entity)
    preferred_entity = preferred_v4_family_sft_for_phase(phase)
    if preferred_entity and preferred_entity in candidate_entities:
        return preferred_entity
    if candidate_entities:
        return candidate_entities[0]
    raise KeyError(f"No SFT entity found for {phase}")


def model_row_from_summary(
    summary_map: dict[tuple[str, str, str], dict[str, str]],
    *,
    prefix: str,
    model_name: str,
    stage: str,
) -> dict[str, str]:
    return summary_map[(f"{prefix}__base_model_{model_name}", model_name, stage)]


def degroot_row_from_abm_summary(
    summary_map: dict[tuple[str, str, str], dict[str, str]],
    *,
    prefix: str,
    stage: str,
) -> dict[str, str]:
    return summary_map[(f"{prefix}__base_model_DeGroot_fit", "DeGroot fit", stage)]


def count_private_stage_totals(csv_path: Path, cache: dict[Path, dict[str, int]]) -> dict[str, int]:
    if csv_path in cache:
        return cache[csv_path]
    counts = {stage: 0 for stage in PRIVATE_STAGES}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            event_type = str(row.get("event_type") or "").strip()
            if event_type == "Initial Opinion":
                counts["initial"] += 1
            elif event_type == "Post Opinion":
                counts["final"] += 1
    cache[csv_path] = counts
    return counts


def count_public_stage_totals(csv_path: Path, cache: dict[Path, dict[str, int]]) -> dict[str, int]:
    if csv_path in cache:
        return cache[csv_path]
    counts = {stage: 0 for stage in PUBLIC_STAGES}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("event_type") or "").strip() != "tweet":
                continue
            round_text = str(row.get("chat_round_order") or "").strip()
            if round_text in {"1", "1.0"}:
                counts["tweet_1"] += 1
            elif round_text in {"3", "3.0"}:
                counts["tweet_3"] += 1
    cache[csv_path] = counts
    return counts


def aggregate_valid_rates(
    status_rows: list[dict[str, str]],
    *,
    stages: tuple[str, ...],
    count_stage_totals,
) -> dict[tuple[str, str, str], tuple[int, int, float]]:
    totals_cache: dict[Path, dict[str, int]] = {}
    aggregate: dict[tuple[str, str, str], list[int]] = {}
    for row in status_rows:
        if row.get("status") != "ok":
            continue
        phase = normalize_phase(str(row.get("phase") or ""))
        raw_entity = str(row.get("entity") or "")
        if not phase or not raw_entity:
            continue
        entity = raw_entity if phase.startswith("base_model_") else group_entity_label(raw_entity)
        source_csv = resolve_source_csv(str(row.get("csv_path") or ""))
        stage_totals = count_stage_totals(source_csv, totals_cache)
        valid_counts = parse_stage_counts(str(row.get("message") or ""), stages)
        for stage in stages:
            key = (phase, entity, stage)
            current = aggregate.setdefault(key, [0, 0])
            current[0] += valid_counts[stage]
            current[1] += stage_totals[stage]
    return {
        key: (valid, total, (valid / total) if total else math.nan)
        for key, (valid, total) in aggregate.items()
    }


def collect_split_entity_members(status_rows: list[dict[str, str]], *, entities: tuple[str, ...]) -> dict[str, list[str]]:
    members = {entity: set() for entity in entities if entity != "human"}
    for row in status_rows:
        if row.get("status") != "ok":
            continue
        phase = normalize_phase(str(row.get("phase") or ""))
        raw_entity = str(row.get("entity") or "")
        if not phase or not raw_entity or phase.startswith("base_model_") or raw_entity == "human":
            continue
        grouped_entity = group_entity_label(raw_entity)
        if grouped_entity not in members:
            continue
        members[grouped_entity].add(raw_entity)
    return {key: sorted(value) for key, value in members.items()}


def neutral_share(row: dict[str, str]) -> tuple[int, int, float]:
    total = int(row["n"])
    neutral = int(row["count_3"]) + int(row["count_4"])
    share = (neutral / total) if total else math.nan
    return total, neutral, share


def sft_entity_display_label(entity: str) -> str:
    return {
        "sft": "SFT",
        "llama_sft": "Llama SFT",
        "gpt_sft": "GPT SFT",
    }.get(entity, entity.replace("_", " ").title())


def sft_entities_for_group(
    summary_map: dict[tuple[str, str, str], dict[str, str]],
    *,
    analysis_group: str,
    stage: str,
) -> list[str]:
    preferred = [entity for entity in ("sft", "llama_sft", "gpt_sft") if (analysis_group, entity, stage) in summary_map]
    if preferred:
        return preferred
    return sorted(
        {
            entity
            for group_key, entity, row_stage in summary_map.keys()
            if group_key == analysis_group and row_stage == stage and "sft" in entity.lower()
        }
    )


def add_text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.03, 0.95, title, fontsize=20, fontweight="bold", va="top")
    y = 0.87
    for line in lines:
        ax.text(0.04, y, line, fontsize=12, va="top")
        y -= 0.055
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_valid_rate_table_page(
    pdf: PdfPages,
    title: str,
    rows: list[list[str]],
    column_labels: list[str],
    footnote: str,
) -> None:
    fig = plt.figure(figsize=(14, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_title(title, fontsize=18, fontweight="bold", pad=18)
    table = ax.table(cellText=rows, colLabels=column_labels, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    fig.text(0.5, 0.04, footnote, ha="center", fontsize=10)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_neutral_share_table_page(
    pdf: PdfPages,
    *,
    title: str,
    summary_map: dict[tuple[str, str, str], dict[str, str]],
    summary_prefix: str,
    split_names: tuple[str, ...],
    stages: tuple[str, ...],
) -> None:
    rows: list[list[str]] = []
    for split_name in split_names:
        for stage in stages:
            analysis_group = f"{summary_prefix}__{split_name}"
            for entity in sft_entities_for_group(summary_map, analysis_group=analysis_group, stage=stage):
                row = summary_map[(analysis_group, entity, stage)]
                total, neutral, share = neutral_share(row)
                rows.append(
                    [
                        split_display_label(split_name),
                        sft_entity_display_label(entity),
                        stage,
                        str(total),
                        str(neutral),
                        format_share(share),
                        format_mean(float(row['mean_likert'])),
                    ]
                )

    fig = plt.figure(figsize=(13.5, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_title(title, fontsize=18, fontweight="bold", pad=20)
    table = ax.table(
        cellText=rows,
        colLabels=["Split", "Model", "Stage", "Responses", "Neutral responses (3-4)", "Neutral share", "Mean Likert"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.15, 1.65)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def should_add_v4_breadth_overview_page(*, report_version: str, collection: str, use_split_masks: bool) -> bool:
    return report_version_family(report_version) == "v4" and collection == "breadth"


def overview_group_name(summary_map: dict[tuple[str, str, str], dict[str, str]], *, prefix: str, collection: str) -> str:
    preferred = f"{prefix}__{collection}_group_overall"
    fallback = f"{prefix}__{collection}_topic_overall"
    if any(group_key == preferred for group_key, _entity, _stage in summary_map):
        return preferred
    if any(group_key == fallback for group_key, _entity, _stage in summary_map):
        return fallback
    raise KeyError(f"No overall summary rows found for {prefix} {collection}")


def histogram_percentages(row: dict[str, str]) -> list[float]:
    total = int(row["n"])
    if total <= 0:
        return [0.0] * len(LIKERT_BUCKETS)
    return [(int(row[f"count_{bucket}"]) / total) * 100.0 for bucket in LIKERT_BUCKETS]


def model_display_label(model_name: str) -> str:
    return BASE_MODEL_V3_DISPLAY_LABELS.get(model_name, model_name)


def add_histogram_subplot(ax: plt.Axes, *, row: dict[str, str], title: str, color: str, show_ylabel: bool) -> None:
    percentages = histogram_percentages(row)
    ax.bar(LIKERT_BUCKETS, percentages, color=color, edgecolor="white", linewidth=0.7, width=0.85)
    mean_value = mean_from_summary_row(row)
    std_value = std_from_summary_row(row)
    ax.set_title(
        f"{title}\n(n={int(row['n'])}, mean={format_mean(mean_value)}, sd={format_mean(std_value)})",
        fontsize=8.4,
        pad=5,
    )
    ax.set_xticks(LIKERT_BUCKETS)
    ax.set_xlim(0.5, 6.5)
    ax.set_ylim(0.0, max(55.0, max(percentages) * 1.15 if percentages else 55.0))
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    if show_ylabel:
        ax.set_ylabel("Share (%)", fontsize=8)
    else:
        ax.set_yticklabels([])
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.35)


def overview_entries(
    summary_map: dict[tuple[str, str, str], dict[str, str]],
    *,
    collection: str,
    prefix: str,
    human_stage_initial: str,
    human_stage_final: str,
    model_stage: str,
) -> list[tuple[dict[str, str], str, str]]:
    overall_group = overview_group_name(summary_map, prefix=prefix, collection=collection)
    sft_entities = sft_entities_for_group(summary_map, analysis_group=overall_group, stage=model_stage)
    if not sft_entities:
        raise KeyError(f"No SFT summary row found for {overall_group} {model_stage}")
    llama_sft_entity = sft_entities[0]

    entries: list[tuple[dict[str, str], str, str]] = [
        (summary_map[(overall_group, "human", human_stage_initial)], f"Human {human_stage_initial.replace('_', ' ').title()}", BASE_MODEL_V3_COLOR_MAP["human"]),
        (summary_map[(overall_group, "human", human_stage_final)], f"Human {human_stage_final.replace('_', ' ').title()}", BASE_MODEL_V3_COLOR_MAP["human"]),
        (
            summary_map[(f"{prefix}__base_model_Llama-3.1-8B-Instruct", "Llama-3.1-8B-Instruct", model_stage)],
            f"{model_display_label('Llama-3.1-8B-Instruct')} {model_stage.replace('_', ' ').title()}",
            BASE_MODEL_V3_COLOR_MAP["Llama-3.1-8B-Instruct"],
        ),
        (
            summary_map[(overall_group, llama_sft_entity, model_stage)],
            f"Llama SFT {model_stage.replace('_', ' ').title()}",
            SFT_OVERVIEW_COLOR,
        ),
    ]

    for model_name in active_base_models("v4"):
        if model_name == "Llama-3.1-8B-Instruct":
            continue
        entries.append(
            (
                summary_map[(f"{prefix}__base_model_{model_name}", model_name, model_stage)],
                f"{model_display_label(model_name)} {model_stage.replace('_', ' ').title()}",
                BASE_MODEL_V3_COLOR_MAP.get(model_name, "#999999"),
            )
        )
    return entries


def add_v4_breadth_overview_histogram_page(
    pdf: PdfPages,
    *,
    public_summary: dict[tuple[str, str, str], dict[str, str]],
    private_summary: dict[tuple[str, str, str], dict[str, str]],
    collection: str,
) -> None:
    public_entries = overview_entries(
        public_summary,
        collection=collection,
        prefix="public_valid_only",
        human_stage_initial="tweet_1",
        human_stage_final="tweet_3",
        model_stage="tweet_3",
    )
    private_entries = overview_entries(
        private_summary,
        collection=collection,
        prefix="private_valid_only",
        human_stage_initial="initial",
        human_stage_final="final",
        model_stage="final",
    )

    fig, axes = plt.subplots(5, 4, figsize=(16, 10))
    fig.suptitle("Histogram Overview: Human Anchors and Model Endpoints", fontsize=18, y=0.985, fontweight="bold")
    fig.text(0.25, 0.952, "Public Opinion", ha="center", va="top", fontsize=12.5)
    fig.text(0.75, 0.952, "Private Opinion", ha="center", va="top", fontsize=12.5)

    for index, (row, title, color) in enumerate(public_entries):
        plot_row = index // 2
        plot_col = index % 2
        add_histogram_subplot(axes[plot_row][plot_col], row=row, title=title, color=color, show_ylabel=plot_col == 0)

    for index, (row, title, color) in enumerate(private_entries):
        plot_row = index // 2
        plot_col = 2 + (index % 2)
        add_histogram_subplot(axes[plot_row][plot_col], row=row, title=title, color=color, show_ylabel=plot_col == 2)

    for row_index in range(5):
        for col_index in range(4):
            axes[row_index][col_index].set_xlabel("Likert", fontsize=8)

    fig.text(
        0.5,
        0.012,
        "Human tweet 1 / initial distributions are shown once because the model-side first-step opinions are human-conditioned; model panels show the endpoint distributions.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.94])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def should_add_v4_breadth_topic_split_pages(*, report_version: str, collection: str, use_split_masks: bool) -> bool:
    return report_version_family(report_version) == "v4" and collection == "breadth" and not use_split_masks


def split_topic_entries(
    *,
    summary_map: dict[tuple[str, str, str], dict[str, str]],
    abm_summary_map: dict[tuple[str, str, str], dict[str, str]],
    status_rows: list[dict[str, str]],
    prefix: str,
    split_name: str,
) -> list[tuple[dict[str, str], str, str]]:
    endpoint_stage = "tweet_3" if prefix == "public_valid_only" else "final"
    anchor_stage = "tweet_1" if prefix == "public_valid_only" else "initial"
    sft_phase = f"{prefix}__{split_name}"
    sft_model = find_sft_model_for_split(status_rows, phase=sft_phase)
    model_rows: list[tuple[dict[str, str], str, str]] = [
        (
            summary_map[(f"{prefix}__{split_name}", "human", anchor_stage)],
            f"Human {anchor_stage.replace('_', ' ').title()}",
            BASE_MODEL_V3_COLOR_MAP["human"],
        ),
        (
            summary_map[(f"{prefix}__{split_name}", "human", endpoint_stage)],
            f"Human {endpoint_stage.replace('_', ' ').title()}",
            BASE_MODEL_V3_COLOR_MAP["human"],
        ),
        (
            aggregate_split_entity_row(
                status_rows,
                phase=sft_phase,
                entity="Llama-3.1-8B-Instruct",
                stage=endpoint_stage,
                prefix=prefix,
            ),
            f"{model_display_label('Llama-3.1-8B-Instruct')} {endpoint_stage.replace('_', ' ').title()}",
            BASE_MODEL_V3_COLOR_MAP["Llama-3.1-8B-Instruct"],
        ),
        (
            aggregate_split_entity_row(
                status_rows,
                phase=sft_phase,
                entity=sft_model,
                stage=endpoint_stage,
                prefix=prefix,
            ),
            f"Llama SFT {endpoint_stage.replace('_', ' ').title()}",
            SFT_OVERVIEW_COLOR,
        ),
        (
            degroot_row_from_abm_summary(abm_summary_map, prefix=prefix, stage=endpoint_stage),
            f"DeGroot fit {endpoint_stage.replace('_', ' ').title()}",
            DEGROOT_OVERVIEW_COLOR,
        ),
    ]
    for model_name in active_base_models("v4"):
        if model_name == "Llama-3.1-8B-Instruct":
            continue
        model_rows.append(
            (
                aggregate_split_entity_row(
                    status_rows,
                    phase=sft_phase,
                    entity=model_name,
                    stage=endpoint_stage,
                    prefix=prefix,
                ),
                f"{model_display_label(model_name)} {endpoint_stage.replace('_', ' ').title()}",
                BASE_MODEL_V3_COLOR_MAP.get(model_name, "#999999"),
            )
        )
    return model_rows


def focused_split_topic_entries(
    *,
    summary_map: dict[tuple[str, str, str], dict[str, str]],
    status_rows: list[dict[str, str]],
    prefix: str,
    split_name: str,
) -> list[tuple[dict[str, str], str, str]]:
    endpoint_stage = "tweet_3" if prefix == "public_valid_only" else "final"
    anchor_stage = "tweet_1" if prefix == "public_valid_only" else "initial"
    sft_phase = f"{prefix}__{split_name}"
    sft_model = find_sft_model_for_split(status_rows, phase=sft_phase)
    return [
        (
            summary_map[(f"{prefix}__{split_name}", "human", anchor_stage)],
            f"Human {anchor_stage.replace('_', ' ').title()}",
            BASE_MODEL_V3_COLOR_MAP["human"],
        ),
        (
            summary_map[(f"{prefix}__{split_name}", "human", endpoint_stage)],
            f"Human {endpoint_stage.replace('_', ' ').title()}",
            BASE_MODEL_V3_COLOR_MAP["human"],
        ),
        (
            aggregate_split_entity_row(
                status_rows,
                phase=sft_phase,
                entity="Llama-3.1-8B-Instruct",
                stage=endpoint_stage,
                prefix=prefix,
            ),
            f"{model_display_label('Llama-3.1-8B-Instruct')} {endpoint_stage.replace('_', ' ').title()}",
            BASE_MODEL_V3_COLOR_MAP["Llama-3.1-8B-Instruct"],
        ),
        (
            aggregate_split_entity_row(
                status_rows,
                phase=sft_phase,
                entity=sft_model,
                stage=endpoint_stage,
                prefix=prefix,
            ),
            f"Llama SFT {endpoint_stage.replace('_', ' ').title()}",
            SFT_OVERVIEW_COLOR,
        ),
    ]


def degroot_fit_full_label(*, report_version: str, split_name: str) -> str | None:
    csv_path = (
        SEAN_REPRODUCE_ROOT
        / report_version_namespace(report_version)
        / "abm"
        / "group_opinion_convergence"
        / "post_train"
        / "csv"
        / f"group_opinion_convergence_models_vs_all_{split_name}_degroot_fit_{report_version}.csv"
    )
    if not csv_path.exists():
        return None
    for row in read_summary_rows(csv_path):
        model_name = str(row.get("model") or "").strip()
        if model_name.startswith("DeGroot fit on "):
            split_phrase = split_name.replace("_", " ")
            if split_phrase in model_name:
                return model_name

            alpha_public = row.get("fit_alpha_public")
            alpha_private = row.get("fit_alpha_private")
            if alpha_public and alpha_private:
                return (
                    f"DeGroot fit on {split_phrase} "
                    f"(alpha_public={float(alpha_public):.3f}, alpha_private={float(alpha_private):.3f})"
                )
            return f"DeGroot fit on {split_phrase}"
    return None


def split_page_footnote(
    *,
    report_version: str,
    split_name: str,
    public_status_rows: list[dict[str, str]],
    private_status_rows: list[dict[str, str]],
    include_base_models: bool,
    include_degroot: bool,
) -> str:
    footnote_pairs: list[tuple[str, str]] = [(model_display_label("Llama-3.1-8B-Instruct"), "Llama-3.1-8B-Instruct")]

    public_phase = f"public_valid_only__{split_name}"
    private_phase = f"private_valid_only__{split_name}"
    public_sft_model = find_sft_model_for_split(public_status_rows, phase=public_phase)
    private_sft_model = find_sft_model_for_split(private_status_rows, phase=private_phase)
    if public_sft_model == private_sft_model:
        footnote_pairs.append(("Llama SFT", public_sft_model))
    else:
        footnote_pairs.extend(
            [
                ("Llama SFT (public)", public_sft_model),
                ("Llama SFT (private)", private_sft_model),
            ]
        )

    if include_base_models:
        for model_name in active_base_models(report_version):
            if model_name == "Llama-3.1-8B-Instruct":
                continue
            footnote_pairs.append((model_display_label(model_name), model_name))

    if include_degroot:
        degroot_label = degroot_fit_full_label(report_version=report_version, split_name=split_name)
        if degroot_label:
            footnote_pairs.append(("DeGroot fit", degroot_label))

    seen_full_names: set[str] = set()
    formatted_pairs: list[str] = []
    for short_label, full_name in footnote_pairs:
        if full_name in seen_full_names:
            continue
        seen_full_names.add(full_name)
        formatted_pairs.append(f"{short_label} = {full_name}")

    return textwrap.fill("Full model names: " + " | ".join(formatted_pairs), width=185)


def add_side_by_side_histogram_page(
    pdf: PdfPages,
    *,
    left_entries: list[tuple[dict[str, str], str, str]],
    right_entries: list[tuple[dict[str, str], str, str]],
    title: str,
    left_title: str,
    right_title: str,
    footer: str | None = None,
) -> None:
    max_entries = max(len(left_entries), len(right_entries))
    cols_per_panel = 4 if max_entries > 4 else 2
    rows_per_panel = max(1, math.ceil(max_entries / cols_per_panel))
    total_rows = rows_per_panel * 2
    figure_width = 16 if cols_per_panel == 4 else 12
    figure_height = 6 * rows_per_panel

    fig, axes = plt.subplots(total_rows, cols_per_panel, figsize=(figure_width, figure_height))
    if total_rows == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=18, y=0.985, fontweight="bold")
    fig.text(0.5, 0.955, left_title, ha="center", va="top", fontsize=12.5)
    fig.text(0.5, 0.495, right_title, ha="center", va="top", fontsize=12.5)

    for index, (row, panel_title, color) in enumerate(left_entries):
        plot_row = index // cols_per_panel
        plot_col = index % cols_per_panel
        add_histogram_subplot(axes[plot_row][plot_col], row=row, title=panel_title, color=color, show_ylabel=plot_col == 0)

    for index, (row, panel_title, color) in enumerate(right_entries):
        plot_row = rows_per_panel + (index // cols_per_panel)
        plot_col = index % cols_per_panel
        add_histogram_subplot(axes[plot_row][plot_col], row=row, title=panel_title, color=color, show_ylabel=plot_col == 0)

    for empty_index in range(len(left_entries), rows_per_panel * cols_per_panel):
        axes[empty_index // cols_per_panel][empty_index % cols_per_panel].axis("off")
    for empty_index in range(len(right_entries), rows_per_panel * cols_per_panel):
        axes[rows_per_panel + (empty_index // cols_per_panel)][empty_index % cols_per_panel].axis("off")

    for row_index in range(total_rows):
        for col_index in range(cols_per_panel):
            if axes[row_index][col_index].has_data():
                axes[row_index][col_index].set_xlabel("Likert", fontsize=8)

    bottom_rect = 0.08 if footer else 0.03
    fig.tight_layout(rect=[0.02, bottom_rect, 0.98, 0.95])
    if footer:
        fig.text(0.5, 0.012, footer, ha="center", va="bottom", fontsize=8.4)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_v4_breadth_split_histogram_pages(
    pdf: PdfPages,
    *,
    report_version: str,
    public_summary: dict[tuple[str, str, str], dict[str, str]],
    private_summary: dict[tuple[str, str, str], dict[str, str]],
    public_abm_summary: dict[tuple[str, str, str], dict[str, str]],
    private_abm_summary: dict[tuple[str, str, str], dict[str, str]],
    public_status_rows: list[dict[str, str]],
    private_status_rows: list[dict[str, str]],
) -> None:
    split_pages = (
        ("breadth_group_train", "Breadth Group Train"),
        ("breadth_group_test", "Breadth Group Test"),
        ("breadth_topic_train", "Breadth Topic Train"),
        ("breadth_topic_test", "Breadth Topic Test"),
    )
    for split_name, split_label in split_pages:
        add_side_by_side_histogram_page(
            pdf,
            left_entries=split_topic_entries(
                summary_map=public_summary,
                abm_summary_map=public_abm_summary,
                status_rows=public_status_rows,
                prefix="public_valid_only",
                split_name=split_name,
            ),
            right_entries=split_topic_entries(
                summary_map=private_summary,
                abm_summary_map=private_abm_summary,
                status_rows=private_status_rows,
                prefix="private_valid_only",
                split_name=split_name,
            ),
            title=f"Histogram Overview: {split_label}",
            left_title="Public Opinion",
            right_title="Private Opinion",
            footer=split_page_footnote(
                report_version=report_version,
                split_name=split_name,
                public_status_rows=public_status_rows,
                private_status_rows=private_status_rows,
                include_base_models=True,
                include_degroot=True,
            ),
        )
        add_side_by_side_histogram_page(
            pdf,
            left_entries=focused_split_topic_entries(
                summary_map=public_summary,
                status_rows=public_status_rows,
                prefix="public_valid_only",
                split_name=split_name,
            ),
            right_entries=focused_split_topic_entries(
                summary_map=private_summary,
                status_rows=private_status_rows,
                prefix="private_valid_only",
                split_name=split_name,
            ),
            title=f"Focused Histogram Overview: {split_label}",
            left_title="Public Opinion",
            right_title="Private Opinion",
            footer=split_page_footnote(
                report_version=report_version,
                split_name=split_name,
                public_status_rows=public_status_rows,
                private_status_rows=private_status_rows,
                include_base_models=False,
                include_degroot=True,
            ),
        )


def add_pair_overlay_page(path: Path, *, title: str, left_title: str, right_title: str, footer: str) -> None:
    fig = plt.figure(figsize=(LANDSCAPE_WIDTH / 72, LANDSCAPE_HEIGHT / 72))
    ax = fig.add_subplot(111)
    ax.axis("off")
    fig.text(0.5, 0.97, title, ha="center", va="top", fontsize=18, fontweight="bold")
    fig.text(0.25, 0.91, left_title, ha="center", va="top", fontsize=13)
    fig.text(0.75, 0.91, right_title, ha="center", va="top", fontsize=13)
    fig.text(0.5, 0.03, footer, ha="center", va="bottom", fontsize=11)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def merge_panel_pdf_pair(
    writer: PdfWriter,
    *,
    overlay_path: Path,
    left_path: Path,
    right_path: Path,
) -> None:
    page = writer.add_blank_page(width=LANDSCAPE_WIDTH, height=LANDSCAPE_HEIGHT)
    overlay_page = PdfReader(str(overlay_path)).pages[0]
    left_page = PdfReader(str(left_path)).pages[0]
    right_page = PdfReader(str(right_path)).pages[0]
    page.merge_page(overlay_page)

    margin = 24
    gap = 18
    top_reserved = 80
    bottom_reserved = 34
    available_width = (LANDSCAPE_WIDTH - (2 * margin) - gap) / 2
    available_height = LANDSCAPE_HEIGHT - top_reserved - bottom_reserved

    for source_page, x_offset in ((left_page, margin), (right_page, margin + available_width + gap)):
        source_width = float(source_page.mediabox.width)
        source_height = float(source_page.mediabox.height)
        scale = min(available_width / source_width, available_height / source_height)
        rendered_width = source_width * scale
        rendered_height = source_height * scale
        x_pos = x_offset + ((available_width - rendered_width) / 2)
        y_pos = bottom_reserved + ((available_height - rendered_height) / 2)
        page.merge_transformed_page(
            source_page,
            Transformation().scale(scale).translate(tx=x_pos, ty=y_pos),
        )


def build_split_rate_footnote(
    split_members: dict[str, list[str]],
    *,
    opinion_label: str,
    entities: tuple[str, ...],
) -> str:
    parts: list[str] = []
    if "llm" in entities:
        parts.append(f"base model = {', '.join(split_members['llm']) or 'none'}")
    if "sft" in entities:
        parts.append(f"sft = {', '.join(split_members['sft']) or 'none'}")
    if "dpo" in entities:
        parts.append(f"dpo = {', '.join(split_members['dpo']) or 'none'}")
    return (
        f"Rates are valid {opinion_label} rows divided by total matching rows in the source opinion CSVs. "
        f"{'; '.join(parts)}."
    )


def build_rate_rows(
    *,
    valid_rates: dict[tuple[str, str, str], tuple[int, int, float]],
    split_names: tuple[str, ...],
    stages: tuple[str, ...],
    entities: tuple[str, ...],
    entity_labels: dict[str, str],
) -> list[list[str]]:
    rows: list[list[str]] = []
    for split_name in split_names:
        for entity in entities:
            stage_1 = valid_rates[(split_name, entity, stages[0])]
            stage_2 = valid_rates[(split_name, entity, stages[1])]
            rows.append(
                [
                    split_display_label(split_name),
                    entity_labels[entity],
                    format_rate_cell(stage_1),
                    format_rate_cell(stage_2),
                ]
            )
    return rows


def build_base_model_rate_rows(
    *,
    valid_rates: dict[tuple[str, str, str], tuple[int, int, float]],
    stages: tuple[str, ...],
    report_version: str,
) -> list[list[str]]:
    rows: list[list[str]] = []
    for model in active_base_models(report_version):
        phase = f"base_model_{model}"
        stage_1 = valid_rates[(phase, model, stages[0])]
        stage_2 = valid_rates[(phase, model, stages[1])]
        rows.append([model, format_rate_cell(stage_1), format_rate_cell(stage_2)])
    return rows


def main() -> int:
    args = parse_args()
    report_version = str(args.report_version)
    entities = active_entities(report_version)
    entity_labels = active_entity_labels(report_version)
    v3_dir = (args.v3_dir or default_input_path(report_version=report_version)).resolve()
    collection = args.collection
    use_split_masks = not args.unfiltered
    split_names = (FILTERED_SPLIT_NAMES_BY_COLLECTION if use_split_masks else UNFILTERED_SPLIT_NAMES_BY_COLLECTION)[collection]
    output_path = (args.output or default_output_path(collection, report_version=report_version, unfiltered=args.unfiltered)).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    v3_private_dir = v3_dir / "private_opinion" / "valid_only"
    v3_public_dir = v3_dir / "public_opinion" / "valid_only"

    v3_private_summary = read_summary_map(v3_private_dir / "histogram_summary_counts.csv")
    v3_public_summary = read_summary_map(v3_public_dir / "histogram_summary_counts.csv")
    v3_private_status_rows = read_status_rows(v3_private_dir / "histogram_collection_status.csv")
    v3_public_status_rows = read_status_rows(v3_public_dir / "histogram_collection_status.csv")
    abm_dir = SEAN_REPRODUCE_ROOT / report_version_namespace(report_version) / "abm" / "histograms"
    abm_private_summary = read_summary_map(abm_dir / "private_opinion" / "valid_only" / "histogram_summary_counts.csv")
    abm_public_summary = read_summary_map(abm_dir / "public_opinion" / "valid_only" / "histogram_summary_counts.csv")

    v3_private_valid_rates = aggregate_valid_rates(
        v3_private_status_rows,
        stages=PRIVATE_STAGES,
        count_stage_totals=count_private_stage_totals,
    )
    v3_public_valid_rates = aggregate_valid_rates(
        v3_public_status_rows,
        stages=PUBLIC_STAGES,
        count_stage_totals=count_public_stage_totals,
    )

    private_split_members = collect_split_entity_members(v3_private_status_rows, entities=entities)
    public_split_members = collect_split_entity_members(v3_public_status_rows, entities=entities)

    split_pairs = [
        (
            split_display_label(split_name),
            v3_public_dir / "splits" / split_name / f"hist_{split_name}_public_valid_only_panels.pdf",
            v3_private_dir / "splits" / split_name / f"hist_{split_name}_private_valid_only_panels.pdf",
        )
        for split_name in split_names
    ]
    extra_pairs = [
        (
            "All Base Models Panels",
            v3_public_dir / "base_aggregate" / "hist_base_all_models_public_valid_only_panels.pdf",
            v3_private_dir / "base_aggregate" / "hist_base_all_models_private_valid_only_panels.pdf",
        ),
    ]
    if report_version_family(report_version) != "v4":
        extra_pairs.append(
            (
                "Base SFT Model Panels",
                v3_public_dir / "base_models" / "Llama-3.1-Tulu-3-8B-SFT" / "hist_base_public_valid_only_panels.pdf",
                v3_private_dir / "base_models" / "Llama-3.1-Tulu-3-8B-SFT" / "hist_base_private_valid_only_panels.pdf",
            )
        )

    required_paths = [
        v3_private_dir / "histogram_summary_counts.csv",
        v3_public_dir / "histogram_summary_counts.csv",
        v3_private_dir / "histogram_collection_status.csv",
        v3_public_dir / "histogram_collection_status.csv",
    ]
    required_paths.extend(path for _, left_path, right_path in split_pairs + extra_pairs for path in (left_path, right_path))
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing histogram inputs:\n{missing_text}")

    intro_lines = [
        f"This report summarizes the standalone {report_version} {collection} histogram outputs after validity rejudging.",
        "Public-opinion panels are shown on the left side of each comparison page; private-opinion panels are shown on the right.",
        (
            f"The split pages in this report cover the {collection} topic/group train/test partitions."
            if use_split_masks
            else f"The pages in this report use the {collection} collection masks only and do not apply any group/topic train/test split masks."
        ),
        "Histogram bars are plotted as response share percentages, and bar labels show both raw counts and percentages.",
        "The pages below use the *_panels histogram artifacts, not the *_overlay artifacts.",
        *v3_mask_setup_lines(use_split_masks=use_split_masks),
    ]

    private_split_rate_rows = build_rate_rows(
        valid_rates=v3_private_valid_rates,
        split_names=split_names,
        stages=PRIVATE_STAGES,
        entities=entities,
        entity_labels=entity_labels,
    )
    public_split_rate_rows = build_rate_rows(
        valid_rates=v3_public_valid_rates,
        split_names=split_names,
        stages=PUBLIC_STAGES,
        entities=entities,
        entity_labels=entity_labels,
    )
    private_base_model_rows = build_base_model_rate_rows(
        valid_rates=v3_private_valid_rates,
        stages=PRIVATE_STAGES,
        report_version=report_version,
    )
    public_base_model_rows = build_base_model_rate_rows(
        valid_rates=v3_public_valid_rates,
        stages=PUBLIC_STAGES,
        report_version=report_version,
    )

    with tempfile.TemporaryDirectory() as temp_dir_text:
        temp_dir = Path(temp_dir_text)
        text_pages_path = temp_dir / "text_pages.pdf"
        writer = PdfWriter()

        with PdfPages(text_pages_path) as pdf:
            add_text_page(pdf, f"Histogram Report: {report_version} {collection.title()} Splits", intro_lines)
            add_valid_rate_table_page(
                pdf,
                title=f"Private Valid Message Rate by {collection.title()} Split and Entity",
                rows=private_split_rate_rows,
                column_labels=["Split", "Entity", "Initial", "Final"],
                footnote=(
                    build_split_rate_footnote(private_split_members, opinion_label="private-opinion", entities=entities)
                    if use_split_masks
                    else "Rates are valid private-opinion rows divided by total matching rows in the source opinion CSVs. This report uses only the collection-level golden masks and no train/test split masks."
                ),
            )
            add_valid_rate_table_page(
                pdf,
                title=f"Public Valid Message Rate by {collection.title()} Split and Entity",
                rows=public_split_rate_rows,
                column_labels=["Split", "Entity", "Tweet 1", "Tweet 3"],
                footnote=(
                    build_split_rate_footnote(public_split_members, opinion_label="public-opinion", entities=entities)
                    if use_split_masks
                    else "Rates are valid public-opinion rows divided by total matching rows in the source opinion CSVs. This report uses only the collection-level golden masks and no train/test split masks."
                ),
            )
            add_valid_rate_table_page(
                pdf,
                title="Private Valid Message Rate by Base Model",
                rows=private_base_model_rows,
                column_labels=["Model", "Initial", "Final"],
                footnote="Private v3 histograms reflect the rejudged initial/post validity sidecars where available.",
            )
            add_valid_rate_table_page(
                pdf,
                title="Public Valid Message Rate by Base Model",
                rows=public_base_model_rows,
                column_labels=["Model", "Tweet 1", "Tweet 3"],
                footnote="Public v3 histograms reflect the rejudged tweet validity sidecars where available.",
            )
            add_neutral_share_table_page(
                pdf,
                title=f"SFT Private Neutral Response Share in {report_version} ({collection.title()} Splits)",
                summary_map=v3_private_summary,
                summary_prefix="private_valid_only",
                split_names=split_names,
                stages=PRIVATE_STAGES,
            )
            add_neutral_share_table_page(
                pdf,
                title=f"SFT Public Neutral Response Share in {report_version} ({collection.title()} Splits)",
                summary_map=v3_public_summary,
                summary_prefix="public_valid_only",
                split_names=split_names,
                stages=PUBLIC_STAGES,
            )
            if should_add_v4_breadth_overview_page(report_version=report_version, collection=collection, use_split_masks=use_split_masks):
                add_v4_breadth_split_histogram_pages(
                    pdf,
                    report_version=report_version,
                    public_summary=v3_public_summary,
                    private_summary=v3_private_summary,
                    public_abm_summary=abm_public_summary,
                    private_abm_summary=abm_private_summary,
                    public_status_rows=v3_public_status_rows,
                    private_status_rows=v3_private_status_rows,
                )

        for page in PdfReader(str(text_pages_path)).pages:
            writer.add_page(page)

        for label, public_path, private_path in split_pairs:
            overlay_path = temp_dir / f"split_{label.replace(' ', '_').lower()}.pdf"
            add_pair_overlay_page(
                overlay_path,
                title=f"Public vs Private Split Panels: {label}",
                left_title="Public",
                right_title="Private",
                footer=(
                    "Left = public opinion panels. Right = private opinion panels. All plots are from the v3 valid-only histogram outputs."
                    if use_split_masks
                    else "Left = public opinion panels. Right = private opinion panels. All plots are from the v3 valid-only collection-mask-only histogram outputs."
                ),
            )
            merge_panel_pdf_pair(writer, overlay_path=overlay_path, left_path=public_path, right_path=private_path)

        for label, public_path, private_path in extra_pairs:
            overlay_path = temp_dir / f"extra_{label.replace(' ', '_').lower()}.pdf"
            add_pair_overlay_page(
                overlay_path,
                title=label,
                left_title="Public",
                right_title="Private",
                footer="Left = public opinion panels. Right = private opinion panels. All plots are from the v3 valid-only histogram outputs.",
            )
            merge_panel_pdf_pair(writer, overlay_path=overlay_path, left_path=public_path, right_path=private_path)

        with output_path.open("wb") as handle:
            writer.write(handle)

    print(f"Wrote v3 histogram report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())