#!/usr/bin/env python3
"""Build opinion histograms for base models and breadth/depth post-training split comparisons."""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generate_plots import (
    HUMAN_DEFAULT_FILENAME,
    LLAMA_SFT_HIGHLIGHT_COLOR,
    LLM_DEFAULT_FILENAME,
    parse_likert,
    read_lines,
    save_csv,
    tqdm,
)
from post_training_breadth_profiles import (
    DEPTH_GROUP_MODELS_V4,
    DEPTH_TOPIC_MODELS_V4,
    GROUP_MODELS_V4,
    TOPIC_MODELS_V4,
)
from neurips_2026.sean_reproduce.v3.masks import filter_rows_for_v3_complete_group_mask
from neurips_2026.sean_reproduce.base_model_v3_style import ordered_base_model_llms_for_report

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
except Exception:  # pragma: no cover
    plt = None
    PercentFormatter = None


LIKERT_LEVELS = [1, 2, 3, 4, 5, 6]
LIKERT_BINS = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]

COLOR_HUMAN = "#222222"
COLOR_LLM = "#1f77b4"
COLOR_GPT_BASE = "#E89B93"
COLOR_SFT = LLAMA_SFT_HIGHLIGHT_COLOR
COLOR_GPT_SFT = "#E89B93"
COLOR_DPO = "#ff7f0e"

GPT_BASE_MODEL = "gpt-4o-mini-2024-07-18"
GPT_DEPTH_TOPIC_SFT_MODEL = "ft:gpt-4o-mini-2024-07-18:camer:sft-20251218-depth-topic-5epochs:CoFxFPcv"
GPT_DEPTH_GROUP_SFT_MODEL = "ft:gpt-4o-mini-2024-07-18:camer:sft-20251218-depth-group-5epochs:CoFzijg0"

TOPIC_MODELS = [
    "Llama-3.1-8B-Instruct",
    "ft:Llama-3.1-8B-SFT-20260210:breadth-topic-3epochs-unsloth",
    "ft:Llama-3.1-8B-Instruct-DPO-20260212:breadth-topic-5epochs",
]

GROUP_MODELS = [
    "Llama-3.1-8B-Instruct",
    "ft:Llama-3.1-8B-SFT-20260210:breadth-group-3epochs-unsloth",
    "ft:Llama-3.1-8B-Instruct-DPO-20260211:breadth-group-5epochs",
]

DEPTH_TOPIC_MODELS = [
    "Llama-3.1-8B-Instruct",
    "ft:Llama-3.1-8B-Instruct-SFT-20251218:depth-topic-5epochs",
    GPT_DEPTH_TOPIC_SFT_MODEL,
    "ft:Llama-3.1-8B-Instruct-DPO-20260105:depth-topic-5epochs",
]

DEPTH_GROUP_MODELS = [
    "Llama-3.1-8B-Instruct",
    "ft:Llama-3.1-8B-Instruct-SFT-20251218:depth-group-5epochs",
    GPT_DEPTH_GROUP_SFT_MODEL,
    "ft:Llama-3.1-8B-Instruct-DPO-20260105:depth-group-5epochs",
]


@dataclass(frozen=True)
class SplitModelSpec:
    key: str
    title: str
    model: str
    color: str


@dataclass(frozen=True)
class SplitPartitionSpec:
    key: str
    title: str
    prefix_file: Path


def default_split_panel_rows(series_specs: tuple[SplitModelSpec, ...]) -> tuple[tuple[str, ...], ...]:
    return (("human", *(series_spec.key for series_spec in series_specs)),)


def split_stage_section_title(stage_spec: StageSpec) -> str:
    if stage_spec.key == "final":
        return "Final Opinion"
    return stage_spec.title


def split_panel_title(*, panel_title: str, stage_spec: StageSpec, n_groups: int, grouped_layout: bool) -> str:
    title = panel_title if grouped_layout else f"{panel_title} {stage_spec.title}"
    return f"{title} (n={n_groups})"


def build_split_model_footnote(series_specs: list[SplitModelSpec]) -> str:
    model_text = " | ".join(f"{series_spec.title}={series_spec.model}" for series_spec in series_specs)
    return textwrap.fill(
        f"Model IDs: {model_text}",
        width=140,
        break_long_words=False,
        break_on_hyphens=False,
    )


@dataclass(frozen=True)
class SplitConfig:
    name: str
    title: str
    partitions: tuple[SplitPartitionSpec, ...]
    model_series: tuple[SplitModelSpec, ...]
    panel_rows: tuple[tuple[str, ...], ...] | None = None


@dataclass(frozen=True)
class StageSpec:
    key: str
    title: str
    event_type: str
    chat_round_order: float | None = None


@dataclass(frozen=True)
class HistogramRunConfig:
    key: str
    title_prefix: str
    output_subdir: str
    stage_specs: tuple[StageSpec, StageSpec]
    valid_only_for_llm: bool
    filter_malformed_text: bool


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


PRIVATE_STAGE_SPECS = (
    StageSpec(key="initial", title="Initial Opinion", event_type="Initial Opinion"),
    StageSpec(key="final", title="Post Opinion", event_type="Post Opinion"),
)

PUBLIC_STAGE_SPECS = (
    StageSpec(key="tweet_1", title="Tweet 1", event_type="tweet", chat_round_order=1.0),
    StageSpec(key="tweet_3", title="Tweet 3", event_type="tweet", chat_round_order=3.0),
)

def build_histogram_run_configs(row_filter: str) -> tuple[HistogramRunConfig, HistogramRunConfig]:
    if row_filter == "all_rows":
        suffix = "all_rows"
        valid_only_for_llm = False
        filter_malformed_text = False
    elif row_filter == "valid_only":
        suffix = "valid_only"
        valid_only_for_llm = True
        filter_malformed_text = True
    else:
        raise ValueError(f"Unsupported row_filter: {row_filter}")

    return (
        HistogramRunConfig(
            key=f"private_{suffix}",
            title_prefix="Private Opinion",
            output_subdir=f"private_opinion/{suffix}",
            stage_specs=PRIVATE_STAGE_SPECS,
            valid_only_for_llm=valid_only_for_llm,
            filter_malformed_text=filter_malformed_text,
        ),
        HistogramRunConfig(
            key=f"public_{suffix}",
            title_prefix="Public Opinion",
            output_subdir=f"public_opinion/{suffix}",
            stage_specs=PUBLIC_STAGE_SPECS,
            valid_only_for_llm=valid_only_for_llm,
            filter_malformed_text=filter_malformed_text,
        ),
    )


def resolve_existing_or_script_relative(path_text: str) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    alt = (Path(__file__).resolve().parent / path_text).resolve()
    if alt.exists():
        return alt
    return path


def find_model_csv(roots: list[Path], data_prefix: str, model: str, filename: str) -> Path | None:
    for root in roots:
        candidate = root / data_prefix / model / filename
        if candidate.exists():
            return candidate
    return None


def find_human_csv(roots: list[Path], data_prefix: str, models_for_fallback: list[str], filename: str) -> Path | None:
    for root in roots:
        direct = root / data_prefix / filename
        if direct.exists():
            return direct
        for model in models_for_fallback:
            candidate = root / data_prefix / model / filename
            if candidate.exists():
                return candidate
    return None


def get_sidecar_csv_path(csv_path: Path, data_root: Path, sidecar_root: Path, suffix: str) -> Path:
    sidecar_path = sidecar_root / csv_path.relative_to(data_root)
    return sidecar_path.with_name(sidecar_path.stem + suffix)


def get_sidecar_suffixes_for_stage_specs(stage_specs: tuple[StageSpec, StageSpec]) -> tuple[str, ...]:
    event_types = {stage_spec.event_type for stage_spec in stage_specs}
    suffixes: list[str] = []
    if event_types & {"Initial Opinion", "Post Opinion"}:
        suffixes.append("__initial_post_rejudge.csv")
    if "tweet" in event_types:
        suffixes.append("__tweet_rejudge.csv")
    return tuple(suffixes)


def load_sidecar_validity_overrides(
    csv_path: Path,
    roots: list[Path],
    sidecar_roots: list[Path],
    sidecar_suffixes: tuple[str, ...],
) -> dict[int, str]:
    if not sidecar_roots:
        return {}
    for data_root in roots:
        try:
            csv_path.relative_to(data_root)
        except ValueError:
            continue
        for sidecar_root in sidecar_roots:
            for sidecar_suffix in sidecar_suffixes:
                sidecar_path = get_sidecar_csv_path(csv_path, data_root, sidecar_root, sidecar_suffix)
                if not sidecar_path.exists():
                    continue
                overrides: dict[int, str] = {}
                with sidecar_path.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        row_index_text = str(row.get("row_index") or "").strip()
                        new_validity = str(row.get("new_validity") or "").strip()
                        if not row_index_text or not new_validity:
                            continue
                        try:
                            row_index = int(row_index_text)
                        except ValueError:
                            continue
                        overrides[row_index] = new_validity
                return overrides
    return {}


def get_row_text_for_quality_filter(row: dict[str, str]) -> str:
    for field_name in ("llm_text", "text", "message", "post_opinion", "initial_opinion"):
        value = str(row.get(field_name) or "").strip()
        if value:
            return value
    return ""


def is_obviously_malformed_text(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return True

    lowered = normalized.lower()
    words = re.findall(r"[a-zA-Z']+", lowered)
    if len(words) <= 2:
        return True

    placeholder_patterns = (
        r"^my response\s*:?\s*(my opinion)?$",
        r"^my opinion$",
        r"^no comments?",
        r"^this chat ended\.?$",
        r"^i hope to interact with others\.?$",
        r"^hey[,! ]+you'?re still online\??$",
        r"^hey guys? whats going on\?\? i havent seen anything happen yet$",
    )
    if any(re.match(pattern, lowered) for pattern in placeholder_patterns):
        return True

    if lowered.startswith(("hello", "hello?", "hello there", "hey", "hi ")):
        stance_terms = ("agree", "disagree", "support", "oppose", "lean", "opinion", "because")
        if not any(term in lowered for term in stance_terms):
            return True

    nonsense_tokens = [token for token in words if token.startswith("nan") and token != "nanoparticles"]
    if nonsense_tokens:
        return True

    if re.search(r"(.)\1{7,}", lowered):
        return True

    return False


def load_stage_values_from_csv(
    csv_path: Path,
    *,
    likert_col: str,
    stage_specs: tuple[StageSpec, StageSpec],
    filter_malformed_text: bool = True,
    validity_field_candidates: tuple[str, ...] | None = None,
    roots: list[Path] | None = None,
    sidecar_roots: list[Path] | None = None,
) -> dict[str, list[float]]:
    sidecar_validity_overrides = load_sidecar_validity_overrides(
        csv_path,
        roots or [],
        sidecar_roots or [],
        get_sidecar_suffixes_for_stage_specs(stage_specs),
    )
    matching_rows: dict[str, list[dict[str, str]]] = {stage_spec.key: [] for stage_spec in stage_specs}
    all_rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row_index, row in enumerate(reader, start=2):
            sidecar_validity = sidecar_validity_overrides.get(row_index)
            if sidecar_validity:
                row = dict(row)
                row["__sidecar_validity"] = sidecar_validity
            all_rows.append(row)

    for row in filter_rows_for_v3_complete_group_mask(all_rows, csv_path=csv_path):
            event = (row.get("event_type") or "").strip()
            round_text = row.get("chat_round_order")
            try:
                round_value = float(round_text) if round_text not in (None, "") else None
            except ValueError:
                round_value = None
            for stage_spec in stage_specs:
                if event != stage_spec.event_type:
                    continue
                if stage_spec.chat_round_order is not None and round_value != stage_spec.chat_round_order:
                    continue
                matching_rows[stage_spec.key].append(row)
                break

    stage_values = {stage_spec.key: [] for stage_spec in stage_specs}
    for stage_spec in stage_specs:
        rows = matching_rows[stage_spec.key]

        def resolve_validity_value(row: dict[str, str]) -> str:
            if not validity_field_candidates:
                return ""
            sidecar_value = str(row.get("__sidecar_validity") or "").strip()
            if sidecar_value:
                return sidecar_value
            for field_name in validity_field_candidates:
                value = str(row.get(field_name) or "").strip()
                if value:
                    return value
            return ""

        has_explicit_validity = any(resolve_validity_value(row) for row in rows)
        for row in rows:
            resolved_validity = resolve_validity_value(row)
            if validity_field_candidates and has_explicit_validity and resolved_validity.upper() != "VALID":
                continue
            if filter_malformed_text and is_obviously_malformed_text(get_row_text_for_quality_filter(row)):
                continue
            val = parse_likert(row.get(likert_col))
            if val is None or val < 1.0 or val > 6.0:
                continue
            stage_values[stage_spec.key].append(val)
    return stage_values


def histogram_counts(values: list[float]) -> dict[int, int]:
    counts = {k: 0 for k in LIKERT_LEVELS}
    for v in values:
        rounded = int(round(v))
        if rounded in counts:
            counts[rounded] += 1
    return counts


def mean_or_nan(values: list[float]) -> float:
    if not values:
        return math.nan
    return sum(values) / len(values)


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return slug if slug else "item"


def annotate_hist_counts(ax: Any, *, raw_counts: list[int], pct_counts: list[float], patches: Any) -> None:
    if not raw_counts or not pct_counts:
        return
    max_count = max(pct_counts) if pct_counts else 0.0
    if max_count <= 0:
        return
    offset = max(0.008, max_count * 0.015)
    for raw_count, pct_count, patch in zip(raw_counts, pct_counts, patches):
        if raw_count <= 0:
            continue
        x = patch.get_x() + patch.get_width() / 2.0
        y = patch.get_height()
        label_y = min(y + offset, 0.985)
        ax.text(x, label_y, f"{pct_count:.1%} ({raw_count})", ha="center", va="bottom", fontsize=8)


def plot_hist(
    ax: Any,
    *,
    values: list[float],
    title: str,
    color: str,
    alpha: float = 0.9,
    label: str | None = None,
) -> None:
    raw_count_map = histogram_counts(values)
    raw_counts = [raw_count_map[level] for level in LIKERT_LEVELS]
    weights = [1.0 / len(values)] * len(values) if values else None
    pct_counts, _bins, patches = ax.hist(
        values,
        bins=LIKERT_BINS,
        weights=weights,
        color=color,
        alpha=alpha,
        edgecolor="white",
        linewidth=0.8,
        label=label,
    )
    ax.set_xticks(LIKERT_LEVELS)
    ax.set_xlim(0.5, 6.5)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_title(title)
    ax.set_ylabel("Share of responses")
    if PercentFormatter is not None:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    annotate_hist_counts(ax, raw_counts=raw_counts, pct_counts=list(pct_counts), patches=patches)


def save_figure(fig: Any, out_png: Path, *, save_png: bool) -> tuple[str, str]:
    out_pdf = out_png.with_suffix(".pdf")
    png_path = ""
    if save_png:
        fig.savefig(out_png, dpi=220, bbox_inches="tight")
        png_path = str(out_png)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return png_path, str(out_pdf)


def plot_base_histograms(
    *,
    model_name: str,
    run_config: HistogramRunConfig,
    human_values: dict[str, list[float]],
    llm_values: dict[str, list[float]],
    output_dir: Path,
    overlay: bool,
    save_png: bool,
) -> list[dict[str, str]]:
    if plt is None:
        return []

    outputs: list[dict[str, str]] = []
    model_slug = slugify(model_name)
    model_output_dir = output_dir / "base_models" / model_slug
    model_output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, len(run_config.stage_specs), figsize=(6 * len(run_config.stage_specs), 8), sharex=True, sharey=True)
    fig.suptitle(f"{run_config.title_prefix}: Base Model {model_name} (Human vs LLM)", fontsize=14)
    for col_idx, stage_spec in enumerate(run_config.stage_specs):
        plot_hist(axes[0, col_idx], values=human_values.get(stage_spec.key, []), title=f"Human {stage_spec.title}", color=COLOR_HUMAN)
        plot_hist(axes[1, col_idx], values=llm_values.get(stage_spec.key, []), title=f"LLM {stage_spec.title}", color=COLOR_LLM)
        axes[1, col_idx].set_xlabel("Likert")
    axes[0, 0].set_ylabel("Share of responses")
    axes[1, 0].set_ylabel("Share of responses")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    png, pdf = save_figure(
        fig,
        model_output_dir / f"hist_base_{run_config.key}_panels.png",
        save_png=save_png,
    )
    outputs.append({"plot_name": f"base_{model_slug}_{run_config.key}_panels", "png_path": png, "pdf_path": pdf})

    if overlay:
        fig2, axes2 = plt.subplots(1, len(run_config.stage_specs), figsize=(6 * len(run_config.stage_specs), 4.5), sharex=True, sharey=True)
        if len(run_config.stage_specs) == 1:
            axes2 = [axes2]
        fig2.suptitle(f"{run_config.title_prefix} Overlay: {model_name} (Human vs LLM)", fontsize=14)
        for col_idx, stage_spec in enumerate(run_config.stage_specs):
            plot_hist(
                axes2[col_idx],
                values=human_values.get(stage_spec.key, []),
                title=stage_spec.title,
                color=COLOR_HUMAN,
                alpha=0.45,
                label="Human",
            )
            plot_hist(
                axes2[col_idx],
                values=llm_values.get(stage_spec.key, []),
                title=stage_spec.title,
                color=COLOR_LLM,
                alpha=0.45,
                label="LLM",
            )
            axes2[col_idx].set_xlabel("Likert")
        axes2[0].set_ylabel("Share of responses")
        for ax in axes2:
            ax.legend(loc="upper left", frameon=True)
        fig2.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        png2, pdf2 = save_figure(
            fig2,
            model_output_dir / f"hist_base_{run_config.key}_overlay.png",
            save_png=save_png,
        )
        outputs.append({"plot_name": f"base_{model_slug}_{run_config.key}_overlay", "png_path": png2, "pdf_path": pdf2})

    return outputs


def plot_base_aggregate_panels(
    *,
    run_config: HistogramRunConfig,
    base_models: list[str],
    base_model_data: dict[str, dict[str, list[float]]],
    output_dir: Path,
    overlay: bool,
    save_png: bool,
) -> list[dict[str, str]]:
    if plt is None or not base_models:
        return []

    outputs: list[dict[str, str]] = []
    base_aggregate_dir = output_dir / "base_aggregate"
    base_aggregate_dir.mkdir(parents=True, exist_ok=True)
    rows = len(base_models)
    num_stage_cols = len(run_config.stage_specs) * 2

    fig, axes = plt.subplots(rows, num_stage_cols, figsize=(4.2 * num_stage_cols, max(2.6 * rows, 4.0)), sharex=True, sharey=True)
    if rows == 1:
        axes = [axes]

    col_titles: list[str] = []
    for stage_spec in run_config.stage_specs:
        col_titles.extend([f"Human {stage_spec.title}", f"LLM {stage_spec.title}"])
    for j, t in enumerate(col_titles):
        axes[0][j].set_title(t)

    for i, model in enumerate(base_models):
        series = base_model_data.get(model, {})
        for stage_idx, stage_spec in enumerate(run_config.stage_specs):
            human_col = stage_idx * 2
            llm_col = human_col + 1
            plot_hist(axes[i][human_col], values=series.get(f"human_{stage_spec.key}", []), title="" if i > 0 else col_titles[human_col], color=COLOR_HUMAN)
            plot_hist(axes[i][llm_col], values=series.get(f"llm_{stage_spec.key}", []), title="" if i > 0 else col_titles[llm_col], color=COLOR_LLM)

        axes[i][0].set_ylabel(f"{model}\nShare of responses", fontsize=9)
        if i == rows - 1:
            for j in range(num_stage_cols):
                axes[i][j].set_xlabel("Likert")

    fig.suptitle(f"Base Models Aggregate Panel: {run_config.title_prefix} (Human vs LLM)", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    png, pdf = save_figure(
        fig,
        base_aggregate_dir / f"hist_base_all_models_{run_config.key}_panels.png",
        save_png=save_png,
    )
    outputs.append({"plot_name": f"base_all_models_{run_config.key}_panels", "png_path": png, "pdf_path": pdf})

    if overlay:
        fig2, axes2 = plt.subplots(rows, len(run_config.stage_specs), figsize=(6 * len(run_config.stage_specs), max(2.6 * rows, 4.0)), sharex=True, sharey=True)
        if rows == 1:
            axes2 = [axes2]

        for stage_idx, stage_spec in enumerate(run_config.stage_specs):
            axes2[0][stage_idx].set_title(stage_spec.title)

        for i, model in enumerate(base_models):
            series = base_model_data.get(model, {})
            for stage_idx, stage_spec in enumerate(run_config.stage_specs):
                plot_hist(
                    axes2[i][stage_idx],
                    values=series.get(f"human_{stage_spec.key}", []),
                    title="" if i > 0 else stage_spec.title,
                    color=COLOR_HUMAN,
                    alpha=0.45,
                    label="Human",
                )
                plot_hist(
                    axes2[i][stage_idx],
                    values=series.get(f"llm_{stage_spec.key}", []),
                    title="" if i > 0 else stage_spec.title,
                    color=COLOR_LLM,
                    alpha=0.45,
                    label="LLM",
                )

            axes2[i][0].set_ylabel(f"{model}\nShare of responses", fontsize=9)
            if i == rows - 1:
                for stage_idx in range(len(run_config.stage_specs)):
                    axes2[i][stage_idx].set_xlabel("Likert")

        for stage_idx in range(len(run_config.stage_specs)):
            axes2[0][stage_idx].legend(loc="upper left", frameon=True)
        fig2.suptitle(f"Base Models Aggregate Overlay Panel: {run_config.title_prefix}", fontsize=14)
        fig2.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        png2, pdf2 = save_figure(
            fig2,
            base_aggregate_dir / f"hist_base_all_models_{run_config.key}_overlay_panel.png",
            save_png=save_png,
        )
        outputs.append({"plot_name": f"base_all_models_{run_config.key}_overlay_panel", "png_path": png2, "pdf_path": pdf2})

    return outputs


def collect_series_for_base_model_pair(
    *,
    prefixes: list[str],
    roots: list[Path],
    sidecar_roots: list[Path],
    model: str,
    human_models_for_fallback: list[str] | None,
    llm_filename: str,
    human_filename: str,
    likert_col: str,
    stage_specs: tuple[StageSpec, StageSpec],
    valid_only_for_llm: bool,
    status_rows: list[dict[str, Any]],
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    human_values = {stage_spec.key: [] for stage_spec in stage_specs}
    llm_values = {stage_spec.key: [] for stage_spec in stage_specs}

    phase = f"base_model_{slugify(model)}"
    for data_prefix in prefixes:
        llm_path = find_model_csv(roots, data_prefix, model, llm_filename)
        if llm_path is None:
            status_rows.append(
                {
                    "phase": phase,
                    "data_prefix": data_prefix,
                    "entity": model,
                    "status": "missing_csv",
                    "csv_path": "",
                    "message": "Model CSV missing.",
                }
            )
            continue

        human_path = find_human_csv(
            roots,
            data_prefix,
            human_models_for_fallback or [model],
            human_filename,
        )
        if human_path is None:
            status_rows.append(
                {
                    "phase": phase,
                    "data_prefix": data_prefix,
                    "entity": "human",
                    "status": "missing_csv",
                    "csv_path": "",
                    "message": "Human CSV missing for this model/prefix.",
                }
            )
            continue

        try:
            human_stage_values = load_stage_values_from_csv(
                human_path,
                likert_col=likert_col,
                stage_specs=stage_specs,
                filter_malformed_text=valid_only_for_llm,
                validity_field_candidates=None,
                roots=roots,
                sidecar_roots=sidecar_roots,
            )
            llm_stage_values = load_stage_values_from_csv(
                llm_path,
                likert_col=likert_col,
                stage_specs=stage_specs,
                filter_malformed_text=valid_only_for_llm,
                validity_field_candidates=("llm_validity", "validity") if valid_only_for_llm else None,
                roots=roots,
                sidecar_roots=sidecar_roots,
            )
        except Exception as exc:
            status_rows.append(
                {
                    "phase": phase,
                    "data_prefix": data_prefix,
                    "entity": model,
                    "status": "csv_error",
                    "csv_path": f"{human_path} | {llm_path}",
                    "message": str(exc),
                }
            )
            continue

        for stage_spec in stage_specs:
            human_values[stage_spec.key].extend(human_stage_values.get(stage_spec.key, []))
            llm_values[stage_spec.key].extend(llm_stage_values.get(stage_spec.key, []))

        status_rows.append(
            {
                "phase": phase,
                "data_prefix": data_prefix,
                "entity": "human",
                "status": "ok",
                "csv_path": str(human_path),
                "message": ", ".join(f"{stage_spec.key}={len(human_stage_values.get(stage_spec.key, []))}" for stage_spec in stage_specs),
            }
        )
        status_rows.append(
            {
                "phase": phase,
                "data_prefix": data_prefix,
                "entity": model,
                "status": "ok",
                "csv_path": str(llm_path),
                "message": ", ".join(f"{stage_spec.key}={len(llm_stage_values.get(stage_spec.key, []))}" for stage_spec in stage_specs),
            }
        )

    return human_values, llm_values


def plot_split_histograms(
    *,
    split_label: str,
    title: str,
    run_config: HistogramRunConfig,
    human_values: dict[str, list[float]],
    series_values: list[tuple[SplitModelSpec, dict[str, list[float]]]],
    panel_rows: tuple[tuple[str, ...], ...] | None,
    n_groups: int,
    output_dir: Path,
    overlay: bool,
    save_png: bool,
) -> list[dict[str, str]]:
    if plt is None:
        return []

    outputs: list[dict[str, str]] = []
    split_output_dir = output_dir / "splits" / split_label
    split_output_dir.mkdir(parents=True, exist_ok=True)
    series_value_map = {series_spec.key: (series_spec, values) for series_spec, values in series_values}
    panel_rows = panel_rows or default_split_panel_rows(tuple(series_spec for series_spec, _ in series_values))
    column_count = max(len(panel_row) for panel_row in panel_rows)
    grid_row_count = len(run_config.stage_specs) * len(panel_rows)
    grouped_layout = len(panel_rows) > 1
    split_model_specs = [series_spec for series_spec, _ in series_values]
    footnote = build_split_model_footnote(split_model_specs)
    fig, axes = plt.subplots(
        grid_row_count,
        column_count,
        figsize=(4.25 * column_count, 2.9 * grid_row_count),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    fig.suptitle(title, fontsize=14)

    for stage_idx, stage_spec in enumerate(run_config.stage_specs):
        for panel_row_idx, panel_row in enumerate(panel_rows):
            grid_row_idx = stage_idx * len(panel_rows) + panel_row_idx
            for col_idx in range(column_count):
                ax = axes[grid_row_idx][col_idx]
                if col_idx >= len(panel_row):
                    ax.axis("off")
                    continue
                panel_key = panel_row[col_idx]
                if panel_key == "human":
                    plot_hist(
                        ax,
                        values=human_values.get(stage_spec.key, []),
                        title=split_panel_title(panel_title="Human", stage_spec=stage_spec, n_groups=n_groups, grouped_layout=grouped_layout),
                        color=COLOR_HUMAN,
                    )
                    continue
                series_spec, values = series_value_map[panel_key]
                plot_hist(
                    ax,
                    values=values.get(stage_spec.key, []),
                    title=split_panel_title(panel_title=series_spec.title, stage_spec=stage_spec, n_groups=n_groups, grouped_layout=grouped_layout),
                    color=series_spec.color,
                )
            axes[grid_row_idx][0].set_ylabel("Share of responses")
    for j in range(column_count):
        axes[-1][j].set_xlabel("Likert")

    fig.text(0.5, 0.015, footnote, ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.95 if not grouped_layout else 0.93))
    if grouped_layout:
        for stage_idx, stage_spec in enumerate(run_config.stage_specs):
            header_row_idx = stage_idx * len(panel_rows)
            header_y = min(0.955, axes[header_row_idx][0].get_position().y1 + 0.012)
            fig.text(
                0.5,
                header_y,
                split_stage_section_title(stage_spec),
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )
    png, pdf = save_figure(
        fig,
        split_output_dir / f"hist_{split_label}_{run_config.key}_panels.png",
        save_png=save_png,
    )
    outputs.append({"plot_name": f"{split_label}_{run_config.key}_panels", "png_path": png, "pdf_path": pdf})

    if overlay:
        fig2, axes2 = plt.subplots(1, len(run_config.stage_specs), figsize=(6 * len(run_config.stage_specs), 4.5), sharex=True, sharey=True)
        if len(run_config.stage_specs) == 1:
            axes2 = [axes2]
        fig2.suptitle(f"{title} (Overlay)", fontsize=14)

        for col_idx, stage_spec in enumerate(run_config.stage_specs):
            plot_hist(axes2[col_idx], values=human_values.get(stage_spec.key, []), title=stage_spec.title, color=COLOR_HUMAN, alpha=0.35, label="Human")
            for series_spec, values in series_values:
                plot_hist(
                    axes2[col_idx],
                    values=values.get(stage_spec.key, []),
                    title=stage_spec.title,
                    color=series_spec.color,
                    alpha=0.35,
                    label=series_spec.title,
                )
            axes2[col_idx].set_xlabel("Likert")
        axes2[0].set_ylabel("Share of responses")
        for ax in axes2:
            ax.legend(loc="upper left", frameon=True)
        fig2.text(0.5, 0.015, footnote, ha="center", va="bottom", fontsize=8)
        fig2.tight_layout(rect=(0.0, 0.05, 1.0, 0.95))
        png2, pdf2 = save_figure(
            fig2,
            split_output_dir / f"hist_{split_label}_{run_config.key}_overlay.png",
            save_png=save_png,
        )
        outputs.append({"plot_name": f"{split_label}_{run_config.key}_overlay", "png_path": png2, "pdf_path": pdf2})

    return outputs


def collect_series_for_model(
    *,
    prefixes: list[str],
    roots: list[Path],
    sidecar_roots: list[Path],
    model: str,
    llm_filename: str,
    likert_col: str,
    stage_specs: tuple[StageSpec, StageSpec],
    valid_only_for_llm: bool,
    status_rows: list[dict[str, Any]],
    phase: str,
) -> dict[str, list[float]]:
    stage_values = {stage_spec.key: [] for stage_spec in stage_specs}
    for data_prefix in prefixes:
        csv_path = find_model_csv(roots, data_prefix, model, llm_filename)
        if csv_path is None:
            status_rows.append(
                {
                    "phase": phase,
                    "data_prefix": data_prefix,
                    "entity": model,
                    "status": "missing_csv",
                    "csv_path": "",
                    "message": "Model CSV missing.",
                }
            )
            continue
        try:
            current_stage_values = load_stage_values_from_csv(
                csv_path,
                likert_col=likert_col,
                stage_specs=stage_specs,
                filter_malformed_text=valid_only_for_llm,
                validity_field_candidates=("llm_validity", "validity") if valid_only_for_llm else None,
                roots=roots,
                sidecar_roots=sidecar_roots,
            )
        except Exception as exc:
            status_rows.append(
                {
                    "phase": phase,
                    "data_prefix": data_prefix,
                    "entity": model,
                    "status": "csv_error",
                    "csv_path": str(csv_path),
                    "message": str(exc),
                }
            )
            continue
        for stage_spec in stage_specs:
            stage_values[stage_spec.key].extend(current_stage_values.get(stage_spec.key, []))
        status_rows.append(
            {
                "phase": phase,
                "data_prefix": data_prefix,
                "entity": model,
                "status": "ok",
                "csv_path": str(csv_path),
                "message": ", ".join(f"{stage_spec.key}={len(current_stage_values.get(stage_spec.key, []))}" for stage_spec in stage_specs),
            }
        )
    return stage_values


def collect_series_for_human(
    *,
    prefixes: list[str],
    roots: list[Path],
    sidecar_roots: list[Path],
    models_for_fallback: list[str],
    human_filename: str,
    likert_col: str,
    stage_specs: tuple[StageSpec, StageSpec],
    valid_only_for_human: bool,
    status_rows: list[dict[str, Any]],
    phase: str,
) -> dict[str, list[float]]:
    stage_values = {stage_spec.key: [] for stage_spec in stage_specs}
    for data_prefix in prefixes:
        csv_path = find_human_csv(roots, data_prefix, models_for_fallback, human_filename)
        if csv_path is None:
            status_rows.append(
                {
                    "phase": phase,
                    "data_prefix": data_prefix,
                    "entity": "human",
                    "status": "missing_csv",
                    "csv_path": "",
                    "message": "Human CSV missing.",
                }
            )
            continue
        try:
            current_stage_values = load_stage_values_from_csv(
                csv_path,
                likert_col=likert_col,
                stage_specs=stage_specs,
                filter_malformed_text=valid_only_for_human,
                validity_field_candidates=("human_validity", "validity") if valid_only_for_human else None,
                roots=roots,
                sidecar_roots=sidecar_roots,
            )
        except Exception as exc:
            status_rows.append(
                {
                    "phase": phase,
                    "data_prefix": data_prefix,
                    "entity": "human",
                    "status": "csv_error",
                    "csv_path": str(csv_path),
                    "message": str(exc),
                }
            )
            continue
        for stage_spec in stage_specs:
            stage_values[stage_spec.key].extend(current_stage_values.get(stage_spec.key, []))
        status_rows.append(
            {
                "phase": phase,
                "data_prefix": data_prefix,
                "entity": "human",
                "status": "ok",
                "csv_path": str(csv_path),
                "message": ", ".join(f"{stage_spec.key}={len(current_stage_values.get(stage_spec.key, []))}" for stage_spec in stage_specs),
            }
        )
    return stage_values


def append_summary_rows(
    *,
    rows: list[dict[str, Any]],
    analysis_group: str,
    entity: str,
    stage: str,
    values: list[float],
) -> None:
    counts = histogram_counts(values)
    rows.append(
        {
            "analysis_group": analysis_group,
            "entity": entity,
            "stage": stage,
            "n": len(values),
            "mean_likert": mean_or_nan(values),
            "count_1": counts[1],
            "count_2": counts[2],
            "count_3": counts[3],
            "count_4": counts[4],
            "count_5": counts[5],
            "count_6": counts[6],
        }
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build opinion histograms for base models and post-training split comparisons.")
    parser.add_argument("--data-root", default="../../result/eval_v3/human_llm", help="Root of human_llm outputs.")
    parser.add_argument("--fallback-data-root", action="append", default=[], help="Additional data root(s).")
    parser.add_argument("--validity-sidecar-root", action="append", default=[], help="Optional sidecar root(s) with validity sidecars to override row validity. Backward-compatible for initial/post sidecars.")
    parser.add_argument("--tweet-validity-sidecar-root", action="append", default=[], help="Optional sidecar root(s) with __tweet_rejudge.csv files to override tweet validity by row_index.")
    parser.add_argument("--data-prefix-file", default="../../inputs/masks_train_test/golden_group_v0/aggregate_universe_mask_golden_v0.txt", help="Base prefixes file.")
    parser.add_argument("--models-file", default="models.txt", help="Base models file.")
    parser.add_argument("--human-filename", default=HUMAN_DEFAULT_FILENAME, help=f"Human CSV filename (default: {HUMAN_DEFAULT_FILENAME}).")
    parser.add_argument("--llm-filename", default=LLM_DEFAULT_FILENAME, help=f"LLM CSV filename (default: {LLM_DEFAULT_FILENAME}).")
    parser.add_argument("--likert-col", default="likert_pred", help="Column name for opinion labels/scores.")
    parser.add_argument("--report-version", default="v3", help="Report/result version label used to choose the post-training model family.")
    parser.add_argument(
        "--row-filter",
        choices=("valid_only", "all_rows"),
        default="valid_only",
        help="Whether to filter to rows marked VALID or include all parseable rows.",
    )
    parser.add_argument("--initial-event", default="Initial Opinion", help='Initial event label (default: "Initial Opinion").')
    parser.add_argument("--final-event", default="Post Opinion", help='Final event label (default: "Post Opinion").')
    parser.add_argument("--output-dir", default="result/radar/histograms", help="Output directory.")

    parser.add_argument(
        "--breadth-prefix-file",
        default="../../inputs/masks_train_test/golden_group_v0/breadth/breadth_universe_mask_golden_v0.txt",
        help="Breadth collection prefixes without split masks.",
    )
    parser.add_argument(
        "--depth-prefix-file",
        default="../../inputs/masks_train_test/golden_group_v0/depth/depth_universe_mask_golden_v0.txt",
        help="Depth collection prefixes without split masks.",
    )

    parser.add_argument(
        "--topic-train-prefix-file",
        default="../../inputs/masks_train_test/golden_group_v0/breadth/breadth_train_topic_mask_golden_v0.txt",
        help="Train topic split prefixes.",
    )
    parser.add_argument(
        "--topic-test-prefix-file",
        default="../../inputs/masks_train_test/golden_group_v0/breadth/breadth_test_topic_mask_golden_v0.txt",
        help="Test topic split prefixes.",
    )
    parser.add_argument(
        "--group-train-prefix-file",
        default="../../inputs/masks_train_test/golden_group_v0/breadth/breadth_train_group_mask_golden_v0.txt",
        help="Train group split prefixes.",
    )
    parser.add_argument(
        "--group-test-prefix-file",
        default="../../inputs/masks_train_test/golden_group_v0/breadth/breadth_test_group_mask_golden_v0.txt",
        help="Test group split prefixes.",
    )

    parser.add_argument(
        "--depth-topic-train-prefix-file",
        default="../../inputs/masks_train_test/golden_group_v0/depth/depth_train_topic_mask_golden_v0.txt",
        help="Depth train topic split prefixes.",
    )
    parser.add_argument(
        "--depth-topic-test-prefix-file",
        default="../../inputs/masks_train_test/golden_group_v0/depth/depth_test_topic_mask_golden_v0.txt",
        help="Depth test topic split prefixes.",
    )
    parser.add_argument(
        "--depth-group-train-prefix-file",
        default="../../inputs/masks_train_test/golden_group_v0/depth/depth_train_group_mask_golden_v0.txt",
        help="Depth train group split prefixes.",
    )
    parser.add_argument(
        "--depth-group-test-prefix-file",
        default="../../inputs/masks_train_test/golden_group_v0/depth/depth_test_group_mask_golden_v0.txt",
        help="Depth test group split prefixes.",
    )

    parser.add_argument("--overlay", dest="overlay", action="store_true", help="Also save overlay histograms.")
    parser.add_argument("--no-overlay", dest="overlay", action="store_false", help="Disable overlay histograms.")
    parser.set_defaults(overlay=True)

    parser.add_argument("--save-png", dest="save_png", action="store_true", help="Also save PNG files alongside PDFs.")
    parser.add_argument("--no-save-png", dest="save_png", action="store_false", help="Disable PNG output.")
    parser.set_defaults(save_png=False)

    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars.")
    return parser


def build_split_configs(args: argparse.Namespace) -> list[SplitConfig]:
    use_v4 = report_version_family(args.report_version) == "v4"

    breadth_topic_models = TOPIC_MODELS_V4 if use_v4 else TOPIC_MODELS
    breadth_group_models = GROUP_MODELS_V4 if use_v4 else GROUP_MODELS

    topic_model_series = [
        SplitModelSpec(key="llm", title="LLM", model=breadth_topic_models[0], color=COLOR_LLM),
        SplitModelSpec(key="sft", title="SFT", model=breadth_topic_models[1], color=COLOR_SFT),
    ]
    if not use_v4:
        topic_model_series.append(SplitModelSpec(key="dpo", title="DPO", model=TOPIC_MODELS[2], color=COLOR_DPO))

    group_model_series = [
        SplitModelSpec(key="llm", title="LLM", model=breadth_group_models[0], color=COLOR_LLM),
        SplitModelSpec(key="sft", title="SFT", model=breadth_group_models[1], color=COLOR_SFT),
    ]
    if not use_v4:
        group_model_series.append(SplitModelSpec(key="dpo", title="DPO", model=GROUP_MODELS[2], color=COLOR_DPO))

    if use_v4:
        depth_topic_model_series = [
            SplitModelSpec(key="llm", title="LLaMA instruct", model=DEPTH_TOPIC_MODELS_V4[0], color=COLOR_LLM),
            SplitModelSpec(key="llama_sft", title="LLaMA SFT", model=DEPTH_TOPIC_MODELS_V4[2], color=COLOR_SFT),
            SplitModelSpec(key="gpt_base", title="gpt4omini", model=DEPTH_TOPIC_MODELS_V4[1], color=COLOR_GPT_BASE),
            SplitModelSpec(key="gpt_sft", title="gpt4omini SFT", model=DEPTH_TOPIC_MODELS_V4[3], color=COLOR_GPT_SFT),
        ]
        depth_group_model_series = [
            SplitModelSpec(key="llm", title="LLaMA instruct", model=DEPTH_GROUP_MODELS_V4[1], color=COLOR_LLM),
            SplitModelSpec(key="llama_sft", title="LLaMA SFT", model=DEPTH_GROUP_MODELS_V4[2], color=COLOR_SFT),
            SplitModelSpec(key="gpt_base", title="gpt4omini", model=DEPTH_GROUP_MODELS_V4[0], color=COLOR_GPT_BASE),
            SplitModelSpec(key="gpt_sft", title="gpt4omini SFT", model=DEPTH_GROUP_MODELS_V4[3], color=COLOR_GPT_SFT),
        ]
        depth_panel_rows = (("human", "llm", "llama_sft"), ("human", "gpt_base", "gpt_sft"))
    else:
        depth_topic_model_series = [
            SplitModelSpec(key="llm", title="LLaMA instruct", model=DEPTH_TOPIC_MODELS[0], color=COLOR_LLM),
            SplitModelSpec(key="gpt_base", title="gpt4omini", model=GPT_BASE_MODEL, color=COLOR_GPT_BASE),
            SplitModelSpec(key="llama_sft", title="LLaMA SFT", model=DEPTH_TOPIC_MODELS[1], color=COLOR_SFT),
            SplitModelSpec(key="gpt_sft", title="gpt4omini SFT", model=DEPTH_TOPIC_MODELS[2], color=COLOR_GPT_SFT),
            SplitModelSpec(key="dpo", title="LLaMA DPO", model=DEPTH_TOPIC_MODELS[3], color=COLOR_DPO),
        ]
        depth_group_model_series = [
            SplitModelSpec(key="llm", title="LLaMA instruct", model=DEPTH_GROUP_MODELS[0], color=COLOR_LLM),
            SplitModelSpec(key="gpt_base", title="gpt4omini", model=GPT_BASE_MODEL, color=COLOR_GPT_BASE),
            SplitModelSpec(key="llama_sft", title="LLaMA SFT", model=DEPTH_GROUP_MODELS[1], color=COLOR_SFT),
            SplitModelSpec(key="gpt_sft", title="gpt4omini SFT", model=DEPTH_GROUP_MODELS[2], color=COLOR_GPT_SFT),
            SplitModelSpec(key="dpo", title="LLaMA DPO", model=DEPTH_GROUP_MODELS[3], color=COLOR_DPO),
        ]
        depth_panel_rows = (("human", "llm", "llama_sft", "dpo"), ("human", "gpt_base", "gpt_sft"))

    return [
        SplitConfig(
            name="breadth_topic",
            title="Breadth Topic Split",
            partitions=(
                SplitPartitionSpec(key="train", title="Train", prefix_file=resolve_existing_or_script_relative(args.topic_train_prefix_file)),
                SplitPartitionSpec(key="test", title="Test", prefix_file=resolve_existing_or_script_relative(args.topic_test_prefix_file)),
                SplitPartitionSpec(key="overall", title="Overall", prefix_file=resolve_existing_or_script_relative(args.breadth_prefix_file)),
            ),
            model_series=tuple(topic_model_series),
        ),
        SplitConfig(
            name="breadth_group",
            title="Breadth Group Split",
            partitions=(
                SplitPartitionSpec(key="train", title="Train", prefix_file=resolve_existing_or_script_relative(args.group_train_prefix_file)),
                SplitPartitionSpec(key="test", title="Test", prefix_file=resolve_existing_or_script_relative(args.group_test_prefix_file)),
                SplitPartitionSpec(key="overall", title="Overall", prefix_file=resolve_existing_or_script_relative(args.breadth_prefix_file)),
            ),
            model_series=tuple(group_model_series),
        ),
        SplitConfig(
            name="depth_topic",
            title="Depth Topic Split",
            partitions=(
                SplitPartitionSpec(key="train", title="Train", prefix_file=resolve_existing_or_script_relative(args.depth_topic_train_prefix_file)),
                SplitPartitionSpec(key="test", title="Test", prefix_file=resolve_existing_or_script_relative(args.depth_topic_test_prefix_file)),
                SplitPartitionSpec(key="overall", title="Overall", prefix_file=resolve_existing_or_script_relative(args.depth_prefix_file)),
            ),
            model_series=tuple(depth_topic_model_series),
            panel_rows=depth_panel_rows,
        ),
        SplitConfig(
            name="depth_group",
            title="Depth Group Split",
            partitions=(
                SplitPartitionSpec(key="train", title="Train", prefix_file=resolve_existing_or_script_relative(args.depth_group_train_prefix_file)),
                SplitPartitionSpec(key="test", title="Test", prefix_file=resolve_existing_or_script_relative(args.depth_group_test_prefix_file)),
                SplitPartitionSpec(key="overall", title="Overall", prefix_file=resolve_existing_or_script_relative(args.depth_prefix_file)),
            ),
            model_series=tuple(depth_group_model_series),
            panel_rows=depth_panel_rows,
        ),
    ]


def main() -> int:
    args = build_arg_parser().parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    roots = [resolve_existing_or_script_relative(args.data_root)]
    roots.extend(resolve_existing_or_script_relative(p) for p in args.fallback_data_root)
    sidecar_roots = [resolve_existing_or_script_relative(p) for p in args.validity_sidecar_root]
    sidecar_roots.extend(resolve_existing_or_script_relative(p) for p in args.tweet_validity_sidecar_root)

    base_prefix_file = resolve_existing_or_script_relative(args.data_prefix_file)
    models_file = resolve_existing_or_script_relative(args.models_file)
    if not base_prefix_file.exists():
        print(f"Base prefix file not found: {base_prefix_file}")
        return 1
    if not models_file.exists():
        print(f"Models file not found: {models_file}")
        return 1

    base_prefixes = read_lines(base_prefix_file)
    base_models = ordered_base_model_llms_for_report(args.report_version, read_lines(models_file))

    split_configs = build_split_configs(args)

    for run_config in build_histogram_run_configs(args.row_filter):
        run_output_dir = output_dir / run_config.output_subdir
        run_output_dir.mkdir(parents=True, exist_ok=True)
        summary_rows: list[dict[str, Any]] = []
        status_rows: list[dict[str, Any]] = []
        plot_rows: list[dict[str, str]] = []
        base_model_data: dict[str, dict[str, list[float]]] = {}

        if not args.no_progress and tqdm is not None:
            print(f"Collecting base-model hist data: {run_config.key}...")

        model_iter: Any = base_models
        if not args.no_progress and tqdm is not None:
            model_iter = tqdm(base_models, total=len(base_models), desc=f"Base model comparisons ({run_config.key})", unit="model")

        for base_model in model_iter:
            human_values, llm_values = collect_series_for_base_model_pair(
                prefixes=base_prefixes,
                roots=roots,
                sidecar_roots=sidecar_roots,
                model=base_model,
                human_models_for_fallback=base_models,
                llm_filename=args.llm_filename,
                human_filename=args.human_filename,
                likert_col=args.likert_col,
                stage_specs=run_config.stage_specs,
                valid_only_for_llm=run_config.valid_only_for_llm,
                status_rows=status_rows,
            )

            analysis_group = f"{run_config.key}__base_model_{slugify(base_model)}"
            for stage_spec in run_config.stage_specs:
                append_summary_rows(rows=summary_rows, analysis_group=analysis_group, entity="human", stage=stage_spec.key, values=human_values.get(stage_spec.key, []))
                append_summary_rows(rows=summary_rows, analysis_group=analysis_group, entity=base_model, stage=stage_spec.key, values=llm_values.get(stage_spec.key, []))
                base_model_data.setdefault(base_model, {})[f"human_{stage_spec.key}"] = human_values.get(stage_spec.key, [])
                base_model_data.setdefault(base_model, {})[f"llm_{stage_spec.key}"] = llm_values.get(stage_spec.key, [])

            plot_rows.extend(
                plot_base_histograms(
                    model_name=base_model,
                    run_config=run_config,
                    human_values=human_values,
                    llm_values=llm_values,
                    output_dir=run_output_dir,
                    overlay=args.overlay,
                    save_png=args.save_png,
                )
            )

        plot_rows.extend(
            plot_base_aggregate_panels(
                run_config=run_config,
                base_models=base_models,
                base_model_data=base_model_data,
                output_dir=run_output_dir,
                overlay=args.overlay,
                save_png=args.save_png,
            )
        )

        for cfg in split_configs:
            for partition in cfg.partitions:
                split_key = f"{cfg.name}_{partition.key}"
                analysis_group = f"{run_config.key}__{split_key}"
                prefix_file = partition.prefix_file
                if not prefix_file.exists():
                    status_rows.append(
                        {
                            "phase": analysis_group,
                            "data_prefix": "",
                            "entity": "all",
                            "status": "missing_prefix_file",
                            "csv_path": str(prefix_file),
                            "message": "Split prefix file missing; skipped.",
                        }
                    )
                    continue

                prefixes = read_lines(prefix_file)
                if not prefixes:
                    status_rows.append(
                        {
                            "phase": analysis_group,
                            "data_prefix": "",
                            "entity": "all",
                            "status": "empty_prefix_file",
                            "csv_path": str(prefix_file),
                            "message": "No prefixes in split file; skipped.",
                        }
                    )
                    continue

                print(f"Collecting split hist data: {analysis_group}")
                model_fallbacks = [series_spec.model for series_spec in cfg.model_series]
                human_values = collect_series_for_human(
                    prefixes=prefixes,
                    roots=roots,
                    sidecar_roots=sidecar_roots,
                    models_for_fallback=model_fallbacks,
                    human_filename=args.human_filename,
                    likert_col=args.likert_col,
                    stage_specs=run_config.stage_specs,
                    valid_only_for_human=run_config.valid_only_for_llm,
                    status_rows=status_rows,
                    phase=analysis_group,
                )
                split_series_values: list[tuple[SplitModelSpec, dict[str, list[float]]]] = []
                for series_spec in cfg.model_series:
                    split_series_values.append(
                        (
                            series_spec,
                            collect_series_for_model(
                                prefixes=prefixes,
                                roots=roots,
                                sidecar_roots=sidecar_roots,
                                model=series_spec.model,
                                llm_filename=args.llm_filename,
                                likert_col=args.likert_col,
                                stage_specs=run_config.stage_specs,
                                valid_only_for_llm=run_config.valid_only_for_llm,
                                status_rows=status_rows,
                                phase=analysis_group,
                            ),
                        )
                    )

                for stage_spec in run_config.stage_specs:
                    append_summary_rows(rows=summary_rows, analysis_group=analysis_group, entity="human", stage=stage_spec.key, values=human_values.get(stage_spec.key, []))
                    for series_spec, series_values in split_series_values:
                        append_summary_rows(
                            rows=summary_rows,
                            analysis_group=analysis_group,
                            entity=series_spec.key,
                            stage=stage_spec.key,
                            values=series_values.get(stage_spec.key, []),
                        )

                split_output = plot_split_histograms(
                    split_label=split_key,
                    title=f"{cfg.title} {partition.title}: {run_config.title_prefix} / Human / Post-Train Models",
                    run_config=run_config,
                    human_values=human_values,
                    series_values=split_series_values,
                    panel_rows=cfg.panel_rows,
                    n_groups=len(prefixes),
                    output_dir=run_output_dir,
                    overlay=args.overlay,
                    save_png=args.save_png,
                )
                plot_rows.extend(split_output)

        summary_path = run_output_dir / "histogram_summary_counts.csv"
        status_path = run_output_dir / "histogram_collection_status.csv"
        plot_path = run_output_dir / "histogram_plot_manifest.csv"

        save_csv(
            summary_path,
            summary_rows,
            [
                "analysis_group",
                "entity",
                "stage",
                "n",
                "mean_likert",
                "count_1",
                "count_2",
                "count_3",
                "count_4",
                "count_5",
                "count_6",
            ],
        )
        save_csv(
            status_path,
            status_rows,
            ["phase", "data_prefix", "entity", "status", "csv_path", "message"],
        )
        save_csv(
            plot_path,
            plot_rows,
            ["plot_name", "png_path", "pdf_path"],
        )

        print(f"Wrote: {summary_path}")
        print(f"Wrote: {status_path}")
        print(f"Wrote: {plot_path}")

    if plt is None:
        print("matplotlib not available; wrote CSV summaries only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
