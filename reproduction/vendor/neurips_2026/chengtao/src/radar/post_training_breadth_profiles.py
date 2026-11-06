#!/usr/bin/env python3
"""Compute/plot post-training profiles for breadth/depth group/topic train/test splits."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from generate_plots import (
    BASE_MODEL_V3_LLM_ORDER,
    DEFAULT_UTTERANCE_EVENT_TYPES,
    HUMAN_DEFAULT_FILENAME,
    LLM_DEFAULT_FILENAME,
    ModelAggregate,
    RADAR_AXES,
    RADAR_AXES_V1,
    PROFILE_Y_LIM,
    append_human_baseline_raw_rows,
    append_human_baseline_score_rows,
    append_reference_score_rows_v1,
    add_v2_bootstrap_cis_to_raw_rows,
    add_model_footnotes,
    build_raw_rows_from_aggregates,
    collect_fieldnames,
    build_score_rows_from_raw_rows,
    compute_prefix_model,
    deterministic_shuffle,
    get_model_color,
    get_model_display_label,
    get_model_footnotes,
    get_v2_display_model_order,
    has_plottable_scores,
    merge_prefix_result_into_aggregate,
    normalize_minmax,
    plot_profile,
    plot_v2_bar_metrics,
    read_lines,
    safe_mean,
    save_csv,
    tqdm,
)
from neurips_2026.sean_reproduce.result_version_config import (
    LLAMA_V4_BREADTH_GROUP_SFT_MODEL,
    LLAMA_V4_BREADTH_TOPIC_SFT_MODEL,
    LLAMA_V4_DEPTH_GROUP_SFT_MODEL,
    LLAMA_V4_DEPTH_TOPIC_SFT_MODEL,
)

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


PROFILE_LINE_STYLES = ("-", "--", ":", "-.")
PROFILE_Y_LIM = (-0.05, 1.05)

AXES_7 = [
    ("stance_difference", "Stance Difference"),
    ("public_stance_alignment", "Public Alignment"),
    ("private_stance_alignment", "Private Alignment"),
    ("public_stance_change_error", "Public Change Error"),
    ("private_stance_change_error", "Private Change Error"),
    ("public_opinion_convergence", "Public Convergence Prediction"),
    ("private_opinion_convergence", "Private Convergence Prediction"),
]

AXIS_GROUP = {
    "stance_difference": "message",
    "public_stance_alignment": "message",
    "private_stance_alignment": "message",
    "public_stance_change_error": "individual",
    "private_stance_change_error": "individual",
    "public_opinion_convergence": "group",
    "private_opinion_convergence": "group",
}

GROUP_TEXT_COLORS = {
    "message": "#1f77b4",
    "group": "#2ca02c",
    "individual": "#d62728",
}

BASE_MODELS = list(BASE_MODEL_V3_LLM_ORDER)
TULU_BASE_MODEL = "Llama-3.1-Tulu-3-8B-SFT"
V4_BASE_MODELS = [model for model in BASE_MODELS if model != TULU_BASE_MODEL]

GPT_4O_MINI_BASE_MODEL = "gpt-4o-mini-2024-07-18"
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
    GPT_4O_MINI_BASE_MODEL,
    "ft:Llama-3.1-8B-Instruct-SFT-20251218:depth-topic-5epochs",
    GPT_DEPTH_TOPIC_SFT_MODEL,
    "ft:Llama-3.1-8B-Instruct-DPO-20260105:depth-topic-5epochs",
]

DEPTH_GROUP_MODELS = [
    GPT_4O_MINI_BASE_MODEL,
    "Llama-3.1-8B-Instruct",
    "ft:Llama-3.1-8B-Instruct-SFT-20251218:depth-group-5epochs",
    GPT_DEPTH_GROUP_SFT_MODEL,
    "ft:Llama-3.1-8B-Instruct-DPO-20260105:depth-group-5epochs",
]

TOPIC_MODELS_V4 = [
    "Llama-3.1-8B-Instruct",
    LLAMA_V4_BREADTH_TOPIC_SFT_MODEL,
]

GROUP_MODELS_V4 = [
    "Llama-3.1-8B-Instruct",
    LLAMA_V4_BREADTH_GROUP_SFT_MODEL,
]

DEPTH_TOPIC_MODELS_V4 = [
    "Llama-3.1-8B-Instruct",
    GPT_4O_MINI_BASE_MODEL,
    LLAMA_V4_DEPTH_TOPIC_SFT_MODEL,
    GPT_DEPTH_TOPIC_SFT_MODEL,
]

DEPTH_GROUP_MODELS_V4 = [
    GPT_4O_MINI_BASE_MODEL,
    "Llama-3.1-8B-Instruct",
    LLAMA_V4_DEPTH_GROUP_SFT_MODEL,
    GPT_DEPTH_GROUP_SFT_MODEL,
]


def unique_models(*model_groups: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for group in model_groups:
        for model in group:
            if model in seen:
                continue
            seen.add(model)
            ordered.append(model)
    return ordered


TOPIC_MODELS_VS_ALL = unique_models(TOPIC_MODELS, BASE_MODELS)
GROUP_MODELS_VS_ALL = unique_models(GROUP_MODELS, BASE_MODELS)
DEPTH_TOPIC_MODELS_VS_ALL = unique_models(DEPTH_TOPIC_MODELS, BASE_MODELS)
DEPTH_GROUP_MODELS_VS_ALL = unique_models(DEPTH_GROUP_MODELS, BASE_MODELS)
TOPIC_MODELS_V4_VS_ALL = unique_models(TOPIC_MODELS_V4, V4_BASE_MODELS)
GROUP_MODELS_V4_VS_ALL = unique_models(GROUP_MODELS_V4, V4_BASE_MODELS)
DEPTH_TOPIC_MODELS_V4_VS_ALL = unique_models(DEPTH_TOPIC_MODELS_V4, V4_BASE_MODELS)
DEPTH_GROUP_MODELS_V4_VS_ALL = unique_models(DEPTH_GROUP_MODELS_V4, V4_BASE_MODELS)


@dataclass(frozen=True)
class PartitionConfig:
    name: str
    file_stem: str
    prefix_file: Path
    models: list[str]
    title: str


@dataclass(frozen=True)
class ModelSetConfig:
    topic_models: list[str]
    group_models: list[str]
    depth_topic_models: list[str]
    depth_group_models: list[str]
    report_stem: str
    title_prefix: str
    file_prefix: str


MODEL_SET_CONFIGS = {
    "post-training": ModelSetConfig(
        topic_models=TOPIC_MODELS,
        group_models=GROUP_MODELS,
        depth_topic_models=DEPTH_TOPIC_MODELS,
        depth_group_models=DEPTH_GROUP_MODELS,
        report_stem="post_training",
        title_prefix="Post-Training",
        file_prefix="",
    ),
    "base-models": ModelSetConfig(
        topic_models=BASE_MODELS,
        group_models=BASE_MODELS,
        depth_topic_models=BASE_MODELS,
        depth_group_models=BASE_MODELS,
        report_stem="base_models_split",
        title_prefix="Base Models",
        file_prefix="base_model_",
    ),
    "post-training-vs-all": ModelSetConfig(
        topic_models=TOPIC_MODELS_VS_ALL,
        group_models=GROUP_MODELS_VS_ALL,
        depth_topic_models=DEPTH_TOPIC_MODELS_VS_ALL,
        depth_group_models=DEPTH_GROUP_MODELS_VS_ALL,
        report_stem="post_training_vs_all",
        title_prefix="Post-Train Models vs All",
        file_prefix="llama_vs_all_",
    ),
    "post-training-v4": ModelSetConfig(
        topic_models=TOPIC_MODELS_V4,
        group_models=GROUP_MODELS_V4,
        depth_topic_models=DEPTH_TOPIC_MODELS_V4,
        depth_group_models=DEPTH_GROUP_MODELS_V4,
        report_stem="post_training",
        title_prefix="Post-Training V4",
        file_prefix="",
    ),
    "post-training-v4-vs-all": ModelSetConfig(
        topic_models=TOPIC_MODELS_V4_VS_ALL,
        group_models=GROUP_MODELS_V4_VS_ALL,
        depth_topic_models=DEPTH_TOPIC_MODELS_V4_VS_ALL,
        depth_group_models=DEPTH_GROUP_MODELS_V4_VS_ALL,
        report_stem="post_training_vs_all",
        title_prefix="Post-Training V4 Models vs All",
        file_prefix="models_vs_all_",
    ),
}

def add_axis_group_legend(fig: Any, anchor: tuple[float, float]) -> None:
    if plt is None:
        return
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=GROUP_TEXT_COLORS["message"], lw=2),
        Line2D([0], [0], color=GROUP_TEXT_COLORS["individual"], lw=2),
        Line2D([0], [0], color=GROUP_TEXT_COLORS["group"], lw=2),
    ]
    labels = ["Message-level metrics", "Individual-level metrics", "Group-level metrics"]
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=anchor,
        bbox_transform=fig.transFigure,
        frameon=True,
        fontsize=9,
        title="Axis Label Colors",
        title_fontsize=9,
        borderaxespad=0.2,
    )


def add_theoretical_bounds_legend(fig: Any, anchor: tuple[float, float]) -> None:
    if plt is None:
        return
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color="#7f7f7f", lw=1.5, linestyle="--"),
        Line2D([0], [0], color="#444444", lw=1.5, linestyle="-"),
    ]
    labels = ["Theoretical lower bound (0.0)", "Theoretical upper bound (1.0)"]
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=anchor,
        bbox_transform=fig.transFigure,
        frameon=True,
        fontsize=9,
        title="Reference Lines",
        title_fontsize=9,
        borderaxespad=0.2,
    )


def plot_profile_7(scores: list[dict[str, Any]], out_png: Path, title: str) -> None:
    if plt is None:
        print("matplotlib not available; skipping profile plots.")
        return
    plot_scores = [row for row in scores if str(row.get("model", "")).strip().lower() != "human"]
    if not plot_scores:
        print("No scores available; skipping profile plots.")
        return

    axis_keys = [k for k, _ in AXES_7]
    axis_labels = [label for _, label in AXES_7]
    x = list(range(len(axis_keys)))

    def _draw_and_save(target_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(12, 6.5))
        cmap = plt.cm.get_cmap("tab10", max(10, len(plot_scores)))
        footnotes = get_model_footnotes([str(row["model"]) for row in plot_scores])

        for idx, row in enumerate(plot_scores):
            model = str(row["model"])
            line_color = get_model_color(model, cmap(idx))
            line_style = PROFILE_LINE_STYLES[idx % len(PROFILE_LINE_STYLES)]
            values = [float(row.get(k, math.nan)) for k in axis_keys]
            if any(math.isnan(v) for v in values):
                continue
            ax.plot(
                x,
                values,
                linewidth=2.0,
                label=get_model_display_label(model),
                color=line_color,
                linestyle=line_style,
                marker="o",
                markersize=4.0,
                markerfacecolor="none",
                markeredgecolor=line_color,
                markeredgewidth=1.2,
            )

        ax.axhline(0.0, color="#7f7f7f", linestyle="--", linewidth=1.0, alpha=0.9, zorder=0)
        ax.axhline(1.0, color="#444444", linestyle="-", linewidth=1.0, alpha=0.9, zorder=0)
        ax.set_ylim(*PROFILE_Y_LIM)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_ylabel("Normalized Score (Higher is Better)")
        ax.set_xticks(x)
        axis_texts = ax.set_xticklabels(axis_labels, rotation=20, ha="right")
        for text_obj, axis_key in zip(axis_texts, axis_keys):
            text_obj.set_color(GROUP_TEXT_COLORS.get(AXIS_GROUP.get(axis_key, ""), "#222222"))
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_title(title)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper left",
                bbox_to_anchor=(0.80, 0.98),
                bbox_transform=fig.transFigure,
                frameon=True,
                fontsize=9,
                title="Models",
                borderaxespad=0.2,
            )
        add_theoretical_bounds_legend(fig, anchor=(0.80, 0.74))
        add_axis_group_legend(fig, anchor=(0.80, 0.52))
        add_model_footnotes(fig, footnotes, y=0.01)
        fig.tight_layout(rect=(0.0, 0.07, 0.78, 1.0))
        fig.savefig(target_path, dpi=220, bbox_inches="tight")
        fig.savefig(target_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)

    _draw_and_save(out_png)


def append_human_baseline_raw_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if any(str(row.get("model", "")).strip().lower() == "human" for row in raw_rows):
        return raw_rows
    return [
        *raw_rows,
        {
            "model": "human",
            "prefixes_ok": 0,
            "prefixes_missing": 0,
            "stance_difference": 0.0,
            "public_stance_alignment": 0.0,
            "public_stance_alignment_abs": 0.0,
            "private_stance_alignment": 0.0,
            "private_stance_alignment_abs": 0.0,
            "public_opinion_convergence_error": 0.0,
            "private_opinion_convergence_error": 0.0,
            "public_stance_change_error": 0.0,
            "public_stance_change_error_abs": 0.0,
            "private_stance_change_error": 0.0,
            "private_stance_change_error_abs": 0.0,
            "lb_stance_difference": math.nan,
            "lb_public_stance_alignment": math.nan,
            "lb_public_stance_alignment_abs": math.nan,
            "lb_private_stance_alignment": math.nan,
            "lb_private_stance_alignment_abs": math.nan,
            "lb_public_opinion_convergence_error": math.nan,
            "lb_private_opinion_convergence_error": math.nan,
            "lb_public_stance_change_error": math.nan,
            "lb_public_stance_change_error_abs": math.nan,
            "lb_private_stance_change_error": math.nan,
            "lb_private_stance_change_error_abs": math.nan,
            "stance_count": 0,
            "public_alignment_count": 0,
            "private_alignment_count": 0,
            "public_group_count": 0,
            "private_group_count": 0,
            "public_change_count": 0,
            "private_change_count": 0,
            "lb_stance_count": 0,
            "lb_public_alignment_count": 0,
            "lb_private_alignment_count": 0,
            "lb_public_group_count": 0,
            "lb_private_group_count": 0,
            "lb_public_change_count": 0,
            "lb_private_change_count": 0,
        },
    ]


def as_float_or_nan(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return math.nan


def build_score_rows_7_from_raw_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def bounded_score(
        value: float,
        *,
        lower_bound: float,
        upper_bound: float,
    ) -> float:
        if any(math.isnan(v) for v in (value, lower_bound, upper_bound)):
            return math.nan
        denom = lower_bound - upper_bound
        if math.isclose(denom, 0.0):
            if math.isclose(value, upper_bound):
                return 1.0
            return math.nan
        raw_score = (lower_bound - value) / denom
        return min(1.0, max(0.0, raw_score))

    def has_bound_field(bound_name: str) -> bool:
        return any(bound_name in row for row in raw_rows)

    def values_map(metric_name: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for row in raw_rows:
            out[str(row["model"])] = as_float_or_nan(row.get(metric_name))
        return out

    use_bound_stance = has_bound_field("lb_stance_difference")
    use_bound_public_align = has_bound_field("lb_public_stance_alignment_abs")
    use_bound_private_align = has_bound_field("lb_private_stance_alignment_abs")
    use_bound_public_conv = has_bound_field("lb_public_opinion_convergence_error")
    use_bound_private_conv = has_bound_field("lb_private_opinion_convergence_error")
    use_bound_public_change = has_bound_field("lb_public_stance_change_error_abs")
    use_bound_private_change = has_bound_field("lb_private_stance_change_error_abs")

    stance_scores = normalize_minmax(values_map("stance_difference"), higher_is_better=False)
    public_align_scores = normalize_minmax(values_map("public_stance_alignment_abs"), higher_is_better=False)
    private_align_scores = normalize_minmax(values_map("private_stance_alignment_abs"), higher_is_better=False)
    public_conv_scores = normalize_minmax(values_map("public_opinion_convergence_error"), higher_is_better=False)
    private_conv_scores = normalize_minmax(values_map("private_opinion_convergence_error"), higher_is_better=False)
    public_change_scores = normalize_minmax(values_map("public_stance_change_error_abs"), higher_is_better=False)
    private_change_scores = normalize_minmax(values_map("private_stance_change_error_abs"), higher_is_better=False)

    score_rows: list[dict[str, Any]] = []
    for row in raw_rows:
        model = str(row["model"])
        if model.strip().lower() == "human":
            score_rows.append(
                {
                    "model": model,
                    "stance_difference": 1.0,
                    "public_stance_alignment": 1.0,
                    "private_stance_alignment": 1.0,
                    "public_stance_change_error": 1.0,
                    "private_stance_change_error": 1.0,
                    "public_opinion_convergence": 1.0,
                    "private_opinion_convergence": 1.0,
                }
            )
            continue

        if use_bound_stance:
            stance_score = bounded_score(
                as_float_or_nan(row.get("stance_difference")),
                lower_bound=as_float_or_nan(row.get("lb_stance_difference")),
                upper_bound=0.0,
            )
        else:
            stance_score = stance_scores.get(model, math.nan)

        if use_bound_public_align:
            public_align_score = bounded_score(
                as_float_or_nan(row.get("public_stance_alignment_abs")),
                lower_bound=as_float_or_nan(row.get("lb_public_stance_alignment_abs")),
                upper_bound=0.0,
            )
        else:
            public_align_score = public_align_scores.get(model, math.nan)

        if use_bound_private_align:
            private_align_score = bounded_score(
                as_float_or_nan(row.get("private_stance_alignment_abs")),
                lower_bound=as_float_or_nan(row.get("lb_private_stance_alignment_abs")),
                upper_bound=0.0,
            )
        else:
            private_align_score = private_align_scores.get(model, math.nan)

        if use_bound_public_change:
            public_change_score = bounded_score(
                as_float_or_nan(row.get("public_stance_change_error_abs")),
                lower_bound=as_float_or_nan(row.get("lb_public_stance_change_error_abs")),
                upper_bound=0.0,
            )
        else:
            public_change_score = public_change_scores.get(model, math.nan)

        if use_bound_private_change:
            private_change_score = bounded_score(
                as_float_or_nan(row.get("private_stance_change_error_abs")),
                lower_bound=as_float_or_nan(row.get("lb_private_stance_change_error_abs")),
                upper_bound=0.0,
            )
        else:
            private_change_score = private_change_scores.get(model, math.nan)

        if use_bound_public_conv:
            public_conv_score = bounded_score(
                as_float_or_nan(row.get("public_opinion_convergence_error")),
                lower_bound=as_float_or_nan(row.get("lb_public_opinion_convergence_error")),
                upper_bound=0.0,
            )
        else:
            public_conv_score = public_conv_scores.get(model, math.nan)

        if use_bound_private_conv:
            private_conv_score = bounded_score(
                as_float_or_nan(row.get("private_opinion_convergence_error")),
                lower_bound=as_float_or_nan(row.get("lb_private_opinion_convergence_error")),
                upper_bound=0.0,
            )
        else:
            private_conv_score = private_conv_scores.get(model, math.nan)

        score_rows.append(
            {
                "model": model,
                "stance_difference": stance_score,
                "public_stance_alignment": public_align_score,
                "private_stance_alignment": private_align_score,
                "public_stance_change_error": public_change_score,
                "private_stance_change_error": private_change_score,
                "public_opinion_convergence": public_conv_score,
                "private_opinion_convergence": private_conv_score,
            }
        )
    return score_rows


def build_raw_rows_from_aggregates_legacy(
    aggregates: dict[str, ModelAggregate],
    models: list[str],
) -> list[dict[str, Any]]:
    raw_rows: list[dict[str, Any]] = []
    for model in models:
        agg = aggregates[model]
        stance_difference = agg.stance_sum / agg.stance_count if agg.stance_count else math.nan
        public_stance_alignment = (
            agg.public_alignment_sum / agg.public_alignment_count if agg.public_alignment_count else math.nan
        )
        private_stance_alignment = (
            agg.private_alignment_sum / agg.private_alignment_count if agg.private_alignment_count else math.nan
        )
        public_stance_alignment_abs = (
            agg.public_alignment_abs_sum / agg.public_alignment_count if agg.public_alignment_count else math.nan
        )
        private_stance_alignment_abs = (
            agg.private_alignment_abs_sum / agg.private_alignment_count if agg.private_alignment_count else math.nan
        )
        public_opinion_convergence_error = safe_mean(agg.public_conv_errors)
        private_opinion_convergence_error = safe_mean(agg.private_conv_errors)
        if agg.public_conv_h_vals and len(agg.public_conv_h_vals) == len(agg.public_conv_l_vals):
            shuffled_public_conv_l = deterministic_shuffle(
                agg.public_conv_l_vals,
                data_prefix="__aggregate__",
                model=model,
                metric_name="public_convergence",
            )
            lb_public_conv_errors = [abs(l - h) for h, l in zip(agg.public_conv_h_vals, shuffled_public_conv_l)]
        else:
            lb_public_conv_errors = agg.lb_public_conv_errors
        if agg.private_conv_h_vals and len(agg.private_conv_h_vals) == len(agg.private_conv_l_vals):
            shuffled_private_conv_l = deterministic_shuffle(
                agg.private_conv_l_vals,
                data_prefix="__aggregate__",
                model=model,
                metric_name="private_convergence",
            )
            lb_private_conv_errors = [abs(l - h) for h, l in zip(agg.private_conv_h_vals, shuffled_private_conv_l)]
        else:
            lb_private_conv_errors = agg.lb_private_conv_errors
        public_stance_change_error = safe_mean(agg.public_change_errors)
        private_stance_change_error = safe_mean(agg.private_change_errors)
        public_stance_change_error_abs = safe_mean(agg.public_change_abs_errors)
        private_stance_change_error_abs = safe_mean(agg.private_change_abs_errors)
        lb_stance_difference = agg.lb_stance_sum / agg.lb_stance_count if agg.lb_stance_count else math.nan
        lb_public_stance_alignment = (
            agg.lb_public_alignment_sum / agg.lb_public_alignment_count if agg.lb_public_alignment_count else math.nan
        )
        lb_private_stance_alignment = (
            agg.lb_private_alignment_sum / agg.lb_private_alignment_count if agg.lb_private_alignment_count else math.nan
        )
        lb_public_stance_alignment_abs = (
            agg.lb_public_alignment_abs_sum / agg.lb_public_alignment_count if agg.lb_public_alignment_count else math.nan
        )
        lb_private_stance_alignment_abs = (
            agg.lb_private_alignment_abs_sum / agg.lb_private_alignment_count if agg.lb_private_alignment_count else math.nan
        )
        lb_public_opinion_convergence_error = safe_mean(lb_public_conv_errors)
        lb_private_opinion_convergence_error = safe_mean(lb_private_conv_errors)
        lb_public_stance_change_error = safe_mean(agg.lb_public_change_errors)
        lb_private_stance_change_error = safe_mean(agg.lb_private_change_errors)
        lb_public_stance_change_error_abs = safe_mean(agg.lb_public_change_abs_errors)
        lb_private_stance_change_error_abs = safe_mean(agg.lb_private_change_abs_errors)

        raw_rows.append(
            {
                "model": model,
                "prefixes_ok": agg.prefixes_ok,
                "prefixes_missing": agg.prefixes_missing,
                "stance_difference": stance_difference,
                "public_stance_alignment": public_stance_alignment,
                "public_stance_alignment_abs": public_stance_alignment_abs,
                "private_stance_alignment": private_stance_alignment,
                "private_stance_alignment_abs": private_stance_alignment_abs,
                "public_opinion_convergence_error": public_opinion_convergence_error,
                "private_opinion_convergence_error": private_opinion_convergence_error,
                "public_stance_change_error": public_stance_change_error,
                "public_stance_change_error_abs": public_stance_change_error_abs,
                "private_stance_change_error": private_stance_change_error,
                "private_stance_change_error_abs": private_stance_change_error_abs,
                "lb_stance_difference": lb_stance_difference,
                "lb_public_stance_alignment": lb_public_stance_alignment,
                "lb_public_stance_alignment_abs": lb_public_stance_alignment_abs,
                "lb_private_stance_alignment": lb_private_stance_alignment,
                "lb_private_stance_alignment_abs": lb_private_stance_alignment_abs,
                "lb_public_opinion_convergence_error": lb_public_opinion_convergence_error,
                "lb_private_opinion_convergence_error": lb_private_opinion_convergence_error,
                "lb_public_stance_change_error": lb_public_stance_change_error,
                "lb_public_stance_change_error_abs": lb_public_stance_change_error_abs,
                "lb_private_stance_change_error": lb_private_stance_change_error,
                "lb_private_stance_change_error_abs": lb_private_stance_change_error_abs,
                "stance_count": agg.stance_count,
                "public_alignment_count": agg.public_alignment_count,
                "private_alignment_count": agg.private_alignment_count,
                "public_group_count": len(agg.public_conv_errors),
                "private_group_count": len(agg.private_conv_errors),
                "public_change_count": len(agg.public_change_errors),
                "private_change_count": len(agg.private_change_errors),
                "lb_stance_count": agg.lb_stance_count,
                "lb_public_alignment_count": agg.lb_public_alignment_count,
                "lb_private_alignment_count": agg.lb_private_alignment_count,
                "lb_public_group_count": len(lb_public_conv_errors),
                "lb_private_group_count": len(lb_private_conv_errors),
                "lb_public_change_count": len(agg.lb_public_change_errors),
                "lb_private_change_count": len(agg.lb_private_change_errors),
            }
        )
    return raw_rows


def compute_for_partition(
    *,
    partition_name: str,
    prefixes: list[str],
    models: list[str],
    metric_version: str,
    roots: list[Path],
    human_filename: str,
    llm_filename: str,
    utterance_event_types: set[str],
    likert_col: str,
    human_text_col: str,
    llm_text_col: str,
    min_key_match_rate: float,
    include_conditioned_alignment: bool,
    no_progress: bool,
    respect_validity_labels: bool,
    initial_post_sidecar_roots: list[Path],
    tweet_sidecar_roots: list[Path],
    apply_v3_additional_group_masks: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    aggregates: dict[str, ModelAggregate] = {model: ModelAggregate(model=model) for model in models}
    status_rows: list[dict[str, Any]] = []
    prefix_results_by_prefix: dict[str, dict[str, Any]] = {}

    prefix_model_pairs = [(data_prefix, model) for data_prefix in prefixes for model in models]
    pair_iter: Any = prefix_model_pairs
    if not no_progress and tqdm is not None:
        pair_iter = tqdm(
            prefix_model_pairs,
            total=len(prefix_model_pairs),
            desc=f"Computing metrics: {partition_name}",
            unit="pair",
        )
    elif not no_progress and tqdm is None:
        print("tqdm not installed; running without progress bar.")

    for data_prefix, model in pair_iter:
        result = compute_prefix_model(
            data_prefix=data_prefix,
            model=model,
            roots=roots,
            human_filename=human_filename,
            llm_filename=llm_filename,
            utterance_event_types=utterance_event_types,
            likert_col=likert_col,
            human_text_col=human_text_col,
            llm_text_col=llm_text_col,
            min_key_match_rate=min_key_match_rate,
            include_conditioned_alignment=include_conditioned_alignment,
            respect_validity_labels=respect_validity_labels,
            initial_post_sidecar_roots=initial_post_sidecar_roots,
            tweet_sidecar_roots=tweet_sidecar_roots,
            apply_v3_additional_group_masks=apply_v3_additional_group_masks,
        )

        status_rows.append(
            {
                "partition": partition_name,
                "data_prefix": data_prefix,
                "model": model,
                "status": result.status,
                "message": result.message,
                "human_path": result.human_path,
                "llm_path": result.llm_path,
                "alignment_mode": result.alignment_mode,
                "key_match_rate": f"{result.key_match_rate:.6f}" if result.status == "ok" else "",
                "aligned_pairs": result.aligned_pairs if result.status == "ok" else "",
            }
        )

        merge_prefix_result_into_aggregate(aggregates[model], result)
        prefix_results_by_prefix.setdefault(data_prefix, {})[model] = result

    ub_mode = "gaussian_likert" if metric_version == "v2" else "jitter"
    raw_rows = build_raw_rows_from_aggregates(aggregates, models, ub_mode=ub_mode)
    if metric_version == "v2":
        raw_rows = add_v2_bootstrap_cis_to_raw_rows(
            raw_rows,
            models=models,
            prefixes=prefixes,
            prefix_results_by_prefix=prefix_results_by_prefix,
        )
    score_rows = build_score_rows_from_raw_rows(raw_rows, metric_version="v0")
    return raw_rows, score_rows, status_rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build breadth/depth post-training profile plots for group/topic train/test.")
    parser.add_argument(
        "--data-root",
        default="../../result/eval_v3/human_llm",
        help="Root of human_llm results (default: result/eval_v3/human_llm).",
    )
    parser.add_argument(
        "--fallback-data-root",
        action="append",
        default=[],
        help="Additional data root(s) to search if files are missing.",
    )
    parser.add_argument(
        "--topic-train-prefix-file",
        default="../../inputs/masks_train_test/golden_group_v0/breadth/breadth_train_topic_mask_golden_v0.txt",
        help="Breadth topic-partition train prefix file.",
    )
    parser.add_argument(
        "--topic-test-prefix-file",
        default="../../inputs/masks_train_test/golden_group_v0/breadth/breadth_test_topic_mask_golden_v0.txt",
        help="Breadth topic-partition test prefix file.",
    )
    parser.add_argument(
        "--group-train-prefix-file",
        default="../../inputs/masks_train_test/golden_group_v0/breadth/breadth_train_group_mask_golden_v0.txt",
        help="Breadth group-partition train prefix file.",
    )
    parser.add_argument(
        "--group-test-prefix-file",
        default="../../inputs/masks_train_test/golden_group_v0/breadth/breadth_test_group_mask_golden_v0.txt",
        help="Breadth group-partition test prefix file.",
    )
    parser.add_argument(
        "--depth-topic-train-prefix-file",
        default="../../inputs/masks_train_test/golden_group_v0/depth/depth_train_topic_mask_golden_v0.txt",
        help="Depth topic-partition train prefix file.",
    )
    parser.add_argument(
        "--depth-topic-test-prefix-file",
        default="../../inputs/masks_train_test/golden_group_v0/depth/depth_test_topic_mask_golden_v0.txt",
        help="Depth topic-partition test prefix file.",
    )
    parser.add_argument(
        "--depth-group-train-prefix-file",
        default="../../inputs/masks_train_test/golden_group_v0/depth/depth_train_group_mask_golden_v0.txt",
        help="Depth group-partition train prefix file.",
    )
    parser.add_argument(
        "--depth-group-test-prefix-file",
        default="../../inputs/masks_train_test/golden_group_v0/depth/depth_test_group_mask_golden_v0.txt",
        help="Depth group-partition test prefix file.",
    )
    parser.add_argument(
        "--human-filename",
        default=HUMAN_DEFAULT_FILENAME,
        help=f"Human CSV filename (default: {HUMAN_DEFAULT_FILENAME}).",
    )
    parser.add_argument(
        "--llm-filename",
        default=LLM_DEFAULT_FILENAME,
        help=f"LLM CSV filename (default: {LLM_DEFAULT_FILENAME}).",
    )
    parser.add_argument(
        "--output-dir",
        default="result/radar/summary_post_training",
        help="Output directory for report assets (default: result/radar/summary_post_training). CSVs are written under output_dir/csv and figures under output_dir/img.",
    )
    parser.add_argument(
        "--likert-col",
        default="likert_pred",
        help="Column for stance labels/scores.",
    )
    parser.add_argument(
        "--human-text-col",
        default="text",
        help="Column for human text.",
    )
    parser.add_argument(
        "--llm-text-col",
        default="llm_text",
        help="Column for LLM text.",
    )
    parser.add_argument(
        "--utterance-event-types",
        default=",".join(DEFAULT_UTTERANCE_EVENT_TYPES),
        help="Comma-separated event types for utterance-level metrics.",
    )
    parser.add_argument(
        "--min-key-match-rate",
        type=float,
        default=0.7,
        help="Minimum key-based row match rate before falling back to index alignment.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar.",
    )
    parser.add_argument(
        "--metric-version",
        choices=("v0", "v0_fixed", "v1", "v2", "spearman", "pearson"),
        default="v0",
        help="Scoring method: v0=original min-max normalization with conditioned alignment included, v0_fixed=min-max normalization with conditioned alignment excluded, v1=Kendall tau-b, v2=v0_fixed-style bounded normalization restricted to the 6 v1-family metrics with permutation/noisy reference lines, spearman=Spearman rho, pearson=Pearson r.",
    )
    parser.add_argument(
        "--model-set",
        choices=tuple(MODEL_SET_CONFIGS.keys()),
        default="post-training",
        help="Model family to score on the breadth split files.",
    )
    parser.add_argument(
        "--respect-validity-labels",
        action="store_true",
        help="Filter invalid rows by blanking their Likert labels. Sidecar overrides are used when available; otherwise original CSV validity columns are used.",
    )
    parser.add_argument(
        "--initial-post-validity-sidecar-root",
        action="append",
        default=[],
        help="Optional sidecar root(s) with __initial_post_rejudge.csv files.",
    )
    parser.add_argument(
        "--tweet-validity-sidecar-root",
        action="append",
        default=[],
        help="Optional sidecar root(s) with __tweet_rejudge.csv files.",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Optional filename suffix override such as _v3. Defaults to the metric-version suffix.",
    )
    parser.add_argument(
        "--skip-v3-additional-group-masks",
        action="store_true",
        help="Do not apply the extra v3 tweet/opinion group masks from masks/groups_with_valid_tweets or masks/groups_with_valid_opinions.",
    )
    return parser


def maybe_adjust_bar_model_order(
    model_order: list[str] | None,
    *,
    partition_name: str,
    model_set: str,
) -> list[str] | None:
    if model_order is None:
        return None
    if model_set not in {"post-training-vs-all", "post-training-v4-vs-all"} or not partition_name.startswith("depth_"):
        return model_order

    adjusted_order = [*model_order]
    tulu_model = "Llama-3.1-Tulu-3-8B-SFT"
    llama_instruct_model = "Llama-3.1-8B-Instruct"
    if tulu_model not in adjusted_order or llama_instruct_model not in adjusted_order:
        return adjusted_order

    adjusted_order.remove(tulu_model)
    insert_at = adjusted_order.index(llama_instruct_model)
    adjusted_order.insert(insert_at, tulu_model)
    return adjusted_order


def main() -> int:
    args = build_arg_parser().parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = output_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    img_dir = output_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

    roots = [Path(args.data_root), *[Path(p) for p in args.fallback_data_root]]
    initial_post_sidecar_roots = [Path(p) for p in args.initial_post_validity_sidecar_root]
    tweet_sidecar_roots = [Path(p) for p in args.tweet_validity_sidecar_root]
    utterance_event_types = {t.strip() for t in args.utterance_event_types.split(",") if t.strip()}
    model_set_cfg = MODEL_SET_CONFIGS[args.model_set]

    configs = [
        PartitionConfig(
            name="breadth_topic_train",
            file_stem=f"{model_set_cfg.file_prefix}breadth_topic_train",
            prefix_file=Path(args.topic_train_prefix_file),
            models=model_set_cfg.topic_models,
            title=f"{model_set_cfg.title_prefix} - Breadth Topic Partition - Train",
        ),
        PartitionConfig(
            name="breadth_topic_test",
            file_stem=f"{model_set_cfg.file_prefix}breadth_topic_test",
            prefix_file=Path(args.topic_test_prefix_file),
            models=model_set_cfg.topic_models,
            title=f"{model_set_cfg.title_prefix} - Breadth Topic Partition - Test",
        ),
        PartitionConfig(
            name="breadth_group_train",
            file_stem=f"{model_set_cfg.file_prefix}breadth_group_train",
            prefix_file=Path(args.group_train_prefix_file),
            models=model_set_cfg.group_models,
            title=f"{model_set_cfg.title_prefix} - Breadth Group Partition - Train",
        ),
        PartitionConfig(
            name="breadth_group_test",
            file_stem=f"{model_set_cfg.file_prefix}breadth_group_test",
            prefix_file=Path(args.group_test_prefix_file),
            models=model_set_cfg.group_models,
            title=f"{model_set_cfg.title_prefix} - Breadth Group Partition - Test",
        ),
        PartitionConfig(
            name="depth_topic_train",
            file_stem=f"{model_set_cfg.file_prefix}depth_topic_train",
            prefix_file=Path(args.depth_topic_train_prefix_file),
            models=model_set_cfg.depth_topic_models,
            title=f"{model_set_cfg.title_prefix} - Depth Topic Partition - Train",
        ),
        PartitionConfig(
            name="depth_topic_test",
            file_stem=f"{model_set_cfg.file_prefix}depth_topic_test",
            prefix_file=Path(args.depth_topic_test_prefix_file),
            models=model_set_cfg.depth_topic_models,
            title=f"{model_set_cfg.title_prefix} - Depth Topic Partition - Test",
        ),
        PartitionConfig(
            name="depth_group_train",
            file_stem=f"{model_set_cfg.file_prefix}depth_group_train",
            prefix_file=Path(args.depth_group_train_prefix_file),
            models=model_set_cfg.depth_group_models,
            title=f"{model_set_cfg.title_prefix} - Depth Group Partition - Train",
        ),
        PartitionConfig(
            name="depth_group_test",
            file_stem=f"{model_set_cfg.file_prefix}depth_group_test",
            prefix_file=Path(args.depth_group_test_prefix_file),
            models=model_set_cfg.depth_group_models,
            title=f"{model_set_cfg.title_prefix} - Depth Group Partition - Test",
        ),
    ]

    score_axes = RADAR_AXES_V1 if args.metric_version in ("v1", "v2", "spearman", "pearson") else RADAR_AXES
    score_fieldnames = ["model"] + [k for k, _ in score_axes]
    plot_y_lim = (0.0, 1.0) if args.metric_version == "v2" else ((-0.05, 1.05) if args.metric_version in ("v1", "spearman", "pearson") else PROFILE_Y_LIM)
    plot_rank_corr_label = {"v1": "Kendall τ-b", "spearman": "Spearman ρ", "pearson": "Pearson r"}.get(args.metric_version, "Kendall τ-b")
    plot_suffix = args.output_suffix or {"v1": "_kendall", "v2": "_v2", "spearman": "_spearman", "pearson": "_pearson"}.get(args.metric_version, "")
    all_status_rows: list[dict[str, Any]] = []

    for cfg in configs:
        if not cfg.prefix_file.exists():
            print(f"Prefix file not found: {cfg.prefix_file}")
            return 1

        prefixes = read_lines(cfg.prefix_file)
        raw_rows, score_rows, status_rows = compute_for_partition(
            partition_name=cfg.name,
            prefixes=prefixes,
            models=cfg.models,
            metric_version=args.metric_version,
            roots=roots,
            human_filename=args.human_filename,
            llm_filename=args.llm_filename,
            utterance_event_types=utterance_event_types,
            likert_col=args.likert_col,
            human_text_col=args.human_text_col,
            llm_text_col=args.llm_text_col,
            min_key_match_rate=args.min_key_match_rate,
            include_conditioned_alignment=args.metric_version == "v0",
            no_progress=args.no_progress,
            respect_validity_labels=args.respect_validity_labels,
            initial_post_sidecar_roots=initial_post_sidecar_roots,
            tweet_sidecar_roots=tweet_sidecar_roots,
            apply_v3_additional_group_masks=not args.skip_v3_additional_group_masks,
        )

        raw_rows = append_human_baseline_raw_rows(raw_rows)
        score_rows = build_score_rows_from_raw_rows(raw_rows, metric_version=args.metric_version)
        score_rows = append_human_baseline_score_rows(score_rows, axes=score_axes)
        if args.metric_version in ("v1", "v2", "spearman", "pearson"):
            include_noisy_ub = not (args.metric_version == "v1" and args.no_jitter_ub)
            score_rows = append_reference_score_rows_v1(
                score_rows,
                raw_rows,
                metric_version=args.metric_version,
                include_noisy_ub=include_noisy_ub,
            )
        score_rows = [
            {"model": str(row.get("model", "")), **{key: row.get(key, math.nan) for key, _ in score_axes}}
            for row in score_rows
        ]

        raw_fieldnames = collect_fieldnames(raw_rows)
        display_model_order = get_v2_display_model_order(raw_rows) if args.metric_version == "v2" else None
        bar_model_order = maybe_adjust_bar_model_order(
            display_model_order,
            partition_name=cfg.name,
            model_set=args.model_set,
        ) if args.metric_version == "v2" else None

        raw_path = csv_dir / f"raw_metrics_{cfg.file_stem}{plot_suffix}.csv"
        score_path = csv_dir / f"profile_scores_{cfg.file_stem}{plot_suffix}.csv"
        plot_path = img_dir / f"profile_{cfg.file_stem}{plot_suffix}.png"
        generate_profile_plots = plot_suffix != "_v3"

        save_csv(raw_path, raw_rows, raw_fieldnames)
        save_csv(score_path, score_rows, score_fieldnames)
        if generate_profile_plots and has_plottable_scores(score_rows, axes=score_axes):
            plot_title = cfg.title
            if args.metric_version == "v1":
                plot_title = f"{cfg.title} (Kendall τ-b)"
                plot_v2_bar_metrics(
                    raw_rows,
                    out_png=bar_path,
                    title=f"{cfg.title} (v2 Bar Plot)",
                    model_order=bar_model_order,
                )
                plot_title = f"{cfg.title} (Spearman ρ)"
            elif args.metric_version == "pearson":
                plot_title = f"{cfg.title} (Pearson r)"
            elif args.metric_version == "v2":
                plot_title = f"{cfg.title} (v2)"
            plot_profile(
                score_rows,
                out_png=plot_path,
                title=plot_title,
                axes=score_axes,
                y_lim=plot_y_lim,
                rank_corr_label=plot_rank_corr_label,
                model_order=display_model_order,
            )
        elif generate_profile_plots:
            print(f"No plottable scores for {cfg.name}; skipping plot.")

        if args.metric_version == "v2":
            bar_path = img_dir / f"bar_{cfg.file_stem}{plot_suffix}.png"
            plot_v2_bar_metrics(
                raw_rows,
                out_png=bar_path,
                title=f"{cfg.title} (v2 Bar Plot)",
                model_order=bar_model_order,
            )
            print(f"Wrote: {bar_path}")

        all_status_rows.extend(status_rows)

        print(f"Wrote: {raw_path}")
        print(f"Wrote: {score_path}")
        if plt is not None and generate_profile_plots:
            print(f"Wrote: {plot_path}")
        elif plot_suffix == "_v3":
            print(f"Skipped v3 profile plot for {cfg.name}; only bar outputs remain supported.")
        else:
            print("Skipped plotting because matplotlib is unavailable.")

    status_fieldnames = [
        "partition",
        "data_prefix",
        "model",
        "status",
        "message",
        "human_path",
        "llm_path",
        "alignment_mode",
        "key_match_rate",
        "aligned_pairs",
    ]
    status_path = csv_dir / f"per_prefix_status_{model_set_cfg.report_stem}{plot_suffix}.csv"
    save_csv(status_path, all_status_rows, status_fieldnames)
    print(f"Wrote: {status_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
