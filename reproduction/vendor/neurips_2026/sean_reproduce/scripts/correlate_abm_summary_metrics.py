#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import re
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

try:
    from adjustText import adjust_text  # type: ignore[reportMissingImports]
except Exception:  # pragma: no cover
    adjust_text = None


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
RADAR_DIR = REPO_ROOT / "neurips_2026" / "chengtao" / "src" / "radar"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(RADAR_DIR) not in sys.path:
    sys.path.insert(0, str(RADAR_DIR))

from neurips_2026.sean_reproduce.scripts.create_v4_vs_v4b_summary_base_model_report import (  # noqa: E402
    MODEL_COLOR_MAP,
    get_model_color,
)

try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import matplotlib.patheffects as patheffects
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"matplotlib is required: {exc}")


DEFAULT_INPUT = (
    REPO_ROOT
    / "neurips_2026"
    / "sean_reproduce"
    / "v4_pub"
    / "abm"
    / "summary_post_training"
    / "report_bar_posttrain_models_vs_all_degroot_breadth_group_train_v4_vs_4b_unfiltered_msg_and_ind_pub.csv"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT.parent / "analysis"

EXCLUDED_MODELS = {"human", "Uniform LB", "Mean Predictor"}
METRIC_SPECS = {
    "public_message_error": {
        "label": "Single-Opinion Prediction Error",
        "columns": ["public_stance_alignment_abs"],
    },
    "public_update_error": {
        "label": "Individual Opinion-Update Prediction Error",
        "columns": ["public_stance_change_error_abs"],
    },
    "public_group_error": {
        "label": "Group Opinion Convergence Prediction Error",
        "columns": ["public_opinion_convergence_error"],
    },
    "private_message_error": {
        "label": "Single-Opinion Prediction Error",
        "columns": ["private_stance_alignment_abs"],
    },
    "private_update_error": {
        "label": "Individual Opinion-Update Prediction Error",
        "columns": ["private_stance_change_error_abs"],
    },
    "private_group_error": {
        "label": "Group Opinion Convergence Prediction Error",
        "columns": ["private_opinion_convergence_error"],
    },
}
PANEL_SPECS = [
    {
        "title": "Public: Single-Opinion vs Opinion-Update",
        "left_metric": "public_message_error",
        "right_metric": "public_update_error",
    },
    {
        "title": "Public: Single-Opinion vs Group Convergence",
        "left_metric": "public_message_error",
        "right_metric": "public_group_error",
    },
    {
        "title": "Private: Single-Opinion vs Opinion-Update",
        "left_metric": "private_message_error",
        "right_metric": "private_update_error",
    },
    {
        "title": "Private: Single-Opinion vs Group Convergence",
        "left_metric": "private_message_error",
        "right_metric": "private_group_error",
    },
]
VERSION_STYLE = {
    "v4": {
        "color": "#1f77b4",
        "marker": "o",
        "label": "Full Dynamics Simulation",
        "output_stem": "full_dynamics_simulation",
    },
    "v4_b": {
        "color": "#d62728",
        "marker": "s",
        "label": "Next Message Prediction",
        "output_stem": "next_message_prediction",
    },
}

BASE_MODEL_NAME = "Llama-3.1-8B-Instruct"
SFT_MODEL_PREFIX = "ft:Llama-3.1-8B-Instruct-SFT-"
HIGHLIGHT_STYLE = {
    BASE_MODEL_NAME: {"marker": "o", "size": 130, "label": "Llama-3.1-8B"},
    "Llama SFT": {"marker": "D", "size": 150, "label": "Llama SFT"},
}
OTHER_POINT_STYLE = {"color": "#b0b0b0", "marker": "o", "size": 64}

LABEL_OFFSET_OVERRIDES = {
    ("v4", "pub", "public_message_error", "public_update_error", "Llama SFT"): (16, 8),
    ("v4", "pub", "public_message_error", "public_update_error", "gpt-5-mini"): (18, -16),
    ("v4", "pub", "public_message_error", "public_group_error", "Llama SFT"): (16, 8),
    ("v4", "pub", "public_message_error", "public_group_error", "gpt-5-mini"): (18, 2),
}

SHORT_LABEL_MAP = {
    "gpt-5-mini": "gpt-5-mini",
    "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
    "gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
    "Mistral-7B-Instruct-v0.3": "Mistral-7B",
    "Qwen2.5-32B-Instruct": "Qwen2.5-32B",
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "Llama-3.1-70B-Instruct": "Llama-3.1-70B",
    "DeGroot fit": "DeGroot fit",
}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Correlate the three ABM summary metrics in a v4/v4_b publication CSV.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    degroot_group = parser.add_mutually_exclusive_group()
    degroot_group.add_argument(
        "--include-degroot",
        dest="include_degroot",
        action="store_true",
        help="Keep the shared DeGroot fit baseline in the correlation scopes and scatter figure.",
    )
    degroot_group.add_argument(
        "--exclude-degroot",
        dest="include_degroot",
        action="store_false",
        help="Drop the shared DeGroot fit baseline from the correlation scopes and scatter figure.",
    )
    parser.set_defaults(include_degroot=False)
    return parser


def _format_model_display_label(model_name: str) -> str:
    normalized = str(model_name).strip()
    if normalized in SHORT_LABEL_MAP:
        return SHORT_LABEL_MAP[normalized]
    if normalized.startswith("ft:Llama-3.1-8B-Instruct-SFT-"):
        return "Llama SFT"
    if ":" in normalized:
        return normalized.split(":")[-1]
    return normalized


def _is_llama_sft(model_name: str) -> bool:
    return str(model_name).strip().startswith(SFT_MODEL_PREFIX)


def _highlight_key(model_name: str) -> str | None:
    normalized = str(model_name).strip()
    if normalized == BASE_MODEL_NAME:
        return BASE_MODEL_NAME
    if _is_llama_sft(normalized):
        return "Llama SFT"
    return None


def _model_point_color(model_name: str) -> str:
    normalized = str(model_name).strip()
    return get_model_color(normalized, MODEL_COLOR_MAP.get(normalized, "#999999"))


def _darken_hex_color(hex_color: str, factor: float = 0.78) -> str:
    normalized = str(hex_color).strip()
    if not normalized.startswith("#") or len(normalized) != 7:
        return normalized
    red = int(normalized[1:3], 16)
    green = int(normalized[3:5], 16)
    blue = int(normalized[5:7], 16)
    red = max(0, min(255, int(round(red * factor))))
    green = max(0, min(255, int(round(green * factor))))
    blue = max(0, min(255, int(round(blue * factor))))
    return f"#{red:02x}{green:02x}{blue:02x}"


def _label_text_color(model_name: str) -> str:
    return _darken_hex_color(_model_point_color(model_name), factor=0.72)


def _apply_label_effects(annotation: object) -> None:
    annotation.set_path_effects(
        [
            patheffects.Stroke(linewidth=1.8, foreground="white", alpha=0.95),
            patheffects.Normal(),
        ]
    )


def _format_pearson_r(value: float) -> str:
    rounded_two = round(value, 2)
    if rounded_two == 1.0 and not math.isclose(value, 1.0):
        return f"{value:.3f}"
    return f"{value:.2f}"


def _prepare_rows(input_csv: Path, *, include_degroot: bool) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    rows = df[~df["model"].isin(EXCLUDED_MODELS)].copy()
    if not include_degroot:
        rows = rows[rows["model"] != "DeGroot fit"].copy()

    for metric_name, metric_spec in METRIC_SPECS.items():
        rows[metric_name] = rows[metric_spec["columns"]].mean(axis=1)

    rows["display_label"] = rows["model"].map(_format_model_display_label)
    rows["highlight_key"] = rows["model"].map(_highlight_key)
    rows["point_label"] = rows["display_label"]
    return rows


def _output_base_name(input_csv: Path, *, include_degroot: bool) -> str:
    match = re.search(r"(breadth|depth)_(group|topic)_(train|test)", input_csv.stem)
    if match:
        partition_stem = "_".join(match.groups())
    else:
        partition_stem = input_csv.stem.replace("report_bar_", "")
    base_name = f"posttrain_{partition_stem}_public_private_1x4_correlation"
    if include_degroot:
        return f"{base_name}_with_degroot"
    return base_name


def _correlation_rows(rows: pd.DataFrame) -> list[dict[str, object]]:
    scopes = {
        "combined": rows,
        "v4": rows[rows["comparison_version"] == "v4"].copy(),
        "v4_b": rows[rows["comparison_version"] == "v4_b"].copy(),
    }
    summary: list[dict[str, object]] = []
    for scope_name, scope_df in scopes.items():
        for panel_spec in PANEL_SPECS:
            left_metric = panel_spec["left_metric"]
            right_metric = panel_spec["right_metric"]
            clean = scope_df[[left_metric, right_metric]].dropna()
            if len(clean) < 3:
                continue
            pearson_result = pearsonr(clean[left_metric], clean[right_metric])
            spearman_result = spearmanr(clean[left_metric], clean[right_metric])
            summary.append(
                {
                    "scope": scope_name,
                    "n": len(clean),
                    "left_metric": left_metric,
                    "right_metric": right_metric,
                    "panel_title": panel_spec["title"],
                    "left_label": METRIC_SPECS[left_metric]["label"],
                    "right_label": METRIC_SPECS[right_metric]["label"],
                    "pearson_r": pearson_result.statistic,
                    "pearson_p": pearson_result.pvalue,
                    "spearman_rho": spearman_result.statistic,
                    "spearman_p": spearman_result.pvalue,
                }
            )
    return summary


def _add_best_fit(ax: object, x_values: pd.Series, y_values: pd.Series, *, color: str) -> None:
    if len(x_values) < 2:
        return
    slope, intercept = np.polyfit(x_values, y_values, 1)
    x_min = float(x_values.min())
    x_max = float(x_values.max())
    x_line = [x_min, x_max]
    y_line = [slope * x_min + intercept, slope * x_max + intercept]
    ax.plot(x_line, y_line, color=color, linewidth=1.5, alpha=0.55, zorder=1)


def _raw_full_name_footnote(version_rows: pd.DataFrame) -> str:
    entries: list[str] = []
    seen: set[str] = set()
    for _, row in version_rows.iterrows():
        display_label = str(row["display_label"])
        full_name = str(row["model"])
        if display_label in seen:
            continue
        seen.add(display_label)
        entries.append(f"{display_label} = {full_name}")
    return " | ".join(entries)


def _set_annotation_alignment(annotation: object, dx: float, dy: float) -> None:
    horizontal_alignment = "left" if dx >= 0 else "right"
    vertical_alignment = "bottom" if dy >= 0 else "top"
    annotation.set_ha(horizontal_alignment)
    annotation.set_va(vertical_alignment)


def _offset_data_position(
    ax: object,
    point_xy: tuple[float, float],
    dx_points: float,
    dy_points: float,
) -> tuple[float, float]:
    display_xy = ax.transData.transform(point_xy)
    point_scale = ax.figure.dpi / 72.0
    offset_display_xy = (
        display_xy[0] + dx_points * point_scale,
        display_xy[1] + dy_points * point_scale,
    )
    return tuple(ax.transData.inverted().transform(offset_display_xy))


def _bbox_overlaps(bbox: object, other_bbox: object, *, padding: float = 0.0) -> bool:
    return not (
        bbox.x1 + padding < other_bbox.x0
        or bbox.x0 - padding > other_bbox.x1
        or bbox.y1 + padding < other_bbox.y0
        or bbox.y0 - padding > other_bbox.y1
    )


def _point_bbox(point_xy: tuple[float, float], *, radius: float = 12.0) -> tuple[float, float, float, float]:
    x_value, y_value = point_xy
    return (x_value - radius, y_value - radius, x_value + radius, y_value + radius)


def _crowding_score(point_xy: tuple[float, float], all_points: list[tuple[float, float]]) -> int:
    x_value, y_value = point_xy
    score = 0
    for other_x, other_y in all_points:
        if other_x == x_value and other_y == y_value:
            continue
        if math.hypot(other_x - x_value, other_y - y_value) <= 90.0:
            score += 1
    return score


def _annotation_inside_axes(annotation_bbox: object, axes_bbox: object, *, margin: float = 2.0) -> bool:
    return (
        annotation_bbox.x0 >= axes_bbox.x0 + margin
        and annotation_bbox.x1 <= axes_bbox.x1 - margin
        and annotation_bbox.y0 >= axes_bbox.y0 + margin
        and annotation_bbox.y1 <= axes_bbox.y1 - margin
    )


def _label_penalty(
    annotation_bbox: object,
    point_boxes: list[tuple[float, float, float, float]],
    occupied_bboxes: list[object],
    axes_bbox: object,
) -> float:
    penalty = 0.0
    if not _annotation_inside_axes(annotation_bbox, axes_bbox):
        penalty += 1_000.0
    for point_box in point_boxes:
        point_left, point_top, point_right, point_bottom = point_box[0], point_box[1], point_box[2], point_box[3]
        if not (
            annotation_bbox.x1 < point_left
            or annotation_bbox.x0 > point_right
            or annotation_bbox.y1 < point_top
            or annotation_bbox.y0 > point_bottom
        ):
            penalty += 200.0
    for occupied_bbox in occupied_bboxes:
        if _bbox_overlaps(annotation_bbox, occupied_bbox, padding=8.0):
            penalty += 500.0
    return penalty


def _manual_label_offset(
    *,
    version_key: str,
    publication_variant: str,
    left_metric: str,
    right_metric: str,
    model_name: str,
) -> tuple[int, int] | None:
    return LABEL_OFFSET_OVERRIDES.get(
        (version_key, publication_variant, left_metric, right_metric, model_name)
    )


def _reposition_axis_annotations(
    ax: object,
    annotations: list[tuple[object, tuple[float, float], str]],
    *,
    version_key: str,
    publication_variant: str,
    left_metric: str,
    right_metric: str,
) -> None:
    if not annotations:
        return
    figure = ax.figure

    for annotation, _, model_name in annotations:
        manual_offset = _manual_label_offset(
            version_key=version_key,
            publication_variant=publication_variant,
            left_metric=left_metric,
            right_metric=right_metric,
            model_name=model_name,
        )
        if manual_offset is not None:
            _set_annotation_alignment(annotation, manual_offset[0], manual_offset[1])
            annotation.set_position(
                _offset_data_position(ax, annotation._label_target_xy, manual_offset[0], manual_offset[1])
            )

    if adjust_text is not None:
        texts = [annotation for annotation, _, _ in annotations]
        target_x = [point_xy[0] for _, point_xy, _ in annotations]
        target_y = [point_xy[1] for _, point_xy, _ in annotations]
        adjust_text(
            texts,
            x=target_x,
            y=target_y,
            target_x=target_x,
            target_y=target_y,
            ax=ax,
            avoid_self=True,
            ensure_inside_axes=True,
            expand_axes=False,
            expand=(1.08, 1.16),
            force_text=(0.28, 0.42),
            force_static=(0.18, 0.28),
            force_pull=(0.03, 0.04),
            force_explode=(0.18, 0.32),
            max_move=(32, 32),
            only_move={"text": "xy", "static": "xy", "explode": "xy", "pull": "xy"},
            min_arrow_len=18,
            iter_lim=300,
            arrowprops={
                "arrowstyle": "-",
                "color": "#7a7a7a",
                "lw": 0.55,
                "alpha": 0.65,
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )
        return
    return


def _render_version_scatter_panels(
    rows: pd.DataFrame,
    summary_rows: list[dict[str, object]],
    *,
    version_key: str,
    publication_variant: str,
    output_path: Path,
) -> list[Path]:
    version_rows = rows[rows["comparison_version"] == version_key].copy()
    include_footnote = publication_variant == "raw"
    include_grid = publication_variant == "raw"
    topic_pub_scale = 1.08 if publication_variant == "pub" and "_topic_" in output_path.stem else 1.0
    point_label_fontsize = 10 * topic_pub_scale
    panel_title_fontsize = 14 * topic_pub_scale
    axis_label_fontsize = 13 * topic_pub_scale
    tick_label_fontsize = 11 * topic_pub_scale
    raw_stat_fontsize = 10.5 * topic_pub_scale
    pub_stat_fontsize = 13 * topic_pub_scale
    fig, axes = plt.subplots(1, 4, figsize=(23.5 * topic_pub_scale, 6.2 * topic_pub_scale))
    fig.subplots_adjust(
        left=0.055,
        right=0.992,
        top=0.83 if include_footnote else 0.9,
        bottom=0.25 if include_footnote else 0.14,
        wspace=0.28,
    )
    summary_lookup = {(row["scope"], row["left_metric"], row["right_metric"]): row for row in summary_rows}
    version_style = VERSION_STYLE[version_key]

    for ax, panel_spec in zip(axes, PANEL_SPECS):
        left_metric = panel_spec["left_metric"]
        right_metric = panel_spec["right_metric"]
        version_summary = summary_lookup[(version_key, left_metric, right_metric)]
        _add_best_fit(ax, version_rows[left_metric], version_rows[right_metric], color="#4a4a4a")
        axis_annotations: list[tuple[object, tuple[float, float], str]] = []

        other_rows = version_rows[version_rows["highlight_key"].isna()].copy()
        if not other_rows.empty:
            for _, row in other_rows.iterrows():
                point_color = _model_point_color(row["model"])
                ax.scatter(
                    [row[left_metric]],
                    [row[right_metric]],
                    color=point_color,
                    marker=OTHER_POINT_STYLE["marker"],
                    s=OTHER_POINT_STYLE["size"],
                    alpha=0.9,
                    edgecolors="white",
                    linewidths=0.6,
                    zorder=2,
                )
                label_dx, label_dy = 5, 5
                annotation = ax.text(
                    *_offset_data_position(
                        ax,
                        (float(row[left_metric]), float(row[right_metric])),
                        label_dx,
                        label_dy,
                    ),
                    row["point_label"],
                    fontsize=point_label_fontsize,
                    fontweight="medium",
                    color=_label_text_color(row["model"]),
                    zorder=4,
                )
                annotation._label_target_xy = (float(row[left_metric]), float(row[right_metric]))
                _set_annotation_alignment(annotation, label_dx, label_dy)
                _apply_label_effects(annotation)
                axis_annotations.append(
                    (
                        annotation,
                        (float(row[left_metric]), float(row[right_metric])),
                        str(row["point_label"]),
                    )
                )

        for highlight_name in (BASE_MODEL_NAME, "Llama SFT"):
            highlight_rows = version_rows[version_rows["highlight_key"] == highlight_name]
            if highlight_rows.empty:
                continue
            highlight_style = HIGHLIGHT_STYLE[highlight_name]
            ax.scatter(
                highlight_rows[left_metric],
                highlight_rows[right_metric],
                color=_model_point_color(highlight_rows.iloc[0]["model"]),
                marker=highlight_style["marker"],
                s=highlight_style["size"],
                alpha=0.98,
                edgecolors="#111111",
                linewidths=1.0,
                zorder=3,
            )
            for _, row in highlight_rows.iterrows():
                label_dx, label_dy = 7, 7
                annotation = ax.text(
                    *_offset_data_position(
                        ax,
                        (float(row[left_metric]), float(row[right_metric])),
                        label_dx,
                        label_dy,
                    ),
                    row["point_label"],
                    fontsize=point_label_fontsize,
                    fontweight="bold",
                    color=_label_text_color(row["model"]),
                    zorder=4,
                )
                annotation._label_target_xy = (float(row[left_metric]), float(row[right_metric]))
                _set_annotation_alignment(annotation, label_dx, label_dy)
                _apply_label_effects(annotation)
                axis_annotations.append(
                    (
                        annotation,
                        (float(row[left_metric]), float(row[right_metric])),
                        str(row["point_label"]),
                    )
                )

        ax.set_title(panel_spec["title"], fontsize=panel_title_fontsize, fontweight="semibold", pad=10 * topic_pub_scale)
        ax.set_xlabel(METRIC_SPECS[left_metric]["label"])
        ax.set_ylabel(METRIC_SPECS[right_metric]["label"])
        ax.tick_params(axis="both", labelsize=tick_label_fontsize)
        ax.xaxis.label.set_size(axis_label_fontsize)
        ax.yaxis.label.set_size(axis_label_fontsize)
        if include_grid:
            ax.grid(True, alpha=0.25)
        else:
            ax.grid(False)
        if include_footnote:
            ax.text(
                0.03,
                0.97,
                f"r={_format_pearson_r(version_summary['pearson_r'])} | rho={version_summary['spearman_rho']:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=raw_stat_fontsize,
                color="#444444",
            )
        else:
            ax.text(
                0.97,
                0.04,
                rf"$r = {_format_pearson_r(version_summary['pearson_r'])}$",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=pub_stat_fontsize,
                color="#444444",
                bbox={
                    "boxstyle": "round,pad=0.16",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.82,
                },
                zorder=4,
            )
        _reposition_axis_annotations(
            ax,
            axis_annotations,
            version_key=version_key,
            publication_variant=publication_variant,
            left_metric=left_metric,
            right_metric=right_metric,
        )

    if include_footnote:
        legend_handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#b0b0b0", markeredgecolor="white", markersize=8, label="Model-colored points"),
            Line2D([0], [0], marker=HIGHLIGHT_STYLE[BASE_MODEL_NAME]["marker"], color="none", markerfacecolor=_model_point_color(BASE_MODEL_NAME), markeredgecolor="#111111", markersize=9, label="Llama-3.1-8B"),
            Line2D([0], [0], marker=HIGHLIGHT_STYLE["Llama SFT"]["marker"], color="none", markerfacecolor=_model_point_color("ft:Llama-3.1-8B-Instruct-SFT-20260430:breadth-group-3epochs-unsloth"), markeredgecolor="#111111", markersize=9, label="Llama SFT"),
            Line2D([0, 1], [0, 1], color="#4a4a4a", lw=1.5, label="Regression line"),
        ]
        fig.legend(legend_handles, [handle.get_label() for handle in legend_handles], loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.985), fontsize=11)
        fig.suptitle(f"{version_style['label']}: Public/Private ABM Metric Correlations (lower error is better)", fontsize=17)
    if include_footnote:
        footnote_text = _raw_full_name_footnote(version_rows)
        if footnote_text:
            fig.text(
                0.01,
                0.03,
                "\n".join(textwrap.wrap(f"Full model names: {footnote_text}", width=180)),
                ha="left",
                va="bottom",
                fontsize=10,
                color="#4f4f4f",
            )
    saved_paths = [output_path]
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    pdf_output_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_output_path, bbox_inches="tight")
    saved_paths.append(pdf_output_path)
    plt.close(fig)
    return saved_paths


def main() -> int:
    args = _build_arg_parser().parse_args()
    input_csv = args.input_csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _prepare_rows(input_csv, include_degroot=bool(args.include_degroot))
    summary_rows = _correlation_rows(rows)
    stem = _output_base_name(input_csv, include_degroot=bool(args.include_degroot))

    rows_output = output_dir / f"{stem}_metric_rows.csv"
    summary_output = output_dir / f"{stem}_correlations.csv"

    rows.to_csv(rows_output, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_output, index=False)
    figure_outputs: list[Path] = []
    for version_key, version_style in VERSION_STYLE.items():
        for publication_variant in ("raw", "pub"):
            figure_output = output_dir / f"{stem}_{version_style['output_stem']}_{publication_variant}.png"
            figure_outputs.extend(
                _render_version_scatter_panels(
                    rows,
                    summary_rows,
                    version_key=version_key,
                    publication_variant=publication_variant,
                    output_path=figure_output,
                )
            )

    print(f"rows_csv={rows_output}")
    print(f"summary_csv={summary_output}")
    for figure_output in figure_outputs:
        print(f"figure_path={figure_output}")
    for row in summary_rows:
        print(
            f"{row['scope']}: {row['left_metric']} vs {row['right_metric']} | "
            f"pearson={row['pearson_r']:.3f} spearman={row['spearman_rho']:.3f} n={row['n']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())