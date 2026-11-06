#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
RADAR_DIR = REPO_ROOT / "neurips_2026" / "chengtao" / "src" / "radar"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(RADAR_DIR) not in sys.path:
    sys.path.insert(0, str(RADAR_DIR))

from generate_plots import (  # noqa: E402
    LIKERT_GAUSSIAN_SIGMA,
    LIKERT_GAUSSIAN_SIGMA_SECONDARY,
    MODEL_COLOR_MAP,
    UNIFORM_LB_MODEL,
    V2_BAR_Y_LIMITS,
    add_model_footnotes,
    as_float_or_nan,
    format_model_display_label,
    get_model_footnotes,
    get_model_color,
    get_v2_display_model_order,
)
from neurips_2026.sean_reproduce.base_model_v3_style import TULU_BASE_MODEL  # noqa: E402

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"matplotlib is required to render the v4 vs v4_b comparison report: {exc}")


COMPARE_ROOT = REPO_ROOT / "neurips_2026" / "sean_reproduce" / "v4_vs_v4b" / "summary_base_model"
DEFAULT_OUTPUT_PDF = COMPARE_ROOT / "report_bar_base_model_v4_vs_4b_unfiltered_msg_and_ind.pdf"

VERSION_SPECS = {
    "v4": {
        "label": "v4",
        "report_pdf": REPO_ROOT / "neurips_2026" / "sean_reproduce" / "v4" / "summary_base_model" / "report_bar_base_model_v4_unfiltered_msg_and_ind.pdf",
        "csv_dir": REPO_ROOT / "neurips_2026" / "sean_reproduce" / "v4" / "summary_base_model" / "csv",
        "offset": -0.18,
        "hatch": None,
        "edgecolor": "#1F1F1F",
        "alpha": 0.92,
        "linestyle": "--",
    },
    "v4_b": {
        "label": "v4_b",
        "report_pdf": REPO_ROOT / "neurips_2026" / "sean_reproduce" / "v4_b" / "summary_base_model" / "report_bar_base_model_v4_b_unfiltered_msg_and_ind.pdf",
        "csv_dir": REPO_ROOT / "neurips_2026" / "sean_reproduce" / "v4_b" / "summary_base_model" / "csv",
        "offset": 0.18,
        "hatch": "///",
        "edgecolor": "#1F1F1F",
        "alpha": 0.75,
        "linestyle": ":",
    },
}

PANEL_SPECS = [
    {
        "key": "all_topics",
        "csv_name": {"v4": "raw_metrics_v4_unfiltered_msg_and_ind.csv", "v4_b": "raw_metrics_v4_b_unfiltered_msg_and_ind.csv"},
        "comparison_csv": "raw_metrics_v4_vs_v4b_unfiltered_msg_and_ind.csv",
        "png_name": "bar_all_models_v4_vs_v4b_unfiltered_msg_and_ind.png",
        "page_title": "DEBATE Alignment Bar Plots (All Topics)",
    },
    {
        "key": "breadth_topics",
        "csv_name": {"v4": "raw_metrics_breadth_topics_v4_unfiltered_msg_and_ind.csv", "v4_b": "raw_metrics_breadth_topics_v4_b_unfiltered_msg_and_ind.csv"},
        "comparison_csv": "raw_metrics_breadth_topics_v4_vs_v4b_unfiltered_msg_and_ind.csv",
        "png_name": "bar_breadth_topics_v4_vs_v4b_unfiltered_msg_and_ind.png",
        "page_title": "DEBATE Alignment Bar Plots (Breadth Topics)",
    },
    {
        "key": "depth_topics",
        "csv_name": {"v4": "raw_metrics_depth_topics_v4_unfiltered_msg_and_ind.csv", "v4_b": "raw_metrics_depth_topics_v4_b_unfiltered_msg_and_ind.csv"},
        "comparison_csv": "raw_metrics_depth_topics_v4_vs_v4b_unfiltered_msg_and_ind.csv",
        "png_name": "bar_depth_topics_v4_vs_v4b_unfiltered_msg_and_ind.png",
        "page_title": "DEBATE Alignment Bar Plots (Depth Topics)",
    },
]

IGNORED_MODELS = {TULU_BASE_MODEL}

METRIC_GROUPS = [
    (
        "Message-level",
        "Opinion\nPrediction Error",
        [
            ("Public Opinion", "public_stance_alignment_abs"),
            ("Private Opinion", "private_stance_alignment_abs"),
        ],
        [
            "public_alignment_count",
            "private_alignment_count",
        ],
        [
            ("lb_public_stance_alignment_abs", "ub_noise_public_stance_alignment_abs", "ub_noise_sigma_half_public_stance_alignment_abs"),
            ("lb_private_stance_alignment_abs", "ub_noise_private_stance_alignment_abs", "ub_noise_sigma_half_private_stance_alignment_abs"),
        ],
    ),
    (
        "Individual-level",
        "Opinion Change\nPrediction Error",
        [
            ("Public Opinion", "public_stance_change_error_abs"),
            ("Private Opinion", "private_stance_change_error_abs"),
        ],
        [
            "public_change_count",
            "private_change_count",
        ],
        [
            ("lb_public_stance_change_error_abs", "ub_noise_public_stance_change_error_abs", "ub_noise_sigma_half_public_stance_change_error_abs"),
            ("lb_private_stance_change_error_abs", "ub_noise_private_stance_change_error_abs", "ub_noise_sigma_half_private_stance_change_error_abs"),
        ],
    ),
    (
        "Group-level",
        "Opinion Convergence\nPrediction Error",
        [
            ("Public Opinion", "public_opinion_convergence_error"),
            ("Private Opinion", "private_opinion_convergence_error"),
        ],
        [
            "public_group_count",
            "private_group_count",
        ],
        [
            ("lb_public_opinion_convergence_error", "ub_noise_public_opinion_convergence_error", "ub_noise_sigma_half_public_opinion_convergence_error"),
            ("lb_private_opinion_convergence_error", "ub_noise_private_opinion_convergence_error", "ub_noise_sigma_half_private_opinion_convergence_error"),
        ],
    ),
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if str(row.get("model", "")).strip() not in IGNORED_MODELS
        ]


def write_comparison_csv(path: Path, rows_by_version: dict[str, list[dict[str, str]]]) -> None:
    fieldnames: list[str] = ["comparison_version"]
    for version_rows in rows_by_version.values():
        if not version_rows:
            continue
        for fieldname in version_rows[0].keys():
            if fieldname not in fieldnames:
                fieldnames.append(fieldname)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for version_key, version_rows in rows_by_version.items():
            for row in version_rows:
                writer.writerow({"comparison_version": version_key, **row})


def _plottable_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if str(row.get("model", "")).strip()
        and str(row.get("model", "")).strip().lower() != "human"
        and str(row.get("model", "")).strip() != UNIFORM_LB_MODEL
        and str(row.get("model", "")).strip() != "Mean Predictor"
    ]


def _row_lookup(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {str(row.get("model", "")).strip(): row for row in rows if str(row.get("model", "")).strip()}


def _draw_error_bar(ax: Any, *, center_x: float, value: float, low: Any, high: Any) -> None:
    low_value = as_float_or_nan(low)
    high_value = as_float_or_nan(high)
    if math.isnan(value) or math.isnan(low_value) or math.isnan(high_value):
        return
    lower_error = max(0.0, value - low_value)
    upper_error = max(0.0, high_value - value)
    if math.isclose(lower_error, 0.0) and math.isclose(upper_error, 0.0):
        return
    ax.errorbar(
        [center_x],
        [value],
        yerr=[[lower_error], [upper_error]],
        fmt="none",
        ecolor="#2F2F2F",
        elinewidth=1.5,
        capsize=4.0,
        capthick=1.5,
        zorder=5,
    )


def _annotate_count(ax: Any, *, center_x: float, count_value: Any, y_value: float) -> None:
    count_number = as_float_or_nan(count_value)
    if math.isnan(count_number):
        return
    ax.text(
        center_x,
        y_value,
        f"n={int(round(count_number))}",
        ha="center",
        va="bottom",
        rotation=90,
        fontsize=6.5,
        color="#666666",
        zorder=6,
    )


def _format_labels(model_names: list[str]) -> list[str]:
    return [format_model_display_label(model_name) for model_name in model_names]


def _draw_version_baselines(
    ax: Any,
    *,
    left_x: float,
    right_x: float,
    bound_fields: tuple[str, str, str],
    uniform_row: dict[str, str] | None,
    reference_row: dict[str, str] | None,
) -> None:
    lb_field, ub_primary_field, ub_secondary_field = bound_fields
    lb_metric_field = lb_field.replace("lb_", "")
    uniform_value = as_float_or_nan((uniform_row or {}).get(lb_metric_field))
    primary_ub = as_float_or_nan((reference_row or {}).get(ub_primary_field))
    secondary_ub = as_float_or_nan((reference_row or {}).get(ub_secondary_field))

    if not math.isnan(uniform_value):
        ax.hlines(uniform_value, left_x, right_x, colors="#7F7F7F", linestyles="--", linewidth=1.5, zorder=2)
    if not math.isnan(primary_ub):
        ax.hlines(primary_ub, left_x, right_x, colors="#D62728", linestyles="--", linewidth=1.5, zorder=2)
    if not math.isnan(secondary_ub):
        ax.hlines(secondary_ub, left_x, right_x, colors="#FF9896", linestyles="--", linewidth=1.5, zorder=2)


def build_page_figure(
    *,
    page_title: str,
    rows_by_version: dict[str, list[dict[str, str]]],
    hide_count_annotations: bool = False,
    model_order: list[str] | None = None,
    shared_models: set[str] | None = None,
) -> Any:
    combined_rows = rows_by_version["v4"] + rows_by_version["v4_b"]
    if model_order is None:
        model_order = get_v2_display_model_order(combined_rows, include_human=False)
    shared_model_set = {str(model_name).strip() for model_name in (shared_models or set())}
    lookup_by_version = {version_key: _row_lookup(rows) for version_key, rows in rows_by_version.items()}
    uniform_rows = {
        version_key: next((row for row in rows if str(row.get("model", "")).strip() == UNIFORM_LB_MODEL), None)
        for version_key, rows in rows_by_version.items()
    }
    reference_rows: dict[str, dict[str, str] | None] = {}
    for version_key, rows in rows_by_version.items():
        reference_rows[version_key] = next((lookup_by_version[version_key].get(model_name) for model_name in model_order if lookup_by_version[version_key].get(model_name) is not None), None)

    labels = _format_labels(model_order)
    num_models = len(model_order)
    public_centers = list(range(num_models))
    private_centers = [position + num_models + 1.6 for position in public_centers]
    separator_x = (public_centers[-1] + private_centers[0]) / 2
    all_centers = public_centers + private_centers
    bar_width = 0.32

    fig, axes = plt.subplots(3, 1, figsize=(17, 10), sharex=False)
    fig.suptitle(page_title, y=0.985, fontsize=14, fontweight="bold")
    fig.text(0.5, 0.948, "Direct Comparison of v4 and v4_b", ha="center", va="center", fontsize=12, fontweight="bold")

    version_handles = [
        Patch(facecolor="#BFBFBF", edgecolor="#1F1F1F", label="v4"),
        Patch(facecolor="#BFBFBF", edgecolor="#1F1F1F", hatch="///", alpha=0.75, label="v4_b"),
    ]
    baseline_handles = [
        Line2D([0], [0], color="#7F7F7F", linestyle="--", linewidth=1.5, label="Uniform LB"),
        Line2D([0], [0], color="#D62728", linestyle="--", linewidth=1.5, label=f"Noise UB (sigma={LIKERT_GAUSSIAN_SIGMA})"),
        Line2D([0], [0], color="#FF9896", linestyle="--", linewidth=1.5, label=f"Noise UB (sigma={LIKERT_GAUSSIAN_SIGMA_SECONDARY})"),
    ]
    fig.legend(
        handles=[*version_handles, *baseline_handles],
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.925),
        fontsize=9,
    )
    fig.text(0.985, 0.908, "Baselines shown once from v4 only", ha="right", va="top", fontsize=8, color="#444444")
    if shared_model_set:
        fig.text(0.985, 0.89, "Shared DeGroot baseline shown once when identical across v4 and v4_b", ha="right", va="top", fontsize=8, color="#444444")
    footnotes = get_model_footnotes(model_order)
    if footnotes:
        add_model_footnotes(fig, footnotes, y=0.02)

    for axis_index, (ax, (level_title, y_label, field_pairs, count_fields, bound_pairs)) in enumerate(zip(axes, METRIC_GROUPS)):
        y_limits = V2_BAR_Y_LIMITS[level_title]
        count_y = y_limits[0] + max(0.01, (y_limits[1] - y_limits[0]) * 0.03)
        for centers, field_pair, bound_pair, block_label in (
            (public_centers, field_pairs[0], bound_pairs[0], "Public"),
            (private_centers, field_pairs[1], bound_pairs[1], "Private"),
        ):
            left_x = centers[0] - 0.6
            right_x = centers[-1] + 0.6
            for version_key in ("v4", "v4_b"):
                version_lookup = lookup_by_version[version_key]
                offset = float(VERSION_SPECS[version_key]["offset"])
                hatch = VERSION_SPECS[version_key]["hatch"]
                edgecolor = str(VERSION_SPECS[version_key]["edgecolor"])
                alpha = float(VERSION_SPECS[version_key]["alpha"])
                metric_field = field_pair[1]
                count_field = count_fields[0] if block_label == "Public" else count_fields[1]
                values = [
                    math.nan if model_name in shared_model_set else as_float_or_nan((version_lookup.get(model_name) or {}).get(metric_field))
                    for model_name in model_order
                ]
                ci_lows = [
                    None if model_name in shared_model_set else (version_lookup.get(model_name) or {}).get(f"{metric_field}_ci_low")
                    for model_name in model_order
                ]
                ci_highs = [
                    None if model_name in shared_model_set else (version_lookup.get(model_name) or {}).get(f"{metric_field}_ci_high")
                    for model_name in model_order
                ]
                counts = [
                    None if model_name in shared_model_set else (version_lookup.get(model_name) or {}).get(count_field)
                    for model_name in model_order
                ]
                colors = [get_model_color(model_name, MODEL_COLOR_MAP.get(model_name, "#999999")) for model_name in model_order]
                bars = ax.bar(
                    [center + offset for center in centers],
                    values,
                    width=bar_width,
                    color=colors,
                    edgecolor=edgecolor,
                    linewidth=0.7,
                    alpha=alpha,
                    hatch=hatch,
                    zorder=3,
                )
                for bar, value, ci_low, ci_high, count_value in zip(bars, values, ci_lows, ci_highs, counts):
                    if math.isnan(value):
                        continue
                    center_x = bar.get_x() + bar.get_width() / 2
                    _draw_error_bar(ax, center_x=center_x, value=value, low=ci_low, high=ci_high)
                    if not hide_count_annotations:
                        _annotate_count(ax, center_x=center_x, count_value=count_value, y_value=count_y)

            if shared_model_set:
                shared_values: list[float] = []
                shared_ci_lows: list[str | None] = []
                shared_ci_highs: list[str | None] = []
                shared_counts: list[str | None] = []
                for model_name in model_order:
                    if model_name not in shared_model_set:
                        shared_values.append(math.nan)
                        shared_ci_lows.append(None)
                        shared_ci_highs.append(None)
                        shared_counts.append(None)
                        continue
                    shared_row = lookup_by_version["v4"].get(model_name) or lookup_by_version["v4_b"].get(model_name) or {}
                    shared_values.append(as_float_or_nan(shared_row.get(metric_field)))
                    shared_ci_lows.append(shared_row.get(f"{metric_field}_ci_low"))
                    shared_ci_highs.append(shared_row.get(f"{metric_field}_ci_high"))
                    shared_counts.append(shared_row.get(count_field))
                shared_bars = ax.bar(
                    centers,
                    shared_values,
                    width=bar_width * 1.35,
                    color=[get_model_color(model_name, MODEL_COLOR_MAP.get(model_name, "#999999")) for model_name in model_order],
                    edgecolor="#1F1F1F",
                    linewidth=0.8,
                    alpha=0.92,
                    zorder=4,
                )
                for bar, value, ci_low, ci_high, count_value in zip(shared_bars, shared_values, shared_ci_lows, shared_ci_highs, shared_counts):
                    if math.isnan(value):
                        continue
                    center_x = bar.get_x() + bar.get_width() / 2
                    _draw_error_bar(ax, center_x=center_x, value=value, low=ci_low, high=ci_high)
                    if not hide_count_annotations:
                        _annotate_count(ax, center_x=center_x, count_value=count_value, y_value=count_y)

            _draw_version_baselines(
                ax,
                left_x=left_x,
                right_x=right_x,
                bound_fields=bound_pair,
                uniform_row=uniform_rows["v4"],
                reference_row=reference_rows["v4"],
            )

            if axis_index == 0:
                center = sum(centers) / len(centers)
                ax.text(center, y_limits[1] - 0.08, block_label, ha="center", va="top", fontsize=10, fontweight="bold")

        ax.axvspan(public_centers[0] - 0.6, public_centers[-1] + 0.6, color="#4C78A8", alpha=0.03)
        ax.axvspan(private_centers[0] - 0.6, private_centers[-1] + 0.6, color="#F58518", alpha=0.03)
        ax.axvline(separator_x, color="#C7C7C7", linewidth=0.9, linestyle="--", alpha=0.8)
        ax.set_title(level_title, loc="left", fontweight="bold")
        ax.set_ylabel(y_label)
        ax.set_ylim(*y_limits)
        ax.set_xlim(public_centers[0] - 0.9, private_centers[-1] + 0.9)
        ax.grid(True, axis="y", alpha=0.24)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelbottom=axis_index == len(axes) - 1, pad=2)
        ax.set_xticks(all_centers)

    axes[-1].set_xticklabels(labels + labels, rotation=40, ha="right")
    for ax in axes[:-1]:
        ax.set_xticklabels([])

    axes[-1].tick_params(axis="x", labelbottom=True, pad=8)
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.29 if footnotes else 0.23, top=0.76, hspace=0.34)
    return fig


def build_cover_figure() -> Any:
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.5, 0.78, "v4 vs v4_b", ha="center", va="center", fontsize=24, fontweight="bold")
    ax.text(0.5, 0.69, "Summary Base-Model Unfiltered Message/Individual Bar Report", ha="center", va="center", fontsize=16)
    ax.text(0.5, 0.54, "Pages compare v4 and v4_b side by side for each model within the same public/private panel.", ha="center", va="center", fontsize=11)
    ax.text(0.5, 0.48, "Solid bars = v4, hatched bars = v4_b.", ha="center", va="center", fontsize=11)
    ax.text(
        0.08,
        0.28,
        "Source reports:\n"
        f"- {VERSION_SPECS['v4']['report_pdf'].relative_to(REPO_ROOT)}\n"
        f"- {VERSION_SPECS['v4_b']['report_pdf'].relative_to(REPO_ROOT)}",
        ha="left",
        va="top",
        fontsize=10,
        family="monospace",
    )
    return fig


def render_report(output_pdf: Path, *, hide_count_annotations: bool = False) -> None:
    output_dir = output_pdf.parent
    img_dir = output_dir / "img"
    csv_dir = output_dir / "csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_pdf) as pdf:
        cover = build_cover_figure()
        pdf.savefig(cover, bbox_inches="tight")
        plt.close(cover)

        for panel_spec in PANEL_SPECS:
            rows_by_version = {
                version_key: read_csv_rows(Path(VERSION_SPECS[version_key]["csv_dir"]) / str(panel_spec["csv_name"][version_key]))
                for version_key in ("v4", "v4_b")
            }
            write_comparison_csv(csv_dir / str(panel_spec["comparison_csv"]), rows_by_version)
            fig = build_page_figure(
                page_title=str(panel_spec["page_title"]),
                rows_by_version=rows_by_version,
                hide_count_annotations=hide_count_annotations,
            )
            pdf.savefig(fig, bbox_inches="tight")
            fig.savefig(img_dir / str(panel_spec["png_name"]), dpi=220)
            plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the v4 vs v4_b summary base-model comparison bar report.")
    parser.add_argument("--output-pdf", type=Path, default=DEFAULT_OUTPUT_PDF)
    parser.add_argument(
        "--hide-count-annotations",
        action="store_true",
        help="Hide n=XXX count annotations in the v4-vs-v4_b comparison pages.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    render_report(args.output_pdf.resolve(), hide_count_annotations=bool(args.hide_count_annotations))
    print(args.output_pdf.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())