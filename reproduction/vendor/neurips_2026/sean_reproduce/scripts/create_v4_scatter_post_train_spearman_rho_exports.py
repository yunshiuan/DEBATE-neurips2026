#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
RADAR_DIR = REPO_ROOT / "neurips_2026" / "chengtao" / "src" / "radar"
if str(RADAR_DIR) not in sys.path:
    sys.path.insert(0, str(RADAR_DIR))

import generate_v3_depth_base_model_scatter_reports as scatter_mod  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "neurips_2026"
    / "sean_reproduce"
    / "v4"
    / "scatter_plot"
    / "post_train"
    / "spearman_rho"
)
DEFAULT_BOOTSTRAP_REPS = 1000
DEFAULT_BOOTSTRAP_SEED = 1
DEFAULT_WORKERS = 1


@dataclass(frozen=True)
class ExportSpec:
    subset: str
    stem: str


@dataclass(frozen=True)
class ExportSuite:
    family_key: str
    source_report: Path
    export_specs: tuple[ExportSpec, ...]


EXPORT_SUITES = (
    ExportSuite(
        family_key="breadth_group",
        source_report=(
            REPO_ROOT
            / "neurips_2026"
            / "sean_reproduce"
            / "v4"
            / "scatter_plot"
            / "post_train"
            / "report_breadth_group_post_train_partner_initial_gap_vs_final_change_scatter_v4.pdf"
        ),
        export_specs=(
            ExportSpec(
                subset="train",
                stem="report_breadth_group_train_post_train_partner_initial_gap_vs_final_change_spearman_rho_v4",
            ),
            ExportSpec(
                subset="test",
                stem="report_breadth_group_test_post_train_partner_initial_gap_vs_final_change_spearman_rho_v4",
            ),
        ),
    ),
    ExportSuite(
        family_key="breadth_group",
        source_report=(
            REPO_ROOT
            / "neurips_2026"
            / "sean_reproduce"
            / "v4"
            / "scatter_plot"
            / "post_train"
            / "report_breadth_group_post_train_partner_initial_gap_vs_final_change_scatter_v4_unfiltered.pdf"
        ),
        export_specs=(
            ExportSpec(
                subset="train",
                stem="report_breadth_group_train_post_train_partner_initial_gap_vs_final_change_spearman_rho_v4_unfiltered",
            ),
            ExportSpec(
                subset="test",
                stem="report_breadth_group_test_post_train_partner_initial_gap_vs_final_change_spearman_rho_v4_unfiltered",
            ),
        ),
    ),
    ExportSuite(
        family_key="breadth_topic",
        source_report=(
            REPO_ROOT
            / "neurips_2026"
            / "sean_reproduce"
            / "v4"
            / "scatter_plot"
            / "post_train"
            / "report_breadth_topic_post_train_partner_initial_gap_vs_final_change_scatter_v4.pdf"
        ),
        export_specs=(
            ExportSpec(
                subset="train",
                stem="report_breadth_topic_train_post_train_partner_initial_gap_vs_final_change_spearman_rho_v4",
            ),
            ExportSpec(
                subset="test",
                stem="report_breadth_topic_test_post_train_partner_initial_gap_vs_final_change_spearman_rho_v4",
            ),
        ),
    ),
    ExportSuite(
        family_key="breadth_topic",
        source_report=(
            REPO_ROOT
            / "neurips_2026"
            / "sean_reproduce"
            / "v4"
            / "scatter_plot"
            / "post_train"
            / "report_breadth_topic_post_train_partner_initial_gap_vs_final_change_scatter_v4_unfiltered.pdf"
        ),
        export_specs=(
            ExportSpec(
                subset="train",
                stem="report_breadth_topic_train_post_train_partner_initial_gap_vs_final_change_spearman_rho_v4_unfiltered",
            ),
            ExportSpec(
                subset="test",
                stem="report_breadth_topic_test_post_train_partner_initial_gap_vs_final_change_spearman_rho_v4_unfiltered",
            ),
        ),
    ),
)

PANEL_ORDER = ("public", "private")
PANEL_TITLE_PREFIX = {"public": "Public", "private": "Private"}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for fieldname in row.keys():
            if fieldname in seen:
                continue
            seen.add(fieldname)
            fieldnames.append(fieldname)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def float_or_nan(value: str) -> float:
    text = str(value).strip()
    if not text:
        return float("nan")
    return float(text)


def correlation_rows(rows: list[dict[str, str]], *, subset: str, panel_key: str) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if str(row.get("row_type", "")).strip() == "correlation_summary"
        and str(row.get("subset", "")).strip() == subset
        and str(row.get("panel_key", "")).strip() == panel_key
    ]


def point_pairs_by_entity(rows: list[dict[str, str]], *, subset: str, panel_key: str) -> dict[str, list[tuple[float, float]]]:
    pairs_by_entity: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        if str(row.get("row_type", "")).strip() != "point":
            continue
        if str(row.get("subset", "")).strip() != subset:
            continue
        if str(row.get("panel_key", "")).strip() != panel_key:
            continue
        entity = str(row.get("entity", "")).strip()
        if not entity:
            continue
        x_value = float_or_nan(row.get("x_value", ""))
        y_value = float_or_nan(row.get("y_value", ""))
        if math.isnan(x_value) or math.isnan(y_value):
            continue
        pairs_by_entity.setdefault(entity, []).append((x_value, y_value))
    return pairs_by_entity


def percentile_value(values: list[float], quantile: float) -> float:
    if not values:
        return math.nan
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = max(0.0, min(float(len(sorted_values) - 1), quantile * float(len(sorted_values) - 1)))
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return sorted_values[lower_index]
    fraction = position - lower_index
    return sorted_values[lower_index] * (1.0 - fraction) + sorted_values[upper_index] * fraction


def bootstrap_spearman_ci(
    point_pairs: list[tuple[float, float]],
    *,
    bootstrap_reps: int,
    seed: int,
) -> tuple[float, float]:
    if len(point_pairs) < 2 or bootstrap_reps <= 0:
        return math.nan, math.nan
    rng = random.Random(seed)
    bootstrap_samples: list[float] = []
    n_points = len(point_pairs)
    for _ in range(bootstrap_reps):
        sample = [point_pairs[rng.randrange(n_points)] for _ in range(n_points)]
        rho = scatter_mod.spearman_rho(
            [x_value for x_value, _ in sample],
            [y_value for _, y_value in sample],
        )
        if not math.isnan(rho):
            bootstrap_samples.append(rho)
    return percentile_value(bootstrap_samples, 0.025), percentile_value(bootstrap_samples, 0.975)
def load_degroot_fit_alphas(*, family_key: str, subset: str) -> tuple[float, float] | None:
    fit_csv_path = (
        REPO_ROOT
        / "neurips_2026"
        / "sean_reproduce"
        / "v4"
        / "abm"
        / "group_opinion_convergence"
        / "post_train"
        / "csv"
        / f"group_opinion_convergence_models_vs_all_{family_key}_{subset}_degroot_fit_v4.csv"
    )
    if not fit_csv_path.exists():
        return None
    with fit_csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            alpha_public = str(row.get("fit_alpha_public", "")).strip()
            alpha_private = str(row.get("fit_alpha_private", "")).strip()
            if not alpha_public or not alpha_private:
                continue
            return float(alpha_public), float(alpha_private)
    return None


def correlation_rows_with_ci(
    rows: list[dict[str, str]],
    *,
    subset: str,
    panel_key: str,
    bootstrap_reps: int,
    bootstrap_seed: int,
    workers: int,
) -> list[dict[str, str | float]]:
    summary_rows = correlation_rows(rows, subset=subset, panel_key=panel_key)
    points_by_entity = point_pairs_by_entity(rows, subset=subset, panel_key=panel_key)

    def enrich_one(index_and_row: tuple[int, dict[str, str]]) -> dict[str, str | float]:
        index, row = index_and_row
        entity = str(row.get("entity", "")).strip()
        ci_low, ci_high = bootstrap_spearman_ci(
            points_by_entity.get(entity, []),
            bootstrap_reps=bootstrap_reps,
            seed=bootstrap_seed + index + (1000 if panel_key == "private" else 0),
        )
        return {
            **row,
            "spearman_rho_ci_low": "" if math.isnan(ci_low) else ci_low,
            "spearman_rho_ci_high": "" if math.isnan(ci_high) else ci_high,
        }

    indexed_rows = list(enumerate(summary_rows))
    max_workers = max(1, int(workers))
    if max_workers == 1 or len(indexed_rows) < 2:
        return [enrich_one(item) for item in indexed_rows]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(indexed_rows))) as executor:
        return list(executor.map(enrich_one, indexed_rows))


def selected_export_suites(source_reports: list[Path]) -> list[ExportSuite]:
    if not source_reports:
        return list(EXPORT_SUITES)

    requested = {path.resolve() for path in source_reports}
    suites = [suite for suite in EXPORT_SUITES if suite.source_report.resolve() in requested]
    missing = sorted(str(path) for path in requested if path not in {suite.source_report.resolve() for suite in suites})
    if missing:
        raise FileNotFoundError(f"Unsupported source report(s): {', '.join(missing)}")
    return suites


def model_id_footnote(rows: list[dict[str, str]]) -> str | None:
    entries: list[str] = []
    seen: set[str] = set()
    for row in rows:
        short_label = str(row.get("entity_display", "")).strip() or str(row.get("entity", "")).strip()
        full_label = str(row.get("model_dir", "")).strip() or str(row.get("entity", "")).strip()
        if not short_label or not full_label or short_label == full_label:
            continue
        footnote_entry = f"{short_label}={full_label}"
        if footnote_entry in seen:
            continue
        seen.add(footnote_entry)
        entries.append(footnote_entry)
    if not entries:
        return None
    return textwrap.fill("Model IDs: " + " | ".join(entries), width=160)


def export_rows(
    rows: list[dict[str, str | float]],
    *,
    subset: str,
    export_relpath: str,
    degroot_fit_alphas: tuple[float, float] | None,
) -> list[dict[str, str]]:
    alpha_public = ""
    alpha_private = ""
    if degroot_fit_alphas is not None:
        alpha_public = str(degroot_fit_alphas[0])
        alpha_private = str(degroot_fit_alphas[1])
    exported_rows: list[dict[str, str]] = []
    for row in rows:
        if str(row.get("row_type", "")).strip() != "correlation_summary":
            continue
        if str(row.get("subset", "")).strip() != subset:
            continue
        exported_rows.append(
            {
                **{key: str(value) if isinstance(value, float) else value for key, value in row.items()},
                "export_relpath": export_relpath,
                "export_metric": "spearman_rho",
                "degroot_fit_alpha_public": alpha_public,
                "degroot_fit_alpha_private": alpha_private,
            }
        )
    return exported_rows


def draw_error_bar(ax: object, *, center_x: float, value: float, low: float, high: float) -> None:
    if math.isnan(value) or math.isnan(low) or math.isnan(high):
        return
    lower_error = max(0.0, value - low)
    upper_error = max(0.0, high - value)
    if math.isclose(lower_error, 0.0) and math.isclose(upper_error, 0.0):
        return
    ax.errorbar(
        [center_x],
        [value],
        yerr=[[lower_error], [upper_error]],
        fmt="none",
        ecolor="#2F2F2F",
        elinewidth=1.4,
        capsize=4.0,
        capthick=1.4,
        zorder=5,
    )


def render_panel(ax: object, *, rows: list[dict[str, str | float]], panel_title: str) -> None:
    labels = [scatter_mod.get_model_display_label(str(row.get("entity", "")).strip()) for row in rows]
    values = [float_or_nan(row.get("spearman_rho", "")) for row in rows]
    ci_lows = [float_or_nan(row.get("spearman_rho_ci_low", "")) for row in rows]
    ci_highs = [float_or_nan(row.get("spearman_rho_ci_high", "")) for row in rows]
    counts = [str(row.get("n_points", "")).strip() for row in rows]
    colors = [scatter_mod.get_model_color(str(row.get("entity", "")).strip(), "#777777") for row in rows]
    x_positions = list(range(len(rows)))
    plotted_values = [0.0 if math.isnan(value) else value for value in values]

    ax.bar(x_positions, plotted_values, color=colors, alpha=0.95)
    ax.axhline(0.0, color="#7f7f7f", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_ylim(-1.0, 1.1)
    ax.grid(True, axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.set_ylabel("Spearman ρ", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title(panel_title, loc="left", fontweight="bold", fontsize=10)

    tick_fontsize = 7 if len(labels) <= 8 else 6
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=tick_fontsize)
    ax.tick_params(axis="x", pad=0)

    for index, (value, ci_low, ci_high) in enumerate(zip(values, ci_lows, ci_highs)):
        draw_error_bar(ax, center_x=float(index), value=value, low=ci_low, high=ci_high)
        label = "n/a" if math.isnan(value) else f"{value:.2f}"
        top_y = value
        if not math.isnan(ci_high):
            top_y = max(top_y, ci_high)
        y_position = min(1.08, top_y + (0.04 if plotted_values[index] >= 0 else -0.07))
        count_text = counts[index] or "n/a"
        ax.text(
            index,
            y_position,
            f"{label}\n(n={count_text})",
            ha="center",
            va="bottom" if plotted_values[index] >= 0 else "top",
            fontsize=7,
            color="#333333",
        )


def render_subset_page(
    rows: list[dict[str, str]],
    *,
    family_key: str,
    subset: str,
    output_pdf: Path,
    output_png: Path,
    bootstrap_reps: int,
    bootstrap_seed: int,
    workers: int,
) -> list[dict[str, str | float]]:
    suite_label = str(rows[0].get("suite_scope", "")).strip() if rows else "Scatter Report"
    subset_label = subset.title()
    panel_rows = {
        panel_key: correlation_rows_with_ci(
            rows,
            subset=subset,
            panel_key=panel_key,
            bootstrap_reps=bootstrap_reps,
            bootstrap_seed=bootstrap_seed,
            workers=workers,
        )
        for panel_key in PANEL_ORDER
    }
    degroot_fit_alphas = load_degroot_fit_alphas(family_key=family_key, subset=subset)

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8))
    for axis, panel_key in zip(axes, PANEL_ORDER):
        render_panel(
            axis,
            rows=panel_rows[panel_key],
            panel_title=f"{PANEL_TITLE_PREFIX[panel_key]} {subset_label}",
        )

    footnote = model_id_footnote([row for panel_key in PANEL_ORDER for row in panel_rows[panel_key]])
    alpha_note = None
    if degroot_fit_alphas is not None:
        alpha_note = (
            f"DeGroot fit alpha: public={degroot_fit_alphas[0]:.3f}, "
            f"private={degroot_fit_alphas[1]:.3f}"
        )
    fig.suptitle(
        f"Spearman Rho Summary\n{suite_label} - {subset_label}",
        x=0.01,
        y=0.985,
        ha="left",
        va="top",
        fontweight="bold",
    )
    if alpha_note:
        fig.text(0.01, 0.10, alpha_note, ha="left", va="bottom", fontsize=8.5, color="#444444")
    if footnote:
        fig.text(0.01, 0.02, footnote, ha="left", va="bottom", fontsize=8, color="#444444")
    bottom_margin = 0.16 if footnote and alpha_note else (0.12 if footnote or alpha_note else 0.04)
    fig.tight_layout(rect=(0.0, bottom_margin, 1.0, 0.90))
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [row for panel_key in PANEL_ORDER for row in panel_rows[panel_key]]


def render_exports(
    *,
    source_reports: list[Path],
    output_dir: Path,
    bootstrap_reps: int,
    bootstrap_seed: int,
    workers: int,
) -> list[Path]:
    rendered_paths: list[Path] = []
    for export_suite in selected_export_suites(source_reports):
        source_report = export_suite.source_report
        source_csv = source_report.with_suffix(".csv")
        if not source_report.exists():
            raise FileNotFoundError(f"Missing source report: {source_report}")
        if not source_csv.exists():
            raise FileNotFoundError(f"Missing source CSV: {source_csv}")

        rows = read_csv_rows(source_csv)
        for export_spec in export_suite.export_specs:
            output_pdf = output_dir / f"{export_spec.stem}.pdf"
            output_png = output_dir / f"{export_spec.stem}.png"
            output_csv = output_dir / f"{export_spec.stem}.csv"
            export_rows_with_ci = render_subset_page(
                rows,
                family_key=export_suite.family_key,
                subset=export_spec.subset,
                output_pdf=output_pdf,
                output_png=output_png,
                bootstrap_reps=bootstrap_reps,
                bootstrap_seed=bootstrap_seed,
                workers=workers,
            )
            degroot_fit_alphas = load_degroot_fit_alphas(
                family_key=export_suite.family_key,
                subset=export_spec.subset,
            )
            write_rows_csv(
                output_csv,
                export_rows(
                    export_rows_with_ci,
                    subset=export_spec.subset,
                    export_relpath=str(output_pdf.relative_to(REPO_ROOT)),
                    degroot_fit_alphas=degroot_fit_alphas,
                ),
            )
            rendered_paths.extend([output_pdf, output_png, output_csv])
    return rendered_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create standalone Spearman rho summary exports from the v4 breadth group/topic post-train scatter reports."
    )
    parser.add_argument(
        "--source-report",
        type=Path,
        action="append",
        default=[],
        help="Optional source report path(s) to limit regeneration; defaults to all configured breadth group/topic partner-gap scatter suites.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of threads to use for bootstrap CI computation within each panel.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    for output_path in render_exports(
        source_reports=[path.resolve() for path in args.source_report],
        output_dir=args.output_dir.resolve(),
        bootstrap_reps=int(args.bootstrap_reps),
        bootstrap_seed=int(args.bootstrap_seed),
        workers=max(1, int(args.workers)),
    ):
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())