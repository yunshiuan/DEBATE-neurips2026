#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import sys
import textwrap
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
    / "v4_pub"
    / "scatter_plot"
    / "post_train"
    / "spearman_rho"
)


@dataclass(frozen=True)
class ExportTarget:
    source_csv: Path
    stem: str


EXPORT_TARGETS = (
    ExportTarget(
        source_csv=(
            REPO_ROOT
            / "neurips_2026"
            / "sean_reproduce"
            / "v4"
            / "scatter_plot"
            / "post_train"
            / "spearman_rho"
            / "report_breadth_group_train_post_train_partner_initial_gap_vs_final_change_spearman_rho_v4.csv"
        ),
        stem="report_breadth_group_train_post_train_partner_initial_gap_vs_final_change_spearman_rho_v4",
    ),
    ExportTarget(
        source_csv=(
            REPO_ROOT
            / "neurips_2026"
            / "sean_reproduce"
            / "v4"
            / "scatter_plot"
            / "post_train"
            / "spearman_rho"
            / "report_breadth_group_test_post_train_partner_initial_gap_vs_final_change_spearman_rho_v4.csv"
        ),
        stem="report_breadth_group_test_post_train_partner_initial_gap_vs_final_change_spearman_rho_v4",
    ),
    ExportTarget(
        source_csv=(
            REPO_ROOT
            / "neurips_2026"
            / "sean_reproduce"
            / "v4"
            / "scatter_plot"
            / "post_train"
            / "spearman_rho"
            / "report_breadth_group_train_post_train_partner_initial_gap_vs_final_change_spearman_rho_v4_unfiltered.csv"
        ),
        stem="report_breadth_group_train_post_train_partner_initial_gap_vs_final_change_spearman_rho_v4_unfiltered",
    ),
    ExportTarget(
        source_csv=(
            REPO_ROOT
            / "neurips_2026"
            / "sean_reproduce"
            / "v4"
            / "scatter_plot"
            / "post_train"
            / "spearman_rho"
            / "report_breadth_group_test_post_train_partner_initial_gap_vs_final_change_spearman_rho_v4_unfiltered.csv"
        ),
        stem="report_breadth_group_test_post_train_partner_initial_gap_vs_final_change_spearman_rho_v4_unfiltered",
    ),
    ExportTarget(
        source_csv=(
            REPO_ROOT
            / "neurips_2026"
            / "sean_reproduce"
            / "v4"
            / "scatter_plot"
            / "post_train"
            / "spearman_rho"
            / "report_breadth_topic_train_post_train_partner_initial_gap_vs_final_change_spearman_rho_v4.csv"
        ),
        stem="report_breadth_topic_train_post_train_partner_initial_gap_vs_final_change_spearman_rho_v4",
    ),
    ExportTarget(
        source_csv=(
            REPO_ROOT
            / "neurips_2026"
            / "sean_reproduce"
            / "v4"
            / "scatter_plot"
            / "post_train"
            / "spearman_rho"
            / "report_breadth_topic_test_post_train_partner_initial_gap_vs_final_change_spearman_rho_v4.csv"
        ),
        stem="report_breadth_topic_test_post_train_partner_initial_gap_vs_final_change_spearman_rho_v4",
    ),
    ExportTarget(
        source_csv=(
            REPO_ROOT
            / "neurips_2026"
            / "sean_reproduce"
            / "v4"
            / "scatter_plot"
            / "post_train"
            / "spearman_rho"
            / "report_breadth_topic_train_post_train_partner_initial_gap_vs_final_change_spearman_rho_v4_unfiltered.csv"
        ),
        stem="report_breadth_topic_train_post_train_partner_initial_gap_vs_final_change_spearman_rho_v4_unfiltered",
    ),
    ExportTarget(
        source_csv=(
            REPO_ROOT
            / "neurips_2026"
            / "sean_reproduce"
            / "v4"
            / "scatter_plot"
            / "post_train"
            / "spearman_rho"
            / "report_breadth_topic_test_post_train_partner_initial_gap_vs_final_change_spearman_rho_v4_unfiltered.csv"
        ),
        stem="report_breadth_topic_test_post_train_partner_initial_gap_vs_final_change_spearman_rho_v4_unfiltered",
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


def float_or_nan(value: str | float) -> float:
    text = str(value).strip()
    if not text:
        return float("nan")
    return float(text)


def panel_rows(rows: list[dict[str, str]], *, panel_key: str) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if str(row.get("row_type", "")).strip() == "correlation_summary"
        and str(row.get("panel_key", "")).strip() == panel_key
    ]


def extract_degroot_alpha_note(rows: list[dict[str, str]]) -> str | None:
    for row in rows:
        alpha_public = str(row.get("degroot_fit_alpha_public", "")).strip()
        alpha_private = str(row.get("degroot_fit_alpha_private", "")).strip()
        if not alpha_public and not alpha_private:
            continue
        parts: list[str] = []
        if alpha_public:
            parts.append(f"public={float(alpha_public):.3f}")
        if alpha_private:
            parts.append(f"private={float(alpha_private):.3f}")
        return "DeGroot fit alpha: " + ", ".join(parts)
    return None


def model_id_footnote(rows: list[dict[str, str]]) -> str | None:
    entries: list[str] = []
    seen: set[str] = set()
    for row in rows:
        short_label = str(row.get("entity_display", "")).strip() or str(row.get("entity", "")).strip()
        full_label = str(row.get("model_dir", "")).strip() or str(row.get("entity", "")).strip()
        if not short_label or not full_label or short_label == full_label:
            continue
        entry = f"{short_label}={full_label}"
        if entry in seen:
            continue
        seen.add(entry)
        entries.append(entry)
    if not entries:
        return None
    return textwrap.fill("Model IDs: " + " | ".join(entries), width=160)


def export_rows(
    rows: list[dict[str, str]],
    *,
    export_relpath: str,
    variant: str,
) -> list[dict[str, str]]:
    exported_rows: list[dict[str, str]] = []
    for row in rows:
        if str(row.get("row_type", "")).strip() != "correlation_summary":
            continue
        exported_rows.append(
            {
                **row,
                "source_export_relpath": str(row.get("export_relpath", "")).strip(),
                "export_relpath": export_relpath,
                "publication_variant": variant,
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


def render_panel(
    ax: object,
    *,
    rows: list[dict[str, str]],
    panel_title: str,
    publication: bool,
) -> None:
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
    if publication:
        ax.grid(False)
    else:
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
        if math.isnan(value):
            label_text = "n/a"
        elif publication:
            label_text = f"{value:.2f}"
        else:
            count_text = counts[index] or "n/a"
            label_text = f"{value:.2f}\n(n={count_text})"
        top_y = value
        if not math.isnan(ci_high):
            top_y = max(top_y, ci_high)
        y_position = min(1.08, top_y + (0.04 if plotted_values[index] >= 0 else -0.07))
        ax.text(
            index,
            y_position,
            label_text,
            ha="center",
            va="bottom" if plotted_values[index] >= 0 else "top",
            fontsize=7,
            color="#333333",
        )


def render_figure(
    rows: list[dict[str, str]],
    *,
    output_pdf: Path,
    output_png: Path,
    publication: bool,
) -> None:
    suite_label = str(rows[0].get("suite_scope", "")).strip() if rows else "Scatter Report"
    subset_label = str(rows[0].get("subset", "")).strip().title() if rows else ""
    panel_payload = {panel_key: panel_rows(rows, panel_key=panel_key) for panel_key in PANEL_ORDER}
    alpha_note = extract_degroot_alpha_note(rows)
    footnote = model_id_footnote([row for panel_key in PANEL_ORDER for row in panel_payload[panel_key]])

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8))
    for axis, panel_key in zip(axes, PANEL_ORDER):
        render_panel(
            axis,
            rows=panel_payload[panel_key],
            panel_title=f"{PANEL_TITLE_PREFIX[panel_key]} {subset_label}",
            publication=publication,
        )

    fig.suptitle(
        f"Spearman Rho Summary\n{suite_label} - {subset_label}",
        x=0.01,
        y=0.985,
        ha="left",
        va="top",
        fontweight="bold",
    )
    if not publication and alpha_note:
        fig.text(0.01, 0.10, alpha_note, ha="left", va="bottom", fontsize=8.5, color="#444444")
    if not publication and footnote:
        fig.text(0.01, 0.02, footnote, ha="left", va="bottom", fontsize=8, color="#444444")
    bottom_margin = 0.16 if (not publication and footnote and alpha_note) else (0.12 if (not publication and (footnote or alpha_note)) else 0.04)
    fig.tight_layout(rect=(0.0, bottom_margin, 1.0, 0.90))
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def selected_targets(source_csvs: list[Path]) -> list[ExportTarget]:
    if not source_csvs:
        return list(EXPORT_TARGETS)
    requested = {path.resolve() for path in source_csvs}
    targets = [target for target in EXPORT_TARGETS if target.source_csv.resolve() in requested]
    missing = sorted(str(path) for path in requested if path not in {target.source_csv.resolve() for target in targets})
    if missing:
        raise FileNotFoundError(f"Unsupported source CSV(s): {', '.join(missing)}")
    return targets


def render_exports(*, output_dir: Path, source_csvs: list[Path]) -> list[Path]:
    rendered_paths: list[Path] = []
    for target in selected_targets(source_csvs):
        if not target.source_csv.exists():
            raise FileNotFoundError(f"Missing source CSV: {target.source_csv}")
        rows = read_csv_rows(target.source_csv)
        for variant in ("raw", "pub"):
            output_pdf = output_dir / f"{target.stem}_{variant}.pdf"
            output_png = output_dir / f"{target.stem}_{variant}.png"
            output_csv = output_dir / f"{target.stem}_{variant}.csv"
            render_figure(
                rows,
                output_pdf=output_pdf,
                output_png=output_png,
                publication=(variant == "pub"),
            )
            write_rows_csv(
                output_csv,
                export_rows(
                    rows,
                    export_relpath=str(output_pdf.relative_to(REPO_ROOT)),
                    variant=variant,
                ),
            )
            rendered_paths.extend([output_pdf, output_png, output_csv])
    return rendered_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create raw and publication Spearman rho exports for the breadth-topic partner-gap post-train v4 surface."
    )
    parser.add_argument(
        "--source-csv",
        type=Path,
        action="append",
        default=[],
        help="Optional source CSV path(s) to limit regeneration; defaults to the breadth-topic train/test filtered and unfiltered Spearman CSVs.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    for output_path in render_exports(
        output_dir=args.output_dir.resolve(),
        source_csvs=[path.resolve() for path in args.source_csv],
    ):
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())