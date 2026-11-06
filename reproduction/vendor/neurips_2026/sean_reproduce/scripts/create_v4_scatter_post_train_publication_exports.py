#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

try:
    from PIL import Image, ImageChops
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Pillow is required to render publication scatter exports: {exc}")

try:
    from pypdf import PdfReader, PdfWriter
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"pypdf is required to extract scatter publication pages: {exc}")


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
)
DEFAULT_RENDER_DPI = 240
DEFAULT_CROP_PADDING = 16

SOURCE_REPORT = (
    REPO_ROOT
    / "neurips_2026"
    / "sean_reproduce"
    / "v4"
    / "scatter_plot"
    / "post_train"
    / "report_breadth_group_post_train_partner_initial_gap_vs_final_change_scatter_v4_unfiltered.pdf"
)
SOURCE_CSV = SOURCE_REPORT.with_suffix(".csv")


@dataclass(frozen=True)
class ExportSpec:
    subset: str
    source_page: int
    stem: str


@dataclass(frozen=True)
class ExportSuite:
    family_key: str
    suite_label: str
    source_report: Path
    export_specs: tuple[ExportSpec, ...]


EXPORT_SUITES = (
    ExportSuite(
        family_key="breadth_group",
        suite_label="Breadth Group Post-Train",
        source_report=SOURCE_REPORT,
        export_specs=(
            ExportSpec(
                subset="train",
                source_page=2,
                stem="report_breadth_group_train_post_train_partner_initial_gap_vs_final_change_scatter_v4_unfiltered",
            ),
            ExportSpec(
                subset="test",
                source_page=3,
                stem="report_breadth_group_test_post_train_partner_initial_gap_vs_final_change_scatter_v4_unfiltered",
            ),
        ),
    ),
    ExportSuite(
        family_key="breadth_topic",
        suite_label="Breadth Topic Post-Train",
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
                source_page=2,
                stem="report_breadth_topic_train_post_train_partner_initial_gap_vs_final_change_scatter_v4",
            ),
            ExportSpec(
                subset="test",
                source_page=3,
                stem="report_breadth_topic_test_post_train_partner_initial_gap_vs_final_change_scatter_v4",
            ),
        ),
    ),
    ExportSuite(
        family_key="breadth_topic",
        suite_label="Breadth Topic Post-Train",
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
                source_page=2,
                stem="report_breadth_topic_train_post_train_partner_initial_gap_vs_final_change_scatter_v4_unfiltered",
            ),
            ExportSpec(
                subset="test",
                source_page=3,
                stem="report_breadth_topic_test_post_train_partner_initial_gap_vs_final_change_scatter_v4_unfiltered",
            ),
        ),
    ),
)

PLOT_PAIR = next(
    plot_pair
    for plot_pair in scatter_mod.MERGED_PLOT_PAIRS
    if plot_pair.report_suffix == "partner_initial_gap_vs_final_change"
)

PANEL_KEY_TO_SPEC = {
    "public": scatter_mod.PLOT_SPEC_BY_KEY[PLOT_PAIR.public_spec_key],
    "private": scatter_mod.PLOT_SPEC_BY_KEY[PLOT_PAIR.private_spec_key],
}

PANEL_TITLES = {
    "train": {"public": "Public Train", "private": "Private Train"},
    "test": {"public": "Public Test", "private": "Private Test"},
}

PAGE_TITLES = {
    "train": "Train Scatter Panels",
    "test": "Test Scatter Panels",
}

def extract_single_page_pdf(source_report: Path, *, page_number: int, output_pdf: Path) -> None:
    reader = PdfReader(str(source_report))
    page_index = page_number - 1
    if page_index < 0 or page_index >= len(reader.pages):
        raise ValueError(f"page {page_number} is out of range for {source_report}")
    writer = PdfWriter()
    writer.add_page(reader.pages[page_index])
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with output_pdf.open("wb") as handle:
        writer.write(handle)


def render_pdf_page_to_png(source_report: Path, *, page_number: int, output_png: Path, dpi: int) -> None:
    ghostscript = shutil.which("gs")
    if ghostscript is None:
        raise RuntimeError("Ghostscript executable 'gs' was not found in PATH")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ghostscript,
            "-dSAFER",
            "-dBATCH",
            "-dNOPAUSE",
            "-sDEVICE=png16m",
            f"-r{dpi}",
            f"-dFirstPage={page_number}",
            f"-dLastPage={page_number}",
            f"-sOutputFile={output_png}",
            str(source_report),
        ],
        check=True,
    )


def crop_whitespace(image: Image.Image, *, padding: int) -> Image.Image:
    rgb_image = image.convert("RGB")
    background = Image.new("RGB", rgb_image.size, (255, 255, 255))
    difference = ImageChops.difference(rgb_image, background)
    bbox = difference.getbbox()
    if bbox is None:
        return rgb_image.copy()
    left, top, right, bottom = bbox
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(rgb_image.width, right + padding)
    bottom = min(rgb_image.height, bottom + padding)
    return rgb_image.crop((left, top, right, bottom))


def save_image_pdf(image: Image.Image, output_pdf: Path, *, dpi: int) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_pdf, "PDF", resolution=float(dpi))


def load_report_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows_csv(csv_path: Path, rows: list[dict[str, str]]) -> Path:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for fieldname in row.keys():
            if fieldname in seen:
                continue
            seen.add(fieldname)
            fieldnames.append(fieldname)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


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

    with fit_csv_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            alpha_public = str(row.get("fit_alpha_public", "")).strip()
            alpha_private = str(row.get("fit_alpha_private", "")).strip()
            if not alpha_public or not alpha_private:
                continue
            return float(alpha_public), float(alpha_private)
    return None


def export_rows(
    rows: list[dict[str, str]],
    *,
    subset: str,
    export_relpath: str,
    export_variant: str,
) -> list[dict[str, str]]:
    exported_rows: list[dict[str, str]] = []
    for row in rows:
        if str(row.get("subset", "")).strip() != subset:
            continue
        exported_rows.append(
            {
                **row,
                "export_variant": export_variant,
                "export_relpath": export_relpath,
            }
        )
    return exported_rows


def float_or_nan(value: str) -> float:
    text = str(value).strip()
    if text == "":
        return float("nan")
    return float(text)


def square_scatter_limits(
    x_lim: tuple[float, float],
    y_lim: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    x_center = (x_lim[0] + x_lim[1]) / 2.0
    y_center = (y_lim[0] + y_lim[1]) / 2.0
    half_span = max(x_lim[1] - x_lim[0], y_lim[1] - y_lim[0]) / 2.0
    return (
        (x_center - half_span, x_center + half_span),
        (y_center - half_span, y_center + half_span),
    )


def legend_order_for_subset(rows: list[dict[str, str]], *, subset: str) -> list[str]:
    ordered_entities: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if str(row.get("row_type", "")).strip() != "correlation_summary":
            continue
        if str(row.get("subset", "")).strip() != subset:
            continue
        entity = str(row.get("entity", "")).strip()
        if not entity or entity in seen:
            continue
        seen.add(entity)
        ordered_entities.append(entity)
    non_human = [entity for entity in ordered_entities if entity != scatter_mod.HUMAN_MODEL]
    if scatter_mod.HUMAN_MODEL in seen:
        return [*non_human, scatter_mod.HUMAN_MODEL]
    return non_human


def entity_points_for_subset(rows: list[dict[str, str]], *, subset: str, panel_key: str) -> dict[str, list[tuple[float, float]]]:
    entity_points: dict[str, list[tuple[float, float]]] = {}
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
        entity_points.setdefault(entity, []).append(
            (
                float_or_nan(row.get("x_value", "")),
                float_or_nan(row.get("y_value", "")),
            )
        )
    return {
        entity: [(x_value, y_value) for x_value, y_value in points if x_value == x_value and y_value == y_value]
        for entity, points in entity_points.items()
    }


def correlation_footnote(
    rows: list[dict[str, str]],
    *,
    subset: str,
    panel_key: str,
    legend_order: list[str],
) -> str:
    summary_by_entity: dict[str, dict[str, str]] = {}
    for row in rows:
        if str(row.get("row_type", "")).strip() != "correlation_summary":
            continue
        if str(row.get("subset", "")).strip() != subset:
            continue
        if str(row.get("panel_key", "")).strip() != panel_key:
            continue
        entity = str(row.get("entity", "")).strip()
        if entity:
            summary_by_entity[entity] = row

    footnote_entries: list[str] = []
    for entity in legend_order:
        summary_row = summary_by_entity.get(entity)
        if summary_row is None:
            continue
        full_name = str(summary_row.get("model_dir", "")).strip() or entity
        rho = float_or_nan(summary_row.get("spearman_rho", ""))
        kendall = float_or_nan(summary_row.get("kendall_tau_c", ""))
        n_points = str(summary_row.get("n_points", "")).strip() or "n/a"
        rho_text = "nan" if rho != rho else f"{rho:.3f}"
        kendall_text = "nan" if kendall != kendall else f"{kendall:.3f}"
        footnote_entries.append(f"{full_name}: n={n_points}, rho={rho_text}, tau_c={kendall_text}")
    return " | ".join(footnote_entries)


def render_publication_axis(
    ax: object,
    *,
    spec: object,
    entity_points: dict[str, list[tuple[float, float]]],
    legend_order: list[str],
    x_lim: tuple[float, float],
    y_lim: tuple[float, float],
    page_index: int,
    panel_index: int,
    panel_title: str,
    show_grid: bool,
    footnote_text: str | None,
) -> list[object]:
    legend_handles: list[object] = []
    draw_order = [entity for entity in legend_order if entity != scatter_mod.HUMAN_MODEL] + [scatter_mod.HUMAN_MODEL]

    for entity_index, entity in enumerate(draw_order):
        points = entity_points.get(entity, [])
        if not points:
            continue
        color = scatter_mod.get_model_color(entity, "#777777")
        is_human = entity == scatter_mod.HUMAN_MODEL
        is_degroot = scatter_mod.is_degroot_model(entity)
        is_primary_highlight = scatter_mod.is_primary_scatter_highlight(entity)

        if is_human:
            point_alpha = 0.18
            point_zorder = 4
            line_width = scatter_mod.LINE_WIDTH + 0.4
            line_alpha = 0.98
            line_zorder = 5
            line_style = "-"
        elif is_degroot:
            point_alpha = 0.08
            point_zorder = 3
            line_width = scatter_mod.LINE_WIDTH + 0.1
            line_alpha = 0.95
            line_zorder = 4
            line_style = "--"
        elif is_primary_highlight:
            point_alpha = 0.12
            point_zorder = 3
            line_width = scatter_mod.LINE_WIDTH + 0.2
            line_alpha = 0.96
            line_zorder = 4
            line_style = "-"
        else:
            point_alpha = 0.05
            point_zorder = 2
            line_width = 1.5
            line_alpha = 0.4
            line_zorder = 3
            line_style = "--"

        x_plot, y_plot = scatter_mod.jitter_points(points, seed=page_index * 1000 + panel_index * 100 + entity_index)
        ax.scatter(
            x_plot,
            y_plot,
            s=scatter_mod.POINT_SIZE,
            alpha=point_alpha,
            color=color,
            edgecolors="none",
            zorder=point_zorder,
        )

        line_fit = scatter_mod.compute_line_fit(points)
        if line_fit is not None:
            slope, intercept = line_fit
            raw_x = [x_value for x_value, _ in points]
            x0 = min(raw_x)
            x1 = max(raw_x)
            line_handle = ax.plot(
                [x0, x1],
                [intercept + slope * x0, intercept + slope * x1],
                color=color,
                linewidth=line_width,
                linestyle=line_style,
                alpha=line_alpha,
                zorder=line_zorder,
                label=scatter_mod.get_model_display_label(entity),
            )[0]
            legend_handles.append(line_handle)

    if spec.x_is_difference:
        ax.axvline(0.0, color="#7f7f7f", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axhline(0.0, color="#7f7f7f", linestyle="--", linewidth=1.0, alpha=0.8)
    if show_grid:
        ax.grid(True, alpha=0.22)
    else:
        ax.grid(False)
    ax.set_axisbelow(True)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(spec.x_label)
    ax.set_ylabel(spec.y_label)
    ax.set_title(panel_title, loc="left", fontweight="bold")
    ax.tick_params(axis="both", labelsize=8)
    if footnote_text:
        footnote_lines = textwrap.wrap(footnote_text, width=72)
        ax.text(
            0.0,
            -0.19,
            "\n".join(footnote_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            color="#444444",
        )
    return legend_handles


def render_publication_page(
    rows: list[dict[str, str]],
    *,
    subset: str,
    suite_label: str,
    show_grid: bool,
    degroot_alpha_note: str | None,
    include_footnotes: bool,
    output_png: Path,
    output_pdf: Path,
    dpi: int,
) -> None:
    legend_order = legend_order_for_subset(rows, subset=subset)
    public_points = entity_points_for_subset(rows, subset=subset, panel_key="public")
    private_points = entity_points_for_subset(rows, subset=subset, panel_key="private")
    public_spec = PANEL_KEY_TO_SPEC["public"]
    private_spec = PANEL_KEY_TO_SPEC["private"]
    public_limits = square_scatter_limits(*scatter_mod.compute_scatter_limits(public_spec, [public_points]))
    private_limits = square_scatter_limits(*scatter_mod.compute_scatter_limits(private_spec, [private_points]))

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 8.8))
    public_handles = render_publication_axis(
        axes[0],
        spec=public_spec,
        entity_points=public_points,
        legend_order=[entity for entity in legend_order if public_points.get(entity)],
        x_lim=public_limits[0],
        y_lim=public_limits[1],
        page_index=1 if subset == "train" else 2,
        panel_index=0,
        panel_title=PANEL_TITLES[subset]["public"],
        show_grid=show_grid,
        footnote_text=(
            correlation_footnote(rows, subset=subset, panel_key="public", legend_order=legend_order)
            if include_footnotes
            else None
        ),
    )
    private_handles = render_publication_axis(
        axes[1],
        spec=private_spec,
        entity_points=private_points,
        legend_order=[entity for entity in legend_order if private_points.get(entity)],
        x_lim=private_limits[0],
        y_lim=private_limits[1],
        page_index=1 if subset == "train" else 2,
        panel_index=1,
        panel_title=PANEL_TITLES[subset]["private"],
        show_grid=show_grid,
        footnote_text=(
            correlation_footnote(rows, subset=subset, panel_key="private", legend_order=legend_order)
            if include_footnotes
            else None
        ),
    )

    legend_handle_map = {handle.get_label(): handle for handle in [*public_handles, *private_handles]}
    fig.legend(
        [legend_handle_map[label] for label in legend_handle_map],
        list(legend_handle_map.keys()),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.19 if include_footnotes else (0.15 if degroot_alpha_note else 0.045)),
        ncol=5,
        frameon=True,
        fontsize=9,
        columnspacing=1.2,
        handlelength=2.3,
    )
    if degroot_alpha_note:
        fig.text(
            0.5,
            0.08,
            degroot_alpha_note,
            ha="center",
            va="center",
            fontsize=9,
            color="#444444",
        )
    fig.suptitle(
        f"{PAGE_TITLES[subset]}\n{suite_label}",
        x=0.01,
        y=0.985,
        ha="left",
        va="top",
        fontweight="bold",
    )
    bottom_rect = 0.31 if include_footnotes else (0.19 if degroot_alpha_note else 0.22)
    fig.tight_layout(rect=(0.0, bottom_rect, 1.0, 0.91))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def render_exports(*, output_dir: Path, dpi: int, crop_padding: int) -> list[Path]:
    rendered_paths: list[Path] = []

    for export_suite in EXPORT_SUITES:
        source_report = export_suite.source_report
        source_csv = source_report.with_suffix(".csv")
        if not source_report.exists():
            raise FileNotFoundError(f"Missing source scatter report: {source_report}")
        if not source_csv.exists():
            raise FileNotFoundError(f"Missing source scatter CSV: {source_csv}")

        source_rows = load_report_rows(source_csv)
        fit_alphas_by_subset = {
            export_spec.subset: load_degroot_fit_alphas(family_key=export_suite.family_key, subset=export_spec.subset)
            for export_spec in export_suite.export_specs
        }

        for export_spec in export_suite.export_specs:
            raw_pdf = output_dir / f"{export_spec.stem}_raw.pdf"
            raw_png = output_dir / f"{export_spec.stem}_raw.png"
            raw_csv = output_dir / f"{export_spec.stem}_raw.csv"

            raw_fit_alphas = fit_alphas_by_subset[export_spec.subset]
            raw_fit_note = None
            if raw_fit_alphas is not None:
                alpha_public, alpha_private = raw_fit_alphas
                raw_fit_note = f"DeGroot fit alpha: public={alpha_public:.3f}, private={alpha_private:.3f}"
            render_publication_page(
                source_rows,
                subset=export_spec.subset,
                suite_label=export_suite.suite_label,
                show_grid=True,
                degroot_alpha_note=raw_fit_note,
                include_footnotes=True,
                output_png=raw_png,
                output_pdf=raw_pdf,
                dpi=dpi,
            )
            write_rows_csv(
                raw_csv,
                export_rows(
                    source_rows,
                    subset=export_spec.subset,
                    export_relpath=str(raw_pdf.relative_to(REPO_ROOT)),
                    export_variant="raw",
                ),
            )

            pub_pdf = output_dir / f"{export_spec.stem}_pub.pdf"
            pub_png = output_dir / f"{export_spec.stem}_pub.png"
            pub_csv = output_dir / f"{export_spec.stem}_pub.csv"
            render_publication_page(
                source_rows,
                subset=export_spec.subset,
                suite_label=export_suite.suite_label,
                show_grid=False,
                degroot_alpha_note=None,
                include_footnotes=False,
                output_png=pub_png,
                output_pdf=pub_pdf,
                dpi=dpi,
            )
            with Image.open(pub_png) as pub_image:
                cropped = crop_whitespace(pub_image, padding=crop_padding)
                cropped.save(pub_png)
                save_image_pdf(cropped, pub_pdf, dpi=dpi)

            write_rows_csv(
                pub_csv,
                export_rows(
                    source_rows,
                    subset=export_spec.subset,
                    export_relpath=str(pub_pdf.relative_to(REPO_ROOT)),
                    export_variant="pub",
                ),
            )

            rendered_paths.extend([raw_pdf, raw_png, raw_csv, pub_pdf, pub_png, pub_csv])

    return rendered_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create raw and publication scatter exports for the v4 breadth-group post-train partner-gap unfiltered report."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=DEFAULT_RENDER_DPI)
    parser.add_argument("--crop-padding", type=int, default=DEFAULT_CROP_PADDING)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    for output_path in render_exports(
        output_dir=args.output_dir.resolve(),
        dpi=int(args.dpi),
        crop_padding=int(args.crop_padding),
    ):
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())