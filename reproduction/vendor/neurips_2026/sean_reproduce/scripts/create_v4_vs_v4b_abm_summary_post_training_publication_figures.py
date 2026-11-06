#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neurips_2026.sean_reproduce.scripts.create_v4_vs_v4b_abm_summary_post_training_report import (  # noqa: E402
    PARTITION_SPECS,
    SHARED_ABM_BASELINE_MODELS,
    VERSION_SPECS,
)
from neurips_2026.sean_reproduce.scripts.create_v4_vs_v4b_summary_post_training_report import (  # noqa: E402
    get_v2_display_model_order,
    maybe_adjust_bar_model_order,
)
from neurips_2026.sean_reproduce.scripts.create_v4_vs_v4b_summary_base_model_report import (  # noqa: E402
    METRIC_GROUPS,
    VERSION_SPECS as BAR_VERSION_SPECS,
    UNIFORM_LB_MODEL,
    as_float_or_nan,
    build_page_figure,
    read_csv_rows,
    write_comparison_csv,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "neurips_2026" / "sean_reproduce" / "v4_pub" / "abm" / "summary_post_training"
PANEL_TITLE_MAP = {
    "Message-level": "Single-Opinion Prediction",
    "Individual-level": "Individual Opinion-Update Prediction",
    "Group-level": "Group Opinion Convergence Prediction",
}
LEGEND_LABEL_MAP = {
    "v4": "Full Dynamics Simulation",
    "v4_b": "Next Message Prediction",
}
PUBLICATION_SHORT_LABELS = {
    "gpt-5-mini": "gpt-5-mini",
    "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
    "gemini-2.5-flash-lite": "gemini-2.5-flash",
    "Mistral-7B-Instruct-v0.3": "Mistral-7B",
    "Qwen2.5-32B-Instruct": "Qwen2.5-32B",
    "Llama-3.1-70B-Instruct": "Llama-3.1-70B",
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "DeGroot fit": "DeGroot fit",
}
PUBLICATION_TITLE_FONTSIZE = 19
PUBLICATION_AXIS_LABEL_FONTSIZE = 14
PUBLICATION_TICK_FONTSIZE = 12.5
PUBLICATION_XTICK_FONTSIZE = 15
PUBLICATION_VALUE_FONTSIZE = 9.5
PUBLICATION_SECTION_FONTSIZE = 21
PUBLICATION_LEGEND_FONTSIZE = 14.5
FAMILY_SPEC = {
    "report_stem": "report_bar_posttrain_models_vs_all_degroot",
    "csv_prefix": "models_vs_all_",
    "model_set": "post-training-v4-vs-all",
    "page_title_prefix": "Post-Train Models vs All DeGroot ABM",
    "variant_suffix": "_unfiltered_msg_and_ind",
}


def _add_raw_alpha_footnote(fig: object, *, rows_by_version: dict[str, list[dict[str, str]]]) -> None:
    alpha_row = None
    for version_key in ("v4", "v4_b"):
        for row in rows_by_version[version_key]:
            alpha_public = str(row.get("abm_alpha_public", "")).strip()
            alpha_private = str(row.get("abm_alpha_private", "")).strip()
            if alpha_public or alpha_private:
                alpha_row = row
                break
        if alpha_row is not None:
            break

    if alpha_row is None:
        return

    alpha_public = str(alpha_row.get("abm_alpha_public", "")).strip()
    alpha_private = str(alpha_row.get("abm_alpha_private", "")).strip()
    fit_scope = str(alpha_row.get("abm_fit_scope", "")).strip()
    if not alpha_public and not alpha_private:
        return

    footnote_parts = ["Fitted DeGroot alpha"]
    if fit_scope:
        footnote_parts.append(f"fit scope={fit_scope}")
    if alpha_public:
        footnote_parts.append(f"alpha_public={float(alpha_public):.3f}")
    if alpha_private:
        footnote_parts.append(f"alpha_private={float(alpha_private):.3f}")

    if hasattr(fig, "text"):
        fig.text(
            0.01,
            0.095,
            " | ".join(footnote_parts),
            ha="left",
            va="bottom",
            fontsize=8,
            color="#555555",
        )


def _annotate_bar_values(fig: object, *, rows_by_version: dict[str, list[dict[str, str]]], model_order: list[str]) -> None:
    axes = list(getattr(fig, "axes", []))
    if len(axes) < len(METRIC_GROUPS):
        return

    lookup_by_version = {
        version_key: {str(row.get("model", "")).strip(): row for row in version_rows}
        for version_key, version_rows in rows_by_version.items()
    }
    num_models = len(model_order)
    public_centers = list(range(num_models))
    private_centers = [position + num_models + 1.6 for position in public_centers]
    shared_models = {str(model_name).strip() for model_name in SHARED_ABM_BASELINE_MODELS}

    for axis, (_, _, field_pairs, _, _) in zip(axes, METRIC_GROUPS):
        y_min, y_max = axis.get_ylim()
        y_offset = (y_max - y_min) * 0.035
        for centers, field_pair in (
            (public_centers, field_pairs[0]),
            (private_centers, field_pairs[1]),
        ):
            metric_field = field_pair[1]
            for version_key in ("v4", "v4_b"):
                offset = float(BAR_VERSION_SPECS[version_key]["offset"])
                version_lookup = lookup_by_version[version_key]
                for center, model_name in zip(centers, model_order):
                    if model_name in shared_models:
                        continue
                    row = version_lookup.get(model_name)
                    if row is None:
                        continue
                    value = as_float_or_nan(row.get(metric_field))
                    if value <= 0:
                        continue
                    ci_high = as_float_or_nan(row.get(f"{metric_field}_ci_high"))
                    top_y = max(value, ci_high if ci_high == ci_high else value)
                    axis.text(
                        center + offset,
                        top_y + y_offset,
                        f"{value:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=PUBLICATION_VALUE_FONTSIZE,
                        color="#333333",
                        rotation=90,
                        zorder=7,
                    )

            for center, model_name in zip(centers, model_order):
                if model_name not in shared_models:
                    continue
                shared_row = lookup_by_version["v4"].get(model_name) or lookup_by_version["v4_b"].get(model_name)
                if shared_row is None:
                    continue
                value = as_float_or_nan(shared_row.get(metric_field))
                if value <= 0:
                    continue
                ci_high = as_float_or_nan(shared_row.get(f"{metric_field}_ci_high"))
                top_y = max(value, ci_high if ci_high == ci_high else value)
                axis.text(
                    center,
                    top_y + y_offset,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=PUBLICATION_VALUE_FONTSIZE,
                    color="#333333",
                    rotation=90,
                    zorder=7,
                )


def _publication_model_label(model_name: str) -> str:
    normalized = str(model_name).strip()
    if normalized in PUBLICATION_SHORT_LABELS:
        return PUBLICATION_SHORT_LABELS[normalized]
    if normalized.startswith("ft:Llama-3.1-8B-Instruct-SFT-") or normalized.startswith("ft:Llama-3.1-8B-SFT-"):
        return "Llama SFT"
    return normalized


def _tighten_y_limits(fig: object, *, rows_by_version: dict[str, list[dict[str, str]]], model_order: list[str], has_value_labels: bool) -> None:
    axes = list(getattr(fig, "axes", []))
    if len(axes) < len(METRIC_GROUPS):
        return

    lookup_v4 = {str(row.get("model", "")).strip(): row for row in rows_by_version["v4"]}
    lookup_v4_b = {str(row.get("model", "")).strip(): row for row in rows_by_version["v4_b"]}
    uniform_row = lookup_v4.get(UNIFORM_LB_MODEL)
    reference_row = next((lookup_v4.get(model_name) for model_name in model_order if lookup_v4.get(model_name) is not None), None)
    shared_models = {str(model_name).strip() for model_name in SHARED_ABM_BASELINE_MODELS}

    for axis_index, (axis, (_, _, field_pairs, _, bound_pairs)) in enumerate(zip(axes, METRIC_GROUPS)):
        y_min, y_max = axis.get_ylim()
        top_values: list[float] = []

        for field_pair in field_pairs:
            metric_field = field_pair[1]
            for version_key, version_rows in rows_by_version.items():
                for row in version_rows:
                    model_name = str(row.get("model", "")).strip()
                    if model_name in shared_models and version_key == "v4_b":
                        continue
                    value = as_float_or_nan(row.get(metric_field))
                    ci_high = as_float_or_nan(row.get(f"{metric_field}_ci_high"))
                    if value == value:
                        top_values.append(value)
                    if ci_high == ci_high:
                        top_values.append(ci_high)

        for bound_fields in bound_pairs:
            lb_field, ub_primary_field, ub_secondary_field = bound_fields
            if uniform_row is not None:
                uniform_value = as_float_or_nan(uniform_row.get(lb_field.replace("lb_", "")))
                if uniform_value == uniform_value:
                    top_values.append(uniform_value)
            if reference_row is not None:
                for bound_field in (ub_primary_field, ub_secondary_field):
                    bound_value = as_float_or_nan(reference_row.get(bound_field))
                    if bound_value == bound_value:
                        top_values.append(bound_value)

        if not top_values:
            continue

        top_value = max(top_values)
        top_padding = max(0.12, (top_value - y_min) * (0.18 if has_value_labels else 0.12))
        new_y_max = top_value + top_padding
        if new_y_max >= y_max:
            continue

        axis.set_ylim(y_min, new_y_max)

        if axis_index == 0:
            label_y = new_y_max - max(0.06, (new_y_max - y_min) * 0.10)
            for text_artist in axis.texts:
                if text_artist.get_text().strip() in {"Public", "Private"}:
                    text_artist.set_y(label_y)


def _load_panel(panel_name: str) -> tuple[str, dict[str, list[dict[str, str]]], list[str]]:
    page_title = next(title for key, title in PARTITION_SPECS if key == panel_name)
    source_stem = f"{FAMILY_SPEC['csv_prefix']}{panel_name}"
    rows_by_version = {
        "v4": read_csv_rows(Path(VERSION_SPECS["v4"]["csv_dir"]) / f"raw_metrics_{source_stem}_v4{FAMILY_SPEC['variant_suffix']}.csv"),
        "v4_b": read_csv_rows(Path(VERSION_SPECS["v4_b"]["csv_dir"]) / f"raw_metrics_{source_stem}_v4_b{FAMILY_SPEC['variant_suffix']}.csv"),
    }
    combined_rows = rows_by_version["v4"] + rows_by_version["v4_b"]
    display_model_order = get_v2_display_model_order(combined_rows, include_human=False)
    model_order = list(
        maybe_adjust_bar_model_order(
            display_model_order,
            partition_name=panel_name,
            model_set=str(FAMILY_SPEC["model_set"]),
        )
    )
    return f"{FAMILY_SPEC['page_title_prefix']} - {page_title}", rows_by_version, model_order


def _clean_publication_figure(fig: object, *, model_order: list[str]) -> None:
    if hasattr(fig, "_suptitle") and getattr(fig, "_suptitle") is not None:
        fig._suptitle.remove()

    for text_artist in list(getattr(fig, "texts", [])):
        text_artist.remove()

    for legend in list(getattr(fig, "legends", [])):
        legend.set_bbox_to_anchor((0.5, 0.095), transform=fig.transFigure)
        for text_artist in legend.get_texts():
            original_text = text_artist.get_text().strip()
            if original_text in LEGEND_LABEL_MAP:
                text_artist.set_text(LEGEND_LABEL_MAP[original_text])
            text_artist.set_fontsize(PUBLICATION_LEGEND_FONTSIZE)
        if legend.get_title() is not None:
            legend.get_title().set_fontsize(PUBLICATION_LEGEND_FONTSIZE)

    for axis_index, axis in enumerate(getattr(fig, "axes", [])):
        title = axis.get_title(loc="left")
        if title in PANEL_TITLE_MAP:
            axis.set_title(PANEL_TITLE_MAP[title], loc="left", fontweight="bold", fontsize=PUBLICATION_TITLE_FONTSIZE)

        for text_artist in list(axis.texts):
            if text_artist.get_text().strip() in {"Public", "Private"}:
                text_artist.remove()

        axis.grid(False)
        for gridline in axis.yaxis.get_gridlines():
            gridline.set_visible(False)

        for patch in list(getattr(axis, "patches", [])):
            alpha = patch.get_alpha()
            if alpha is not None and alpha <= 0.05:
                patch.remove()

        axis.set_ylabel(axis.get_ylabel(), fontsize=PUBLICATION_AXIS_LABEL_FONTSIZE)
        axis.tick_params(axis="y", labelsize=PUBLICATION_TICK_FONTSIZE)
        axis.tick_params(axis="x", labelsize=PUBLICATION_XTICK_FONTSIZE)

        if axis_index == len(getattr(fig, "axes", [])) - 1:
            short_labels = [_publication_model_label(model_name) for model_name in model_order]
            axis.set_xticklabels(short_labels + short_labels, rotation=90, ha="center", va="top")
            for tick_label in axis.get_xticklabels():
                tick_label.set_fontsize(PUBLICATION_XTICK_FONTSIZE)

    fig.set_size_inches(18.5, 12.0, forward=True)
    fig.text(0.285, 0.905, "Public Opinion", ha="center", va="center", fontsize=PUBLICATION_SECTION_FONTSIZE, fontweight="bold")
    fig.text(0.77, 0.905, "Private Opinion", ha="center", va="center", fontsize=PUBLICATION_SECTION_FONTSIZE, fontweight="bold")

    if hasattr(fig, "subplots_adjust"):
        fig.subplots_adjust(left=0.095, right=0.995, bottom=0.245, top=0.86, hspace=0.36)


def _save_figure(fig: object, *, output_dir: Path, stem: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    return pdf_path, png_path


def _save_report_csv(*, output_dir: Path, stem: str, rows_by_version: dict[str, list[dict[str, str]]]) -> Path:
    csv_path = output_dir / f"{stem}.csv"
    write_comparison_csv(csv_path, rows_by_version)
    return csv_path


def render_publication_exports(*, output_dir: Path) -> list[Path]:
    saved_paths: list[Path] = []
    figures_to_close: list[object] = []

    for partition_name, _ in PARTITION_SPECS:
        page_title, rows_by_version, model_order = _load_panel(partition_name)
        stem_base = f"report_bar_posttrain_models_vs_all_degroot_{partition_name}_v4_vs_4b_unfiltered_msg_and_ind"

        raw_figure = build_page_figure(
            page_title=page_title,
            rows_by_version=rows_by_version,
            hide_count_annotations=False,
            model_order=model_order,
            shared_models=SHARED_ABM_BASELINE_MODELS,
        )
        _add_raw_alpha_footnote(raw_figure, rows_by_version=rows_by_version)
        _annotate_bar_values(raw_figure, rows_by_version=rows_by_version, model_order=model_order)
        _tighten_y_limits(raw_figure, rows_by_version=rows_by_version, model_order=model_order, has_value_labels=True)
        raw_paths = _save_figure(raw_figure, output_dir=output_dir, stem=f"{stem_base}_raw")
        saved_paths.extend(raw_paths)
        figures_to_close.append(raw_figure)

        pub_figure = build_page_figure(
            page_title=page_title,
            rows_by_version=rows_by_version,
            hide_count_annotations=True,
            model_order=model_order,
            shared_models=SHARED_ABM_BASELINE_MODELS,
        )
        _clean_publication_figure(pub_figure, model_order=model_order)
        _annotate_bar_values(pub_figure, rows_by_version=rows_by_version, model_order=model_order)
        _tighten_y_limits(pub_figure, rows_by_version=rows_by_version, model_order=model_order, has_value_labels=True)
        pub_paths = _save_figure(pub_figure, output_dir=output_dir, stem=f"{stem_base}_pub")
        saved_paths.extend(pub_paths)
        saved_paths.append(_save_report_csv(output_dir=output_dir, stem=f"{stem_base}_pub", rows_by_version=rows_by_version))
        figures_to_close.append(pub_figure)

    try:
        import matplotlib.pyplot as plt  # noqa: WPS433

        for figure in figures_to_close:
            plt.close(figure)
    except Exception:
        pass

    return saved_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create raw and publication-cleaned exports for the v4-vs-v4_b ABM summary post-training vs-all unfiltered comparison family."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    for output_path in render_publication_exports(output_dir=args.output_dir.resolve()):
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())