#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
SEAN_REPRODUCE_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]
TRANSITION_SRC = REPO_ROOT / "neurips_2026" / "leo" / "transition" / "src"

for path in (REPO_ROOT, SCRIPT_DIR, TRANSITION_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_transition_layout as layout_mod  # noqa: E402
from neurips_2026.sean_reproduce.result_version_config import v4_family_sft_model  # noqa: E402


DEFAULT_SOURCE_ROOT = SEAN_REPRODUCE_ROOT / "v4" / "abm" / "transition" / "tweet_likert"
DEFAULT_OUTPUT_ROOT = SEAN_REPRODUCE_ROOT / "v4_pub" / "abm" / "transition"
DEFAULT_OUTPUT_FILENAME = "report_transition_ind_public_private_depth_group_train_selected_models_v4_unfiltered_pub.pdf"
DEFAULT_PRIVATE_SOURCE_ROOT = SEAN_REPRODUCE_ROOT / "v4" / "abm" / "transition" / "opinion_stance"
DEFAULT_METRIC_VERSION = "v4"

FIGURE_WIDTH = 21.0
FIGURE_HEIGHT = 9.7
SUPTITLE_FONTSIZE = 18
ROW_LABEL_FONTSIZE = 17
AXIS_LABEL_FONTSIZE = 11
TICK_LABEL_FONTSIZE = 13
TITLE_FONT_BONUS = 3.0
CELL_TEXT_FONTSIZE_PUB = 11
CELL_TEXT_FONTSIZE_RAW = 10
PANEL_INFO_FONTSIZE_RAW = 10
PANEL_INFO_FONTSIZE_PUB = 12
COLORBAR_LABEL_FONTSIZE = 12
COLORBAR_TICK_FONTSIZE = 11
FOOTNOTE_FONTSIZE = 10


def build_panel_specs(collection_type: str, split_type: str) -> list[tuple[str, str, tuple[int, int], str | None]]:
    sft_model_id = v4_family_sft_model(collection_type, split_type)
    return [
        ("human", "human", (0, 0), None),
        ("base: Llama-3.1-8B-Instruct", "Llama-3.1-8B-Instruct", (0, 2), "Llama-3.1-8B\nInstruct"),
        (f"sft: {sft_model_id}", sft_model_id.replace(":", "_"), (1, 0), "LLaMA SFT"),
        ("gpt-5-mini", "gpt-5-mini", (1, 1), None),
        ("DeGroot fit", "DeGroot_fit", (1, 2), "DeGroot fit"),
    ]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a quick-test v4_pub ABM transition publication panel.")
    parser.add_argument("--public-source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--private-source-root", type=Path, default=DEFAULT_PRIVATE_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--collection-type", choices=["breadth", "depth"], default="depth")
    parser.add_argument("--split-type", choices=["group", "topic"], default="group")
    parser.add_argument("--partition-type", choices=["train", "test"], default="train")
    parser.add_argument("--filtered-state", choices=["filtered", "unfiltered"], default="unfiltered")
    parser.add_argument("--variant", choices=["raw", "pub", "both"], default="both")
    parser.add_argument("--output-filename", default=None)
    return parser


def default_output_filename(collection_type: str, split_type: str, partition_type: str, filtered_state: str) -> str:
    state_suffix = "_unfiltered" if filtered_state == "unfiltered" else ""
    return f"report_transition_ind_public_private_{collection_type}_{split_type}_{partition_type}_selected_models_v4{state_suffix}_pub.pdf"


def normalize_output_stem(output_filename: str) -> str:
    stem = Path(output_filename).stem
    if stem.endswith("_pub") or stem.endswith("_raw"):
        return stem.rsplit("_", 1)[0]
    return stem


def variant_output_path(output_root: Path, output_stem: str, variant: str) -> Path:
    return (output_root / f"{output_stem}_{variant}").with_suffix(".pdf")


def normalized_frobenius_diff(model_matrix: np.ndarray, human_matrix: np.ndarray) -> float | None:
    if model_matrix.shape != human_matrix.shape:
        return None
    diff_norm = np.linalg.norm(model_matrix - human_matrix, ord="fro")
    human_norm = np.linalg.norm(human_matrix, ord="fro")
    return float(diff_norm if human_norm == 0 else diff_norm / human_norm)


def full_model_identifier(panel_label: str, model_dir: str) -> str:
    if panel_label.startswith("base: ") or panel_label.startswith("sft: "):
        return panel_label.split(": ", 1)[1]
    return layout_mod._restore_table_model_id(model_dir)


def group_count(collection_type: str, split_type: str, partition_type: str, *, value_type: str) -> int:
    prefixes = layout_mod._section_prefixes(
        collection_type,
        split_type,
        partition_type,
        metric_version=DEFAULT_METRIC_VERSION,
        value_type=value_type,
        group_validity_threshold=None,
        use_split_masks=True,
    )
    return len(prefixes)


def degroot_alpha_pair(collection_type: str, model_dir: str) -> tuple[float, float] | None:
    manifest_path = SEAN_REPRODUCE_ROOT / DEFAULT_METRIC_VERSION / "abm" / "transition" / "_eval" / collection_type / "degroot_proxy_manifest.csv"
    if not manifest_path.exists():
        return None

    model_name = layout_mod._restore_table_model_id(model_dir)
    alpha_pairs: set[tuple[float, float]] = set()
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("model_dir", "")).strip() != model_name:
                continue
            alpha_public = str(row.get("alpha_public", "")).strip()
            alpha_private = str(row.get("alpha_private", "")).strip()
            if not alpha_public or not alpha_private:
                continue
            alpha_pairs.add((float(alpha_public), float(alpha_private)))
    if not alpha_pairs:
        return None
    return sorted(alpha_pairs)[0]


def model_footnote(
    panel_specs: list[tuple[str, str, tuple[int, int], str | None]],
    *,
    collection_type: str,
) -> str:
    entries: list[str] = []
    for panel_label, model_dir, _position, display_label_override in panel_specs:
        display_label = display_label_override or layout_mod._display_panel_label(panel_label)
        display_label = display_label.replace("\n", " ")
        full_label = full_model_identifier(panel_label, model_dir)
        if model_dir == "DeGroot_fit":
            alpha_pair = degroot_alpha_pair(collection_type, model_dir)
            if alpha_pair is not None:
                full_label = f"{full_label} (alpha_public={alpha_pair[0]:.3f}, alpha_private={alpha_pair[1]:.3f})"
        entries.append(f"{display_label} = {full_label}")
    return "Full model names: " + " | ".join(entries)


def matrix_path(
    source_root: Path,
    collection_type: str,
    split_type: str,
    partition_type: str,
    model_dir: str,
    *,
    raw_subdir: str,
    matrix_name: str,
) -> Path:
    return (
        source_root
        / "matrices"
        / raw_subdir
        / "tables"
        / f"{collection_type}__{split_type}__{partition_type}__{model_dir}__{matrix_name}.csv"
    )


def draw_publication_heatmap(
    ax,
    matrix: np.ndarray,
    label: str,
    *,
    display_label_override: str | None = None,
    vmax: float,
    value_type: str,
    variant: str,
    panel_info_text: str | None,
):
    labels, x_label, y_label = layout_mod._heatmap_axes_config("ind", value_type)
    total = int(np.asarray(matrix).sum())
    percent_matrix = (matrix / total) if total else np.zeros_like(matrix, dtype=float)
    image = ax.imshow(percent_matrix, cmap=layout_mod.V4_TRANSITION_BLUE_CMAP, vmin=0.0, vmax=vmax)

    axis_len = matrix.shape[0]
    ax.set_xticks(np.arange(axis_len))
    ax.set_yticks(np.arange(axis_len))
    ax.set_xticklabels(labels[:axis_len])
    ax.set_yticklabels(labels[:axis_len])
    ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    ax.set_xlabel(x_label, fontsize=AXIS_LABEL_FONTSIZE, labelpad=4)
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_FONTSIZE)

    display_label = display_label_override or layout_mod._display_panel_label(label)
    title_fill = layout_mod._panel_base_color(label)
    title_fontsize = layout_mod._panel_title_fontsize(display_label) + TITLE_FONT_BONUS
    ax.add_patch(
        layout_mod.Rectangle(
            (0.0, 1.055),
            1.0,
            0.16,
            transform=ax.transAxes,
            facecolor=title_fill,
            edgecolor="none",
            alpha=0.42,
            clip_on=False,
            zorder=2,
        )
    )
    ax.text(
        0.5,
        1.135,
        display_label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=title_fontsize,
        color="black",
        zorder=3,
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_edgecolor("black")

    threshold = vmax * 0.55 if total else 0.0
    rows, cols = matrix.shape
    for row in range(rows):
        for col in range(cols):
            count = int(matrix[row, col])
            percentage = percent_matrix[row, col] * 100.0
            text_color = "white" if total and percent_matrix[row, col] >= threshold else "black"
            cell_text = f"{percentage:.0f}%" if variant == "pub" else f"{count}\n{percentage:.0f}%"
            ax.text(
                col,
                row,
                cell_text,
                ha="center",
                va="center",
                color=text_color,
                fontsize=CELL_TEXT_FONTSIZE_PUB if variant == "pub" else CELL_TEXT_FONTSIZE_RAW,
            )

    if panel_info_text is not None:
        panel_info_y = -0.24 if variant == "pub" else -0.12
        panel_info_fontsize = PANEL_INFO_FONTSIZE_PUB if variant == "pub" else PANEL_INFO_FONTSIZE_RAW
        ax.text(
            0.5,
            panel_info_y,
            panel_info_text,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=panel_info_fontsize,
            clip_on=False,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.3},
        )

    return image


def write_sidecar_csv(
    output_path: Path,
    *,
    collection_type: str,
    split_type: str,
    partition_type: str,
    publication_variant: str,
    panel_rows: list[tuple[int, str, str, str, np.ndarray, str, float | None, int]],
) -> None:
    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "collection_type",
                "split_type",
                "partition_type",
                "publication_variant",
                "panel_order",
                "panel_label",
                "display_label",
                "model_dir",
                "opinion_kind",
                "l2_norm_vs_human",
                "group_count",
                "row_likert",
                "col_likert",
                "count",
                "share",
                "panel_total",
            ],
        )
        writer.writeheader()
        for panel_order, panel_label, display_label, model_dir, matrix, opinion_kind, l2_norm, panel_group_count in panel_rows:
            total = int(np.asarray(matrix).sum())
            for row_index in range(matrix.shape[0]):
                for col_index in range(matrix.shape[1]):
                    count = int(matrix[row_index, col_index])
                    share = (count / total) if total else 0.0
                    writer.writerow(
                        {
                            "collection_type": collection_type,
                            "split_type": split_type,
                            "partition_type": partition_type,
                            "publication_variant": publication_variant,
                            "panel_order": panel_order,
                            "panel_label": panel_label,
                            "display_label": display_label,
                            "model_dir": model_dir,
                            "opinion_kind": opinion_kind,
                            "l2_norm_vs_human": "" if l2_norm is None else f"{l2_norm:.6f}",
                            "group_count": panel_group_count,
                            "row_likert": row_index + 1,
                            "col_likert": col_index + 1,
                            "count": count,
                            "share": f"{share:.6f}",
                            "panel_total": total,
                        }
                    )


def render_publication_panel(
    public_source_root: Path,
    private_source_root: Path,
    output_path: Path,
    *,
    collection_type: str,
    split_type: str,
    partition_type: str,
    filtered_state: str,
    variant: str,
) -> None:
    panel_specs = build_panel_specs(collection_type, split_type)
    panel_matrices: list[tuple[str, str, np.ndarray, tuple[int, int], str, str, str, float | None, int]] = []
    global_vmax = 0.0
    human_public_matrix: np.ndarray | None = None
    human_private_matrix: np.ndarray | None = None

    human_public_path = matrix_path(
        public_source_root,
        collection_type,
        split_type,
        partition_type,
        "human",
        raw_subdir="individual_likert",
        matrix_name="indiv_matrix",
    )
    human_private_path = matrix_path(
        private_source_root,
        collection_type,
        split_type,
        partition_type,
        "human",
        raw_subdir="individual_likert",
        matrix_name="indiv_matrix",
    )
    human_public_matrix = layout_mod._load_matrix(human_public_path)
    human_private_matrix = layout_mod._load_matrix(human_private_path)
    if human_public_matrix is None:
        raise FileNotFoundError(f"Missing public human transition matrix: {human_public_path}")
    if human_private_matrix is None:
        raise FileNotFoundError(f"Missing private human transition matrix: {human_private_path}")

    public_group_count = group_count(collection_type, split_type, partition_type, value_type="likert_pred")
    private_group_count = group_count(collection_type, split_type, partition_type, value_type="slider_value")

    for column_index, (label, model_dir, _position, display_label_override) in enumerate(panel_specs):
        display_label = display_label_override or layout_mod._display_panel_label(label)
        public_path = matrix_path(
            public_source_root,
            collection_type,
            split_type,
            partition_type,
            model_dir,
            raw_subdir="individual_likert",
            matrix_name="indiv_matrix",
        )
        private_path = matrix_path(
            private_source_root,
            collection_type,
            split_type,
            partition_type,
            model_dir,
            raw_subdir="individual_likert",
            matrix_name="indiv_matrix",
        )
        public_matrix = layout_mod._load_matrix(public_path)
        private_matrix = layout_mod._load_matrix(private_path)
        if public_matrix is None:
            raise FileNotFoundError(f"Missing public transition matrix: {public_path}")
        if private_matrix is None:
            raise FileNotFoundError(f"Missing private transition matrix: {private_path}")
        for matrix, opinion_kind, value_type, position, human_matrix, panel_group_count in (
            (public_matrix, "public", "likert_pred", (0, column_index), human_public_matrix, public_group_count),
            (private_matrix, "private", "slider_value", (1, column_index), human_private_matrix, private_group_count),
        ):
            total = int(np.asarray(matrix).sum())
            percent_matrix = (matrix / total) if total else np.zeros_like(matrix, dtype=float)
            global_vmax = max(global_vmax, float(percent_matrix.max()) if total else 0.0)
            l2_norm = None if label == "human" else normalized_frobenius_diff(matrix, human_matrix)
            panel_matrices.append((label, model_dir, matrix, position, display_label, opinion_kind, value_type, l2_norm, panel_group_count))
    if global_vmax <= 0.0:
        global_vmax = 1.0

    figure = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    bottom_margin = 0.13 if variant == "pub" else 0.16
    grid = figure.add_gridspec(2, 5, left=0.055, right=0.905, top=0.88, bottom=bottom_margin, wspace=0.38, hspace=0.42)
    state_title = "Unfiltered" if filtered_state == "unfiltered" else "Filtered"
    if variant != "pub":
        figure.suptitle(
            f"ABM Transition Matrices: {collection_type.title()} {split_type.title()} {partition_type.title()} (Public and Private, {state_title})",
            fontsize=SUPTITLE_FONTSIZE,
            y=0.975,
        )

    figure.text(0.018, 0.69, "Public opinion", rotation=90, va="center", ha="center", fontsize=ROW_LABEL_FONTSIZE)
    figure.text(0.018, 0.275, "Private opinion", rotation=90, va="center", ha="center", fontsize=ROW_LABEL_FONTSIZE)

    colorbar_image = None
    panel_rows_for_csv: list[tuple[int, str, str, str, np.ndarray, str, float | None, int]] = []
    for panel_order, (label, model_dir, matrix, (row, col), display_label, opinion_kind, value_type, l2_norm, panel_group_count) in enumerate(panel_matrices, start=1):
        ax = figure.add_subplot(grid[row, col])
        panel_total = int(np.asarray(matrix).sum())
        panel_info_text = None
        if variant == "raw":
            first_line = f"groups = {panel_group_count} | cell count = {panel_total}"
            second_line = None if l2_norm is None else f"L2 vs human = {l2_norm:.4f}"
            panel_info_text = first_line if second_line is None else f"{first_line}\n{second_line}"
        elif l2_norm is not None:
            panel_info_text = f"L2 vs human = {l2_norm:.4f}"
        colorbar_image = draw_publication_heatmap(
            ax,
            matrix,
            label,
            display_label_override=display_label,
            vmax=global_vmax,
            value_type=value_type,
            variant=variant,
            panel_info_text=panel_info_text,
        )
        panel_rows_for_csv.append((panel_order, label, display_label, model_dir, matrix, opinion_kind, l2_norm, panel_group_count))

    if colorbar_image is not None:
        colorbar = figure.colorbar(
            colorbar_image,
            ax=figure.axes,
            fraction=0.025,
            pad=0.02,
            shrink=0.88,
        )
        colorbar.set_label("Share of transitions (%)", fontsize=COLORBAR_LABEL_FONTSIZE)
        colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        colorbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)

    if variant == "raw":
        footnote = model_footnote(panel_specs, collection_type=collection_type)
        figure.text(
            0.055,
            0.04,
            textwrap.fill(footnote, width=180),
            ha="left",
            va="bottom",
            fontsize=FOOTNOTE_FONTSIZE,
            color="#444444",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(figure)
    write_sidecar_csv(
        output_path,
        collection_type=collection_type,
        split_type=split_type,
        partition_type=partition_type,
        publication_variant=variant,
        panel_rows=panel_rows_for_csv,
    )


def main() -> int:
    args = build_arg_parser().parse_args()
    output_filename = args.output_filename or default_output_filename(
        args.collection_type,
        args.split_type,
        args.partition_type,
        args.filtered_state,
    )
    output_stem = normalize_output_stem(output_filename)
    variants = [args.variant] if args.variant in {"raw", "pub"} else ["raw", "pub"]
    for variant in variants:
        output_path = variant_output_path(args.output_root, output_stem, variant)
        render_publication_panel(
            args.public_source_root.resolve(),
            args.private_source_root.resolve(),
            output_path.resolve(),
            collection_type=args.collection_type,
            split_type=args.split_type,
            partition_type=args.partition_type,
            filtered_state=args.filtered_state,
            variant=variant,
        )
        print(output_path.with_suffix(".pdf"))
        print(output_path.with_suffix(".png"))
        print(output_path.with_suffix(".csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())