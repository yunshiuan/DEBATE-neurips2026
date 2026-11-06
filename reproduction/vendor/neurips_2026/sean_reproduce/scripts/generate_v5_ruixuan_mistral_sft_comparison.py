#!/usr/bin/env python3
"""Render a controlled Mistral-7B base/SFT v5 Ruixuan comparison bundle.

The Mistral adapters are evaluated against their matching Mistral-SFT human
and validity CSVs.  The canonical pre-SFT Mistral output is then scored on
that same reference surface, and every learned-model metric uses only prefixes
where both base and SFT are usable.  This prevents label-surface or source
coverage differences from masquerading as an SFT effect.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
SCRIPT_DIR = SCRIPT_PATH.parent
for import_path in (REPO_ROOT, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import generate_v5_leo_qwen4b_comparison as shared  # noqa: E402

renderer = shared.renderer


MISTRAL_BASE = "Mistral-7B-Instruct-v0.3"
MISTRAL_GROUP = "ft:Mistral-7B-Instruct-v0.3-SFT-20260725-bd-unsloth:bd-group-3epochs"
MISTRAL_TOPIC = "ft:Mistral-7B-Instruct-v0.3-SFT-20260725-bd-unsloth:bd-topic-3epochs"

DEFAULT_MISTRAL_ROOT = Path("data/imports/ruixuan_mistral_adapters_simulation_eval_v5_20260727/eval_v5/mistral-7b/human_llm")
DEFAULT_BASE_ROOT = REPO_ROOT / "neurips_2026" / "chengtao" / "result" / "eval_v3" / "human_llm"
DEFAULT_MASK_ROOT = Path("data/imports/v5_bd_data_ruixuan_20260716/data/sft_data/masks")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "neurips_2026" / "sean_reproduce" / "v5_ruixuan" / "mistral_sft_comparison"
DEFAULT_HTML_PATH = REPO_ROOT / "neurips_2026" / "sean_reproduce" / "v5_ruixuan" / "SFT_Mistral.html"

PARTITIONS = ("group_train", "group_test", "topic_train", "topic_test")
PARTITION_OUTPUT_DIRS = {
    "group_train": "group_train_set",
    "group_test": "group_test_set",
    "topic_train": "topic_train_set",
    "topic_test": "topic_test_set",
}
PARTITION_MASKS = {
    "group_train": "group_train_prefixes.txt",
    "group_test": "group_test_prefixes.txt",
    "topic_train": "topic_train_prefixes.txt",
    "topic_test": "topic_test_prefixes.txt",
}
MODEL_COLORS = {
    renderer.HUMAN_MODEL: "#222222",
    MISTRAL_BASE: "#D89021",
    MISTRAL_GROUP: "#C45B18",
    MISTRAL_TOPIC: "#C45B18",
    renderer.DEGROOT_MODEL: "#7F7F7F",
}
METRICS = (
    ("Public single", "public_stance_alignment_abs"),
    ("Private single", "private_stance_alignment_abs"),
    ("Public update", "public_stance_change_error_abs"),
    ("Private update", "private_stance_change_error_abs"),
    ("Public convergence", "public_opinion_convergence_error"),
    ("Private convergence", "private_opinion_convergence_error"),
)


def is_group(partition: str) -> bool:
    return partition.startswith("group_")


def sft_model(partition: str) -> str:
    return MISTRAL_GROUP if is_group(partition) else MISTRAL_TOPIC


def output_dir_for(output_dir: Path, partition: str) -> Path:
    return output_dir / PARTITION_OUTPUT_DIRS[partition]


def model_label(model: str) -> str:
    return {
        renderer.HUMAN_MODEL: "Human",
        MISTRAL_BASE: "Mistral-7B base",
        MISTRAL_GROUP: "Mistral-7B SFT (group)",
        MISTRAL_TOPIC: "Mistral-7B SFT (topic)",
        renderer.DEGROOT_MODEL: "DeGroot fit",
    }.get(model, model)


def publication_model_label(model: str) -> str:
    return {
        renderer.HUMAN_MODEL: "Human",
        MISTRAL_BASE: "Mistral-7B\nbase",
        MISTRAL_GROUP: "Mistral SFT",
        MISTRAL_TOPIC: "Mistral SFT",
        renderer.DEGROOT_MODEL: "DeGroot fit",
    }.get(model, model_label(model))


def source_name(model: str) -> str:
    if model == MISTRAL_BASE:
        return "canonical eval-v3 Mistral-7B base"
    if model in {MISTRAL_GROUP, MISTRAL_TOPIC}:
        return "supplied v5 Mistral-7B SFT"
    if model == renderer.DEGROOT_MODEL:
        return "fitted DeGroot reference"
    return "unknown"


def comparison_models(partition: str) -> list[str]:
    return [MISTRAL_BASE, sft_model(partition)]


def read_prefixes(mask_root: Path, partition: str) -> list[str]:
    return [line.strip() for line in (mask_root / PARTITION_MASKS[partition]).read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical_human_path(mistral_root: Path, partition: str, prefix: str, version: str) -> Path:
    return mistral_root / prefix / sft_model(partition) / str(renderer.VERSION_SPECS[version]["human_filename"])


def configure_shared_helpers() -> None:
    """Reuse the controlled metric/profile implementation with Mistral names."""
    shared.QWEN_BASE = MISTRAL_BASE
    shared.QWEN_GROUP = MISTRAL_GROUP
    shared.QWEN_TOPIC = MISTRAL_TOPIC
    shared.PARTITION_MASKS = dict(PARTITION_MASKS)
    shared.comparison_models = comparison_models
    shared.source_name = source_name
    shared.model_label = model_label
    shared.publication_model_label = publication_model_label
    shared.MODEL_COLORS = dict(MODEL_COLORS)

    def model_root(model: str, *, qwen_root: Path, llama_root: Path, llama_base_root: Path) -> Path:
        del llama_root
        if model == MISTRAL_BASE:
            return llama_base_root
        if model in {MISTRAL_GROUP, MISTRAL_TOPIC}:
            return qwen_root
        raise ValueError(f"Unknown Mistral comparison model: {model}")

    shared.model_root = model_root


def configure_renderer() -> None:
    renderer.GROUP_MODEL = MISTRAL_GROUP
    renderer.TOPIC_MODEL = MISTRAL_TOPIC
    renderer.PARTITION_SPECS = {
        "group_train": {"mask": PARTITION_MASKS["group_train"], "model": MISTRAL_GROUP, "fit_partition": "group_train"},
        "group_test": {"mask": PARTITION_MASKS["group_test"], "model": MISTRAL_GROUP, "fit_partition": "group_train"},
        "topic_train": {"mask": PARTITION_MASKS["topic_train"], "model": MISTRAL_TOPIC, "fit_partition": "topic_train"},
        "topic_test": {"mask": PARTITION_MASKS["topic_test"], "model": MISTRAL_TOPIC, "fit_partition": "topic_train"},
    }
    renderer.BASE_MODELS = [MISTRAL_BASE]
    renderer.FOCUSED_BASE_MODEL = MISTRAL_BASE
    renderer.FOCUSED_OTHER_BASE_MODEL = MISTRAL_BASE
    renderer.MODEL_COLORS = dict(MODEL_COLORS)
    for model, color in MODEL_COLORS.items():
        renderer.MODEL_COLOR_MAP[model] = color
    renderer.FINE_TUNED_DISPLAY_MAP[MISTRAL_GROUP] = "Mistral SFT"
    renderer.FINE_TUNED_DISPLAY_MAP[MISTRAL_TOPIC] = "Mistral SFT"
    # M1/A3 runs through the v4 bar-style postprocessor, which owns its own
    # publication-label map rather than the renderer's fine-tuned map.
    # Register both adapter IDs there so raw Hugging Face identifiers never
    # become x-axis labels.
    renderer.bar_style.PUBLICATION_SHORT_LABELS[MISTRAL_GROUP] = "Mistral SFT"
    renderer.bar_style.PUBLICATION_SHORT_LABELS[MISTRAL_TOPIC] = "Mistral SFT"
    renderer.model_label = model_label
    renderer.publication_model_label = publication_model_label


def configure_partition(partition: str) -> None:
    del partition
    renderer.BASE_MODELS = [MISTRAL_BASE]
    renderer.FOCUSED_BASE_MODEL = MISTRAL_BASE
    renderer.FOCUSED_OTHER_BASE_MODEL = MISTRAL_BASE
    renderer.PROFILE_CACHE.clear()


def profile_series(*, mistral_root: Path, base_root: Path, mask_root: Path, partition: str, version: str, alpha_spec: dict[str, float]) -> dict[str, list[dict[str, str]]]:
    """Use one Mistral-SFT human reference for human, base, SFT, and DeGroot."""
    rows_by_model: dict[str, list[dict[str, str]]] = {
        renderer.HUMAN_MODEL: [],
        MISTRAL_BASE: [],
        sft_model(partition): [],
        renderer.DEGROOT_MODEL: [],
    }
    llm_filename = str(renderer.VERSION_SPECS[version]["llm_filename"])
    for prefix in read_prefixes(mask_root, partition):
        human_path = canonical_human_path(mistral_root, partition, prefix, version)
        base_path = base_root / prefix / MISTRAL_BASE / llm_filename
        sft_path = mistral_root / prefix / sft_model(partition) / llm_filename
        if not (human_path.is_file() and base_path.is_file() and sft_path.is_file()):
            continue
        try:
            human_rows = shared.compact_dynamics_rows(shared.read_rows(human_path, root=mistral_root))
            base_rows = shared.compact_dynamics_rows(shared.read_rows(base_path, root=base_root))
            sft_rows = shared.compact_dynamics_rows(shared.read_rows(sft_path, root=mistral_root))
        except Exception:
            continue
        rows_by_model[renderer.HUMAN_MODEL].extend(human_rows)
        rows_by_model[MISTRAL_BASE].extend(base_rows)
        rows_by_model[sft_model(partition)].extend(sft_rows)
        try:
            rows_by_model[renderer.DEGROOT_MODEL].extend(
                renderer.build_degroot_proxy_rows(
                    human_rows,
                    data_prefix=prefix,
                    likert_col=renderer.LIKERT_COL,
                    alpha_public=float(alpha_spec["alpha_public"]),
                    alpha_private=float(alpha_spec["alpha_private"]),
                )
            )
        except ValueError:
            pass
    return rows_by_model


def install_profile_override(*, mistral_root: Path, base_root: Path, mask_root: Path) -> None:
    def profile_rows_for_partition(*, result_root: Path, base_result_root: Path, data_root: Path, partition: str, version: str, alpha_spec: dict[str, float]) -> dict[str, list[dict[str, str]]]:
        del result_root, base_result_root, data_root
        return profile_series(mistral_root=mistral_root, base_root=base_root, mask_root=mask_root, partition=partition, version=version, alpha_spec=alpha_spec)

    renderer.profile_rows_for_partition = profile_rows_for_partition


def render_alignment_bars(*, rows: list[dict[str, Any]], partition: str, output_path: Path) -> None:
    """Render M1/A3 with the canonical geometry and stable Mistral labels.

    The shared renderer's bar postprocessor uses a second label registry.  A
    final explicit tick assignment keeps the adapter's long checkpoint ID out
    of the exported publication-style panel even if that registry is imported
    under a separate module instance.
    """
    rows_by_version = renderer.comparison_rows_by_version(rows, partition=partition)
    model_order = renderer.comparison_model_order(partition)
    figure = renderer.build_page_figure(
        page_title=f"v5 combined BD: {renderer.short_partition(partition)}",
        rows_by_version=rows_by_version,
        hide_count_annotations=True,
        model_order=model_order,
        shared_models=set(),
    )
    renderer.bar_style._clean_publication_figure(figure, model_order=model_order)
    renderer.bar_style._tighten_y_limits(
        figure,
        rows_by_version=rows_by_version,
        model_order=model_order,
        has_value_labels=False,
    )
    axes = list(figure.axes)
    if axes:
        axes[-1].set_xticklabels(
            ["Mistral-7B", "Mistral SFT", "DeGroot fit"] * 2,
            rotation=90,
            ha="center",
            va="top",
            fontsize=renderer.bar_style.PUBLICATION_XTICK_FONTSIZE,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    renderer.write_csv(
        output_path.with_suffix(".csv"),
        [row for version_rows in rows_by_version.values() for row in version_rows],
    )


def render_transition(*, mistral_root: Path, base_root: Path, mask_root: Path, partition: str, alpha_spec: dict[str, float], output_path: Path) -> None:
    series = profile_series(mistral_root=mistral_root, base_root=base_root, mask_root=mask_root, partition=partition, version="v0", alpha_spec=alpha_spec)
    specs = [
        (renderer.HUMAN_MODEL, "human", "Human"),
        (MISTRAL_BASE, "base: Mistral-7B-Instruct-v0.3", "Mistral-7B\nbase"),
        (sft_model(partition), "sft: Mistral", "Mistral SFT"),
        (renderer.DEGROOT_MODEL, "DeGroot fit", "DeGroot fit"),
    ]
    matrices = {(kind, model): renderer.transition_matrix(series[model], kind=kind) for kind in ("public", "private") for model, _, _ in specs}
    maximum = max(((matrix / int(matrix.sum())).max() for matrix in matrices.values() if int(matrix.sum())), default=1.0)
    fig = plt.figure(figsize=(18.0, renderer.transition_style.FIGURE_HEIGHT))
    grid = fig.add_gridspec(2, len(specs), left=0.075, right=0.89, top=0.88, bottom=0.13, wspace=0.42, hspace=0.42)
    fig.text(0.025, 0.69, "Public opinion", rotation=90, va="center", ha="center", fontsize=renderer.transition_style.ROW_LABEL_FONTSIZE)
    fig.text(0.025, 0.275, "Private opinion", rotation=90, va="center", ha="center", fontsize=renderer.transition_style.ROW_LABEL_FONTSIZE)
    output_rows: list[dict[str, Any]] = []
    image: Any | None = None
    for row_index, kind in enumerate(("public", "private")):
        human_matrix = matrices[(kind, renderer.HUMAN_MODEL)]
        for column_index, (model, style_label, display_label) in enumerate(specs):
            matrix = matrices[(kind, model)]
            l2 = None if model == renderer.HUMAN_MODEL else renderer.normalized_frobenius_diff(matrix, human_matrix)
            image = renderer.transition_style.draw_publication_heatmap(
                fig.add_subplot(grid[row_index, column_index]), matrix, style_label,
                display_label_override=display_label, vmax=maximum,
                value_type="likert_pred" if kind == "public" else "slider_value", variant="pub",
                panel_info_text=None if l2 is None else f"L2 vs human = {l2:.4f}",
            )
            total = int(matrix.sum())
            for start in range(6):
                for end in range(6):
                    output_rows.append({"partition": partition, "simulation_version": "v0", "opinion_type": kind, "model": model, "model_label": publication_model_label(model), "l2_norm_vs_human": "" if l2 is None else l2, "start_likert": start + 1, "end_likert": end + 1, "count": int(matrix[start, end]), "share": int(matrix[start, end]) / total if total else 0.0, "panel_total": total})
    if image is not None:
        colorbar = fig.colorbar(image, ax=fig.axes, fraction=0.027, pad=0.02, shrink=0.88)
        colorbar.set_label("Share of transitions (%)", fontsize=renderer.transition_style.COLORBAR_LABEL_FONTSIZE)
        colorbar.ax.yaxis.set_major_formatter(renderer.transition_style.PercentFormatter(xmax=1.0, decimals=0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    shared.write_csv(output_path.with_suffix(".csv"), output_rows)


def render_histogram(*, mistral_root: Path, base_root: Path, mask_root: Path, partition: str, alpha_spec: dict[str, float], kind: str, output_path: Path) -> None:
    series = profile_series(mistral_root=mistral_root, base_root=base_root, mask_root=mask_root, partition=partition, version="v0", alpha_spec=alpha_spec)
    panels = [(renderer.HUMAN_MODEL, "first"), (renderer.HUMAN_MODEL, "final"), (MISTRAL_BASE, "final"), (sft_model(partition), "final")]
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 10.4))
    rows: list[dict[str, Any]] = []
    for index, (axis, (model, stage)) in enumerate(zip(np.asarray(axes).reshape(-1), panels)):
        first, final = renderer.stage_values(series[model], kind=kind)
        values = first if stage == "first" else final
        counts = [sum(1 for value in values if int(round(value)) == score) for score in range(1, 7)]
        total = len(values)
        summary = {"n": str(total), "mean_likert": str(renderer.safe_mean(values)), **{f"count_{score}": str(count) for score, count in zip(range(1, 7), counts)}}
        title = "Human anchor" if model == renderer.HUMAN_MODEL and stage == "first" else "Human final" if model == renderer.HUMAN_MODEL else publication_model_label(model).replace("\n", " ")
        renderer.histogram_style.add_histogram_subplot(axis, row=summary, title=title, color=MODEL_COLORS[model], show_ylabel=index % 2 == 0)
        axis.set_title(f"{title}\n(mean={renderer.histogram_style.mean_from_summary_row(summary):.2f}, sd={renderer.histogram_style.std_from_summary_row(summary):.2f})", fontsize=15, pad=8)
        axis.grid(False)
        axis.set_xlabel("Likert", fontsize=13)
        if index % 2 == 0:
            axis.set_ylabel("Share (%)", fontsize=13)
        for score, count in zip(range(1, 7), counts):
            rows.append({"partition": partition, "simulation_version": "v0", "opinion_type": kind, "model": model, "model_label": publication_model_label(model), "stage": stage, "likert": score, "count": count, "n": total, "percent": 100 * count / total if total else math.nan})
    fig.suptitle(f"Histogram of {kind.title()} Opinion Stance", fontsize=25, y=0.985, fontweight="bold")
    fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    shared.write_csv(output_path.with_suffix(".csv"), rows)


def render_error_pairs(*, rows: list[dict[str, Any]], partition: str, output_path: Path) -> None:
    """Render the M2/A4 error-pair view without a meaningless n=2 correlation.

    The canonical correlation panel fits an across-model correlation, which
    requires at least three learned series.  This focused ablation has exactly
    base and SFT, so it instead shows their locations in the same four error
    spaces, retaining both prediction modes and all source values in the CSV.
    """
    pairs = (
        ("Public: single vs update", "public_stance_alignment_abs", "public_stance_change_error_abs"),
        ("Public: single vs convergence", "public_stance_alignment_abs", "public_opinion_convergence_error"),
        ("Private: single vs update", "private_stance_alignment_abs", "private_stance_change_error_abs"),
        ("Private: single vs convergence", "private_stance_alignment_abs", "private_opinion_convergence_error"),
    )
    selected = [row for row in rows if str(row.get("partition")) == partition and str(row.get("model")) in {MISTRAL_BASE, sft_model(partition)}]
    fig, axes = plt.subplots(1, 4, figsize=(19.0, 4.8))
    for axis, (title, x_metric, y_metric) in zip(axes, pairs):
        for row in selected:
            model = str(row["model"])
            version = str(row["simulation_version"])
            marker = "o" if version == "v0" else "s"
            label = f"{publication_model_label(model).replace(chr(10), ' ')} · {'full dynamics' if version == 'v0' else 'next message'}"
            axis.scatter(shared.finite(row[x_metric]), shared.finite(row[y_metric]), s=92, marker=marker, color=MODEL_COLORS[model], edgecolor="#303030", linewidth=0.6, label=label, zorder=3)
        axis.set_title(title, fontsize=12, fontweight="bold")
        axis.set_xlabel("Single-opinion error" if x_metric.endswith("alignment_abs") else "Error", fontsize=11)
        axis.set_ylabel("Opinion-update error" if y_metric.endswith("change_error_abs") else "Convergence error", fontsize=11)
        axis.grid(True, alpha=0.22, linewidth=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Error-pair placement for the Mistral base/SFT ablation", fontsize=17, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0.01, 0.10, 0.99, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    shared.write_csv(output_path.with_suffix(".csv"), selected)


def render_all(*, rows: list[dict[str, Any]], alphas: dict[tuple[str, str], dict[str, float]], mistral_root: Path, base_root: Path, mask_root: Path, output_dir: Path, bootstrap_reps: int) -> None:
    for partition in PARTITIONS:
        configure_partition(partition)
        directory = output_dir_for(output_dir, partition)
        directory.mkdir(parents=True, exist_ok=True)
        ids = ("M1", "M2", "M3", "M4", "M5", "A6") if is_group(partition) else ("A3", "A4", "A5", "A8", "A9", "A7")
        alignment, correlation, convergence, transition, partner, histogram = ids
        stem = f"{partition}_mistral_sft"
        print(f"Rendering Mistral comparison panels for {partition}...", flush=True)
        render_alignment_bars(rows=rows, partition=partition, output_path=directory / f"{alignment}_{stem}_alignment_bars")
        render_error_pairs(rows=rows, partition=partition, output_path=directory / f"{correlation}_{stem}_error_correlations")
        renderer.render_convergence_composite(result_root=mistral_root, base_result_root=base_root, data_root=mask_root, partition=partition, alpha_spec=alphas[(partition, "v0")], output_path=directory / f"{convergence}_{stem}_convergence_sd")
        render_transition(mistral_root=mistral_root, base_root=base_root, mask_root=mask_root, partition=partition, alpha_spec=alphas[(partition, "v0")], output_path=directory / f"{transition}_{stem}_transition_matrices")
        renderer.render_partner_gap_composite(result_root=mistral_root, base_result_root=base_root, data_root=mask_root, partition=partition, alpha_spec=alphas[(partition, "v0")], output_path=directory / f"{partner}_{stem}_partner_gap", bootstrap_reps=bootstrap_reps)
        for kind in ("public", "private"):
            render_histogram(mistral_root=mistral_root, base_root=base_root, mask_root=mask_root, partition=partition, alpha_spec=alphas[(partition, "v0")], kind=kind, output_path=directory / f"{histogram}_{stem}_{kind}_histograms")


def table_html(headers: list[str], rows: list[list[str]]) -> tuple[str, str]:
    markdown = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    markdown.extend("| " + " | ".join(row) + " |" for row in rows)
    rendered = "<table><thead><tr>" + "".join(f"<th>{html.escape(header)}</th>" for header in headers) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>" for row in rows) + "</tbody></table>"
    return rendered, "\n".join(markdown)


def rebuttal_table(rows: list[dict[str, Any]]) -> tuple[str, str, int, int]:
    lookup = {(str(row["partition"]), str(row["simulation_version"]), str(row["model"])): row for row in rows}
    headers = ["Split", "Mode", "Matched n"] + [label for label, _ in METRICS]
    table_rows: list[list[str]] = []
    lower = total = 0
    for partition in ("group_test", "topic_test"):
        for version in ("v0", "v1_5"):
            base = lookup[(partition, version, MISTRAL_BASE)]
            sft = lookup[(partition, version, sft_model(partition))]
            cells = [partition.replace("_", " ").title(), "Full dynamics" if version == "v0" else "Next message", str(int(shared.finite(base["prefixes_ok"])))]
            for _, metric in METRICS:
                base_value, sft_value = shared.finite(base[metric]), shared.finite(sft[metric])
                delta = sft_value - base_value
                lower += int(delta < 0)
                total += 1
                cells.append(f"{base_value:.3f} → {sft_value:.3f} ({delta:+.3f})")
            table_rows.append(cells)
    return (*table_html(headers, table_rows), lower, total)


def figure_csv_path(output_dir: Path, partition: str, figure_id: str, suffix: str) -> Path:
    return output_dir_for(output_dir, partition) / f"{figure_id}_{partition}_mistral_sft_{suffix}.csv"


def read_output_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def find_row(rows: list[dict[str, str]], **required: str) -> dict[str, str]:
    for row in rows:
        if all(str(row.get(field, "")) == value for field, value in required.items()):
            return row
    raise KeyError(f"Missing exported row matching {required}")


def rebuttal_label(model: str) -> str:
    return {
        renderer.HUMAN_MODEL: "Human",
        MISTRAL_BASE: "Mistral base",
        MISTRAL_GROUP: "Mistral SFT",
        MISTRAL_TOPIC: "Mistral SFT",
    }.get(model, model_label(model))


def format_rho(row: dict[str, str]) -> str:
    rho = shared.finite(row["spearman_rho"])
    low = shared.finite(row["spearman_rho_ci_low"])
    high = shared.finite(row["spearman_rho_ci_high"])
    count = int(shared.finite(row["n"]))
    return f"ρ={rho:.3f} [{low:.3f}, {high:.3f}]; n={count}"


def rebuttal_summary(rows: list[dict[str, Any]], output_dir: Path) -> tuple[str, str, int, int]:
    """Build Qwen-style, figure-native held-out tables for the Mistral report."""
    by_key = {(str(row["partition"]), str(row["simulation_version"]), str(row["model"])): row for row in rows}
    lower_count = comparison_count = 0

    error_headers = ["Split", "Mode", "Matched n", "Series"] + [label for label, _ in METRICS]
    error_rows: list[list[str]] = []
    for partition in ("group_test", "topic_test"):
        for version in ("v0", "v1_5"):
            base = by_key[(partition, version, MISTRAL_BASE)]
            sft = by_key[(partition, version, sft_model(partition))]
            cells: list[str] = []
            for _, metric in METRICS:
                base_value, sft_value = shared.finite(base[metric]), shared.finite(sft[metric])
                delta = sft_value - base_value
                lower_count += int(delta < 0)
                comparison_count += 1
                cells.append(f"{base_value:.3f} → {sft_value:.3f} ({delta:+.3f})")
            error_rows.append([
                partition.replace("_", " ").title(),
                "Full dynamics" if version == "v0" else "Next message",
                str(int(shared.finite(base["prefixes_ok"]))),
                "Mistral base → SFT (Δ)",
                *cells,
            ])
    error_html, error_markdown = shared.markdown_html_table(error_headers, error_rows)

    convergence_headers = ["Panel", "Series", "Public mean Δ SD", "Private mean Δ SD"]
    convergence_rows: list[list[str]] = []
    for partition in ("group_test", "topic_test"):
        figure_id = "M3" if is_group(partition) else "A5"
        source_rows = read_output_csv(figure_csv_path(output_dir, partition, figure_id, "convergence_sd"))
        for model in (renderer.HUMAN_MODEL, MISTRAL_BASE, sft_model(partition)):
            public = find_row(source_rows, panel="public_convergence", model=model)
            private = find_row(source_rows, panel="private_convergence", model=model)
            convergence_rows.append([
                f"{figure_id} · {partition.replace('_', ' ').title()}",
                rebuttal_label(model),
                f"{shared.finite(public['mean_delta_sd']):+.3f}",
                f"{shared.finite(private['mean_delta_sd']):+.3f}",
            ])
    convergence_html, convergence_markdown = shared.markdown_html_table(convergence_headers, convergence_rows)

    transition_headers = ["Panel", "Series", "Public L2 vs human", "Private L2 vs human"]
    transition_rows: list[list[str]] = []
    for partition in ("group_test", "topic_test"):
        figure_id = "M4" if is_group(partition) else "A8"
        source_rows = read_output_csv(figure_csv_path(output_dir, partition, figure_id, "transition_matrices"))
        for model in (MISTRAL_BASE, sft_model(partition)):
            public = find_row(source_rows, model=model, opinion_type="public")
            private = find_row(source_rows, model=model, opinion_type="private")
            transition_rows.append([
                f"{figure_id} · {partition.replace('_', ' ').title()}",
                rebuttal_label(model),
                f"{shared.finite(public['l2_norm_vs_human']):.3f}",
                f"{shared.finite(private['l2_norm_vs_human']):.3f}",
            ])
    transition_html, transition_markdown = shared.markdown_html_table(transition_headers, transition_rows)

    partner_headers = ["Panel", "Series", "Public partner-gap Spearman", "Private partner-gap Spearman"]
    partner_rows: list[list[str]] = []
    for partition in ("group_test", "topic_test"):
        figure_id = "M5" if is_group(partition) else "A9"
        source_rows = read_output_csv(figure_csv_path(output_dir, partition, figure_id, "partner_gap"))
        for model in (renderer.HUMAN_MODEL, MISTRAL_BASE, sft_model(partition)):
            public = find_row(source_rows, panel="public_spearman", model=model)
            private = find_row(source_rows, panel="private_spearman", model=model)
            partner_rows.append([
                f"{figure_id} · {partition.replace('_', ' ').title()}",
                rebuttal_label(model),
                format_rho(public),
                format_rho(private),
            ])
    partner_html, partner_markdown = shared.markdown_html_table(partner_headers, partner_rows)

    sections = [
        ("M1 / A3 — stance prediction errors", "Each cell is Mistral base → SFT (SFT − base). Lower is better.", error_html, error_markdown),
        ("M3 / A5 — opinion convergence", "Mean change in cross-participant stance SD (final − initial). More negative values indicate greater convergence; the human row is the target trajectory.", convergence_html, convergence_markdown),
        ("M4 / A8 — public/private transition matrices", "L2 distance between a model and the human transition matrix. Lower is better.", transition_html, transition_markdown),
        ("M5 / A9 — partner-gap relationship", "Spearman ρ between signed partner-minus-self initial gap and opinion change, with the plotted 95% paired-bootstrap interval and point count.", partner_html, partner_markdown),
    ]
    rendered = "".join(f'<h3>{html.escape(title)}</h3><p>{html.escape(description)}</p><div class="table-scroll">{table}</div>' for title, description, table, _ in sections)
    markdown = "\n\n".join(f"**{title}**\n\n{description}\n\n{table}" for title, description, _, table in sections)
    return rendered, markdown, lower_count, comparison_count


def figure_specs(partition: str) -> list[tuple[str, str, list[str]]]:
    ids = ("M1", "M2", "M3", "M4", "M5", "A6") if is_group(partition) else ("A3", "A4", "A5", "A8", "A9", "A7")
    alignment, correlation, convergence, transition, partner, histogram = ids
    stem = f"{partition}_mistral_sft"
    return [
        (alignment, "All three stance errors", [f"{alignment}_{stem}_alignment_bars"]),
        (correlation, "Error-pair placement", [f"{correlation}_{stem}_error_correlations"]),
        (convergence, "Opinion convergence", [f"{convergence}_{stem}_convergence_sd"]),
        (transition, "Public/private transition matrices", [f"{transition}_{stem}_transition_matrices"]),
        (partner, "Partner-gap relationship", [f"{partner}_{stem}_partner_gap"]),
        (histogram, "Opinion-distribution histograms", [f"{histogram}_{stem}_public_histograms", f"{histogram}_{stem}_private_histograms"]),
    ]


def _write_html_legacy(*, html_path: Path, output_dir: Path, rows: list[dict[str, Any]]) -> None:
    table, markdown, lower, total = rebuttal_table(rows)
    nav = ['<a href="#overview">Overview</a>', '<a href="#heldout">Held-out table</a>']
    sections: list[str] = []
    for partition in PARTITIONS:
        section_id = partition.replace("_", "-")
        nav.append(f'<a href="#{section_id}">{html.escape(partition.replace("_", " ").title())}</a>')
        cards: list[str] = []
        for figure_id, title, stems in figure_specs(partition):
            primary, *extras = stems
            base = f"mistral_sft_comparison/{PARTITION_OUTPUT_DIRS[partition]}/{primary}"
            extra = ""
            if extras:
                label = "Private-channel histogram" if title == "Opinion-distribution histograms" else "Auxiliary correlation renderings"
                extra = f'<details><summary>{label}</summary>' + "".join(f'<a class="asset" href="mistral_sft_comparison/{PARTITION_OUTPUT_DIRS[partition]}/{name}.png"><img src="mistral_sft_comparison/{PARTITION_OUTPUT_DIRS[partition]}/{name}.png" alt="{html.escape(figure_id)} auxiliary"></a>' for name in extras) + "</details>"
            cards.append(f'<article class="card"><h3>{figure_id} · {html.escape(title)}</h3><a class="asset" href="{base}.png"><img src="{base}.png" alt="{html.escape(figure_id)} {html.escape(title)}"></a><p><a href="{base}.png">PNG</a> · <a href="{base}.pdf">PDF</a> · <a href="{base}.csv">CSV</a></p>{extra}</article>')
        sections.append(f'<section id="{section_id}"><h2>{html.escape(partition.replace("_", " ").title())}</h2><p class="subtitle">Matched Mistral base/SFT comparison; M1/A3 and M2/A4 show full dynamics and next-message prediction, while dynamics panels use full dynamics.</p>{"".join(cards)}</section>')
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Mistral-7B SFT — v5 Ruixuan</title><style>
:root{{--ink:#26384b;--muted:#5c6d7d;--line:#d8e1eb;--paper:#f6f8fb;--accent:#075d8c;--active:#dceefa}}*{{box-sizing:border-box}}body{{margin:0;font:15px/1.5 ui-sans-serif,system-ui,sans-serif;color:var(--ink)}}aside{{position:fixed;inset:0 auto 0 0;width:250px;padding:25px 18px;overflow:auto;border-right:1px solid var(--line);background:var(--paper)}}aside h1{{font-size:18px;line-height:1.3;margin:0 0 7px}}aside p{{color:var(--muted);font-size:13px;margin:0 0 18px}}aside a{{display:block;color:#33526b;text-decoration:none;padding:8px 10px;border-radius:6px;margin:2px 0}}aside a.active{{color:#034c73;background:var(--active);font-weight:700}}main{{max-width:1760px;margin-left:250px;padding:30px 42px 64px}}h2{{scroll-margin-top:20px;border-bottom:2px solid var(--line);padding-bottom:8px;margin-top:48px}}h3{{margin:0 0 10px;font-size:21px}}.notice{{padding:14px 17px;border-left:4px solid #c77916;background:#fff8e9;margin:18px 0}}.subtitle{{color:var(--muted);max-width:1100px}}.card{{border:1px solid var(--line);border-radius:10px;padding:20px;margin:22px 0;box-shadow:0 1px 3px #1526370d}}.asset{{display:block}}img{{width:100%;height:auto;border:1px solid #e4eaf0}}.card p a,details summary{{color:var(--accent);font-weight:650}}details{{margin-top:14px}}details summary{{cursor:pointer}}details .asset{{margin-top:14px}}.table-scroll{{overflow-x:auto;border:1px solid var(--line);border-radius:8px}}table{{border-collapse:collapse;width:100%;min-width:1080px;font-size:13px}}th,td{{text-align:left;padding:7px 9px;border-bottom:1px solid var(--line);white-space:nowrap;font-variant-numeric:tabular-nums}}th{{background:var(--paper)}}textarea{{display:block;width:100%;min-height:260px;resize:vertical;padding:12px;border:1px solid var(--line);border-radius:8px;font:12px/1.45 ui-monospace,monospace}}button{{border:0;border-radius:6px;padding:8px 11px;margin:12px 0 8px;background:var(--accent);color:#fff;cursor:pointer;font:inherit;font-weight:650}}@media(max-width:900px){{aside{{position:static;width:auto;border-right:0;border-bottom:1px solid var(--line)}}aside a{{display:inline-block}}main{{margin:0;padding:24px 16px}}}}
</style></head><body><aside><h1>Mistral-7B SFT comparison</h1><p>v5 Ruixuan masks · Mistral base versus SFT</p>{"".join(nav)}</aside><main><section id="overview"><h2>Overview</h2><p>This isolated bundle compares canonical Mistral-7B-Instruct-v0.3 base output with the supplied group/topic Mistral SFT adapters across the same paper-style stance and opinion-dynamics panels.</p><div class="notice"><strong>Comparison contract.</strong> The canonical base package and the new Mistral delivery have different human/validity files. Both model outputs are therefore evaluated against the matching Mistral-SFT human/validity reference for each partition. A metric aggregate includes only prefixes with usable results for both base and SFT.</div><p><strong>Unavailable family:</strong> the delivery does not provide root-level <code>llm_report_v0_bd.csv</code>/<code>llm_report_v1_5_bd.csv</code> aggregate inputs, so A1/A2 are not rendered.</p><p><a href="mistral_sft_comparison/csv/summary_metrics.csv">Summary metrics CSV</a> · <a href="mistral_sft_comparison/csv/coverage_summary.csv">coverage CSV</a> · <a href="mistral_sft_comparison/README.md">README</a></p></section><section id="heldout"><h2>Held-out Mistral base → SFT results</h2><p><strong>Lower is better.</strong> SFT lowers {lower}/{total} public/private stance-error values below. Each cell is Mistral base → SFT (SFT − base); <code>n</code> is the two-model common usable-prefix count.</p><div class="table-scroll">{table}</div><h3>Copyable Markdown</h3><button data-copy-target="mistral-markdown">Copy Markdown</button><textarea id="mistral-markdown" readonly spellcheck="false">{html.escape(markdown)}</textarea></section>{"".join(sections)}</main><script>const links=[...document.querySelectorAll("aside a[href^='#']")],targets=links.map(a=>[a,document.querySelector(a.getAttribute("href"))]).filter(([,e])=>e);const observer=new IntersectionObserver(es=>{{const v=es.filter(e=>e.isIntersecting).sort((a,b)=>b.intersectionRatio-a.intersectionRatio);if(!v.length)return;links.forEach(a=>a.classList.toggle("active",a.getAttribute("href")==="#"+v[0].target.id));}},{{rootMargin:"-12% 0px -72% 0px",threshold:[0,.15,.4]}});targets.forEach(([,e])=>observer.observe(e));document.querySelectorAll("[data-copy-target]").forEach(b=>b.addEventListener("click",async()=>{{const t=document.getElementById(b.dataset.copyTarget);t.focus();t.select();try{{await navigator.clipboard.writeText(t.value)}}catch(_e){{document.execCommand("copy")}}const o=b.textContent;b.textContent="Copied";setTimeout(()=>b.textContent=o,1400)}}));</script></body></html>''', encoding="utf-8")


def write_html(
    *,
    html_path: Path,
    output_dir: Path,
    rows: list[dict[str, Any]],
    alphas: dict[tuple[str, str], dict[str, float]],
    bootstrap_reps: int,
) -> None:
    """Write the Mistral report using the Qwen report's browsing/table layout."""
    sections: list[str] = []
    sidebar: list[str] = []
    for partition in PARTITIONS:
        label = partition.replace("_", " ").title()
        section_id = partition.replace("_", "-")
        sidebar.append(f'<a href="#{section_id}">{html.escape(label)}</a>')
        directory = PARTITION_OUTPUT_DIRS[partition]
        cards: list[str] = []
        for figure_id, title, stems in figure_specs(partition):
            primary, *auxiliary = stems
            primary_base = f"mistral_sft_comparison/{directory}/{primary}"
            extra = ""
            if auxiliary:
                detail_label = "Private-channel histogram" if title == "Opinion-distribution histograms" else "Auxiliary correlation renderings"
                extra = f'<details class="diagnostics"><summary>{detail_label}</summary>' + "".join(
                    f'<a class="asset" href="mistral_sft_comparison/{directory}/{name}.png"><img loading="lazy" src="mistral_sft_comparison/{directory}/{name}.png" alt="{html.escape(figure_id)} auxiliary figure"></a>'
                    for name in auxiliary
                ) + "</details>"
            cards.append(
                f'<article class="card" id="{section_id}-{figure_id.lower()}">'
                f'<h3>{html.escape(figure_id)} · {html.escape(title)}</h3>'
                f'<a class="asset" href="{primary_base}.png"><img loading="lazy" src="{primary_base}.png" alt="{html.escape(figure_id)} {html.escape(title)}"></a>'
                f'<p><a href="{primary_base}.png">PNG</a> · <a href="{primary_base}.pdf">PDF</a> · <a href="{primary_base}.csv">CSV</a></p>{extra}</article>'
            )
        sections.append(
            f'<section id="{section_id}"><h2>{html.escape(label)}</h2>'
            '<p class="subtitle">Mistral base and the split-matched Mistral SFT adapter are evaluated on the supplied Ruixuan mask and Mistral-SFT human/validity reference. M1/A3 and M2/A4 include full dynamics and next-message prediction; dynamics panels use full dynamics.</p>'
            + "".join(cards) + "</section>"
        )
    alpha_rows = "".join(
        f'<tr><td>{html.escape(partition.replace("_", " "))}</td><td>{html.escape(version)}</td><td>{shared.finite(values.get("alpha_public")):.6f}</td><td>{shared.finite(values.get("alpha_private")):.6f}</td><td>{shared.finite(values.get("fit_prefix_count")):.0f}</td></tr>'
        for (partition, version), values in sorted(alphas.items())
    )
    rebuttal_tables, rebuttal_markdown, lower_count, comparison_count = rebuttal_summary(rows, output_dir)
    command = html.escape(
        'export PYTHON=python\n'
        'export PYTHONPATH="$PWD/src"\n'
        'export MPLCONFIGDIR=outputs/tmp/codex-mpl-v5-mistral\n'
        '"$PYTHON" neurips_2026/sean_reproduce/scripts/generate_v5_ruixuan_mistral_sft_comparison.py'
    )
    document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mistral-7B SFT — v5 Ruixuan</title><style>
:root {{ color-scheme:light; --ink:#26384b; --muted:#5c6d7d; --line:#d8e1eb; --paper:#f6f8fb; --accent:#075d8c; --active:#dceefa; }}
* {{ box-sizing:border-box; }} body {{ margin:0; font:15px/1.5 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--ink); background:#fff; }}
aside {{ position:fixed; inset:0 auto 0 0; width:250px; padding:25px 18px; overflow:auto; border-right:1px solid var(--line); background:var(--paper); }} aside h1 {{ font-size:18px; line-height:1.3; margin:0 0 7px; }} aside p {{ color:var(--muted); font-size:13px; margin:0 0 18px; }} aside a {{ display:block; color:#33526b; text-decoration:none; padding:8px 10px; border-radius:6px; margin:2px 0; }} aside a.active {{ color:#034c73; background:var(--active); font-weight:700; }}
main {{ max-width:1760px; margin-left:250px; padding:30px 42px 64px; }} h2 {{ scroll-margin-top:20px; border-bottom:2px solid var(--line); padding-bottom:8px; margin-top:48px; }} h3 {{ margin:0 0 10px; font-size:21px; }}
.notice {{ padding:14px 17px; border-left:4px solid #c77916; background:#fff8e9; margin:18px 0; }} .subtitle {{ color:var(--muted); max-width:1100px; }} code,pre {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }} pre {{ white-space:pre-wrap; overflow-wrap:anywhere; background:#172430; color:#e9f3f9; padding:14px; border-radius:8px; }}
.card {{ border:1px solid var(--line); border-radius:10px; padding:20px; margin:22px 0; background:#fff; box-shadow:0 1px 3px #1526370d; }} .asset {{ display:block; }} img {{ width:100%; height:auto; border:1px solid #e4eaf0; background:#fff; }} .card p a {{ color:var(--accent); font-weight:650; }}
.diagnostics {{ margin-top:14px; }} .diagnostics summary {{ cursor:pointer; color:var(--accent); font-weight:650; }} .diagnostics .asset {{ margin-top:14px; }} table {{ border-collapse:collapse; width:100%; max-width:850px; }} th,td {{ text-align:left; padding:7px 9px; border-bottom:1px solid var(--line); }} th {{ background:var(--paper); }} .rebuttal-table {{ max-width:none; font-size:13px; }} .rebuttal-table td {{ white-space:nowrap; font-variant-numeric:tabular-nums; }} .table-scroll {{ overflow-x:auto; border:1px solid var(--line); border-radius:8px; margin:0 0 24px; }} textarea {{ display:block; width:100%; min-height:760px; resize:vertical; padding:12px; border:1px solid var(--line); border-radius:8px; font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace; color:var(--ink); background:#fbfcfe; }} .copy-button {{ border:0; border-radius:6px; padding:8px 11px; margin:0 0 8px; background:var(--accent); color:#fff; cursor:pointer; font:inherit; font-weight:650; }} .copy-button:hover {{ background:#034c73; }}
@media (max-width:900px) {{ aside {{ position:static; width:auto; border-right:0; border-bottom:1px solid var(--line); }} aside a {{ display:inline-block; }} main {{ margin:0; padding:24px 16px; }} }}
</style></head><body>
<aside><h1>Mistral-7B SFT comparison</h1><p>v5 Ruixuan masks · Mistral base/SFT</p><a href="#overview">Overview</a><a href="#rebuttal-summary">Rebuttal table</a>{''.join(sidebar)}<a href="#reproduce">Reproduce</a></aside>
<main><section id="overview"><h2>Overview</h2><p>This report tests whether SFT improves Mistral-7B on the three stance-error families, both public/private channels, group/topic generalization, full dynamics, next-message prediction, and opinion convergence, using the supplied Ruixuan v5 surface.</p>
<div class="notice"><strong>Comparison contract.</strong> The canonical base package and the new Mistral delivery have different human/validity files. Both model outputs are evaluated against the matching Mistral-SFT human CSV and validity labels for a partition; every learned-model aggregate is then restricted to the base/SFT common usable-prefix intersection. Group partitions use the group-SFT human surface and topic partitions use the topic-SFT human surface. Coverage remains explicit in <a href="mistral_sft_comparison/csv/coverage_summary.csv">coverage_summary.csv</a>.</div>
<p><strong>Unavailable family:</strong> the delivery has no <code>llm_report_v0_bd.csv</code>, <code>llm_report_v1_5_bd.csv</code>, or Jina/ROUGE aggregate reports, so A1/A2 beyond-stance panels are intentionally absent.</p><p>Summary metrics: <a href="mistral_sft_comparison/csv/summary_metrics.csv">CSV</a> · Per-prefix status: <a href="mistral_sft_comparison/csv/per_prefix_status.csv">CSV</a>.</p></section>
<section id="rebuttal-summary"><h2>Rebuttal-ready held-out results</h2><p>Mistral SFT lowers {lower_count}/{comparison_count} public/private M1/A3 error values on the matched group- and topic-generalization test sets. The tables follow the paper panels, so M1/A3 has both full dynamics and next-message rows while M3/A5, M4/A8, and M5/A9 use full dynamics.</p>{rebuttal_tables}<h3>Copyable Markdown</h3><p>Use this directly in the NeurIPS rebuttal response field.</p><button class="copy-button" type="button" data-copy-target="rebuttal-markdown">Copy Markdown</button><textarea id="rebuttal-markdown" rows="50" readonly spellcheck="false">{html.escape(rebuttal_markdown)}</textarea></section>
{''.join(sections)}
<section id="reproduce"><h2>Reproduce</h2><pre>{command}</pre><p>Outputs are written under <code>mistral_sft_comparison/</code>; this HTML page remains at <code>v5_ruixuan/SFT_Mistral.html</code>.</p><h3>DeGroot fits on the common reference surface</h3><table><thead><tr><th>Partition</th><th>Simulation</th><th>α public</th><th>α private</th><th>Human prefixes</th></tr></thead><tbody>{alpha_rows}</tbody></table></section>
</main><script>
const links=[...document.querySelectorAll('aside a[href^="#"]')]; const targets=links.map(a=>[a,document.querySelector(a.getAttribute('href'))]).filter(([,e])=>e);
const observer=new IntersectionObserver(entries=>{{const visible=entries.filter(e=>e.isIntersecting).sort((a,b)=>b.intersectionRatio-a.intersectionRatio);if(!visible.length)return;const active=visible[0].target.id;links.forEach(a=>a.classList.toggle('active',a.getAttribute('href')==='#'+active));}},{{rootMargin:'-12% 0px -72% 0px',threshold:[0,.15,.4]}});targets.forEach(([,e])=>observer.observe(e));
document.querySelectorAll('[data-copy-target]').forEach(button=>button.addEventListener('click',async()=>{{const target=document.getElementById(button.dataset.copyTarget);target.focus();target.select();try{{await navigator.clipboard.writeText(target.value);}}catch(_error){{document.execCommand('copy');}}const original=button.textContent;button.textContent='Copied';setTimeout(()=>button.textContent=original,1400);}}));
</script></body></html>'''
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(document, encoding="utf-8")


def write_readme(*, output_dir: Path, mistral_root: Path, base_root: Path, mask_root: Path, alphas: dict[tuple[str, str], dict[str, float]], bootstrap_reps: int) -> None:
    lines = [
        "# Mistral-7B v5 SFT comparison", "",
        "This bundle compares canonical `Mistral-7B-Instruct-v0.3` pre-SFT output with the supplied group/topic Mistral SFT adapters.", "",
        "## Controlled comparison contract", "",
        "- Masks are the existing Ruixuan v5 group/topic masks.",
        "- Both base and SFT are evaluated against the matching Mistral-SFT human/validity CSV for a partition.",
        "- Learned-model metrics use only prefixes with usable base and SFT results; see `csv/coverage_summary.csv`.",
        "- M1/A3 and M2/A4 include `v0` full dynamics and `v1_5` next-message prediction. M3/A5, M4/A8, M5/A9, and A6/A7 use `v0` as in the paper.",
        "- A1/A2 are unavailable: there are no root-level `llm_report_{v0,v1_5}_bd.csv` aggregate inputs.", "",
        "## Inputs", "",
        f"- Mistral SFT root: `{mistral_root}`", f"- Canonical pre-SFT Mistral root: `{base_root}`", f"- Ruixuan v5 masks: `{mask_root}`", "",
        "## DeGroot fits", "", "| Partition | Simulation | alpha_public | alpha_private | human prefixes |", "|---|---|---:|---:|---:|",
    ]
    for (partition, version), values in sorted(alphas.items()):
        lines.append(f"| {partition} | {version} | {shared.finite(values.get('alpha_public')):.6f} | {shared.finite(values.get('alpha_private')):.6f} | {shared.finite(values.get('fit_prefix_count')):.0f} |")
    lines.extend(["", "## Reproduce", "", "```bash", "export PYTHON=python", "export PYTHONPATH=\"$PWD/src\"", "export MPLCONFIGDIR=outputs/tmp/codex-mpl-v5-mistral", f"\"$PYTHON\" neurips_2026/sean_reproduce/scripts/generate_v5_ruixuan_mistral_sft_comparison.py --mistral-root \"{mistral_root}\" --base-root \"{base_root}\" --mask-root \"{mask_root}\" --output-dir \"$PWD/neurips_2026/sean_reproduce/v5_ruixuan/mistral_sft_comparison\" --html-path \"$PWD/neurips_2026/sean_reproduce/v5_ruixuan/SFT_Mistral.html\" --bootstrap-reps {bootstrap_reps}", "```", ""])
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def read_alphas(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {(str(row["partition"]), str(row["simulation_version"])): {field: shared.finite(value) for field, value in row.items() if field not in {"partition", "simulation_version"}} for row in csv.DictReader(handle)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render controlled Mistral-7B base/SFT v5 Ruixuan figures.")
    parser.add_argument("--mistral-root", type=Path, default=DEFAULT_MISTRAL_ROOT)
    parser.add_argument("--base-root", type=Path, default=DEFAULT_BASE_ROOT)
    parser.add_argument("--mask-root", type=Path, default=DEFAULT_MASK_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--html-path", type=Path, default=DEFAULT_HTML_PATH)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--reuse-summary", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mistral_root, base_root, mask_root = args.mistral_root.expanduser().resolve(), args.base_root.expanduser().resolve(), args.mask_root.expanduser().resolve()
    output_dir, html_path = args.output_dir.expanduser().resolve(), args.html_path.expanduser().resolve()
    if not all(path.is_dir() for path in (mistral_root, base_root, mask_root)):
        raise SystemExit("Mistral SFT root, canonical base root, and mask root must all exist.")
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_shared_helpers()
    configure_renderer()
    shared.install_profile_override(qwen_root=mistral_root, llama_root=mistral_root, llama_base_root=base_root, mask_root=mask_root)
    summary_path, status_path, alpha_path = output_dir / "csv" / "summary_metrics.csv", output_dir / "csv" / "per_prefix_status.csv", output_dir / "csv" / "degroot_fit_alphas.csv"
    if args.reuse_summary:
        if not all(path.is_file() for path in (summary_path, status_path, alpha_path)):
            raise SystemExit("--reuse-summary requires the existing summary, status, and alpha CSVs.")
        with summary_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        with status_path.open(encoding="utf-8", newline="") as handle:
            statuses = list(csv.DictReader(handle))
        alphas = read_alphas(alpha_path)
    else:
        rows, statuses, alphas = shared.compute_summary(qwen_root=mistral_root, llama_root=mistral_root, llama_base_root=base_root, mask_root=mask_root, bootstrap_reps=max(0, args.bootstrap_reps))
        for row in rows:
            row["human_reference"] = f"Mistral SFT {('group' if is_group(str(row['partition'])) else 'topic')} human surface"
        for status in statuses:
            status["human_reference"] = str(status.get("human_reference", "")).replace("Qwen SFT", "Mistral SFT").replace("Qwen", "Mistral")
            status["message"] = str(status.get("message", "")).replace("Qwen SFT", "Mistral SFT").replace("Qwen", "Mistral")
        shared.write_csv(summary_path, rows)
        shared.write_csv(status_path, statuses)
        shared.write_csv(alpha_path, [{"partition": partition, "simulation_version": version, **values} for (partition, version), values in sorted(alphas.items())])
    shared.write_csv(output_dir / "csv" / "coverage_summary.csv", shared.summarize_coverage(statuses))
    render_all(rows=rows, alphas=alphas, mistral_root=mistral_root, base_root=base_root, mask_root=mask_root, output_dir=output_dir, bootstrap_reps=max(0, args.bootstrap_reps))
    write_readme(output_dir=output_dir, mistral_root=mistral_root, base_root=base_root, mask_root=mask_root, alphas=alphas, bootstrap_reps=max(0, args.bootstrap_reps))
    write_html(
        html_path=html_path,
        output_dir=output_dir,
        rows=rows,
        alphas=alphas,
        bootstrap_reps=max(0, args.bootstrap_reps),
    )
    print(html_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
