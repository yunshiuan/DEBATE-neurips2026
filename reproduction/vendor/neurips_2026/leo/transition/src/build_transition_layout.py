from __future__ import annotations

import argparse
import io
import shutil
import sys
import textwrap
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from pypdf import PdfReader, PdfWriter

from utils import (
    collection_golden_mask_path,
    golden_group_mask_v0_path,
    human_transition_csv_filename,
    load_eval_dataframe,
    metric_version_filters_invalid,
    metric_version_uses_rejudged_sidecars,
    normalize_metric_version,
    prepare_golden_dps,
    result_root,
    train_or_test,
)


SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neurips_2026.sean_reproduce.result_version_config import (
    include_close_far_transitions,
    LLAMA_V4_BREADTH_GROUP_SFT_MODEL,
    LLAMA_V4_BREADTH_TOPIC_SFT_MODEL,
    LLAMA_V4_DEPTH_GROUP_SFT_MODEL,
    LLAMA_V4_DEPTH_TOPIC_SFT_MODEL,
)
from neurips_2026.sean_reproduce.v3.masks import repo_relative_display_path, wrapped_path_lines
from neurips_2026.chengtao.src.radar.generate_plots import get_model_color

DEFAULT_WORK_ROOT = SCRIPT_ROOT / "work"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "neurips_2026" / "sean_reproduce" / "v0" / "transition"


BASE_MODELS = {
    "Mistral-7B-Instruct-v0.3",
    "Llama-3.1-Tulu-3-8B-SFT",
    "Llama-3.1-70B-Instruct",
    "Qwen2.5-32B-Instruct",
    "gpt-4o-mini-2024-07-18",
    "gpt-5-mini",
    "Llama-3.1-8B-Instruct",
    "gemini-2.5-flash-lite",
}
POST_TRAINING_MODELS = {
    "ft:gpt-4o-mini-2024-07-18:camer:sft-20251218-depth-topic-5epochs:CoFxFPcv",
    "ft:gpt-4o-mini-2024-07-18:camer:sft-20251218-depth-group-5epochs:CoFzijg0",
    "ft:Llama-3.1-8B-Instruct-SFT-20251218:depth-group-5epochs",
    "ft:Llama-3.1-8B-Instruct-SFT-20251218:depth-topic-5epochs",
    "ft:Llama-3.1-8B-Instruct-DPO-20260105:depth-group-5epochs",
    "ft:Llama-3.1-8B-Instruct-DPO-20260105:depth-topic-5epochs",
    "ft:Llama-3.1-8B-SFT-20260210:breadth-group-3epochs-unsloth",
    "ft:Llama-3.1-8B-SFT-20260210:breadth-topic-3epochs-unsloth",
    "ft:Llama-3.1-8B-Instruct-DPO-20260212:breadth-topic-5epochs",
    "ft:Llama-3.1-8B-Instruct-DPO-20260211:breadth-group-5epochs",
    "ft:gpt-4o-mini-2024-07-18:camer:sft-20260112-breadth-group-5epochs:CxGGMXoF",
    "ft:gpt-4o-mini-2024-07-18:camer:sft-20260112-breadth-topic-5epochs:CxG6nPck",
}
POST_TRAINING_MODELS_V4 = {
    "ft:gpt-4o-mini-2024-07-18:camer:sft-20251218-depth-topic-5epochs:CoFxFPcv",
    "ft:gpt-4o-mini-2024-07-18:camer:sft-20251218-depth-group-5epochs:CoFzijg0",
    LLAMA_V4_DEPTH_GROUP_SFT_MODEL,
    LLAMA_V4_DEPTH_TOPIC_SFT_MODEL,
    LLAMA_V4_BREADTH_GROUP_SFT_MODEL,
    LLAMA_V4_BREADTH_TOPIC_SFT_MODEL,
}
COLLECTION_TYPES = ["breadth", "depth"]
SPLIT_TYPES = ["overall", "group", "topic"]
PARTITION_TYPES = ["train", "test"]
PANELS_PER_PAGE = 9
PARTITION_ORDER = {"test": 0, "train": 1, None: 2}
HUMAN_MODEL_DIR = "Llama-3.1-8B-Instruct"
HUMAN_CSV_FILENAME = human_transition_csv_filename()
V4_TRANSITION_BLUE_CMAP = LinearSegmentedColormap.from_list(
    "v4_transition_blue",
    ["#F4F8FC", "#D9E6F3", "#9AB8D8", "#5E89B7", "#2F5D8A"],
)
TRANSITION_HUMAN_COLOR = "#4D4D4D"
TRANSITION_DEGROOT_COLOR = "#9A9A9A"
TRANSITION_FALLBACK_COLOR = "#2F5D8A"

PLOT_CONFIG = {
    "cf": {
        "raw_subdir": "close_far",
        "result_subdir": "close_far",
        "artifact_prefix": "close_far",
        "stypes": ["overall", "pos", "neg", "neu"],
    },
    "pn": {
        "raw_subdir": "polarized_neutral",
        "result_subdir": "polarized_neutral",
        "artifact_prefix": "polarized_neutral",
        "stypes": [None],
    },
    "ind": {
        "raw_subdir": "individual_likert",
        "result_subdir": "individual_likert",
        "artifact_prefix": "individual_likert",
        "stypes": [None],
    },
}
FINAL_REPORT_FILENAMES = {
    "cf": "report_transition_cf.pdf",
    "pn": "report_transition_pn.pdf",
    "ind": "report_transition_ind.pdf",
}


def _scope_label(value_type: str) -> str:
    return "private" if value_type == "slider_value" else "public"


def _collection_report_filename(
    plot_type: str,
    collection_type: str,
    value_type: str,
    output_suffix: str = "",
) -> str:
    stem = FINAL_REPORT_FILENAMES[plot_type].removesuffix('.pdf')
    return f"{stem}_{_scope_label(value_type)}_{collection_type}{output_suffix}.pdf"


def _validity_label(metric_version: str) -> str:
    metric_version = normalize_metric_version(metric_version)
    if metric_version.startswith("v4") or metric_version == "v3":
        return metric_version
    if metric_version == "v0_fixed":
        return "v0_fixed"
    return "v0"


def _repo_root() -> Path:
    return REPO_ROOT


def _path_block(label: str, paths: list[Path]) -> str:
    lines = [f"{label}:"]
    for path in paths:
        lines.extend(wrapped_path_lines(path, width=88, indent="  - ", subsequent_indent="    "))
    return "\n".join(lines)


def _results_matrix_dir(results_root: Path, plot_type: str) -> Path:
    config = PLOT_CONFIG[plot_type]
    return results_root / "matrices" / config["result_subdir"]


def _report_intro_lines(
    plot_type: str,
    results_root: Path,
    *,
    metric_version: str,
    value_type: str,
    group_validity_threshold: float | None,
    use_split_masks: bool = True,
) -> list[str]:
    metric_version = normalize_metric_version(metric_version)
    mask_path = golden_group_mask_v0_path()
    collection_mask_paths = [collection_golden_mask_path(collection_type) for collection_type in COLLECTION_TYPES]
    split_mask_paths = [
        collection_golden_mask_path(collection_type).parent / f"{collection_type}_{partition}_{split}_mask_golden_v0.txt"
        for collection_type in COLLECTION_TYPES
        for split in ("group", "topic")
        for partition in ("train", "test")
    ]
    repo_root = _repo_root()
    eval_root = repo_root / "neurips_2026" / "chengtao" / "result" / "eval_v3" / "human_llm"
    source_label = "initial-vs-final opinion stance" if value_type == "slider_value" else "round-1 to round-3 tweet Likert"
    common_lines = [
        _path_block("Aggregate golden mask used", [mask_path]),
        _path_block("Collection golden masks used", collection_mask_paths),
    ]
    if use_split_masks:
        common_lines.extend(
            [
                _path_block("Split masks used", split_mask_paths),
                f"Input eval CSV root: {repo_relative_display_path(eval_root)}",
                f"Published results root: {repo_relative_display_path(results_root)}",
                "Included groups and train/test partitions: canonical golden_group_v0 txt masks only.",
                f"Metric version: {metric_version}",
                f"Transition source: {source_label}",
            ]
        )
    else:
        common_lines.extend(
            [
                "Split masks used: none; this variant keeps only the collection golden mask.",
                f"Input eval CSV root: {repo_relative_display_path(eval_root)}",
                f"Published results root: {repo_relative_display_path(results_root)}",
                "Included groups: collection golden masks only; no additional group/topic train/test split masks applied.",
                f"Metric version: {metric_version}",
                f"Transition source: {source_label}",
            ]
        )
    if metric_version == "v3" or metric_version.startswith("v4"):
        common_lines.append(
            f"{metric_version} applies canonical rejudged validity sidecars from rejudged_labels/{{initial_post,tweet}} and excludes INVALID rows from matrix counts."
        )
    elif metric_version == "v0_fixed":
        common_lines.append(
            "v0_fixed uses the existing validity labels only and excludes INVALID rows from matrix counts without applying rejudged sidecars."
        )
    else:
        common_lines.append(
            "v0 keeps the legacy all-rows counting rule: rows with parseable Likert values are counted even when marked INVALID."
        )
    if group_validity_threshold is not None:
        common_lines.append(
            f"Group validity filter: keep groups whose human valid share on the transition-relevant rows is at least {group_validity_threshold:.2f}."
        )

    if plot_type == "cf":
        matrix_lines = [
            "Matrix definition: 5x5 close/far transition matrix.",
            "Each axis counts how many of the 4 players are close versus far from the topic-bias bucket.",
            "Close/far is computed from the selected transition source for each player's start and end stance.",
            "The neg/neu/pos slices are bucketed by the topic-bias prior; overall is their sum.",
        ]
    elif plot_type == "pn":
        matrix_lines = [
            "Matrix definition: 5x5 polarized/neutral composition transition matrix.",
            "Each axis counts how many of the 4 players are polarized versus neutral.",
            "Polarized means Likert in {1,2,5,6}; neutral means {0,3,4}.",
            "The matrix compares start-versus-end group compositions from the selected transition source.",
        ]
    else:
        matrix_lines = [
            "Matrix definition: 6x6 individual Likert transition matrix.",
            "Each cell counts one player moving from a start-state Likert bin to an end-state Likert bin.",
            "Values come from the selected transition source on the kept groups.",
            "This report aggregates those player-level transitions across breadth and depth splits.",
        ]

    return [f"Transition Report: {plot_type.upper()}", *common_lines, *matrix_lines]


def _render_intro_page(
    plot_type: str,
    results_root: Path,
    *,
    metric_version: str,
    value_type: str,
    group_validity_threshold: float | None,
    use_split_masks: bool = True,
) -> plt.Figure:
    lines = _report_intro_lines(
        plot_type,
        results_root,
        metric_version=metric_version,
        value_type=value_type,
        group_validity_threshold=group_validity_threshold,
        use_split_masks=use_split_masks,
    )
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.axis("off")
    ax.text(0.03, 0.96, lines[0], fontsize=20, fontweight="bold", va="top", ha="left", transform=ax.transAxes)
    body = "\n\n".join(lines[1:])
    ax.text(0.03, 0.88, body, fontsize=12, va="top", ha="left", transform=ax.transAxes, wrap=True)
    return fig


def _intro_pdf_reader(
    plot_type: str,
    results_root: Path,
    *,
    metric_version: str,
    value_type: str,
    group_validity_threshold: float | None,
    use_split_masks: bool = True,
) -> PdfReader:
    fig = _render_intro_page(
        plot_type,
        results_root,
        metric_version=metric_version,
        value_type=value_type,
        group_validity_threshold=group_validity_threshold,
        use_split_masks=use_split_masks,
    )
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    buffer.seek(0)
    return PdfReader(buffer)


def _split_sort_rank(split_type: str) -> int:
    if split_type == "overall":
        return 2
    if split_type == "group":
        return 0
    if split_type == "topic":
        return 1
    return 3


def _section_sort_key(section: tuple[str, str, str | None]) -> tuple[int, int, int, str]:
    collection_type, split_type, partition_type = section
    return (
        COLLECTION_TYPES.index(collection_type),
        PARTITION_ORDER.get(partition_type, 3),
        _split_sort_rank(split_type),
        "" if partition_type is None else str(partition_type),
    )


def _validity_column_for_path(csv_path: Path, df: pd.DataFrame, respect_validity_labels: bool) -> str | None:
    if respect_validity_labels and "effective_validity" in df.columns:
        return "effective_validity"
    if "opinion_llm_" in csv_path.name and "llm_validity" in df.columns:
        return "llm_validity"
    if "opinion_human_" in csv_path.name and "human_validity" in df.columns:
        return "human_validity"
    if "validity" in df.columns:
        return "validity"
    return None


def _transition_mask(df: pd.DataFrame, value_type: str) -> pd.Series:
    event_type = df.get("event_type", pd.Series(index=df.index, dtype=object)).astype(str)
    if value_type == "likert_pred":
        rounds = pd.to_numeric(df.get("chat_round_order"), errors="coerce")
        return (event_type == "tweet") & rounds.isin([1.0, 3.0])
    if value_type == "slider_value":
        return event_type.isin(["Initial Opinion", "Post Opinion"])
    raise ValueError(f"Unsupported value_type={value_type}")


def _group_validity_share(data_prefix: str, value_type: str) -> float:
    csv_path = result_root() / data_prefix / HUMAN_MODEL_DIR / HUMAN_CSV_FILENAME
    if not csv_path.exists():
        return float("nan")

    df = load_eval_dataframe(
        csv_path,
        respect_validity_labels=True,
        apply_rejudged_sidecars=False,
        likert_col="likert_pred",
    )
    if "event_type" not in df.columns or "effective_validity" not in df.columns:
        return float("nan")

    mask = _transition_mask(df, value_type)
    validity = df.loc[mask, "effective_validity"].astype(str).str.upper().str.strip()
    labelled = validity.isin(["VALID", "INVALID"])
    if not labelled.any():
        return float("nan")
    valid = (validity[labelled] == "VALID").sum()
    return float(valid) / float(labelled.sum())


def _section_prefixes(
    collection_type: str,
    split_type: str,
    partition_type: str | None,
    *,
    metric_version: str,
    value_type: str,
    group_validity_threshold: float | None,
    use_split_masks: bool = True,
) -> list[str]:
    prefixes = prepare_golden_dps(collection_type)
    if metric_version_filters_invalid(metric_version) and group_validity_threshold is not None:
        filtered_prefixes: list[str] = []
        for data_prefix in prefixes:
            validity_share = _group_validity_share(data_prefix, value_type)
            if pd.notna(validity_share) and validity_share >= group_validity_threshold:
                filtered_prefixes.append(data_prefix)
        prefixes = filtered_prefixes

    if use_split_masks and split_type in {"group", "topic"} and partition_type is not None:
        prefixes = [
            data_prefix
            for data_prefix in prefixes
            if train_or_test(data_prefix, split_type, collection_type) == partition_type
        ]
    return prefixes


def _aggregate_validity_stats(
    prefixes: list[str],
    value_type: str,
    *,
    respect_validity_labels: bool,
    apply_rejudged_sidecars: bool,
) -> tuple[int, int]:
    valid_count = 0
    labelled_count = 0
    for data_prefix in prefixes:
        csv_path = result_root() / data_prefix / HUMAN_MODEL_DIR / HUMAN_CSV_FILENAME
        if not csv_path.exists():
            continue
        df = load_eval_dataframe(
            csv_path,
            respect_validity_labels=respect_validity_labels,
            apply_rejudged_sidecars=apply_rejudged_sidecars,
            likert_col="likert_pred",
        )
        validity_column = _validity_column_for_path(csv_path, df, respect_validity_labels)
        if validity_column is None:
            continue
        mask = _transition_mask(df, value_type)
        validity = df.loc[mask, validity_column].astype(str).str.upper().str.strip()
        labelled = validity.isin(["VALID", "INVALID"])
        labelled_count += int(labelled.sum())
        valid_count += int((validity[labelled] == "VALID").sum())
    return valid_count, labelled_count


def _summary_table_rows(
    *,
    metric_version: str,
    value_type: str,
    group_validity_threshold: float | None,
    collection_filter: str | None = None,
    overall_only: bool = False,
    use_split_masks: bool = True,
) -> list[list[str]]:
    return [
        list(row)
        for row in _summary_table_rows_cached(
            metric_version,
            value_type,
            group_validity_threshold,
            collection_filter,
            overall_only,
            use_split_masks,
        )
    ]


@lru_cache(maxsize=None)
def _summary_table_rows_cached(
    metric_version: str,
    value_type: str,
    group_validity_threshold: float | None,
    collection_filter: str | None,
    overall_only: bool,
    use_split_masks: bool,
) -> tuple[tuple[str, ...], ...]:
    rows: list[list[str]] = []
    section_collection_types = [collection_filter] if collection_filter is not None else COLLECTION_TYPES
    sections: list[tuple[str, str, str | None]] = []
    for collection_type in section_collection_types:
        if overall_only:
            sections.append((collection_type, "overall", None))
            continue
        for split_type in SPLIT_TYPES:
            partitions = [None] if split_type == "overall" else PARTITION_TYPES
            for partition_type in partitions:
                sections.append((collection_type, split_type, partition_type))

    for collection_type, split_type, partition_type in sorted(sections, key=_section_sort_key):
        prefixes = _section_prefixes(
            collection_type,
            split_type,
            partition_type,
            metric_version=metric_version,
            value_type=value_type,
            group_validity_threshold=group_validity_threshold,
            use_split_masks=use_split_masks,
        )
        v0_valid, labelled_count = _aggregate_validity_stats(
            prefixes,
            value_type,
            respect_validity_labels=False,
            apply_rejudged_sidecars=False,
        )
        current_valid, _ = _aggregate_validity_stats(
            prefixes,
            value_type,
            respect_validity_labels=metric_version_filters_invalid(metric_version),
            apply_rejudged_sidecars=metric_version_uses_rejudged_sidecars(metric_version),
        )
        v0_rate = (100.0 * v0_valid / labelled_count) if labelled_count else float("nan")
        current_rate = (100.0 * current_valid / labelled_count) if labelled_count else float("nan")
        partition_label = "overall" if partition_type is None else partition_type
        delta_valid = current_valid - v0_valid
        rows.append(
            [
                collection_type,
                split_type,
                partition_label,
                str(len(prefixes)),
                str(labelled_count),
                "n/a" if np.isnan(v0_rate) else f"{v0_rate:.1f}% ({v0_valid})",
                "n/a" if np.isnan(current_rate) else f"{current_rate:.1f}% ({current_valid})",
                f"{delta_valid:+d}",
            ]
        )
    return tuple(tuple(row) for row in rows)


def _render_summary_page(
    plot_type: str,
    results_root: Path,
    *,
    metric_version: str,
    value_type: str,
    group_validity_threshold: float | None,
    collection_filter: str | None = None,
    overall_only: bool = False,
    use_split_masks: bool = True,
) -> plt.Figure:
    metric_version = normalize_metric_version(metric_version)
    del plot_type, results_root
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.axis("off")
    ax.text(0.03, 0.97, "Transition Validity Summary", fontsize=18, fontweight="bold", va="top", ha="left", transform=ax.transAxes)
    note_lines = [
        "Human transition rows only, aggregated over the groups included in this report.",
        "v0 uses the legacy all-rows counting rule based on parseable Likert labels.",
        f"{_validity_label(metric_version)} valid rate applies the active report version's validity handling.",
        f"Delta valid rows = {_validity_label(metric_version)} valid count minus v0 valid count.",
    ]
    if metric_version == "v3" or metric_version.startswith("v4"):
        note_lines.append(f"{metric_version} uses canonical rejudged sidecar overrides where available.")
    elif metric_version == "v0_fixed":
        note_lines.append("v0_fixed uses existing validity labels only and does not apply rejudged sidecars.")
    if metric_version_filters_invalid(metric_version) and group_validity_threshold is not None:
        note_lines.append(
            f"Included groups already satisfy the {_validity_label(metric_version)} human-valid-share threshold of {group_validity_threshold:.2f}."
        )
    if not use_split_masks:
        note_lines.append("Train/test rows are shown for layout parity only; no additional split masks were applied.")
    ax.text(0.03, 0.90, "\n".join(note_lines), fontsize=11, va="top", ha="left", transform=ax.transAxes)

    table_bottom = 0.08
    table_top = max(0.64, 0.88 - (0.03 * len(note_lines)))

    table = ax.table(
        cellText=_summary_table_rows(
            metric_version=metric_version,
            value_type=value_type,
            group_validity_threshold=group_validity_threshold,
            collection_filter=collection_filter,
            overall_only=overall_only,
            use_split_masks=use_split_masks,
        ),
        colLabels=[
            "collection",
            "split",
            "partition",
            "groups",
            "labelled rows",
            "v0 valid rate",
            f"{_validity_label(metric_version)} valid rate",
            "delta valid",
        ],
        cellLoc="center",
        loc="center",
        bbox=[0.03, table_bottom, 0.94, table_top - table_bottom],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.3)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e8eef7")
    return fig


def _summary_pdf_reader(
    plot_type: str,
    results_root: Path,
    *,
    metric_version: str,
    value_type: str,
    group_validity_threshold: float | None,
    collection_filter: str | None = None,
    overall_only: bool = False,
    use_split_masks: bool = True,
) -> PdfReader:
    fig = _render_summary_page(
        plot_type,
        results_root,
        metric_version=metric_version,
        value_type=value_type,
        group_validity_threshold=group_validity_threshold,
        collection_filter=collection_filter,
        overall_only=overall_only,
        use_split_masks=use_split_masks,
    )
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    buffer.seek(0)
    return PdfReader(buffer)


def _normalize_model_id(model_name: str) -> str:
    return (model_name.split("/", 1)[1] if "/" in model_name else model_name).replace(":", "_")


def post_training_model_ids_for_metric_version(metric_version: str) -> set[str]:
    normalized_version = normalize_metric_version(metric_version)
    models = POST_TRAINING_MODELS_V4 if normalized_version.startswith("v4") else POST_TRAINING_MODELS
    return {_normalize_model_id(model_name) for model_name in models}


BASE_MODEL_IDS = {_normalize_model_id(model_name) for model_name in BASE_MODELS}
ACTIVE_POST_TRAINING_MODEL_IDS = post_training_model_ids_for_metric_version("v3")


def _plot_filename(plot_type: str, stype: str | None, ext: str) -> str:
    if plot_type == "cf":
        return f"cf_matrix_{stype}{ext}"
    if plot_type == "pn":
        return f"pn_matrix{ext}"
    return f"indiv_matrix{ext}"


def _section_root(base_dir: Path, collection_type: str, split_type: str, partition_type: str | None) -> Path:
    if split_type in {"group", "topic"}:
        return base_dir / collection_type / split_type / str(partition_type)
    return base_dir / collection_type / split_type


def _discover_model_ids(root: Path) -> list[str]:
    if not root.exists():
        return []
    names = [child.name for child in root.iterdir() if child.is_dir() and child.name not in {"human", "overall", "breadth", "depth"}]
    return sorted(names)


def _load_matrix(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        return np.loadtxt(path, delimiter=",", dtype=int)
    except Exception:
        return None


def _heatmap_axes_config(plot_type: str, value_type: str) -> tuple[list[str], str, str]:
    if plot_type == "cf":
        labels = ["0c4f", "1c3f", "2c2f", "3c1f", "4c0f"]
        if value_type == "slider_value":
            return labels, "Final opinion (close/far to topic bias)", "Initial opinion (close/far to topic bias)"
        return labels, "Tweet 3 (close/far to topic bias)", "Tweet 1 (close/far to topic bias)"
    if plot_type == "pn":
        labels = ["4p0n", "3p1n", "2p2n", "1p3n", "0p4n"]
        if value_type == "slider_value":
            return labels, "Final opinion composition", "Initial opinion composition"
        return labels, "Tweet 3 composition", "Tweet 1 composition"
    labels = ["1", "2", "3", "4", "5", "6"]
    if value_type == "slider_value":
        return labels, "Final opinion stance", "Initial opinion stance"
    return labels, "Round 3 tweet Likert", "Round 1 tweet Likert"


def _panel_model_id(label: str) -> str:
    normalized = str(label).strip()
    if normalized == "human":
        return "human"
    if normalized.startswith("base: "):
        return normalized.removeprefix("base: ").strip()
    if normalized.startswith(("sft: ", "dpo: ", "ft: ")):
        _prefix, model_id = normalized.split(": ", 1)
        return model_id.strip()
    return normalized


def _panel_base_color(label: str) -> str:
    model_id = _panel_model_id(label)
    if model_id == "human":
        return TRANSITION_HUMAN_COLOR
    if model_id in {"DeGroot fit", "DeGroot fit topic", "DeGroot fit group"} or model_id.startswith("DeGroot alpha="):
        return TRANSITION_DEGROOT_COLOR
    return str(get_model_color(model_id, TRANSITION_FALLBACK_COLOR))


def _draw_panel_title(ax, label: str, matrix: np.ndarray | None, *, color_theme: str | None = None) -> None:
    display_label = _display_panel_label(label)
    total = _matrix_total(matrix)
    title_fill = _panel_base_color(label) if color_theme == "v4" else None
    title_color = "black"
    title_fontsize = _panel_title_fontsize(display_label)
    if title_fill is not None:
        title_fontsize += 0.9
        ax.add_patch(
            Rectangle(
                (0.0, 1.055),
                1.0,
                0.15,
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
        1.13,
        display_label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=title_fontsize,
        color=title_color,
        zorder=3,
    )
    if total is not None:
        ax.text(
            0.5,
            1.03,
            f"n={total}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="black",
        )


def _draw_matrix_heatmap(
    ax,
    matrix: np.ndarray,
    title: str,
    label: str,
    plot_type: str,
    value_type: str,
    *,
    color_theme: str | None = None,
) -> None:
    labels, x_label, y_label = _heatmap_axes_config(plot_type, value_type)
    total = int(np.asarray(matrix).sum())
    percent_matrix = (matrix / total) if total else np.zeros_like(matrix, dtype=float)
    vmax = float(percent_matrix.max()) if total else 1.0
    if color_theme == "v4":
        color_map = V4_TRANSITION_BLUE_CMAP
    else:
        color_map = "Oranges" if "DeGroot" in title else "Blues"
    image = ax.imshow(percent_matrix, cmap=color_map, vmin=0.0, vmax=vmax)

    axis_len = matrix.shape[0]
    ax.set_xticks(np.arange(axis_len))
    ax.set_yticks(np.arange(axis_len))
    ax.set_xticklabels(labels[:axis_len])
    ax.set_yticklabels(labels[:axis_len])
    ax.set_xlabel(x_label, fontsize=8)
    ax.set_ylabel(y_label, fontsize=8)
    _draw_panel_title(ax, label, matrix, color_theme=color_theme)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_edgecolor("black")

    threshold = vmax * 0.55 if total else 0.0
    rows, cols = matrix.shape
    for row in range(rows):
        for col in range(cols):
            percentage = percent_matrix[row, col] * 100.0
            count = int(matrix[row, col])
            text_color = "white" if total and percent_matrix[row, col] >= threshold else "black"
            ax.text(
                col,
                row,
                f"{percentage:.0f}%\n({count})",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )

    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Share of transition counts", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)


def _draw_image(
    ax,
    image_path: Path,
    title: str,
    label: str,
    plot_type: str,
    value_type: str,
    matrix_fallback: np.ndarray | None = None,
    *,
    color_theme: str | None = None,
) -> None:
    image_loaded = False
    if image_path.exists() and image_path.stat().st_size > 0:
        try:
            ax.imshow(plt.imread(image_path))
            ax.axis("off")
            ax.set_title(title, fontsize=_panel_title_fontsize(title))
            image_loaded = True
        except Exception:
            image_loaded = False
    if not image_loaded and matrix_fallback is not None:
        _draw_matrix_heatmap(ax, matrix_fallback, title, label, plot_type, value_type, color_theme=color_theme)
    elif not image_loaded:
        ax.axis("off")
        ax.set_title(title, fontsize=_panel_title_fontsize(title))
        ax.text(0.5, 0.5, f"Missing\n{image_path.name}", ha="center", va="center", fontsize=8)


def _matrix_total(matrix: np.ndarray | None) -> int | None:
    if matrix is None:
        return None
    return int(np.asarray(matrix).sum())


def _normalized_frobenius_diff(model_matrix: np.ndarray | None, human_matrix: np.ndarray | None) -> float | None:
    if model_matrix is None or human_matrix is None or model_matrix.shape != human_matrix.shape:
        return None
    diff_norm = np.linalg.norm(model_matrix - human_matrix, ord="fro")
    human_norm = np.linalg.norm(human_matrix, ord="fro")
    return float(diff_norm if human_norm == 0 else diff_norm / human_norm)


def _model_category(model_id: str) -> str | None:
    if model_id in BASE_MODEL_IDS:
        return "base_model"
    if model_id in ACTIVE_POST_TRAINING_MODEL_IDS:
        return "post_training"
    if model_id.startswith("DeGroot alpha="):
        return "degroot"
    if model_id in {"DeGroot fit", "DeGroot fit topic", "DeGroot fit group"}:
        return "degroot"
    return None


def _ordered_sections(raw_root: Path) -> list[tuple[str, str, str | None]]:
    all_sections: list[tuple[str, str, str | None]] = []
    for collection_type in COLLECTION_TYPES:
        for split_type in SPLIT_TYPES:
            partitions = PARTITION_TYPES if split_type in {"group", "topic"} else [None]
            for partition_type in partitions:
                section_root = _section_root(raw_root, collection_type, split_type, partition_type)
                if section_root.exists():
                    all_sections.append((collection_type, split_type, partition_type))

    return sorted(set(all_sections), key=_section_sort_key)


def _restore_table_model_id(model_id: str) -> str:
    if model_id.startswith("DeGroot_alpha_"):
        return f"DeGroot alpha={model_id.removeprefix('DeGroot_alpha_')}"
    mapping = {
        "DeGroot_fit": "DeGroot fit",
        "DeGroot_fit_topic": "DeGroot fit topic",
        "DeGroot_fit_group": "DeGroot fit group",
    }
    return mapping.get(model_id, model_id)


def _parse_table_csv_name(plot_type: str, csv_path: Path) -> tuple[str, str, str | None, str, str | None] | None:
    parts = csv_path.stem.split("__")
    if len(parts) not in {4, 5}:
        return None

    collection_type = parts[0]
    split_type = parts[1]
    partition_type: str | None = None
    if split_type in {"group", "topic"}:
        if len(parts) != 5:
            return None
        partition_type = parts[2]
        model_id = _restore_table_model_id(parts[3])
        matrix_name = parts[4]
    else:
        if len(parts) != 4:
            return None
        model_id = _restore_table_model_id(parts[2])
        matrix_name = parts[3]

    if plot_type == "cf":
        if not matrix_name.startswith("cf_matrix_"):
            return None
        stype: str | None = matrix_name.removeprefix("cf_matrix_")
    elif plot_type == "pn":
        if matrix_name != "pn_matrix":
            return None
        stype = None
    else:
        if matrix_name != "indiv_matrix":
            return None
        stype = None

    return collection_type, split_type, partition_type, model_id, stype


def _table_index(plot_type: str, results_root: Path) -> dict[tuple[str, str, str | None, str, str | None], Path]:
    table_dir = _results_matrix_dir(results_root, plot_type) / "tables"
    index: dict[tuple[str, str, str | None, str, str | None], Path] = {}
    if not table_dir.exists():
        return index

    for csv_path in sorted(table_dir.glob("*.csv")):
        parsed = _parse_table_csv_name(plot_type, csv_path)
        if parsed is None:
            continue
        index[parsed] = csv_path
    return index


def _ordered_sections_from_index(index: dict[tuple[str, str, str | None, str, str | None], Path]) -> list[tuple[str, str, str | None]]:
    all_sections = {(collection_type, split_type, partition_type) for collection_type, split_type, partition_type, _, _ in index}
    return sorted(all_sections, key=_section_sort_key)


def _base_model_for_post_training(model_id: str) -> str | None:
    if model_id.startswith("ft_Llama-3.1-8B"):
        return "Llama-3.1-8B-Instruct"
    if model_id.startswith("ft_gpt-4o-mini-2024-07-18"):
        return "gpt-4o-mini-2024-07-18"
    return None


def _post_training_label_prefix(model_id: str) -> str:
    lower = model_id.lower()
    if "dpo" in lower:
        return "dpo"
    if "sft" in lower:
        return "sft"
    return "ft"


def _family_sort_key(model_id: str) -> tuple[int, str]:
    base_model_id = _base_model_for_post_training(model_id) or model_id
    if base_model_id == "Llama-3.1-8B-Instruct":
        return (0, model_id)
    if base_model_id == "gpt-4o-mini-2024-07-18":
        return (1, model_id)
    return (2, model_id)


def _post_training_sort_key(model_id: str) -> tuple[int, int, str]:
    lower = model_id.lower()
    prefix = _post_training_label_prefix(model_id)
    prefix_rank = {"sft": 0, "dpo": 1, "ft": 2}.get(prefix, 3)
    split_rank = 0 if "topic" in lower else 1 if "group" in lower else 2
    return (_family_sort_key(model_id)[0], prefix_rank * 10 + split_rank, model_id)


def _base_model_display_name(model_id: str) -> str:
    mapping = {
        "Mistral-7B-Instruct-v0.3": "Mistral 7B",
        "Llama-3.1-Tulu-3-8B-SFT": "LLaMA Tulu SFT",
        "Llama-3.1-70B-Instruct": "LLaMA 70B",
        "Qwen2.5-32B-Instruct": "Qwen 32B",
        "gpt-4o-mini-2024-07-18": "GPT-4o-mini",
        "gpt-5-mini": "GPT-5-mini",
        "Llama-3.1-8B-Instruct": "LLaMA 8B",
        "gemini-2.5-flash-lite": "Gemini 2.5\nFlash Lite",
    }
    if model_id in mapping:
        return mapping[model_id]
    if model_id.startswith("DeGroot alpha="):
        return f"DeGroot\n{model_id.removeprefix('DeGroot ')}"
    if model_id == "DeGroot fit":
        return "DeGroot\nfit"
    if model_id == "DeGroot fit topic":
        return "DeGroot fit\ntopic"
    if model_id == "DeGroot fit group":
        return "DeGroot fit\ngroup"
    return textwrap.fill(model_id.replace("_", " "), width=16)


def _post_training_display_name(model_id: str, prefix: str) -> str:
    family = _base_model_display_name(_base_model_for_post_training(model_id) or model_id).replace("\n", " ")
    lower = model_id.lower()
    split = "topic" if "topic" in lower else "group" if "group" in lower else None
    method = prefix.upper()
    detail = method if split is None else f"{method} {split}"
    return f"{family}\n{detail}"


def _display_panel_label(label: str) -> str:
    if label == "human":
        return "Human"
    if label.startswith("base: "):
        return f"Base\n{_base_model_display_name(label.removeprefix('base: ').strip())}"
    if label.startswith(("sft: ", "dpo: ", "ft: ")):
        prefix, model_id = label.split(": ", 1)
        return _post_training_display_name(model_id.strip(), prefix)
    return _base_model_display_name(label)


def _panel_title_fontsize(title: str) -> float:
    line_count = title.count("\n") + 1
    if line_count >= 4:
        return 7.5
    if line_count == 3:
        return 8.0
    return 9.0


def _panel_title(label: str, matrix: np.ndarray | None) -> str:
    label = _display_panel_label(label)
    total = _matrix_total(matrix)
    if total is None:
        return label
    return f"{label}\nn={total}"


def _rows_for_base_model_section(section_root: Path, plot_type: str, stype: str | None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_id in _discover_model_ids(section_root):
        if _model_category(model_id) != "base_model":
            continue
        rows.append(
            {
                "label": model_id,
                "plot": section_root / model_id / _plot_filename(plot_type, stype, ".png"),
                "matrix": section_root / model_id / _plot_filename(plot_type, stype, ".csv"),
            }
        )
    return rows


def _rows_for_base_model_section_from_index(
    index: dict[tuple[str, str, str | None, str, str | None], Path],
    collection_type: str,
    split_type: str,
    partition_type: str | None,
    stype: str | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_id in sorted({model_id for ctype, stype_name, partition, model_id, bucket in index if (ctype, stype_name, partition, bucket) == (collection_type, split_type, partition_type, stype)}):
        if _model_category(model_id) != "base_model":
            continue
        rows.append({"label": model_id, "plot": Path(""), "matrix": index[(collection_type, split_type, partition_type, model_id, stype)]})
    return rows


def _rows_for_post_training_section(
    section_root: Path,
    plot_type: str,
    stype: str | None,
    collection_type: str,
    split_type: str,
) -> list[dict[str, object]]:
    post_training_model_ids = []
    for model_id in _discover_model_ids(section_root):
        if _model_category(model_id) != "post_training":
            continue
        lower = model_id.lower()
        if collection_type not in lower:
            continue
        if split_type in {"topic", "group"} and split_type not in lower:
            continue
        post_training_model_ids.append(model_id)

    if not post_training_model_ids:
        return []

    rows: list[dict[str, object]] = []
    grouped_model_ids: dict[str, list[str]] = {}
    for model_id in sorted(post_training_model_ids, key=_post_training_sort_key):
        base_model_id = _base_model_for_post_training(model_id)
        if base_model_id is None:
            continue
        grouped_model_ids.setdefault(base_model_id, []).append(model_id)

    for base_model_id in sorted(grouped_model_ids, key=_family_sort_key):
        base_model_path = section_root / base_model_id
        if base_model_path.exists():
            rows.append(
                {
                    "label": f"base: {base_model_id}",
                    "plot": base_model_path / _plot_filename(plot_type, stype, ".png"),
                    "matrix": base_model_path / _plot_filename(plot_type, stype, ".csv"),
                }
            )
        for model_id in grouped_model_ids[base_model_id]:
            rows.append(
                {
                    "label": f"{_post_training_label_prefix(model_id)}: {model_id}",
                    "plot": section_root / model_id / _plot_filename(plot_type, stype, ".png"),
                    "matrix": section_root / model_id / _plot_filename(plot_type, stype, ".csv"),
                }
            )
    return rows


def _rows_for_post_training_section_from_index(
    index: dict[tuple[str, str, str | None, str, str | None], Path],
    collection_type: str,
    split_type: str,
    partition_type: str | None,
    stype: str | None,
) -> list[dict[str, object]]:
    post_training_model_ids = []
    candidate_model_ids = sorted({model_id for ctype, split_name, partition, model_id, bucket in index if (ctype, split_name, partition, bucket) == (collection_type, split_type, partition_type, stype)})
    for model_id in candidate_model_ids:
        if _model_category(model_id) != "post_training":
            continue
        lower = model_id.lower()
        if collection_type not in lower:
            continue
        if split_type in {"topic", "group"} and split_type not in lower:
            continue
        post_training_model_ids.append(model_id)

    if not post_training_model_ids:
        return []

    rows: list[dict[str, object]] = []
    grouped_model_ids: dict[str, list[str]] = {}
    for model_id in sorted(post_training_model_ids, key=_post_training_sort_key):
        base_model_id = _base_model_for_post_training(model_id)
        if base_model_id is None:
            continue
        grouped_model_ids.setdefault(base_model_id, []).append(model_id)

    for base_model_id in sorted(grouped_model_ids, key=_family_sort_key):
        base_key = (collection_type, split_type, partition_type, base_model_id, stype)
        if base_key in index:
            rows.append({"label": f"base: {base_model_id}", "plot": Path(""), "matrix": index[base_key]})
        for model_id in grouped_model_ids[base_model_id]:
            rows.append(
                {
                    "label": f"{_post_training_label_prefix(model_id)}: {model_id}",
                    "plot": Path(""),
                    "matrix": index[(collection_type, split_type, partition_type, model_id, stype)],
                }
            )
    return rows


def _rows_for_combined_section_from_index(
    index: dict[tuple[str, str, str | None, str, str | None], Path],
    collection_type: str,
    split_type: str,
    partition_type: str | None,
    stype: str | None,
) -> list[dict[str, object]]:
    candidate_model_ids = sorted(
        {
            model_id
            for ctype, split_name, partition, model_id, bucket in index
            if (ctype, split_name, partition, bucket) == (collection_type, split_type, partition_type, stype)
            and model_id != "human"
        }
    )
    present_base_models = {model_id for model_id in candidate_model_ids if _model_category(model_id) == "base_model"}
    degroot_model_ids = [model_id for model_id in candidate_model_ids if _model_category(model_id) == "degroot"]
    family_map: dict[str, list[str]] = {model_id: [] for model_id in present_base_models}
    orphan_post_training: list[str] = []

    for model_id in sorted(candidate_model_ids, key=_post_training_sort_key):
        if _model_category(model_id) != "post_training":
            continue
        lower = model_id.lower()
        if collection_type not in lower:
            continue
        if split_type in {"topic", "group"} and split_type not in lower:
            continue
        base_model_id = _base_model_for_post_training(model_id)
        if base_model_id is None:
            orphan_post_training.append(model_id)
            continue
        family_map.setdefault(base_model_id, []).append(model_id)

    rows: list[dict[str, object]] = []
    for family_id in sorted(family_map, key=_family_sort_key):
        base_key = (collection_type, split_type, partition_type, family_id, stype)
        family_members = family_map[family_id]
        if base_key in index:
            base_label = f"base: {family_id}" if family_members else family_id
            rows.append({"label": base_label, "plot": Path(""), "matrix": index[base_key]})
        for model_id in family_members:
            rows.append(
                {
                    "label": f"{_post_training_label_prefix(model_id)}: {model_id}",
                    "plot": Path(""),
                    "matrix": index[(collection_type, split_type, partition_type, model_id, stype)],
                }
            )

    for model_id in sorted(degroot_model_ids):
        rows.append(
            {
                "label": model_id,
                "plot": Path(""),
                "matrix": index[(collection_type, split_type, partition_type, model_id, stype)],
            }
        )

    for model_id in orphan_post_training:
        rows.append(
            {
                "label": f"{_post_training_label_prefix(model_id)}: {model_id}",
                "plot": Path(""),
                "matrix": index[(collection_type, split_type, partition_type, model_id, stype)],
            }
        )
    return rows


def _copy_csvs(plot_type: str, work_root: Path, results_root: Path) -> None:
    config = PLOT_CONFIG[plot_type]
    raw_root = work_root / config["raw_subdir"]
    target_csv_dir = _results_matrix_dir(results_root, plot_type) / "tables"
    target_csv_dir.mkdir(parents=True, exist_ok=True)
    if not raw_root.exists():
        return
    for csv_path in sorted(raw_root.rglob("*.csv")):
        relative = csv_path.relative_to(raw_root)
        filename = "__".join(relative.parts)
        shutil.copy2(csv_path, target_csv_dir / filename)


def _render_category_outputs(
    plot_type: str,
    category: str,
    work_root: Path,
    results_root: Path,
    *,
    value_type: str,
    collection_filter: str | None = None,
    overall_only: bool = False,
    output_suffix: str = "",
    use_split_matrices: bool = True,
    split_filter: str | None = None,
    partition_filter: str | None = None,
    page_filter: int | None = None,
    color_theme: str | None = None,
) -> None:
    config = PLOT_CONFIG[plot_type]
    del work_root
    index = _table_index(plot_type, results_root)
    out_dir = _results_matrix_dir(results_root, plot_type) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if collection_filter is None else f"_{collection_filter}"
    suffix = f"{suffix}{output_suffix}"
    pdf_path = out_dir / f"{config['artifact_prefix']}_{category}{suffix}.pdf"
    png_path = out_dir / f"{config['artifact_prefix']}_{category}{suffix}.png"

    first_page_saved = False
    page_count = 0

    def save_page(fig: plt.Figure, pdf: PdfPages) -> None:
        nonlocal first_page_saved, page_count
        pdf.savefig(fig)
        if not first_page_saved:
            fig.savefig(png_path, dpi=300)
            first_page_saved = True
        plt.close(fig)
        page_count += 1

    with PdfPages(pdf_path) as pdf:
        for collection_type, split_type, partition_type in _ordered_sections_from_index(index):
            if collection_filter is not None and collection_type != collection_filter:
                continue
            if overall_only and split_type != "overall":
                continue
            if split_filter is not None and split_type != split_filter:
                continue
            if partition_filter is not None and partition_type != partition_filter:
                continue
            source_split_type = split_type if use_split_matrices else "overall"
            source_partition_type = partition_type if use_split_matrices else None
            for stype in config["stypes"]:
                if category == "base_model":
                    rows = _rows_for_base_model_section_from_index(
                        index,
                        collection_type,
                        source_split_type,
                        source_partition_type,
                        stype,
                    )
                elif category == "post_training":
                    rows = _rows_for_post_training_section_from_index(
                        index,
                        collection_type,
                        source_split_type,
                        source_partition_type,
                        stype,
                    )
                else:
                    rows = _rows_for_combined_section_from_index(
                        index,
                        collection_type,
                        source_split_type,
                        source_partition_type,
                        stype,
                    )
                if not rows:
                    continue

                print(
                    "[layout]",
                    f"plot_type={plot_type}",
                    f"category={category}",
                    f"collection={collection_type}",
                    f"split={split_type}",
                    f"partition={partition_type or 'overall'}",
                    f"stype={stype or 'overall'}",
                    f"rows={len(rows)}",
                )

                human_matrix = index.get((collection_type, source_split_type, source_partition_type, "human", stype))
                human_plot = Path("")

                subtitle = f"plot_type={plot_type}, category={category}, collection={collection_type}, split={split_type}"
                if partition_type is not None:
                    subtitle += f", partition={partition_type}"
                if stype is not None:
                    subtitle += f", stype={stype}"

                display_rows = list(rows)
                if human_matrix is not None:
                    display_rows = [
                        {"label": "human", "plot": human_plot, "matrix": human_matrix, "is_summary": True},
                        *display_rows,
                    ]
                for page_start in range(0, len(display_rows), PANELS_PER_PAGE):
                    page_rows = display_rows[page_start : page_start + PANELS_PER_PAGE]
                    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
                    ax_list = list(axes.flatten())
                    page_number = (page_start // PANELS_PER_PAGE) + 1
                    total_pages = int(np.ceil(len(display_rows) / PANELS_PER_PAGE))
                    if page_filter is not None and page_number != page_filter:
                        plt.close(fig)
                        continue
                    title = subtitle if total_pages == 1 else f"{subtitle}, page={page_number}/{total_pages}"
                    fig.suptitle(title, fontsize=14)

                    for idx, row in enumerate(page_rows):
                        ax = ax_list[idx]
                        matrix = _load_matrix(row["matrix"])
                        _draw_image(
                            ax,
                            row["plot"],
                            _panel_title(str(row["label"]), matrix),
                            str(row["label"]),
                            plot_type,
                            value_type,
                            matrix,
                            color_theme=color_theme,
                        )
                        if not row.get("is_summary"):
                            if human_matrix is not None:
                                diff = _normalized_frobenius_diff(matrix, _load_matrix(human_matrix))
                                diff_text = "N/A" if diff is None else f"{diff:.4f}"
                            else:
                                diff_text = "N/A (no human matrix)"
                            ax.text(
                                0.5,
                                -0.23,
                                f"frobenius norm\ndiff with human\n= {diff_text}",
                                transform=ax.transAxes,
                                va="top",
                                ha="center",
                                fontsize=8,
                                clip_on=False,
                                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
                            )
                    for idx in range(len(page_rows), len(ax_list)):
                        ax_list[idx].axis("off")
                    plt.tight_layout(rect=[0, 0, 1, 0.96])
                    save_page(fig, pdf)

        if page_count == 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.axis("off")
            ax.text(0.5, 0.5, f"No {category} outputs available for {plot_type}", ha="center", va="center", fontsize=14)
            save_page(fig, pdf)


def _render_report(
    plot_type: str,
    results_root: Path,
    *,
    metric_version: str,
    value_type: str,
    group_validity_threshold: float | None,
    collection_filter: str,
    overall_only: bool = False,
    output_suffix: str = "",
    use_split_masks: bool = True,
) -> None:
    config = PLOT_CONFIG[plot_type]
    report_path = results_root / _collection_report_filename(
        plot_type,
        collection_filter,
        value_type,
        output_suffix,
    )
    img_dir = _results_matrix_dir(results_root, plot_type) / "figures"
    img_dir.mkdir(parents=True, exist_ok=True)
    category_pdfs = [
        img_dir / f"{config['artifact_prefix']}_{category}_{collection_filter}{output_suffix}.pdf"
        for category in ("combined",)
    ]
    existing_category_pdfs = [path for path in category_pdfs if path.exists()]

    print("[report]", f"plot_type={plot_type}", f"collection={collection_filter}", f"path={report_path}")

    if existing_category_pdfs:
        writer = PdfWriter()
        intro_reader = _intro_pdf_reader(
            plot_type,
            results_root,
            metric_version=metric_version,
            value_type=value_type,
            group_validity_threshold=group_validity_threshold,
            use_split_masks=use_split_masks,
        )
        for page in intro_reader.pages:
            writer.add_page(page)
        summary_reader = _summary_pdf_reader(
            plot_type,
            results_root,
            metric_version=metric_version,
            value_type=value_type,
            group_validity_threshold=group_validity_threshold,
            collection_filter=collection_filter,
            overall_only=overall_only,
            use_split_masks=use_split_masks,
        )
        for page in summary_reader.pages:
            writer.add_page(page)
        for path in existing_category_pdfs:
            reader = PdfReader(str(path))
            for page in reader.pages:
                writer.add_page(page)
        with report_path.open("wb") as handle:
            writer.write(handle)
        return

    with PdfPages(report_path) as pdf:
        fig = _render_intro_page(
            plot_type,
            results_root,
            metric_version=metric_version,
            value_type=value_type,
            group_validity_threshold=group_validity_threshold,
            use_split_masks=use_split_masks,
        )
        pdf.savefig(fig)
        plt.close(fig)
        fig = _render_summary_page(
            plot_type,
            results_root,
            metric_version=metric_version,
            value_type=value_type,
            group_validity_threshold=group_validity_threshold,
            collection_filter=collection_filter,
            overall_only=overall_only,
            use_split_masks=use_split_masks,
        )
        pdf.savefig(fig)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.axis("off")
        ax.text(0.5, 0.5, f"Missing {config['artifact_prefix']}_combined.pdf", ha="center", va="center", fontsize=14)
        pdf.savefig(fig)
        plt.close(fig)


def build_layout(
    work_root: Path,
    results_root: Path,
    *,
    metric_version: str,
    value_type: str,
    group_validity_threshold: float | None,
) -> None:
    global ACTIVE_POST_TRAINING_MODEL_IDS
    ACTIVE_POST_TRAINING_MODEL_IDS = post_training_model_ids_for_metric_version(metric_version)
    color_theme = "v4" if metric_version in {"v4", "v4_b"} else None
    plot_types = ("cf", "pn", "ind") if include_close_far_transitions(metric_version) else ("pn", "ind")
    for plot_type in plot_types:
        print("[build_layout]", f"plot_type={plot_type}", f"results_root={results_root}")
        _copy_csvs(plot_type, work_root, results_root)
        for collection_type in COLLECTION_TYPES:
            print("[build_layout]", f"plot_type={plot_type}", f"collection={collection_type}", "render_category_outputs")
            _render_category_outputs(
                plot_type,
                "combined",
                work_root,
                results_root,
                value_type=value_type,
                collection_filter=collection_type,
                color_theme=color_theme,
            )
            print("[build_layout]", f"plot_type={plot_type}", f"collection={collection_type}", "render_report")
            _render_report(
                plot_type,
                results_root,
                metric_version=metric_version,
                value_type=value_type,
                group_validity_threshold=group_validity_threshold,
                collection_filter=collection_type,
            )
            if plot_type == "ind":
                print(
                    "[build_layout]",
                    f"plot_type={plot_type}",
                    f"collection={collection_type}",
                    "render_category_outputs_unfiltered",
                )
                _render_category_outputs(
                    plot_type,
                    "combined",
                    work_root,
                    results_root,
                    value_type=value_type,
                    collection_filter=collection_type,
                    output_suffix="_unfiltered",
                    use_split_matrices=False,
                    color_theme=color_theme,
                )
                print(
                    "[build_layout]",
                    f"plot_type={plot_type}",
                    f"collection={collection_type}",
                    "render_report_unfiltered",
                )
                _render_report(
                    plot_type,
                    results_root,
                    metric_version=metric_version,
                    value_type=value_type,
                    group_validity_threshold=group_validity_threshold,
                    collection_filter=collection_type,
                    output_suffix="_unfiltered",
                    use_split_masks=False,
                )


def clean_work_outputs(work_root: Path) -> None:
    for dirname in ("close_far", "polarized_neutral", "individual_likert"):
        path = work_root / dirname
        if path.exists():
            shutil.rmtree(path)


def clean_outputs(work_root: Path, results_root: Path) -> None:
    del work_root
    for dirname in ("matrices", "reports"):
        path = results_root / dirname
        if path.exists():
            shutil.rmtree(path)
    for dirname in ("cf_plots", "pn_plots", "indiv_plots", "cf_matrix", "pn_matrix", "ind_matrix"):
        path = results_root / dirname
        if path.exists():
            shutil.rmtree(path)
    report_filenames = ["transition_summary.pdf", *FINAL_REPORT_FILENAMES.values()]
    for plot_type in FINAL_REPORT_FILENAMES:
        for collection_type in COLLECTION_TYPES:
            for value_type in ("likert_pred", "slider_value"):
                report_filenames.append(_collection_report_filename(plot_type, collection_type, value_type))
                if plot_type == "ind":
                    report_filenames.append(
                        _collection_report_filename(
                            plot_type,
                            collection_type,
                            value_type,
                            "_unfiltered",
                        )
                    )
    for filename in report_filenames:
        path = results_root / filename
        if path.exists():
            path.unlink()
    for filename in (
        "build_transition_layout.py",
        "gen_doc_base.py",
        "plot.py",
        "plot_bias.py",
        "plot_indiv.py",
        "utils.py",
    ):
        path = results_root / filename
        if path.exists():
            path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build presentation-oriented transition matrix results")
    parser.add_argument("--work-root", default=str(DEFAULT_WORK_ROOT))
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--clean", action="store_true", help="Remove old transition output directories before reconstruction")
    parser.add_argument("--keep-work", action="store_true", help="Keep intermediate work files after a successful build")
    parser.add_argument("--metric-version", choices=["v0", "v0_fixed", "v0_fix", "v3", "v4", "v4_b"], default="v0")
    parser.add_argument("--value-type", choices=["likert_pred", "slider_value"], default="likert_pred")
    parser.add_argument("--group-validity-threshold", type=float, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    work_root = Path(arguments.work_root).resolve()
    results_root = Path(arguments.results_root).resolve()
    if arguments.clean:
        clean_outputs(work_root, results_root)
    build_layout(
        work_root,
        results_root,
        metric_version=arguments.metric_version,
        value_type=arguments.value_type,
        group_validity_threshold=arguments.group_validity_threshold,
    )
    if not arguments.keep_work:
        clean_work_outputs(work_root)