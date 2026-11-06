#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
RADAR_DIR = REPO_ROOT / "neurips_2026" / "chengtao" / "src" / "radar"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(RADAR_DIR) not in sys.path:
    sys.path.insert(0, str(RADAR_DIR))

from generate_plots import MODEL_COLOR_MAP, format_model_display_label, get_model_color, get_v2_display_model_order  # noqa: E402


RAW_TABLE_DIR = REPO_ROOT / "neurips_2026" / "sean_reproduce" / "v4" / "raw_result_tables"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "neurips_2026" / "sean_reproduce" / "v4_pub" / "beyond_stance"

MAINTABLE_FILES = {
    "breadth": RAW_TABLE_DIR / "Mini-Twitter_ Reproducing Result Tables - Leo_V4_Maintable_Breadth.csv",
    "depth": RAW_TABLE_DIR / "Mini-Twitter_ Reproducing Result Tables - Leo_V4_Maintable_Depth.csv",
}
LLAMA_SFT_BREADTH_FILE = RAW_TABLE_DIR / "Mini-Twitter_ Reproducing Result Tables - Leo_V4_Llama_sft_Breadth.csv"
LLAMA_SFT_DEPTH_FILE = RAW_TABLE_DIR / "Mini-Twitter_ Reproducing Result Tables - Leo_V4_Llama_sft_Depth.csv"
BREADTH_LLAMA_SFT_PLOT_MODEL = "ft:Llama-3.1-8B-Instruct-SFT-20260501:breadth-avg-for-beyond-stance"
DEPTH_LLAMA_SFT_PLOT_MODEL = "ft:Llama-3.1-8B-Instruct-SFT-20260501:depth-avg-for-beyond-stance"
OUTDATED_OUTPUT_FILENAMES = {
    "llama_sft_depth_pre_post_deltas.pdf",
    "llama_sft_depth_pre_post_deltas.png",
}
LEGACY_UNSUFFIXED_FIGURE_FILENAMES = {
    "breadth_beyond_stance.pdf",
    "breadth_beyond_stance.png",
    "depth_beyond_stance.pdf",
    "depth_beyond_stance.png",
    "llama_sft_depth_group_test_beyond_stance.pdf",
    "llama_sft_depth_group_test_beyond_stance.png",
    "llama_sft_depth_group_train_beyond_stance.pdf",
    "llama_sft_depth_group_train_beyond_stance.png",
    "llama_sft_depth_topic_test_beyond_stance.pdf",
    "llama_sft_depth_topic_test_beyond_stance.png",
    "llama_sft_depth_topic_train_beyond_stance.pdf",
    "llama_sft_depth_topic_train_beyond_stance.png",
    "llama_sft_breadth_group_test_beyond_stance.pdf",
    "llama_sft_breadth_group_test_beyond_stance.png",
    "llama_sft_breadth_group_train_beyond_stance.pdf",
    "llama_sft_breadth_group_train_beyond_stance.png",
    "llama_sft_breadth_topic_test_beyond_stance.pdf",
    "llama_sft_breadth_topic_test_beyond_stance.png",
    "llama_sft_breadth_topic_train_beyond_stance.pdf",
    "llama_sft_breadth_topic_train_beyond_stance.png",
}

SIMULATION_MODE_SHORT = {
    "Simulation Mode 2: Tweet-guided Conversation Simulation (v1b)": "tweet_guided",
    "Simulation Mode 3: Full Conversation Simulation (v0)": "full_conversation",
}
SIMULATION_MODE_DISPLAY = {
    "Simulation Mode 2: Tweet-guided Conversation Simulation (v1b)": "Tweet-guided",
    "Simulation Mode 3: Full Conversation Simulation (v0)": "Full conversation",
}
GENERALIZATION_DISPLAY = {
    "Group\nGeneralization": "Group Generalization",
    "Topic\nGeneralization": "Topic Generalization",
}
MODEL_SHORT_LABELS = {
    "gpt-4o-mini-2024-07-18": "GPT-4o mini",
    "gpt-5-mini": "GPT-5 mini",
    "gemini-2.5-flash-lite": "Gemini 2.5 flash-lite",
    "Llama-3.1-Tulu-3-8B-SFT": "Tulu 3 8B SFT",
    "Llama-3.1-8B-Instruct": "Llama 3.1 8B",
    "Llama-3.1-70B-Instruct": "Llama 3.1 70B",
    "Mistral-7B-Instruct-v0.3": "Mistral 7B",
    "Qwen2.5-32B-Instruct": "Qwen2.5 32B",
    BREADTH_LLAMA_SFT_PLOT_MODEL: "Llama SFT",
    DEPTH_LLAMA_SFT_PLOT_MODEL: "Llama SFT",
    "pre-SFT": "Pre-SFT",
    "post-SFT": "Post-SFT",
}
MODEL_PUBLIC_TICK_LABELS = {
    "gpt-4o-mini-2024-07-18": "GPT-4o mini",
    "gpt-5-mini": "GPT-5 mini",
    "gemini-2.5-flash-lite": "Gemini flash-lite",
    "Llama-3.1-Tulu-3-8B-SFT": "Tulu 3 8B",
    "Llama-3.1-8B-Instruct": "Llama 8B",
    "Llama-3.1-70B-Instruct": "Llama 70B",
    "Mistral-7B-Instruct-v0.3": "Mistral 7B",
    "Qwen2.5-32B-Instruct": "Qwen 32B",
    BREADTH_LLAMA_SFT_PLOT_MODEL: "Llama SFT",
    DEPTH_LLAMA_SFT_PLOT_MODEL: "Llama SFT",
    "pre-SFT": "Llama 8B",
    "post-SFT": "Llama SFT",
}
PANEL_SECTION_GAP = 2.0
CSV_SUBDIR = "csv"
MAINTABLE_SUBDIR = "maintable"
SFT_BREADTH_SUBDIR = "sft_breadth"
SFT_DEPTH_SUBDIR = "sft_depth"
METRIC_COLUMNS = {
    "semantic_similarity_tweet": "semantic similarity tweet",
    "semantic_similarity_post_opinion": "semantic similarity post opinion",
    "n_word_diff_tweet": "n word diff tweet",
    "n_word_diff_post_opinion": "n word diff post opinion",
    "rouge_l_tweet": "rouge L tweet",
    "rouge_l_post_opinion": "rouge L post opinion",
    "valid_response_rate": "valid response rate",
}
MAINTABLE_ROW_SPECS = [
    ("Semantic Similarity", "semantic_similarity_tweet", "semantic_similarity_post_opinion", (0.0, 1.05)),
    ("ROUGE-L", "rouge_l_tweet", "rouge_l_post_opinion", (0.0, 0.60)),
    ("Message Length Difference", "n_word_diff_tweet", "n_word_diff_post_opinion", None),
    ("On-Topic Rate", "valid_response_rate", "valid_response_rate", (0.0, 1.05)),
]
FIGURE_ROW_GROUP_SPECS = [
    ("", MAINTABLE_ROW_SPECS[:3]),
    ("_on_topic_rate", MAINTABLE_ROW_SPECS[3:]),
]
PLOT_EXCLUDED_MODELS = {"Llama-3.1-Tulu-3-8B-SFT"}
SIMULATION_MODE_PLOT_SPECS = {
    "Simulation Mode 3: Full Conversation Simulation (v0)": {
        "label": "Full Dynamics Simulation",
        "offset": -0.18,
        "hatch": None,
        "edgecolor": "#1F1F1F",
        "alpha": 0.92,
    },
    "Simulation Mode 2: Tweet-guided Conversation Simulation (v1b)": {
        "label": "Next Message Prediction",
        "offset": 0.18,
        "hatch": "///",
        "edgecolor": "#1F1F1F",
        "alpha": 0.75,
    },
}
PLOT_VARIANTS = {
    "raw": {
        "figsize_maintable": (17.0, 11.5),
        "figsize_llama": (11.5, 11.5),
        "annotation_fontsize": 7.5,
        "row_title_fontsize": 15,
        "row_title_pad": 4,
        "panel_label_fontsize": 10,
        "show_background_spans": True,
        "show_grid": True,
        "show_suptitle": True,
        "show_subtitle": True,
        "show_inline_panel_labels": True,
        "show_figure_panel_labels": False,
        "legend_loc": "upper center",
        "legend_bbox": (0.5, 0.955),
        "legend_fontsize": 10,
        "tick_rotation": 40,
        "llama_tick_rotation": 35,
        "tick_fontsize": 12,
        "y_tick_fontsize": 11,
        "top": 0.90,
        "bottom": 0.20,
        "left": 0.07,
        "right": 0.995,
        "hspace": 0.30,
        "panel_label_y": 0.06,
    },
    "pub": {
        "figsize_maintable": (17.5, 12.2),
        "figsize_llama": (17.5, 12.2),
        "annotation_fontsize": 9.5,
        "row_title_fontsize": 20,
        "row_title_pad": 12,
        "panel_label_fontsize": 24,
        "show_background_spans": False,
        "show_grid": False,
        "show_suptitle": False,
        "show_subtitle": False,
        "show_inline_panel_labels": False,
        "show_figure_panel_labels": True,
        "legend_loc": "lower center",
        "legend_bbox": (0.5, 0.015),
        "legend_fontsize": 16,
        "tick_rotation": 90,
        "llama_tick_rotation": 90,
        "tick_fontsize": 16,
        "y_tick_fontsize": 14,
        "top": 0.90,
        "bottom": 0.28,
        "left": 0.085,
        "right": 0.995,
        "hspace": 0.34,
        "panel_label_y": 0.992,
    },
}


def read_csv_rows(csv_path: Path) -> list[list[str]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        return [row for row in csv.reader(handle)]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_float(value: float) -> str:
    return f"{value:.3f}"


def safe_float(value: str) -> float:
    return float(value.strip())


def model_label(model_name: str) -> str:
    return MODEL_SHORT_LABELS.get(model_name, model_name)


def publication_model_label(model_name: str) -> str:
    if model_name in {BREADTH_LLAMA_SFT_PLOT_MODEL, DEPTH_LLAMA_SFT_PLOT_MODEL}:
        return "Llama\nSFT"
    return format_model_display_label(model_name)


def plot_tick_label(model_name: str, *, variant: str) -> str:
    if variant == "pub":
        if "Llama-3.1-8B-Instruct-SFT" in model_name:
            return "Llama SFT"
        return MODEL_PUBLIC_TICK_LABELS.get(model_name, model_label(model_name))
    return publication_model_label(model_name)


def variant_stem(stem: str, *, variant: str) -> str:
    return f"{stem}_{variant}"


def grouped_figure_stem(base_stem: str, group_suffix: str) -> str:
    return f"{base_stem}{group_suffix}"


def figure_size(*, style: dict[str, object], family: str, row_count: int) -> tuple[float, float]:
    base_size = tuple(style["figsize_maintable"] if family == "maintable" else style["figsize_llama"])
    base_width = float(base_size[0])
    base_height = float(base_size[1])
    scaled_height = max(base_height * (row_count / len(MAINTABLE_ROW_SPECS)), 4.2)
    return (base_width, scaled_height)


def output_section_dir(output_dir: Path, section: str) -> Path:
    path = output_dir / section
    path.mkdir(parents=True, exist_ok=True)
    return path


def maintable_stem(split: str, group_suffix: str) -> str:
    return f"{split}_rate" if group_suffix == "_on_topic_rate" else f"{split}_main"


def sft_stem(generalization_slug: str, split_slug: str, group_suffix: str) -> str:
    mode_suffix = "rate" if group_suffix == "_on_topic_rate" else "main"
    return f"{generalization_slug}_{split_slug}_{mode_suffix}"


def parse_maintable_rows(path: Path, split: str) -> list[dict[str, object]]:
    parsed: list[dict[str, object]] = []
    current_mode = ""
    header: list[str] | None = None
    for raw_row in read_csv_rows(path):
        row = [cell.strip() for cell in raw_row]
        if not any(row):
            continue
        first_cell = row[0]
        if first_cell.startswith("Simulation Mode"):
            current_mode = first_cell
            header = None
            continue
        if first_cell == "Model":
            header = row
            continue
        if not header or not current_mode:
            continue
        if first_cell.startswith("Round Level Metrics"):
            continue
        if not row[METRIC_INDEX(header, METRIC_COLUMNS["semantic_similarity_tweet"])]:
            continue
        row_dict = dict(zip(header, row))
        parsed.append(
            {
                "source_table": split,
                "simulation_mode": current_mode,
                "simulation_mode_short": SIMULATION_MODE_SHORT[current_mode],
                "model": row_dict["Model"],
                **{
                    output_key: safe_float(row_dict[input_key])
                    for output_key, input_key in METRIC_COLUMNS.items()
                },
            }
        )
    return parsed


def METRIC_INDEX(header: list[str], metric_name: str) -> int:
    return header.index(metric_name)


def parse_llama_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    current_context: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            row = {key: (value.strip() if isinstance(value, str) else value) for key, value in raw_row.items()}
            if not any(row.values()):
                continue
            for context_key in ("Generalization Type", "Simulation Mode", "Train/Test"):
                if row[context_key]:
                    current_context[context_key] = row[context_key]
                else:
                    row[context_key] = current_context[context_key]
            rows.append(
                {
                    "generalization_type": row["Generalization Type"],
                    "simulation_mode": row["Simulation Mode"],
                    "simulation_mode_short": SIMULATION_MODE_SHORT[row["Simulation Mode"]],
                    "train_test": row["Train/Test"],
                    "model_state": row["Model"],
                    "real_model": row["Real Model"],
                    **{
                        output_key: safe_float(row[input_key])
                        for output_key, input_key in METRIC_COLUMNS.items()
                    },
                }
            )
    return rows


def build_maintable_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["source_table"]), str(row["simulation_mode"]))
        grouped.setdefault(key, []).append(row)
    for (split, simulation_mode), subset in grouped.items():
        avg_semantic_tweet = mean(float(item["semantic_similarity_tweet"]) for item in subset)
        avg_semantic_post = mean(float(item["semantic_similarity_post_opinion"]) for item in subset)
        avg_word_tweet = mean(float(item["n_word_diff_tweet"]) for item in subset)
        avg_word_post = mean(float(item["n_word_diff_post_opinion"]) for item in subset)
        avg_rouge_tweet = mean(float(item["rouge_l_tweet"]) for item in subset)
        avg_rouge_post = mean(float(item["rouge_l_post_opinion"]) for item in subset)
        avg_valid = mean(float(item["valid_response_rate"]) for item in subset)
        best_tweet = max(subset, key=lambda item: float(item["semantic_similarity_tweet"]))
        best_post = max(subset, key=lambda item: float(item["semantic_similarity_post_opinion"]))
        best_valid = max(subset, key=lambda item: float(item["valid_response_rate"]))
        summaries.append(
            {
                "source_table": split,
                "simulation_mode": simulation_mode,
                "simulation_mode_short": SIMULATION_MODE_SHORT[simulation_mode],
                "avg_semantic_similarity_tweet": format_float(avg_semantic_tweet),
                "avg_semantic_similarity_post_opinion": format_float(avg_semantic_post),
                "avg_n_word_diff_tweet": format_float(avg_word_tweet),
                "avg_n_word_diff_post_opinion": format_float(avg_word_post),
                "avg_rouge_l_tweet": format_float(avg_rouge_tweet),
                "avg_rouge_l_post_opinion": format_float(avg_rouge_post),
                "avg_valid_response_rate": format_float(avg_valid),
                "best_tweet_semantic_model": str(best_tweet["model"]),
                "best_tweet_semantic_value": format_float(float(best_tweet["semantic_similarity_tweet"])),
                "best_post_semantic_model": str(best_post["model"]),
                "best_post_semantic_value": format_float(float(best_post["semantic_similarity_post_opinion"])),
                "best_valid_model": str(best_valid["model"]),
                "best_valid_value": format_float(float(best_valid["valid_response_rate"])),
            }
        )
    summaries.sort(key=lambda row: (row["source_table"], row["simulation_mode_short"]))
    return summaries


def build_llama_comparison(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], dict[str, dict[str, object]]] = {}
    for row in rows:
        key = (str(row["generalization_type"]), str(row["simulation_mode"]), str(row["train_test"]))
        grouped.setdefault(key, {})[str(row["model_state"])] = row
    comparisons: list[dict[str, object]] = []
    for (generalization_type, simulation_mode, train_test), states in grouped.items():
        if "pre-SFT" not in states or "post-SFT" not in states:
            continue
        pre = states["pre-SFT"]
        post = states["post-SFT"]
        comparisons.append(
            {
                "generalization_type": generalization_type,
                "simulation_mode": simulation_mode,
                "simulation_mode_short": SIMULATION_MODE_SHORT[simulation_mode],
                "train_test": train_test,
                "pre_semantic_similarity_tweet": format_float(float(pre["semantic_similarity_tweet"])),
                "post_semantic_similarity_tweet": format_float(float(post["semantic_similarity_tweet"])),
                "delta_semantic_similarity_tweet": format_float(float(post["semantic_similarity_tweet"]) - float(pre["semantic_similarity_tweet"])),
                "pre_semantic_similarity_post_opinion": format_float(float(pre["semantic_similarity_post_opinion"])),
                "post_semantic_similarity_post_opinion": format_float(float(post["semantic_similarity_post_opinion"])),
                "delta_semantic_similarity_post_opinion": format_float(float(post["semantic_similarity_post_opinion"]) - float(pre["semantic_similarity_post_opinion"])),
                "pre_n_word_diff_tweet": format_float(float(pre["n_word_diff_tweet"])),
                "post_n_word_diff_tweet": format_float(float(post["n_word_diff_tweet"])),
                "delta_n_word_diff_tweet": format_float(float(post["n_word_diff_tweet"]) - float(pre["n_word_diff_tweet"])),
                "pre_n_word_diff_post_opinion": format_float(float(pre["n_word_diff_post_opinion"])),
                "post_n_word_diff_post_opinion": format_float(float(post["n_word_diff_post_opinion"])),
                "delta_n_word_diff_post_opinion": format_float(float(post["n_word_diff_post_opinion"]) - float(pre["n_word_diff_post_opinion"])),
                "pre_rouge_l_tweet": format_float(float(pre["rouge_l_tweet"])),
                "post_rouge_l_tweet": format_float(float(post["rouge_l_tweet"])),
                "delta_rouge_l_tweet": format_float(float(post["rouge_l_tweet"]) - float(pre["rouge_l_tweet"])),
                "pre_rouge_l_post_opinion": format_float(float(pre["rouge_l_post_opinion"])),
                "post_rouge_l_post_opinion": format_float(float(post["rouge_l_post_opinion"])),
                "delta_rouge_l_post_opinion": format_float(float(post["rouge_l_post_opinion"]) - float(pre["rouge_l_post_opinion"])),
                "pre_valid_response_rate": format_float(float(pre["valid_response_rate"])),
                "post_valid_response_rate": format_float(float(post["valid_response_rate"])),
                "delta_valid_response_rate": format_float(float(post["valid_response_rate"]) - float(pre["valid_response_rate"])),
            }
        )
    comparisons.sort(key=lambda row: (row["generalization_type"], row["simulation_mode_short"], row["train_test"]))
    return comparisons


def build_depth_llama_sft_plot_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        if str(row["model_state"]) != "post-SFT":
            continue
        grouped.setdefault(str(row["simulation_mode"]), []).append(row)

    synthetic_rows: list[dict[str, object]] = []
    for simulation_mode, subset in grouped.items():
        if not subset:
            continue
        synthetic_rows.append(
            {
                "source_table": "depth",
                "simulation_mode": simulation_mode,
                "simulation_mode_short": SIMULATION_MODE_SHORT[simulation_mode],
                "model": DEPTH_LLAMA_SFT_PLOT_MODEL,
                **{
                    metric_key: mean(float(item[metric_key]) for item in subset)
                    for metric_key in METRIC_COLUMNS
                },
            }
        )
    return synthetic_rows


def build_breadth_llama_sft_plot_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        if str(row["model_state"]) != "post-SFT":
            continue
        grouped.setdefault(str(row["simulation_mode"]), []).append(row)

    synthetic_rows: list[dict[str, object]] = []
    for simulation_mode, subset in grouped.items():
        if not subset:
            continue
        synthetic_rows.append(
            {
                "source_table": "breadth",
                "simulation_mode": simulation_mode,
                "simulation_mode_short": SIMULATION_MODE_SHORT[simulation_mode],
                "model": BREADTH_LLAMA_SFT_PLOT_MODEL,
                **{
                    metric_key: mean(float(item[metric_key]) for item in subset)
                    for metric_key in METRIC_COLUMNS
                },
            }
        )
    return synthetic_rows


def _base_depth_plot_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if str(row["source_table"]) == "depth"
        and str(row["model"]) not in PLOT_EXCLUDED_MODELS
        and str(row["model"]) != DEPTH_LLAMA_SFT_PLOT_MODEL
    ]


def _base_breadth_plot_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if str(row["source_table"]) == "breadth"
        and str(row["model"]) not in PLOT_EXCLUDED_MODELS
        and str(row["model"]) != BREADTH_LLAMA_SFT_PLOT_MODEL
    ]


def _insert_post_sft_model(base_model_order: list[str], post_sft_model: str) -> list[str]:
    ordered = [*base_model_order]
    if post_sft_model in ordered:
        return ordered
    if "Llama-3.1-8B-Instruct" in ordered:
        insert_at = ordered.index("Llama-3.1-8B-Instruct") + 1
        ordered.insert(insert_at, post_sft_model)
        return ordered
    ordered.append(post_sft_model)
    return ordered


def mean(values: list[float] | tuple[float, ...] | object) -> float:
    seq = list(values)
    return sum(seq) / len(seq)


def build_markdown_summary(
    *,
    maintable_summary: list[dict[str, object]],
    llama_comparison: list[dict[str, object]],
    output_dir: Path,
) -> Path:
    lines = [
        "# Beyond Stance Exports",
        "",
        "Focused v4 exports for the beyond-stance metrics derived from the three raw result tables.",
        "",
        "Included metrics: semantic similarity, Message Length Difference, ROUGE-L, and On-Topic Rate for tweets and post opinions.",
        "",
        "## Layout",
        "",
        "- `csv/maintable_rows.csv`: row-level breadth/depth model summaries for the selected metrics.",
        "- `csv/maintable_summary.csv`: average-by-mode rollup plus best model per public/private semantic similarity and validity.",
        "- `csv/sft_depth_rows.csv` and `csv/sft_depth_delta.csv`: depth pre/post-SFT rows and deltas.",
        "- `csv/sft_breadth_rows.csv` and `csv/sft_breadth_delta.csv`: breadth pre/post-SFT rows and deltas.",
        "- `maintable/{breadth,depth}_{main,rate}_{raw,pub}.*`: maintable figures, where `main` is the 3-row plot and `rate` is the standalone On-Topic Rate plot.",
        "- `sft_depth/{group,topic}_{train,test}_{main,rate}_{raw,pub}.*`: split-specific depth figures with paired plot-data CSVs.",
        "- `sft_breadth/{group,topic}_{train,test}_{main,rate}_{raw,pub}.*`: split-specific breadth figures with paired plot-data CSVs.",
        "",
        "## Maintable Averages",
        "",
        "| Split | Simulation Mode | Avg Tweet Semantic | Avg Post Semantic | Avg Tweet ROUGE-L | Avg Post ROUGE-L | Avg Tweet Message Length Difference | Avg Post Message Length Difference | Avg On-Topic Rate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in maintable_summary:
        lines.append(
            "| {split} | {mode} | {tweet_sem} | {post_sem} | {tweet_rouge} | {post_rouge} | {tweet_word} | {post_word} | {valid} |".format(
                split=row["source_table"],
                mode=SIMULATION_MODE_DISPLAY[str(row["simulation_mode"])],
                tweet_sem=row["avg_semantic_similarity_tweet"],
                post_sem=row["avg_semantic_similarity_post_opinion"],
                tweet_rouge=row["avg_rouge_l_tweet"],
                post_rouge=row["avg_rouge_l_post_opinion"],
                tweet_word=row["avg_n_word_diff_tweet"],
                post_word=row["avg_n_word_diff_post_opinion"],
                valid=row["avg_valid_response_rate"],
            )
        )
    lines.extend([
        "",
        "## Llama SFT Depth Deltas",
        "",
        "| Generalization | Simulation Mode | Split | Tweet Semantic Δ | Post Semantic Δ | Tweet ROUGE-L Δ | Post ROUGE-L Δ | Tweet Message Length Difference Δ | Post Message Length Difference Δ | On-Topic Rate Δ |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in llama_comparison:
        lines.append(
            "| {generalization} | {mode} | {split} | {tweet_sem} | {post_sem} | {tweet_rouge} | {post_rouge} | {tweet_word} | {post_word} | {valid} |".format(
                generalization=GENERALIZATION_DISPLAY[str(row["generalization_type"])],
                mode=SIMULATION_MODE_DISPLAY[str(row["simulation_mode"])],
                split=row["train_test"],
                tweet_sem=row["delta_semantic_similarity_tweet"],
                post_sem=row["delta_semantic_similarity_post_opinion"],
                tweet_rouge=row["delta_rouge_l_tweet"],
                post_rouge=row["delta_rouge_l_post_opinion"],
                tweet_word=row["delta_n_word_diff_tweet"],
                post_word=row["delta_n_word_diff_post_opinion"],
                valid=row["delta_valid_response_rate"],
            )
        )
    lines.extend([
        "",
        "Breadth SFT outputs now follow the same naming/layout pattern as depth via `llama_sft_breadth_*` CSVs and figures.",
    ])
    readme_path = output_dir / "README.md"
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return readme_path


def _cleanup_outdated_outputs(output_dir: Path) -> None:
    for filename in OUTDATED_OUTPUT_FILENAMES | LEGACY_UNSUFFIXED_FIGURE_FILENAMES:
        path = output_dir / filename
        if path.exists():
            path.unlink()
    for child in output_dir.iterdir():
        if child.is_file() and child.name != "README.md":
            child.unlink()


def _ordered_model_names(rows: list[dict[str, object]]) -> list[str]:
    model_rows = [{"model": str(row["model"])} for row in rows if str(row["model"]) not in PLOT_EXCLUDED_MODELS]
    ordered = [str(model_name) for model_name in get_v2_display_model_order(model_rows, include_human=False)]
    present_models = {str(row["model"]) for row in rows}
    for plot_model in (BREADTH_LLAMA_SFT_PLOT_MODEL, DEPTH_LLAMA_SFT_PLOT_MODEL):
        if plot_model in present_models and plot_model not in ordered:
            if "Llama-3.1-8B-Instruct" in ordered:
                insert_at = ordered.index("Llama-3.1-8B-Instruct") + 1
                ordered.insert(insert_at, plot_model)
            else:
                ordered.append(plot_model)
    return ordered


def _annotate_bar_value(
    ax: object,
    *,
    center_x: float,
    value: float,
    row_limits: tuple[float, float],
    fontsize: float,
) -> None:
    lower, upper = row_limits
    offset = max((upper - lower) * 0.03, 0.02)
    if value >= 0:
        y_value = min(value + offset, upper - offset * 0.5)
        va = "bottom"
    else:
        y_value = max(value - offset, lower + offset * 0.5)
        va = "top"
    ax.text(
        center_x,
        y_value,
        f"{value:.2f}",
        ha="center",
        va=va,
        rotation=90,
        fontsize=fontsize,
        color="#333333",
        zorder=6,
    )


def _row_limits_for_metric(rows: list[dict[str, object]], public_field: str, private_field: str) -> tuple[float, float]:
    values = [float(row[public_field]) for row in rows] + [float(row[private_field]) for row in rows]
    min_value = min(values)
    max_value = max(values)
    padding = max((max_value - min_value) * 0.18, 1.0)
    lower = min(min_value - padding, -0.5)
    upper = max(max_value + padding, 0.5)
    if lower < 0 < upper:
        return (lower, upper)
    return (min(lower, 0.0), max(upper, 0.0))


def _render_metric_row(
    *,
    ax: object,
    row_title: str,
    public_centers: list[float],
    private_centers: list[float],
    model_order: list[str],
    lookup_by_mode: dict[str, dict[str, dict[str, object]]],
    public_field: str,
    private_field: str,
    row_limits: tuple[float, float],
    row_index: int,
    variant: str,
    single_row: bool = False,
) -> None:
    style = PLOT_VARIANTS[variant]
    bar_width = 0.32
    colors = [
        get_model_color(
            model_name,
            MODEL_COLOR_MAP.get(
                model_name,
                "#4C78A8" if model_name in {BREADTH_LLAMA_SFT_PLOT_MODEL, DEPTH_LLAMA_SFT_PLOT_MODEL} else "#999999",
            ),
        )
        for model_name in model_order
    ]

    for simulation_mode, plot_spec in SIMULATION_MODE_PLOT_SPECS.items():
        mode_lookup = lookup_by_mode.get(simulation_mode, {})
        public_values = [
            math.nan if model_name not in mode_lookup else float(mode_lookup[model_name][public_field])
            for model_name in model_order
        ]
        private_values = [
            math.nan if model_name not in mode_lookup else float(mode_lookup[model_name][private_field])
            for model_name in model_order
        ]
        public_bars = ax.bar(
            [center + float(plot_spec["offset"]) for center in public_centers],
            public_values,
            width=bar_width,
            color=colors,
            edgecolor=str(plot_spec["edgecolor"]),
            linewidth=0.7,
            alpha=float(plot_spec["alpha"]),
            hatch=plot_spec["hatch"],
            zorder=3,
        )
        private_bars = ax.bar(
            [center + float(plot_spec["offset"]) for center in private_centers],
            private_values,
            width=bar_width,
            color=colors,
            edgecolor=str(plot_spec["edgecolor"]),
            linewidth=0.7,
            alpha=float(plot_spec["alpha"]),
            hatch=plot_spec["hatch"],
            zorder=3,
        )

        for bar, value in zip(public_bars, public_values):
            if math.isnan(value):
                continue
            _annotate_bar_value(
                ax,
                center_x=bar.get_x() + bar.get_width() / 2,
                value=value,
                row_limits=row_limits,
                fontsize=float(style["annotation_fontsize"]),
            )
        for bar, value in zip(private_bars, private_values):
            if math.isnan(value):
                continue
            _annotate_bar_value(
                ax,
                center_x=bar.get_x() + bar.get_width() / 2,
                value=value,
                row_limits=row_limits,
                fontsize=float(style["annotation_fontsize"]),
            )

    separator_x = (public_centers[-1] + private_centers[0]) / 2
    if bool(style["show_background_spans"]):
        ax.axvspan(public_centers[0] - 0.6, public_centers[-1] + 0.6, color="#4C78A8", alpha=0.03)
        ax.axvspan(private_centers[0] - 0.6, private_centers[-1] + 0.6, color="#F58518", alpha=0.03)
    ax.axvline(separator_x, color="#C7C7C7", linewidth=0.9, linestyle="--", alpha=0.8)
    if variant == "pub" and single_row:
        ax.set_ylabel(
            row_title,
            fontweight="bold",
            fontsize=float(style["row_title_fontsize"]) * 0.8,
            labelpad=12,
        )
    else:
        ax.set_title(
            row_title,
            loc="left",
            fontweight="bold",
            fontsize=float(style["row_title_fontsize"]),
            pad=float(style["row_title_pad"]),
        )
    ax.set_ylim(*row_limits)
    ax.set_xlim(public_centers[0] - 0.9, private_centers[-1] + 0.9)
    if bool(style["show_grid"]):
        ax.grid(True, axis="y", alpha=0.24)
    else:
        ax.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelbottom=False)
    ax.tick_params(axis="y", labelsize=float(style["y_tick_fontsize"]))
    if row_limits[0] < 0 < row_limits[1]:
        ax.axhline(0.0, color="#666666", linewidth=0.9, zorder=2)
    if row_index == 0 and bool(style["show_inline_panel_labels"]):
        ax.text(
            sum(public_centers) / len(public_centers),
            row_limits[1] - (row_limits[1] - row_limits[0]) * float(style["panel_label_y"]),
            "Public Opinion",
            ha="center",
            va="top",
            fontsize=float(style["panel_label_fontsize"]),
            fontweight="bold",
        )
        ax.text(
            sum(private_centers) / len(private_centers),
            row_limits[1] - (row_limits[1] - row_limits[0]) * float(style["panel_label_y"]),
            "Private Opinion",
            ha="center",
            va="top",
            fontsize=float(style["panel_label_fontsize"]),
            fontweight="bold",
        )


def _add_figure_panel_labels(fig: object, *, variant: str, single_row: bool = False) -> None:
    style = PLOT_VARIANTS[variant]
    if not bool(style["show_figure_panel_labels"]):
        return
    if single_row:
        public_x = 0.43
        private_x = 0.79
        label_y = min(float(style["panel_label_y"]) + 0.015, 0.99)
        label_fontsize = float(style["panel_label_fontsize"]) * 0.78
    else:
        public_x = 0.27
        private_x = 0.74
        label_y = float(style["panel_label_y"])
        label_fontsize = float(style["panel_label_fontsize"])
    fig.text(public_x, label_y, "Public Opinion", ha="center", va="top", fontsize=label_fontsize, fontweight="bold")
    fig.text(private_x, label_y, "Private Opinion", ha="center", va="top", fontsize=label_fontsize, fontweight="bold")


def _legend_bbox(*, variant: str, single_row: bool) -> tuple[float, float]:
    style = PLOT_VARIANTS[variant]
    default_bbox = tuple(style["legend_bbox"])
    if variant == "pub" and single_row:
        return (0.5, 0.93)
    return (float(default_bbox[0]), float(default_bbox[1]))


def _legend_loc(*, variant: str, single_row: bool) -> str:
    if variant == "pub" and single_row:
        return "upper center"
    return str(PLOT_VARIANTS[variant]["legend_loc"])


def _subplot_bounds(*, variant: str, single_row: bool) -> tuple[float, float]:
    style = PLOT_VARIANTS[variant]
    top = float(style["top"])
    bottom = float(style["bottom"])
    if variant == "pub" and single_row:
        return (0.80, max(bottom, 0.40))
    return (top, bottom)


def render_maintable_figure(*, split: str, rows: list[dict[str, object]], output_dir: Path) -> list[Path]:
    saved_paths: list[Path] = []
    split_rows = [row for row in rows if str(row["source_table"]) == split and str(row["model"]) not in PLOT_EXCLUDED_MODELS]
    if not split_rows:
        return saved_paths

    ordered_models = _ordered_model_names(split_rows)
    num_models = len(ordered_models)
    public_centers = list(range(num_models))
    private_centers = [position + num_models + PANEL_SECTION_GAP for position in public_centers]
    all_centers = public_centers + private_centers
    lookup_by_mode = {
        simulation_mode: {
            str(row["model"]): row
            for row in split_rows
            if str(row["simulation_mode"]) == simulation_mode
        }
        for simulation_mode in SIMULATION_MODE_PLOT_SPECS
    }

    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor="#BFBFBF", edgecolor="#1F1F1F", label=str(SIMULATION_MODE_PLOT_SPECS["Simulation Mode 3: Full Conversation Simulation (v0)"]["label"])),
        Patch(
            facecolor="#BFBFBF",
            edgecolor="#1F1F1F",
            hatch="///",
            alpha=0.75,
            label=str(SIMULATION_MODE_PLOT_SPECS["Simulation Mode 2: Tweet-guided Conversation Simulation (v1b)"]["label"]),
        ),
    ]

    for group_suffix, row_specs in FIGURE_ROW_GROUP_SPECS:
        for variant in ("raw", "pub"):
            style = PLOT_VARIANTS[variant]
            single_row = len(row_specs) == 1
            top, bottom = _subplot_bounds(variant=variant, single_row=single_row)
            labels = [plot_tick_label(model_name, variant=variant) for model_name in ordered_models]
            fig, axes = plt.subplots(
                len(row_specs),
                1,
                figsize=figure_size(style=style, family="maintable", row_count=len(row_specs)),
                sharex=False,
            )
            if hasattr(axes, "bar"):
                axes = [axes]
            else:
                axes = list(axes)
            if bool(style["show_suptitle"]):
                fig.suptitle(f"Beyond-Stance Metrics: {split.title()}", y=0.985, fontsize=14, fontweight="bold")
            _add_figure_panel_labels(fig, variant=variant, single_row=single_row)
            fig.legend(
                handles=legend_handles,
                loc=_legend_loc(variant=variant, single_row=single_row),
                ncol=2,
                frameon=False,
                bbox_to_anchor=_legend_bbox(variant=variant, single_row=single_row),
                fontsize=float(style["legend_fontsize"]),
            )

            for row_index, (axis, (row_title, public_field, private_field, fixed_limits)) in enumerate(zip(axes, row_specs)):
                row_source_rows = [
                    row
                    for row in split_rows
                    if str(row["simulation_mode"]) in SIMULATION_MODE_PLOT_SPECS
                ]
                row_limits = fixed_limits if fixed_limits is not None else _row_limits_for_metric(row_source_rows, public_field, private_field)
                _render_metric_row(
                    ax=axis,
                    row_title=row_title,
                    public_centers=public_centers,
                    private_centers=private_centers,
                    model_order=ordered_models,
                    lookup_by_mode=lookup_by_mode,
                    public_field=public_field,
                    private_field=private_field,
                    row_limits=row_limits,
                    row_index=row_index,
                    variant=variant,
                    single_row=single_row,
                )

            axes[-1].set_xticks(all_centers)
            axes[-1].set_xticklabels(labels + labels, rotation=float(style["tick_rotation"]), ha="right" if float(style["tick_rotation"]) < 90 else "center")
            axes[-1].tick_params(axis="x", labelbottom=True, pad=8, labelsize=float(style["tick_fontsize"]))
            for axis in axes[:-1]:
                axis.set_xticklabels([])

            fig.subplots_adjust(
                left=float(style["left"]),
                right=float(style["right"]),
                bottom=bottom,
                top=top,
                hspace=float(style["hspace"]),
            )
            stem = variant_stem(maintable_stem(split, group_suffix), variant=variant)
            section_dir = output_section_dir(output_dir, MAINTABLE_SUBDIR)
            pdf_path = section_dir / f"{stem}.pdf"
            png_path = section_dir / f"{stem}.png"
            fig.savefig(pdf_path, bbox_inches="tight")
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            saved_paths.extend([pdf_path, png_path])
    return saved_paths


def render_llama_split_figures(*, rows: list[dict[str, object]], base_depth_rows: list[dict[str, object]], output_dir: Path) -> list[Path]:
    saved_paths: list[Path] = []
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["generalization_type"]), str(row["train_test"]))
        grouped.setdefault(key, []).append(row)

    from matplotlib.patches import Patch

    base_model_order = _ordered_model_names(base_depth_rows)
    base_lookup_by_mode = {
        simulation_mode: {
            str(row["model"]): row
            for row in base_depth_rows
            if str(row["simulation_mode"]) == simulation_mode
        }
        for simulation_mode in SIMULATION_MODE_PLOT_SPECS
    }

    for (generalization_type, train_test), subset in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        post_row = next((row for row in subset if str(row["model_state"]) == "post-SFT"), None)
        if post_row is None:
            continue
        model_order = _insert_post_sft_model(base_model_order, str(post_row["real_model"]))
        num_models = len(model_order)
        public_centers = list(range(num_models))
        private_centers = [position + num_models + PANEL_SECTION_GAP for position in public_centers]
        all_centers = public_centers + private_centers

        lookup_by_mode: dict[str, dict[str, dict[str, object]]] = {}
        for simulation_mode in SIMULATION_MODE_PLOT_SPECS:
            mode_rows = [row for row in subset if str(row["simulation_mode"]) == simulation_mode]
            mode_lookup: dict[str, dict[str, object]] = dict(base_lookup_by_mode.get(simulation_mode, {}))
            for row in mode_rows:
                if str(row["model_state"]) == "pre-SFT":
                    mode_lookup["Llama-3.1-8B-Instruct"] = row
                elif str(row["model_state"]) == "post-SFT":
                    mode_lookup[str(row["real_model"])] = row
            lookup_by_mode[simulation_mode] = mode_lookup

        legend_handles = [
            Patch(facecolor="#BFBFBF", edgecolor="#1F1F1F", label=str(SIMULATION_MODE_PLOT_SPECS["Simulation Mode 3: Full Conversation Simulation (v0)"]["label"])),
            Patch(
                facecolor="#BFBFBF",
                edgecolor="#1F1F1F",
                hatch="///",
                alpha=0.75,
                label=str(SIMULATION_MODE_PLOT_SPECS["Simulation Mode 2: Tweet-guided Conversation Simulation (v1b)"]["label"]),
            ),
        ]
        generalization_slug = "group" if generalization_type.startswith("Group") else "topic"
        split_slug = str(train_test).lower()
        stem = f"{generalization_slug}_{split_slug}"
        saved_paths.append(
            write_split_plot_data_csv(
                output_dir=output_dir,
                stem=stem,
                split_family="depth",
                generalization_type=generalization_type,
                train_test=str(train_test),
                model_order=model_order,
                lookup_by_mode=lookup_by_mode,
            )
        )
        for group_suffix, row_specs in FIGURE_ROW_GROUP_SPECS:
            for variant in ("raw", "pub"):
                style = PLOT_VARIANTS[variant]
                single_row = len(row_specs) == 1
                top, bottom = _subplot_bounds(variant=variant, single_row=single_row)
                labels = [plot_tick_label(model_name, variant=variant) for model_name in model_order]
                fig, axes = plt.subplots(
                    len(row_specs),
                    1,
                    figsize=figure_size(style=style, family="llama", row_count=len(row_specs)),
                    sharex=False,
                )
                if hasattr(axes, "bar"):
                    axes = [axes]
                else:
                    axes = list(axes)
                if bool(style["show_suptitle"]):
                    fig.suptitle(
                        f"Llama SFT Depth: {GENERALIZATION_DISPLAY[generalization_type]} / {train_test}",
                        y=0.985,
                        fontsize=14,
                        fontweight="bold",
                    )
                if bool(style["show_subtitle"]):
                    fig.text(0.5, 0.952, "Depth base-model roster with the split-specific LLAMA SFT row inserted", ha="center", va="center", fontsize=10)
                _add_figure_panel_labels(fig, variant=variant, single_row=single_row)
                fig.legend(
                    handles=legend_handles,
                    loc=_legend_loc(variant=variant, single_row=single_row),
                    ncol=2,
                    frameon=False,
                    bbox_to_anchor=_legend_bbox(variant=variant, single_row=single_row),
                    fontsize=float(style["legend_fontsize"]),
                )

                for row_index, (axis, (row_title, public_field, private_field, fixed_limits)) in enumerate(zip(axes, row_specs)):
                    row_limits = fixed_limits if fixed_limits is not None else _row_limits_for_metric(subset, public_field, private_field)
                    _render_metric_row(
                        ax=axis,
                        row_title=row_title,
                        public_centers=public_centers,
                        private_centers=private_centers,
                        model_order=model_order,
                        lookup_by_mode=lookup_by_mode,
                        public_field=public_field,
                        private_field=private_field,
                        row_limits=row_limits,
                        row_index=row_index,
                        variant=variant,
                        single_row=single_row,
                    )

                axes[-1].set_xticks(all_centers)
                axes[-1].set_xticklabels(labels + labels, rotation=float(style["llama_tick_rotation"]), ha="right" if float(style["llama_tick_rotation"]) < 90 else "center")
                axes[-1].tick_params(axis="x", labelbottom=True, pad=8, labelsize=float(style["tick_fontsize"]))
                for axis in axes[:-1]:
                    axis.set_xticklabels([])

                fig.subplots_adjust(
                    left=float(style["left"]),
                    right=float(style["right"]),
                    bottom=bottom,
                    top=top,
                    hspace=float(style["hspace"]),
                )
                base_stem = sft_stem(generalization_slug, split_slug, group_suffix)
                section_dir = output_section_dir(output_dir, SFT_DEPTH_SUBDIR)
                pdf_path = section_dir / f"{variant_stem(base_stem, variant=variant)}.pdf"
                png_path = section_dir / f"{variant_stem(base_stem, variant=variant)}.png"
                fig.savefig(pdf_path, bbox_inches="tight")
                fig.savefig(png_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                saved_paths.extend([pdf_path, png_path])

    return saved_paths


def write_split_plot_data_csv(
    *,
    output_dir: Path,
    stem: str,
    split_family: str,
    generalization_type: str,
    train_test: str,
    model_order: list[str],
    lookup_by_mode: dict[str, dict[str, dict[str, object]]],
) -> Path:
    rows: list[dict[str, object]] = []
    for simulation_mode, mode_lookup in lookup_by_mode.items():
        for plot_order, model_name in enumerate(model_order, start=1):
            if model_name not in mode_lookup:
                continue
            source_row = mode_lookup[model_name]
            rows.append(
                {
                    "split_family": split_family,
                    "generalization_type": generalization_type,
                    "train_test": train_test,
                    "simulation_mode": simulation_mode,
                    "simulation_mode_short": SIMULATION_MODE_SHORT[simulation_mode],
                    "plot_model_order": plot_order,
                    "plot_model": model_name,
                    "plot_label": plot_tick_label(model_name, variant="pub"),
                    "model_state": str(source_row.get("model_state", "base")),
                    "real_model": str(source_row.get("real_model", model_name)),
                    **{metric_key: source_row[metric_key] for metric_key in METRIC_COLUMNS},
                }
            )

    section_dir = output_section_dir(output_dir, SFT_BREADTH_SUBDIR if split_family == "breadth" else SFT_DEPTH_SUBDIR)
    csv_path = section_dir / f"{stem}_plot_data.csv"
    write_csv(
        csv_path,
        [
            "split_family",
            "generalization_type",
            "train_test",
            "simulation_mode",
            "simulation_mode_short",
            "plot_model_order",
            "plot_model",
            "plot_label",
            "model_state",
            "real_model",
            *METRIC_COLUMNS.keys(),
        ],
        rows,
    )
    return csv_path


def render_llama_breadth_split_figures(*, rows: list[dict[str, object]], base_breadth_rows: list[dict[str, object]], output_dir: Path) -> list[Path]:
    saved_paths: list[Path] = []
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["generalization_type"]), str(row["train_test"]))
        grouped.setdefault(key, []).append(row)

    from matplotlib.patches import Patch

    base_model_order = _ordered_model_names(base_breadth_rows)
    base_lookup_by_mode = {
        simulation_mode: {
            str(row["model"]): row
            for row in base_breadth_rows
            if str(row["simulation_mode"]) == simulation_mode
        }
        for simulation_mode in SIMULATION_MODE_PLOT_SPECS
    }

    for (generalization_type, train_test), subset in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        post_row = next((row for row in subset if str(row["model_state"]) == "post-SFT"), None)
        if post_row is None:
            continue
        model_order = _insert_post_sft_model(base_model_order, str(post_row["real_model"]))
        num_models = len(model_order)
        public_centers = list(range(num_models))
        private_centers = [position + num_models + PANEL_SECTION_GAP for position in public_centers]
        all_centers = public_centers + private_centers

        lookup_by_mode: dict[str, dict[str, dict[str, object]]] = {}
        for simulation_mode in SIMULATION_MODE_PLOT_SPECS:
            mode_rows = [row for row in subset if str(row["simulation_mode"]) == simulation_mode]
            mode_lookup: dict[str, dict[str, object]] = dict(base_lookup_by_mode.get(simulation_mode, {}))
            for row in mode_rows:
                if str(row["model_state"]) == "pre-SFT":
                    mode_lookup["Llama-3.1-8B-Instruct"] = row
                elif str(row["model_state"]) == "post-SFT":
                    mode_lookup[str(row["real_model"])] = row
            lookup_by_mode[simulation_mode] = mode_lookup

        legend_handles = [
            Patch(facecolor="#BFBFBF", edgecolor="#1F1F1F", label=str(SIMULATION_MODE_PLOT_SPECS["Simulation Mode 3: Full Conversation Simulation (v0)"]["label"])),
            Patch(
                facecolor="#BFBFBF",
                edgecolor="#1F1F1F",
                hatch="///",
                alpha=0.75,
                label=str(SIMULATION_MODE_PLOT_SPECS["Simulation Mode 2: Tweet-guided Conversation Simulation (v1b)"]["label"]),
            ),
        ]
        generalization_slug = "group" if generalization_type.startswith("Group") else "topic"
        split_slug = str(train_test).lower()
        stem = f"{generalization_slug}_{split_slug}"
        saved_paths.append(
            write_split_plot_data_csv(
                output_dir=output_dir,
                stem=stem,
                split_family="breadth",
                generalization_type=generalization_type,
                train_test=str(train_test),
                model_order=model_order,
                lookup_by_mode=lookup_by_mode,
            )
        )
        for group_suffix, row_specs in FIGURE_ROW_GROUP_SPECS:
            for variant in ("raw", "pub"):
                style = PLOT_VARIANTS[variant]
                single_row = len(row_specs) == 1
                top, bottom = _subplot_bounds(variant=variant, single_row=single_row)
                labels = [plot_tick_label(model_name, variant=variant) for model_name in model_order]
                fig, axes = plt.subplots(
                    len(row_specs),
                    1,
                    figsize=figure_size(style=style, family="llama", row_count=len(row_specs)),
                    sharex=False,
                )
                if hasattr(axes, "bar"):
                    axes = [axes]
                else:
                    axes = list(axes)
                if bool(style["show_suptitle"]):
                    fig.suptitle(
                        f"Llama SFT Breadth: {GENERALIZATION_DISPLAY[generalization_type]} / {train_test}",
                        y=0.985,
                        fontsize=14,
                        fontweight="bold",
                    )
                if bool(style["show_subtitle"]):
                    fig.text(0.5, 0.952, "Breadth base-model roster with the split-specific LLAMA SFT row inserted", ha="center", va="center", fontsize=10)
                _add_figure_panel_labels(fig, variant=variant, single_row=single_row)
                fig.legend(
                    handles=legend_handles,
                    loc=_legend_loc(variant=variant, single_row=single_row),
                    ncol=2,
                    frameon=False,
                    bbox_to_anchor=_legend_bbox(variant=variant, single_row=single_row),
                    fontsize=float(style["legend_fontsize"]),
                )

                for row_index, (axis, (row_title, public_field, private_field, fixed_limits)) in enumerate(zip(axes, row_specs)):
                    row_limits = fixed_limits if fixed_limits is not None else _row_limits_for_metric(subset, public_field, private_field)
                    _render_metric_row(
                        ax=axis,
                        row_title=row_title,
                        public_centers=public_centers,
                        private_centers=private_centers,
                        model_order=model_order,
                        lookup_by_mode=lookup_by_mode,
                        public_field=public_field,
                        private_field=private_field,
                        row_limits=row_limits,
                        row_index=row_index,
                        variant=variant,
                        single_row=single_row,
                    )

                axes[-1].set_xticks(all_centers)
                axes[-1].set_xticklabels(labels + labels, rotation=float(style["llama_tick_rotation"]), ha="right" if float(style["llama_tick_rotation"]) < 90 else "center")
                axes[-1].tick_params(axis="x", labelbottom=True, pad=8, labelsize=float(style["tick_fontsize"]))
                for axis in axes[:-1]:
                    axis.set_xticklabels([])

                fig.subplots_adjust(
                    left=float(style["left"]),
                    right=float(style["right"]),
                    bottom=bottom,
                    top=top,
                    hspace=float(style["hspace"]),
                )
                base_stem = sft_stem(generalization_slug, split_slug, group_suffix)
                section_dir = output_section_dir(output_dir, SFT_BREADTH_SUBDIR)
                pdf_path = section_dir / f"{variant_stem(base_stem, variant=variant)}.pdf"
                png_path = section_dir / f"{variant_stem(base_stem, variant=variant)}.png"
                fig.savefig(pdf_path, bbox_inches="tight")
                fig.savefig(png_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                saved_paths.extend([pdf_path, png_path])

    return saved_paths


def export_beyond_stance(*, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_outdated_outputs(output_dir)
    maintable_rows: list[dict[str, object]] = []
    for split, path in MAINTABLE_FILES.items():
        maintable_rows.extend(parse_maintable_rows(path, split))
    breadth_llama_rows = parse_llama_rows(LLAMA_SFT_BREADTH_FILE)
    breadth_llama_rows.sort(key=lambda row: (str(row["generalization_type"]), str(row["simulation_mode_short"]), str(row["train_test"]), str(row["model_state"])))
    breadth_llama_comparison = build_llama_comparison(breadth_llama_rows)
    maintable_rows.extend(build_breadth_llama_sft_plot_rows(breadth_llama_rows))

    llama_rows = parse_llama_rows(LLAMA_SFT_DEPTH_FILE)
    llama_rows.sort(key=lambda row: (str(row["generalization_type"]), str(row["simulation_mode_short"]), str(row["train_test"]), str(row["model_state"])))
    llama_comparison = build_llama_comparison(llama_rows)
    maintable_rows.extend(build_depth_llama_sft_plot_rows(llama_rows))
    maintable_rows.sort(key=lambda row: (str(row["source_table"]), str(row["simulation_mode_short"]), -float(row["semantic_similarity_tweet"])))
    maintable_summary = build_maintable_summary(maintable_rows)
    base_breadth_rows = _base_breadth_plot_rows(maintable_rows)
    base_depth_rows = _base_depth_plot_rows(maintable_rows)

    saved_paths: list[Path] = []
    csv_dir = output_section_dir(output_dir, CSV_SUBDIR)
    maintable_fieldnames = [
        "source_table",
        "simulation_mode",
        "simulation_mode_short",
        "model",
        *METRIC_COLUMNS.keys(),
    ]
    maintable_csv_path = csv_dir / "maintable_rows.csv"
    write_csv(maintable_csv_path, maintable_fieldnames, maintable_rows)
    saved_paths.append(maintable_csv_path)

    maintable_summary_csv_path = csv_dir / "maintable_summary.csv"
    write_csv(
        maintable_summary_csv_path,
        [
            "source_table",
            "simulation_mode",
            "simulation_mode_short",
            "avg_semantic_similarity_tweet",
            "avg_semantic_similarity_post_opinion",
            "avg_n_word_diff_tweet",
            "avg_n_word_diff_post_opinion",
            "avg_rouge_l_tweet",
            "avg_rouge_l_post_opinion",
            "avg_valid_response_rate",
            "best_tweet_semantic_model",
            "best_tweet_semantic_value",
            "best_post_semantic_model",
            "best_post_semantic_value",
            "best_valid_model",
            "best_valid_value",
        ],
        maintable_summary,
    )
    saved_paths.append(maintable_summary_csv_path)

    breadth_llama_rows_csv_path = csv_dir / "sft_breadth_rows.csv"
    write_csv(
        breadth_llama_rows_csv_path,
        [
            "generalization_type",
            "simulation_mode",
            "simulation_mode_short",
            "train_test",
            "model_state",
            "real_model",
            *METRIC_COLUMNS.keys(),
        ],
        breadth_llama_rows,
    )
    saved_paths.append(breadth_llama_rows_csv_path)

    breadth_llama_comparison_csv_path = csv_dir / "sft_breadth_delta.csv"
    write_csv(
        breadth_llama_comparison_csv_path,
        [
            "generalization_type",
            "simulation_mode",
            "simulation_mode_short",
            "train_test",
            "pre_semantic_similarity_tweet",
            "post_semantic_similarity_tweet",
            "delta_semantic_similarity_tweet",
            "pre_semantic_similarity_post_opinion",
            "post_semantic_similarity_post_opinion",
            "delta_semantic_similarity_post_opinion",
            "pre_n_word_diff_tweet",
            "post_n_word_diff_tweet",
            "delta_n_word_diff_tweet",
            "pre_n_word_diff_post_opinion",
            "post_n_word_diff_post_opinion",
            "delta_n_word_diff_post_opinion",
            "pre_rouge_l_tweet",
            "post_rouge_l_tweet",
            "delta_rouge_l_tweet",
            "pre_rouge_l_post_opinion",
            "post_rouge_l_post_opinion",
            "delta_rouge_l_post_opinion",
            "pre_valid_response_rate",
            "post_valid_response_rate",
            "delta_valid_response_rate",
        ],
        breadth_llama_comparison,
    )
    saved_paths.append(breadth_llama_comparison_csv_path)

    llama_rows_csv_path = csv_dir / "sft_depth_rows.csv"
    write_csv(
        llama_rows_csv_path,
        [
            "generalization_type",
            "simulation_mode",
            "simulation_mode_short",
            "train_test",
            "model_state",
            "real_model",
            *METRIC_COLUMNS.keys(),
        ],
        llama_rows,
    )
    saved_paths.append(llama_rows_csv_path)

    llama_comparison_csv_path = csv_dir / "sft_depth_delta.csv"
    write_csv(
        llama_comparison_csv_path,
        [
            "generalization_type",
            "simulation_mode",
            "simulation_mode_short",
            "train_test",
            "pre_semantic_similarity_tweet",
            "post_semantic_similarity_tweet",
            "delta_semantic_similarity_tweet",
            "pre_semantic_similarity_post_opinion",
            "post_semantic_similarity_post_opinion",
            "delta_semantic_similarity_post_opinion",
            "pre_n_word_diff_tweet",
            "post_n_word_diff_tweet",
            "delta_n_word_diff_tweet",
            "pre_n_word_diff_post_opinion",
            "post_n_word_diff_post_opinion",
            "delta_n_word_diff_post_opinion",
            "pre_rouge_l_tweet",
            "post_rouge_l_tweet",
            "delta_rouge_l_tweet",
            "pre_rouge_l_post_opinion",
            "post_rouge_l_post_opinion",
            "delta_rouge_l_post_opinion",
            "pre_valid_response_rate",
            "post_valid_response_rate",
            "delta_valid_response_rate",
        ],
        llama_comparison,
    )
    saved_paths.append(llama_comparison_csv_path)

    saved_paths.append(build_markdown_summary(maintable_summary=maintable_summary, llama_comparison=llama_comparison, output_dir=output_dir))
    for split in MAINTABLE_FILES:
        saved_paths.extend(render_maintable_figure(split=split, rows=maintable_rows, output_dir=output_dir))
    saved_paths.extend(render_llama_breadth_split_figures(rows=breadth_llama_rows, base_breadth_rows=base_breadth_rows, output_dir=output_dir))
    saved_paths.extend(render_llama_split_figures(rows=llama_rows, base_depth_rows=base_depth_rows, output_dir=output_dir))
    return saved_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create beyond-stance tables and figures from the v4 raw result tables.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    for output_path in export_beyond_stance(output_dir=args.output_dir.resolve()):
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())