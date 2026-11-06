#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import sys
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
RADAR_SRC = REPO_ROOT / "neurips_2026" / "chengtao" / "src" / "radar"
if str(RADAR_SRC) not in sys.path:
    sys.path.insert(0, str(RADAR_SRC))
TRANSITION_SRC = REPO_ROOT / "neurips_2026" / "leo" / "transition" / "src"
if str(TRANSITION_SRC) not in sys.path:
    sys.path.insert(0, str(TRANSITION_SRC))

from neurips_2026.sean_reproduce.base_model_v3_style import (  # noqa: E402
    CANONICAL_V3_HUMAN_SOURCE_MODEL,
    BASE_MODEL_V3_COLOR_MAP,
    BASE_MODEL_V3_DISPLAY_LABELS,
    BASE_MODEL_V3_LLM_ORDER,
    ordered_base_model_llms_for_report,
)
from neurips_2026.chengtao.src.radar.post_training_breadth_profiles import (  # noqa: E402
    DEPTH_GROUP_MODELS_V4,
    DEPTH_TOPIC_MODELS_V4,
    GROUP_MODELS_V4,
    TOPIC_MODELS_V4,
)
from neurips_2026.sean_reproduce.result_version_config import (  # noqa: E402
    apply_transition_simulation_version_for_result,
    report_family_for_version,
    report_namespace_for_version,
)
from neurips_2026.sean_reproduce.v3.masks import enable_v3_complete_group_masks  # noqa: E402

from utils import (  # noqa: E402
    collection_golden_mask_path,
    collection_split_mask_path,
    get_topic,
    human_transition_csv_filename,
    initial_post_rejudged_root,
    llm_transition_csv_filename,
    load_eval_dataframe,
    prepare_golden_dps,
    read_eval_csv_rows,
    result_root,
    split_membership,
    transition_simulation_version,
    tweet_rejudged_root,
)


SUMMARY_BASE_MODEL_COLOR_MAP = dict(BASE_MODEL_V3_COLOR_MAP)

BASE_MODELS = [
    {"model_dir": model_name, "display": BASE_MODEL_V3_DISPLAY_LABELS[model_name]}
    for model_name in BASE_MODEL_V3_LLM_ORDER
]
BASE_MODEL_SPECS_BY_MODEL = {spec["model_dir"]: spec for spec in BASE_MODELS}

HUMAN_SERIES = {
    "label": "human",
    "model_dir": CANONICAL_V3_HUMAN_SOURCE_MODEL,
    "display": BASE_MODEL_V3_DISPLAY_LABELS["human"],
}
LLM_CSV_FILENAME = llm_transition_csv_filename()
HUMAN_CSV_FILENAME = human_transition_csv_filename()
OVERALL_TOPIC_KEY = "__overall__"
GPT_4O_MINI_BASE_MODEL = "gpt-4o-mini-2024-07-18"
GPT_DEPTH_TOPIC_SFT_MODEL = "ft:gpt-4o-mini-2024-07-18:camer:sft-20251218-depth-topic-5epochs:CoFxFPcv"
GPT_DEPTH_GROUP_SFT_MODEL = "ft:gpt-4o-mini-2024-07-18:camer:sft-20251218-depth-group-5epochs:CoFzijg0"
GPT_4O_MINI_SFT_COLOR = "#B85F56"
DEGROOT_FIT_MODEL_NAME = "DeGroot fit"
DEGROOT_FIT_COLOR = "#9A9A9A"
LINE_PLOT_LLAMA_SFT_COLOR = "#3F6B97"

COMBINED_EVENTS = [
    ("initial_opinion", "Initial", "Initial Opinion", None),
    ("tweet_1", "Tweet 1", "tweet", 1.0),
    ("tweet_2", "Tweet 2", "tweet", 2.0),
    ("tweet_3", "Tweet 3", "tweet", 3.0),
    ("post_opinion", "Post", "Post Opinion", None),
]
SHARED_HUMAN_ANCHOR_KEYS = {"initial_opinion", "tweet_1"}
SEAN_REPRODUCE_ROOT = REPO_ROOT / "neurips_2026" / "sean_reproduce"
REQUIRED_COMPLETE_PARTICIPANTS = 4
PUBLIC_POSITION_KEYS = ("tweet_1", "tweet_2", "tweet_3")
PRIVATE_POSITION_KEYS = ("initial_opinion", "post_opinion")

POST_TRAINING_MODEL_DIRS = {
    "breadth": {
        "topic": {
            "instruct": "Llama-3.1-8B-Instruct",
            "sft": "ft:Llama-3.1-8B-SFT-20260210:breadth-topic-3epochs-unsloth",
            "dpo": "ft:Llama-3.1-8B-Instruct-DPO-20260212:breadth-topic-5epochs",
        },
        "group": {
            "instruct": "Llama-3.1-8B-Instruct",
            "sft": "ft:Llama-3.1-8B-SFT-20260210:breadth-group-3epochs-unsloth",
            "dpo": "ft:Llama-3.1-8B-Instruct-DPO-20260211:breadth-group-5epochs",
        },
    },
    "depth": {
        "topic": {
            "instruct": "Llama-3.1-8B-Instruct",
            "sft": "ft:Llama-3.1-8B-Instruct-SFT-20251218:depth-topic-5epochs",
            "extra_sfts": [GPT_DEPTH_TOPIC_SFT_MODEL],
            "dpo": "ft:Llama-3.1-8B-Instruct-DPO-20260105:depth-topic-5epochs",
        },
        "group": {
            "instruct": "Llama-3.1-8B-Instruct",
            "sft": "ft:Llama-3.1-8B-Instruct-SFT-20251218:depth-group-5epochs",
            "extra_sfts": [GPT_DEPTH_GROUP_SFT_MODEL],
            "dpo": "ft:Llama-3.1-8B-Instruct-DPO-20260105:depth-group-5epochs",
        },
    },
}

POST_TRAINING_MODEL_DIRS_V4 = {
    "breadth": {
        "topic": {
            "instruct": TOPIC_MODELS_V4[0],
            "sft": TOPIC_MODELS_V4[1],
        },
        "group": {
            "instruct": GROUP_MODELS_V4[0],
            "sft": GROUP_MODELS_V4[1],
        },
    },
    "depth": {
        "topic": {
            "instruct": DEPTH_TOPIC_MODELS_V4[0],
            "sft": DEPTH_TOPIC_MODELS_V4[2],
            "extra_sfts": [GPT_DEPTH_TOPIC_SFT_MODEL],
        },
        "group": {
            "instruct": DEPTH_GROUP_MODELS_V4[1],
            "sft": DEPTH_GROUP_MODELS_V4[2],
            "extra_sfts": [GPT_DEPTH_GROUP_SFT_MODEL],
        },
    },
}

INITIAL_POST_REJUDGE_SUFFIX = "__initial_post_rejudge.csv"
TWEET_REJUDGE_SUFFIX = "__tweet_rejudge.csv"
REFERENCE_LINE_SPECS = [
    ("One-Step Toward First Partner", "#1F77B4"),
    ("50% SD Reduction", "#7F7F7F"),
]
REFERENCE_LINE_WIDTH = 2.4
REFERENCE_LINE_ALPHA = 1.0
REFERENCE_LINE_DASH_PATTERN = (0, (1.2, 2.0))
REFERENCE_LINE_PATH_EFFECTS = [
    pe.Stroke(linewidth=4.0, foreground="white", alpha=0.95),
    pe.Normal(),
]
REFERENCE_MARKER = "D"
REFERENCE_MARKER_SIZE = 7.0
REFERENCE_MARKER_EDGE_WIDTH = 1.2
REFERENCE_MARKER_X_OFFSET = 0.12
LINE_COLOR_DARKEN_FACTOR = 0.82
BACKGROUND_MODEL_ALPHA = 0.28
BOOTSTRAP_REPS = 1000
BOOTSTRAP_SEED = 1


def normalize_topic(topic: str) -> str:
    return str(topic).replace("_", " ").strip().lower()


def collection_label(collection_type: str) -> str:
    return str(collection_type).strip().lower().capitalize()


def comparison_label(comparison_set: str) -> str:
    return "Base-Model" if comparison_set == "base-models" else "Post-Training"


def comparison_slug(comparison_set: str) -> str:
    return "base_model" if comparison_set == "base-models" else "post_train"


def report_version_label(report_version: str) -> str:
    return str(report_version)


def report_version_family(report_version: str) -> str:
    return report_family_for_version(report_version)


def report_version_namespace(report_version: str) -> str:
    return report_namespace_for_version(report_version)


def default_output_dir(comparison_set: str, report_version: str) -> Path:
    family_root = SEAN_REPRODUCE_ROOT / report_version_namespace(report_version) / "line_plot"
    return family_root / ("base_model" if comparison_set == "base-models" else "post_train")


def skip_v4_breadth_topic_post_train(collection_type: str, split: str | None, report_version: str) -> bool:
    return report_version_family(report_version) == "v4" and collection_type == "breadth" and split == "topic"


def include_base_model_series_in_post_training(report_version: str) -> bool:
    return report_version_family(report_version) == "v4"


def skip_breadth_per_topic_pages_in_post_training(
    collection_type: str,
    comparison_set: str,
    report_version: str,
) -> bool:
    return (
        comparison_set == "post-training"
        and report_version_family(report_version) == "v4"
        and collection_type == "breadth"
    )


def post_training_model_dirs_for_report(report_version: str) -> dict[str, dict[str, dict[str, object]]]:
    if report_version_family(report_version) == "v4":
        return POST_TRAINING_MODEL_DIRS_V4
    return POST_TRAINING_MODEL_DIRS


def selection_slug(collection_type: str, split: str | None, partition: str) -> str:
    if split is None or partition == "all":
        return collection_type
    return f"{collection_type}_{split}_{partition}"


def selection_context_label(collection_type: str, split: str | None, partition: str) -> str:
    base = f"Golden {collection_label(collection_type)} Groups"
    if split is None or partition == "all":
        return base
    return f"{base} ({split.capitalize()} Split, {partition.capitalize()} Partition)"


def selection_display_label(collection_type: str, split: str | None, partition: str) -> str:
    pieces = [collection_label(collection_type)]
    if split is not None:
        pieces.append(split.capitalize())
    if partition != "all":
        pieces.append(partition.capitalize())
    return " ".join(pieces)


def default_report_path(
    output_dir: Path,
    collection_type: str,
    comparison_set: str,
    split: str | None,
    partition: str,
    report_version: str,
) -> Path:
    return output_dir / (
        f"report_{selection_slug(collection_type, split, partition)}_"
        f"{comparison_slug(comparison_set)}_line_plots_{report_version_label(report_version)}.pdf"
    )


def build_report_page_specs(scope_label: str) -> list[dict[str, object]]:
    return [
        {
            "metric_name": "group_mean",
            "metric_kind": "mean",
            "page_title": f"Tweet Group Mean Stance Across {scope_label}",
            "y_label": "Mean group stance (Likert)",
            "event_keys": ["tweet_1", "tweet_2", "tweet_3"],
        },
        {
            "metric_name": "group_mean",
            "metric_kind": "mean",
            "page_title": f"Initial/Post Opinion Group Mean Stance Across {scope_label}",
            "y_label": "Mean group stance (Likert)",
            "event_keys": ["initial_opinion", "post_opinion"],
        },
        {
            "metric_name": "group_sd",
            "metric_kind": "sd",
            "page_title": f"Tweet Group Stance SD Across {scope_label}",
            "y_label": "Mean within-group stance SD",
            "event_keys": ["tweet_1", "tweet_2", "tweet_3"],
        },
        {
            "metric_name": "group_sd",
            "metric_kind": "sd",
            "page_title": f"Initial/Post Opinion Group Stance SD Across {scope_label}",
            "y_label": "Mean within-group stance SD",
            "event_keys": ["initial_opinion", "post_opinion"],
        },
    ]


def build_plot_specs(scope_label: str, file_prefix_root: str) -> list[dict[str, object]]:
    return [
        {
            "metric_name": "group_mean",
            "title_prefix": f"Group Mean Stance Across {scope_label}",
            "y_label": "Mean group stance (Likert)",
            "file_prefix": f"{file_prefix_root}_group_mean",
            "metric_kind": "mean",
        },
        {
            "metric_name": "group_sd",
            "title_prefix": f"Within-Group Stance SD Across {scope_label}",
            "y_label": "Mean within-group stance SD",
            "file_prefix": f"{file_prefix_root}_group_sd",
            "metric_kind": "sd",
        },
    ]


def post_training_color_for_series(series_key: str) -> str:
    if series_key == HUMAN_SERIES["label"]:
        return SUMMARY_BASE_MODEL_COLOR_MAP["human"]
    if series_key == DEGROOT_FIT_MODEL_NAME:
        return DEGROOT_FIT_COLOR
    if series_key == GPT_4O_MINI_BASE_MODEL:
        return SUMMARY_BASE_MODEL_COLOR_MAP[GPT_4O_MINI_BASE_MODEL]
    if series_key in {GPT_DEPTH_TOPIC_SFT_MODEL, GPT_DEPTH_GROUP_SFT_MODEL}:
        return GPT_4O_MINI_SFT_COLOR
    lower = str(series_key).lower()
    if "dpo" in lower:
        return "#D77A61"
    if "sft" in lower:
        return LINE_PLOT_LLAMA_SFT_COLOR
    return "#999999"


def post_training_display_label(model_dir: str) -> str:
    if model_dir == DEGROOT_FIT_MODEL_NAME:
        return DEGROOT_FIT_MODEL_NAME
    if model_dir in BASE_MODEL_V3_DISPLAY_LABELS:
        return BASE_MODEL_V3_DISPLAY_LABELS[model_dir]
    if model_dir in {GPT_DEPTH_TOPIC_SFT_MODEL, GPT_DEPTH_GROUP_SFT_MODEL}:
        return "gpt-4o-mini SFT"
    lower = str(model_dir).lower()
    if "dpo" in lower:
        return "Llama DPO"
    if "llama" in lower and "sft" in lower:
        return "Llama SFT"
    if "sft" in lower:
        return "SFT"
    return str(model_dir)


def build_series_specs(
    comparison_set: str,
    collection_type: str,
    split: str | None,
    report_version: str,
) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    added_labels: set[str] = set()

    def append_spec(label: str, model_dir: str, csv_name: str, display: str) -> None:
        if label in added_labels:
            return
        specs.append(
            {
                "label": label,
                "model_dir": model_dir,
                "csv_name": csv_name,
                "display": display,
            }
        )
        added_labels.add(label)

    append_spec(
        HUMAN_SERIES["label"],
        HUMAN_SERIES["model_dir"],
        HUMAN_CSV_FILENAME,
        HUMAN_SERIES["display"],
    )
    if comparison_set == "base-models":
        for model_name in ordered_base_model_llms_for_report(
            report_version,
            [spec["model_dir"] for spec in BASE_MODELS],
        ):
            model_spec = BASE_MODEL_SPECS_BY_MODEL[model_name]
            append_spec(
                model_spec["model_dir"],
                model_spec["model_dir"],
                LLM_CSV_FILENAME,
                model_spec["display"],
            )
        return specs

    if split is None:
        raise ValueError("post-training comparison requires --split topic or --split group")

    if include_base_model_series_in_post_training(report_version):
        for model_name in ordered_base_model_llms_for_report(
            report_version,
            [spec["model_dir"] for spec in BASE_MODELS],
        ):
            model_spec = BASE_MODEL_SPECS_BY_MODEL[model_name]
            append_spec(
                model_spec["model_dir"],
                model_spec["model_dir"],
                LLM_CSV_FILENAME,
                model_spec["display"],
            )

    model_dirs = post_training_model_dirs_for_report(report_version)[collection_type][split]
    append_spec(
        model_dirs["instruct"],
        model_dirs["instruct"],
        LLM_CSV_FILENAME,
        post_training_display_label(model_dirs["instruct"]),
    )
    prefer_llama_sft_after_instruct = report_version_family(report_version) == "v4"
    if not prefer_llama_sft_after_instruct and collection_type == "depth":
        append_spec(
            GPT_4O_MINI_BASE_MODEL,
            GPT_4O_MINI_BASE_MODEL,
            LLM_CSV_FILENAME,
            "gpt-4o-mini",
        )
    append_spec(
        model_dirs["sft"],
        model_dirs["sft"],
        LLM_CSV_FILENAME,
        post_training_display_label(model_dirs["sft"]),
    )
    if prefer_llama_sft_after_instruct and collection_type == "depth":
        append_spec(
            GPT_4O_MINI_BASE_MODEL,
            GPT_4O_MINI_BASE_MODEL,
            LLM_CSV_FILENAME,
            "gpt-4o-mini",
        )
    for extra_sft in model_dirs.get("extra_sfts", []):
        append_spec(
            extra_sft,
            extra_sft,
            LLM_CSV_FILENAME,
            post_training_display_label(extra_sft),
        )
    dpo_model = model_dirs.get("dpo")
    if dpo_model:
        append_spec(
            str(dpo_model),
            str(dpo_model),
            LLM_CSV_FILENAME,
            post_training_display_label(str(dpo_model)),
        )
    return specs


def selected_prefixes(collection_type: str, split: str | None, partition: str) -> list[str]:
    prefixes = prepare_golden_dps(collection_type)
    if split is None or partition == "all":
        return prefixes
    membership = split_membership(split, collection_type)
    return [prefix for prefix in prefixes if membership.get(prefix) == partition]


def selected_mask_path(collection_type: str, split: str | None, partition: str) -> Path:
    if split is None or partition == "all":
        return collection_golden_mask_path(collection_type)
    return collection_split_mask_path(collection_type, split, partition)


def display_topic(topic: str) -> str:
    return str(topic).replace("_", " ").strip()


def topic_display_label(topic_key: str, topic_display_map: dict[str, str]) -> str:
    if topic_key == OVERALL_TOPIC_KEY:
        return "Overall"
    return topic_display_map.get(topic_key, display_topic(topic_key))


def slugify(value: str) -> str:
    text = str(value).strip().lower()
    return "_".join(part for part in "".join(ch if ch.isalnum() else " " for ch in text).split()) or "plot"


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def wrap_text(text: str, width: int) -> str:
    return textwrap.fill(str(text), width=width, break_long_words=False, break_on_hyphens=False)


def darken_hex_color(color: str, factor: float = LINE_COLOR_DARKEN_FACTOR) -> str:
    text = str(color)
    if not text.startswith("#") or len(text) != 7:
        return text
    channels = [int(text[index:index + 2], 16) for index in (1, 3, 5)]
    darkened = [max(0, min(255, int(round(channel * factor)))) for channel in channels]
    return f"#{darkened[0]:02X}{darkened[1]:02X}{darkened[2]:02X}"


def blend_hex_colors(color: str, target: str, weight: float) -> str:
    if not color.startswith("#") or len(color) != 7 or not target.startswith("#") or len(target) != 7:
        return color
    base_channels = [int(color[index:index + 2], 16) for index in (1, 3, 5)]
    target_channels = [int(target[index:index + 2], 16) for index in (1, 3, 5)]
    blended = [
        max(0, min(255, int(round((1.0 - weight) * base + weight * target_channel))))
        for base, target_channel in zip(base_channels, target_channels)
    ]
    return f"#{blended[0]:02X}{blended[1]:02X}{blended[2]:02X}"


def color_for_series(series_key: str) -> str:
    if series_key in SUMMARY_BASE_MODEL_COLOR_MAP:
        base_color = SUMMARY_BASE_MODEL_COLOR_MAP[series_key]
    else:
        base_color = post_training_color_for_series(series_key)
    if series_key == "Llama-3.1-8B-Instruct":
        return "#A0CBE8"
    if series_key == "Llama-3.1-70B-Instruct":
        return "#4E79A7"
    return base_color


def is_emphasized_model_series(series_key: str, model_dir: str, display: str) -> bool:
    lower = " ".join([str(series_key), str(model_dir), str(display)]).lower()
    return "llama" in lower and "sft" in lower and "tulu" not in lower


def is_llama_instruct_series(series_key: str, model_dir: str) -> bool:
    return series_key == "Llama-3.1-8B-Instruct" or model_dir == "Llama-3.1-8B-Instruct"


def is_base_model_series(series_key: str, model_dir: str) -> bool:
    return any(str(value).strip() in BASE_MODEL_V3_LLM_ORDER for value in (series_key, model_dir))


def is_degroot_fit_series(series_key: str, model_dir: str, display: str) -> bool:
    return any(str(value).strip() == DEGROOT_FIT_MODEL_NAME for value in (series_key, model_dir, display))


def series_line_style(
    series_key: str,
    model_dir: str = "",
    display: str = "",
    *,
    is_human: bool = False,
) -> dict[str, float]:
    if is_human:
        return {"linewidth": 1.9, "markersize": 4.6, "elinewidth": 1.0, "zorder": 6.0, "alpha": 1.0, "linestyle": "-"}
    if is_llama_instruct_series(series_key, model_dir):
        return {"linewidth": 1.9, "markersize": 4.4, "elinewidth": 1.0, "zorder": 5.7, "alpha": 1.0, "linestyle": "-"}
    if is_emphasized_model_series(series_key, model_dir, display):
        return {"linewidth": 1.9, "markersize": 4.4, "elinewidth": 1.0, "zorder": 5.5, "alpha": 1.0, "linestyle": "-"}
    if is_degroot_fit_series(series_key, model_dir, display):
        return {"linewidth": 1.2, "markersize": 3.8, "elinewidth": 0.8, "zorder": 4.2, "alpha": 1.0, "linestyle": "--"}
    if is_base_model_series(series_key, model_dir):
        return {"linewidth": 1.05, "markersize": 3.7, "elinewidth": 0.8, "zorder": 4.0, "alpha": 1.0, "linestyle": "--"}
    return {"linewidth": 1.05, "markersize": 3.7, "elinewidth": 0.8, "zorder": 4.0, "alpha": 1.0, "linestyle": "-"}


def corresponding_abm_line_report_csv_path(
    collection_type: str,
    comparison_set: str,
    split: str | None,
    partition: str,
    report_version: str,
) -> Path | None:
    if comparison_set != "post-training" or collection_type != "breadth" or split is None:
        return None
    abm_output_dir = SEAN_REPRODUCE_ROOT / report_version_namespace(report_version) / "abm" / "line_plot" / "post_train"
    return default_report_path(
        abm_output_dir,
        collection_type,
        comparison_set,
        split,
        partition,
        report_version,
    ).with_suffix(".csv")


def load_degroot_fit_overall_series(
    collection_type: str,
    comparison_set: str,
    split: str | None,
    partition: str,
    report_version: str,
) -> dict[str, dict[str, object]]:
    report_csv_path = corresponding_abm_line_report_csv_path(
        collection_type,
        comparison_set,
        split,
        partition,
        report_version,
    )
    if report_csv_path is None or not report_csv_path.exists():
        return {}

    page_payloads: dict[str, dict[str, object]] = {}
    with report_csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("topic_key", "")).strip() != OVERALL_TOPIC_KEY:
                continue
            if str(row.get("series_key", "")).strip() != DEGROOT_FIT_MODEL_NAME:
                continue
            page_title = str(row.get("page_title", "")).strip()
            if not page_title:
                continue
            payload = page_payloads.setdefault(
                page_title,
                {
                    "series_key": DEGROOT_FIT_MODEL_NAME,
                    "display": str(row.get("series_display", DEGROOT_FIT_MODEL_NAME)).strip() or DEGROOT_FIT_MODEL_NAME,
                    "model_dir": str(row.get("model_dir", DEGROOT_FIT_MODEL_NAME)).strip() or DEGROOT_FIT_MODEL_NAME,
                    "color": color_for_series(DEGROOT_FIT_MODEL_NAME),
                    "count": 0,
                    "line_points": [],
                    "change_value": float("nan"),
                    "change_se": float("nan"),
                },
            )
            row_type = str(row.get("row_type", "")).strip()
            if row_type == "line_point":
                event_index = parse_round(row.get("event_index"))
                plotted_value = parse_numeric(row.get("plotted_value"))
                plotted_se = parse_numeric(row.get("plotted_se"))
                if event_index is None or plotted_value is None:
                    continue
                payload["line_points"].append(
                    {
                        "event_index": int(event_index),
                        "value": float(plotted_value),
                        "error": float("nan") if plotted_se is None else float(plotted_se),
                    }
                )
                count_value = parse_round(row.get("n_complete_groups"))
                if count_value is not None:
                    payload["count"] = max(int(payload["count"]), int(count_value))
            elif row_type == "overall_change":
                change_value = parse_numeric(row.get("change_value"))
                change_se = parse_numeric(row.get("change_se"))
                if change_value is not None:
                    payload["change_value"] = float(change_value)
                if change_se is not None:
                    payload["change_se"] = float(change_se)
                count_value = parse_round(row.get("n_complete_groups"))
                if count_value is not None:
                    payload["count"] = max(int(payload["count"]), int(count_value))

    resolved_payloads: dict[str, dict[str, object]] = {}
    for page_title, payload in page_payloads.items():
        line_points = sorted(payload.pop("line_points", []), key=lambda point: int(point["event_index"]))
        if not line_points:
            continue
        resolved_payloads[page_title] = {
            **payload,
            "values": [float(point["value"]) for point in line_points],
            "errors": [float(point["error"]) for point in line_points],
            "group_store": {},
        }
    return resolved_payloads


def append_external_overall_series(
    model_series: list[dict[str, object]],
    page_title: str,
    external_series_by_page: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    external_series = external_series_by_page.get(page_title)
    if external_series is None:
        return model_series
    existing_keys = {str(model.get("series_key", "")) for model in model_series}
    if str(external_series.get("series_key", "")) in existing_keys:
        return model_series
    return [
        *model_series,
        {
            **external_series,
            "values": list(external_series.get("values", [])),
            "errors": list(external_series.get("errors", [])),
        },
    ]


def mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.asarray(values, dtype=float).mean())


def bootstrap_standard_error(values: list[float]) -> float:
    if len(values) <= 1:
        return float("nan")
    return float(np.asarray(values, dtype=float).std(ddof=1))


def safe_pstdev(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    return float(np.asarray(values, dtype=float).std(ddof=0))


def init_metric_store() -> dict[str, dict[str, dict[str, dict[str, dict[str, float]]]]]:
    return {
        "group_mean": {},
        "group_sd": {},
    }


def append_metric(
    store: dict[str, dict[str, dict[str, dict[str, dict[str, float]]]]],
    metric_name: str,
    topic_key: str,
    series_key: str,
    group_key: str,
    position_key: str,
    value: float,
) -> None:
    topic_store = store[metric_name].setdefault(topic_key, {})
    series_store = topic_store.setdefault(series_key, {})
    group_store = series_store.setdefault(group_key, {})
    group_store[position_key] = float(value)


def extract_complete_groups(
    group_store: dict[str, dict[str, float]],
    position_keys: list[str],
) -> dict[str, dict[str, float]]:
    return {
        group_key: value_map
        for group_key, value_map in group_store.items()
        if all(position_key in value_map for position_key in position_keys)
    }


def prefix_from_group_key(group_key: str) -> str:
    return str(group_key).split("::", 1)[0]


def build_prefix_position_values(
    complete_groups: dict[str, dict[str, float]],
    position_key: str,
) -> dict[str, list[float]]:
    prefix_values: dict[str, list[float]] = {}
    for group_key, value_map in complete_groups.items():
        value = value_map.get(position_key)
        if value is None:
            continue
        prefix_values.setdefault(prefix_from_group_key(group_key), []).append(float(value))
    return prefix_values


def row_position_key(row: dict[str, str]) -> str | None:
    event_type = str(row.get("event_type") or "").strip()
    if event_type == "Initial Opinion":
        return "initial_opinion"
    if event_type == "Post Opinion":
        return "post_opinion"
    if event_type != "tweet":
        return None
    round_idx = parse_round(row.get("chat_round_order"))
    if round_idx in (1, 2, 3):
        return f"tweet_{round_idx}"
    return None


def summarize_complete_group_metrics(
    rows: list[dict[str, str]],
    *,
    data_prefix: str,
) -> dict[str, dict[str, dict[str, tuple[float, float]]]]:
    grouped_values: dict[str, dict[str, dict[str, list[float]]]] = {}

    for row in rows:
        position_key = row_position_key(row)
        if position_key is None:
            continue
        participant = participant_key(row)
        if not participant:
            continue
        value = parse_numeric(row.get("likert_pred"))
        if value is None:
            continue
        group_id = str(row.get("group_id") or "").strip() or data_prefix
        grouped_values.setdefault(group_id, {}).setdefault(position_key, {}).setdefault(participant, []).append(value)

    summaries = {"public": {}, "private": {}}
    family_specs = (("public", PUBLIC_POSITION_KEYS), ("private", PRIVATE_POSITION_KEYS))
    for group_id, position_store in grouped_values.items():
        for family_name, required_positions in family_specs:
            if any(position_key not in position_store for position_key in required_positions):
                continue
            eligible_participants = sorted(
                set.intersection(*(set(position_store[position_key].keys()) for position_key in required_positions))
            )
            if len(eligible_participants) != REQUIRED_COMPLETE_PARTICIPANTS:
                continue

            group_key = data_prefix if group_id == data_prefix else f"{data_prefix}::{group_id}"
            position_summaries: dict[str, tuple[float, float]] = {}
            for position_key in required_positions:
                participant_values = [
                    float(np.asarray(position_store[position_key][participant], dtype=float).mean())
                    for participant in eligible_participants
                ]
                summary = summarize_group_values(participant_values)
                if summary is None:
                    position_summaries = {}
                    break
                position_summaries[position_key] = summary
            if position_summaries:
                summaries[family_name][group_key] = position_summaries

    return summaries


def extract_event_values(
    df: pd.DataFrame,
    event_type: str,
    round_value: float | None,
) -> list[float]:
    out = df.copy()
    out["event_type"] = out["event_type"].astype(str)
    out["chat_round_order"] = pd.to_numeric(out.get("chat_round_order"), errors="coerce")
    out["likert_pred"] = pd.to_numeric(out.get("likert_pred"), errors="coerce")

    mask = out["event_type"] == event_type
    if round_value is not None:
        mask &= out["chat_round_order"] == float(round_value)

    out = out.loc[mask].copy()
    out = out.dropna(subset=["likert_pred"])

    dedupe_col = None
    for candidate in ("worker_id", "empirica_id", "pseudo_id"):
        if candidate in out.columns:
            dedupe_col = candidate
            break
    if dedupe_col is not None:
        out = out.drop_duplicates(subset=[dedupe_col], keep="first")

    return out["likert_pred"].astype(float).tolist()


def summarize_group_values(values: list[float]) -> tuple[float, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=0))


def load_eval_df(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Eval CSV not found: {csv_path}")
    return load_eval_dataframe(
        csv_path,
        respect_validity_labels=True,
        apply_rejudged_sidecars=True,
        likert_col="likert_pred",
    )


def load_eval_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Eval CSV not found: {csv_path}")
    return read_eval_csv_rows(
        csv_path,
        respect_validity_labels=True,
        apply_rejudged_sidecars=True,
        likert_col="likert_pred",
    )


def sidecar_path_for_csv(csv_path: Path, sidecar_root: Path, suffix: str) -> Path | None:
    try:
        relative_path = csv_path.relative_to(result_root())
    except ValueError:
        return None
    sidecar_path = sidecar_root / relative_path
    return sidecar_path.with_name(sidecar_path.stem + suffix)


def parse_numeric(value: str | float | int | None) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_round(value: str | float | int | None) -> int | None:
    numeric = parse_numeric(value)
    if numeric is None:
        return None
    return int(numeric)


def participant_key(row: dict[str, str]) -> str | None:
    for key in ("empirica_id", "worker_id", "sender_id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return None


def step_toward(current: float, target: float) -> float:
    if target > current:
        return min(6.0, current + 1.0)
    if target < current:
        return max(1.0, current - 1.0)
    return current


def compute_sd_baseline_targets(rows: list[dict[str, str]], data_prefix: str) -> dict[str, dict[str, float]]:
    round1_public: dict[str, dict[str, float]] = {}
    initial_private: dict[str, dict[str, float]] = {}
    first_partner: dict[str, dict[str, str]] = {}

    def partner_key(row: dict[str, str]) -> str | None:
        for key in ("sender_id", "worker_id", "empirica_id"):
            value = str(row.get(key) or "").strip()
            if value:
                return value
        return None

    for row in rows:
        pid = partner_key(row)
        if not pid:
            continue

        gid = str(row.get("group_id") or "").strip() or data_prefix
        event_type = str(row.get("event_type") or "").strip()
        stance = parse_numeric(row.get("likert_pred"))

        if event_type == "Initial Opinion":
            initial_stance = parse_numeric(row.get("likert_pred"))
            if initial_stance is None:
                initial_stance = parse_numeric(row.get("sliderValue"))
            if initial_stance is not None:
                initial_private.setdefault(gid, {})[pid] = initial_stance
            continue

        if event_type != "tweet":
            continue

        round_idx = parse_round(row.get("chat_round_order"))
        if round_idx != 1 or stance is None:
            continue

        round1_public.setdefault(gid, {})[pid] = stance
        recipient = str(row.get("recipient_id") or "").strip()
        if recipient:
            first_partner.setdefault(gid, {})[pid] = recipient

    tweet_targets = {label: [] for label, _ in REFERENCE_LINE_SPECS}
    opinion_targets = {label: [] for label, _ in REFERENCE_LINE_SPECS}

    all_group_ids = sorted(set(round1_public) | set(initial_private) | set(first_partner))
    for gid in all_group_ids:
        public_init = round1_public.get(gid, {})
        private_init = initial_private.get(gid, {})
        partners = first_partner.get(gid, {})

        private_participants = [
            pid for pid in sorted(private_init)
            if pid in partners and partners[pid] in private_init
        ]
        if len(private_participants) >= 2:
            private_init_vals = [private_init[pid] for pid in private_participants]
            shifted_private_vals = [
                step_toward(private_init[pid], private_init[partners[pid]])
                for pid in private_participants
            ]
            one_step_private = safe_pstdev(shifted_private_vals)
            if not np.isnan(one_step_private):
                opinion_targets[REFERENCE_LINE_SPECS[0][0]].append(one_step_private)
            half_private = 0.5 * safe_pstdev(private_init_vals)
            if not np.isnan(half_private):
                opinion_targets[REFERENCE_LINE_SPECS[1][0]].append(half_private)

        tweet_participants = [
            pid for pid in sorted(public_init)
            if pid in partners and partners[pid] in public_init
        ]
        if len(tweet_participants) >= 2:
            tweet_init_vals = [public_init[pid] for pid in tweet_participants]
            shifted_public_vals = [
                step_toward(public_init[pid], public_init[partners[pid]])
                for pid in tweet_participants
            ]
            one_step_public = safe_pstdev(shifted_public_vals)
            if not np.isnan(one_step_public):
                tweet_targets[REFERENCE_LINE_SPECS[0][0]].append(one_step_public)
            half_public = 0.5 * safe_pstdev(tweet_init_vals)
            if not np.isnan(half_public):
                tweet_targets[REFERENCE_LINE_SPECS[1][0]].append(half_public)

    return {
        "tweet": {label: mean_or_nan(values) for label, values in tweet_targets.items()},
        "opinion": {label: mean_or_nan(values) for label, values in opinion_targets.items()},
    }


def collect_plot_inputs(
    collection_type: str,
    *,
    split: str | None,
    partition: str,
    series_specs: list[dict[str, str]],
    apply_rejudged_sidecars: bool,
) -> tuple[
    list[str],
    dict[str, str],
    dict[str, int],
    dict[str, dict[str, dict[str, dict[str, dict[str, float]]]]],
    dict[str, dict[str, dict[str, float]]],
    dict[str, bool],
]:
    golden_prefixes = selected_prefixes(collection_type, split, partition)
    metric_store = init_metric_store()
    topic_display_map: dict[str, str] = {}
    topic_counts: Counter[str] = Counter()
    baseline_store: dict[str, dict[str, dict[str, list[float]]]] = {
        OVERALL_TOPIC_KEY: {
            "tweet": {label: [] for label, _ in REFERENCE_LINE_SPECS},
            "opinion": {label: [] for label, _ in REFERENCE_LINE_SPECS},
        }
    }
    report_metadata = {
        "initial_post_rejudged_active": False,
        "tweet_rejudged_active": False,
    }

    for data_prefix in golden_prefixes:
        raw_topic = get_topic(data_prefix)
        topic_key = normalize_topic(raw_topic)
        topic_display_map.setdefault(topic_key, display_topic(raw_topic))
        human_csv_path = result_root() / data_prefix / HUMAN_SERIES["model_dir"] / HUMAN_CSV_FILENAME
        if not human_csv_path.exists():
            continue

        topic_counts[topic_key] += 1
        baseline_store.setdefault(
            topic_key,
            {
                "tweet": {label: [] for label, _ in REFERENCE_LINE_SPECS},
                "opinion": {label: [] for label, _ in REFERENCE_LINE_SPECS},
            },
        )

        human_rows = read_eval_csv_rows(
            human_csv_path,
            respect_validity_labels=True,
            apply_rejudged_sidecars=apply_rejudged_sidecars,
            likert_col="likert_pred",
        )
        baseline_targets = compute_sd_baseline_targets(human_rows, data_prefix)
        for axis_key in ("tweet", "opinion"):
            for label, value in baseline_targets[axis_key].items():
                if np.isnan(value):
                    continue
                baseline_store[topic_key][axis_key][label].append(value)
                baseline_store[OVERALL_TOPIC_KEY][axis_key][label].append(value)

        for series_spec in series_specs:
            series_key = series_spec["label"]
            model_dir = series_spec["model_dir"]
            csv_name = series_spec["csv_name"]
            csv_path = result_root() / data_prefix / model_dir / csv_name
            if not csv_path.exists():
                continue
            if apply_rejudged_sidecars and not report_metadata["initial_post_rejudged_active"]:
                initial_post_sidecar = sidecar_path_for_csv(csv_path, initial_post_rejudged_root(), INITIAL_POST_REJUDGE_SUFFIX)
                report_metadata["initial_post_rejudged_active"] = bool(initial_post_sidecar and initial_post_sidecar.exists())
            if apply_rejudged_sidecars and not report_metadata["tweet_rejudged_active"]:
                tweet_sidecar = sidecar_path_for_csv(csv_path, tweet_rejudged_root(), TWEET_REJUDGE_SUFFIX)
                report_metadata["tweet_rejudged_active"] = bool(tweet_sidecar and tweet_sidecar.exists())
            eval_rows = read_eval_csv_rows(
                csv_path,
                respect_validity_labels=True,
                apply_rejudged_sidecars=apply_rejudged_sidecars,
                likert_col="likert_pred",
            )
            family_summaries = summarize_complete_group_metrics(eval_rows, data_prefix=data_prefix)
            for family_summary in family_summaries.values():
                for group_key, position_summaries in family_summary.items():
                    for position_key, (mean_value, sd_value) in position_summaries.items():
                        append_metric(metric_store, "group_mean", topic_key, series_key, group_key, position_key, mean_value)
                        append_metric(metric_store, "group_sd", topic_key, series_key, group_key, position_key, sd_value)
                        append_metric(metric_store, "group_mean", OVERALL_TOPIC_KEY, series_key, group_key, position_key, mean_value)
                        append_metric(metric_store, "group_sd", OVERALL_TOPIC_KEY, series_key, group_key, position_key, sd_value)

    summarized_baselines = {
        topic_key: {
            axis_key: {label: mean_or_nan(values) for label, values in axis_store.items()}
            for axis_key, axis_store in topic_store.items()
        }
        for topic_key, topic_store in baseline_store.items()
    }

    return golden_prefixes, topic_display_map, dict(topic_counts), metric_store, summarized_baselines, report_metadata


def summarize_across_groups(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    mean_value = float(array.mean())
    if array.size <= 1:
        return mean_value, 0.0
    se_value = float(array.std(ddof=1) / np.sqrt(array.size))
    return mean_value, se_value


def build_series_stats(
    series_store: dict[str, dict[str, dict[str, float]]],
    series_key: str,
    position_keys: list[str],
) -> dict[str, object]:
    group_store = series_store.get(series_key, {})
    complete_groups = extract_complete_groups(group_store, position_keys)

    if not complete_groups:
        return {
            "values": [float("nan")] * len(position_keys),
            "errors": [float("nan")] * len(position_keys),
            "count": 0,
        }

    values: list[float] = []
    errors: list[float] = []
    for position_key in position_keys:
        position_values = [value_map[position_key] for value_map in complete_groups.values()]
        mean_value, se_value = summarize_across_groups(position_values)
        values.append(mean_value)
        errors.append(se_value)
    return {"values": values, "errors": errors, "count": len(complete_groups)}


def build_panel_count_note(
    human_series: dict[str, object],
    model_series: list[dict[str, object]],
    wrap_width: int = 54,
) -> str:
    line_counts = [f"Human={human_series['count']}"]
    line_counts.extend(f"{model['display']}={model['count']}" for model in model_series)
    return wrap_text("Line n: " + " | ".join(line_counts), wrap_width)


def append_opacity_footnote(base_footnote: str | None) -> str:
    opacity_note = (
        "Opacity: highlighted lines alpha=1.00; non-highlight lines alpha=1.00; "
        "baseline markers alpha=1.00"
    )
    if base_footnote:
        return f"{base_footnote} | {opacity_note}"
    return opacity_note


def apply_shared_human_anchors(
    human_series: dict[str, object],
    model_series: list[dict[str, object]],
    position_keys: list[str],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    anchor_indices = [
        index for index, position_key in enumerate(position_keys)
        if position_key in SHARED_HUMAN_ANCHOR_KEYS
    ]

    anchored_human = {
        **human_series,
        "values": list(human_series["values"]),
        "errors": list(human_series["errors"]),
    }
    anchored_models: list[dict[str, object]] = []
    for model in model_series:
        anchored_model = {
            **model,
            "values": list(model["values"]),
            "errors": list(model["errors"]),
        }
        for index in anchor_indices:
            if index >= len(anchored_model["values"]) or index >= len(anchored_human["values"]):
                continue
            anchored_model["values"][index] = anchored_human["values"][index]
            anchored_model["errors"][index] = anchored_human["errors"][index]
        anchored_models.append(anchored_model)
    return anchored_human, anchored_models


def series_change_value(series: dict[str, object]) -> float:
    precomputed_change = parse_numeric(series.get("change_value"))
    if precomputed_change is not None:
        return float(precomputed_change)
    values = list(series.get("values", []))
    if len(values) < 2:
        return float("nan")
    start_value = float(values[0])
    end_value = float(values[-1])
    if np.isnan(start_value) or np.isnan(end_value):
        return float("nan")
    return end_value - start_value


def series_change_standard_error_for_export(
    series: dict[str, object],
    position_keys: list[str],
    anchor_series: dict[str, object],
) -> float:
    precomputed_change_se = parse_numeric(series.get("change_se"))
    if precomputed_change_se is not None:
        return float(precomputed_change_se)
    return bootstrap_change_standard_error_for_export(series, position_keys, anchor_series)


def build_model_id_note(model_series: list[dict[str, object]], wrap_width: int = 120) -> str:
    model_ids = " | ".join(f"{model['display']}={model['model_dir']}" for model in model_series)
    return wrap_text(f"Model IDs: {model_ids}", wrap_width)


def event_label_for_key(event_key: str) -> str:
    for candidate_key, label, _event_type, _round_value in COMBINED_EVENTS:
        if candidate_key == event_key:
            return label
    return str(event_key)


def build_page_group_specs(
    comparison_set: str,
    collection_type: str,
    series_specs: list[dict[str, str]],
) -> list[dict[str, object]]:
    post_train_specs = series_specs[1:]
    if comparison_set == "post-training" and collection_type == "depth":
        gpt_sft_models = {GPT_DEPTH_TOPIC_SFT_MODEL, GPT_DEPTH_GROUP_SFT_MODEL}
        llama_specs = [
            spec
            for spec in post_train_specs
            if spec["model_dir"] not in {GPT_4O_MINI_BASE_MODEL, *gpt_sft_models}
        ]
        gpt_specs = [
            spec
            for spec in post_train_specs
            if spec["model_dir"] in {GPT_4O_MINI_BASE_MODEL, *gpt_sft_models}
        ]
        has_llama_dpo = any("dpo" in str(spec["model_dir"]).lower() for spec in llama_specs)
        return [
            {
                "title_suffix": "LLaMA Base vs SFT vs DPO" if has_llama_dpo else "LLaMA Base vs SFT",
                "allowed_model_dirs": [spec["model_dir"] for spec in llama_specs],
                "footnote": build_model_id_note(llama_specs),
            },
            {
                "title_suffix": "gpt-4o-mini Base vs SFT",
                "allowed_model_dirs": [spec["model_dir"] for spec in gpt_specs],
                "footnote": build_model_id_note(gpt_specs),
            },
        ]
    return [{"title_suffix": "", "allowed_model_dirs": None, "footnote": None}]


def build_reference_lines(
    metric_kind: str,
    x_labels: list[str],
    topic_key: str,
    baseline_targets: dict[str, dict[str, float]] | None,
    human_series: dict[str, object],
) -> list[dict[str, object]]:
    if metric_kind != "sd":
        return []

    baseline_targets = baseline_targets or {}
    human_values = list(human_series.get("values", []))
    start_value = float(human_values[0]) if human_values else float("nan")
    reference_lines = []
    for label, color in REFERENCE_LINE_SPECS:
        target_value = baseline_targets.get(topic_key, {}).get(label, float("nan"))
        if np.isnan(start_value) or np.isnan(target_value):
            line_x_positions = np.asarray([float(max(0, len(x_labels) - 1)) + REFERENCE_MARKER_X_OFFSET], dtype=float)
            series_values = [float("nan")] * len(line_x_positions)
        else:
            line_x_positions = np.asarray([float(max(0, len(x_labels) - 1)) + REFERENCE_MARKER_X_OFFSET], dtype=float)
            series_values = [float(target_value)]
        reference_lines.append(
            {
                "label": label,
                "color": color,
                "values": series_values,
                "x_positions": line_x_positions,
                "marker_only": True,
            }
        )
    return reference_lines


def create_topic_plot(
    ax: plt.Axes,
    title: str,
    y_label: str,
    x_labels: list[str],
    human_series: dict[str, object],
    model_series: list[dict[str, object]],
    metric_kind: str,
    show_ylabel: bool,
    show_legend: bool,
    reference_lines: list[dict[str, object]] | None = None,
    panel_note: str | None = None,
    title_wrap_width: int = 52,
    show_y_grid: bool = True,
) -> None:
    x_positions = np.arange(len(x_labels), dtype=float)
    reference_lines = reference_lines or []
    human_style = series_line_style(HUMAN_SERIES["label"], HUMAN_SERIES["model_dir"], HUMAN_SERIES["display"], is_human=True)
    ordered_model_series = sorted(
        model_series,
        key=lambda model: (
            series_line_style(
                str(model.get("series_key", "")),
                str(model.get("model_dir", "")),
                str(model.get("display", "")),
            )["zorder"],
            str(model.get("display", "")),
        ),
    )

    human_label = f"{HUMAN_SERIES['display']} (n={human_series['count']})"
    ax.errorbar(
        x_positions,
        human_series["values"],
        yerr=human_series["errors"],
        marker="o",
        linestyle=human_style["linestyle"],
        linewidth=human_style["linewidth"],
        markersize=human_style["markersize"],
        elinewidth=human_style["elinewidth"],
        capsize=2.5,
        color=color_for_series(HUMAN_SERIES["label"]),
        label=human_label,
        alpha=human_style["alpha"],
        zorder=human_style["zorder"],
    )

    for model in ordered_model_series:
        model_style = series_line_style(
            str(model.get("series_key", "")),
            str(model.get("model_dir", "")),
            str(model.get("display", "")),
        )
        label = f"{model['display']} (n={model['count']})"
        ax.errorbar(
            x_positions,
            model["values"],
            yerr=model["errors"],
            marker="o",
            linestyle=model_style["linestyle"],
            linewidth=model_style["linewidth"],
            markersize=model_style["markersize"],
            elinewidth=model_style["elinewidth"],
            capsize=2.0,
            color=model["color"],
            label=label,
            alpha=model_style["alpha"],
            zorder=model_style["zorder"],
        )

    for reference_line in reference_lines:
        series_values = reference_line.get("values")
        if not isinstance(series_values, list) or not any(not np.isnan(value) for value in series_values):
            continue
        line_x_positions = reference_line.get("x_positions", x_positions)
        if bool(reference_line.get("marker_only")):
            plotted_lines = ax.plot(
                line_x_positions,
                series_values,
                color=str(reference_line["color"]),
                linestyle="None",
                marker=REFERENCE_MARKER,
                markersize=REFERENCE_MARKER_SIZE,
                markeredgecolor="white",
                markeredgewidth=REFERENCE_MARKER_EDGE_WIDTH,
                alpha=REFERENCE_LINE_ALPHA,
                zorder=1,
            )
        else:
            plotted_lines = ax.plot(
                line_x_positions,
                series_values,
                color=str(reference_line["color"]),
                linewidth=REFERENCE_LINE_WIDTH,
                linestyle=REFERENCE_LINE_DASH_PATTERN,
                alpha=REFERENCE_LINE_ALPHA,
                zorder=1,
            )
        plotted_lines[0].set_path_effects(REFERENCE_LINE_PATH_EFFECTS)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=20)
    ax.set_xlim(-0.2, float(max(0, len(x_labels) - 1)) + REFERENCE_MARKER_X_OFFSET + 0.10)
    ax.set_title(wrap_text(title, title_wrap_width), fontsize=10, pad=8)
    if show_ylabel:
        ax.set_ylabel(y_label)
    if show_y_grid:
        ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.5)

    upper_values: list[float] = []
    lower_values: list[float] = []
    for series in [human_series, *ordered_model_series]:
        for value, error in zip(series["values"], series["errors"]):
            if np.isnan(value):
                continue
            error_value = float(0.0 if np.isnan(error) else error)
            upper_values.append(float(value) + error_value)
            lower_values.append(float(value) - error_value)
    for reference_line in reference_lines:
        series_values = reference_line.get("values")
        if isinstance(series_values, list):
            for value in series_values:
                if not np.isnan(value):
                    upper_values.append(float(value))
                    lower_values.append(float(value))

    if metric_kind == "mean":
        if upper_values and lower_values:
            lower = min(lower_values)
            upper = max(upper_values)
            spread = upper - lower
            lower_padding = max(0.06, spread * 0.12 if spread > 0 else 0.12)
            upper_padding = max(0.08, spread * 0.12 if spread > 0 else 0.14)
            ax.set_ylim(max(1.0, lower - lower_padding), min(6.0, upper + upper_padding))
        else:
            ax.set_ylim(1.0, 6.0)
    else:
        if upper_values and lower_values:
            lower = min(lower_values)
            upper = max(upper_values)
            spread = upper - lower
            lower_padding = max(0.025, spread * 0.12 if spread > 0 else 0.05)
            upper_padding = max(0.04, spread * 0.10 if spread > 0 else 0.06)
            ax.set_ylim(max(0.0, lower - lower_padding), upper + upper_padding)
        else:
            ax.set_ylim(0.0, 0.5)

    if show_legend:
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, ncol=1, fontsize=9)

    if panel_note:
        ax.text(
            0.015,
            0.02,
            panel_note,
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=6.3,
            color="#444444",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.2},
        )


def create_topic_png(
    output_path: Path,
    title: str,
    y_label: str,
    position_keys: list[str],
    x_labels: list[str],
    human_series: dict[str, object],
    model_series: list[dict[str, object]],
    metric_kind: str,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    anchored_human, anchored_models = apply_shared_human_anchors(human_series, model_series, position_keys)
    create_topic_plot(
        ax=ax,
        title=title,
        y_label=y_label,
        x_labels=x_labels,
        human_series=anchored_human,
        model_series=anchored_models,
        metric_kind=metric_kind,
        show_ylabel=True,
        show_legend=True,
        title_wrap_width=72,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def add_topic_count_page(
    pdf: PdfPages,
    collection_type: str,
    comparison_set: str,
    split: str | None,
    partition: str,
    report_version: str,
    apply_rejudged_sidecars: bool,
    total_groups: int,
    ordered_topic_keys: list[str],
    topic_display_map: dict[str, str],
    topic_counts: dict[str, int],
    report_metadata: dict[str, bool],
) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")

    label = collection_label(collection_type)
    mask_path = selected_mask_path(collection_type, split, partition)
    eval_root_path = result_root()
    comparison_name = comparison_label(comparison_set)
    lines = [
        f"{report_version_label(report_version)} {selection_display_label(collection_type, split, partition)} {comparison_name} Line Plot Report",
        "",
        f"Transition simulation version: {transition_simulation_version()}",
        f"Comparison set: {comparison_name}",
        f"Eval tree: {display_path(eval_root_path)}",
        f"LLM CSV filename: {LLM_CSV_FILENAME}",
        f"Human CSV filename: {HUMAN_CSV_FILENAME}",
        f"Selected mask: {display_path(mask_path)}",
        (
            "Validity labels: inline validity columns plus rejudged sidecar overrides when available"
            if apply_rejudged_sidecars
            else "Validity labels: inline validity columns from eval CSVs only"
        ),
        "",
        f"Selected golden groups: {total_groups}",
        "Topic counts:",
    ]
    if apply_rejudged_sidecars and report_metadata.get("initial_post_rejudged_active"):
        lines.insert(8, f"Rejudged initial/post sidecars: {display_path(initial_post_rejudged_root())}")
    if apply_rejudged_sidecars and report_metadata.get("tweet_rejudged_active"):
        insert_at = 9 if report_metadata.get("initial_post_rejudged_active") else 8
        lines.insert(insert_at, f"Rejudged tweet sidecars: {display_path(tweet_rejudged_root())}")
    for topic_key in ordered_topic_keys:
        lines.append(f"- {topic_display_label(topic_key, topic_display_map)}: {topic_counts.get(topic_key, 0)}")

    y_position = 0.96
    for index, line in enumerate(lines):
        wrapped_line = wrap_text(line, 76) if line.startswith("-") or ": " in line else line
        if index == 0:
            fontsize = 18
            step = 0.085
        elif line == "":
            fontsize = 12
            step = 0.03
        else:
            fontsize = 10.5
            step = 0.04 + 0.016 * max(0, wrapped_line.count("\n"))
        ax.text(0.04, y_position, wrapped_line, va="top", ha="left", fontsize=fontsize)
        y_position -= step

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_legend_panel(
    ax: plt.Axes,
    position_count: int,
    model_legend_series: list[dict[str, object]],
    reference_lines: list[dict[str, object]] | None = None,
) -> None:
    ax.axis("off")
    reference_lines = reference_lines or []
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=color_for_series(HUMAN_SERIES["label"]),
            marker="o",
            linestyle=series_line_style(HUMAN_SERIES["label"], HUMAN_SERIES["model_dir"], HUMAN_SERIES["display"], is_human=True)["linestyle"],
            linewidth=series_line_style(HUMAN_SERIES["label"], HUMAN_SERIES["model_dir"], HUMAN_SERIES["display"], is_human=True)["linewidth"],
            markersize=series_line_style(HUMAN_SERIES["label"], HUMAN_SERIES["model_dir"], HUMAN_SERIES["display"], is_human=True)["markersize"],
            alpha=series_line_style(HUMAN_SERIES["label"], HUMAN_SERIES["model_dir"], HUMAN_SERIES["display"], is_human=True)["alpha"],
            label=HUMAN_SERIES["display"],
        )
    ]
    for model_spec in model_legend_series:
        model_style = series_line_style(
            str(model_spec.get("series_key", "")),
            str(model_spec.get("model_dir", "")),
            str(model_spec.get("display", "")),
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=str(model_spec["color"]),
                marker="o",
                linestyle=model_style["linestyle"],
                linewidth=model_style["linewidth"],
                markersize=model_style["markersize"],
                alpha=model_style["alpha"],
                label=str(model_spec["display"]),
            )
        )

    for reference_line in reference_lines:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=str(reference_line["color"]),
                linewidth=0,
                marker=REFERENCE_MARKER,
                markersize=REFERENCE_MARKER_SIZE,
                markeredgecolor="white",
                markeredgewidth=REFERENCE_MARKER_EDGE_WIDTH,
                alpha=REFERENCE_LINE_ALPHA,
                label=str(reference_line["label"]),
            )
        )

    model_legend = ax.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=False,
        fontsize=8.0,
        ncol=2,
        columnspacing=1.0,
        handlelength=2.2,
        bbox_to_anchor=(0.0, 1.0),
    )
    ax.add_artist(model_legend)

    ax.text(
        0.02,
        0.36,
        (
            "Page notes:\n"
            f"- Each line uses only groups where the same exact {REQUIRED_COMPLETE_PARTICIPANTS} participants survive across the {position_count} plotted positions.\n"
            "- Model lines are human-anchored at Initial Opinion and Tweet 1.\n"
            "- Each panel shows those per-line counts in a small footnote.\n"
            "- Standalone PNGs are only written when explicitly requested."
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.0,
    )


def add_overall_summary_page(
    pdf: PdfPages,
    page_specs: list[dict[str, object]],
    overall_series_by_page: list[dict[str, dict[str, object]]],
    sd_baseline_targets: dict[str, dict[str, dict[str, float]]],
    title_suffix: str = "",
    page_footnote: str | None = None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes_flat = list(axes.flatten())

    for index, page_spec in enumerate(page_specs):
        ax = axes_flat[index]
        series_payload = overall_series_by_page[index]
        anchored_human, anchored_models = apply_shared_human_anchors(
            series_payload["human"],
            series_payload["models"],
            list(page_spec["event_keys"]),
        )
        x_labels = [label for event_key, label, _, _ in COMBINED_EVENTS if event_key in page_spec["event_keys"]]
        baseline_axis = None
        if page_spec["metric_kind"] == "sd":
            baseline_axis = "tweet" if all(event_key.startswith("tweet") for event_key in page_spec["event_keys"]) else "opinion"
        create_topic_plot(
            ax=ax,
            title=f"Overall\n{page_spec['page_title']}",
            y_label=str(page_spec["y_label"]),
            x_labels=x_labels,
            human_series=anchored_human,
            model_series=anchored_models,
            metric_kind=str(page_spec["metric_kind"]),
            show_ylabel=index % 2 == 0,
            show_legend=False,
            reference_lines=build_reference_lines(
                str(page_spec["metric_kind"]),
                x_labels,
                OVERALL_TOPIC_KEY,
                {
                    topic_key: sd_baseline_targets.get(topic_key, {}).get(baseline_axis, {})
                    for topic_key in [OVERALL_TOPIC_KEY]
                } if baseline_axis is not None else None,
                anchored_human,
            ),
            panel_note=build_panel_count_note(anchored_human, anchored_models, wrap_width=70),
            title_wrap_width=52,
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=color_for_series(HUMAN_SERIES["label"]),
            marker="o",
            linestyle=series_line_style(HUMAN_SERIES["label"], HUMAN_SERIES["model_dir"], HUMAN_SERIES["display"], is_human=True)["linestyle"],
            linewidth=series_line_style(HUMAN_SERIES["label"], HUMAN_SERIES["model_dir"], HUMAN_SERIES["display"], is_human=True)["linewidth"],
            markersize=series_line_style(HUMAN_SERIES["label"], HUMAN_SERIES["model_dir"], HUMAN_SERIES["display"], is_human=True)["markersize"],
            alpha=series_line_style(HUMAN_SERIES["label"], HUMAN_SERIES["model_dir"], HUMAN_SERIES["display"], is_human=True)["alpha"],
            label=HUMAN_SERIES["display"],
        )
    ]
    series_payload = overall_series_by_page[0]
    for model in series_payload["models"]:
        model_style = series_line_style(
            str(model.get("series_key", "")),
            str(model.get("model_dir", "")),
            str(model.get("display", "")),
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=str(model["color"]),
                marker="o",
                linestyle=model_style["linestyle"],
                linewidth=model_style["linewidth"],
                markersize=model_style["markersize"],
                alpha=model_style["alpha"],
                label=str(model["display"]),
            )
        )
    for label, color in REFERENCE_LINE_SPECS:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=0,
                marker=REFERENCE_MARKER,
                markersize=REFERENCE_MARKER_SIZE,
                markeredgecolor="white",
                markeredgewidth=REFERENCE_MARKER_EDGE_WIDTH,
                alpha=REFERENCE_LINE_ALPHA,
                label=label,
            )
        )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.07 if page_footnote else 0.01),
        ncol=4,
        frameon=False,
        fontsize=9,
        columnspacing=1.4,
        handlelength=2.3,
    )
    overall_title = "Overall Panels Across All Topics"
    if title_suffix:
        overall_title = f"{overall_title} - {title_suffix}"
    fig.suptitle(overall_title, fontsize=16, y=0.98)
    if page_footnote:
        fig.text(0.5, 0.015, page_footnote, ha="center", va="bottom", fontsize=7.2)
    fig.tight_layout(rect=[0, 0.11 if page_footnote else 0.06, 1, 0.96])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_overall_delta_bar_page(
    pdf: PdfPages,
    page_specs: list[dict[str, object]],
    overall_series_by_page: list[dict[str, dict[str, object]]],
    sd_baseline_targets: dict[str, dict[str, dict[str, float]]],
    title_suffix: str = "",
    page_footnote: str | None = None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes_flat = list(axes.flatten())
    legend_handles: list[Line2D] = []

    def bootstrap_change_standard_error(
        series: dict[str, object],
        position_keys: list[str],
        anchor_series: dict[str, object],
    ) -> float:
        primary_group_store = series.get("group_store")
        anchor_group_store = anchor_series.get("group_store")
        if not isinstance(primary_group_store, dict) or not isinstance(anchor_group_store, dict):
            return float("nan")

        primary_complete_groups = extract_complete_groups(primary_group_store, position_keys)
        anchor_complete_groups = extract_complete_groups(anchor_group_store, position_keys)
        if not primary_complete_groups and not anchor_complete_groups:
            return float("nan")

        start_key = str(position_keys[0])
        end_key = str(position_keys[-1])
        start_source_groups = anchor_complete_groups if start_key in SHARED_HUMAN_ANCHOR_KEYS else primary_complete_groups
        end_source_groups = anchor_complete_groups if end_key in SHARED_HUMAN_ANCHOR_KEYS else primary_complete_groups

        start_prefix_values = build_prefix_position_values(start_source_groups, start_key)
        end_prefix_values = build_prefix_position_values(end_source_groups, end_key)
        available_prefixes = sorted(set(start_prefix_values) | set(end_prefix_values))
        if not available_prefixes:
            return float("nan")

        rng = np.random.default_rng(BOOTSTRAP_SEED)
        bootstrap_changes: list[float] = []
        for _ in range(BOOTSTRAP_REPS):
            sampled_prefixes = rng.choice(available_prefixes, size=len(available_prefixes), replace=True)
            sampled_start: list[float] = []
            sampled_end: list[float] = []
            for sampled_prefix in sampled_prefixes:
                sampled_start.extend(start_prefix_values.get(str(sampled_prefix), []))
                sampled_end.extend(end_prefix_values.get(str(sampled_prefix), []))
            if not sampled_start or not sampled_end:
                continue
            bootstrap_changes.append(float(np.mean(sampled_end) - np.mean(sampled_start)))

        return bootstrap_standard_error(bootstrap_changes)

    for index, page_spec in enumerate(page_specs):
        ax = axes_flat[index]
        series_payload = overall_series_by_page[index]
        position_keys = list(page_spec["event_keys"])
        anchored_human, anchored_models = apply_shared_human_anchors(
            series_payload["human"],
            series_payload["models"],
            position_keys,
        )
        all_series = [
            {
                "display": HUMAN_SERIES["display"],
                "color": color_for_series(HUMAN_SERIES["label"]),
                "count": int(anchored_human["count"]),
                "change": series_change_value(anchored_human),
                "change_se": series_change_standard_error_for_export(anchored_human, position_keys, anchored_human),
            },
            *[
                {
                    "display": str(model["display"]),
                    "color": str(model["color"]),
                    "count": int(model["count"]),
                    "change": series_change_value(model),
                    "change_se": series_change_standard_error_for_export(model, position_keys, anchored_human),
                }
                for model in anchored_models
            ],
        ]
        positions = np.arange(len(all_series), dtype=float)
        changes = [series_info["change"] for series_info in all_series]
        change_ses = [
            float(series_info["change_se"])
            if not np.isnan(float(series_info["change_se"]))
            else 0.0
            for series_info in all_series
        ]
        bar_heights = [0.0 if np.isnan(value) else value for value in changes]
        bars = ax.bar(
            positions,
            bar_heights,
            color=[series_info["color"] for series_info in all_series],
            edgecolor="white",
            linewidth=0.7,
            width=0.8,
        )
        ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.9)

        reference_lines: list[dict[str, object]] = []
        if page_spec["metric_kind"] == "sd":
            baseline_axis = "tweet" if all(event_key.startswith("tweet") for event_key in position_keys) else "opinion"
            start_value = float(anchored_human["values"][0]) if anchored_human["values"] else float("nan")
            for label, color in REFERENCE_LINE_SPECS:
                target_value = sd_baseline_targets.get(OVERALL_TOPIC_KEY, {}).get(baseline_axis, {}).get(label, float("nan"))
                if np.isnan(start_value) or np.isnan(target_value):
                    continue
                reference_value = float(target_value - start_value)
                reference_lines.append({"label": label, "color": color, "value": reference_value})
                baseline_line = ax.axhline(
                    reference_value,
                    color=color,
                    linewidth=REFERENCE_LINE_WIDTH,
                    linestyle=REFERENCE_LINE_DASH_PATTERN,
                    alpha=REFERENCE_LINE_ALPHA,
                    zorder=1,
                )
                baseline_line.set_path_effects(REFERENCE_LINE_PATH_EFFECTS)

        ax.grid(True, axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.set_xticks(positions)
        ax.set_xticklabels([series_info["display"] for series_info in all_series], rotation=28, ha="right")
        ax.set_ylabel("End - start")
        ax.set_title(wrap_text(f"Overall Change\n{page_spec['page_title']}", 54), fontsize=10, pad=8)

        finite_changes = [value for value in changes if not np.isnan(value)]
        finite_error_bounds = [
            value - error
            for value, error in zip(changes, change_ses)
            if not np.isnan(value)
        ] + [
            value + error
            for value, error in zip(changes, change_ses)
            if not np.isnan(value)
        ]
        finite_reference_values = [
            float(reference_line["value"])
            for reference_line in reference_lines
            if not np.isnan(float(reference_line["value"]))
        ]
        all_finite_values = finite_changes + finite_error_bounds + finite_reference_values
        if all_finite_values:
            lower = min(all_finite_values)
            upper = max(all_finite_values)
        else:
            lower, upper = -0.1, 0.1
        spread = upper - lower
        padding = max(0.08, spread * 0.2 if spread > 0 else 0.12)
        y_bottom = lower - padding
        y_top = upper + padding
        ax.set_ylim(y_bottom, y_top)
        count_y = y_bottom + max(0.01, (y_top - y_bottom) * 0.03)
        label_offset = max(0.03, (y_top - y_bottom) * 0.035)

        for bar, series_info in zip(bars, all_series):
            error_size = float(series_info["change_se"])
            if np.isnan(error_size):
                error_size = 0.0
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                count_y,
                f"n={series_info['count']}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=6.5,
                color="#666666",
            )
            if np.isnan(series_info["change"]):
                continue
            ax.errorbar(
                [bar.get_x() + bar.get_width() / 2],
                [float(series_info["change"])],
                yerr=[[error_size], [error_size]],
                fmt="none",
                ecolor="#2F2F2F",
                elinewidth=1.4,
                capsize=5.0,
                capthick=2.0,
                zorder=5,
            )
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                float(series_info["change"]) + (label_offset if float(series_info["change"]) >= 0 else -label_offset),
                f"{float(series_info['change']):.2f}",
                ha="center",
                va="bottom" if float(series_info["change"]) >= 0 else "top",
                fontsize=7,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.4},
                zorder=6,
            )

        if not legend_handles:
            legend_handles = [
                Line2D([0], [0], color=series_info["color"], marker="s", linewidth=0, markersize=7, label=series_info["display"])
                for series_info in all_series
            ]
        existing_legend_labels = {handle.get_label() for handle in legend_handles}
        for reference_line in reference_lines:
            if str(reference_line["label"]) in existing_legend_labels:
                continue
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=reference_line["color"],
                    linewidth=REFERENCE_LINE_WIDTH,
                    linestyle=REFERENCE_LINE_DASH_PATTERN,
                    label=reference_line["label"],
                )
            )
            existing_legend_labels.add(str(reference_line["label"]))

    overall_title = "Overall Change Bars Across All Topics"
    if title_suffix:
        overall_title = f"{overall_title} - {title_suffix}"
    fig.suptitle(overall_title, fontsize=16, y=0.98)
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.07 if page_footnote else 0.01),
            ncol=4,
            frameon=False,
            fontsize=9,
            columnspacing=1.4,
            handlelength=1.4,
        )
    if page_footnote:
        fig.text(0.5, 0.015, page_footnote, ha="center", va="bottom", fontsize=7.2)
    fig.tight_layout(rect=[0, 0.11 if page_footnote else 0.06, 1, 0.96])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def bootstrap_change_standard_error_for_export(
    series: dict[str, object],
    position_keys: list[str],
    anchor_series: dict[str, object],
) -> float:
    primary_group_store = series.get("group_store")
    anchor_group_store = anchor_series.get("group_store")
    if not isinstance(primary_group_store, dict) or not isinstance(anchor_group_store, dict):
        return float("nan")

    primary_complete_groups = extract_complete_groups(primary_group_store, position_keys)
    anchor_complete_groups = extract_complete_groups(anchor_group_store, position_keys)
    if not primary_complete_groups and not anchor_complete_groups:
        return float("nan")

    start_key = str(position_keys[0])
    end_key = str(position_keys[-1])
    start_source_groups = anchor_complete_groups if start_key in SHARED_HUMAN_ANCHOR_KEYS else primary_complete_groups
    end_source_groups = anchor_complete_groups if end_key in SHARED_HUMAN_ANCHOR_KEYS else primary_complete_groups

    start_prefix_values = build_prefix_position_values(start_source_groups, start_key)
    end_prefix_values = build_prefix_position_values(end_source_groups, end_key)
    available_prefixes = sorted(set(start_prefix_values) | set(end_prefix_values))
    if not available_prefixes:
        return float("nan")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_changes: list[float] = []
    for _ in range(BOOTSTRAP_REPS):
        sampled_prefixes = rng.choice(available_prefixes, size=len(available_prefixes), replace=True)
        sampled_start: list[float] = []
        sampled_end: list[float] = []
        for sampled_prefix in sampled_prefixes:
            sampled_start.extend(start_prefix_values.get(str(sampled_prefix), []))
            sampled_end.extend(end_prefix_values.get(str(sampled_prefix), []))
        if not sampled_start or not sampled_end:
            continue
        bootstrap_changes.append(float(np.mean(sampled_end) - np.mean(sampled_start)))

    return bootstrap_standard_error(bootstrap_changes)


def write_report_csv(
    csv_path: Path,
    *,
    report_path: Path,
    collection_type: str,
    comparison_set: str,
    split: str | None,
    partition: str,
    report_version: str,
    topic_display_map: dict[str, str],
    ordered_topic_keys: list[str],
    sd_baseline_targets: dict[str, dict[str, dict[str, float]]],
    page_specs: list[dict[str, object]],
    section_payloads: list[dict[str, object]],
) -> Path:
    rows: list[dict[str, object]] = []
    metadata = {
        "report_relpath": display_path(report_path),
        "report_version": report_version,
        "collection_type": collection_type,
        "comparison_set": comparison_set,
        "split": "" if split is None else split,
        "partition": partition,
    }

    for section_payload in section_payloads:
        section_suffix = str(section_payload["title_suffix"])
        overall_series_by_page = list(section_payload["overall_series_by_page"])
        page_payloads = list(section_payload["page_payloads"])

        for page_spec, page_payload in zip(page_specs, page_payloads):
            position_keys = [str(event_key) for event_key in page_payload["position_keys"]]
            topic_series_map = dict(page_payload["topic_series_map"])
            metric_name = str(page_spec["metric_name"])
            metric_kind = str(page_spec["metric_kind"])
            page_title = str(page_payload["page_title"])
            baseline_axis = None
            if metric_kind == "sd":
                baseline_axis = "tweet" if all(event_key.startswith("tweet") for event_key in position_keys) else "opinion"

            for topic_key in [OVERALL_TOPIC_KEY]:
                topic_label = topic_display_label(topic_key, topic_display_map)
                series_payload = topic_series_map[topic_key]
                anchored_human, anchored_models = apply_shared_human_anchors(
                    series_payload["human"],
                    series_payload["models"],
                    position_keys,
                )

                human_rows = [(series_payload["human"], anchored_human, "human")]
                model_rows = [(raw_model, anchored_model, "model") for raw_model, anchored_model in zip(series_payload["models"], anchored_models)]

                for raw_series, plotted_series, series_role in [*human_rows, *model_rows]:
                    for event_index, event_key in enumerate(position_keys):
                        anchored_to_human = bool(series_role == "model" and event_key in SHARED_HUMAN_ANCHOR_KEYS)
                        rows.append(
                            {
                                **metadata,
                                "row_type": "line_point",
                                "section_title_suffix": section_suffix,
                                "page_title": page_title,
                                "topic_key": topic_key,
                                "topic_display": topic_label,
                                "metric_name": metric_name,
                                "metric_kind": metric_kind,
                                "event_key": event_key,
                                "event_label": event_label_for_key(event_key),
                                "event_index": event_index,
                                "series_role": series_role,
                                "series_key": str(raw_series["series_key"]),
                                "series_display": str(raw_series.get("display", HUMAN_SERIES["display"])),
                                "model_dir": "" if series_role == "human" else str(raw_series.get("model_dir", "")),
                                "n_complete_groups": int(plotted_series["count"]),
                                "plotted_value": float(plotted_series["values"][event_index]),
                                "plotted_se": float(plotted_series["errors"][event_index]),
                                "raw_value": float(raw_series["values"][event_index]),
                                "raw_se": float(raw_series["errors"][event_index]),
                                "anchored_to_human": anchored_to_human,
                                "change_value": "",
                                "change_se": "",
                                "reference_label": "",
                            }
                        )

                if baseline_axis is not None:
                    reference_lines = build_reference_lines(
                        metric_kind,
                        [event_label_for_key(event_key) for event_key in position_keys],
                        topic_key,
                        {OVERALL_TOPIC_KEY: sd_baseline_targets.get(OVERALL_TOPIC_KEY, {}).get(baseline_axis, {})},
                        anchored_human,
                    )
                    for reference_line in reference_lines:
                        x_positions = list(reference_line["x_positions"])
                        values = list(reference_line["values"])
                        for point_index, (x_position, value) in enumerate(zip(x_positions, values)):
                            rows.append(
                                {
                                    **metadata,
                                    "row_type": "reference_point",
                                    "section_title_suffix": section_suffix,
                                    "page_title": page_title,
                                    "topic_key": topic_key,
                                    "topic_display": topic_label,
                                    "metric_name": metric_name,
                                    "metric_kind": metric_kind,
                                    "event_key": "",
                                    "event_label": "",
                                    "event_index": int(x_position),
                                    "series_role": "reference",
                                    "series_key": str(reference_line["label"]),
                                    "series_display": str(reference_line["label"]),
                                    "model_dir": "",
                                    "n_complete_groups": "",
                                    "plotted_value": float(value),
                                    "plotted_se": "",
                                    "raw_value": "",
                                    "raw_se": "",
                                    "anchored_to_human": False,
                                    "change_value": "",
                                    "change_se": "",
                                    "reference_label": str(reference_line["label"]),
                                }
                            )

        for page_spec, overall_payload in zip(page_specs, overall_series_by_page):
            position_keys = [str(event_key) for event_key in page_spec["event_keys"]]
            metric_name = str(page_spec["metric_name"])
            metric_kind = str(page_spec["metric_kind"])
            page_title = str(page_spec["page_title"])
            if section_suffix:
                page_title = f"{page_title} - {section_suffix}"
            anchored_human, anchored_models = apply_shared_human_anchors(
                overall_payload["human"],
                overall_payload["models"],
                position_keys,
            )

            all_series = [(overall_payload["human"], anchored_human, "human")]
            all_series.extend((raw_model, anchored_model, "model") for raw_model, anchored_model in zip(overall_payload["models"], anchored_models))
            for raw_series, plotted_series, series_role in all_series:
                rows.append(
                    {
                        **metadata,
                        "row_type": "overall_change",
                        "section_title_suffix": section_suffix,
                        "page_title": page_title,
                        "topic_key": OVERALL_TOPIC_KEY,
                        "topic_display": "Overall",
                        "metric_name": metric_name,
                        "metric_kind": metric_kind,
                        "event_key": "",
                        "event_label": "",
                        "event_index": "",
                        "series_role": series_role,
                        "series_key": str(raw_series["series_key"]),
                        "series_display": str(raw_series.get("display", HUMAN_SERIES["display"])),
                        "model_dir": "" if series_role == "human" else str(raw_series.get("model_dir", "")),
                        "n_complete_groups": int(plotted_series["count"]),
                        "plotted_value": "",
                        "plotted_se": "",
                        "raw_value": "",
                        "raw_se": "",
                        "anchored_to_human": False,
                        "change_value": float(series_change_value(plotted_series)),
                        "change_se": float(series_change_standard_error_for_export(plotted_series, position_keys, anchored_human)),
                        "reference_label": "",
                    }
                )

            if metric_kind == "sd":
                baseline_axis = "tweet" if all(event_key.startswith("tweet") for event_key in position_keys) else "opinion"
                start_value = float(anchored_human["values"][0]) if anchored_human["values"] else float("nan")
                for label, _color in REFERENCE_LINE_SPECS:
                    target_value = sd_baseline_targets.get(OVERALL_TOPIC_KEY, {}).get(baseline_axis, {}).get(label, float("nan"))
                    if np.isnan(start_value) or np.isnan(target_value):
                        change_value = float("nan")
                    else:
                        change_value = float(target_value - start_value)
                    rows.append(
                        {
                            **metadata,
                            "row_type": "overall_change_reference",
                            "section_title_suffix": section_suffix,
                            "page_title": page_title,
                            "topic_key": OVERALL_TOPIC_KEY,
                            "topic_display": "Overall",
                            "metric_name": metric_name,
                            "metric_kind": metric_kind,
                            "event_key": "",
                            "event_label": "",
                            "event_index": "",
                            "series_role": "reference",
                            "series_key": label,
                            "series_display": label,
                            "model_dir": "",
                            "n_complete_groups": "",
                            "plotted_value": "",
                            "plotted_se": "",
                            "raw_value": "",
                            "raw_se": "",
                            "anchored_to_human": False,
                            "change_value": change_value,
                            "change_se": "",
                            "reference_label": label,
                        }
                    )

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def paginate_topic_keys(ordered_topic_keys: list[str]) -> list[list[str]]:
    if not ordered_topic_keys:
        return []

    pages: list[list[str]] = []
    first_page_topics = ordered_topic_keys[:8]
    pages.append(first_page_topics)

    remaining = ordered_topic_keys[8:]
    for start in range(0, len(remaining), 8):
        pages.append(remaining[start:start + 8])
    return pages


def add_grid_page(
    pdf: PdfPages,
    page_title: str,
    y_label: str,
    ordered_topic_keys: list[str],
    topic_display_map: dict[str, str],
    topic_counts: dict[str, int],
    total_groups: int,
    metric_kind: str,
    topic_series_map: dict[str, dict[str, object]],
    position_keys: list[str],
    x_labels: list[str],
    baseline_targets: dict[str, dict[str, float]] | None = None,
    page_footnote: str | None = None,
) -> None:
    baseline_targets = baseline_targets or {}
    topic_pages = paginate_topic_keys(ordered_topic_keys)

    for page_index, page_topics in enumerate(topic_pages, start=1):
        fig, axes = plt.subplots(3, 3, figsize=(16, 12))
        axes_flat = list(axes.flatten())

        for index, topic_key in enumerate(page_topics):
            ax = axes_flat[index]
            title = topic_display_label(topic_key, topic_display_map)
            series_payload = topic_series_map[topic_key]
            anchored_human, anchored_models = apply_shared_human_anchors(
                series_payload["human"],
                series_payload["models"],
                position_keys,
            )
            create_topic_plot(
                ax=ax,
                title=title,
                y_label=y_label,
                x_labels=x_labels,
                human_series=anchored_human,
                model_series=anchored_models,
                metric_kind=metric_kind,
                show_ylabel=index % 3 == 0,
                show_legend=False,
                reference_lines=build_reference_lines(
                    metric_kind,
                    x_labels,
                    topic_key,
                    baseline_targets,
                    anchored_human,
                ),
                panel_note=build_panel_count_note(anchored_human, anchored_models),
                title_wrap_width=64,
            )

        legend_axis = axes_flat[8]
        first_topic_key = page_topics[0] if page_topics else OVERALL_TOPIC_KEY
        build_legend_panel(
            legend_axis,
            len(x_labels),
            topic_series_map[first_topic_key]["models"],
            [{"label": label, "color": color} for label, color in REFERENCE_LINE_SPECS] if metric_kind == "sd" else [],
        )

        for axis in axes_flat[len(page_topics):8]:
            axis.axis("off")

        total_pages = len(topic_pages)
        if total_pages > 1:
            current_page_title = f"{page_title} (Page {page_index}/{total_pages})"
        else:
            current_page_title = page_title
        fig.suptitle(current_page_title, fontsize=16, y=0.98)
        if page_footnote:
            fig.text(0.5, 0.015, page_footnote, ha="center", va="bottom", fontsize=7.2)
        fig.tight_layout(rect=[0, 0.05 if page_footnote else 0, 1, 0.97])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def render_outputs(
    output_dir: Path,
    report_path: Path,
    collection_type: str,
    comparison_set: str,
    split: str | None,
    partition: str,
    report_version: str,
    apply_rejudged_sidecars: bool,
    series_specs: list[dict[str, str]],
    golden_prefixes: list[str],
    topic_display_map: dict[str, str],
    topic_counts: dict[str, int],
    metric_store: dict[str, dict[str, dict[str, dict[str, dict[str, float]]]]],
    sd_baseline_targets: dict[str, dict[str, dict[str, float]]],
    report_metadata: dict[str, bool],
    page_specs: list[dict[str, object]],
    plot_specs: list[dict[str, object]],
    write_pngs: bool,
    write_pdf: bool = True,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_csv_path = report_path.with_suffix(".csv")

    topic_keys = {topic_key for topic_key in topic_display_map if topic_key != OVERALL_TOPIC_KEY}
    topic_keys.update(topic_key for topic_key in topic_counts if topic_key != OVERALL_TOPIC_KEY)
    topic_keys.update(topic_key for topic_key in sd_baseline_targets if topic_key != OVERALL_TOPIC_KEY)
    for metric_topics in metric_store.values():
        topic_keys.update(topic_key for topic_key in metric_topics if topic_key != OVERALL_TOPIC_KEY)
    for topic_key in topic_keys:
        topic_display_map.setdefault(topic_key, display_topic(topic_key))
    ordered_topic_keys = sorted(topic_keys, key=lambda topic_key: topic_display_label(topic_key, topic_display_map))
    position_keys = [event_key for event_key, _, _, _ in COMBINED_EVENTS]
    generated_paths: list[Path] = []
    page_group_specs = build_page_group_specs(comparison_set, collection_type, series_specs)
    skip_topic_pages = skip_breadth_per_topic_pages_in_post_training(
        collection_type,
        comparison_set,
        report_version,
    )
    external_overall_series_by_page = load_degroot_fit_overall_series(
        collection_type,
        comparison_set,
        split,
        partition,
        report_version,
    )

    if write_pngs:
        for plot_spec in plot_specs:
            metric_name = str(plot_spec["metric_name"])
            for topic_key in [OVERALL_TOPIC_KEY] + ordered_topic_keys:
                if skip_topic_pages and topic_key != OVERALL_TOPIC_KEY:
                    continue
                topic_metric_store = metric_store[metric_name].get(topic_key, {})
                human_series = build_series_stats(topic_metric_store, HUMAN_SERIES["label"], position_keys)
                model_series = []
                for model_spec in series_specs[1:]:
                    stats = build_series_stats(topic_metric_store, model_spec["label"], position_keys)
                    model_series.append(
                        {
                            "display": model_spec["display"],
                            "model_dir": model_spec["model_dir"],
                            "color": color_for_series(model_spec["label"]),
                            "values": stats["values"],
                            "errors": stats["errors"],
                            "count": stats["count"],
                        }
                    )

                topic_slug = "overall" if topic_key == OVERALL_TOPIC_KEY else slugify(topic_display_label(topic_key, topic_display_map))
                output_path = output_dir / f"{plot_spec['file_prefix']}_{topic_slug}.png"
                title = topic_display_label(topic_key, topic_display_map)
                create_topic_png(
                    output_path=output_path,
                    title=f"{plot_spec['title_prefix']}\n{title}",
                    y_label=str(plot_spec["y_label"]),
                    position_keys=position_keys,
                    x_labels=[label for _, label, _, _ in COMBINED_EVENTS],
                    human_series=human_series,
                    model_series=model_series,
                    metric_kind=str(plot_spec["metric_kind"]),
                )
                generated_paths.append(output_path)

    section_payloads: list[dict[str, object]] = []
    for page_group_spec in page_group_specs:
        page_payloads: list[dict[str, object]] = []
        overall_series_by_page: list[dict[str, dict[str, object]]] = []
        allowed_model_dirs = page_group_spec["allowed_model_dirs"]
        for page_spec in page_specs:
            x_labels = [label for event_key, label, _, _ in COMBINED_EVENTS if event_key in page_spec["event_keys"]]
            topic_series_map: dict[str, dict[str, object]] = {}

            for topic_key in [OVERALL_TOPIC_KEY] + ordered_topic_keys:
                topic_metric_store = metric_store[page_spec["metric_name"]].get(topic_key, {})
                human_series = build_series_stats(topic_metric_store, HUMAN_SERIES["label"], page_spec["event_keys"])
                model_series = []
                for model_spec in series_specs[1:]:
                    stats = build_series_stats(topic_metric_store, model_spec["label"], page_spec["event_keys"])
                    model_series.append(
                        {
                            "series_key": model_spec["label"],
                            "display": model_spec["display"],
                            "model_dir": model_spec["model_dir"],
                            "color": color_for_series(model_spec["label"]),
                            "values": stats["values"],
                            "errors": stats["errors"],
                            "count": stats["count"],
                            "group_store": topic_metric_store.get(model_spec["label"], {}),
                        }
                    )
                if topic_key == OVERALL_TOPIC_KEY:
                    model_series = append_external_overall_series(
                        model_series,
                        str(page_spec["page_title"]),
                        external_overall_series_by_page,
                    )
                if allowed_model_dirs is not None:
                    model_series = [model for model in model_series if model["model_dir"] in allowed_model_dirs]
                topic_series_map[topic_key] = {
                    "human": {
                        **human_series,
                        "series_key": HUMAN_SERIES["label"],
                        "group_store": topic_metric_store.get(HUMAN_SERIES["label"], {}),
                    },
                    "models": model_series,
                }
            overall_series_by_page.append(topic_series_map[OVERALL_TOPIC_KEY])

            baseline_axis = None
            if page_spec["metric_kind"] == "sd":
                baseline_axis = "tweet" if all(event_key.startswith("tweet") for event_key in page_spec["event_keys"]) else "opinion"
            page_title = str(page_spec["page_title"])
            if page_group_spec["title_suffix"]:
                page_title = f"{page_title} - {page_group_spec['title_suffix']}"
            page_payloads.append(
                {
                    "page_title": page_title,
                    "y_label": page_spec["y_label"],
                    "metric_kind": page_spec["metric_kind"],
                    "topic_series_map": topic_series_map,
                    "position_keys": page_spec["event_keys"],
                    "x_labels": x_labels,
                    "baseline_targets": {
                        topic_key: sd_baseline_targets.get(topic_key, {}).get(baseline_axis, {})
                        for topic_key in [OVERALL_TOPIC_KEY] + ordered_topic_keys
                    } if baseline_axis is not None else None,
                }
            )

        section_payloads.append(
            {
                "title_suffix": str(page_group_spec["title_suffix"]),
                "footnote": append_opacity_footnote(page_group_spec["footnote"]),
                "overall_series_by_page": overall_series_by_page,
                "page_payloads": page_payloads,
            }
        )

    if write_pdf:
        with PdfPages(report_path) as pdf:
            add_topic_count_page(
                pdf=pdf,
                collection_type=collection_type,
                comparison_set=comparison_set,
                split=split,
                partition=partition,
                report_version=report_version,
                apply_rejudged_sidecars=apply_rejudged_sidecars,
                total_groups=len(golden_prefixes),
                ordered_topic_keys=ordered_topic_keys,
                topic_display_map=topic_display_map,
                topic_counts=topic_counts,
                report_metadata=report_metadata,
            )

            for section_payload in section_payloads:
                add_overall_summary_page(
                    pdf=pdf,
                    page_specs=page_specs,
                    overall_series_by_page=section_payload["overall_series_by_page"],
                    sd_baseline_targets=sd_baseline_targets,
                    title_suffix=str(section_payload["title_suffix"]),
                    page_footnote=section_payload["footnote"],
                )
                add_overall_delta_bar_page(
                    pdf=pdf,
                    page_specs=page_specs,
                    overall_series_by_page=section_payload["overall_series_by_page"],
                    sd_baseline_targets=sd_baseline_targets,
                    title_suffix=str(section_payload["title_suffix"]),
                    page_footnote=section_payload["footnote"],
                )

            for section_payload in section_payloads:
                for page_payload in section_payload["page_payloads"]:
                    if skip_topic_pages:
                        continue
                    add_grid_page(
                        pdf=pdf,
                        page_title=page_payload["page_title"],
                        y_label=page_payload["y_label"],
                        ordered_topic_keys=ordered_topic_keys,
                        topic_display_map=topic_display_map,
                        topic_counts=topic_counts,
                        total_groups=len(golden_prefixes),
                        metric_kind=page_payload["metric_kind"],
                        topic_series_map=page_payload["topic_series_map"],
                        position_keys=page_payload["position_keys"],
                        x_labels=page_payload["x_labels"],
                        baseline_targets=page_payload["baseline_targets"],
                        page_footnote=section_payload["footnote"],
                    )

        generated_paths.append(report_path)

    write_report_csv(
        report_csv_path,
        report_path=report_path,
        collection_type=collection_type,
        comparison_set=comparison_set,
        split=split,
        partition=partition,
        report_version=report_version,
        topic_display_map=topic_display_map,
        ordered_topic_keys=ordered_topic_keys,
        sd_baseline_targets=sd_baseline_targets,
        page_specs=page_specs,
        section_payloads=section_payloads,
    )
    generated_paths.append(report_csv_path)

    return generated_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate v3 breadth/depth base-model group mean and SD line-plot PDF reports."
    )
    parser.add_argument("--comparison-set", choices=["base-models", "post-training"], default="base-models")
    parser.add_argument("--collection-type", choices=["depth", "breadth"], default="depth")
    parser.add_argument("--split", choices=["topic", "group"], default=None)
    parser.add_argument("--partition", choices=["all", "train", "test"], default="all")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--report-version", default="v3")
    parser.add_argument("--write-pngs", action="store_true")
    parser.add_argument("--write-csv-only", action="store_true")
    parser.add_argument("--disable-rejudged-sidecars", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_transition_simulation_version_for_result(args.report_version)
    if report_version_family(args.report_version) in {"v3", "v4"}:
        enable_v3_complete_group_masks()
    if args.partition != "all" and args.split is None:
        raise ValueError("--partition train/test requires --split topic/group")
    if args.comparison_set == "post-training" and args.split is None:
        raise ValueError("post-training reports require --split topic/group")

    output_dir = args.output_dir or default_output_dir(args.comparison_set, args.report_version)
    report_path = args.report_path or default_report_path(
        output_dir,
        args.collection_type,
        args.comparison_set,
        args.split,
        args.partition,
        args.report_version,
    )
    scope_label = selection_context_label(args.collection_type, args.split, args.partition)
    page_specs = build_report_page_specs(scope_label)
    plot_specs = build_plot_specs(scope_label, selection_slug(args.collection_type, args.split, args.partition))
    series_specs = build_series_specs(args.comparison_set, args.collection_type, args.split, args.report_version)
    apply_rejudged_sidecars = not args.disable_rejudged_sidecars
    golden_prefixes, topic_display_map, topic_counts, metric_store, sd_baseline_targets, report_metadata = collect_plot_inputs(
        args.collection_type,
        split=args.split,
        partition=args.partition,
        series_specs=series_specs,
        apply_rejudged_sidecars=apply_rejudged_sidecars,
    )
    generated_paths = render_outputs(
        output_dir=output_dir,
        report_path=report_path,
        collection_type=args.collection_type,
        comparison_set=args.comparison_set,
        split=args.split,
        partition=args.partition,
        report_version=args.report_version,
        apply_rejudged_sidecars=apply_rejudged_sidecars,
        series_specs=series_specs,
        golden_prefixes=golden_prefixes,
        topic_display_map=topic_display_map,
        topic_counts=topic_counts,
        metric_store=metric_store,
        sd_baseline_targets=sd_baseline_targets,
        report_metadata=report_metadata,
        page_specs=page_specs,
        plot_specs=plot_specs,
        write_pngs=args.write_pngs,
        write_pdf=not args.write_csv_only,
    )

    print(f"Loaded {len(golden_prefixes)} selected golden {args.collection_type} prefixes.")
    if args.write_pngs:
        print(f"Wrote {len(generated_paths)} plots to {output_dir}")
        for path in generated_paths:
            print(display_path(path))
    else:
        print("Skipped standalone PNG generation.")
    if args.write_csv_only:
        print(f"Wrote CSV export to {display_path(report_path.with_suffix('.csv'))}")
    else:
        print(f"Wrote PDF report to {display_path(report_path)}")
        print(display_path(report_path))


if __name__ == "__main__":
    main()
