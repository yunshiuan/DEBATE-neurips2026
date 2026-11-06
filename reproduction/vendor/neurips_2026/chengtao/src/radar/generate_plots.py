#!/usr/bin/env python3
"""Build DEBATE metrics/plot artifacts from human/LLM simulation CSVs.

Current suite:
1) Message-level stance difference (lower absolute difference is better)
2) Public stance alignment (LLM tweet - human tweet; closer to zero is better)
3) Private stance alignment (LLM private opinion - human private opinion; closer to zero is better)
4) Public stance change error ([human tweet3 - tweet1] - [LLM tweet3 - tweet1]; closer to zero is better)
5) Private stance change error ([human final - initial] - [LLM final - initial]; closer to zero is better)
6) Stance update direction accuracy (higher is better)
7) Public opinion convergence prediction (difference from human group-level convergence; closer to zero is better)
8) Private opinion convergence prediction (difference from human group-level convergence; closer to zero is better)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import re
import statistics
import sys
import textwrap
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neurips_2026.sean_reproduce.base_model_v3_style import (  # noqa: E402
    BASE_MODEL_V3_COLOR_MAP,
    BASE_MODEL_V3_LLM_ORDER,
    ordered_base_model_llms_for_report,
)
from neurips_2026.sean_reproduce.result_version_config import (  # noqa: E402
    LLAMA_V4_BREADTH_GROUP_SFT_MODEL,
    LLAMA_V4_BREADTH_TOPIC_SFT_MODEL,
    LLAMA_V4_DEPTH_GROUP_SFT_MODEL,
    LLAMA_V4_DEPTH_TOPIC_SFT_MODEL,
)
from neurips_2026.sean_reproduce.v3.masks import filter_rows_for_v3_complete_group_mask  # noqa: E402

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency for plotting only
    plt = None

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency for progress display
    tqdm = None


HUMAN_DEFAULT_FILENAME = "opinion_human_memory_gpt-5-mini_v0.csv"
LLM_DEFAULT_FILENAME = "opinion_llm_memory_gpt-5-mini_v0.csv"

DEFAULT_UTTERANCE_EVENT_TYPES = ("Initial Opinion", "tweet", "message_sent", "Post Opinion")

LIKERT_LABEL_TO_NUM = {
    "certainly disagree": 1.0,
    "probably disagree": 2.0,
    "lean disagree": 3.0,
    "lean agree": 4.0,
    "probably agree": 5.0,
    "certainly agree": 6.0,
}

LIKERT_KEY_FIELDS = (
    "group_id",
    "chat_round_order",
    "event_order",
    "webpage_order",
    "event_type",
    "message_id",
    "empirica_id",
    "worker_id",
    "sender_id",
    "recipient_id",
    "chat_order",
)

RADAR_AXES = [
    ("stance_difference", "Stance Difference"),
    ("public_stance_alignment", "Public Alignment"),
    ("private_stance_alignment", "Private Alignment"),
    ("public_stance_change_error", "Public Change Error"),
    ("private_stance_change_error", "Private Change Error"),
    ("stance_update_direction_accuracy", "Direction Accuracy"),
    ("public_opinion_convergence", "Public Convergence Prediction"),
    ("private_opinion_convergence", "Private Convergence Prediction"),
]

# v1: 6-metric Kendall tau-b axes (drops Stance Difference and Direction Accuracy)
RADAR_AXES_V1 = [
    ("public_stance_alignment", "Public Alignment"),
    ("private_stance_alignment", "Private Alignment"),
    ("public_stance_change_error", "Public Change Error"),
    ("private_stance_change_error", "Private Change Error"),
    ("public_opinion_convergence", "Public Convergence Prediction"),
    ("private_opinion_convergence", "Private Convergence Prediction"),
]

# Reference model identifiers for special plot styling (v1)
PERM_LB_MODEL = "Perm. LB"
MEAN_PREDICTOR_MODEL = "Mean Predictor"
UNIFORM_LB_MODEL = "Uniform LB"
NOISY_UB_PREFIX = "Noisy UB"
LIKERT_GAUSSIAN_SIGMA = 1.0
LIKERT_GAUSSIAN_SIGMA_SECONDARY = 0.5


def noisy_ub_gaussian_label(sigma: float) -> str:
    return f"{NOISY_UB_PREFIX} (Gaussian on Likert labels; sigma={sigma})"


NOISY_UB_GAUSSIAN_LABEL = noisy_ub_gaussian_label(LIKERT_GAUSSIAN_SIGMA)
NOISY_UB_GAUSSIAN_SIGMA_SECONDARY_LABEL = noisy_ub_gaussian_label(LIKERT_GAUSSIAN_SIGMA_SECONDARY)
NOISY_UB_JITTER_LABEL = f"{NOISY_UB_PREFIX} (\u00b11 jitter)"

LLAMA_8B_INSTRUCT_MODEL = "Llama-3.1-8B-Instruct"
LLAMA_SFT_HIGHLIGHT_COLOR = "#2F5D8A"
LLAMA_SFT_MODEL_PREFIXES = (
    "ft:Llama-3.1-8B-SFT-",
    "ft:Llama-3.1-8B-Instruct-SFT-",
)
DEGROOT_MODEL_NAMES = {"DeGroot fit", "DeGroot fit topic", "DeGroot fit group"}
DEGROOT_MODEL_PREFIX = "DeGroot alpha="

AXIS_GROUP = {
    "stance_difference": "message",
    "public_stance_alignment": "message",
    "private_stance_alignment": "message",
    "public_stance_change_error": "individual",
    "private_stance_change_error": "individual",
    "stance_update_direction_accuracy": "individual",
    "public_opinion_convergence": "group",
    "private_opinion_convergence": "group",
}

GROUP_TEXT_COLORS = {
    "message": "#1f77b4",     # blue
    "group": "#2ca02c",       # green
    "individual": "#d62728",  # red
}

MODEL_COLOR_MAP = {
    **BASE_MODEL_V3_COLOR_MAP,
    "gpt-4o-mini-2024-07-18-length_control": "#F2B5A7",
    "ft:Llama-3.1-8B-SFT-20260210:breadth-topic-3epochs-unsloth": LLAMA_SFT_HIGHLIGHT_COLOR,
    "ft:Llama-3.1-8B-Instruct-DPO-20260212:breadth-topic-5epochs": "#8EA9DC",
    "ft:Llama-3.1-8B-SFT-20260210:breadth-group-3epochs-unsloth": LLAMA_SFT_HIGHLIGHT_COLOR,
    "ft:Llama-3.1-8B-Instruct-DPO-20260211:breadth-group-5epochs": "#6F8FCE",
    "ft:Llama-3.1-8B-Instruct-SFT-20251218:depth-topic-5epochs": LLAMA_SFT_HIGHLIGHT_COLOR,
    "ft:Llama-3.1-8B-Instruct-DPO-20260105:depth-topic-5epochs": "#6C8CCE",
    "ft:Llama-3.1-8B-Instruct-SFT-20251218:depth-group-5epochs": LLAMA_SFT_HIGHLIGHT_COLOR,
    "ft:Llama-3.1-8B-Instruct-DPO-20260105:depth-group-5epochs": "#597CC5",
    LLAMA_V4_BREADTH_TOPIC_SFT_MODEL: LLAMA_SFT_HIGHLIGHT_COLOR,
    LLAMA_V4_BREADTH_GROUP_SFT_MODEL: LLAMA_SFT_HIGHLIGHT_COLOR,
    LLAMA_V4_DEPTH_TOPIC_SFT_MODEL: LLAMA_SFT_HIGHLIGHT_COLOR,
    LLAMA_V4_DEPTH_GROUP_SFT_MODEL: LLAMA_SFT_HIGHLIGHT_COLOR,
    "ft:gpt-4o-mini-2024-07-18:camer:sft-20251218-depth-topic-5epochs:CoFxFPcv": "#A63A32",
    "ft:gpt-4o-mini-2024-07-18:camer:sft-20251218-depth-group-5epochs:CoFzijg0": "#A63A32",
    MEAN_PREDICTOR_MODEL: "#595959",
}

V2_FIXED_MODEL_ORDER = list(BASE_MODEL_V3_LLM_ORDER)

V2_BAR_METRIC_FIELDS = [
    "public_stance_alignment_abs",
    "private_stance_alignment_abs",
    "public_stance_change_error_abs",
    "private_stance_change_error_abs",
    "public_opinion_convergence_error",
    "private_opinion_convergence_error",
]
V2_BOOTSTRAP_REPS = 1000
V2_BOOTSTRAP_SEED = 1
V2_BAR_Y_LIMITS = {
    "Message-level": (-0.02, 3.2),
    "Individual-level": (-0.02, 3.1),
    "Group-level": (-0.02, 2.25),
}

V2_EXCLUDED_ORDER_MODELS = {
    PERM_LB_MODEL,
    MEAN_PREDICTOR_MODEL,
    UNIFORM_LB_MODEL,
    NOISY_UB_GAUSSIAN_LABEL,
    NOISY_UB_GAUSSIAN_SIGMA_SECONDARY_LABEL,
    NOISY_UB_JITTER_LABEL,
}

DEPTH_TOPICS = {
    'A_"body_cleanse,"_in_which_you_consume_only_particular_kinds_of_nutrients_over_1-3_days,_helps_your_body_to_eliminate_toxins',
    "Angels_are_real",
    "The_United_States_has_the_highest_federal_income_tax_rate_of_any_Western_country",
    "Everything_that_happens_can_eventually_be_explained_by_science",
    "Regular_fasting_will_improve_your_health",
    "The_US_deficit_increased_after_President_Obama_was_elected",
    "The_position_of_the_planets_at_the_time_of_your_birth_can_influence_your_personality",
}

PROFILE_LINE_STYLES = ("-", "--", ":", "-.")
PROFILE_Y_LIM = (-0.05, 1.05)

FINE_TUNED_DISPLAY_MAP = {
    "ft:Llama-3.1-8B-SFT-20260210:breadth-topic-3epochs-unsloth": "Llama SFT",
    "ft:Llama-3.1-8B-Instruct-DPO-20260212:breadth-topic-5epochs": "LLaMA DPO Topic",
    "ft:Llama-3.1-8B-SFT-20260210:breadth-group-3epochs-unsloth": "Llama SFT",
    "ft:Llama-3.1-8B-Instruct-DPO-20260211:breadth-group-5epochs": "LLaMA DPO Group",
    LLAMA_V4_BREADTH_TOPIC_SFT_MODEL: "Llama SFT",
    LLAMA_V4_BREADTH_GROUP_SFT_MODEL: "Llama SFT",
    "ft:Llama-3.1-8B-Instruct-SFT-20251218:depth-topic-5epochs": "Llama SFT",
    "ft:Llama-3.1-8B-Instruct-DPO-20260105:depth-topic-5epochs": "LLaMA DPO Topic",
    LLAMA_V4_DEPTH_TOPIC_SFT_MODEL: "Llama SFT",
    "ft:Llama-3.1-8B-Instruct-SFT-20251218:depth-group-5epochs": "Llama SFT",
    "ft:Llama-3.1-8B-Instruct-DPO-20260105:depth-group-5epochs": "LLaMA DPO Group",
    LLAMA_V4_DEPTH_GROUP_SFT_MODEL: "Llama SFT",
    "ft:gpt-4o-mini-2024-07-18:camer:sft-20251218-depth-topic-5epochs:CoFxFPcv": "GPT SFT Topic",
    "ft:gpt-4o-mini-2024-07-18:camer:sft-20251218-depth-group-5epochs:CoFzijg0": "GPT SFT Group",
}


@dataclass
class Profile:
    initial_private: dict[str, dict[str, float]] = field(default_factory=lambda: defaultdict(dict))
    final_private: dict[str, dict[str, float]] = field(default_factory=lambda: defaultdict(dict))
    tweet_round: dict[str, dict[str, dict[int, float]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(dict))
    )


@dataclass
class PrefixModelResult:
    status: str
    data_prefix: str
    model: str
    message: str
    human_path: str = ""
    llm_path: str = ""
    alignment_mode: str = ""
    key_match_rate: float = 0.0
    aligned_pairs: int = 0
    stance_sum: float = 0.0
    stance_count: int = 0
    public_alignment_sum: float = 0.0
    public_alignment_abs_sum: float = 0.0
    public_alignment_count: int = 0
    private_alignment_sum: float = 0.0
    private_alignment_abs_sum: float = 0.0
    private_alignment_count: int = 0
    public_conv_errors: list[float] = field(default_factory=list)
    private_conv_errors: list[float] = field(default_factory=list)
    public_conv_h_vals: list[float] = field(default_factory=list)
    public_conv_l_vals: list[float] = field(default_factory=list)
    private_conv_h_vals: list[float] = field(default_factory=list)
    private_conv_l_vals: list[float] = field(default_factory=list)
    public_change_errors: list[float] = field(default_factory=list)
    public_change_abs_errors: list[float] = field(default_factory=list)
    private_change_errors: list[float] = field(default_factory=list)
    private_change_abs_errors: list[float] = field(default_factory=list)
    direction_true: list[int] = field(default_factory=list)
    direction_pred: list[int] = field(default_factory=list)
    lb_stance_sum: float = 0.0
    lb_stance_count: int = 0
    lb_public_alignment_sum: float = 0.0
    lb_public_alignment_abs_sum: float = 0.0
    lb_public_alignment_count: int = 0
    lb_private_alignment_sum: float = 0.0
    lb_private_alignment_abs_sum: float = 0.0
    lb_private_alignment_count: int = 0
    lb_public_conv_errors: list[float] = field(default_factory=list)
    lb_private_conv_errors: list[float] = field(default_factory=list)
    lb_public_change_errors: list[float] = field(default_factory=list)
    lb_public_change_abs_errors: list[float] = field(default_factory=list)
    lb_private_change_errors: list[float] = field(default_factory=list)
    lb_private_change_abs_errors: list[float] = field(default_factory=list)
    lb_direction_true: list[int] = field(default_factory=list)
    lb_direction_pred: list[int] = field(default_factory=list)
    # raw paired values for Kendall tau-b scoring (v1)
    public_align_h_vals: list[float] = field(default_factory=list)
    public_align_l_vals: list[float] = field(default_factory=list)
    private_align_h_vals: list[float] = field(default_factory=list)
    private_align_l_vals: list[float] = field(default_factory=list)
    public_change_h_vals: list[float] = field(default_factory=list)
    public_change_l_vals: list[float] = field(default_factory=list)
    private_change_h_vals: list[float] = field(default_factory=list)
    private_change_l_vals: list[float] = field(default_factory=list)
    ub_public_align_h_vals: list[float] = field(default_factory=list)
    ub_public_align_l_vals: list[float] = field(default_factory=list)
    ub_private_align_h_vals: list[float] = field(default_factory=list)
    ub_private_align_l_vals: list[float] = field(default_factory=list)
    ub_public_change_h_vals: list[float] = field(default_factory=list)
    ub_public_change_l_vals: list[float] = field(default_factory=list)
    ub_private_change_h_vals: list[float] = field(default_factory=list)
    ub_private_change_l_vals: list[float] = field(default_factory=list)
    ub_public_conv_h_vals: list[float] = field(default_factory=list)
    ub_public_conv_l_vals: list[float] = field(default_factory=list)
    ub_private_conv_h_vals: list[float] = field(default_factory=list)
    ub_private_conv_l_vals: list[float] = field(default_factory=list)
    ub_sigma_half_public_align_h_vals: list[float] = field(default_factory=list)
    ub_sigma_half_public_align_l_vals: list[float] = field(default_factory=list)
    ub_sigma_half_private_align_h_vals: list[float] = field(default_factory=list)
    ub_sigma_half_private_align_l_vals: list[float] = field(default_factory=list)
    ub_sigma_half_public_change_h_vals: list[float] = field(default_factory=list)
    ub_sigma_half_public_change_l_vals: list[float] = field(default_factory=list)
    ub_sigma_half_private_change_h_vals: list[float] = field(default_factory=list)
    ub_sigma_half_private_change_l_vals: list[float] = field(default_factory=list)
    ub_sigma_half_public_conv_h_vals: list[float] = field(default_factory=list)
    ub_sigma_half_public_conv_l_vals: list[float] = field(default_factory=list)
    ub_sigma_half_private_conv_h_vals: list[float] = field(default_factory=list)
    ub_sigma_half_private_conv_l_vals: list[float] = field(default_factory=list)
    uniform_public_alignment_abs_sum: float = 0.0
    uniform_public_alignment_count: int = 0
    uniform_private_alignment_abs_sum: float = 0.0
    uniform_private_alignment_count: int = 0
    uniform_public_conv_errors: list[float] = field(default_factory=list)
    uniform_private_conv_errors: list[float] = field(default_factory=list)
    uniform_public_change_abs_errors: list[float] = field(default_factory=list)
    uniform_private_change_abs_errors: list[float] = field(default_factory=list)


@dataclass
class ModelAggregate:
    model: str
    prefixes_ok: int = 0
    prefixes_missing: int = 0
    stance_sum: float = 0.0
    stance_count: int = 0
    public_alignment_sum: float = 0.0
    public_alignment_abs_sum: float = 0.0
    public_alignment_count: int = 0
    private_alignment_sum: float = 0.0
    private_alignment_abs_sum: float = 0.0
    private_alignment_count: int = 0
    public_conv_errors: list[float] = field(default_factory=list)
    private_conv_errors: list[float] = field(default_factory=list)
    public_conv_h_vals: list[float] = field(default_factory=list)
    public_conv_l_vals: list[float] = field(default_factory=list)
    private_conv_h_vals: list[float] = field(default_factory=list)
    private_conv_l_vals: list[float] = field(default_factory=list)
    public_change_errors: list[float] = field(default_factory=list)
    public_change_abs_errors: list[float] = field(default_factory=list)
    private_change_errors: list[float] = field(default_factory=list)
    private_change_abs_errors: list[float] = field(default_factory=list)
    direction_true: list[int] = field(default_factory=list)
    direction_pred: list[int] = field(default_factory=list)
    lb_stance_sum: float = 0.0
    lb_stance_count: int = 0
    lb_public_alignment_sum: float = 0.0
    lb_public_alignment_abs_sum: float = 0.0
    lb_public_alignment_count: int = 0
    lb_private_alignment_sum: float = 0.0
    lb_private_alignment_abs_sum: float = 0.0
    lb_private_alignment_count: int = 0
    lb_public_conv_errors: list[float] = field(default_factory=list)
    lb_private_conv_errors: list[float] = field(default_factory=list)
    lb_public_change_errors: list[float] = field(default_factory=list)
    lb_public_change_abs_errors: list[float] = field(default_factory=list)
    lb_private_change_errors: list[float] = field(default_factory=list)
    lb_private_change_abs_errors: list[float] = field(default_factory=list)
    lb_direction_true: list[int] = field(default_factory=list)
    lb_direction_pred: list[int] = field(default_factory=list)
    # raw paired values for Kendall tau-b scoring (v1)
    public_align_h_vals: list[float] = field(default_factory=list)
    public_align_l_vals: list[float] = field(default_factory=list)
    private_align_h_vals: list[float] = field(default_factory=list)
    private_align_l_vals: list[float] = field(default_factory=list)
    public_change_h_vals: list[float] = field(default_factory=list)
    public_change_l_vals: list[float] = field(default_factory=list)
    private_change_h_vals: list[float] = field(default_factory=list)
    private_change_l_vals: list[float] = field(default_factory=list)
    ub_public_align_h_vals: list[float] = field(default_factory=list)
    ub_public_align_l_vals: list[float] = field(default_factory=list)
    ub_private_align_h_vals: list[float] = field(default_factory=list)
    ub_private_align_l_vals: list[float] = field(default_factory=list)
    ub_public_change_h_vals: list[float] = field(default_factory=list)
    ub_public_change_l_vals: list[float] = field(default_factory=list)
    ub_private_change_h_vals: list[float] = field(default_factory=list)
    ub_private_change_l_vals: list[float] = field(default_factory=list)
    ub_public_conv_h_vals: list[float] = field(default_factory=list)
    ub_public_conv_l_vals: list[float] = field(default_factory=list)
    ub_private_conv_h_vals: list[float] = field(default_factory=list)
    ub_private_conv_l_vals: list[float] = field(default_factory=list)
    ub_sigma_half_public_align_h_vals: list[float] = field(default_factory=list)
    ub_sigma_half_public_align_l_vals: list[float] = field(default_factory=list)
    ub_sigma_half_private_align_h_vals: list[float] = field(default_factory=list)
    ub_sigma_half_private_align_l_vals: list[float] = field(default_factory=list)
    ub_sigma_half_public_change_h_vals: list[float] = field(default_factory=list)
    ub_sigma_half_public_change_l_vals: list[float] = field(default_factory=list)
    ub_sigma_half_private_change_h_vals: list[float] = field(default_factory=list)
    ub_sigma_half_private_change_l_vals: list[float] = field(default_factory=list)
    ub_sigma_half_public_conv_h_vals: list[float] = field(default_factory=list)
    ub_sigma_half_public_conv_l_vals: list[float] = field(default_factory=list)
    ub_sigma_half_private_conv_h_vals: list[float] = field(default_factory=list)
    ub_sigma_half_private_conv_l_vals: list[float] = field(default_factory=list)
    uniform_public_alignment_abs_sum: float = 0.0
    uniform_public_alignment_count: int = 0
    uniform_private_alignment_abs_sum: float = 0.0
    uniform_private_alignment_count: int = 0
    uniform_public_conv_errors: list[float] = field(default_factory=list)
    uniform_private_conv_errors: list[float] = field(default_factory=list)
    uniform_public_change_abs_errors: list[float] = field(default_factory=list)
    uniform_private_change_abs_errors: list[float] = field(default_factory=list)


def get_model_display_label(model_name: str) -> str:
    return FINE_TUNED_DISPLAY_MAP.get(model_name, model_name)


def is_llama_sft_model(model_name: str) -> bool:
    model_text = str(model_name).strip()
    return any(model_text.startswith(prefix) for prefix in LLAMA_SFT_MODEL_PREFIXES)


def is_degroot_model(model_name: str) -> bool:
    model_text = str(model_name).strip()
    return model_text.startswith(DEGROOT_MODEL_PREFIX) or model_text in DEGROOT_MODEL_NAMES


def format_model_display_label(model_name: str) -> str:
    label = get_model_display_label(model_name)
    replacements = {
        "human": "Human",
        "-Instruct": "\nInstruct",
        "-mini-": "-mini\n",
        "-flash-": "-flash\n",
        "-lite": "\nlite",
    }
    for old, new in replacements.items():
        if old == model_name:
            return new
        if old in label:
            label = label.replace(old, new, 1)
    return label


def get_model_footnotes(model_names: list[str]) -> list[str]:
    footnotes: list[str] = []
    seen: set[str] = set()
    for model_name in model_names:
        display_name = FINE_TUNED_DISPLAY_MAP.get(model_name)
        if display_name is None or model_name in seen:
            continue
        seen.add(model_name)
        footnotes.append(f"{display_name} = {model_name}")
    return footnotes


def add_model_footnotes(fig: Any, footnotes: list[str], *, y: float = 0.01) -> None:
    if not footnotes:
        return
    footnote_text = "Short labels: " + " | ".join(footnotes)
    fig.text(
        0.01,
        y,
        "\n".join(textwrap.wrap(footnote_text, width=120, break_long_words=False, break_on_hyphens=False)),
        ha="left",
        va="bottom",
        fontsize=8,
        color="#555555",
    )


def read_lines(path: Path) -> list[str]:
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            lines.append(line)
    return lines


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_likert(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    maybe_num = maybe_float(text)
    if maybe_num is not None:
        return maybe_num

    label = text.lower()
    if label in LIKERT_LABEL_TO_NUM:
        return LIKERT_LABEL_TO_NUM[label]

    return None


def parse_round(value: Any) -> int | None:
    maybe_num = maybe_float(value)
    if maybe_num is None:
        return None
    return int(round(maybe_num))


def sign_label(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def safe_mean(values: list[float]) -> float:
    if not values:
        return math.nan
    return sum(values) / len(values)


def mean_absolute_error_to_mean(values: list[float]) -> float:
    if not values:
        return math.nan
    mean_value = safe_mean(values)
    return safe_mean([abs(value - mean_value) for value in values])


def safe_pstdev(values: list[float]) -> float:
    if len(values) < 2:
        return math.nan
    return statistics.pstdev(values)


def kendall_tau_b(x: list[float], y: list[float]) -> float:
    """Kendall tau-b rank correlation between paired sequences x and y."""
    n = len(x)
    if n < 2:
        return math.nan
    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0 and dy == 0:
                pass  # joint tie — not counted anywhere
            elif dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1
    n0 = concordant + discordant
    denom = math.sqrt((n0 + ties_x) * (n0 + ties_y))
    if denom == 0:
        return math.nan
    return (concordant - discordant) / denom


def tau_to_score(tau: float) -> float:
    """Return raw Kendall tau-b value; caller scales axis as needed."""
    return tau


def spearman_rho(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation between paired sequences x and y.

    Uses average ranks for ties.  Returns nan if n < 2 or variance is zero.
    """
    n = len(x)
    if n < 2:
        return math.nan

    def avg_ranks(vals: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and vals[order[j]] == vals[order[i]]:
                j += 1
            avg = (i + j + 1) / 2.0  # 1-based average rank
            for k in range(i, j):
                ranks[order[k]] = avg
            i = j
        return ranks

    rx = avg_ranks(x)
    ry = avg_ranks(y)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    denom = math.sqrt(
        sum((r - mean_rx) ** 2 for r in rx) * sum((r - mean_ry) ** 2 for r in ry)
    )
    if denom == 0:
        return math.nan
    return num / denom


def pearson_r(x: list[float], y: list[float]) -> float:
    """Pearson correlation between paired sequences x and y."""
    n = len(x)
    if n < 2:
        return math.nan
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denom = math.sqrt(
        sum((v - mean_x) ** 2 for v in x) * sum((v - mean_y) ** 2 for v in y)
    )
    if denom == 0:
        return math.nan
    return num / denom


def participant_id(row: dict[str, str]) -> str | None:
    for key in ("empirica_id", "worker_id", "sender_id"):
        value = (row.get(key) or "").strip()
        if value:
            return value
    return None


def group_id(row: dict[str, str], default_group: str) -> str:
    value = (row.get("group_id") or "").strip()
    return value if value else default_group


def row_key(row: dict[str, str], idx: int) -> str:
    parts: list[str] = []
    for field in LIKERT_KEY_FIELDS:
        if field not in row:
            continue
        value = (row.get(field) or "").strip()
        if value:
            parts.append(f"{field}={value}")
    if not parts:
        return f"__index__={idx}"
    return "|".join(parts)


def align_rows(
    human_rows: list[dict[str, str]],
    llm_rows: list[dict[str, str]],
    min_key_match_rate: float,
) -> tuple[list[tuple[dict[str, str], dict[str, str]]], str, float]:
    buckets: dict[str, deque[dict[str, str]]] = defaultdict(deque)
    for i, row in enumerate(llm_rows):
        buckets[row_key(row, i)].append(row)

    pairs: list[tuple[dict[str, str], dict[str, str]]] = []
    key_matches = 0
    for i, hrow in enumerate(human_rows):
        key = row_key(hrow, i)
        if buckets[key]:
            pairs.append((hrow, buckets[key].popleft()))
            key_matches += 1

    key_match_rate = key_matches / max(1, len(human_rows))
    if key_match_rate < min_key_match_rate:
        n = min(len(human_rows), len(llm_rows))
        return [(human_rows[i], llm_rows[i]) for i in range(n)], "index", key_match_rate
    return pairs, "key", key_match_rate


WORD_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    curr = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = prev[j] if prev[j] >= curr[j - 1] else curr[j - 1]
        prev, curr = curr, prev
    return prev[-1]


def rouge_l_f1(reference: str, prediction: str) -> float:
    ref_tokens = tokenize(reference)
    pred_tokens = tokenize(prediction)
    if not ref_tokens or not pred_tokens:
        return 0.0
    lcs = lcs_length(ref_tokens, pred_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    denom = precision + recall
    if denom == 0:
        return 0.0
    return 2.0 * precision * recall / denom


def get_sidecar_csv_path(csv_path: Path, data_roots: list[Path], sidecar_root: Path, suffix: str) -> Path | None:
    for data_root in data_roots:
        try:
            relative_path = csv_path.relative_to(data_root)
        except ValueError:
            continue
        sidecar_path = sidecar_root / relative_path
        return sidecar_path.with_name(sidecar_path.stem + suffix)
    return None


def load_sidecar_validity_overrides(
    csv_path: Path,
    data_roots: list[Path],
    sidecar_roots: list[Path],
    suffix: str,
) -> dict[int, str]:
    overrides: dict[int, str] = {}
    if not sidecar_roots:
        return overrides

    for sidecar_root in sidecar_roots:
        sidecar_path = get_sidecar_csv_path(csv_path, data_roots, sidecar_root, suffix)
        if sidecar_path is None or not sidecar_path.exists():
            continue
        with sidecar_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                row_index = str(row.get("row_index") or "").strip()
                new_validity = str(row.get("new_validity") or "").strip().upper()
                if not row_index or not new_validity:
                    continue
                overrides[int(row_index)] = new_validity
    return overrides


def determine_validity_column(path: Path, row: dict[str, str]) -> str | None:
    if "opinion_llm_" in path.name and "llm_validity" in row:
        return "llm_validity"
    if "opinion_human_" in path.name and "human_validity" in row:
        return "human_validity"
    if "validity" in row:
        return "validity"
    return None


def apply_validity_overrides(
    rows: list[dict[str, str]],
    *,
    path: Path,
    likert_col: str,
    data_roots: list[Path],
    initial_post_sidecar_roots: list[Path],
    tweet_sidecar_roots: list[Path],
) -> list[dict[str, str]]:
    if not path.name.startswith("opinion_"):
        return rows

    initial_post_overrides = load_sidecar_validity_overrides(
        path,
        data_roots,
        initial_post_sidecar_roots,
        "__initial_post_rejudge.csv",
    )
    tweet_overrides = load_sidecar_validity_overrides(
        path,
        data_roots,
        tweet_sidecar_roots,
        "__tweet_rejudge.csv",
    )

    adjusted_rows: list[dict[str, str]] = []
    for row_index, row in enumerate(rows, start=2):
        event_type = str(row.get("event_type") or "").strip()
        validity_column = determine_validity_column(path, row)
        effective_validity = str(row.get(validity_column) or row.get("validity") or "").strip().upper() if validity_column else ""

        if event_type in {"Initial Opinion", "Post Opinion"} and row_index in initial_post_overrides:
            effective_validity = initial_post_overrides[row_index]
        elif event_type == "tweet" and row_index in tweet_overrides:
            effective_validity = tweet_overrides[row_index]

        if effective_validity == "INVALID":
            adjusted_row = dict(row)
            adjusted_row[likert_col] = ""
            adjusted_rows.append(adjusted_row)
            continue

        adjusted_rows.append(row)

    return adjusted_rows


def read_csv_rows(
    path: Path,
    *,
    respect_validity_labels: bool = False,
    likert_col: str | None = None,
    data_roots: list[Path] | None = None,
    initial_post_sidecar_roots: list[Path] | None = None,
    tweet_sidecar_roots: list[Path] | None = None,
    apply_v3_additional_group_masks: bool = True,
) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not apply_v3_additional_group_masks:
        if not respect_validity_labels or not likert_col:
            return rows
        return apply_validity_overrides(
            rows,
            path=path,
            likert_col=likert_col,
            data_roots=data_roots or [],
            initial_post_sidecar_roots=initial_post_sidecar_roots or [],
            tweet_sidecar_roots=tweet_sidecar_roots or [],
        )
    if not respect_validity_labels or not likert_col:
        return filter_rows_for_v3_complete_group_mask(rows, csv_path=path)
    adjusted_rows = apply_validity_overrides(
        rows,
        path=path,
        likert_col=likert_col,
        data_roots=data_roots or [],
        initial_post_sidecar_roots=initial_post_sidecar_roots or [],
        tweet_sidecar_roots=tweet_sidecar_roots or [],
    )
    return filter_rows_for_v3_complete_group_mask(adjusted_rows, csv_path=path)


def resolve_file(
    roots: list[Path],
    data_prefix: str,
    model: str,
    filename: str,
    allow_without_model: bool,
) -> Path | None:
    for root in roots:
        model_path = root / data_prefix / model / filename
        if model_path.exists():
            return model_path
        if allow_without_model:
            no_model_path = root / data_prefix / filename
            if no_model_path.exists():
                return no_model_path
    return None


def build_profile(rows: list[dict[str, str]], default_group: str, likert_col: str) -> Profile:
    profile = Profile()
    tweet_values: dict[str, dict[str, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for row in rows:
        pid = participant_id(row)
        if not pid:
            continue
        stance = parse_likert(row.get(likert_col))
        if stance is None:
            continue
        gid = group_id(row, default_group)
        event_type = (row.get("event_type") or "").strip()

        if event_type == "Initial Opinion":
            profile.initial_private[gid][pid] = stance
        elif event_type == "Post Opinion":
            profile.final_private[gid][pid] = stance
        elif event_type == "tweet":
            round_idx = parse_round(row.get("chat_round_order"))
            if round_idx in (1, 2, 3):
                tweet_values[gid][pid][round_idx].append(stance)

    for gid, participants in tweet_values.items():
        for pid, rounds in participants.items():
            for round_idx, vals in rounds.items():
                if vals:
                    profile.tweet_round[gid][pid][round_idx] = sum(vals) / len(vals)

    return profile


def compute_group_deltas(profile: Profile) -> tuple[dict[str, float], dict[str, float]]:
    delta_public: dict[str, float] = {}
    delta_private: dict[str, float] = {}

    groups = set(profile.initial_private.keys()) | set(profile.final_private.keys()) | set(profile.tweet_round.keys())
    for gid in groups:
        init = profile.initial_private.get(gid, {})
        final = profile.final_private.get(gid, {})
        tweet = profile.tweet_round.get(gid, {})

        private_participants = sorted(set(init.keys()) & set(final.keys()))
        if len(private_participants) >= 2:
            init_vals = [init[p] for p in private_participants]
            final_vals = [final[p] for p in private_participants]
            init_sd = safe_pstdev(init_vals)
            final_sd = safe_pstdev(final_vals)
            if not math.isnan(init_sd) and not math.isnan(final_sd):
                delta_private[gid] = final_sd - init_sd

        tweet_participants = [p for p, rounds in tweet.items() if 1 in rounds and 3 in rounds]
        if len(tweet_participants) >= 2:
            round1_vals = [tweet[p][1] for p in tweet_participants]
            round3_vals = [tweet[p][3] for p in tweet_participants]
            round1_sd = safe_pstdev(round1_vals)
            round3_sd = safe_pstdev(round3_vals)
            if not math.isnan(round1_sd) and not math.isnan(round3_sd):
                delta_public[gid] = round3_sd - round1_sd

    return delta_public, delta_private


def macro_f1_fixed_labels(y_true: list[int], y_pred: list[int], labels: tuple[int, ...]) -> float:
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return math.nan

    f1s: list[float] = []
    for label in labels:
        tp = 0
        fp = 0
        fn = 0
        for yt, yp in zip(y_true, y_pred):
            if yt == label and yp == label:
                tp += 1
            elif yt != label and yp == label:
                fp += 1
            elif yt == label and yp != label:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        f1s.append(f1)
    return sum(f1s) / len(f1s)


def direction_accuracy(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return math.nan
    valid_pairs = [(yt, yp) for yt, yp in zip(y_true, y_pred) if yt != 0]
    if not valid_pairs:
        return math.nan
    correct = sum(1 for yt, yp in valid_pairs if yt == yp)
    return correct / len(valid_pairs)


def deterministic_shuffle(values: list[float], *, data_prefix: str, model: str, metric_name: str) -> list[float]:
    shuffled = list(values)
    seed_text = f"{data_prefix}|{model}|{metric_name}"
    seed = int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    return shuffled


def deterministic_ordinal_jitter(
    values: list[float], *, data_prefix: str, model: str, metric_name: str,
    likert_min: float = 1.0, likert_max: float = 6.0,
) -> list[float]:
    """Apply deterministic center-weighted ±1 ordinal jitter to Likert values.

    Each value is shifted by −1, 0, or +1 with probabilities (1/4, 1/2, 1/4)
    and clamped to [likert_min, likert_max].  Models a predictor that is exact
    ~50% of the time and off by one point the other 50%.
    """
    seed_text = f"{data_prefix}|{model}|{metric_name}|ordinal_jitter_v1"
    seed = int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    weights = [-1, 0, 0, 1]  # centre-weighted: 0 appears twice → P(0)=.5, P(±1)=.25 each
    return [max(likert_min, min(likert_max, v + rng.choice(weights))) for v in values]


def deterministic_gaussian_noise(
    values: list[float], *, data_prefix: str, model: str, metric_name: str,
    sigma: float, lower_bound: float | None = None, upper_bound: float | None = None,
) -> list[float]:
    """Apply deterministic Gaussian noise with optional clipping.

    This is used for the Noisy UB reference rows. Unlike ordinal jitter, it can
    be applied to derived quantities such as change and convergence values.
    """
    if not values:
        return []
    seed_text = f"{data_prefix}|{model}|{metric_name}|gaussian_noise_v1"
    seed = int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    noisy: list[float] = []
    for value in values:
        perturbed = value + rng.gauss(0.0, sigma)
        if lower_bound is not None:
            perturbed = max(lower_bound, perturbed)
        if upper_bound is not None:
            perturbed = min(upper_bound, perturbed)
        noisy.append(perturbed)
    return noisy


def deterministic_gaussian_likert_rows(
    rows: list[dict[str, str]], *, data_prefix: str, likert_col: str, sigma: float,
) -> list[dict[str, str]]:
    """Create deterministic noisy Likert labels with a single shared sigma.

    The noise is injected at the discrete 1..6 labeling stage, then all
    alignment/change/convergence metrics are derived from those noisy labels.
    """
    noisy_rows: list[dict[str, str]] = []
    for idx, row in enumerate(rows):
        noisy_row = dict(row)
        stance = parse_likert(row.get(likert_col))
        if stance is None:
            noisy_rows.append(noisy_row)
            continue
        row_id = row_key(row, idx)
        noisy_value = deterministic_gaussian_noise(
            [stance],
            data_prefix=data_prefix,
            model="__likert__",
            metric_name=f"likert_label|{row_id}|{idx}",
            sigma=sigma,
            lower_bound=1.0,
            upper_bound=6.0,
        )[0]
        rounded_value = min(6, max(1, int(math.floor(noisy_value + 0.5))))
        noisy_row[likert_col] = str(rounded_value)
        noisy_rows.append(noisy_row)
    return noisy_rows


def deterministic_uniform_likert_rows(
    rows: list[dict[str, str]], *, data_prefix: str, likert_col: str,
) -> list[dict[str, str]]:
    uniform_rows: list[dict[str, str]] = []
    for idx, row in enumerate(rows):
        uniform_row = dict(row)
        stance = parse_likert(row.get(likert_col))
        if stance is None:
            uniform_rows.append(uniform_row)
            continue
        row_id = row_key(row, idx)
        seed_text = f"{data_prefix}|uniform_likert|{row_id}|{idx}"
        seed = int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest()[:8], "big")
        rng = random.Random(seed)
        uniform_row[likert_col] = str(rng.randint(1, 6))
        uniform_rows.append(uniform_row)
    return uniform_rows


def compute_individual_deltas(
    human_profile: Profile,
    llm_profile: Profile,
) -> tuple[list[float], list[float], list[float], list[float]]:
    public_h_deltas: list[float] = []
    public_l_deltas: list[float] = []
    private_h_deltas: list[float] = []
    private_l_deltas: list[float] = []

    groups = (
        set(human_profile.initial_private.keys())
        | set(human_profile.final_private.keys())
        | set(human_profile.tweet_round.keys())
        | set(llm_profile.final_private.keys())
        | set(llm_profile.tweet_round.keys())
    )

    for gid in groups:
        h_tweets = human_profile.tweet_round.get(gid, {})
        l_tweets = llm_profile.tweet_round.get(gid, {})
        participants_for_public = sorted(set(h_tweets.keys()) & set(l_tweets.keys()))
        for pid in participants_for_public:
            h_rounds = h_tweets[pid]
            l_rounds = l_tweets[pid]
            if not all(r in h_rounds for r in (1, 3)):
                continue
            if not all(r in l_rounds for r in (1, 3)):
                continue

            h_delta_public = h_rounds[3] - h_rounds[1]
            l_delta_public = l_rounds[3] - l_rounds[1]
            public_h_deltas.append(h_delta_public)
            public_l_deltas.append(l_delta_public)

        h_init = human_profile.initial_private.get(gid, {})
        h_final = human_profile.final_private.get(gid, {})
        l_init = llm_profile.initial_private.get(gid, {})
        l_final = llm_profile.final_private.get(gid, {})
        participants_for_private = sorted(set(h_init.keys()) & set(h_final.keys()) & set(l_init.keys()) & set(l_final.keys()))
        for pid in participants_for_private:
            h_delta_private = h_final[pid] - h_init[pid]
            l_delta_private = l_final[pid] - l_init[pid]
            private_h_deltas.append(h_delta_private)
            private_l_deltas.append(l_delta_private)

    return public_h_deltas, public_l_deltas, private_h_deltas, private_l_deltas


def compute_prefix_model_from_rows(
    *,
    data_prefix: str,
    model: str,
    human_rows: list[dict[str, str]],
    llm_rows: list[dict[str, str]],
    utterance_event_types: set[str],
    likert_col: str,
    min_key_match_rate: float,
    include_conditioned_alignment: bool,
    human_path: str = "",
    llm_path: str = "",
) -> PrefixModelResult:
    pairs, alignment_mode, key_match_rate = align_rows(human_rows, llm_rows, min_key_match_rate)
    result = PrefixModelResult(
        status="ok",
        data_prefix=data_prefix,
        model=model,
        message="ok",
        human_path=human_path,
        llm_path=llm_path,
        alignment_mode=alignment_mode,
        key_match_rate=key_match_rate,
        aligned_pairs=len(pairs),
    )

    human_row_indices = {id(row): idx for idx, row in enumerate(human_rows)}
    llm_row_indices = {id(row): idx for idx, row in enumerate(llm_rows)}
    noisy_human_rows = deterministic_gaussian_likert_rows(
        human_rows,
        data_prefix=data_prefix,
        likert_col=likert_col,
        sigma=LIKERT_GAUSSIAN_SIGMA,
    )
    noisy_sigma_half_human_rows = deterministic_gaussian_likert_rows(
        human_rows,
        data_prefix=data_prefix,
        likert_col=likert_col,
        sigma=LIKERT_GAUSSIAN_SIGMA_SECONDARY,
    )
    uniform_human_rows = deterministic_uniform_likert_rows(
        human_rows,
        data_prefix=data_prefix,
        likert_col=likert_col,
    )
    noisy_proxy_rows = [{**row, likert_col: ""} for row in llm_rows]
    noisy_sigma_half_proxy_rows = [{**row, likert_col: ""} for row in llm_rows]
    uniform_proxy_rows = [{**row, likert_col: ""} for row in llm_rows]

    stance_h: list[float] = []
    stance_l: list[float] = []
    public_h: list[float] = []
    public_l: list[float] = []
    private_h: list[float] = []
    private_l: list[float] = []
    ub_public_h: list[float] = []
    ub_public_l: list[float] = []
    ub_private_h: list[float] = []
    ub_private_l: list[float] = []
    ub_sigma_half_public_h: list[float] = []
    ub_sigma_half_public_l: list[float] = []
    ub_sigma_half_private_h: list[float] = []
    ub_sigma_half_private_l: list[float] = []
    uniform_public_h: list[float] = []
    uniform_public_l: list[float] = []
    uniform_private_h: list[float] = []
    uniform_private_l: list[float] = []

    for hrow, lrow in pairs:
        event_type = (hrow.get("event_type") or "").strip()
        if utterance_event_types and event_type not in utterance_event_types:
            continue

        human_idx = human_row_indices.get(id(hrow))
        llm_idx = llm_row_indices.get(id(lrow))
        noisy_h_stance = None
        noisy_sigma_half_h_stance = None
        uniform_h_stance = None
        if human_idx is not None and llm_idx is not None:
            noisy_proxy_rows[llm_idx][likert_col] = noisy_human_rows[human_idx].get(likert_col, "")
            noisy_sigma_half_proxy_rows[llm_idx][likert_col] = noisy_sigma_half_human_rows[human_idx].get(likert_col, "")
            uniform_proxy_rows[llm_idx][likert_col] = uniform_human_rows[human_idx].get(likert_col, "")
            noisy_h_stance = parse_likert(noisy_human_rows[human_idx].get(likert_col))
            noisy_sigma_half_h_stance = parse_likert(noisy_sigma_half_human_rows[human_idx].get(likert_col))
            uniform_h_stance = parse_likert(uniform_human_rows[human_idx].get(likert_col))

        h_stance = parse_likert(hrow.get(likert_col))
        l_stance = parse_likert(lrow.get(likert_col))
        if h_stance is not None and l_stance is not None:
            stance_h.append(h_stance)
            stance_l.append(l_stance)
            result.stance_sum += abs(h_stance - l_stance)
            result.stance_count += 1
            if event_type == "tweet":
                round_idx = parse_round(hrow.get("chat_round_order"))
                if round_idx is not None and (include_conditioned_alignment or round_idx >= 2):
                    public_h.append(h_stance)
                    public_l.append(l_stance)
                    if noisy_h_stance is not None:
                        ub_public_h.append(h_stance)
                        ub_public_l.append(noisy_h_stance)
                    if noisy_sigma_half_h_stance is not None:
                        ub_sigma_half_public_h.append(h_stance)
                        ub_sigma_half_public_l.append(noisy_sigma_half_h_stance)
                    if uniform_h_stance is not None:
                        uniform_public_h.append(h_stance)
                        uniform_public_l.append(uniform_h_stance)
                        result.uniform_public_alignment_abs_sum += abs(uniform_h_stance - h_stance)
                        result.uniform_public_alignment_count += 1
                    alignment_error = l_stance - h_stance
                    result.public_alignment_sum += alignment_error
                    result.public_alignment_abs_sum += abs(alignment_error)
                    result.public_alignment_count += 1
            elif event_type in ({"Initial Opinion", "Post Opinion"} if include_conditioned_alignment else {"Post Opinion"}):
                private_h.append(h_stance)
                private_l.append(l_stance)
                if noisy_h_stance is not None:
                    ub_private_h.append(h_stance)
                    ub_private_l.append(noisy_h_stance)
                if noisy_sigma_half_h_stance is not None:
                    ub_sigma_half_private_h.append(h_stance)
                    ub_sigma_half_private_l.append(noisy_sigma_half_h_stance)
                if uniform_h_stance is not None:
                    uniform_private_h.append(h_stance)
                    uniform_private_l.append(uniform_h_stance)
                    result.uniform_private_alignment_abs_sum += abs(uniform_h_stance - h_stance)
                    result.uniform_private_alignment_count += 1
                alignment_error = l_stance - h_stance
                result.private_alignment_sum += alignment_error
                result.private_alignment_abs_sum += abs(alignment_error)
                result.private_alignment_count += 1

    if stance_h and stance_l:
        stance_l_shuffled = deterministic_shuffle(
            stance_l, data_prefix=data_prefix, model=model, metric_name="stance_difference"
        )
        result.lb_stance_sum = sum(abs(h - l) for h, l in zip(stance_h, stance_l_shuffled))
        result.lb_stance_count = len(stance_h)

    if public_h and public_l:
        public_l_shuffled = deterministic_shuffle(
            public_l, data_prefix=data_prefix, model=model, metric_name="public_stance_alignment"
        )
        shuffled_errors = [l - h for h, l in zip(public_h, public_l_shuffled)]
        result.lb_public_alignment_sum = sum(shuffled_errors)
        result.lb_public_alignment_abs_sum = sum(abs(err) for err in shuffled_errors)
        result.lb_public_alignment_count = len(public_h)

    if private_h and private_l:
        private_l_shuffled = deterministic_shuffle(
            private_l, data_prefix=data_prefix, model=model, metric_name="private_stance_alignment"
        )
        shuffled_errors = [l - h for h, l in zip(private_h, private_l_shuffled)]
        result.lb_private_alignment_sum = sum(shuffled_errors)
        result.lb_private_alignment_abs_sum = sum(abs(err) for err in shuffled_errors)
        result.lb_private_alignment_count = len(private_h)

    result.public_align_h_vals.extend(public_h)
    result.public_align_l_vals.extend(public_l)
    result.private_align_h_vals.extend(private_h)
    result.private_align_l_vals.extend(private_l)
    result.ub_public_align_h_vals.extend(ub_public_h)
    result.ub_public_align_l_vals.extend(ub_public_l)
    result.ub_private_align_h_vals.extend(ub_private_h)
    result.ub_private_align_l_vals.extend(ub_private_l)
    result.ub_sigma_half_public_align_h_vals.extend(ub_sigma_half_public_h)
    result.ub_sigma_half_public_align_l_vals.extend(ub_sigma_half_public_l)
    result.ub_sigma_half_private_align_h_vals.extend(ub_sigma_half_private_h)
    result.ub_sigma_half_private_align_l_vals.extend(ub_sigma_half_private_l)

    human_profile = build_profile(human_rows, default_group=data_prefix, likert_col=likert_col)
    llm_profile = build_profile(llm_rows, default_group=data_prefix, likert_col=likert_col)
    noisy_proxy_profile = build_profile(noisy_proxy_rows, default_group=data_prefix, likert_col=likert_col)
    noisy_sigma_half_proxy_profile = build_profile(noisy_sigma_half_proxy_rows, default_group=data_prefix, likert_col=likert_col)
    uniform_proxy_profile = build_profile(uniform_proxy_rows, default_group=data_prefix, likert_col=likert_col)

    human_public_delta, human_private_delta = compute_group_deltas(human_profile)
    llm_public_delta, llm_private_delta = compute_group_deltas(llm_profile)
    ub_public_delta, ub_private_delta = compute_group_deltas(noisy_proxy_profile)
    ub_sigma_half_public_delta, ub_sigma_half_private_delta = compute_group_deltas(noisy_sigma_half_proxy_profile)
    uniform_public_delta, uniform_private_delta = compute_group_deltas(uniform_proxy_profile)

    for gid in sorted(set(human_public_delta.keys()) & set(llm_public_delta.keys())):
        result.public_conv_errors.append(abs(llm_public_delta[gid] - human_public_delta[gid]))
    for gid in sorted(set(human_private_delta.keys()) & set(llm_private_delta.keys())):
        result.private_conv_errors.append(abs(llm_private_delta[gid] - human_private_delta[gid]))

    public_overlap_gids = sorted(set(human_public_delta.keys()) & set(llm_public_delta.keys()))
    if public_overlap_gids:
        h_public_vals = [human_public_delta[gid] for gid in public_overlap_gids]
        l_public_vals = [llm_public_delta[gid] for gid in public_overlap_gids]
        result.public_conv_h_vals.extend(h_public_vals)
        result.public_conv_l_vals.extend(l_public_vals)

    private_overlap_gids = sorted(set(human_private_delta.keys()) & set(llm_private_delta.keys()))
    if private_overlap_gids:
        h_private_vals = [human_private_delta[gid] for gid in private_overlap_gids]
        l_private_vals = [llm_private_delta[gid] for gid in private_overlap_gids]
        result.private_conv_h_vals.extend(h_private_vals)
        result.private_conv_l_vals.extend(l_private_vals)

    ub_public_overlap_gids = sorted(set(human_public_delta.keys()) & set(ub_public_delta.keys()))
    if ub_public_overlap_gids:
        result.ub_public_conv_h_vals.extend([human_public_delta[gid] for gid in ub_public_overlap_gids])
        result.ub_public_conv_l_vals.extend([ub_public_delta[gid] for gid in ub_public_overlap_gids])

    ub_private_overlap_gids = sorted(set(human_private_delta.keys()) & set(ub_private_delta.keys()))
    if ub_private_overlap_gids:
        result.ub_private_conv_h_vals.extend([human_private_delta[gid] for gid in ub_private_overlap_gids])
        result.ub_private_conv_l_vals.extend([ub_private_delta[gid] for gid in ub_private_overlap_gids])

    ub_sigma_half_public_overlap_gids = sorted(set(human_public_delta.keys()) & set(ub_sigma_half_public_delta.keys()))
    if ub_sigma_half_public_overlap_gids:
        result.ub_sigma_half_public_conv_h_vals.extend([human_public_delta[gid] for gid in ub_sigma_half_public_overlap_gids])
        result.ub_sigma_half_public_conv_l_vals.extend([ub_sigma_half_public_delta[gid] for gid in ub_sigma_half_public_overlap_gids])

    ub_sigma_half_private_overlap_gids = sorted(set(human_private_delta.keys()) & set(ub_sigma_half_private_delta.keys()))
    if ub_sigma_half_private_overlap_gids:
        result.ub_sigma_half_private_conv_h_vals.extend([human_private_delta[gid] for gid in ub_sigma_half_private_overlap_gids])
        result.ub_sigma_half_private_conv_l_vals.extend([ub_sigma_half_private_delta[gid] for gid in ub_sigma_half_private_overlap_gids])

    for gid in sorted(set(human_public_delta.keys()) & set(uniform_public_delta.keys())):
        result.uniform_public_conv_errors.append(abs(uniform_public_delta[gid] - human_public_delta[gid]))
    for gid in sorted(set(human_private_delta.keys()) & set(uniform_private_delta.keys())):
        result.uniform_private_conv_errors.append(abs(uniform_private_delta[gid] - human_private_delta[gid]))

    public_h_deltas, public_l_deltas, private_h_deltas, private_l_deltas = compute_individual_deltas(
        human_profile,
        llm_profile,
    )
    ub_public_h_deltas, ub_public_l_deltas, ub_private_h_deltas, ub_private_l_deltas = compute_individual_deltas(
        human_profile,
        noisy_proxy_profile,
    )
    ub_sigma_half_public_h_deltas, ub_sigma_half_public_l_deltas, ub_sigma_half_private_h_deltas, ub_sigma_half_private_l_deltas = compute_individual_deltas(
        human_profile,
        noisy_sigma_half_proxy_profile,
    )
    uniform_public_h_deltas, uniform_public_l_deltas, uniform_private_h_deltas, uniform_private_l_deltas = compute_individual_deltas(
        human_profile,
        uniform_proxy_profile,
    )
    public_change_errors = [h - l for h, l in zip(public_h_deltas, public_l_deltas)]
    private_change_errors = [h - l for h, l in zip(private_h_deltas, private_l_deltas)]
    public_change_abs_errors = [abs(err) for err in public_change_errors]
    private_change_abs_errors = [abs(err) for err in private_change_errors]
    direction_true = [sign_label(h) for h in private_h_deltas]
    direction_pred = [sign_label(l) for l in private_l_deltas]
    result.public_change_errors.extend(public_change_errors)
    result.public_change_abs_errors.extend(public_change_abs_errors)
    result.private_change_errors.extend(private_change_errors)
    result.private_change_abs_errors.extend(private_change_abs_errors)
    result.direction_true.extend(direction_true)
    result.direction_pred.extend(direction_pred)

    if public_h_deltas and public_l_deltas:
        public_l_deltas_shuffled = deterministic_shuffle(
            public_l_deltas, data_prefix=data_prefix, model=model, metric_name="public_change_error"
        )
        lb_public_errors = [h - l for h, l in zip(public_h_deltas, public_l_deltas_shuffled)]
        result.lb_public_change_errors.extend(lb_public_errors)
        result.lb_public_change_abs_errors.extend(abs(err) for err in lb_public_errors)

    if private_h_deltas and private_l_deltas:
        private_l_deltas_shuffled = deterministic_shuffle(
            private_l_deltas, data_prefix=data_prefix, model=model, metric_name="private_change_error"
        )
        lb_private_errors = [h - l for h, l in zip(private_h_deltas, private_l_deltas_shuffled)]
        result.lb_private_change_errors.extend(lb_private_errors)
        result.lb_private_change_abs_errors.extend(abs(err) for err in lb_private_errors)

        result.lb_direction_true.extend(sign_label(h) for h in private_h_deltas)
        result.lb_direction_pred.extend(sign_label(l) for l in private_l_deltas_shuffled)

    result.public_change_h_vals.extend(public_h_deltas)
    result.public_change_l_vals.extend(public_l_deltas)
    result.private_change_h_vals.extend(private_h_deltas)
    result.private_change_l_vals.extend(private_l_deltas)
    result.ub_public_change_h_vals.extend(ub_public_h_deltas)
    result.ub_public_change_l_vals.extend(ub_public_l_deltas)
    result.ub_private_change_h_vals.extend(ub_private_h_deltas)
    result.ub_private_change_l_vals.extend(ub_private_l_deltas)
    result.ub_sigma_half_public_change_h_vals.extend(ub_sigma_half_public_h_deltas)
    result.ub_sigma_half_public_change_l_vals.extend(ub_sigma_half_public_l_deltas)
    result.ub_sigma_half_private_change_h_vals.extend(ub_sigma_half_private_h_deltas)
    result.ub_sigma_half_private_change_l_vals.extend(ub_sigma_half_private_l_deltas)
    result.uniform_public_change_abs_errors.extend(
        abs(h - l) for h, l in zip(uniform_public_h_deltas, uniform_public_l_deltas)
    )
    result.uniform_private_change_abs_errors.extend(
        abs(h - l) for h, l in zip(uniform_private_h_deltas, uniform_private_l_deltas)
    )

    return result


def compute_prefix_model(
    data_prefix: str,
    model: str,
    roots: list[Path],
    human_filename: str,
    llm_filename: str,
    utterance_event_types: set[str],
    likert_col: str,
    human_text_col: str,
    llm_text_col: str,
    min_key_match_rate: float,
    include_conditioned_alignment: bool,
    respect_validity_labels: bool,
    initial_post_sidecar_roots: list[Path],
    tweet_sidecar_roots: list[Path],
    apply_v3_additional_group_masks: bool = True,
) -> PrefixModelResult:
    human_path = resolve_file(roots, data_prefix, model, human_filename, allow_without_model=True)
    llm_path = resolve_file(roots, data_prefix, model, llm_filename, allow_without_model=False)

    if human_path is None:
        return PrefixModelResult(
            status="missing_human",
            data_prefix=data_prefix,
            model=model,
            message="Human CSV not found in configured roots.",
        )
    if llm_path is None:
        return PrefixModelResult(
            status="missing_llm",
            data_prefix=data_prefix,
            model=model,
            message="LLM CSV not found in configured roots.",
            human_path=str(human_path),
        )

    try:
        human_rows = read_csv_rows(
            human_path,
            respect_validity_labels=respect_validity_labels,
            likert_col=likert_col,
            data_roots=roots,
            initial_post_sidecar_roots=initial_post_sidecar_roots,
            tweet_sidecar_roots=tweet_sidecar_roots,
            apply_v3_additional_group_masks=apply_v3_additional_group_masks,
        )
        llm_rows = read_csv_rows(
            llm_path,
            respect_validity_labels=respect_validity_labels,
            likert_col=likert_col,
            data_roots=roots,
            initial_post_sidecar_roots=initial_post_sidecar_roots,
            tweet_sidecar_roots=tweet_sidecar_roots,
            apply_v3_additional_group_masks=apply_v3_additional_group_masks,
        )
    except Exception as exc:
        return PrefixModelResult(
            status="csv_error",
            data_prefix=data_prefix,
            model=model,
            message=f"Failed to parse CSV: {exc}",
            human_path=str(human_path),
            llm_path=str(llm_path),
        )

    return compute_prefix_model_from_rows(
        data_prefix=data_prefix,
        model=model,
        human_rows=human_rows,
        llm_rows=llm_rows,
        utterance_event_types=utterance_event_types,
        likert_col=likert_col,
        min_key_match_rate=min_key_match_rate,
        include_conditioned_alignment=include_conditioned_alignment,
        human_path=str(human_path),
        llm_path=str(llm_path),
    )


def normalize_minmax(values_by_model: dict[str, float], higher_is_better: bool) -> dict[str, float]:
    valid_items = [(model, value) for model, value in values_by_model.items() if not math.isnan(value)]
    if not valid_items:
        return {model: math.nan for model in values_by_model}

    vals = [value for _, value in valid_items]
    lo = min(vals)
    hi = max(vals)
    if math.isclose(lo, hi):
        return {model: (1.0 if not math.isnan(value) else math.nan) for model, value in values_by_model.items()}

    scores: dict[str, float] = {}
    for model, value in values_by_model.items():
        if math.isnan(value):
            scores[model] = math.nan
            continue
        if higher_is_better:
            score = (value - lo) / (hi - lo)
        else:
            score = (hi - value) / (hi - lo)
        scores[model] = min(1.0, max(0.0, score))
    return scores


def save_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_fieldnames(rows: list[dict[str, Any]], *, default: str = "model") -> list[str]:
    if not rows:
        return [default]
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames or [default]


def as_float_or_nan(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    maybe = maybe_float(value)
    if maybe is None:
        return math.nan
    return maybe


def append_human_baseline_raw_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if any(str(row.get("model", "")).strip().lower() == "human" for row in raw_rows):
        return raw_rows

    baseline_row = {
        "model": "human",
        "prefixes_ok": 0,
        "prefixes_missing": 0,
        "stance_difference": 0.0,
        "public_stance_alignment": 0.0,
        "public_stance_alignment_abs": 0.0,
        "private_stance_alignment": 0.0,
        "private_stance_alignment_abs": 0.0,
        "public_opinion_convergence_error": 0.0,
        "private_opinion_convergence_error": 0.0,
        "public_stance_change_error": 0.0,
        "public_stance_change_error_abs": 0.0,
        "private_stance_change_error": 0.0,
        "private_stance_change_error_abs": 0.0,
        "stance_update_direction_accuracy": 1.0,
        "lb_stance_difference": math.nan,
        "lb_public_stance_alignment": math.nan,
        "lb_public_stance_alignment_abs": math.nan,
        "lb_private_stance_alignment": math.nan,
        "lb_private_stance_alignment_abs": math.nan,
        "lb_public_opinion_convergence_error": math.nan,
        "lb_private_opinion_convergence_error": math.nan,
        "lb_public_stance_change_error": math.nan,
        "lb_public_stance_change_error_abs": math.nan,
        "lb_private_stance_change_error": math.nan,
        "lb_private_stance_change_error_abs": math.nan,
        "lb_stance_update_direction_accuracy": math.nan,
        "stance_count": 0,
        "public_alignment_count": 0,
        "private_alignment_count": 0,
        "public_group_count": 0,
        "private_group_count": 0,
        "public_change_count": 0,
        "private_change_count": 0,
        "direction_count": 0,
        "lb_stance_count": 0,
        "lb_public_alignment_count": 0,
        "lb_private_alignment_count": 0,
        "lb_public_group_count": 0,
        "lb_private_group_count": 0,
        "lb_public_change_count": 0,
        "lb_private_change_count": 0,
        "lb_direction_count": 0,
        "public_alignment_tau": 1.0,
        "private_alignment_tau": 1.0,
        "public_change_error_tau": 1.0,
        "private_change_error_tau": 1.0,
        "public_convergence_tau": 1.0,
        "private_convergence_tau": 1.0,
        "lb_perm_public_alignment_tau": math.nan,
        "lb_perm_private_alignment_tau": math.nan,
        "lb_perm_public_change_error_tau": math.nan,
        "lb_perm_private_change_error_tau": math.nan,
        "lb_perm_public_convergence_tau": math.nan,
        "lb_perm_private_convergence_tau": math.nan,
        "ub_noise_public_alignment_tau": math.nan,
        "ub_noise_private_alignment_tau": math.nan,
        "ub_noise_public_change_error_tau": math.nan,
        "ub_noise_private_change_error_tau": math.nan,
        "ub_noise_public_convergence_tau": math.nan,
        "ub_noise_private_convergence_tau": math.nan,
        "public_alignment_rho": 1.0,
        "private_alignment_rho": 1.0,
        "public_change_error_rho": 1.0,
        "private_change_error_rho": 1.0,
        "public_convergence_rho": 1.0,
        "private_convergence_rho": 1.0,
        "lb_perm_public_alignment_rho": math.nan,
        "lb_perm_private_alignment_rho": math.nan,
        "lb_perm_public_change_error_rho": math.nan,
        "lb_perm_private_change_error_rho": math.nan,
        "lb_perm_public_convergence_rho": math.nan,
        "lb_perm_private_convergence_rho": math.nan,
        "ub_noise_public_alignment_rho": math.nan,
        "ub_noise_private_alignment_rho": math.nan,
        "ub_noise_public_change_error_rho": math.nan,
        "ub_noise_private_change_error_rho": math.nan,
        "ub_noise_public_convergence_rho": math.nan,
        "ub_noise_private_convergence_rho": math.nan,
        "public_alignment_pearson": 1.0,
        "private_alignment_pearson": 1.0,
        "public_change_error_pearson": 1.0,
        "private_change_error_pearson": 1.0,
        "public_convergence_pearson": 1.0,
        "private_convergence_pearson": 1.0,
        "lb_perm_public_alignment_pearson": math.nan,
        "lb_perm_private_alignment_pearson": math.nan,
        "lb_perm_public_change_error_pearson": math.nan,
        "lb_perm_private_change_error_pearson": math.nan,
        "lb_perm_public_convergence_pearson": math.nan,
        "lb_perm_private_convergence_pearson": math.nan,
        "ub_noise_public_alignment_pearson": math.nan,
        "ub_noise_private_alignment_pearson": math.nan,
        "ub_noise_public_change_error_pearson": math.nan,
        "ub_noise_private_change_error_pearson": math.nan,
        "ub_noise_public_convergence_pearson": math.nan,
        "ub_noise_private_convergence_pearson": math.nan,
    }
    return [*raw_rows, baseline_row]


def append_human_baseline_score_rows(
    score_rows: list[dict[str, Any]],
    axes: list[tuple[str, str]] | None = None,
    human_score: float = 1.0,
) -> list[dict[str, Any]]:
    if any(str(row.get("model", "")).strip().lower() == "human" for row in score_rows):
        return score_rows
    if axes is None:
        axes = RADAR_AXES
    baseline_row = {"model": "human"}
    for key, _ in axes:
        baseline_row[key] = human_score
    return [*score_rows, baseline_row]


def append_reference_score_rows_v1(
    score_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    metric_version: str = "v1",
    include_noisy_ub: bool = True,
) -> list[dict[str, Any]]:
    """Append Perm. LB and Ordinal Jitter UB reference lines to v1-family score rows for plotting."""
    special = {"human", PERM_LB_MODEL, MEAN_PREDICTOR_MODEL, UNIFORM_LB_MODEL}
    model_raw = [
        r for r in raw_rows
        if str(r.get("model", "")).strip().lower() not in {"human"}
        and str(r.get("model", "")) not in special
        and not str(r.get("model", "")).startswith(NOISY_UB_PREFIX)
    ]
    if not model_raw:
        return score_rows

    if metric_version == "v2":
        field_specs = [
            ("public_stance_alignment", "lb_public_stance_alignment_abs", "ub_noise_public_stance_alignment_abs"),
            ("private_stance_alignment", "lb_private_stance_alignment_abs", "ub_noise_private_stance_alignment_abs"),
            ("public_stance_change_error", "lb_public_stance_change_error_abs", "ub_noise_public_stance_change_error_abs"),
            ("private_stance_change_error", "lb_private_stance_change_error_abs", "ub_noise_private_stance_change_error_abs"),
            ("public_opinion_convergence", "lb_public_opinion_convergence_error", "ub_noise_public_opinion_convergence_error"),
            ("private_opinion_convergence", "lb_private_opinion_convergence_error", "ub_noise_private_opinion_convergence_error"),
        ]
        sigma_half_key_map = {
            "ub_noise_public_stance_alignment_abs": "ub_noise_sigma_half_public_stance_alignment_abs",
            "ub_noise_private_stance_alignment_abs": "ub_noise_sigma_half_private_stance_alignment_abs",
            "ub_noise_public_stance_change_error_abs": "ub_noise_sigma_half_public_stance_change_error_abs",
            "ub_noise_private_stance_change_error_abs": "ub_noise_sigma_half_private_stance_change_error_abs",
            "ub_noise_public_opinion_convergence_error": "ub_noise_sigma_half_public_opinion_convergence_error",
            "ub_noise_private_opinion_convergence_error": "ub_noise_sigma_half_private_opinion_convergence_error",
        }

        def bounded_score_v2(value: float, lower_bound: float) -> float:
            if any(math.isnan(v) for v in (value, lower_bound)):
                return math.nan
            denom = lower_bound
            if math.isclose(denom, 0.0):
                if math.isclose(value, 0.0):
                    return 1.0
                return math.nan
            return min(1.0, max(0.0, (lower_bound - value) / denom))

        lb_row: dict[str, Any] = {"model": PERM_LB_MODEL}
        for axis_key, _, _ in field_specs:
            lb_row[axis_key] = 0.0

        ub_row: dict[str, Any] = {"model": NOISY_UB_GAUSSIAN_LABEL}
        ub_sigma_half_row: dict[str, Any] = {"model": NOISY_UB_GAUSSIAN_SIGMA_SECONDARY_LABEL}
        for axis_key, lb_key, ub_key in field_specs:
            vals = [
                bounded_score_v2(as_float_or_nan(r.get(ub_key)), as_float_or_nan(r.get(lb_key)))
                for r in model_raw
            ]
            valid = [v for v in vals if not math.isnan(v)]
            ub_row[axis_key] = sum(valid) / len(valid) if valid else math.nan
            sigma_half_vals = [
                bounded_score_v2(as_float_or_nan(r.get(sigma_half_key_map[ub_key])), as_float_or_nan(r.get(lb_key)))
                for r in model_raw
            ]
            sigma_half_valid = [v for v in sigma_half_vals if not math.isnan(v)]
            ub_sigma_half_row[axis_key] = sum(sigma_half_valid) / len(sigma_half_valid) if sigma_half_valid else math.nan

        rows = [*score_rows, lb_row]
        if include_noisy_ub:
            rows.append(ub_row)
            rows.append(ub_sigma_half_row)
        return rows

    axis_keys = [k for k, _ in RADAR_AXES_V1]
    suffix = {
        "v1": "tau",
        "spearman": "rho",
        "pearson": "pearson",
    }.get(metric_version, "tau")
    lb_tau_keys = [
        f"lb_perm_public_alignment_{suffix}",
        f"lb_perm_private_alignment_{suffix}",
        f"lb_perm_public_change_error_{suffix}",
        f"lb_perm_private_change_error_{suffix}",
        f"lb_perm_public_convergence_{suffix}",
        f"lb_perm_private_convergence_{suffix}",
    ]
    ub_tau_keys = [
        f"ub_noise_public_alignment_{suffix}",
        f"ub_noise_private_alignment_{suffix}",
        f"ub_noise_public_change_error_{suffix}",
        f"ub_noise_private_change_error_{suffix}",
        f"ub_noise_public_convergence_{suffix}",
        f"ub_noise_private_convergence_{suffix}",
    ]

    lb_row: dict[str, Any] = {"model": PERM_LB_MODEL}
    for axis_key, lb_key in zip(axis_keys, lb_tau_keys):
        vals = [as_float_or_nan(r.get(lb_key)) for r in model_raw]
        valid = [v for v in vals if not math.isnan(v)]
        lb_row[axis_key] = sum(valid) / len(valid) if valid else math.nan

    ub_row: dict[str, Any] = {"model": NOISY_UB_JITTER_LABEL}
    for axis_key, ub_key in zip(axis_keys, ub_tau_keys):
        vals = [as_float_or_nan(r.get(ub_key)) for r in model_raw]
        valid = [v for v in vals if not math.isnan(v)]
        ub_row[axis_key] = sum(valid) / len(valid) if valid else math.nan

    rows = [*score_rows, lb_row]
    if include_noisy_ub:
        rows.append(ub_row)
    return rows


def build_score_rows_from_raw_rows(raw_rows: list[dict[str, Any]], metric_version: str = "v0") -> list[dict[str, Any]]:
    def bounded_score(
        value: float,
        *,
        lower_bound: float,
        upper_bound: float,
        higher_is_better: bool,
    ) -> float:
        if any(math.isnan(v) for v in (value, lower_bound, upper_bound)):
            return math.nan
        denom = (upper_bound - lower_bound) if higher_is_better else (lower_bound - upper_bound)
        if math.isclose(denom, 0.0):
            if math.isclose(value, upper_bound):
                return 1.0
            return math.nan
        raw_score = ((value - lower_bound) / denom) if higher_is_better else ((lower_bound - value) / denom)
        return min(1.0, max(0.0, raw_score))

    def has_bound_field(bound_name: str) -> bool:
        return any(bound_name in row for row in raw_rows)

    def values_map(metric_name: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for row in raw_rows:
            out[str(row["model"])] = as_float_or_nan(row.get(metric_name))
        return out

    use_bound_stance = has_bound_field("lb_stance_difference")
    use_bound_public_align = has_bound_field("lb_public_stance_alignment_abs")
    use_bound_private_align = has_bound_field("lb_private_stance_alignment_abs")
    use_bound_public_conv = has_bound_field("lb_public_opinion_convergence_error")
    use_bound_private_conv = has_bound_field("lb_private_opinion_convergence_error")
    use_bound_public_change = has_bound_field("lb_public_stance_change_error_abs")
    use_bound_private_change = has_bound_field("lb_private_stance_change_error_abs")
    use_bound_direction = has_bound_field("lb_stance_update_direction_accuracy")

    stance_scores = normalize_minmax(values_map("stance_difference"), higher_is_better=False)
    public_align_scores = normalize_minmax(values_map("public_stance_alignment_abs"), higher_is_better=False)
    private_align_scores = normalize_minmax(values_map("private_stance_alignment_abs"), higher_is_better=False)
    public_conv_scores = normalize_minmax(values_map("public_opinion_convergence_error"), higher_is_better=False)
    private_conv_scores = normalize_minmax(values_map("private_opinion_convergence_error"), higher_is_better=False)
    public_change_scores = normalize_minmax(values_map("public_stance_change_error_abs"), higher_is_better=False)
    private_change_scores = normalize_minmax(values_map("private_stance_change_error_abs"), higher_is_better=False)
    direction_scores = normalize_minmax(values_map("stance_update_direction_accuracy"), higher_is_better=True)

    score_rows: list[dict[str, Any]] = []
    for row in raw_rows:
        model = str(row["model"])
        if model == UNIFORM_LB_MODEL:
            continue
        if model.strip().lower() == "human":
            score_rows.append(
                {
                    "model": model,
                    "stance_difference": 1.0,
                    "public_stance_alignment": 1.0,
                    "private_stance_alignment": 1.0,
                    "public_opinion_convergence": 1.0,
                    "private_opinion_convergence": 1.0,
                    "public_stance_change_error": 1.0,
                    "private_stance_change_error": 1.0,
                    "stance_update_direction_accuracy": 1.0,
                }
            )
            continue

        if use_bound_stance:
            stance_score = bounded_score(
                as_float_or_nan(row.get("stance_difference")),
                lower_bound=as_float_or_nan(row.get("lb_stance_difference")),
                upper_bound=0.0,
                higher_is_better=False,
            )
        else:
            stance_score = stance_scores.get(model, math.nan)

        if use_bound_public_align:
            public_align_score = bounded_score(
                as_float_or_nan(row.get("public_stance_alignment_abs")),
                lower_bound=as_float_or_nan(row.get("lb_public_stance_alignment_abs")),
                upper_bound=0.0,
                higher_is_better=False,
            )
        else:
            public_align_score = public_align_scores.get(model, math.nan)

        if use_bound_private_align:
            private_align_score = bounded_score(
                as_float_or_nan(row.get("private_stance_alignment_abs")),
                lower_bound=as_float_or_nan(row.get("lb_private_stance_alignment_abs")),
                upper_bound=0.0,
                higher_is_better=False,
            )
        else:
            private_align_score = private_align_scores.get(model, math.nan)

        if use_bound_public_conv:
            public_conv_score = bounded_score(
                as_float_or_nan(row.get("public_opinion_convergence_error")),
                lower_bound=as_float_or_nan(row.get("lb_public_opinion_convergence_error")),
                upper_bound=0.0,
                higher_is_better=False,
            )
        else:
            public_conv_score = public_conv_scores.get(model, math.nan)

        if use_bound_private_conv:
            private_conv_score = bounded_score(
                as_float_or_nan(row.get("private_opinion_convergence_error")),
                lower_bound=as_float_or_nan(row.get("lb_private_opinion_convergence_error")),
                upper_bound=0.0,
                higher_is_better=False,
            )
        else:
            private_conv_score = private_conv_scores.get(model, math.nan)

        if use_bound_public_change:
            public_change_score = bounded_score(
                as_float_or_nan(row.get("public_stance_change_error_abs")),
                lower_bound=as_float_or_nan(row.get("lb_public_stance_change_error_abs")),
                upper_bound=0.0,
                higher_is_better=False,
            )
        else:
            public_change_score = public_change_scores.get(model, math.nan)

        if use_bound_private_change:
            private_change_score = bounded_score(
                as_float_or_nan(row.get("private_stance_change_error_abs")),
                lower_bound=as_float_or_nan(row.get("lb_private_stance_change_error_abs")),
                upper_bound=0.0,
                higher_is_better=False,
            )
        else:
            private_change_score = private_change_scores.get(model, math.nan)

        if use_bound_direction:
            direction_score = bounded_score(
                as_float_or_nan(row.get("stance_update_direction_accuracy")),
                lower_bound=as_float_or_nan(row.get("lb_stance_update_direction_accuracy")),
                upper_bound=1.0,
                higher_is_better=True,
            )
        else:
            direction_score = direction_scores.get(model, math.nan)

        # v1: override 6 metrics with raw Kendall tau-b values
        if metric_version == "v1":
            public_align_score = tau_to_score(as_float_or_nan(row.get("public_alignment_tau")))
            private_align_score = tau_to_score(as_float_or_nan(row.get("private_alignment_tau")))
            public_change_score = tau_to_score(as_float_or_nan(row.get("public_change_error_tau")))
            private_change_score = tau_to_score(as_float_or_nan(row.get("private_change_error_tau")))
            public_conv_score = tau_to_score(as_float_or_nan(row.get("public_convergence_tau")))
            private_conv_score = tau_to_score(as_float_or_nan(row.get("private_convergence_tau")))
            score_rows.append(
                {
                    "model": model,
                    "public_stance_alignment": public_align_score,
                    "private_stance_alignment": private_align_score,
                    "public_stance_change_error": public_change_score,
                    "private_stance_change_error": private_change_score,
                    "public_opinion_convergence": public_conv_score,
                    "private_opinion_convergence": private_conv_score,
                }
            )
            continue

        # v2: keep the v0_fixed-style bounded scores but restrict to the 6 v1-family axes
        if metric_version == "v2":
            score_rows.append(
                {
                    "model": model,
                    "public_stance_alignment": public_align_score,
                    "private_stance_alignment": private_align_score,
                    "public_stance_change_error": public_change_score,
                    "private_stance_change_error": private_change_score,
                    "public_opinion_convergence": public_conv_score,
                    "private_opinion_convergence": private_conv_score,
                }
            )
            continue

        # Spearman rho: override 6 metrics with raw Spearman values
        if metric_version == "spearman":
            score_rows.append(
                {
                    "model": model,
                    "public_stance_alignment": as_float_or_nan(row.get("public_alignment_rho")),
                    "private_stance_alignment": as_float_or_nan(row.get("private_alignment_rho")),
                    "public_stance_change_error": as_float_or_nan(row.get("public_change_error_rho")),
                    "private_stance_change_error": as_float_or_nan(row.get("private_change_error_rho")),
                    "public_opinion_convergence": as_float_or_nan(row.get("public_convergence_rho")),
                    "private_opinion_convergence": as_float_or_nan(row.get("private_convergence_rho")),
                }
            )
            continue

        # Pearson r: override 6 metrics with raw Pearson values
        if metric_version == "pearson":
            score_rows.append(
                {
                    "model": model,
                    "public_stance_alignment": as_float_or_nan(row.get("public_alignment_pearson")),
                    "private_stance_alignment": as_float_or_nan(row.get("private_alignment_pearson")),
                    "public_stance_change_error": as_float_or_nan(row.get("public_change_error_pearson")),
                    "private_stance_change_error": as_float_or_nan(row.get("private_change_error_pearson")),
                    "public_opinion_convergence": as_float_or_nan(row.get("public_convergence_pearson")),
                    "private_opinion_convergence": as_float_or_nan(row.get("private_convergence_pearson")),
                }
            )
            continue

        score_rows.append(
            {
                "model": model,
                "stance_difference": stance_score,
                "public_stance_alignment": public_align_score,
                "private_stance_alignment": private_align_score,
                "public_opinion_convergence": public_conv_score,
                "private_opinion_convergence": private_conv_score,
                "public_stance_change_error": public_change_score,
                "private_stance_change_error": private_change_score,
                "stance_update_direction_accuracy": direction_score,
            }
        )
    return score_rows


def infer_plot_input_type(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "unknown"
    cols = set(rows[0].keys())
    score_cols = {"model"} | {k for k, _ in RADAR_AXES}
    raw_cols = {
        "model",
        "stance_difference",
        "public_stance_alignment_abs",
        "private_stance_alignment_abs",
        "public_opinion_convergence_error",
        "private_opinion_convergence_error",
        "public_stance_change_error_abs",
        "private_stance_change_error_abs",
        "stance_update_direction_accuracy",
    }
    if score_cols.issubset(cols):
        return "scores"
    if raw_cols.issubset(cols):
        return "raw"
    return "unknown"


def extract_topic_from_data_prefix(data_prefix: str) -> str:
    parts = data_prefix.split("_")
    # Expected pattern: YYYYMMDD_HHMMSS_TOPIC_NAME_UNIQUE_ID
    if len(parts) >= 4 and re.fullmatch(r"\d{8}", parts[0]) and re.fullmatch(r"\d{6}", parts[1]):
        return "_".join(parts[2:-1])
    return data_prefix


def is_depth_topic(data_prefix: str) -> bool:
    return extract_topic_from_data_prefix(data_prefix) in DEPTH_TOPICS


def build_model_aggregates(models: list[str]) -> dict[str, ModelAggregate]:
    return {model: ModelAggregate(model=model) for model in models}


def extract_v2_bar_metrics_from_aggregate(agg: ModelAggregate) -> dict[str, float]:
    return {
        "public_stance_alignment_abs": (
            agg.public_alignment_abs_sum / agg.public_alignment_count if agg.public_alignment_count else math.nan
        ),
        "private_stance_alignment_abs": (
            agg.private_alignment_abs_sum / agg.private_alignment_count if agg.private_alignment_count else math.nan
        ),
        "public_stance_change_error_abs": safe_mean(agg.public_change_abs_errors),
        "private_stance_change_error_abs": safe_mean(agg.private_change_abs_errors),
        "public_opinion_convergence_error": safe_mean(agg.public_conv_errors),
        "private_opinion_convergence_error": safe_mean(agg.private_conv_errors),
    }


def percentile_value(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered_values = sorted(values)
    if len(ordered_values) == 1:
        return ordered_values[0]
    rank = (len(ordered_values) - 1) * q
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)
    if lower_index == upper_index:
        return ordered_values[lower_index]
    lower_value = ordered_values[lower_index]
    upper_value = ordered_values[upper_index]
    fraction = rank - lower_index
    return lower_value + (upper_value - lower_value) * fraction


def add_v2_bootstrap_cis_to_raw_rows(
    raw_rows: list[dict[str, Any]],
    *,
    models: list[str],
    prefixes: list[str],
    prefix_results_by_prefix: dict[str, dict[str, PrefixModelResult]],
    bootstrap_reps: int = V2_BOOTSTRAP_REPS,
    seed: int = V2_BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    updated_rows = [dict(row) for row in raw_rows]
    if not updated_rows or bootstrap_reps <= 0:
        return updated_rows

    available_prefixes = [prefix for prefix in prefixes if prefix in prefix_results_by_prefix]
    if not available_prefixes:
        return updated_rows

    bootstrap_samples: dict[str, dict[str, list[float]]] = {
        model: {field: [] for field in V2_BAR_METRIC_FIELDS}
        for model in models
    }
    rng = random.Random(seed)

    for _ in range(bootstrap_reps):
        sampled_prefixes = [available_prefixes[rng.randrange(len(available_prefixes))] for _ in available_prefixes]
        bootstrap_aggregates = build_model_aggregates(models)
        for sampled_prefix in sampled_prefixes:
            sampled_results = prefix_results_by_prefix.get(sampled_prefix, {})
            for model in models:
                result = sampled_results.get(model)
                if result is None:
                    continue
                merge_prefix_result_into_aggregate(bootstrap_aggregates[model], result)
        for model in models:
            metric_values = extract_v2_bar_metrics_from_aggregate(bootstrap_aggregates[model])
            for field_name, metric_value in metric_values.items():
                if not math.isnan(metric_value):
                    bootstrap_samples[model][field_name].append(metric_value)

    for row in updated_rows:
        model_name = str(row.get("model", "")).strip()
        for field_name in V2_BAR_METRIC_FIELDS:
            samples = bootstrap_samples.get(model_name, {}).get(field_name, [])
            row[f"{field_name}_ci_low"] = percentile_value(samples, 0.025)
            row[f"{field_name}_ci_high"] = percentile_value(samples, 0.975)

    return updated_rows


def merge_prefix_result_into_aggregate(agg: ModelAggregate, result: PrefixModelResult) -> None:
    if result.status != "ok":
        agg.prefixes_missing += 1
        return

    agg.prefixes_ok += 1
    agg.stance_sum += result.stance_sum
    agg.stance_count += result.stance_count
    agg.public_alignment_sum += result.public_alignment_sum
    agg.public_alignment_abs_sum += result.public_alignment_abs_sum
    agg.public_alignment_count += result.public_alignment_count
    agg.private_alignment_sum += result.private_alignment_sum
    agg.private_alignment_abs_sum += result.private_alignment_abs_sum
    agg.private_alignment_count += result.private_alignment_count
    agg.public_conv_errors.extend(result.public_conv_errors)
    agg.private_conv_errors.extend(result.private_conv_errors)
    agg.public_conv_h_vals.extend(result.public_conv_h_vals)
    agg.public_conv_l_vals.extend(result.public_conv_l_vals)
    agg.private_conv_h_vals.extend(result.private_conv_h_vals)
    agg.private_conv_l_vals.extend(result.private_conv_l_vals)
    agg.public_change_errors.extend(result.public_change_errors)
    agg.public_change_abs_errors.extend(result.public_change_abs_errors)
    agg.private_change_errors.extend(result.private_change_errors)
    agg.private_change_abs_errors.extend(result.private_change_abs_errors)
    agg.direction_true.extend(result.direction_true)
    agg.direction_pred.extend(result.direction_pred)
    agg.lb_stance_sum += result.lb_stance_sum
    agg.lb_stance_count += result.lb_stance_count
    agg.lb_public_alignment_sum += result.lb_public_alignment_sum
    agg.lb_public_alignment_abs_sum += result.lb_public_alignment_abs_sum
    agg.lb_public_alignment_count += result.lb_public_alignment_count
    agg.lb_private_alignment_sum += result.lb_private_alignment_sum
    agg.lb_private_alignment_abs_sum += result.lb_private_alignment_abs_sum
    agg.lb_private_alignment_count += result.lb_private_alignment_count
    agg.lb_public_conv_errors.extend(result.lb_public_conv_errors)
    agg.lb_private_conv_errors.extend(result.lb_private_conv_errors)
    agg.lb_public_change_errors.extend(result.lb_public_change_errors)
    agg.lb_public_change_abs_errors.extend(result.lb_public_change_abs_errors)
    agg.lb_private_change_errors.extend(result.lb_private_change_errors)
    agg.lb_private_change_abs_errors.extend(result.lb_private_change_abs_errors)
    agg.lb_direction_true.extend(result.lb_direction_true)
    agg.lb_direction_pred.extend(result.lb_direction_pred)
    agg.public_align_h_vals.extend(result.public_align_h_vals)
    agg.public_align_l_vals.extend(result.public_align_l_vals)
    agg.private_align_h_vals.extend(result.private_align_h_vals)
    agg.private_align_l_vals.extend(result.private_align_l_vals)
    agg.public_change_h_vals.extend(result.public_change_h_vals)
    agg.public_change_l_vals.extend(result.public_change_l_vals)
    agg.private_change_h_vals.extend(result.private_change_h_vals)
    agg.private_change_l_vals.extend(result.private_change_l_vals)
    agg.ub_public_align_h_vals.extend(result.ub_public_align_h_vals)
    agg.ub_public_align_l_vals.extend(result.ub_public_align_l_vals)
    agg.ub_private_align_h_vals.extend(result.ub_private_align_h_vals)
    agg.ub_private_align_l_vals.extend(result.ub_private_align_l_vals)
    agg.ub_public_change_h_vals.extend(result.ub_public_change_h_vals)
    agg.ub_public_change_l_vals.extend(result.ub_public_change_l_vals)
    agg.ub_private_change_h_vals.extend(result.ub_private_change_h_vals)
    agg.ub_private_change_l_vals.extend(result.ub_private_change_l_vals)
    agg.ub_public_conv_h_vals.extend(result.ub_public_conv_h_vals)
    agg.ub_public_conv_l_vals.extend(result.ub_public_conv_l_vals)
    agg.ub_private_conv_h_vals.extend(result.ub_private_conv_h_vals)
    agg.ub_private_conv_l_vals.extend(result.ub_private_conv_l_vals)
    agg.ub_sigma_half_public_align_h_vals.extend(result.ub_sigma_half_public_align_h_vals)
    agg.ub_sigma_half_public_align_l_vals.extend(result.ub_sigma_half_public_align_l_vals)
    agg.ub_sigma_half_private_align_h_vals.extend(result.ub_sigma_half_private_align_h_vals)
    agg.ub_sigma_half_private_align_l_vals.extend(result.ub_sigma_half_private_align_l_vals)
    agg.ub_sigma_half_public_change_h_vals.extend(result.ub_sigma_half_public_change_h_vals)
    agg.ub_sigma_half_public_change_l_vals.extend(result.ub_sigma_half_public_change_l_vals)
    agg.ub_sigma_half_private_change_h_vals.extend(result.ub_sigma_half_private_change_h_vals)
    agg.ub_sigma_half_private_change_l_vals.extend(result.ub_sigma_half_private_change_l_vals)
    agg.ub_sigma_half_public_conv_h_vals.extend(result.ub_sigma_half_public_conv_h_vals)
    agg.ub_sigma_half_public_conv_l_vals.extend(result.ub_sigma_half_public_conv_l_vals)
    agg.ub_sigma_half_private_conv_h_vals.extend(result.ub_sigma_half_private_conv_h_vals)
    agg.ub_sigma_half_private_conv_l_vals.extend(result.ub_sigma_half_private_conv_l_vals)
    agg.uniform_public_alignment_abs_sum += result.uniform_public_alignment_abs_sum
    agg.uniform_public_alignment_count += result.uniform_public_alignment_count
    agg.uniform_private_alignment_abs_sum += result.uniform_private_alignment_abs_sum
    agg.uniform_private_alignment_count += result.uniform_private_alignment_count
    agg.uniform_public_conv_errors.extend(result.uniform_public_conv_errors)
    agg.uniform_private_conv_errors.extend(result.uniform_private_conv_errors)
    agg.uniform_public_change_abs_errors.extend(result.uniform_public_change_abs_errors)
    agg.uniform_private_change_abs_errors.extend(result.uniform_private_change_abs_errors)


def build_raw_rows_from_aggregates(
    aggregates: dict[str, ModelAggregate],
    models: list[str],
    ub_mode: str = "jitter",
) -> list[dict[str, Any]]:
    raw_rows: list[dict[str, Any]] = []
    for model in models:
        agg = aggregates[model]
        stance_difference = agg.stance_sum / agg.stance_count if agg.stance_count else math.nan
        public_stance_alignment = (
            agg.public_alignment_sum / agg.public_alignment_count if agg.public_alignment_count else math.nan
        )
        private_stance_alignment = (
            agg.private_alignment_sum / agg.private_alignment_count if agg.private_alignment_count else math.nan
        )
        public_stance_alignment_abs = (
            agg.public_alignment_abs_sum / agg.public_alignment_count if agg.public_alignment_count else math.nan
        )
        private_stance_alignment_abs = (
            agg.private_alignment_abs_sum / agg.private_alignment_count if agg.private_alignment_count else math.nan
        )
        public_opinion_convergence_error = safe_mean(agg.public_conv_errors)
        private_opinion_convergence_error = safe_mean(agg.private_conv_errors)
        if agg.public_conv_h_vals and len(agg.public_conv_h_vals) == len(agg.public_conv_l_vals):
            shuffled_public_conv_l = deterministic_shuffle(
                agg.public_conv_l_vals,
                data_prefix="__aggregate__",
                model=model,
                metric_name="public_convergence",
            )
            lb_public_conv_errors = [abs(l - h) for h, l in zip(agg.public_conv_h_vals, shuffled_public_conv_l)]
        else:
            lb_public_conv_errors = agg.lb_public_conv_errors
        if agg.private_conv_h_vals and len(agg.private_conv_h_vals) == len(agg.private_conv_l_vals):
            shuffled_private_conv_l = deterministic_shuffle(
                agg.private_conv_l_vals,
                data_prefix="__aggregate__",
                model=model,
                metric_name="private_convergence",
            )
            lb_private_conv_errors = [abs(l - h) for h, l in zip(agg.private_conv_h_vals, shuffled_private_conv_l)]
        else:
            lb_private_conv_errors = agg.lb_private_conv_errors
        public_stance_change_error = safe_mean(agg.public_change_errors)
        private_stance_change_error = safe_mean(agg.private_change_errors)
        public_stance_change_error_abs = safe_mean(agg.public_change_abs_errors)
        private_stance_change_error_abs = safe_mean(agg.private_change_abs_errors)
        stance_update_direction_accuracy = direction_accuracy(agg.direction_true, agg.direction_pred)
        lb_stance_difference = agg.lb_stance_sum / agg.lb_stance_count if agg.lb_stance_count else math.nan
        lb_public_stance_alignment = (
            agg.lb_public_alignment_sum / agg.lb_public_alignment_count if agg.lb_public_alignment_count else math.nan
        )
        lb_private_stance_alignment = (
            agg.lb_private_alignment_sum / agg.lb_private_alignment_count if agg.lb_private_alignment_count else math.nan
        )
        lb_public_stance_alignment_abs = (
            agg.lb_public_alignment_abs_sum / agg.lb_public_alignment_count if agg.lb_public_alignment_count else math.nan
        )
        lb_private_stance_alignment_abs = (
            agg.lb_private_alignment_abs_sum / agg.lb_private_alignment_count if agg.lb_private_alignment_count else math.nan
        )
        lb_public_opinion_convergence_error = safe_mean(lb_public_conv_errors)
        lb_private_opinion_convergence_error = safe_mean(lb_private_conv_errors)
        lb_public_stance_change_error = safe_mean(agg.lb_public_change_errors)
        lb_private_stance_change_error = safe_mean(agg.lb_private_change_errors)
        lb_public_stance_change_error_abs = safe_mean(agg.lb_public_change_abs_errors)
        lb_private_stance_change_error_abs = safe_mean(agg.lb_private_change_abs_errors)
        lb_stance_update_direction_accuracy = direction_accuracy(agg.lb_direction_true, agg.lb_direction_pred)
        public_alignment_tau = kendall_tau_b(agg.public_align_h_vals, agg.public_align_l_vals)
        private_alignment_tau = kendall_tau_b(agg.private_align_h_vals, agg.private_align_l_vals)
        public_change_error_tau = kendall_tau_b(agg.public_change_h_vals, agg.public_change_l_vals)
        private_change_error_tau = kendall_tau_b(agg.private_change_h_vals, agg.private_change_l_vals)
        public_convergence_tau = kendall_tau_b(agg.public_conv_h_vals, agg.public_conv_l_vals)
        private_convergence_tau = kendall_tau_b(agg.private_conv_h_vals, agg.private_conv_l_vals)
        # Permutation LB: shuffle l_vals to destroy correlation → null tau
        lb_perm_public_alignment_tau = kendall_tau_b(
            agg.public_align_h_vals,
            deterministic_shuffle(agg.public_align_l_vals, data_prefix="__aggregate__", model=model, metric_name="lb_perm_public_alignment"),
        )
        lb_perm_private_alignment_tau = kendall_tau_b(
            agg.private_align_h_vals,
            deterministic_shuffle(agg.private_align_l_vals, data_prefix="__aggregate__", model=model, metric_name="lb_perm_private_alignment"),
        )
        lb_perm_public_change_error_tau = kendall_tau_b(
            agg.public_change_h_vals,
            deterministic_shuffle(agg.public_change_l_vals, data_prefix="__aggregate__", model=model, metric_name="lb_perm_public_change_error"),
        )
        lb_perm_private_change_error_tau = kendall_tau_b(
            agg.private_change_h_vals,
            deterministic_shuffle(agg.private_change_l_vals, data_prefix="__aggregate__", model=model, metric_name="lb_perm_private_change_error"),
        )
        lb_perm_public_convergence_tau = kendall_tau_b(
            agg.public_conv_h_vals,
            deterministic_shuffle(agg.public_conv_l_vals, data_prefix="__aggregate__", model=model, metric_name="lb_perm_public_convergence"),
        )
        lb_perm_private_convergence_tau = kendall_tau_b(
            agg.private_conv_h_vals,
            deterministic_shuffle(agg.private_conv_l_vals, data_prefix="__aggregate__", model=model, metric_name="lb_perm_private_convergence"),
        )
        if ub_mode == "gaussian_likert":
            # v2 noisy UB: perturb the discrete human Likert labels once, then derive all
            # downstream metrics from that noisy label process on the aligned subset.
            ub_noise_public_alignment_tau = kendall_tau_b(
                agg.ub_public_align_h_vals,
                agg.ub_public_align_l_vals,
            )
            ub_noise_private_alignment_tau = kendall_tau_b(
                agg.ub_private_align_h_vals,
                agg.ub_private_align_l_vals,
            )
            ub_noise_public_change_error_tau = kendall_tau_b(
                agg.ub_public_change_h_vals,
                agg.ub_public_change_l_vals,
            )
            ub_noise_private_change_error_tau = kendall_tau_b(
                agg.ub_private_change_h_vals,
                agg.ub_private_change_l_vals,
            )
            ub_noise_public_convergence_tau = kendall_tau_b(
                agg.ub_public_conv_h_vals,
                agg.ub_public_conv_l_vals,
            )
            ub_noise_private_convergence_tau = kendall_tau_b(
                agg.ub_private_conv_h_vals,
                agg.ub_private_conv_l_vals,
            )

            def noisy_abs_error(human_values: list[float], noisy_values: list[float]) -> float:
                if not human_values or not noisy_values:
                    return math.nan
                return safe_mean([abs(noisy - human) for human, noisy in zip(human_values, noisy_values)])

            ub_noise_public_stance_alignment_abs = noisy_abs_error(agg.ub_public_align_h_vals, agg.ub_public_align_l_vals)
            ub_noise_private_stance_alignment_abs = noisy_abs_error(agg.ub_private_align_h_vals, agg.ub_private_align_l_vals)
            ub_noise_public_stance_change_error_abs = noisy_abs_error(agg.ub_public_change_h_vals, agg.ub_public_change_l_vals)
            ub_noise_private_stance_change_error_abs = noisy_abs_error(agg.ub_private_change_h_vals, agg.ub_private_change_l_vals)
            ub_noise_public_opinion_convergence_error = noisy_abs_error(agg.ub_public_conv_h_vals, agg.ub_public_conv_l_vals)
            ub_noise_private_opinion_convergence_error = noisy_abs_error(agg.ub_private_conv_h_vals, agg.ub_private_conv_l_vals)
            ub_noise_sigma_half_public_stance_alignment_abs = noisy_abs_error(
                agg.ub_sigma_half_public_align_h_vals,
                agg.ub_sigma_half_public_align_l_vals,
            )
            ub_noise_sigma_half_private_stance_alignment_abs = noisy_abs_error(
                agg.ub_sigma_half_private_align_h_vals,
                agg.ub_sigma_half_private_align_l_vals,
            )
            ub_noise_sigma_half_public_stance_change_error_abs = noisy_abs_error(
                agg.ub_sigma_half_public_change_h_vals,
                agg.ub_sigma_half_public_change_l_vals,
            )
            ub_noise_sigma_half_private_stance_change_error_abs = noisy_abs_error(
                agg.ub_sigma_half_private_change_h_vals,
                agg.ub_sigma_half_private_change_l_vals,
            )
            ub_noise_sigma_half_public_opinion_convergence_error = noisy_abs_error(
                agg.ub_sigma_half_public_conv_h_vals,
                agg.ub_sigma_half_public_conv_l_vals,
            )
            ub_noise_sigma_half_private_opinion_convergence_error = noisy_abs_error(
                agg.ub_sigma_half_private_conv_h_vals,
                agg.ub_sigma_half_private_conv_l_vals,
            )
            ub_noise_public_alignment_rho = spearman_rho(
                agg.ub_public_align_h_vals,
                agg.ub_public_align_l_vals,
            )
            ub_noise_private_alignment_rho = spearman_rho(
                agg.ub_private_align_h_vals,
                agg.ub_private_align_l_vals,
            )
            ub_noise_public_change_error_rho = spearman_rho(
                agg.ub_public_change_h_vals,
                agg.ub_public_change_l_vals,
            )
            ub_noise_private_change_error_rho = spearman_rho(
                agg.ub_private_change_h_vals,
                agg.ub_private_change_l_vals,
            )
            ub_noise_public_convergence_rho = spearman_rho(
                agg.ub_public_conv_h_vals,
                agg.ub_public_conv_l_vals,
            )
            ub_noise_private_convergence_rho = spearman_rho(
                agg.ub_private_conv_h_vals,
                agg.ub_private_conv_l_vals,
            )
            ub_noise_public_alignment_pearson = pearson_r(
                agg.ub_public_align_h_vals,
                agg.ub_public_align_l_vals,
            )
            ub_noise_private_alignment_pearson = pearson_r(
                agg.ub_private_align_h_vals,
                agg.ub_private_align_l_vals,
            )
            ub_noise_public_change_error_pearson = pearson_r(
                agg.ub_public_change_h_vals,
                agg.ub_public_change_l_vals,
            )
            ub_noise_private_change_error_pearson = pearson_r(
                agg.ub_private_change_h_vals,
                agg.ub_private_change_l_vals,
            )
            ub_noise_public_convergence_pearson = pearson_r(
                agg.ub_public_conv_h_vals,
                agg.ub_public_conv_l_vals,
            )
            ub_noise_private_convergence_pearson = pearson_r(
                agg.ub_private_conv_h_vals,
                agg.ub_private_conv_l_vals,
            )
        else:
            # Legacy noisy UB for v1-family reproduction: centre-weighted ordinal jitter
            # applied to the derived values.
            jitter_public_align = deterministic_ordinal_jitter(
                agg.public_align_h_vals,
                data_prefix="__aggregate__",
                model=model,
                metric_name="ub_noise_public_alignment",
            )
            jitter_private_align = deterministic_ordinal_jitter(
                agg.private_align_h_vals,
                data_prefix="__aggregate__",
                model=model,
                metric_name="ub_noise_private_alignment",
            )
            jitter_public_change = deterministic_ordinal_jitter(
                agg.public_change_h_vals,
                data_prefix="__aggregate__",
                model=model,
                metric_name="ub_noise_public_change_error",
            )
            jitter_private_change = deterministic_ordinal_jitter(
                agg.private_change_h_vals,
                data_prefix="__aggregate__",
                model=model,
                metric_name="ub_noise_private_change_error",
            )
            jitter_public_conv = deterministic_ordinal_jitter(
                agg.public_conv_h_vals,
                data_prefix="__aggregate__",
                model=model,
                metric_name="ub_noise_public_convergence",
            )
            jitter_private_conv = deterministic_ordinal_jitter(
                agg.private_conv_h_vals,
                data_prefix="__aggregate__",
                model=model,
                metric_name="ub_noise_private_convergence",
            )

            ub_noise_public_alignment_tau = kendall_tau_b(agg.public_align_h_vals, jitter_public_align)
            ub_noise_private_alignment_tau = kendall_tau_b(agg.private_align_h_vals, jitter_private_align)
            ub_noise_public_change_error_tau = kendall_tau_b(agg.public_change_h_vals, jitter_public_change)
            ub_noise_private_change_error_tau = kendall_tau_b(agg.private_change_h_vals, jitter_private_change)
            ub_noise_public_convergence_tau = kendall_tau_b(agg.public_conv_h_vals, jitter_public_conv)
            ub_noise_private_convergence_tau = kendall_tau_b(agg.private_conv_h_vals, jitter_private_conv)
            ub_noise_public_stance_alignment_abs = safe_mean(
                [abs(jitter - human) for human, jitter in zip(agg.public_align_h_vals, jitter_public_align)]
            )
            ub_noise_private_stance_alignment_abs = safe_mean(
                [abs(jitter - human) for human, jitter in zip(agg.private_align_h_vals, jitter_private_align)]
            )
            ub_noise_public_stance_change_error_abs = safe_mean(
                [abs(jitter - human) for human, jitter in zip(agg.public_change_h_vals, jitter_public_change)]
            )
            ub_noise_private_stance_change_error_abs = safe_mean(
                [abs(jitter - human) for human, jitter in zip(agg.private_change_h_vals, jitter_private_change)]
            )
            ub_noise_public_opinion_convergence_error = safe_mean(
                [abs(jitter - human) for human, jitter in zip(agg.public_conv_h_vals, jitter_public_conv)]
            )
            ub_noise_private_opinion_convergence_error = safe_mean(
                [abs(jitter - human) for human, jitter in zip(agg.private_conv_h_vals, jitter_private_conv)]
            )
            ub_noise_public_alignment_rho = spearman_rho(agg.public_align_h_vals, jitter_public_align)
            ub_noise_private_alignment_rho = spearman_rho(agg.private_align_h_vals, jitter_private_align)
            ub_noise_public_change_error_rho = spearman_rho(agg.public_change_h_vals, jitter_public_change)
            ub_noise_private_change_error_rho = spearman_rho(agg.private_change_h_vals, jitter_private_change)
            ub_noise_public_convergence_rho = spearman_rho(agg.public_conv_h_vals, jitter_public_conv)
            ub_noise_private_convergence_rho = spearman_rho(agg.private_conv_h_vals, jitter_private_conv)
            ub_noise_public_alignment_pearson = pearson_r(agg.public_align_h_vals, jitter_public_align)
            ub_noise_private_alignment_pearson = pearson_r(agg.private_align_h_vals, jitter_private_align)
            ub_noise_public_change_error_pearson = pearson_r(agg.public_change_h_vals, jitter_public_change)
            ub_noise_private_change_error_pearson = pearson_r(agg.private_change_h_vals, jitter_private_change)
            ub_noise_public_convergence_pearson = pearson_r(agg.public_conv_h_vals, jitter_public_conv)
            ub_noise_private_convergence_pearson = pearson_r(agg.private_conv_h_vals, jitter_private_conv)
            ub_noise_sigma_half_public_stance_alignment_abs = math.nan
            ub_noise_sigma_half_private_stance_alignment_abs = math.nan
            ub_noise_sigma_half_public_stance_change_error_abs = math.nan
            ub_noise_sigma_half_private_stance_change_error_abs = math.nan
            ub_noise_sigma_half_public_opinion_convergence_error = math.nan
            ub_noise_sigma_half_private_opinion_convergence_error = math.nan

        # Spearman rho — same h/l pairs as above, different correlation function
        public_alignment_rho = spearman_rho(agg.public_align_h_vals, agg.public_align_l_vals)
        private_alignment_rho = spearman_rho(agg.private_align_h_vals, agg.private_align_l_vals)
        public_change_error_rho = spearman_rho(agg.public_change_h_vals, agg.public_change_l_vals)
        private_change_error_rho = spearman_rho(agg.private_change_h_vals, agg.private_change_l_vals)
        public_convergence_rho = spearman_rho(agg.public_conv_h_vals, agg.public_conv_l_vals)
        private_convergence_rho = spearman_rho(agg.private_conv_h_vals, agg.private_conv_l_vals)
        lb_perm_public_alignment_rho = spearman_rho(
            agg.public_align_h_vals,
            deterministic_shuffle(agg.public_align_l_vals, data_prefix="__aggregate__", model=model, metric_name="lb_perm_public_alignment"),
        )
        lb_perm_private_alignment_rho = spearman_rho(
            agg.private_align_h_vals,
            deterministic_shuffle(agg.private_align_l_vals, data_prefix="__aggregate__", model=model, metric_name="lb_perm_private_alignment"),
        )
        lb_perm_public_change_error_rho = spearman_rho(
            agg.public_change_h_vals,
            deterministic_shuffle(agg.public_change_l_vals, data_prefix="__aggregate__", model=model, metric_name="lb_perm_public_change_error"),
        )
        lb_perm_private_change_error_rho = spearman_rho(
            agg.private_change_h_vals,
            deterministic_shuffle(agg.private_change_l_vals, data_prefix="__aggregate__", model=model, metric_name="lb_perm_private_change_error"),
        )
        lb_perm_public_convergence_rho = spearman_rho(
            agg.public_conv_h_vals,
            deterministic_shuffle(agg.public_conv_l_vals, data_prefix="__aggregate__", model=model, metric_name="lb_perm_public_convergence"),
        )
        lb_perm_private_convergence_rho = spearman_rho(
            agg.private_conv_h_vals,
            deterministic_shuffle(agg.private_conv_l_vals, data_prefix="__aggregate__", model=model, metric_name="lb_perm_private_convergence"),
        )
        # Pearson r — same h/l pairs as above, linear correlation on raw values
        public_alignment_pearson = pearson_r(agg.public_align_h_vals, agg.public_align_l_vals)
        private_alignment_pearson = pearson_r(agg.private_align_h_vals, agg.private_align_l_vals)
        public_change_error_pearson = pearson_r(agg.public_change_h_vals, agg.public_change_l_vals)
        private_change_error_pearson = pearson_r(agg.private_change_h_vals, agg.private_change_l_vals)
        public_convergence_pearson = pearson_r(agg.public_conv_h_vals, agg.public_conv_l_vals)
        private_convergence_pearson = pearson_r(agg.private_conv_h_vals, agg.private_conv_l_vals)
        lb_perm_public_alignment_pearson = pearson_r(
            agg.public_align_h_vals,
            deterministic_shuffle(agg.public_align_l_vals, data_prefix="__aggregate__", model=model, metric_name="lb_perm_public_alignment"),
        )
        lb_perm_private_alignment_pearson = pearson_r(
            agg.private_align_h_vals,
            deterministic_shuffle(agg.private_align_l_vals, data_prefix="__aggregate__", model=model, metric_name="lb_perm_private_alignment"),
        )
        lb_perm_public_change_error_pearson = pearson_r(
            agg.public_change_h_vals,
            deterministic_shuffle(agg.public_change_l_vals, data_prefix="__aggregate__", model=model, metric_name="lb_perm_public_change_error"),
        )
        lb_perm_private_change_error_pearson = pearson_r(
            agg.private_change_h_vals,
            deterministic_shuffle(agg.private_change_l_vals, data_prefix="__aggregate__", model=model, metric_name="lb_perm_private_change_error"),
        )
        lb_perm_public_convergence_pearson = pearson_r(
            agg.public_conv_h_vals,
            deterministic_shuffle(agg.public_conv_l_vals, data_prefix="__aggregate__", model=model, metric_name="lb_perm_public_convergence"),
        )
        lb_perm_private_convergence_pearson = pearson_r(
            agg.private_conv_h_vals,
            deterministic_shuffle(agg.private_conv_l_vals, data_prefix="__aggregate__", model=model, metric_name="lb_perm_private_convergence"),
        )

        raw_rows.append(
            {
                "model": model,
                "prefixes_ok": agg.prefixes_ok,
                "prefixes_missing": agg.prefixes_missing,
                "stance_difference": stance_difference,
                "public_stance_alignment": public_stance_alignment,
                "public_stance_alignment_abs": public_stance_alignment_abs,
                "private_stance_alignment": private_stance_alignment,
                "private_stance_alignment_abs": private_stance_alignment_abs,
                "public_opinion_convergence_error": public_opinion_convergence_error,
                "private_opinion_convergence_error": private_opinion_convergence_error,
                "public_stance_change_error": public_stance_change_error,
                "public_stance_change_error_abs": public_stance_change_error_abs,
                "private_stance_change_error": private_stance_change_error,
                "private_stance_change_error_abs": private_stance_change_error_abs,
                "stance_update_direction_accuracy": stance_update_direction_accuracy,
                "lb_stance_difference": lb_stance_difference,
                "lb_public_stance_alignment": lb_public_stance_alignment,
                "lb_public_stance_alignment_abs": lb_public_stance_alignment_abs,
                "lb_private_stance_alignment": lb_private_stance_alignment,
                "lb_private_stance_alignment_abs": lb_private_stance_alignment_abs,
                "lb_public_opinion_convergence_error": lb_public_opinion_convergence_error,
                "lb_private_opinion_convergence_error": lb_private_opinion_convergence_error,
                "lb_public_stance_change_error": lb_public_stance_change_error,
                "lb_public_stance_change_error_abs": lb_public_stance_change_error_abs,
                "lb_private_stance_change_error": lb_private_stance_change_error,
                "lb_private_stance_change_error_abs": lb_private_stance_change_error_abs,
                "lb_stance_update_direction_accuracy": lb_stance_update_direction_accuracy,
                "ub_noise_public_stance_alignment_abs": ub_noise_public_stance_alignment_abs,
                "ub_noise_private_stance_alignment_abs": ub_noise_private_stance_alignment_abs,
                "ub_noise_public_stance_change_error_abs": ub_noise_public_stance_change_error_abs,
                "ub_noise_private_stance_change_error_abs": ub_noise_private_stance_change_error_abs,
                "ub_noise_public_opinion_convergence_error": ub_noise_public_opinion_convergence_error,
                "ub_noise_private_opinion_convergence_error": ub_noise_private_opinion_convergence_error,
                "ub_noise_sigma_half_public_stance_alignment_abs": ub_noise_sigma_half_public_stance_alignment_abs,
                "ub_noise_sigma_half_private_stance_alignment_abs": ub_noise_sigma_half_private_stance_alignment_abs,
                "ub_noise_sigma_half_public_stance_change_error_abs": ub_noise_sigma_half_public_stance_change_error_abs,
                "ub_noise_sigma_half_private_stance_change_error_abs": ub_noise_sigma_half_private_stance_change_error_abs,
                "ub_noise_sigma_half_public_opinion_convergence_error": ub_noise_sigma_half_public_opinion_convergence_error,
                "ub_noise_sigma_half_private_opinion_convergence_error": ub_noise_sigma_half_private_opinion_convergence_error,
                "stance_count": agg.stance_count,
                "public_alignment_count": agg.public_alignment_count,
                "private_alignment_count": agg.private_alignment_count,
                "public_group_count": len(agg.public_conv_errors),
                "private_group_count": len(agg.private_conv_errors),
                "public_change_count": len(agg.public_change_errors),
                "private_change_count": len(agg.private_change_errors),
                "direction_count": len(agg.direction_true),
                "lb_stance_count": agg.lb_stance_count,
                "lb_public_alignment_count": agg.lb_public_alignment_count,
                "lb_private_alignment_count": agg.lb_private_alignment_count,
                "lb_public_group_count": len(lb_public_conv_errors),
                "lb_private_group_count": len(lb_private_conv_errors),
                "lb_public_change_count": len(agg.lb_public_change_errors),
                "lb_private_change_count": len(agg.lb_private_change_errors),
                "lb_direction_count": len(agg.lb_direction_true),
                "public_alignment_tau": public_alignment_tau,
                "private_alignment_tau": private_alignment_tau,
                "public_change_error_tau": public_change_error_tau,
                "private_change_error_tau": private_change_error_tau,
                "public_convergence_tau": public_convergence_tau,
                "private_convergence_tau": private_convergence_tau,
                "lb_perm_public_alignment_tau": lb_perm_public_alignment_tau,
                "lb_perm_private_alignment_tau": lb_perm_private_alignment_tau,
                "lb_perm_public_change_error_tau": lb_perm_public_change_error_tau,
                "lb_perm_private_change_error_tau": lb_perm_private_change_error_tau,
                "lb_perm_public_convergence_tau": lb_perm_public_convergence_tau,
                "lb_perm_private_convergence_tau": lb_perm_private_convergence_tau,
                "ub_noise_public_alignment_tau": ub_noise_public_alignment_tau,
                "ub_noise_private_alignment_tau": ub_noise_private_alignment_tau,
                "ub_noise_public_change_error_tau": ub_noise_public_change_error_tau,
                "ub_noise_private_change_error_tau": ub_noise_private_change_error_tau,
                "ub_noise_public_convergence_tau": ub_noise_public_convergence_tau,
                "ub_noise_private_convergence_tau": ub_noise_private_convergence_tau,
                "public_alignment_rho": public_alignment_rho,
                "private_alignment_rho": private_alignment_rho,
                "public_change_error_rho": public_change_error_rho,
                "private_change_error_rho": private_change_error_rho,
                "public_convergence_rho": public_convergence_rho,
                "private_convergence_rho": private_convergence_rho,
                "lb_perm_public_alignment_rho": lb_perm_public_alignment_rho,
                "lb_perm_private_alignment_rho": lb_perm_private_alignment_rho,
                "lb_perm_public_change_error_rho": lb_perm_public_change_error_rho,
                "lb_perm_private_change_error_rho": lb_perm_private_change_error_rho,
                "lb_perm_public_convergence_rho": lb_perm_public_convergence_rho,
                "lb_perm_private_convergence_rho": lb_perm_private_convergence_rho,
                "ub_noise_public_alignment_rho": ub_noise_public_alignment_rho,
                "ub_noise_private_alignment_rho": ub_noise_private_alignment_rho,
                "ub_noise_public_change_error_rho": ub_noise_public_change_error_rho,
                "ub_noise_private_change_error_rho": ub_noise_private_change_error_rho,
                "ub_noise_public_convergence_rho": ub_noise_public_convergence_rho,
                "ub_noise_private_convergence_rho": ub_noise_private_convergence_rho,
                "public_alignment_pearson": public_alignment_pearson,
                "private_alignment_pearson": private_alignment_pearson,
                "public_change_error_pearson": public_change_error_pearson,
                "private_change_error_pearson": private_change_error_pearson,
                "public_convergence_pearson": public_convergence_pearson,
                "private_convergence_pearson": private_convergence_pearson,
                "lb_perm_public_alignment_pearson": lb_perm_public_alignment_pearson,
                "lb_perm_private_alignment_pearson": lb_perm_private_alignment_pearson,
                "lb_perm_public_change_error_pearson": lb_perm_public_change_error_pearson,
                "lb_perm_private_change_error_pearson": lb_perm_private_change_error_pearson,
                "lb_perm_public_convergence_pearson": lb_perm_public_convergence_pearson,
                "lb_perm_private_convergence_pearson": lb_perm_private_convergence_pearson,
                "ub_noise_public_alignment_pearson": ub_noise_public_alignment_pearson,
                "ub_noise_private_alignment_pearson": ub_noise_private_alignment_pearson,
                "ub_noise_public_change_error_pearson": ub_noise_public_change_error_pearson,
                "ub_noise_private_change_error_pearson": ub_noise_private_change_error_pearson,
                "ub_noise_public_convergence_pearson": ub_noise_public_convergence_pearson,
                "ub_noise_private_convergence_pearson": ub_noise_private_convergence_pearson,
            }
        )

    if ub_mode == "gaussian_likert" and raw_rows and not any(
        str(row.get("model", "")).strip() == MEAN_PREDICTOR_MODEL for row in raw_rows
    ):
        source_agg = max(
            (aggregates[model] for model in models),
            key=lambda agg: (
                len(agg.public_align_h_vals)
                + len(agg.private_align_h_vals)
                + len(agg.public_change_h_vals)
                + len(agg.private_change_h_vals)
                + len(agg.public_conv_h_vals)
                + len(agg.private_conv_h_vals)
            ),
            default=None,
        )
        if source_agg is not None:
            baseline_row: dict[str, Any] = {
                key: math.nan
                for row in raw_rows
                for key in row.keys()
            }
            baseline_row["model"] = MEAN_PREDICTOR_MODEL
            baseline_row["prefixes_ok"] = source_agg.prefixes_ok
            baseline_row["prefixes_missing"] = source_agg.prefixes_missing
            baseline_row["public_stance_alignment_abs"] = mean_absolute_error_to_mean(source_agg.public_align_h_vals)
            baseline_row["private_stance_alignment_abs"] = mean_absolute_error_to_mean(source_agg.private_align_h_vals)
            baseline_row["public_stance_change_error_abs"] = mean_absolute_error_to_mean(source_agg.public_change_h_vals)
            baseline_row["private_stance_change_error_abs"] = mean_absolute_error_to_mean(source_agg.private_change_h_vals)
            baseline_row["public_opinion_convergence_error"] = mean_absolute_error_to_mean(source_agg.public_conv_h_vals)
            baseline_row["private_opinion_convergence_error"] = mean_absolute_error_to_mean(source_agg.private_conv_h_vals)
            baseline_row["public_alignment_count"] = len(source_agg.public_align_h_vals)
            baseline_row["private_alignment_count"] = len(source_agg.private_align_h_vals)
            baseline_row["public_change_count"] = len(source_agg.public_change_h_vals)
            baseline_row["private_change_count"] = len(source_agg.private_change_h_vals)
            baseline_row["public_group_count"] = len(source_agg.public_conv_h_vals)
            baseline_row["private_group_count"] = len(source_agg.private_conv_h_vals)

            for lb_key in (
                "lb_public_stance_alignment_abs",
                "lb_private_stance_alignment_abs",
                "lb_public_stance_change_error_abs",
                "lb_private_stance_change_error_abs",
                "lb_public_opinion_convergence_error",
                "lb_private_opinion_convergence_error",
            ):
                valid = [as_float_or_nan(row.get(lb_key)) for row in raw_rows]
                valid = [value for value in valid if not math.isnan(value)]
                baseline_row[lb_key] = safe_mean(valid)

            raw_rows.append(baseline_row)

    if ub_mode == "gaussian_likert" and raw_rows and not any(
        str(row.get("model", "")).strip() == UNIFORM_LB_MODEL for row in raw_rows
    ):
        source_agg = max(
            (aggregates[model] for model in models),
            key=lambda agg: (
                agg.uniform_public_alignment_count
                + agg.uniform_private_alignment_count
                + len(agg.uniform_public_change_abs_errors)
                + len(agg.uniform_private_change_abs_errors)
                + len(agg.uniform_public_conv_errors)
                + len(agg.uniform_private_conv_errors)
            ),
            default=None,
        )
        if source_agg is not None:
            uniform_row: dict[str, Any] = {
                key: math.nan
                for row in raw_rows
                for key in row.keys()
            }
            uniform_row["model"] = UNIFORM_LB_MODEL
            uniform_row["prefixes_ok"] = source_agg.prefixes_ok
            uniform_row["prefixes_missing"] = source_agg.prefixes_missing
            uniform_row["public_stance_alignment_abs"] = (
                source_agg.uniform_public_alignment_abs_sum / source_agg.uniform_public_alignment_count
                if source_agg.uniform_public_alignment_count else math.nan
            )
            uniform_row["private_stance_alignment_abs"] = (
                source_agg.uniform_private_alignment_abs_sum / source_agg.uniform_private_alignment_count
                if source_agg.uniform_private_alignment_count else math.nan
            )
            uniform_row["public_stance_change_error_abs"] = safe_mean(source_agg.uniform_public_change_abs_errors)
            uniform_row["private_stance_change_error_abs"] = safe_mean(source_agg.uniform_private_change_abs_errors)
            uniform_row["public_opinion_convergence_error"] = safe_mean(source_agg.uniform_public_conv_errors)
            uniform_row["private_opinion_convergence_error"] = safe_mean(source_agg.uniform_private_conv_errors)
            raw_rows.append(uniform_row)
    return raw_rows


def has_plottable_scores(score_rows: list[dict[str, Any]], axes: list | None = None) -> bool:
    axis_keys = [k for k, _ in (axes if axes is not None else RADAR_AXES)]
    for row in score_rows:
        values = [as_float_or_nan(row.get(k)) for k in axis_keys]
        if not any(math.isnan(v) for v in values):
            return True
    return False


def get_model_color(model: str, fallback: Any) -> Any:
    return MODEL_COLOR_MAP.get(model, fallback)


def is_human_model(model: str) -> bool:
    return model.strip().lower() == "human"


def sort_rows_by_model_order(rows: list[dict[str, Any]], model_order: list[str] | None = None) -> list[dict[str, Any]]:
    requested_order = {model_name: index for index, model_name in enumerate(model_order or [])}
    return sorted(
        rows,
        key=lambda row: (
            0 if is_human_model(str(row.get("model", "")).strip()) else 1,
            requested_order.get(str(row.get("model", "")).strip(), len(requested_order)),
            str(row.get("model", "")).strip(),
        ),
    )


def _fixed_v2_llm_order(candidate_models: set[str]) -> list[str]:
    ordered_llm_models = [model_name for model_name in V2_FIXED_MODEL_ORDER if model_name in candidate_models]
    llama_dpo_models = sorted(
        model_name
        for model_name in candidate_models
        if model_name.startswith("ft:Llama-3.1-8B-Instruct-DPO-")
    )
    llama_sft_models = sorted(
        model_name
        for model_name in candidate_models
        if is_llama_sft_model(model_name)
    )
    degroot_models = sorted(model_name for model_name in candidate_models if is_degroot_model(model_name))
    covered_models = set(ordered_llm_models) | set(llama_dpo_models) | set(llama_sft_models) | set(degroot_models)
    uncatalogued_llm_models = sorted(candidate_models - covered_models)
    if llama_sft_models and LLAMA_8B_INSTRUCT_MODEL in ordered_llm_models:
        instruct_index = ordered_llm_models.index(LLAMA_8B_INSTRUCT_MODEL) + 1
        ordered_llm_models = [
            *ordered_llm_models[:instruct_index],
            *llama_sft_models,
            *ordered_llm_models[instruct_index:],
        ]
    else:
        ordered_llm_models = [*ordered_llm_models, *llama_sft_models]
    return [*ordered_llm_models, *llama_dpo_models, *uncatalogued_llm_models, *degroot_models]


def get_v2_display_model_order(
    raw_rows: list[dict[str, Any]],
    *,
    include_human: bool = True,
    allowed_models: list[str] | None = None,
) -> list[str]:
    allowed_set = set(allowed_models) if allowed_models is not None else None
    candidate_models: set[str] = set()
    for row in raw_rows:
        model_name = str(row.get("model", "")).strip()
        if not model_name:
            continue
        if model_name in V2_EXCLUDED_ORDER_MODELS:
            continue
        if allowed_set is not None and model_name not in allowed_set and not is_human_model(model_name):
            continue
        if not include_human and is_human_model(model_name):
            continue
        candidate_models.add(model_name)

    human_models = sorted(model_name for model_name in candidate_models if is_human_model(model_name))
    candidate_llm_models = {model_name for model_name in candidate_models if not is_human_model(model_name)}
    ordered_llm_models = _fixed_v2_llm_order(candidate_llm_models)
    return [*human_models, *ordered_llm_models]


def add_axis_group_legend(fig: Any, anchor: tuple[float, float]) -> None:
    if plt is None:
        return
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=GROUP_TEXT_COLORS["message"], lw=2),
        Line2D([0], [0], color=GROUP_TEXT_COLORS["individual"], lw=2),
        Line2D([0], [0], color=GROUP_TEXT_COLORS["group"], lw=2),
    ]
    labels = ["Message-level metrics", "Individual-level metrics", "Group-level metrics"]
    fig.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=anchor,
        bbox_transform=fig.transFigure,
        frameon=True,
        fontsize=9,
        ncols=3,
        title="Axis Label Colors",
        title_fontsize=9,
        borderaxespad=0.2,
    )


def add_theoretical_bounds_legend(fig: Any, anchor: tuple[float, float]) -> None:
    if plt is None:
        return
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color="#7f7f7f", lw=1.5, linestyle="--"),
        Line2D([0], [0], color="#444444", lw=1.5, linestyle="-"),
    ]
    labels = ["Theoretical lower bound (0.0)", "Theoretical upper bound (1.0)"]
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=anchor,
        bbox_transform=fig.transFigure,
        frameon=True,
        fontsize=9,
        title="Reference Lines",
        title_fontsize=9,
        borderaxespad=0.2,
    )


def plot_profile(
    scores: list[dict[str, Any]],
    out_png: Path,
    title: str,
    axes: list[tuple[str, str]] | None = None,
    y_lim: tuple[float, float] | None = None,
    rank_corr_label: str = "Kendall tau-b",
    model_order: list[str] | None = None,
) -> None:
    if plt is None:
        print("matplotlib not available; skipping profile plots.")
        return
    plot_scores = [row for row in scores if str(row.get("model", "")).strip().lower() != "human"]
    if model_order is not None:
        plot_scores = sort_rows_by_model_order(plot_scores, model_order)
    if not plot_scores:
        print("No scores available; skipping profile plots.")
        return

    axis_keys = [k for k, _ in (axes or RADAR_AXES)]
    axis_labels = [label for _, label in (axes or RADAR_AXES)]
    x = list(range(len(axis_keys)))
    effective_y_lim = y_lim if y_lim is not None else PROFILE_Y_LIM
    is_tau = y_lim is not None and y_lim[0] < 0

    def _draw_and_save(target_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(12, 6.5))
        cmap = plt.cm.get_cmap("tab10", max(10, len(plot_scores)))
        footnotes = get_model_footnotes([str(row["model"]) for row in plot_scores])

        for idx, row in enumerate(plot_scores):
            model = str(row["model"])
            is_perm_lb = model == PERM_LB_MODEL
            is_mean_predictor = model == MEAN_PREDICTOR_MODEL
            is_noisy_ub = model.startswith(NOISY_UB_PREFIX)
            line_color = get_model_color(model, cmap(idx))
            line_style = PROFILE_LINE_STYLES[idx % len(PROFILE_LINE_STYLES)]
            if is_perm_lb:
                line_color = "#888888"
                line_style = (0, (4, 2))
            elif is_mean_predictor:
                line_color = get_model_color(model, "#595959")
                line_style = (0, (6, 2, 1, 2))
            elif is_noisy_ub:
                line_color = "#cc00cc"
                line_style = (0, (2, 1, 1, 1))
            values = [float(row.get(k, math.nan)) for k in axis_keys]
            if any(math.isnan(v) for v in values):
                continue
            lw = 1.5 if is_perm_lb else (2.5 if is_noisy_ub else 2.0)
            marker = "x" if is_perm_lb else ("^" if is_mean_predictor else ("s" if is_noisy_ub else "o"))
            ax.plot(
                x,
                values,
                linewidth=lw,
                label=get_model_display_label(model),
                color=line_color,
                linestyle=line_style,
                marker=marker,
                markersize=4.0,
                markerfacecolor="none",
                markeredgecolor=line_color,
                markeredgewidth=1.2,
            )

        ax.axhline(0.0, color="#7f7f7f", linestyle="--", linewidth=1.0, alpha=0.9, zorder=0)
        if not is_tau:
            ax.axhline(1.0, color="#444444", linestyle="-", linewidth=1.0, alpha=0.9, zorder=0)
        ax.set_ylim(*effective_y_lim)
        if is_tau:
            ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_ylabel(rank_corr_label)
        else:
            ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_ylabel("Normalized Score (Higher is Better)")
        ax.set_xticks(x)
        axis_texts = ax.set_xticklabels(axis_labels, rotation=20, ha="right")
        for text_obj, axis_key in zip(axis_texts, axis_keys):
            group = AXIS_GROUP.get(axis_key, "")
            text_obj.set_color(GROUP_TEXT_COLORS.get(group, "#222222"))
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_title(title)
        handles, labels = ax.get_legend_handles_labels()
        filtered_pairs = [
            (handle, label)
            for handle, label in zip(handles, labels)
            if label != PERM_LB_MODEL
        ]
        if filtered_pairs:
            handles, labels = zip(*filtered_pairs)
            fig.legend(
                handles,
                labels,
                loc="upper left",
                bbox_to_anchor=(0.80, 0.98),
                bbox_transform=fig.transFigure,
                frameon=True,
                fontsize=9,
                title="Models",
                borderaxespad=0.2,
            )
        # Axis label colour legend placed horizontally at the bottom of the figure
        add_axis_group_legend(fig, anchor=(0.01, 0.02))
        add_model_footnotes(fig, footnotes, y=0.01)
        fig.tight_layout(rect=(0.0, 0.10, 0.78, 1.0))
        fig.savefig(target_path, dpi=220, bbox_inches="tight")
        fig.savefig(target_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)

    _draw_and_save(out_png)


def plot_v2_bar_metrics(
    raw_rows: list[dict[str, Any]],
    out_png: Path,
    title: str,
    model_order: list[str] | None = None,
    y_limit_overrides: dict[str, tuple[float, float]] | None = None,
) -> None:
    if plt is None:
        print("matplotlib not available; skipping v2 bar plots.")
        return

    special_models = {"human", *V2_EXCLUDED_ORDER_MODELS}
    uniform_row = next(
        (row for row in raw_rows if str(row.get("model", "")).strip() == UNIFORM_LB_MODEL),
        None,
    )
    plot_rows = [
        row for row in raw_rows
        if str(row.get("model", "")).strip()
        and str(row.get("model", "")).strip().lower() != "human"
        and str(row.get("model", "")).strip() not in special_models
    ]
    if not plot_rows:
        print("No raw rows available; skipping v2 bar plots.")
        return

    reference_row = plot_rows[0]

    def _message_level_rank_key(row: dict[str, Any]) -> float:
        values = [
            as_float_or_nan(row.get("public_stance_alignment_abs")),
            as_float_or_nan(row.get("private_stance_alignment_abs")),
        ]
        finite_values = [value for value in values if not math.isnan(value)]
        if not finite_values:
            return math.inf
        return sum(finite_values) / len(finite_values)

    if model_order:
        plot_rows = sort_rows_by_model_order(plot_rows, model_order)
    else:
        plot_rows = sorted(plot_rows, key=lambda row: (_message_level_rank_key(row), str(row["model"])))

    metric_groups = [
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

    def _draw_and_save(target_path: Path) -> None:
        from matplotlib.lines import Line2D

        fig, axes = plt.subplots(3, 1, figsize=(16, 9.5), sharex=True)

        model_names = [str(row["model"]) for row in plot_rows]
        footnotes = get_model_footnotes(model_names)
        num_models = len(model_names)
        public_positions = list(range(num_models))
        group_gap = 1.8
        private_positions = [position + num_models + group_gap for position in public_positions]
        all_positions = public_positions + private_positions
        bar_width = 0.78
        value_label_fontsize = 8
        model_colors = {
            model_name: get_model_color(model_name, plt.get_cmap("tab10")(index % 10))
            for index, model_name in enumerate(model_names)
        }

        formatted_labels = [format_model_display_label(name) for name in model_names]

        def _draw_error_bar(
            ax: Any,
            center_x: float,
            value: float,
            low: Any,
            high: Any,
            y_limits: tuple[float, float] | None = None,
        ) -> None:
            low_value = as_float_or_nan(low)
            high_value = as_float_or_nan(high)
            if math.isnan(value) or math.isnan(low_value) or math.isnan(high_value):
                return
            if y_limits is not None:
                low_value = max(low_value, y_limits[0])
                high_value = min(high_value, y_limits[1])
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
                elinewidth=2.0,
                capsize=9.0,
                capthick=2.0,
                zorder=5,
            )

        def _annotate_count(ax: Any, center_x: float, count_value: Any, y_value: float) -> None:
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
            )

        public_center = sum(public_positions) / len(public_positions)
        private_center = sum(private_positions) / len(private_positions)
        separator_x = (public_positions[-1] + private_positions[0]) / 2
        ub_line_specs = [
            ("#D62728", f"Noise UB (sigma={LIKERT_GAUSSIAN_SIGMA})"),
            ("#FF9896", f"Noise UB (sigma={LIKERT_GAUSSIAN_SIGMA_SECONDARY})"),
        ]

        for axis_idx, (ax, (level_title, y_label, field_pairs, count_fields, bound_pairs)) in enumerate(zip(axes, metric_groups)):
            max_value = -math.inf
            min_value = math.inf
            label_entries: list[tuple[float, float, str]] = []
            fixed_limits = (y_limit_overrides or {}).get(level_title, V2_BAR_Y_LIMITS.get(level_title))
            public_values = [as_float_or_nan(row.get(field_pairs[0][1])) for row in plot_rows]
            private_values = [as_float_or_nan(row.get(field_pairs[1][1])) for row in plot_rows]
            public_counts = [row.get(count_fields[0]) for row in plot_rows]
            private_counts = [row.get(count_fields[1]) for row in plot_rows]
            public_ci_lows = [row.get(f"{field_pairs[0][1]}_ci_low") for row in plot_rows]
            public_ci_highs = [row.get(f"{field_pairs[0][1]}_ci_high") for row in plot_rows]
            private_ci_lows = [row.get(f"{field_pairs[1][1]}_ci_low") for row in plot_rows]
            private_ci_highs = [row.get(f"{field_pairs[1][1]}_ci_high") for row in plot_rows]

            for positions, series_values, ci_lows, ci_highs in (
                (public_positions, public_values, public_ci_lows, public_ci_highs),
                (private_positions, private_values, private_ci_lows, private_ci_highs),
            ):
                bars = ax.bar(
                    positions,
                    series_values,
                    width=bar_width,
                    color=[model_colors[str(row["model"])] for row in plot_rows],
                    edgecolor="white",
                    linewidth=0.6,
                )
                for bar, value, ci_low, ci_high in zip(bars, series_values, ci_lows, ci_highs):
                    if math.isnan(value):
                        continue
                    max_value = max(max_value, value)
                    min_value = min(min_value, value)
                    center_x = bar.get_x() + bar.get_width() / 2
                    _draw_error_bar(ax, center_x, value, ci_low, ci_high, fixed_limits)
                    low_value = as_float_or_nan(ci_low)
                    high_value = as_float_or_nan(ci_high)
                    if not math.isnan(low_value):
                        min_value = min(min_value, low_value)
                    if not math.isnan(high_value):
                        max_value = max(max_value, high_value)
                    label_anchor = high_value if not math.isnan(high_value) else value
                    if fixed_limits is not None:
                        label_anchor = min(label_anchor, fixed_limits[1] - 0.06)
                    label_entries.append((center_x, label_anchor, f"{value:.2f}"))

            block_specs = [
                (public_positions, public_center, "Public", bound_pairs[0]),
                (private_positions, private_center, "Private", bound_pairs[1]),
            ]
            for positions, _center, _label, (lb_field, primary_ub_field, secondary_ub_field) in block_specs:
                lb_metric_field = lb_field.replace("lb_", "")
                lb_value = as_float_or_nan((uniform_row or {}).get(lb_metric_field))
                ub_values = [
                    as_float_or_nan(reference_row.get(primary_ub_field)),
                    as_float_or_nan(reference_row.get(secondary_ub_field)),
                ]
                for bound_value in [lb_value, *ub_values]:
                    if not math.isnan(bound_value):
                        max_value = max(max_value, bound_value)
                        min_value = min(min_value, bound_value)

            if math.isinf(max_value) or math.isinf(min_value):
                max_value = 0.0
                min_value = 0.0

            if axis_idx == 0:
                rank_y = max_value + max(0.035, (max_value - min_value) * 0.06)
                for model_idx, position in enumerate(public_positions):
                    ax.text(
                        position,
                        rank_y,
                        str(model_idx + 1),
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color="#666666",
                    )

            upper_padding = max(0.16, (max_value - min_value) * 0.34 if max_value > min_value else 0.26)
            y_top = max_value + upper_padding if max_value > 0 else 1.0
            y_bottom = min(0.0, min_value) - 0.02
            if fixed_limits is not None:
                y_bottom, y_top = fixed_limits
                upper_padding = max(0.16, y_top - max_value)
            count_y = y_bottom + max(0.01, (y_top - y_bottom) * 0.03)
            label_offset = max(0.03, (y_top - y_bottom) * 0.035)

            for center_x, label_anchor, label_text in label_entries:
                ax.text(
                    center_x,
                    label_anchor + label_offset,
                    label_text,
                    ha="center",
                    va="bottom",
                    rotation=0,
                    fontsize=7,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.4},
                    zorder=6,
                )

            for positions, center, label, (lb_field, primary_ub_field, secondary_ub_field) in block_specs:
                left_x = positions[0] - 0.42
                right_x = positions[-1] + 0.42
                lb_metric_field = lb_field.replace("lb_", "")
                lb_value = as_float_or_nan((uniform_row or {}).get(lb_metric_field))
                if not math.isnan(lb_value):
                    ax.hlines(
                        lb_value,
                        left_x,
                        right_x,
                        colors="#7F7F7F",
                        linestyles=":",
                        linewidth=1.6,
                        zorder=3,
                    )
                for ub_field, (color, legend_label) in zip(
                    (primary_ub_field, secondary_ub_field),
                    ub_line_specs,
                ):
                    ub_value = as_float_or_nan(reference_row.get(ub_field))
                    if math.isnan(ub_value):
                        continue
                    ax.hlines(
                        ub_value,
                        left_x,
                        right_x,
                        colors=color,
                        linestyles=":",
                        linewidth=1.6,
                        zorder=3,
                        label=legend_label if axis_idx == 0 and label == "Public" else None,
                    )

            ax.axvspan(public_positions[0] - 0.5, public_positions[-1] + 0.5, color="#4C78A8", alpha=0.03)
            ax.axvspan(private_positions[0] - 0.5, private_positions[-1] + 0.5, color="#F58518", alpha=0.03)
            ax.axvline(separator_x, color="#C7C7C7", linewidth=0.9, linestyle="--", alpha=0.75)
            if axis_idx == 0:
                ax.text(public_center, y_top - upper_padding * 0.18, "Public", ha="center", va="top", fontsize=10, fontweight="bold")
                ax.text(private_center, y_top - upper_padding * 0.18, "Private", ha="center", va="top", fontsize=10, fontweight="bold")

            for position, count_value in zip(public_positions, public_counts):
                _annotate_count(ax, position, count_value, count_y)
            for position, count_value in zip(private_positions, private_counts):
                _annotate_count(ax, position, count_value, count_y)

            ax.set_title(level_title, loc="left", fontweight="bold")
            ax.set_ylabel(y_label)
            ax.grid(True, axis="y", alpha=0.25)
            ax.set_axisbelow(True)
            ax.set_xlim(public_positions[0] - 0.9, private_positions[-1] + 0.9)
            ax.set_ylim(y_bottom, y_top)
            ax.tick_params(axis="x", labelbottom=axis_idx == len(axes) - 1, pad=2)

        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.legend(
            handles=[
                Line2D([0], [0], color="#7F7F7F", linestyle=":", linewidth=1.8, label=UNIFORM_LB_MODEL),
                Line2D([0], [0], color="#D62728", linestyle=":", linewidth=1.8, label=f"Noise UB (sigma={LIKERT_GAUSSIAN_SIGMA})"),
                Line2D([0], [0], color="#FF9896", linestyle=":", linewidth=1.8, label=f"Noise UB (sigma={LIKERT_GAUSSIAN_SIGMA_SECONDARY})"),
            ],
            loc="upper right",
            bbox_to_anchor=(0.985, 0.93),
            frameon=False,
            fontsize=9,
        )
        for axis_idx, ax in enumerate(axes):
            ax.set_xticks(all_positions)
            if axis_idx == len(axes) - 1:
                ax.set_xticklabels(formatted_labels + formatted_labels, rotation=48, ha="right")
            else:
                ax.set_xticklabels([])
        add_model_footnotes(fig, footnotes, y=0.01)
        fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.92), h_pad=2.0)
        fig.savefig(target_path, dpi=220, bbox_inches="tight")
        fig.savefig(target_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)

    _draw_and_save(out_png)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate DEBATE metrics and profile plots (combined entry point).")
    parser.add_argument(
        "--data-root",
        default="../../result/eval_v3/human_llm",
        help="Root of human_llm results (default: result/eval_v3/human_llm).",
    )
    parser.add_argument(
        "--fallback-data-root",
        action="append",
        default=[],
        help="Additional data root(s) to search if files are missing.",
    )
    parser.add_argument(
        "--data-prefix-file",
        default="../../inputs/masks_train_test/golden_group_v0/aggregate_universe_mask_golden_v0.txt",
        help="Text file listing data_prefix values.",
    )
    parser.add_argument(
        "--models-file",
        default="models.txt",
        help="Text file listing model names.",
    )
    parser.add_argument(
        "--human-filename",
        default=HUMAN_DEFAULT_FILENAME,
        help=f"Human CSV filename (default: {HUMAN_DEFAULT_FILENAME}).",
    )
    parser.add_argument(
        "--llm-filename",
        default=LLM_DEFAULT_FILENAME,
        help=f"LLM CSV filename (default: {LLM_DEFAULT_FILENAME}).",
    )
    parser.add_argument(
        "--output-dir",
        default="result/radar",
        help="Output directory for report assets (default: result/radar). CSVs are written under output_dir/csv and figures under output_dir/img.",
    )
    parser.add_argument(
        "--likert-col",
        default="likert_pred",
        help="Column for stance labels/scores.",
    )
    parser.add_argument(
        "--human-text-col",
        default="text",
        help="Column for human text.",
    )
    parser.add_argument(
        "--llm-text-col",
        default="llm_text",
        help="Column for LLM text.",
    )
    parser.add_argument(
        "--utterance-event-types",
        default=",".join(DEFAULT_UTTERANCE_EVENT_TYPES),
        help="Comma-separated event types for utterance-level metrics.",
    )
    parser.add_argument(
        "--min-key-match-rate",
        type=float,
        default=0.7,
        help="Minimum key-based row match rate before falling back to index alignment.",
    )
    parser.add_argument(
        "--metric-version",
        choices=("v0", "v0_fixed", "v1", "v2", "spearman", "pearson"),
        default="v0",
        help="Scoring method: v0=original min-max normalization with conditioned alignment included, v0_fixed=min-max normalization with conditioned alignment excluded, v1=Kendall tau-b, v2=v0_fixed-style bounded normalization restricted to the 6 v1-family metrics with permutation/noisy reference lines, spearman=Spearman rho, pearson=Pearson r.",
    )
    parser.add_argument(
        "--no-jitter-ub",
        action="store_true",
        help="Disable the ordinal ±1 jitter upper-bound reference line (v1 only).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip metric computation and render profile plot from existing metrics CSV.",
    )
    parser.add_argument(
        "--plot-input-csv",
        default="",
        help=(
            "Input CSV for --plot-only. If omitted, tries {output_dir}/radar_scores.csv, "
            "then {output_dir}/raw_metrics.csv."
        ),
    )
    parser.add_argument(
        "--plot-input-type",
        choices=("auto", "raw", "scores"),
        default="auto",
        help="Input CSV type for --plot-only (raw metrics or normalized scores).",
    )
    parser.add_argument(
        "--profile-title",
        default="DEBATE Alignment Profile (Higher is Better)",
        help="Title for the all-model profile plot.",
    )
    parser.add_argument(
        "--respect-validity-labels",
        action="store_true",
        help="Filter invalid rows by blanking their Likert labels. Sidecar overrides are used when available; otherwise original CSV validity columns are used.",
    )
    parser.add_argument(
        "--initial-post-validity-sidecar-root",
        action="append",
        default=[],
        help="Optional sidecar root(s) with __initial_post_rejudge.csv files.",
    )
    parser.add_argument(
        "--tweet-validity-sidecar-root",
        action="append",
        default=[],
        help="Optional sidecar root(s) with __tweet_rejudge.csv files.",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Optional filename suffix override such as _v3. Defaults to the metric-version suffix.",
    )
    parser.add_argument(
        "--report-version",
        default="v3",
        help="Optional artifact family selector. v4 excludes Tulu from the base-model roster.",
    )
    parser.add_argument(
        "--skip-v3-additional-group-masks",
        action="store_true",
        help="Do not apply the extra v3 tweet/opinion group masks from masks/groups_with_valid_tweets or masks/groups_with_valid_opinions.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    data_root = Path(args.data_root)
    fallback_roots = [Path(p) for p in args.fallback_data_root]
    roots = [data_root] + fallback_roots
    initial_post_sidecar_roots = [Path(p) for p in args.initial_post_validity_sidecar_root]
    tweet_sidecar_roots = [Path(p) for p in args.tweet_validity_sidecar_root]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = output_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    img_dir = output_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

    score_axes = RADAR_AXES_V1 if args.metric_version in ("v1", "v2", "spearman", "pearson") else RADAR_AXES
    score_fieldnames = ["model"] + [k for k, _ in score_axes]
    plot_y_lim = (0.0, 1.0) if args.metric_version == "v2" else ((-0.05, 1.05) if args.metric_version in ("v1", "spearman", "pearson") else PROFILE_Y_LIM)
    plot_rank_corr_label = {"v1": "Kendall \u03c4-b", "spearman": "Spearman \u03c1", "pearson": "Pearson r"}.get(args.metric_version, "Kendall \u03c4-b")
    plot_suffix = args.output_suffix or {"v1": "_kendall", "v2": "_v2", "spearman": "_spearman", "pearson": "_pearson"}.get(args.metric_version, "")
    raw_metrics_name = f"raw_metrics{plot_suffix}.csv"
    radar_scores_name = f"radar_scores{plot_suffix}.csv"
    raw_metrics_depth_name = f"raw_metrics_depth_topics{plot_suffix}.csv"
    radar_scores_depth_name = f"radar_scores_depth_topics{plot_suffix}.csv"
    raw_metrics_breadth_name = f"raw_metrics_breadth_topics{plot_suffix}.csv"
    radar_scores_breadth_name = f"radar_scores_breadth_topics{plot_suffix}.csv"
    status_name = f"per_prefix_status{plot_suffix}.csv"
    plot_title = args.profile_title
    if args.metric_version == "v1" and plot_title == "DEBATE Alignment Profile (Higher is Better)":
        plot_title = "DEBATE Alignment Profile (Kendall \u03c4-b)"
    elif args.metric_version == "spearman" and plot_title == "DEBATE Alignment Profile (Higher is Better)":
        plot_title = "DEBATE Alignment Profile (Spearman \u03c1)"
    elif args.metric_version == "pearson" and plot_title == "DEBATE Alignment Profile (Higher is Better)":
        plot_title = "DEBATE Alignment Profile (Pearson r)"
    elif args.metric_version == "v2" and plot_title == "DEBATE Alignment Profile (Higher is Better)":
        plot_title = "DEBATE Alignment Profile (v2)"

    if args.plot_only:
        if args.plot_input_csv:
            plot_input_path = Path(args.plot_input_csv)
        else:
            score_default = csv_dir / radar_scores_name
            raw_default = csv_dir / raw_metrics_name
            if score_default.exists():
                plot_input_path = score_default
            elif raw_default.exists():
                plot_input_path = raw_default
            else:
                print(
                    "No plot input found. Provide --plot-input-csv or create "
                    f"{score_default} / {raw_default} first."
                )
                return 1

        if not plot_input_path.exists():
            print(f"Plot input CSV not found: {plot_input_path}")
            return 1

        plot_rows = read_csv_rows(plot_input_path)
        input_type = args.plot_input_type
        if input_type == "auto":
            input_type = infer_plot_input_type(plot_rows)
        if input_type == "unknown":
            print(
                "Could not infer input CSV type. Expected either raw metrics columns or normalized score columns. "
                "Use --plot-input-type raw|scores."
            )
            return 1

        if input_type == "raw":
            raw_rows_for_plot = [
                row for row in plot_rows if str(row.get("model", "")).strip().lower() != "human"
            ]
            score_rows = build_score_rows_from_raw_rows(raw_rows_for_plot, metric_version=args.metric_version)
            save_csv(csv_dir / radar_scores_name, score_rows, score_fieldnames)
        else:
            score_rows = []
            for row in plot_rows:
                if str(row.get("model", "")).strip().lower() == "human":
                    continue
                out_row: dict[str, Any] = {"model": str(row.get("model", ""))}
                for key, _ in score_axes:
                    out_row[key] = as_float_or_nan(row.get(key))
                score_rows.append(out_row)

        display_model_order = get_v2_display_model_order(raw_rows) if args.metric_version == "v2" else None

        plot_profile(
            score_rows,
            out_png=img_dir / "profile_all_models.png",
            title=plot_title,
            axes=score_axes,
            y_lim=plot_y_lim,
            model_order=display_model_order,
        )

        print(f"Plot input: {plot_input_path} ({input_type})")
        print(f"Wrote: {img_dir / 'profile_all_models.png'}")
        return 0

    data_prefixes = read_lines(Path(args.data_prefix_file))
    models = ordered_base_model_llms_for_report(args.report_version, read_lines(Path(args.models_file)))

    utterance_event_types = {
        t.strip() for t in args.utterance_event_types.split(",") if t.strip()
    }

    aggregates = build_model_aggregates(models)
    depth_aggregates = build_model_aggregates(models)
    breadth_aggregates = build_model_aggregates(models)
    status_rows: list[dict[str, Any]] = []
    prefix_results_by_prefix: dict[str, dict[str, PrefixModelResult]] = defaultdict(dict)
    depth_prefix_results_by_prefix: dict[str, dict[str, PrefixModelResult]] = defaultdict(dict)
    breadth_prefix_results_by_prefix: dict[str, dict[str, PrefixModelResult]] = defaultdict(dict)

    prefix_model_pairs = [(data_prefix, model) for data_prefix in data_prefixes for model in models]
    pair_iter: Any = prefix_model_pairs
    if not args.no_progress and tqdm is not None:
        pair_iter = tqdm(
            prefix_model_pairs,
            total=len(prefix_model_pairs),
            desc="Computing metrics",
            unit="pair",
        )
    elif not args.no_progress and tqdm is None:
        print("tqdm not installed; running without progress bar.")

    for data_prefix, model in pair_iter:
        result = compute_prefix_model(
            data_prefix=data_prefix,
            model=model,
            roots=roots,
            human_filename=args.human_filename,
            llm_filename=args.llm_filename,
            utterance_event_types=utterance_event_types,
            likert_col=args.likert_col,
            human_text_col=args.human_text_col,
            llm_text_col=args.llm_text_col,
            min_key_match_rate=args.min_key_match_rate,
            include_conditioned_alignment=args.metric_version == "v0",
            respect_validity_labels=args.respect_validity_labels,
            initial_post_sidecar_roots=initial_post_sidecar_roots,
            tweet_sidecar_roots=tweet_sidecar_roots,
            apply_v3_additional_group_masks=not args.skip_v3_additional_group_masks,
        )

        status_rows.append(
            {
                "data_prefix": data_prefix,
                "model": model,
                "status": result.status,
                "message": result.message,
                "human_path": result.human_path,
                "llm_path": result.llm_path,
                "alignment_mode": result.alignment_mode,
                "key_match_rate": f"{result.key_match_rate:.6f}" if result.status == "ok" else "",
                "aligned_pairs": result.aligned_pairs if result.status == "ok" else "",
            }
        )

        merge_prefix_result_into_aggregate(aggregates[model], result)
        prefix_results_by_prefix[data_prefix][model] = result
        if is_depth_topic(data_prefix):
            merge_prefix_result_into_aggregate(depth_aggregates[model], result)
            depth_prefix_results_by_prefix[data_prefix][model] = result
        else:
            merge_prefix_result_into_aggregate(breadth_aggregates[model], result)
            breadth_prefix_results_by_prefix[data_prefix][model] = result

    ub_mode = "gaussian_likert" if args.metric_version == "v2" else "jitter"
    raw_rows = build_raw_rows_from_aggregates(aggregates, models, ub_mode=ub_mode)
    depth_raw_rows = build_raw_rows_from_aggregates(depth_aggregates, models, ub_mode=ub_mode)
    breadth_raw_rows = build_raw_rows_from_aggregates(breadth_aggregates, models, ub_mode=ub_mode)
    if args.metric_version == "v2":
        raw_rows = add_v2_bootstrap_cis_to_raw_rows(
            raw_rows,
            models=models,
            prefixes=data_prefixes,
            prefix_results_by_prefix=prefix_results_by_prefix,
        )
        depth_prefixes = [data_prefix for data_prefix in data_prefixes if is_depth_topic(data_prefix)]
        breadth_prefixes = [data_prefix for data_prefix in data_prefixes if not is_depth_topic(data_prefix)]
        depth_raw_rows = add_v2_bootstrap_cis_to_raw_rows(
            depth_raw_rows,
            models=models,
            prefixes=depth_prefixes,
            prefix_results_by_prefix=depth_prefix_results_by_prefix,
        )
        breadth_raw_rows = add_v2_bootstrap_cis_to_raw_rows(
            breadth_raw_rows,
            models=models,
            prefixes=breadth_prefixes,
            prefix_results_by_prefix=breadth_prefix_results_by_prefix,
        )

    raw_rows = [row for row in raw_rows if str(row.get("model", "")).strip().lower() != "human"]
    depth_raw_rows = [row for row in depth_raw_rows if str(row.get("model", "")).strip().lower() != "human"]
    breadth_raw_rows = [row for row in breadth_raw_rows if str(row.get("model", "")).strip().lower() != "human"]
    score_rows = build_score_rows_from_raw_rows(raw_rows, metric_version=args.metric_version)
    depth_score_rows = build_score_rows_from_raw_rows(depth_raw_rows, metric_version=args.metric_version)
    breadth_score_rows = build_score_rows_from_raw_rows(breadth_raw_rows, metric_version=args.metric_version)
    if args.metric_version in ("v1", "v2", "spearman", "pearson"):
        include_noisy_ub = not (args.metric_version == "v1" and args.no_jitter_ub)
        score_rows = append_reference_score_rows_v1(
            score_rows,
            raw_rows,
            metric_version=args.metric_version,
            include_noisy_ub=include_noisy_ub,
        )
        depth_score_rows = append_reference_score_rows_v1(
            depth_score_rows,
            depth_raw_rows,
            metric_version=args.metric_version,
            include_noisy_ub=include_noisy_ub,
        )
        breadth_score_rows = append_reference_score_rows_v1(
            breadth_score_rows,
            breadth_raw_rows,
            metric_version=args.metric_version,
            include_noisy_ub=include_noisy_ub,
        )

    raw_fieldnames = collect_fieldnames(raw_rows)
    status_fieldnames = [
        "data_prefix",
        "model",
        "status",
        "message",
        "human_path",
        "llm_path",
        "alignment_mode",
        "key_match_rate",
        "aligned_pairs",
    ]

    save_csv(csv_dir / raw_metrics_name, raw_rows, raw_fieldnames)
    save_csv(csv_dir / radar_scores_name, score_rows, score_fieldnames)
    save_csv(csv_dir / raw_metrics_depth_name, depth_raw_rows, raw_fieldnames)
    save_csv(csv_dir / radar_scores_depth_name, depth_score_rows, score_fieldnames)
    save_csv(csv_dir / raw_metrics_breadth_name, breadth_raw_rows, raw_fieldnames)
    save_csv(csv_dir / radar_scores_breadth_name, breadth_score_rows, score_fieldnames)
    save_csv(csv_dir / status_name, status_rows, status_fieldnames)

    display_model_order = get_v2_display_model_order(raw_rows) if args.metric_version == "v2" else None
    depth_display_model_order = get_v2_display_model_order(depth_raw_rows) if args.metric_version == "v2" else None
    breadth_display_model_order = get_v2_display_model_order(breadth_raw_rows) if args.metric_version == "v2" else None

    generate_profile_plots = plot_suffix != "_v3"
    if generate_profile_plots:
        plot_profile(
            score_rows,
            out_png=img_dir / f"profile_all_models{plot_suffix}.png",
            title=plot_title,
            axes=score_axes,
            y_lim=plot_y_lim,
            rank_corr_label=plot_rank_corr_label,
            model_order=display_model_order,
        )
        if has_plottable_scores(depth_score_rows, axes=score_axes):
            plot_profile(
                depth_score_rows,
                out_png=img_dir / f"profile_depth_topics{plot_suffix}.png",
                title=f"{plot_title} (Depth Topics)",
                axes=score_axes,
                y_lim=plot_y_lim,
                rank_corr_label=plot_rank_corr_label,
                model_order=depth_display_model_order,
            )
        if has_plottable_scores(breadth_score_rows, axes=score_axes):
            plot_profile(
                breadth_score_rows,
                out_png=img_dir / f"profile_breadth_topics{plot_suffix}.png",
                title=f"{plot_title} (Breadth Topics)",
                axes=score_axes,
                y_lim=plot_y_lim,
                rank_corr_label=plot_rank_corr_label,
                model_order=breadth_display_model_order,
            )

    print(f"Wrote: {csv_dir / raw_metrics_name}")
    print(f"Wrote: {csv_dir / radar_scores_name}")
    print(f"Wrote: {csv_dir / raw_metrics_depth_name}")
    print(f"Wrote: {csv_dir / radar_scores_depth_name}")
    print(f"Wrote: {csv_dir / raw_metrics_breadth_name}")
    print(f"Wrote: {csv_dir / radar_scores_breadth_name}")
    print(f"Wrote: {csv_dir / status_name}")
    if plt is not None:
        if generate_profile_plots:
            print(f"Wrote: {img_dir / f'profile_all_models{plot_suffix}.png'}")
            if has_plottable_scores(depth_score_rows, axes=score_axes):
                print(f"Wrote: {img_dir / f'profile_depth_topics{plot_suffix}.png'}")
            else:
                print("Skipped depth-topic profile plot due to insufficient valid scores.")
            if has_plottable_scores(breadth_score_rows, axes=score_axes):
                print(f"Wrote: {img_dir / f'profile_breadth_topics{plot_suffix}.png'}")
            else:
                print("Skipped breadth-topic profile plot due to insufficient valid scores.")
        else:
            print("Skipped v3 profile plots; only bar plots remain supported for v3 summary outputs.")
        if args.metric_version == "v2":
            plot_v2_bar_metrics(
                raw_rows,
                out_png=img_dir / f"bar_all_models{plot_suffix}.png",
                title="DEBATE Alignment Bar Plots (All Topics)",
                model_order=display_model_order,
                y_limit_overrides={"Group-level": (-0.02, 1.6)},
            )
            plot_v2_bar_metrics(
                breadth_raw_rows,
                out_png=img_dir / f"bar_breadth_topics{plot_suffix}.png",
                title="DEBATE Alignment Bar Plots (Breadth Topics)",
                model_order=breadth_display_model_order,
                y_limit_overrides={"Group-level": (-0.02, 1.6)},
            )
            plot_v2_bar_metrics(
                depth_raw_rows,
                out_png=img_dir / f"bar_depth_topics{plot_suffix}.png",
                title="DEBATE Alignment Bar Plots (Depth Topics)",
                model_order=depth_display_model_order,
                y_limit_overrides={"Group-level": (-0.02, 1.6)},
            )
            print(f"Wrote: {img_dir / f'bar_all_models{plot_suffix}.png'}")
            print(f"Wrote: {img_dir / f'bar_breadth_topics{plot_suffix}.png'}")
            print(f"Wrote: {img_dir / f'bar_depth_topics{plot_suffix}.png'}")
    else:
        print("Skipped plotting because matplotlib is unavailable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
