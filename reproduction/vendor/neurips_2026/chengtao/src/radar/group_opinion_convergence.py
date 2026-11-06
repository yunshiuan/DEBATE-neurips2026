#!/usr/bin/env python3
"""Generate raw group opinion convergence summary plots for v2."""

from __future__ import annotations

import argparse
import math
import random
import re
import tempfile
import textwrap
from pathlib import Path
from typing import Any

from post_training_breadth_profiles import BASE_MODELS, MODEL_SET_CONFIGS, unique_models
from neurips_2026.sean_reproduce.base_model_v3_style import CANONICAL_V3_HUMAN_SOURCE_MODEL, ordered_base_model_llms_for_report
from neurips_2026.sean_reproduce.result_version_config import (
    LLAMA_V4_BREADTH_GROUP_SFT_MODEL,
    LLAMA_V4_BREADTH_TOPIC_SFT_MODEL,
    LLAMA_V4_DEPTH_GROUP_SFT_MODEL,
    LLAMA_V4_DEPTH_TOPIC_SFT_MODEL,
    report_family_for_version,
)
from neurips_2026.sean_reproduce.v3.masks import configured_private_mask_root, configured_public_mask_root, repo_relative_display_path, wrapped_path_lines

from generate_plots import (
    BASE_MODEL_V3_LLM_ORDER,
    DEPTH_TOPICS,
    HUMAN_DEFAULT_FILENAME,
    LLM_DEFAULT_FILENAME,
    build_profile,
    collect_fieldnames,
    compute_group_deltas,
    format_model_display_label,
    get_model_color,
    get_model_footnotes,
    get_v2_display_model_order,
    parse_likert,
    parse_round,
    read_csv_rows,
    read_lines,
    resolve_file,
    save_csv,
    safe_pstdev,
    sort_rows_by_model_order,
    tqdm,
)

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
except Exception:  # pragma: no cover
    plt = None
    PdfPages = None

try:
    from pypdf import PdfWriter
except Exception:  # pragma: no cover
    PdfWriter = None


def is_depth_topic(data_prefix: str) -> bool:
    parts = data_prefix.split("_")
    if len(parts) >= 4 and parts[0].isdigit() and parts[1].isdigit():
        topic = "_".join(parts[2:-1])
    else:
        topic = data_prefix
    return topic in DEPTH_TOPICS


def safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def extract_topic_name(data_prefix: str) -> str:
    parts = data_prefix.split("_")
    if len(parts) >= 4 and parts[0].isdigit() and parts[1].isdigit():
        return "_".join(parts[2:-1])
    return data_prefix


def slugify(value: str) -> str:
    return "_".join(part for part in "".join(ch if ch.isalnum() else " " for ch in str(value)).lower().split()) or "topic"


POST_TRAIN_PARTITION_SPECS = [
    ("breadth_topic_train", "../../inputs/masks_train_test/golden_group_v0/breadth/breadth_train_topic_mask_golden_v0.txt", "topic_models", "Breadth Topic Partition - Train"),
    ("breadth_topic_test", "../../inputs/masks_train_test/golden_group_v0/breadth/breadth_test_topic_mask_golden_v0.txt", "topic_models", "Breadth Topic Partition - Test"),
    ("breadth_group_train", "../../inputs/masks_train_test/golden_group_v0/breadth/breadth_train_group_mask_golden_v0.txt", "group_models", "Breadth Group Partition - Train"),
    ("breadth_group_test", "../../inputs/masks_train_test/golden_group_v0/breadth/breadth_test_group_mask_golden_v0.txt", "group_models", "Breadth Group Partition - Test"),
    ("depth_topic_train", "../../inputs/masks_train_test/golden_group_v0/depth/depth_train_topic_mask_golden_v0.txt", "topic_models", "Depth Topic Partition - Train"),
    ("depth_topic_test", "../../inputs/masks_train_test/golden_group_v0/depth/depth_test_topic_mask_golden_v0.txt", "topic_models", "Depth Topic Partition - Test"),
    ("depth_group_train", "../../inputs/masks_train_test/golden_group_v0/depth/depth_train_group_mask_golden_v0.txt", "group_models", "Depth Group Partition - Train"),
    ("depth_group_test", "../../inputs/masks_train_test/golden_group_v0/depth/depth_test_group_mask_golden_v0.txt", "group_models", "Depth Group Partition - Test"),
]

POST_TRAIN_REPORT_SPECS = [
    ("post-training", "models", "report_group_opinion_convergence_posttrain_models_v2.pdf"),
    ("post-training-vs-all", "models_vs_all", "report_group_opinion_convergence_posttrain_models_vs_all_v2.pdf"),
    ("post-training-v4", "models_v4", "report_group_opinion_convergence_posttrain_models_v4.pdf"),
    ("post-training-v4-vs-all", "models_vs_all", "report_group_opinion_convergence_posttrain_models_vs_all_v4.pdf"),
]

POST_TRAIN_BREADTH_TOPIC_MODELS = [
    "Llama-3.1-8B-Instruct",
    "ft:Llama-3.1-8B-SFT-20260210:breadth-topic-3epochs-unsloth",
    "ft:Llama-3.1-8B-Instruct-DPO-20260212:breadth-topic-5epochs",
]

POST_TRAIN_BREADTH_GROUP_MODELS = [
    "Llama-3.1-8B-Instruct",
    "ft:Llama-3.1-8B-SFT-20260210:breadth-group-3epochs-unsloth",
    "ft:Llama-3.1-8B-Instruct-DPO-20260211:breadth-group-5epochs",
]

POST_TRAIN_DEPTH_TOPIC_MODELS = [
    "Llama-3.1-8B-Instruct",
    "gpt-4o-mini-2024-07-18",
    "ft:Llama-3.1-8B-Instruct-SFT-20251218:depth-topic-5epochs",
    "ft:gpt-4o-mini-2024-07-18:camer:sft-20251218-depth-topic-5epochs:CoFxFPcv",
    "ft:Llama-3.1-8B-Instruct-DPO-20260105:depth-topic-5epochs",
]

POST_TRAIN_DEPTH_GROUP_MODELS = [
    "Llama-3.1-8B-Instruct",
    "gpt-4o-mini-2024-07-18",
    "ft:Llama-3.1-8B-Instruct-SFT-20251218:depth-group-5epochs",
    "ft:gpt-4o-mini-2024-07-18:camer:sft-20251218-depth-group-5epochs:CoFzijg0",
    "ft:Llama-3.1-8B-Instruct-DPO-20260105:depth-group-5epochs",
]

POST_TRAIN_BREADTH_TOPIC_MODELS_V4 = [
    "Llama-3.1-8B-Instruct",
    LLAMA_V4_BREADTH_TOPIC_SFT_MODEL,
]

POST_TRAIN_BREADTH_GROUP_MODELS_V4 = [
    "Llama-3.1-8B-Instruct",
    LLAMA_V4_BREADTH_GROUP_SFT_MODEL,
]

POST_TRAIN_DEPTH_TOPIC_MODELS_V4 = [
    "Llama-3.1-8B-Instruct",
    "gpt-4o-mini-2024-07-18",
    LLAMA_V4_DEPTH_TOPIC_SFT_MODEL,
    "ft:gpt-4o-mini-2024-07-18:camer:sft-20251218-depth-topic-5epochs:CoFxFPcv",
]

POST_TRAIN_DEPTH_GROUP_MODELS_V4 = [
    "Llama-3.1-8B-Instruct",
    "gpt-4o-mini-2024-07-18",
    LLAMA_V4_DEPTH_GROUP_SFT_MODEL,
    "ft:gpt-4o-mini-2024-07-18:camer:sft-20251218-depth-group-5epochs:CoFzijg0",
]

TULU_BASE_MODEL = "Llama-3.1-Tulu-3-8B-SFT"
V4_BASE_MODELS = [model for model in BASE_MODELS if model != TULU_BASE_MODEL]

POST_TRAIN_BREADTH_TOPIC_MODELS_V4_VS_ALL = unique_models(POST_TRAIN_BREADTH_TOPIC_MODELS_V4, V4_BASE_MODELS)
POST_TRAIN_BREADTH_GROUP_MODELS_V4_VS_ALL = unique_models(POST_TRAIN_BREADTH_GROUP_MODELS_V4, V4_BASE_MODELS)
POST_TRAIN_DEPTH_TOPIC_MODELS_V4_VS_ALL = unique_models(POST_TRAIN_DEPTH_TOPIC_MODELS_V4, V4_BASE_MODELS)
POST_TRAIN_DEPTH_GROUP_MODELS_V4_VS_ALL = unique_models(POST_TRAIN_DEPTH_GROUP_MODELS_V4, V4_BASE_MODELS)

REPO_ROOT = Path(__file__).resolve().parents[4]
AGGREGATE_GOLDEN_MASK = REPO_ROOT / "neurips_2026" / "chengtao" / "inputs" / "masks_train_test" / "golden_group_v0" / "aggregate_universe_mask_golden_v0.txt"
BREADTH_GOLDEN_MASK = REPO_ROOT / "neurips_2026" / "chengtao" / "inputs" / "masks_train_test" / "golden_group_v0" / "breadth" / "breadth_universe_mask_golden_v0.txt"
DEPTH_GOLDEN_MASK = REPO_ROOT / "neurips_2026" / "chengtao" / "inputs" / "masks_train_test" / "golden_group_v0" / "depth" / "depth_universe_mask_golden_v0.txt"


def _append_v3_mask_cover_body(cover_body: str, *, output_suffix: str, uses_split_masks: bool) -> str:
    if output_suffix not in {"_v3", "_v3_b", "_v4", "_v4_b"}:
        return cover_body
    mask_lines = [
        "",
        "Mask setup:",
        "Base golden masks:",
        *wrapped_path_lines(AGGREGATE_GOLDEN_MASK, indent="  Aggregate: "),
        *wrapped_path_lines(BREADTH_GOLDEN_MASK, indent="  Breadth collection: "),
        *wrapped_path_lines(DEPTH_GOLDEN_MASK, indent="  Depth collection: "),
    ]
    if uses_split_masks:
        mask_lines.append("Split masks: train/test group/topic masks are applied on partitioned post-training pages as named by each page title.")
    else:
        mask_lines.append("Split masks: none for this report; pages use only the aggregate or breadth/depth collection masks listed above.")
    mask_lines.extend(
        [
            "Additional complete-group masks:",
            *wrapped_path_lines(configured_public_mask_root(), indent="  Public/tweet: "),
            *wrapped_path_lines(configured_private_mask_root(), indent="  Private/opinion: "),
        ]
    )
    return "\n".join([cover_body.strip(), *mask_lines])


def skip_v4_family_breadth_topic_per_topic_report(report_version: str, reference_suffix: str) -> bool:
    return report_family_for_version(report_version) == "v4" and reference_suffix.startswith("breadth_topic")


def resolve_post_train_partition_models(*, model_set_key: str, suffix: str) -> list[str] | None:
    if model_set_key not in {"post-training", "post-training-vs-all", "post-training-v4", "post-training-v4-vs-all"}:
        return None

    if model_set_key == "post-training-v4":
        if suffix.startswith("breadth_topic"):
            return POST_TRAIN_BREADTH_TOPIC_MODELS_V4
        if suffix.startswith("breadth_group"):
            return POST_TRAIN_BREADTH_GROUP_MODELS_V4
        if suffix.startswith("depth_topic"):
            return POST_TRAIN_DEPTH_TOPIC_MODELS_V4
        if suffix.startswith("depth_group"):
            return POST_TRAIN_DEPTH_GROUP_MODELS_V4
        return None

    if model_set_key == "post-training-v4-vs-all":
        if suffix.startswith("breadth_topic"):
            return POST_TRAIN_BREADTH_TOPIC_MODELS_V4_VS_ALL
        if suffix.startswith("breadth_group"):
            return POST_TRAIN_BREADTH_GROUP_MODELS_V4_VS_ALL
        if suffix.startswith("depth_topic"):
            return POST_TRAIN_DEPTH_TOPIC_MODELS_V4_VS_ALL
        if suffix.startswith("depth_group"):
            return POST_TRAIN_DEPTH_GROUP_MODELS_V4_VS_ALL
        return None

    if suffix.startswith("breadth_topic"):
        models = POST_TRAIN_BREADTH_TOPIC_MODELS
    elif suffix.startswith("breadth_group"):
        models = POST_TRAIN_BREADTH_GROUP_MODELS
    elif suffix.startswith("depth_topic"):
        models = POST_TRAIN_DEPTH_TOPIC_MODELS
    elif suffix.startswith("depth_group"):
        models = POST_TRAIN_DEPTH_GROUP_MODELS
    else:
        return None

    if model_set_key == "post-training-vs-all":
        return unique_models(models, BASE_MODELS)
    return models


DISPLAY_LABEL_OVERRIDES = {
    "ft:Llama-3.1-8B-Instruct-SFT-20251218:depth-group-5epochs": "Llama SFT",
    "ft:Llama-3.1-8B-Instruct-DPO-20260105:depth-group-5epochs": "LLaMA DPO Group",
    "ft:Llama-3.1-8B-SFT-20260210:breadth-group-3epochs-unsloth": "Llama SFT",
    "ft:Llama-3.1-8B-Instruct-DPO-20260211:breadth-group-5epochs": "LLaMA DPO Group",
    LLAMA_V4_BREADTH_GROUP_SFT_MODEL: "Llama SFT",
    LLAMA_V4_DEPTH_GROUP_SFT_MODEL: "Llama SFT",
}


def format_convergence_model_label(model_name: str) -> str:
    return DISPLAY_LABEL_OVERRIDES.get(model_name, format_model_display_label(model_name))


def build_wrapped_model_footnotes(model_names: list[str], *, width: int = 135) -> list[str]:
    wrapped_lines: list[str] = []
    seen: set[str] = set()
    for model_name in model_names:
        if model_name in seen:
            continue
        seen.add(model_name)
        display_name = DISPLAY_LABEL_OVERRIDES.get(model_name)
        if display_name is None:
            display_name = next(iter(get_model_footnotes([model_name])), None)
            if display_name is not None and " = " in display_name:
                display_name = display_name.split(" = ", 1)[0]
        if display_name is None:
            continue
        footnote = f"{display_name} = {model_name}"
        wrapped_lines.extend(
            textwrap.wrap(
                footnote,
                width=width,
                break_long_words=False,
                break_on_hyphens=False,
            )
            or [footnote]
        )
    return wrapped_lines


def add_wrapped_model_footnotes(fig: Any, footnotes: list[str], *, y: float = 0.01) -> None:
    if not footnotes:
        return
    fig.text(
        0.01,
        y,
        "Short labels: " + "\n".join(footnotes),
        ha="left",
        va="bottom",
        fontsize=8,
        color="#555555",
    )

CONSENSUS_FLOOR_MODEL = "One-Step Toward First Partner"
HALF_SD_REDUCTION_MODEL = "50% SD Reduction"
REFERENCE_LINE_SPECS = [
    (CONSENSUS_FLOOR_MODEL, "#1F77B4", CONSENSUS_FLOOR_MODEL),
    (HALF_SD_REDUCTION_MODEL, "#7F7F7F", HALF_SD_REDUCTION_MODEL),
]
REQUIRED_COMPLETE_PARTICIPANTS = 4


def participant_key(row: dict[str, str]) -> str | None:
    for key in ("empirica_id", "worker_id", "sender_id"):
        value = (row.get(key) or "").strip()
        if value:
            return value
    return None


def build_first_partner_map(rows: list[dict[str, str]], *, data_prefix: str, likert_col: str) -> dict[str, dict[str, str]]:
    first_partner: dict[str, dict[str, str]] = {}
    for row in rows:
        pid = participant_key(row)
        if not pid:
            continue
        if (row.get("event_type") or "").strip() != "tweet":
            continue
        if parse_round(row.get("chat_round_order")) != 1:
            continue
        if parse_likert(row.get(likert_col)) is None:
            continue
        recipient = (row.get("recipient_id") or "").strip()
        if not recipient:
            continue
        gid = (row.get("group_id") or "").strip() or data_prefix
        first_partner.setdefault(gid, {})[pid] = recipient
    return first_partner


def strict_private_participants(profile: Any, gid: str) -> list[str]:
    init = profile.initial_private.get(gid, {})
    final = profile.final_private.get(gid, {})
    participants = sorted(set(init.keys()) & set(final.keys()))
    if len(participants) != REQUIRED_COMPLETE_PARTICIPANTS:
        return []
    return participants


def strict_public_participants(profile: Any, gid: str) -> list[str]:
    tweet = profile.tweet_round.get(gid, {})
    participants = sorted(
        participant
        for participant, rounds in tweet.items()
        if 1 in rounds and 2 in rounds and 3 in rounds
    )
    if len(participants) != REQUIRED_COMPLETE_PARTICIPANTS:
        return []
    return participants


def compute_strict_group_deltas(profile: Any) -> tuple[dict[str, float], dict[str, float]]:
    delta_public: dict[str, float] = {}
    delta_private: dict[str, float] = {}

    groups = set(profile.initial_private.keys()) | set(profile.final_private.keys()) | set(profile.tweet_round.keys())
    for gid in groups:
        init = profile.initial_private.get(gid, {})
        final = profile.final_private.get(gid, {})
        tweet = profile.tweet_round.get(gid, {})

        private_participants = strict_private_participants(profile, gid)
        if private_participants:
            init_vals = [init[participant] for participant in private_participants]
            final_vals = [final[participant] for participant in private_participants]
            init_sd = safe_pstdev(init_vals)
            final_sd = safe_pstdev(final_vals)
            if not math.isnan(init_sd) and not math.isnan(final_sd):
                delta_private[gid] = final_sd - init_sd

        public_participants = strict_public_participants(profile, gid)
        if public_participants:
            round1_vals = [tweet[participant][1] for participant in public_participants]
            round3_vals = [tweet[participant][3] for participant in public_participants]
            round1_sd = safe_pstdev(round1_vals)
            round3_sd = safe_pstdev(round3_vals)
            if not math.isnan(round1_sd) and not math.isnan(round3_sd):
                delta_public[gid] = round3_sd - round1_sd

    return delta_public, delta_private


def compute_profile_delta_lists(
    rows: list[dict[str, str]], *, data_prefix: str, likert_col: str,
) -> tuple[list[float], list[float]]:
    profile = build_profile(rows, default_group=data_prefix, likert_col=likert_col)
    public_delta, private_delta = compute_strict_group_deltas(profile)
    return list(public_delta.values()), list(private_delta.values())


def compute_consensus_floor_lists(
    rows: list[dict[str, str]], *, data_prefix: str, likert_col: str,
) -> tuple[list[float], list[float]]:
    round1_public: dict[str, dict[str, float]] = {}
    initial_private: dict[str, dict[str, float]] = {}
    first_partner: dict[str, dict[str, str]] = {}

    def partner_key(row: dict[str, str]) -> str | None:
        for key in ("sender_id", "worker_id", "empirica_id"):
            value = (row.get(key) or "").strip()
            if value:
                return value
        return None

    def step_toward(current: float, target: float) -> float:
        if target > current:
            return min(6.0, current + 1.0)
        if target < current:
            return max(1.0, current - 1.0)
        return current

    for row in rows:
        pid = partner_key(row)
        if not pid:
            continue
        gid = (row.get("group_id") or "").strip() or data_prefix
        event_type = (row.get("event_type") or "").strip()
        stance = parse_likert(row.get(likert_col))

        if event_type == "Initial Opinion" and stance is not None:
            initial_private.setdefault(gid, {})[pid] = stance
            continue

        if event_type != "tweet":
            continue

        round_idx = parse_round(row.get("chat_round_order"))
        if round_idx != 1 or stance is None:
            continue
        round1_public.setdefault(gid, {}).setdefault(pid, stance)
        recipient = (row.get("recipient_id") or "").strip()
        if recipient:
            first_partner.setdefault(gid, {}).setdefault(pid, recipient)

    public_floor: list[float] = []
    private_floor: list[float] = []

    for gid in sorted(set(round1_public.keys()) | set(initial_private.keys()) | set(first_partner.keys())):
        public_init = round1_public.get(gid, {})
        private_init = initial_private.get(gid, {})
        partners = first_partner.get(gid, {})

        private_participants = [
            pid for pid in sorted(private_init.keys())
            if pid in partners and partners[pid] in private_init
        ]
        if len(private_participants) >= 2:
            init_vals = [private_init[p] for p in private_participants]
            init_sd = safe_pstdev(init_vals)
            if not math.isnan(init_sd):
                shifted_vals = [step_toward(private_init[p], private_init[partners[p]]) for p in private_participants]
                shifted_sd = safe_pstdev(shifted_vals)
                if not math.isnan(shifted_sd):
                    private_floor.append(shifted_sd - init_sd)

        tweet_participants = [
            pid for pid in sorted(public_init.keys())
            if pid in partners and partners[pid] in public_init
        ]
        if len(tweet_participants) >= 2:
            round1_vals = [public_init[p] for p in tweet_participants]
            round1_sd = safe_pstdev(round1_vals)
            if not math.isnan(round1_sd):
                shifted_vals = [step_toward(public_init[p], public_init[partners[p]]) for p in tweet_participants]
                shifted_sd = safe_pstdev(shifted_vals)
                if not math.isnan(shifted_sd):
                    public_floor.append(shifted_sd - round1_sd)

    return public_floor, private_floor


def compute_half_sd_reduction_lists(
    rows: list[dict[str, str]], *, data_prefix: str, likert_col: str,
) -> tuple[list[float], list[float]]:
    profile = build_profile(rows, default_group=data_prefix, likert_col=likert_col)
    public_floor: list[float] = []
    private_floor: list[float] = []

    for gid in sorted(set(profile.initial_private.keys()) | set(profile.final_private.keys()) | set(profile.tweet_round.keys())):
        init = profile.initial_private.get(gid, {})
        final = profile.final_private.get(gid, {})
        tweet = profile.tweet_round.get(gid, {})

        private_participants = sorted(set(init.keys()) & set(final.keys()))
        if len(private_participants) >= 2:
            init_vals = [init[p] for p in private_participants]
            init_sd = safe_pstdev(init_vals)
            if not math.isnan(init_sd):
                private_floor.append(-0.5 * init_sd)

        tweet_participants = [participant for participant, rounds in tweet.items() if 1 in rounds and 3 in rounds]
        if len(tweet_participants) >= 2:
            round1_vals = [tweet[p][1] for p in tweet_participants]
            round1_sd = safe_pstdev(round1_vals)
            if not math.isnan(round1_sd):
                public_floor.append(-0.5 * round1_sd)

    return public_floor, private_floor


def collect_convergence_rows(
    *,
    roots: list[Path],
    prefixes: list[str],
    models: list[str],
    human_filename: str,
    llm_filename: str,
    likert_col: str,
    no_progress: bool,
    respect_validity_labels: bool,
    initial_post_sidecar_roots: list[Path],
    tweet_sidecar_roots: list[Path],
) -> list[dict[str, Any]]:
    human_public: list[float] = []
    human_private: list[float] = []
    model_public: dict[str, list[float]] = {model: [] for model in models}
    model_private: dict[str, list[float]] = {model: [] for model in models}
    reference_public: dict[str, list[float]] = {model_name: [] for model_name, _, _ in REFERENCE_LINE_SPECS}
    reference_private: dict[str, list[float]] = {model_name: [] for model_name, _, _ in REFERENCE_LINE_SPECS}

    prefix_iter: Any = prefixes
    if not no_progress and tqdm is not None:
        prefix_iter = tqdm(prefixes, total=len(prefixes), desc="Convergence summaries", unit="prefix")
    elif not no_progress and tqdm is None:
        print("tqdm not installed; running without progress bar.")

    def resolve_human_path(data_prefix: str) -> Path | None:
        # Evaluation exports often duplicate the human CSV inside each model subdirectory
        # rather than storing a single shared copy at the prefix root.
        human_path = resolve_file(
            roots,
            data_prefix,
            CANONICAL_V3_HUMAN_SOURCE_MODEL,
            human_filename,
            allow_without_model=False,
        )
        if human_path is not None:
            return human_path

        human_path = resolve_file(roots, data_prefix, "", human_filename, allow_without_model=True)
        if human_path is not None:
            return human_path

        for model in models:
            human_path = resolve_file(roots, data_prefix, model, human_filename, allow_without_model=False)
            if human_path is not None:
                return human_path

        return None

    for data_prefix in prefix_iter:
        human_path = resolve_human_path(data_prefix)
        if human_path is None:
            continue
        human_rows = read_csv_rows(
            human_path,
            respect_validity_labels=respect_validity_labels,
            likert_col=likert_col,
            data_roots=roots,
            initial_post_sidecar_roots=initial_post_sidecar_roots,
            tweet_sidecar_roots=tweet_sidecar_roots,
        )
        human_public_delta, human_private_delta = compute_profile_delta_lists(
            human_rows,
            data_prefix=data_prefix,
            likert_col=likert_col,
        )
        human_public.extend(human_public_delta)
        human_private.extend(human_private_delta)

        consensus_floor_public, consensus_floor_private = compute_consensus_floor_lists(
            human_rows,
            data_prefix=data_prefix,
            likert_col=likert_col,
        )
        reference_public[CONSENSUS_FLOOR_MODEL].extend(consensus_floor_public)
        reference_private[CONSENSUS_FLOOR_MODEL].extend(consensus_floor_private)

        half_sd_public, half_sd_private = compute_half_sd_reduction_lists(
            human_rows,
            data_prefix=data_prefix,
            likert_col=likert_col,
        )
        reference_public[HALF_SD_REDUCTION_MODEL].extend(half_sd_public)
        reference_private[HALF_SD_REDUCTION_MODEL].extend(half_sd_private)

        for model in models:
            llm_path = resolve_file(roots, data_prefix, model, llm_filename, allow_without_model=False)
            if llm_path is None:
                continue
            llm_rows = read_csv_rows(
                llm_path,
                respect_validity_labels=respect_validity_labels,
                likert_col=likert_col,
                data_roots=roots,
                initial_post_sidecar_roots=initial_post_sidecar_roots,
                tweet_sidecar_roots=tweet_sidecar_roots,
            )
            llm_public_delta, llm_private_delta = compute_profile_delta_lists(
                llm_rows,
                data_prefix=data_prefix,
                likert_col=likert_col,
            )
            model_public[model].extend(llm_public_delta)
            model_private[model].extend(llm_private_delta)

    rows: list[dict[str, Any]] = [
        {
            "model": "human",
            "public_mean_delta_sd": safe_mean(human_public),
            "public_group_count": len(human_public),
            "private_mean_delta_sd": safe_mean(human_private),
            "private_group_count": len(human_private),
        }
    ]
    for model in models:
        rows.append(
            {
                "model": model,
                "public_mean_delta_sd": safe_mean(model_public[model]),
                "public_group_count": len(model_public[model]),
                "private_mean_delta_sd": safe_mean(model_private[model]),
                "private_group_count": len(model_private[model]),
            }
        )
    for model_name, _, _ in REFERENCE_LINE_SPECS:
        rows.append(
            {
                "model": model_name,
                "public_mean_delta_sd": safe_mean(reference_public[model_name]),
                "public_group_count": len(reference_public[model_name]),
                "private_mean_delta_sd": safe_mean(reference_private[model_name]),
                "private_group_count": len(reference_private[model_name]),
            }
        )
    return rows


def collect_convergence_rows_with_bootstrap(
    *,
    roots: list[Path],
    prefixes: list[str],
    models: list[str],
    human_filename: str,
    llm_filename: str,
    likert_col: str,
    no_progress: bool,
    bootstrap_reps: int = 1000,
    seed: int = 1,
    respect_validity_labels: bool = False,
    initial_post_sidecar_roots: list[Path] | None = None,
    tweet_sidecar_roots: list[Path] | None = None,
) -> list[dict[str, Any]]:
    model_names = ["human", *models, *[model_name for model_name, _, _ in REFERENCE_LINE_SPECS]]
    prefix_public: dict[str, dict[str, list[float]]] = {prefix: {model_name: [] for model_name in model_names} for prefix in prefixes}
    prefix_private: dict[str, dict[str, list[float]]] = {prefix: {model_name: [] for model_name in model_names} for prefix in prefixes}

    prefix_iter: Any = prefixes
    if not no_progress and tqdm is not None:
        prefix_iter = tqdm(prefixes, total=len(prefixes), desc="Convergence summaries", unit="prefix")
    elif not no_progress and tqdm is None:
        print("tqdm not installed; running without progress bar.")

    def resolve_human_path(data_prefix: str) -> Path | None:
        human_path = resolve_file(
            roots,
            data_prefix,
            CANONICAL_V3_HUMAN_SOURCE_MODEL,
            human_filename,
            allow_without_model=False,
        )
        if human_path is not None:
            return human_path
        human_path = resolve_file(roots, data_prefix, "", human_filename, allow_without_model=True)
        if human_path is not None:
            return human_path
        for model in models:
            human_path = resolve_file(roots, data_prefix, model, human_filename, allow_without_model=False)
            if human_path is not None:
                return human_path
        return None

    for data_prefix in prefix_iter:
        human_path = resolve_human_path(data_prefix)
        if human_path is None:
            continue
        human_rows = read_csv_rows(
            human_path,
            respect_validity_labels=respect_validity_labels,
            likert_col=likert_col,
            data_roots=roots,
            initial_post_sidecar_roots=initial_post_sidecar_roots or [],
            tweet_sidecar_roots=tweet_sidecar_roots or [],
        )
        human_public_delta, human_private_delta = compute_profile_delta_lists(
            human_rows,
            data_prefix=data_prefix,
            likert_col=likert_col,
        )
        prefix_public[data_prefix]["human"].extend(human_public_delta)
        prefix_private[data_prefix]["human"].extend(human_private_delta)

        consensus_floor_public, consensus_floor_private = compute_consensus_floor_lists(
            human_rows,
            data_prefix=data_prefix,
            likert_col=likert_col,
        )
        prefix_public[data_prefix][CONSENSUS_FLOOR_MODEL].extend(consensus_floor_public)
        prefix_private[data_prefix][CONSENSUS_FLOOR_MODEL].extend(consensus_floor_private)

        half_sd_public, half_sd_private = compute_half_sd_reduction_lists(
            human_rows,
            data_prefix=data_prefix,
            likert_col=likert_col,
        )
        prefix_public[data_prefix][HALF_SD_REDUCTION_MODEL].extend(half_sd_public)
        prefix_private[data_prefix][HALF_SD_REDUCTION_MODEL].extend(half_sd_private)

        for model in models:
            llm_path = resolve_file(roots, data_prefix, model, llm_filename, allow_without_model=False)
            if llm_path is None:
                continue
            llm_rows = read_csv_rows(
                llm_path,
                respect_validity_labels=respect_validity_labels,
                likert_col=likert_col,
                data_roots=roots,
                initial_post_sidecar_roots=initial_post_sidecar_roots or [],
                tweet_sidecar_roots=tweet_sidecar_roots or [],
            )
            llm_public_delta, llm_private_delta = compute_profile_delta_lists(
                llm_rows,
                data_prefix=data_prefix,
                likert_col=likert_col,
            )
            prefix_public[data_prefix][model].extend(llm_public_delta)
            prefix_private[data_prefix][model].extend(llm_private_delta)

    rows: list[dict[str, Any]] = []
    bootstrap_public: dict[str, list[float]] = {model_name: [] for model_name in model_names}
    bootstrap_private: dict[str, list[float]] = {model_name: [] for model_name in model_names}
    available_prefixes = [prefix for prefix in prefixes if any(prefix_public[prefix][model_name] or prefix_private[prefix][model_name] for model_name in model_names)]

    if available_prefixes and bootstrap_reps > 0:
        rng = random.Random(seed)
        for _ in range(bootstrap_reps):
            sampled_prefixes = [available_prefixes[rng.randrange(len(available_prefixes))] for _ in available_prefixes]
            for model_name in model_names:
                public_values: list[float] = []
                private_values: list[float] = []
                for sampled_prefix in sampled_prefixes:
                    public_values.extend(prefix_public[sampled_prefix][model_name])
                    private_values.extend(prefix_private[sampled_prefix][model_name])
                public_mean = safe_mean(public_values)
                private_mean = safe_mean(private_values)
                if not math.isnan(public_mean):
                    bootstrap_public[model_name].append(public_mean)
                if not math.isnan(private_mean):
                    bootstrap_private[model_name].append(private_mean)

    def bootstrap_standard_error(samples: list[float]) -> float:
        if len(samples) <= 1:
            return math.nan
        mean_value = sum(samples) / len(samples)
        variance = sum((sample - mean_value) ** 2 for sample in samples) / (len(samples) - 1)
        return math.sqrt(variance)

    for model_name in model_names:
        public_values = [value for prefix in prefixes for value in prefix_public[prefix][model_name]]
        private_values = [value for prefix in prefixes for value in prefix_private[prefix][model_name]]
        rows.append(
            {
                "model": model_name,
                "public_mean_delta_sd": safe_mean(public_values),
                "public_group_count": len(public_values),
                "private_mean_delta_sd": safe_mean(private_values),
                "private_group_count": len(private_values),
                "public_mean_delta_sd_bootstrap_se": bootstrap_standard_error(bootstrap_public[model_name]),
                "private_mean_delta_sd_bootstrap_se": bootstrap_standard_error(bootstrap_private[model_name]),
            }
        )

    return rows


def plot_convergence_rows(
    rows: list[dict[str, Any]],
    *,
    out_png: Path,
    title: str,
    model_order: list[str] | None = None,
    write_png: bool = True,
) -> None:
    if plt is None:
        print("matplotlib not available; skipping group opinion convergence plots.")
        return

    reference_models = {model_name for model_name, _, _ in REFERENCE_LINE_SPECS}
    bar_rows = [row for row in rows if str(row.get("model", "")) not in reference_models]
    bar_rows = sort_rows_by_model_order(bar_rows, model_order)
    reference_rows = {str(row["model"]): row for row in rows if str(row.get("model", "")) in reference_models}

    model_names = [str(row["model"]) for row in bar_rows]
    footnotes = build_wrapped_model_footnotes(model_names)
    positions = list(range(len(model_names)))
    formatted_labels = [format_convergence_model_label(name) for name in model_names]
    colors = ["#4A4A4A" if name == "human" else get_model_color(name, "#999999") for name in model_names]
    metrics = [
        ("Public", "public_mean_delta_sd", "public_group_count"),
        ("Private", "private_mean_delta_sd", "private_group_count"),
    ]

    footnote_line_count = max(1, len(footnotes))
    model_count = max(1, len(model_names))
    fig_height = 7.2 + min(2.4, 0.20 * max(0, model_count - 4)) + min(0.9, 0.14 * max(0, footnote_line_count - 1))
    bottom_margin = 0.24 + min(0.04, 0.008 * max(0, footnote_line_count - 1))

    fig, axes = plt.subplots(1, 2, figsize=(18, fig_height), sharex=False)

    for ax, (panel_title, field_name, count_field) in zip(axes, metrics):
        values = [float(row.get(field_name, math.nan)) for row in bar_rows]
        count_values = [row.get(count_field) for row in bar_rows]
        finite_values = [value for value in values if not math.isnan(value)]
        reference_values = [
            float(reference_rows[model_name].get(field_name, math.nan))
            for model_name, _, _ in REFERENCE_LINE_SPECS
            if model_name in reference_rows
        ]
        finite_reference_values = [value for value in reference_values if not math.isnan(value)]
        all_finite_values = finite_values + finite_reference_values
        lower = min(all_finite_values) if all_finite_values else -0.1
        upper = max(all_finite_values) if all_finite_values else 0.1
        spread = upper - lower
        padding = max(0.08, spread * 0.2 if spread > 0 else 0.12)

        bars = ax.bar(positions, values, color=colors, edgecolor="white", linewidth=0.6, width=0.8)
        bootstrap_ses = [float(row.get(f"{field_name}_bootstrap_se", math.nan)) for row in bar_rows]
        ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.9)
        for model_name, line_color, legend_label in REFERENCE_LINE_SPECS:
            ref_row = reference_rows.get(model_name)
            if ref_row is None:
                continue
            ref_value = float(ref_row.get(field_name, math.nan))
            if math.isnan(ref_value):
                continue
            ax.axhline(ref_value, color=line_color, linewidth=1.8, linestyle=":", alpha=0.95, label=legend_label)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.set_ylabel("Δ group SD")
        ax.set_title(f"{panel_title} Opinion Convergence", loc="left", fontweight="bold")
        y_bottom = lower - padding
        y_top = upper + padding
        count_y = y_bottom + max(0.01, (y_top - y_bottom) * 0.03)
        ax.set_ylim(y_bottom, y_top)
        label_offset = max(0.03, (y_top - y_bottom) * 0.035)

        for bar, value, count_value, bootstrap_se in zip(bars, values, count_values, bootstrap_ses):
            count_number = float(count_value) if count_value not in (None, "") else math.nan
            if math.isnan(count_number) or count_number <= 0:
                count_label = "(n=0)"
            else:
                count_label = f"n={int(round(count_number))}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                count_y,
                count_label,
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=6.5,
                color="#666666",
            )
            if math.isnan(value):
                continue
            error_size = bootstrap_se if not math.isnan(bootstrap_se) else 0.0
            low_value = value - error_size
            high_value = value + error_size
            ax.errorbar(
                [bar.get_x() + bar.get_width() / 2],
                [value],
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
                high_value + label_offset,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                rotation=0,
                fontsize=7,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.4},
                zorder=6,
            )
        ax.set_xticks(positions)
        ax.set_xticklabels(formatted_labels, rotation=34, ha="right")
        ax.tick_params(axis="x", pad=2)

    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        unique_pairs: list[tuple[Any, str]] = []
        seen_labels: set[str] = set()
        for handle, label in zip(handles, labels):
            if label in seen_labels:
                continue
            seen_labels.add(label)
            unique_pairs.append((handle, label))
        if unique_pairs:
            legend_handles, legend_labels = zip(*unique_pairs)
            fig.legend(
                legend_handles,
                legend_labels,
                loc="upper right",
                bbox_to_anchor=(0.985, 0.965),
                bbox_transform=fig.transFigure,
                frameon=True,
                fontsize=10,
            )
    fig.suptitle(f"{title}\nNegative values mean reduced SD (more convergence)", fontsize=14, fontweight="bold")
    add_wrapped_model_footnotes(fig, footnotes, y=0.01)
    fig.subplots_adjust(left=0.055, right=0.985, bottom=bottom_margin, top=0.86, wspace=0.14)
    if write_png:
        fig.savefig(out_png, dpi=220)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


def _write_temp_page_pdf(fig: Any) -> Path:
    handle = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    temp_path = Path(handle.name)
    handle.close()
    with PdfPages(temp_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return temp_path


def build_report_pdf(
    output_dir: Path,
    figures: list[tuple[str, Path]],
    *,
    report_name: str,
    cover_title: str,
    cover_body: str,
) -> Path | None:
    if plt is None or PdfPages is None:
        print("matplotlib PDF backend not available; skipping report PDF.")
        return None

    out_path = output_dir / report_name
    if PdfWriter is None:
        raise RuntimeError("pypdf with PdfWriter support is required to build vector report PDFs.")

    temp_paths: list[Path] = []
    try:
        merger = PdfWriter()
        cover_fig = plt.figure(figsize=(11, 8.5))
        cover_ax = cover_fig.add_axes([0, 0, 1, 1])
        cover_ax.set_axis_off()
        cover_ax.set_facecolor("white")

        wrapped_title_lines = textwrap.wrap(
            cover_title.strip(),
            width=42,
            break_long_words=False,
            break_on_hyphens=False,
        ) or [cover_title.strip()]
        wrapped_cover_lines: list[str] = []
        for raw_line in cover_body.strip().split("\n"):
            stripped = raw_line.strip()
            if not stripped:
                wrapped_cover_lines.append("")
                continue
            wrapped_cover_lines.extend(
                textwrap.wrap(
                    stripped,
                    width=88,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
                or [stripped]
            )
        title_y = 0.92
        title_step = 0.050
        title_fontsize = 22 if len(wrapped_title_lines) == 1 else 19
        for idx, title_line in enumerate(wrapped_title_lines):
            cover_ax.text(
                0.5,
                title_y - idx * title_step,
                title_line,
                ha="center",
                va="top",
                fontsize=title_fontsize,
                fontweight="bold",
            )

        body_top = title_y - title_step * len(wrapped_title_lines) - 0.03
        body_bottom = 0.18
        line_count_units = sum(0.32 if not line.strip() else 1.0 for line in wrapped_cover_lines)
        available_height = max(0.18, body_top - body_bottom)
        body_spacing = min(0.040, available_height / line_count_units) if line_count_units else 0.036
        blank_spacing = body_spacing * 0.32
        body_fontsize = 12 if body_spacing >= 0.034 else 11
        current_y = body_top
        for line in wrapped_cover_lines:
            if not line.strip():
                current_y -= blank_spacing
                continue
            cover_ax.text(
                0.5,
                current_y,
                line,
                ha="center",
                va="top",
                fontsize=body_fontsize,
            )
            current_y -= body_spacing

        output_lines = textwrap.wrap(
            f"Output directory: {output_dir}",
            width=90,
            break_long_words=False,
            break_on_hyphens=False,
        )
        output_y = 0.12
        output_step = 0.028
        for idx, output_line in enumerate(output_lines):
            cover_ax.text(
                0.06,
                output_y - idx * output_step,
                output_line,
                ha="left",
                va="top",
                fontsize=10,
                family="monospace",
            )
        cover_path = _write_temp_page_pdf(cover_fig)
        temp_paths.append(cover_path)
        merger.append(str(cover_path))

        for page_title, image_path in figures:
            figure_pdf_path = image_path.with_suffix(".pdf")
            if not figure_pdf_path.exists():
                raise FileNotFoundError(
                    f"Missing figure PDF for report page '{page_title}': {figure_pdf_path}"
                )
            merger.append(str(figure_pdf_path))

        with out_path.open("wb") as out_file:
            merger.write(out_file)
    finally:
        for temp_path in temp_paths:
            if temp_path.exists():
                temp_path.unlink()

    return out_path


def generate_output_set(
    *,
    output_dir: Path,
    subsets: list[tuple[str, list[str], list[str], str]],
    report_version: str,
    roots: list[Path],
    human_filename: str,
    llm_filename: str,
    likert_col: str,
    no_progress: bool,
    report_name: str,
    cover_title: str,
    cover_body: str,
    output_suffix: str,
    respect_validity_labels: bool,
    initial_post_sidecar_roots: list[Path],
    tweet_sidecar_roots: list[Path],
    report_only: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    img_dir = output_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = output_dir / "csv"
    if not report_only:
        csv_dir.mkdir(parents=True, exist_ok=True)

    report_figures: list[tuple[str, Path]] = []
    for suffix, subset_prefixes, subset_models, title in subsets:
        rows = collect_convergence_rows_with_bootstrap(
            roots=roots,
            prefixes=subset_prefixes,
            models=subset_models,
            human_filename=human_filename,
            llm_filename=llm_filename,
            likert_col=likert_col,
            no_progress=no_progress,
            respect_validity_labels=respect_validity_labels,
            initial_post_sidecar_roots=initial_post_sidecar_roots,
            tweet_sidecar_roots=tweet_sidecar_roots,
        )
        csv_path = csv_dir / f"group_opinion_convergence_{suffix}{output_suffix}.csv"
        png_path = img_dir / f"group_opinion_convergence_{suffix}{output_suffix}.png"
        reference_order = load_reference_model_order(
            output_dir=output_dir,
            suffix=suffix,
            subset_models=subset_models,
            output_suffix=output_suffix,
            report_version=report_version,
        )
        if not report_only:
            save_csv(csv_path, rows, collect_fieldnames(rows))
        plot_convergence_rows(rows, out_png=png_path, title=title, model_order=reference_order, write_png=not report_only)
        report_figures.append((title, png_path))
        if not report_only:
            print(f"Wrote: {csv_path}")
            print(f"Wrote: {png_path}")
        print(f"Wrote: {png_path.with_suffix('.pdf')}")

    report_path = build_report_pdf(
        output_dir,
        report_figures,
        report_name=report_name,
        cover_title=cover_title,
        cover_body=cover_body,
    )
    if report_path is not None:
        print(f"Wrote: {report_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate raw group opinion convergence figures for human and LLMs.")
    parser.add_argument(
        "--data-root",
        default="../../result/eval_v3/human_llm",
        help="Root of human_llm results (default: ../../result/eval_v3/human_llm).",
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
        help="Text file listing the models to plot.",
    )
    parser.add_argument(
        "--report-version",
        default="v3",
        help="Optional artifact family selector. v4 excludes Tulu from the base-model roster.",
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
        "--likert-col",
        default="likert_pred",
        help="Column for stance labels/scores.",
    )
    parser.add_argument(
        "--output-dir",
        default="../../../sean_reproduce/v2/group_opinion_convergence",
        help="Output directory for report assets (default: ../../../sean_reproduce/v2/group_opinion_convergence). CSVs are written under output_dir/csv and figures under output_dir/img.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar.",
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
        default="_v2",
        help="Filename suffix for generated CSVs/plots, e.g. _v3.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Skip CSV and PNG writes; keep only per-page PDFs plus the assembled report PDF.",
    )
    parser.add_argument(
        "--only-base-model-report",
        action="store_true",
        help="Generate only base_model/report_group_opinion_convergence*.pdf and skip per-topic and post-training outputs.",
    )
    parser.add_argument(
        "--skip-base-model-reports",
        action="store_true",
        help="Skip base_model outputs entirely and generate only the requested post-training reports.",
    )
    parser.add_argument(
        "--post-train-report-spec",
        action="append",
        choices=tuple(model_set_key for model_set_key, _, _ in POST_TRAIN_REPORT_SPECS),
        default=[],
        help="Optional post-training report set(s) to generate. Defaults to all registered post-training report specs.",
    )
    return parser


def load_reference_model_order(*, output_dir: Path, suffix: str, subset_models: list[str], output_suffix: str, report_version: str) -> list[str]:
    if output_dir.name == "base_model":
        return ordered_base_model_llms_for_report(report_version, subset_models)

    version_root = output_dir.parent.parent
    reference_csv = version_root / "summary_post_training" / "csv" / f"raw_metrics_{suffix}{output_suffix}.csv"

    if not reference_csv.exists():
        return subset_models

    reference_rows = read_csv_rows(reference_csv)
    return get_v2_display_model_order(reference_rows, allowed_models=subset_models)


def generate_per_topic_report(
    *,
    output_dir: Path,
    prefixes: list[str],
    models: list[str],
    report_version: str,
    roots: list[Path],
    human_filename: str,
    llm_filename: str,
    likert_col: str,
    no_progress: bool,
    report_name: str,
    cover_title: str,
    cover_body: str,
    output_suffix: str,
    respect_validity_labels: bool,
    initial_post_sidecar_roots: list[Path],
    tweet_sidecar_roots: list[Path],
    reference_suffix: str,
    topic_scope_label: str,
    report_only: bool,
) -> None:
    img_dir = output_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = output_dir / "csv"
    if not report_only:
        csv_dir.mkdir(parents=True, exist_ok=True)

    figures: list[tuple[str, Path]] = []
    reference_order = load_reference_model_order(
        output_dir=output_dir,
        suffix=reference_suffix,
        subset_models=models,
        output_suffix=output_suffix,
        report_version=report_version,
    )

    overall_rows = collect_convergence_rows_with_bootstrap(
        roots=roots,
        prefixes=prefixes,
        models=models,
        human_filename=human_filename,
        llm_filename=llm_filename,
        likert_col=likert_col,
        no_progress=no_progress,
        respect_validity_labels=respect_validity_labels,
        initial_post_sidecar_roots=initial_post_sidecar_roots,
        tweet_sidecar_roots=tweet_sidecar_roots,
    )
    overall_csv_path = csv_dir / f"group_opinion_convergence_{reference_suffix}_overall_per_topic{output_suffix}.csv"
    overall_png_path = img_dir / f"group_opinion_convergence_{reference_suffix}_overall_per_topic{output_suffix}.png"
    if not report_only:
        save_csv(overall_csv_path, overall_rows, collect_fieldnames(overall_rows))
    plot_convergence_rows(
        overall_rows,
        out_png=overall_png_path,
        title=f"Group Opinion Convergence — {topic_scope_label} Topics (Overall)",
        model_order=reference_order,
        write_png=not report_only,
    )
    figures.append((f"Group Opinion Convergence — {topic_scope_label} Topics (Overall)", overall_png_path))
    if not report_only:
        print(f"Wrote: {overall_csv_path}")
        print(f"Wrote: {overall_png_path}")
    print(f"Wrote: {overall_png_path.with_suffix('.pdf')}")

    topic_to_prefixes: dict[str, list[str]] = {}
    for prefix in prefixes:
        topic_to_prefixes.setdefault(extract_topic_name(prefix), []).append(prefix)

    for topic_name in sorted(topic_to_prefixes):
        topic_prefixes = topic_to_prefixes[topic_name]
        rows = collect_convergence_rows_with_bootstrap(
            roots=roots,
            prefixes=topic_prefixes,
            models=models,
            human_filename=human_filename,
            llm_filename=llm_filename,
            likert_col=likert_col,
            no_progress=True,
            respect_validity_labels=respect_validity_labels,
            initial_post_sidecar_roots=initial_post_sidecar_roots,
            tweet_sidecar_roots=tweet_sidecar_roots,
        )
        topic_slug = slugify(topic_name)
        csv_path = csv_dir / f"group_opinion_convergence_{reference_suffix}_{topic_slug}{output_suffix}.csv"
        png_path = img_dir / f"group_opinion_convergence_{reference_suffix}_{topic_slug}{output_suffix}.png"
        title = f"Group Opinion Convergence — {topic_name.replace('_', ' ')}"
        if not report_only:
            save_csv(csv_path, rows, collect_fieldnames(rows))
        plot_convergence_rows(rows, out_png=png_path, title=title, model_order=reference_order, write_png=not report_only)
        figures.append((title, png_path))
        if not report_only:
            print(f"Wrote: {csv_path}")
            print(f"Wrote: {png_path}")
        print(f"Wrote: {png_path.with_suffix('.pdf')}")

    report_path = build_report_pdf(
        output_dir,
        figures,
        report_name=report_name,
        cover_title=cover_title,
        cover_body=cover_body,
    )
    if report_path is not None:
        print(f"Wrote: {report_path}")


def main() -> int:
    args = build_arg_parser().parse_args()

    roots = [Path(args.data_root), *[Path(path) for path in args.fallback_data_root]]
    initial_post_sidecar_roots = [Path(path) for path in args.initial_post_validity_sidecar_root]
    tweet_sidecar_roots = [Path(path) for path in args.tweet_validity_sidecar_root]
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    prefixes = read_lines(Path(args.data_prefix_file))
    models = ordered_base_model_llms_for_report(args.report_version, read_lines(Path(args.models_file)))

    base_model_subsets = [
        ("all_topics", prefixes, models, "Group Opinion Convergence — All Topics"),
        (
            "breadth_topics",
            [prefix for prefix in prefixes if not is_depth_topic(prefix)],
            models,
            "Group Opinion Convergence — Breadth Topics",
        ),
        (
            "depth_topics",
            [prefix for prefix in prefixes if is_depth_topic(prefix)],
            models,
            "Group Opinion Convergence — Depth Topics",
        ),
    ]

    if not args.skip_base_model_reports:
        generate_output_set(
            output_dir=output_root / "base_model",
            subsets=base_model_subsets,
            report_version=args.report_version,
            roots=roots,
            human_filename=args.human_filename,
            llm_filename=args.llm_filename,
            likert_col=args.likert_col,
            no_progress=args.no_progress,
            report_name=f"report_group_opinion_convergence{args.output_suffix}.pdf",
            cover_title=f"Group Opinion Convergence Report ({args.output_suffix.lstrip('_') or 'v2'})",
            cover_body=_append_v3_mask_cover_body((
                "This report collects the raw convergence figures for all topics, breadth topics, and depth topics.\n"
                "Bars show mean Δ group SD by model; error bars show bootstrap standard errors from resampling data_prefix values with replacement 1000 times; the dotted lines mark two simple references.\n"
                "For public opinion, each participant's synthetic final stance moves one Likert step toward their first partner's round-1 tweet stance.\n"
                "For private opinion, it moves one Likert step toward that same partner's initial opinion stance.\n"
                "The second line is a direct 50% reduction of initial group SD."
            ), output_suffix=args.output_suffix, uses_split_masks=False),
            output_suffix=args.output_suffix,
            respect_validity_labels=args.respect_validity_labels,
            initial_post_sidecar_roots=initial_post_sidecar_roots,
            tweet_sidecar_roots=tweet_sidecar_roots,
            report_only=args.report_only,
        )

    if args.only_base_model_report:
        return 0

    base_model_output_dir = output_root / "base_model"
    depth_prefixes = [prefix for prefix in prefixes if is_depth_topic(prefix)]
    breadth_prefixes = [prefix for prefix in prefixes if not is_depth_topic(prefix)]

    if not args.skip_base_model_reports:
        generate_per_topic_report(
            output_dir=base_model_output_dir,
            prefixes=depth_prefixes,
            models=models,
            report_version=args.report_version,
            roots=roots,
            human_filename=args.human_filename,
            llm_filename=args.llm_filename,
            likert_col=args.likert_col,
            no_progress=args.no_progress,
            report_name=f"report_depth_per_topic_group_opinion_convergence{args.output_suffix}.pdf",
            cover_title=f"Depth Per-Topic Group Opinion Convergence Report ({args.output_suffix.lstrip('_') or 'v2'})",
            cover_body=_append_v3_mask_cover_body((
                "This report breaks the base-model group opinion convergence bars out topic by topic for depth topics.\n"
                "Page 2 shows the overall depth-topic bar plot, followed by one page per topic.\n"
                "Bars show mean Δ group SD by model; error bars show bootstrap standard errors from resampling data_prefix values with replacement 1000 times; the dotted lines mark two simple references."
            ), output_suffix=args.output_suffix, uses_split_masks=False),
            output_suffix=args.output_suffix,
            respect_validity_labels=args.respect_validity_labels,
            initial_post_sidecar_roots=initial_post_sidecar_roots,
            tweet_sidecar_roots=tweet_sidecar_roots,
            reference_suffix="depth_topics",
            topic_scope_label="Depth",
            report_only=args.report_only,
        )

        if not skip_v4_family_breadth_topic_per_topic_report(args.report_version, "breadth_topics"):
            generate_per_topic_report(
                output_dir=base_model_output_dir,
                prefixes=breadth_prefixes,
                models=models,
                report_version=args.report_version,
                roots=roots,
                human_filename=args.human_filename,
                llm_filename=args.llm_filename,
                likert_col=args.likert_col,
                no_progress=args.no_progress,
                report_name=f"report_breadth_per_topic_group_opinion_convergence{args.output_suffix}.pdf",
                cover_title=f"Breadth Per-Topic Group Opinion Convergence Report ({args.output_suffix.lstrip('_') or 'v2'})",
                cover_body=_append_v3_mask_cover_body((
                    "This report breaks the base-model group opinion convergence bars out topic by topic for breadth topics.\n"
                    "Page 2 shows the overall breadth-topic bar plot, followed by one page per topic.\n"
                    "Bars show mean Δ group SD by model; error bars show bootstrap standard errors from resampling data_prefix values with replacement 1000 times; the dotted lines mark two simple references."
                ), output_suffix=args.output_suffix, uses_split_masks=False),
                output_suffix=args.output_suffix,
                respect_validity_labels=args.respect_validity_labels,
                initial_post_sidecar_roots=initial_post_sidecar_roots,
                tweet_sidecar_roots=tweet_sidecar_roots,
                reference_suffix="breadth_topics",
                topic_scope_label="Breadth",
                report_only=args.report_only,
            )

    post_train_output_dir = output_root / "post_train"
    selected_post_train_specs = [
        spec for spec in POST_TRAIN_REPORT_SPECS
        if not args.post_train_report_spec or spec[0] in set(args.post_train_report_spec)
    ]

    for model_set_key, report_label, report_name in selected_post_train_specs:
        model_set_cfg = MODEL_SET_CONFIGS[model_set_key]
        post_train_subsets: list[tuple[str, list[str], list[str], str]] = []
        post_train_partition_specs: list[tuple[str, str, list[str], list[str], str]] = []
        report_prefix = "models_vs_all_" if model_set_key in {"post-training-vs-all", "post-training-v4-vs-all"} else model_set_cfg.file_prefix
        for suffix, prefix_file, model_attr, partition_label in POST_TRAIN_PARTITION_SPECS:
            partition_prefix_file = Path(prefix_file)
            if not partition_prefix_file.exists():
                print(f"Prefix file not found: {partition_prefix_file}")
                return 1
            partition_prefixes = read_lines(partition_prefix_file)
            partition_models = resolve_post_train_partition_models(model_set_key=model_set_key, suffix=suffix)
            if partition_models is None:
                partition_models = getattr(model_set_cfg, model_attr)
            post_train_subsets.append(
                (
                    f"{model_set_cfg.file_prefix}{suffix}",
                    partition_prefixes,
                    partition_models,
                    f"{model_set_cfg.title_prefix} - {partition_label}",
                )
            )
            post_train_partition_specs.append(
                (
                    f"{model_set_cfg.file_prefix}{suffix}",
                    f"{report_prefix}{suffix}",
                    partition_prefixes,
                    partition_models,
                    partition_label,
                )
            )

        generate_output_set(
            output_dir=post_train_output_dir,
            subsets=post_train_subsets,
            report_version=args.report_version,
            roots=roots,
            human_filename=args.human_filename,
            llm_filename=args.llm_filename,
            likert_col=args.likert_col,
            no_progress=args.no_progress,
            report_name=re.sub(r"_(?:v\d(?:_fixed|_b)?)\.pdf$", f"{args.output_suffix}.pdf", report_name),
            cover_title=f"Post-Training Group Opinion Convergence Report ({report_label}, {args.output_suffix.lstrip('_') or 'v2'})",
            cover_body=_append_v3_mask_cover_body((
                "This report collects the raw breadth and depth split convergence figures for the post-training setting.\n"
                "Bars show mean Δ group SD by model; error bars show bootstrap standard errors from resampling data_prefix values with replacement 1000 times; the dotted lines mark two simple references.\n"
                "For public opinion, each participant's synthetic final stance moves one Likert step toward their first partner's round-1 tweet stance.\n"
                "For private opinion, it moves one Likert step toward that same partner's initial opinion stance.\n"
                "The second line is a direct 50% reduction of initial group SD."
            ), output_suffix=args.output_suffix, uses_split_masks=True),
            output_suffix=args.output_suffix,
            respect_validity_labels=args.respect_validity_labels,
            initial_post_sidecar_roots=initial_post_sidecar_roots,
            tweet_sidecar_roots=tweet_sidecar_roots,
            report_only=args.report_only,
        )

        if model_set_key in {"post-training-vs-all", "post-training-v4-vs-all"}:
            continue

        for reference_suffix, report_suffix, partition_prefixes, partition_models, partition_label in post_train_partition_specs:
            if skip_v4_family_breadth_topic_per_topic_report(args.report_version, reference_suffix):
                continue
            generate_per_topic_report(
                output_dir=post_train_output_dir,
                prefixes=partition_prefixes,
                models=partition_models,
                report_version=args.report_version,
                roots=roots,
                human_filename=args.human_filename,
                llm_filename=args.llm_filename,
                likert_col=args.likert_col,
                no_progress=args.no_progress,
                report_name=f"report_{report_suffix}_per_topic_group_opinion_convergence{args.output_suffix}.pdf",
                cover_title=(
                    f"Per-Topic Group Opinion Convergence Report "
                    f"({model_set_cfg.title_prefix} - {partition_label}, {args.output_suffix.lstrip('_') or 'v2'})"
                ),
                cover_body=_append_v3_mask_cover_body((
                    "This report breaks the post-training group opinion convergence bars out topic by topic for the selected partition.\n"
                    "Page 2 shows the overall bar plot for this partition, followed by one page per topic.\n"
                    "Bars show mean Δ group SD by model; error bars show bootstrap standard errors from resampling data_prefix values with replacement 1000 times; the dotted lines mark two simple references."
                ), output_suffix=args.output_suffix, uses_split_masks=True),
                output_suffix=args.output_suffix,
                respect_validity_labels=args.respect_validity_labels,
                initial_post_sidecar_roots=initial_post_sidecar_roots,
                tweet_sidecar_roots=tweet_sidecar_roots,
                reference_suffix=reference_suffix,
                topic_scope_label=f"{model_set_cfg.title_prefix} - {partition_label}",
                report_only=args.report_only,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())