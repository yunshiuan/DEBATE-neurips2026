#!/usr/bin/env python3
"""Generate v3 scatter plot reports for base-model and post-train comparisons.

Outputs are written under sean_reproduce/v3/scatter_plot:
- base_model/
  - depth base-model reports with overall, correlation-summary, and per-topic pages
  - breadth base-model reports with overall and correlation-summary pages
- post_train/
    - breadth topic/group train/test post-train Llama vs SFT/DPO reports with overall and correlation-summary pages
    - depth topic/group train/test post-train Llama vs SFT/DPO reports with overall and correlation-summary pages

All reports use the canonical v3 rejudged validity sidecars when available.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
SEAN_REPRODUCE_SCRIPTS_DIR = REPO_ROOT / "neurips_2026" / "sean_reproduce" / "scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SEAN_REPRODUCE_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SEAN_REPRODUCE_SCRIPTS_DIR))

from generate_plots import (
    HUMAN_DEFAULT_FILENAME,
    LLAMA_8B_INSTRUCT_MODEL,
    LLM_DEFAULT_FILENAME,
    get_model_color,
    get_model_display_label,
    get_v2_display_model_order,
    group_id,
    is_degroot_model,
    is_llama_sft_model,
    parse_likert,
    parse_round,
    participant_id,
    read_csv_rows,
    read_lines,
    resolve_file,
    spearman_rho,
    tqdm,
)
from post_training_breadth_profiles import (
    DEPTH_GROUP_MODELS_V4,
    DEPTH_TOPIC_MODELS_V4,
    GROUP_MODELS_V4,
    TOPIC_MODELS_V4,
)
from neurips_2026.sean_reproduce.result_version_config import (
    has_canonical_result_namespace,
    human_csv_filename_for_result,
    llm_csv_filename_for_result,
    report_family_for_version,
    report_namespace_for_version,
)
from neurips_2026.sean_reproduce.base_model_v3_style import ordered_base_model_llms_for_report
from neurips_2026.sean_reproduce.v3.masks import (
    configured_private_mask_root,
    configured_public_mask_root,
    enable_v3_complete_group_masks,
    repo_relative_display_path,
)
from generate_v3_degroot_abm_post_train_report import (
    POST_TRAIN_PARTITION_SPECS,
    fit_best_alphas,
    partition_pair_key,
    resolve_partition_prefix_file,
)
from generate_v3_degroot_abm_proxy_tree_common import (
    ProxyEvalModelSpec,
    load_valid_human_rows_by_prefix,
    materialize_proxy_eval_tree,
)
from generate_v3_degroot_abm_report import FULL_FIT_MODEL_NAME

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
except Exception:  # pragma: no cover
    plt = None
    PdfPages = None


DEFAULT_INITIAL_POST_SIDECAR_ROOT = "../../../rejudged_labels/eval_v3/initial_post"
DEFAULT_TWEET_SIDECAR_ROOT = "../../../rejudged_labels/eval_v3/tweet"
DEFAULT_DEPTH_MASK = "../../inputs/masks_train_test/golden_group_v0/depth/depth_universe_mask_golden_v0.txt"
DEFAULT_BREADTH_MASK = "../../inputs/masks_train_test/golden_group_v0/breadth/breadth_universe_mask_golden_v0.txt"
DEFAULT_BREADTH_TOPIC_TRAIN_MASK = "../../inputs/masks_train_test/golden_group_v0/breadth/breadth_train_topic_mask_golden_v0.txt"
DEFAULT_BREADTH_TOPIC_TEST_MASK = "../../inputs/masks_train_test/golden_group_v0/breadth/breadth_test_topic_mask_golden_v0.txt"
DEFAULT_BREADTH_GROUP_TRAIN_MASK = "../../inputs/masks_train_test/golden_group_v0/breadth/breadth_train_group_mask_golden_v0.txt"
DEFAULT_BREADTH_GROUP_TEST_MASK = "../../inputs/masks_train_test/golden_group_v0/breadth/breadth_test_group_mask_golden_v0.txt"
DEFAULT_DEPTH_TOPIC_TRAIN_MASK = "../../inputs/masks_train_test/golden_group_v0/depth/depth_train_topic_mask_golden_v0.txt"
DEFAULT_DEPTH_TOPIC_TEST_MASK = "../../inputs/masks_train_test/golden_group_v0/depth/depth_test_topic_mask_golden_v0.txt"
DEFAULT_DEPTH_GROUP_TRAIN_MASK = "../../inputs/masks_train_test/golden_group_v0/depth/depth_train_group_mask_golden_v0.txt"
DEFAULT_DEPTH_GROUP_TEST_MASK = "../../inputs/masks_train_test/golden_group_v0/depth/depth_test_group_mask_golden_v0.txt"
POINT_ALPHA = 0.12
POINT_SIZE = 16
LINE_WIDTH = 2.2
HUMAN_MODEL = "human"
GPT_4O_MINI_BASE_MODEL = "gpt-4o-mini-2024-07-18"
GPT_DEPTH_TOPIC_SFT_MODEL = "ft:gpt-4o-mini-2024-07-18:camer:sft-20251218-depth-topic-5epochs:CoFxFPcv"
GPT_DEPTH_GROUP_SFT_MODEL = "ft:gpt-4o-mini-2024-07-18:camer:sft-20251218-depth-group-5epochs:CoFzijg0"

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
    GPT_4O_MINI_BASE_MODEL,
    "Llama-3.1-8B-Instruct",
    "ft:Llama-3.1-8B-Instruct-SFT-20251218:depth-topic-5epochs",
    GPT_DEPTH_TOPIC_SFT_MODEL,
    "ft:Llama-3.1-8B-Instruct-DPO-20260105:depth-topic-5epochs",
]

POST_TRAIN_DEPTH_GROUP_MODELS = [
    GPT_4O_MINI_BASE_MODEL,
    "Llama-3.1-8B-Instruct",
    "ft:Llama-3.1-8B-Instruct-SFT-20251218:depth-group-5epochs",
    GPT_DEPTH_GROUP_SFT_MODEL,
    "ft:Llama-3.1-8B-Instruct-DPO-20260105:depth-group-5epochs",
]


def report_version_family(report_version: str) -> str:
    return report_family_for_version(report_version)


def report_version_namespace(report_version: str) -> str:
    return report_namespace_for_version(report_version)


def default_output_dir(report_version: str) -> str:
    return f"../../../sean_reproduce/{report_version_namespace(report_version)}/scatter_plot"


def post_train_model_groups(report_version: str) -> dict[str, list[str]]:
    if report_version_family(report_version) == "v4":
        return {
            "breadth_topic": list(TOPIC_MODELS_V4),
            "breadth_group": list(GROUP_MODELS_V4),
            "depth_topic": list(DEPTH_TOPIC_MODELS_V4),
            "depth_group": list(DEPTH_GROUP_MODELS_V4),
        }
    return {
        "breadth_topic": list(POST_TRAIN_BREADTH_TOPIC_MODELS),
        "breadth_group": list(POST_TRAIN_BREADTH_GROUP_MODELS),
        "depth_topic": list(POST_TRAIN_DEPTH_TOPIC_MODELS),
        "depth_group": list(POST_TRAIN_DEPTH_GROUP_MODELS),
    }


def skip_v4_breadth_topic_post_train(family_key: str, report_version: str) -> bool:
    return False


@dataclass
class ConversationProfile:
    initial_private: dict[str, dict[str, float]] = field(default_factory=lambda: defaultdict(dict))
    final_private: dict[str, dict[str, float]] = field(default_factory=lambda: defaultdict(dict))
    tweet_round: dict[str, dict[str, dict[int, float]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(dict))
    )
    first_partner: dict[str, dict[str, str]] = field(default_factory=lambda: defaultdict(dict))


@dataclass(frozen=True)
class PlotSpec:
    key: str
    report_suffix: str
    page_title: str
    x_label: str
    y_label: str
    x_default: tuple[float, float]
    y_default: tuple[float, float]
    x_is_difference: bool
    point_builder: Callable[[ConversationProfile], list[tuple[float, float]]]


@dataclass(frozen=True)
class ReportSuite:
    report_prefix: str
    output_subdir: str
    models: list[str]
    prefix_file: Path
    overall_label: str
    topic_label: str
    include_topic_pages: bool
    progress_label: str
    scatter_panel_specs: tuple[tuple[str, tuple[str, ...]], ...] | None = None


@dataclass(frozen=True)
class MergedPostTrainSuite:
    report_prefix: str
    output_subdir: str
    models: list[str]
    train_prefix_file: Path
    test_prefix_file: Path
    scope_label: str
    progress_label_train: str
    progress_label_test: str
    scatter_panel_specs: tuple[tuple[str, tuple[str, ...]], ...] | None = None


@dataclass(frozen=True)
class MergedPlotPair:
    report_suffix: str
    public_spec_key: str
    private_spec_key: str
    public_label: str
    private_label: str


def resolve_existing_or_script_relative(path_text: str) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    alt = (Path(__file__).resolve().parent / path_text).resolve()
    if alt.exists():
        return alt
    return path


def extract_topic_name(data_prefix: str) -> str:
    parts = data_prefix.split("_")
    if len(parts) >= 4 and parts[0].isdigit() and parts[1].isdigit():
        return "_".join(parts[2:-1])
    return data_prefix


def resolve_human_path(roots: list[Path], data_prefix: str, models: list[str], filename: str) -> Path | None:
    human_path = resolve_file(roots, data_prefix, "", filename, allow_without_model=True)
    if human_path is not None:
        return human_path
    for model in models:
        human_path = resolve_file(roots, data_prefix, model, filename, allow_without_model=False)
        if human_path is not None:
            return human_path
    return None


def build_conversation_profile(rows: list[dict[str, str]], *, default_group: str, likert_col: str) -> ConversationProfile:
    profile = ConversationProfile()
    tweet_values: dict[str, dict[str, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    raw_first_partner: dict[str, dict[str, str]] = defaultdict(dict)
    participant_aliases: dict[str, dict[str, str]] = defaultdict(dict)

    for row in rows:
        pid = participant_id(row)
        if not pid:
            continue
        gid = group_id(row, default_group=default_group)
        for alias_key in ("empirica_id", "worker_id", "sender_id"):
            alias_value = (row.get(alias_key) or "").strip()
            if alias_value:
                participant_aliases[gid][alias_value] = pid
        event_type = (row.get("event_type") or "").strip()
        stance = parse_likert(row.get(likert_col))

        if event_type == "Initial Opinion" and stance is not None:
            profile.initial_private[gid][pid] = stance
            continue

        if event_type == "Post Opinion" and stance is not None:
            profile.final_private[gid][pid] = stance
            continue

        if event_type != "tweet":
            continue

        round_idx = parse_round(row.get("chat_round_order"))
        if round_idx == 1:
            recipient_id = (row.get("recipient_id") or "").strip()
            if recipient_id:
                raw_first_partner[gid].setdefault(pid, recipient_id)
        if round_idx in {1, 3} and stance is not None:
            tweet_values[gid][pid][round_idx].append(stance)

    for gid, participants in tweet_values.items():
        for pid, rounds in participants.items():
            for round_idx, values in rounds.items():
                if values:
                    profile.tweet_round[gid][pid][round_idx] = sum(values) / len(values)

    for gid, pid_to_partner in raw_first_partner.items():
        aliases = participant_aliases.get(gid, {})
        for pid, partner_alias in pid_to_partner.items():
            profile.first_partner[gid][pid] = aliases.get(partner_alias, partner_alias)

    return profile


def build_tweet_start_vs_change_points(profile: ConversationProfile) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for gid, participants in profile.tweet_round.items():
        for pid, rounds in participants.items():
            if 1 not in rounds or 3 not in rounds:
                continue
            points.append((rounds[1], rounds[3] - rounds[1]))
    return points


def build_initial_vs_change_points(profile: ConversationProfile) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    groups = set(profile.initial_private.keys()) | set(profile.final_private.keys())
    for gid in groups:
        init_map = profile.initial_private.get(gid, {})
        final_map = profile.final_private.get(gid, {})
        for pid in sorted(set(init_map.keys()) & set(final_map.keys())):
            points.append((init_map[pid], final_map[pid] - init_map[pid]))
    return points


def build_partner_tweet_gap_vs_change_points(profile: ConversationProfile) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    groups = set(profile.tweet_round.keys()) | set(profile.first_partner.keys())
    for gid in groups:
        rounds_by_pid = profile.tweet_round.get(gid, {})
        first_partner = profile.first_partner.get(gid, {})
        for pid, partner_id in first_partner.items():
            self_rounds = rounds_by_pid.get(pid, {})
            partner_rounds = rounds_by_pid.get(partner_id, {})
            if 1 not in self_rounds or 3 not in self_rounds or 1 not in partner_rounds:
                continue
            x_value = partner_rounds[1] - self_rounds[1]
            y_value = self_rounds[3] - self_rounds[1]
            points.append((x_value, y_value))
    return points


def build_partner_initial_gap_vs_change_points(profile: ConversationProfile) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    groups = set(profile.initial_private.keys()) | set(profile.final_private.keys()) | set(profile.first_partner.keys())
    for gid in groups:
        init_map = profile.initial_private.get(gid, {})
        final_map = profile.final_private.get(gid, {})
        first_partner = profile.first_partner.get(gid, {})
        for pid, partner_id in first_partner.items():
            if pid not in init_map or pid not in final_map or partner_id not in init_map:
                continue
            x_value = init_map[partner_id] - init_map[pid]
            y_value = final_map[pid] - init_map[pid]
            points.append((x_value, y_value))
    return points


PLOT_SPECS = [
    PlotSpec(
        key="tweet_start_vs_change",
        report_suffix="tweet1_vs_tweet_change",
        page_title="Tweet 1 Stance vs Tweet 3 - Tweet 1",
        x_label="Tweet 1 stance",
        y_label="Tweet 3 stance - Tweet 1 stance",
        x_default=(0.5, 6.5),
        y_default=(-5.5, 5.5),
        x_is_difference=False,
        point_builder=build_tweet_start_vs_change_points,
    ),
    PlotSpec(
        key="initial_vs_change",
        report_suffix="initial_vs_final_change",
        page_title="Initial Opinion Stance vs Final - Initial",
        x_label="Initial opinion stance",
        y_label="Final opinion stance - Initial opinion stance",
        x_default=(0.5, 6.5),
        y_default=(-5.5, 5.5),
        x_is_difference=False,
        point_builder=build_initial_vs_change_points,
    ),
    PlotSpec(
        key="partner_tweet_gap_vs_change",
        report_suffix="partner_tweet_gap_vs_tweet_change",
        page_title="Partner Tweet 1 Gap vs Tweet 3 - Tweet 1",
        x_label="First partner's Tweet 1 stance - Self Tweet 1 stance",
        y_label="Tweet 3 stance - Tweet 1 stance",
        x_default=(-5.5, 5.5),
        y_default=(-5.5, 5.5),
        x_is_difference=True,
        point_builder=build_partner_tweet_gap_vs_change_points,
    ),
    PlotSpec(
        key="partner_initial_gap_vs_change",
        report_suffix="partner_initial_gap_vs_final_change",
        page_title="Partner Initial Gap vs Final - Initial",
        x_label="First partner's initial stance - Self initial stance",
        y_label="Final opinion stance - Initial opinion stance",
        x_default=(-5.5, 5.5),
        y_default=(-5.5, 5.5),
        x_is_difference=True,
        point_builder=build_partner_initial_gap_vs_change_points,
    ),
]

PLOT_SPEC_BY_KEY = {spec.key: spec for spec in PLOT_SPECS}

MERGED_PLOT_PAIRS = [
    MergedPlotPair(
        report_suffix="initial_vs_final_change",
        public_spec_key="tweet_start_vs_change",
        private_spec_key="initial_vs_change",
        public_label="Public opinion",
        private_label="Private opinion",
    ),
    MergedPlotPair(
        report_suffix="partner_initial_gap_vs_final_change",
        public_spec_key="partner_tweet_gap_vs_change",
        private_spec_key="partner_initial_gap_vs_change",
        public_label="Public opinion",
        private_label="Private opinion",
    ),
]


def collect_plot_data(
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
    progress_label: str,
    apply_v3_additional_group_masks: bool,
) -> tuple[dict[str, dict[str, dict[str, list[tuple[float, float]]]]], list[str]]:
    store: dict[str, dict[str, dict[str, list[tuple[float, float]]]]] = {
        spec.key: defaultdict(lambda: defaultdict(list)) for spec in PLOT_SPECS
    }
    topic_names: set[str] = set()

    prefix_iter = prefixes
    if not no_progress and tqdm is not None:
        prefix_iter = tqdm(prefixes, total=len(prefixes), desc=progress_label, unit="prefix")
    elif not no_progress and tqdm is None:
        print("tqdm not installed; running without progress bar.")

    for data_prefix in prefix_iter:
        topic_name = extract_topic_name(data_prefix)
        topic_names.add(topic_name)

        human_path = resolve_human_path(roots, data_prefix, models, human_filename)
        if human_path is not None:
            human_rows = read_csv_rows(
                human_path,
                respect_validity_labels=respect_validity_labels,
                likert_col=likert_col,
                data_roots=roots,
                initial_post_sidecar_roots=initial_post_sidecar_roots,
                tweet_sidecar_roots=tweet_sidecar_roots,
                apply_v3_additional_group_masks=apply_v3_additional_group_masks,
            )
            human_profile = build_conversation_profile(human_rows, default_group=data_prefix, likert_col=likert_col)
            for spec in PLOT_SPECS:
                points = spec.point_builder(human_profile)
                if points:
                    store[spec.key]["overall"][HUMAN_MODEL].extend(points)
                    store[spec.key][topic_name][HUMAN_MODEL].extend(points)

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
                apply_v3_additional_group_masks=apply_v3_additional_group_masks,
            )
            llm_profile = build_conversation_profile(llm_rows, default_group=data_prefix, likert_col=likert_col)
            for spec in PLOT_SPECS:
                points = spec.point_builder(llm_profile)
                if points:
                    store[spec.key]["overall"][model].extend(points)
                    store[spec.key][topic_name][model].extend(points)

    return store, sorted(topic_names)


def compute_line_fit(points: list[tuple[float, float]]) -> tuple[float, float] | None:
    if len(points) < 2:
        return None
    x_vals = [x for x, _ in points]
    y_vals = [y for _, y in points]
    mean_x = sum(x_vals) / len(x_vals)
    mean_y = sum(y_vals) / len(y_vals)
    denom = sum((x - mean_x) ** 2 for x in x_vals)
    if denom == 0.0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_vals, y_vals)) / denom
    intercept = mean_y - slope * mean_x
    return slope, intercept


def kendall_tau_c(x_values: list[float], y_values: list[float]) -> float:
    n = len(x_values)
    if n < 2:
        return math.nan
    unique_x = sorted(set(x_values))
    unique_y = sorted(set(y_values))
    m = min(len(unique_x), len(unique_y))
    if m <= 1:
        return math.nan

    concordant = 0
    discordant = 0
    for index in range(n):
        for other_index in range(index + 1, n):
            dx = x_values[index] - x_values[other_index]
            dy = y_values[index] - y_values[other_index]
            if dx == 0 or dy == 0:
                continue
            if (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1

    return (2.0 * m * (concordant - discordant)) / ((n * n) * (m - 1))


def padded_limits(values: list[float], default: tuple[float, float]) -> tuple[float, float]:
    if not values:
        return default
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        span = max(0.5, abs(low) * 0.15)
        return low - span, high + span
    pad = max(0.2, (high - low) * 0.08)
    return low - pad, high + pad


def jitter_points(points: list[tuple[float, float]], *, seed: int) -> tuple[list[float], list[float]]:
    rng = random.Random(seed)
    x_plot = [x + rng.uniform(-0.06, 0.06) for x, _ in points]
    y_plot = [y + rng.uniform(-0.06, 0.06) for _, y in points]
    return x_plot, y_plot


def format_topic_title(topic_name: str) -> str:
    return topic_name.replace("_", " ")


def build_display_order(models: list[str]) -> list[str]:
    synthetic_rows = [{"model": HUMAN_MODEL}, *({"model": model} for model in models)]
    return get_v2_display_model_order(synthetic_rows, allowed_models=models)


def dedupe_preserving_order(models: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for model_name in models:
        if model_name in seen:
            continue
        seen.add(model_name)
        ordered.append(model_name)
    return ordered


def background_base_models(base_models: list[str]) -> list[str]:
    return [model_name for model_name in base_models if model_name != LLAMA_8B_INSTRUCT_MODEL]


def is_primary_scatter_highlight(entity: str) -> bool:
    return entity == HUMAN_MODEL or entity == LLAMA_8B_INSTRUCT_MODEL or is_llama_sft_model(entity)


def build_post_train_full_fit_eval_root(
    *,
    family_key: str,
    roots: list[Path],
    human_filename: str,
    llm_filename: str,
    likert_col: str,
    initial_post_sidecar_roots: list[Path],
    tweet_sidecar_roots: list[Path],
    output_root: Path,
    report_version: str,
) -> Path | None:
    prefix_model_specs: dict[str, list[ProxyEvalModelSpec]] = {}
    family_prefixes: list[str] = []

    for suffix, prefix_file, _model_attr, _partition_label in POST_TRAIN_PARTITION_SPECS:
        if partition_pair_key(suffix) != family_key:
            continue
        if skip_v4_breadth_topic_post_train(family_key, report_version):
            continue

        raw_prefixes = read_lines(resolve_partition_prefix_file(prefix_file))
        valid_rows_by_prefix, _skipped_rows = load_valid_human_rows_by_prefix(
            roots=roots,
            prefixes=raw_prefixes,
            human_filename=human_filename,
            likert_col=likert_col,
            initial_post_sidecar_roots=initial_post_sidecar_roots,
            tweet_sidecar_roots=tweet_sidecar_roots,
        )
        valid_prefixes = [data_prefix for data_prefix in raw_prefixes if data_prefix in valid_rows_by_prefix]
        if not valid_prefixes:
            continue

        fit_result = fit_best_alphas(
            roots=roots,
            prefixes=valid_prefixes,
            human_filename=human_filename,
            llm_filename=llm_filename,
            likert_col=likert_col,
            initial_post_sidecar_roots=initial_post_sidecar_roots,
            tweet_sidecar_roots=tweet_sidecar_roots,
            coarse_steps=20,
            tol=1e-4,
            max_iter=64,
        )
        family_prefixes.extend(valid_prefixes)
        for data_prefix in valid_prefixes:
            prefix_model_specs[data_prefix] = [
                ProxyEvalModelSpec(
                    model_dir=FULL_FIT_MODEL_NAME,
                    alpha_public=float(fit_result["alpha_public"]),
                    alpha_private=float(fit_result["alpha_private"]),
                    fit_scope=suffix,
                )
            ]

    if not prefix_model_specs:
        return None

    family_eval_root = output_root / "_eval" / "standard_post_train_full_fit" / family_key
    materialize_proxy_eval_tree(
        roots=roots,
        prefixes=sorted(set(family_prefixes)),
        human_filename=human_filename,
        llm_filename=llm_filename,
        likert_col=likert_col,
        initial_post_sidecar_roots=initial_post_sidecar_roots,
        tweet_sidecar_roots=tweet_sidecar_roots,
        output_root=family_eval_root,
        prefix_model_specs=prefix_model_specs,
    )
    return family_eval_root


def render_scatter_axis(
    ax: object,
    *,
    spec: PlotSpec,
    entity_points: dict[str, list[tuple[float, float]]],
    legend_order: list[str],
    x_lim: tuple[float, float],
    y_lim: tuple[float, float],
    page_index: int,
    panel_index: int = 0,
    panel_title: str | None = None,
    footnote_width: int = 150,
    legend_fontsize: int = 9,
    footnote_fontsize: int = 8,
) -> None:
    legend_handles: dict[str, object] = {}
    rho_entries: list[str] = []
    draw_order = [entity for entity in legend_order if entity != HUMAN_MODEL] + [HUMAN_MODEL]

    for entity_index, entity in enumerate(draw_order):
        points = entity_points.get(entity, [])
        if not points:
            continue
        color = get_model_color(entity, "#777777")
        display_label = get_model_display_label(entity)
        is_human = entity == HUMAN_MODEL
        is_degroot = is_degroot_model(entity)
        is_primary_highlight = is_primary_scatter_highlight(entity)

        if is_human:
            point_alpha = 0.18
            point_zorder = 4
            line_width = LINE_WIDTH + 0.4
            line_alpha = 0.98
            line_zorder = 5
            line_style = "-"
        elif is_degroot:
            point_alpha = 0.08
            point_zorder = 3
            line_width = LINE_WIDTH + 0.1
            line_alpha = 0.95
            line_zorder = 4
            line_style = "--"
        elif is_primary_highlight:
            point_alpha = 0.12
            point_zorder = 3
            line_width = LINE_WIDTH + 0.2
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

        x_plot, y_plot = jitter_points(points, seed=page_index * 1000 + panel_index * 100 + entity_index)
        ax.scatter(
            x_plot,
            y_plot,
            s=POINT_SIZE,
            alpha=point_alpha,
            color=color,
            edgecolors="none",
            zorder=point_zorder,
        )

        line_fit = compute_line_fit(points)
        if line_fit is not None:
            slope, intercept = line_fit
            raw_x = [x for x, _ in points]
            x0 = min(raw_x)
            x1 = max(raw_x)
            legend_handles[entity] = ax.plot(
                [x0, x1],
                [intercept + slope * x0, intercept + slope * x1],
                color=color,
                linewidth=line_width,
                linestyle=line_style,
                alpha=line_alpha,
                zorder=line_zorder,
            )[0]

        rho = spearman_rho([x for x, _ in points], [y for _, y in points])
        rho_text = f"{rho:.2f}" if not math.isnan(rho) else "nan"
        rho_entries.append(f"{display_label}: rho={rho_text}, n={len(points)}")

    if spec.x_is_difference:
        ax.axvline(0.0, color="#7f7f7f", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axhline(0.0, color="#7f7f7f", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.grid(True, alpha=0.22)
    ax.set_axisbelow(True)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_xlabel(spec.x_label)
    ax.set_ylabel(spec.y_label)

    if panel_title:
        ax.set_title(panel_title, loc="left", fontweight="bold")

    legend_entities = [entity for entity in legend_order if entity in legend_handles]
    if legend_entities:
        ax.legend(
            [legend_handles[entity] for entity in legend_entities],
            [get_model_display_label(entity) for entity in legend_entities],
            loc="upper right",
            frameon=True,
            fontsize=legend_fontsize,
        )

    footnote_lines = textwrap.wrap("Spearman rho: " + " | ".join(rho_entries), width=footnote_width)
    ax.text(
        0.0,
        -0.23,
        "\n".join(footnote_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=footnote_fontsize,
        color="#444444",
    )


def filter_entity_points(
    entity_points: dict[str, list[tuple[float, float]]],
    allowed_entities: tuple[str, ...] | list[str] | None,
) -> dict[str, list[tuple[float, float]]]:
    if allowed_entities is None:
        return {entity: points for entity, points in entity_points.items() if points}
    allowed = set(allowed_entities)
    allowed.add(HUMAN_MODEL)
    return {entity: points for entity, points in entity_points.items() if entity in allowed and points}


def compute_scatter_limits(
    spec: PlotSpec,
    panels: list[dict[str, list[tuple[float, float]]]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    all_x = [x for entity_points in panels for points in entity_points.values() for x, _ in points]
    all_y = [y for entity_points in panels for points in entity_points.values() for _, y in points]
    x_lim = padded_limits(all_x, spec.x_default)
    y_lim = padded_limits(all_y, spec.y_default)
    if not spec.x_is_difference:
        x_lim = (max(spec.x_default[0], x_lim[0]), min(spec.x_default[1], x_lim[1]))
    return x_lim, y_lim


def render_cover_page(
    pdf: PdfPages,
    *,
    suite_label: str,
    plot_pair: MergedPlotPair,
    legend_order: list[str],
    scatter_panel_specs: tuple[tuple[str, tuple[str, ...]], ...] | None,
    apply_v3_additional_group_masks: bool,
) -> None:
    fig = plt.figure(figsize=(11.0, 8.5))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.axis("off")

    model_labels = [get_model_display_label(entity) for entity in legend_order if entity != HUMAN_MODEL]
    scatter_pages = [
        f"One train scatter page for {plot_pair.public_label.lower()} and {plot_pair.private_label.lower()}",
        f"One test scatter page for {plot_pair.public_label.lower()} and {plot_pair.private_label.lower()}",
    ]
    if scatter_panel_specs:
        scatter_pages = [
            f"One train scatter page for {panel_title}" for panel_title, _ in scatter_panel_specs
        ] + [
            f"One test scatter page for {panel_title}" for panel_title, _ in scatter_panel_specs
        ]

    mask_lines = [
        "Masks",
        "- Base page masks: golden_group_v0 aggregate / collection / split txt masks, depending on the page",
    ]
    if apply_v3_additional_group_masks:
        mask_lines.extend(
            [
                f"- Public/tweet additional mask root: {repo_relative_display_path(configured_public_mask_root())}",
                f"- Private/opinion additional mask root: {repo_relative_display_path(configured_private_mask_root())}",
            ]
        )
    else:
        mask_lines.append("- Additional tweet/opinion group masks: disabled for this unfiltered variant")

    lines = [
        suite_label,
        "Consolidated post-train scatter report",
        "",
        "Contents",
        f"- {plot_pair.public_label}: split across separate train and test pages",
        f"- {plot_pair.private_label}: split across separate train and test pages",
        *[f"- {line}" for line in scatter_pages],
        "- One train correlation-summary page with Spearman and Kendall bar plots",
        "- One test correlation-summary page with Spearman and Kendall bar plots",
        "",
        "Page layout",
        "- Train pages: Public left, Private right",
        "- Test pages: Public left, Private right",
        "",
        *mask_lines,
        "",
        "Models",
        f"- Human reference plus: {', '.join(model_labels)}",
    ]

    ax.text(0.08, 0.90, lines[0], fontsize=24, fontweight="bold", ha="left", va="top", color="#1a3a5c")
    ax.text(0.08, 0.84, lines[1], fontsize=14, ha="left", va="top", color="#3b4f63")
    ax.text(0.08, 0.76, "\n".join(lines[3:]), fontsize=11, ha="left", va="top", color="#222222", linespacing=1.35)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_summary_subplot(
    axes: tuple[object, object],
    *,
    entity_points: dict[str, list[tuple[float, float]]],
    legend_order: list[str],
    panel_title: str,
) -> None:
    entities = [entity for entity in legend_order if entity_points.get(entity)]
    if not entities:
        for axis, title in ((axes[0], "Spearman ρ"), (axes[1], "Kendall τ-c")):
            axis.set_title(f"{panel_title} ({title})", loc="left", fontweight="bold", fontsize=9)
            axis.text(0.5, 0.5, "No data", transform=axis.transAxes, ha="center", va="center", color="#666666")
            axis.set_axis_off()
        return

    correlations = []
    for entity in entities:
        points = entity_points[entity]
        x_vals = [x for x, _ in points]
        y_vals = [y for _, y in points]
        correlations.append(
            {
                "label": get_model_display_label(entity),
                "color": get_model_color(entity, "#777777"),
                "rho": spearman_rho(x_vals, y_vals),
                "tau_c": kendall_tau_c(x_vals, y_vals),
                "n": len(points),
            }
        )

    labels = [entry["label"] for entry in correlations]
    colors = [entry["color"] for entry in correlations]
    x_positions = list(range(len(correlations)))
    tick_fontsize = 7 if len(labels) <= 8 else 6
    for axis, metric_key, title in (
        (axes[0], "rho", "Spearman ρ"),
        (axes[1], "tau_c", "Kendall τ-c"),
    ):
        values = [entry[metric_key] for entry in correlations]
        plotted_values = [0.0 if math.isnan(value) else value for value in values]
        axis.bar(x_positions, plotted_values, color=colors, alpha=0.95)
        axis.axhline(0.0, color="#7f7f7f", linestyle="--", linewidth=1.0, alpha=0.8)
        axis.set_ylim(-1.0, 1.0)
        axis.grid(True, axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        axis.set_ylabel(title, fontsize=8)
        axis.tick_params(axis="y", labelsize=8)
        axis.set_title(f"{panel_title} ({title})", loc="left", fontweight="bold", fontsize=9, pad=1)
        for index, value in enumerate(values):
            label = "n/a" if math.isnan(value) else f"{value:.2f}"
            y_position = plotted_values[index] + (0.04 if plotted_values[index] >= 0 else -0.07)
            axis.text(
                index,
                y_position,
                f"{label}\n(n={correlations[index]['n']})",
                ha="center",
                va="bottom" if plotted_values[index] >= 0 else "top",
                fontsize=7,
                color="#333333",
            )
        axis.set_xticks(x_positions)
        axis.set_xticklabels(labels, rotation=20, ha="right", fontsize=tick_fontsize)
        axis.tick_params(axis="x", pad=0)


def render_scatter_split_page(
    pdf: PdfPages,
    *,
    suite_label: str,
    page_title: str,
    plot_pair: MergedPlotPair,
    subset_store: dict[str, dict[str, list[tuple[float, float]]]],
    legend_order: list[str],
    page_index: int,
    subset_label: str,
    allowed_entities: tuple[str, ...] | None = None,
) -> None:
    public_spec = PLOT_SPEC_BY_KEY[plot_pair.public_spec_key]
    private_spec = PLOT_SPEC_BY_KEY[plot_pair.private_spec_key]
    public_points = filter_entity_points(subset_store.get(public_spec.key, {}).get("overall", {}), allowed_entities)
    private_points = filter_entity_points(subset_store.get(private_spec.key, {}).get("overall", {}), allowed_entities)
    public_limits = compute_scatter_limits(public_spec, [public_points])
    private_limits = compute_scatter_limits(private_spec, [private_points])

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 7.8))
    panel_defs = [
        (axes[0], f"Public {subset_label.title()}", public_spec, public_points, public_limits, 0),
        (axes[1], f"Private {subset_label.title()}", private_spec, private_points, private_limits, 1),
    ]
    for ax, panel_title, spec, entity_points, limits, panel_index in panel_defs:
        x_lim, y_lim = limits
        render_scatter_axis(
            ax,
            spec=spec,
            entity_points=entity_points,
            legend_order=[entity for entity in legend_order if entity_points.get(entity)],
            x_lim=x_lim,
            y_lim=y_lim,
            page_index=page_index,
            panel_index=panel_index,
            panel_title=panel_title,
            footnote_width=60,
            legend_fontsize=7,
            footnote_fontsize=7,
        )
        ax.tick_params(axis="both", labelsize=8)

    fig.suptitle(
        f"{page_title}\n{suite_label}",
        x=0.01,
        y=0.985,
        ha="left",
        va="top",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.92))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_summary_split_page(
    pdf: PdfPages,
    *,
    suite_label: str,
    page_title: str,
    plot_pair: MergedPlotPair,
    subset_store: dict[str, dict[str, list[tuple[float, float]]]],
    legend_order: list[str],
    subset_label: str,
) -> None:
    public_spec = PLOT_SPEC_BY_KEY[plot_pair.public_spec_key]
    private_spec = PLOT_SPEC_BY_KEY[plot_pair.private_spec_key]

    fig = plt.figure(figsize=(14.2, 8.6))
    outer = fig.add_gridspec(1, 2, wspace=0.24)
    panel_defs = [
        (f"Public {subset_label.title()}", subset_store.get(public_spec.key, {}).get("overall", {}), outer[0, 0]),
        (f"Private {subset_label.title()}", subset_store.get(private_spec.key, {}).get("overall", {}), outer[0, 1]),
    ]

    for panel_title, entity_points, cell in panel_defs:
        inner = cell.subgridspec(2, 1, hspace=0.24)
        axes = (fig.add_subplot(inner[0, 0]), fig.add_subplot(inner[1, 0]))
        render_summary_subplot(
            axes,
            entity_points=entity_points,
            legend_order=[entity for entity in legend_order if entity_points.get(entity)],
            panel_title=panel_title,
        )

    fig.suptitle(
        f"{page_title}\n{suite_label}",
        x=0.01,
        y=0.99,
        ha="left",
        va="top",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.94))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_base_cover_page(
    pdf: PdfPages,
    *,
    suite: ReportSuite,
    plot_pair: MergedPlotPair,
    legend_order: list[str],
    apply_v3_additional_group_masks: bool,
) -> None:
    fig = plt.figure(figsize=(11.0, 8.5))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.axis("off")

    model_labels = [get_model_display_label(entity) for entity in legend_order if entity != HUMAN_MODEL]
    page_lines = [
        "- One merged scatter page for overall public and private opinion panels",
        "- One merged correlation-summary page for overall public and private opinion panels",
    ]
    if suite.include_topic_pages:
        page_lines.append(f"- One merged scatter page per {suite.topic_label.lower()}")

    mask_lines = [
        "Masks",
        "- Base page masks: golden_group_v0 aggregate / collection / split txt masks, depending on the page",
    ]
    if apply_v3_additional_group_masks:
        mask_lines.extend(
            [
                f"- Public/tweet additional mask root: {repo_relative_display_path(configured_public_mask_root())}",
                f"- Private/opinion additional mask root: {repo_relative_display_path(configured_private_mask_root())}",
            ]
        )
    else:
        mask_lines.append("- Additional tweet/opinion group masks: disabled for this unfiltered variant")

    lines = [
        suite.overall_label,
        "Consolidated base-model scatter report",
        "",
        "Contents",
        f"- {plot_pair.public_label}: merged into the same report as {plot_pair.private_label.lower()}",
        f"- {plot_pair.private_label}: merged into the same report as {plot_pair.public_label.lower()}",
        *page_lines,
        "",
        "Panel layout",
        "- Left: Public opinion",
        "- Right: Private opinion",
        "",
        *mask_lines,
        "",
        "Models",
        f"- Human reference plus: {', '.join(model_labels)}",
    ]

    ax.text(0.08, 0.90, lines[0], fontsize=24, fontweight="bold", ha="left", va="top", color="#1a3a5c")
    ax.text(0.08, 0.84, lines[1], fontsize=14, ha="left", va="top", color="#3b4f63")
    ax.text(0.08, 0.76, "\n".join(lines[3:]), fontsize=11, ha="left", va="top", color="#222222", linespacing=1.35)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_dual_scatter_page(
    pdf: PdfPages,
    *,
    suite: ReportSuite,
    page_title: str,
    plot_pair: MergedPlotPair,
    store: dict[str, dict[str, dict[str, list[tuple[float, float]]]]],
    legend_order: list[str],
    scope_key: str,
    scope_label: str,
    page_index: int,
) -> None:
    public_spec = PLOT_SPEC_BY_KEY[plot_pair.public_spec_key]
    private_spec = PLOT_SPEC_BY_KEY[plot_pair.private_spec_key]
    public_points = store.get(public_spec.key, {}).get(scope_key, {})
    private_points = store.get(private_spec.key, {}).get(scope_key, {})
    public_limits = compute_scatter_limits(public_spec, [public_points])
    private_limits = compute_scatter_limits(private_spec, [private_points])

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 7.6))
    panel_defs = [
        (axes[0], plot_pair.public_label, public_spec, public_points, public_limits, 0),
        (axes[1], plot_pair.private_label, private_spec, private_points, private_limits, 1),
    ]
    for ax, panel_title, spec, entity_points, limits, panel_index in panel_defs:
        x_lim, y_lim = limits
        render_scatter_axis(
            ax,
            spec=spec,
            entity_points=entity_points,
            legend_order=[entity for entity in legend_order if entity_points.get(entity)],
            x_lim=x_lim,
            y_lim=y_lim,
            page_index=page_index,
            panel_index=panel_index,
            panel_title=panel_title,
            footnote_width=64,
            legend_fontsize=8,
            footnote_fontsize=7,
        )
        ax.tick_params(axis="both", labelsize=8)

    fig.suptitle(
        f"{page_title}\n{scope_label}",
        x=0.01,
        y=0.98,
        ha="left",
        va="top",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.92))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_dual_summary_page(
    pdf: PdfPages,
    *,
    suite: ReportSuite,
    page_title: str,
    plot_pair: MergedPlotPair,
    store: dict[str, dict[str, dict[str, list[tuple[float, float]]]]],
    legend_order: list[str],
    scope_key: str,
    scope_label: str,
) -> None:
    public_spec = PLOT_SPEC_BY_KEY[plot_pair.public_spec_key]
    private_spec = PLOT_SPEC_BY_KEY[plot_pair.private_spec_key]
    public_points = store.get(public_spec.key, {}).get(scope_key, {})
    private_points = store.get(private_spec.key, {}).get(scope_key, {})

    fig = plt.figure(figsize=(14.2, 8.4))
    outer = fig.add_gridspec(1, 2, wspace=0.24)
    panel_defs = [
        (plot_pair.public_label, public_points, outer[0, 0]),
        (plot_pair.private_label, private_points, outer[0, 1]),
    ]

    for panel_title, entity_points, cell in panel_defs:
        inner = cell.subgridspec(2, 1, hspace=0.10)
        axes = (fig.add_subplot(inner[0, 0]), fig.add_subplot(inner[1, 0]))
        render_summary_subplot(
            axes,
            entity_points=entity_points,
            legend_order=[entity for entity in legend_order if entity_points.get(entity)],
            panel_title=panel_title,
        )

    fig.suptitle(
        f"{page_title}\n{scope_label}",
        x=0.01,
        y=0.99,
        ha="left",
        va="top",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_merged_base_reports(
    *,
    output_root: Path,
    report_version: str,
    suite: ReportSuite,
    store: dict[str, dict[str, dict[str, list[tuple[float, float]]]]],
    topics: list[str],
    apply_v3_additional_group_masks: bool,
    write_pdf: bool = True,
) -> list[Path]:
    if plt is None or PdfPages is None:
        raise RuntimeError("matplotlib with PdfPages support is required to render scatter reports.")

    output_dir = output_root / suite.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_paths: list[Path] = []
    legend_order = build_display_order(suite.models)

    for plot_pair in MERGED_PLOT_PAIRS:
        report_path = output_dir / f"report_{suite.report_prefix}_{plot_pair.report_suffix}_scatter_{report_version}.pdf"
        if write_pdf:
            with PdfPages(report_path) as pdf:
                render_base_cover_page(
                    pdf,
                    suite=suite,
                    plot_pair=plot_pair,
                    legend_order=legend_order,
                    apply_v3_additional_group_masks=apply_v3_additional_group_masks,
                )
                render_dual_scatter_page(
                    pdf,
                    suite=suite,
                    page_title="Overall Scatter Panels",
                    plot_pair=plot_pair,
                    store=store,
                    legend_order=legend_order,
                    scope_key="overall",
                    scope_label=suite.overall_label,
                    page_index=1,
                )
                render_dual_summary_page(
                    pdf,
                    suite=suite,
                    page_title="Overall Correlation Summary",
                    plot_pair=plot_pair,
                    store=store,
                    legend_order=legend_order,
                    scope_key="overall",
                    scope_label=suite.overall_label,
                )

                if suite.include_topic_pages:
                    for topic_index, topic_name in enumerate(topics, start=2):
                        public_points = store.get(plot_pair.public_spec_key, {}).get(topic_name, {})
                        private_points = store.get(plot_pair.private_spec_key, {}).get(topic_name, {})
                        if not any(public_points.get(entity) for entity in legend_order) and not any(private_points.get(entity) for entity in legend_order):
                            continue
                        render_dual_scatter_page(
                            pdf,
                            suite=suite,
                            page_title="Topic Scatter Panels",
                            plot_pair=plot_pair,
                            store=store,
                            legend_order=legend_order,
                            scope_key=topic_name,
                            scope_label=f"{suite.topic_label}: {format_topic_title(topic_name)}",
                            page_index=topic_index,
                        )

        write_merged_base_report_csv(
            report_path.with_suffix(".csv"),
            report_path=report_path,
            report_version=report_version,
            suite=suite,
            plot_pair=plot_pair,
            store=store,
            legend_order=legend_order,
        )

        report_paths.append(report_path)
        if write_pdf:
            print(f"Wrote: {report_path}")
        print(f"Wrote CSV: {report_path.with_suffix('.csv')}")

    return report_paths


def render_scatter_page(
    pdf: PdfPages,
    *,
    spec: PlotSpec,
    suite: ReportSuite,
    topic_name: str,
    entity_points: dict[str, list[tuple[float, float]]],
    legend_order: list[str],
    page_index: int,
) -> None:
    all_x = [x for points in entity_points.values() for x, _ in points]
    all_y = [y for points in entity_points.values() for _, y in points]
    x_lim = padded_limits(all_x, spec.x_default)
    y_lim = padded_limits(all_y, spec.y_default)

    if not spec.x_is_difference:
        x_lim = (max(spec.x_default[0], x_lim[0]), min(spec.x_default[1], x_lim[1]))

    scope_label = suite.overall_label if topic_name == "overall" else f"{suite.topic_label}: {format_topic_title(topic_name)}"

    if suite.scatter_panel_specs:
        fig, axes = plt.subplots(1, len(suite.scatter_panel_specs), figsize=(14.4, 7.0), sharex=True, sharey=True)
        axes_list = list(axes if isinstance(axes, (list, tuple)) else getattr(axes, "flat", [axes]))
        if hasattr(axes, "flat"):
            axes_list = list(axes.flat)
        for panel_index, ((panel_title, panel_models), ax) in enumerate(zip(suite.scatter_panel_specs, axes_list)):
            panel_order = [entity for entity in legend_order if entity in panel_models]
            panel_points = {
                entity: entity_points[entity]
                for entity in [*panel_order, HUMAN_MODEL]
                if entity_points.get(entity)
            }
            render_scatter_axis(
                ax,
                spec=spec,
                entity_points=panel_points,
                legend_order=[*panel_order, HUMAN_MODEL],
                x_lim=x_lim,
                y_lim=y_lim,
                page_index=page_index,
                panel_index=panel_index,
                panel_title=panel_title,
                footnote_width=58,
            )
        fig.suptitle(f"{spec.page_title}\n{scope_label}", x=0.01, y=0.98, ha="left", va="top", fontweight="bold")
        fig.tight_layout(rect=(0.0, 0.11, 1.0, 0.90))
    else:
        fig, ax = plt.subplots(figsize=(10.8, 7.2))
        render_scatter_axis(
            ax,
            spec=spec,
            entity_points=entity_points,
            legend_order=legend_order,
            x_lim=x_lim,
            y_lim=y_lim,
            page_index=page_index,
            footnote_width=150,
        )
        ax.set_title(f"{spec.page_title}\n{scope_label}", loc="left", fontweight="bold")
        fig.tight_layout(rect=(0.0, 0.09, 1.0, 1.0))

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_summary_page(
    pdf: PdfPages,
    *,
    spec: PlotSpec,
    suite: ReportSuite,
    entity_points: dict[str, list[tuple[float, float]]],
    legend_order: list[str],
) -> None:
    entities = [entity for entity in legend_order if entity_points.get(entity)]
    if not entities:
        return

    fig, axes = plt.subplots(2, 1, figsize=(11.0, 7.4), sharex=True)
    correlations = []
    for entity in entities:
        points = entity_points[entity]
        x_vals = [x for x, _ in points]
        y_vals = [y for _, y in points]
        correlations.append(
            {
                "entity": entity,
                "label": get_model_display_label(entity),
                "color": get_model_color(entity, "#777777"),
                "rho": spearman_rho(x_vals, y_vals),
                "tau_c": kendall_tau_c(x_vals, y_vals),
                "n": len(points),
            }
        )

    labels = [entry["label"] for entry in correlations]
    colors = [entry["color"] for entry in correlations]
    x_positions = list(range(len(correlations)))

    for axis, metric_key, title in (
        (axes[0], "rho", "Spearman ρ"),
        (axes[1], "tau_c", "Kendall τ-c"),
    ):
        values = [entry[metric_key] for entry in correlations]
        plotted_values = [0.0 if math.isnan(value) else value for value in values]
        axis.bar(x_positions, plotted_values, color=colors, alpha=0.95)
        axis.axhline(0.0, color="#7f7f7f", linestyle="--", linewidth=1.0, alpha=0.8)
        axis.set_ylim(-1.0, 1.0)
        axis.set_ylabel(title)
        axis.grid(True, axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        for index, value in enumerate(values):
            label = "n/a" if math.isnan(value) else f"{value:.2f}"
            y_position = plotted_values[index] + (0.04 if plotted_values[index] >= 0 else -0.07)
            axis.text(
                index,
                y_position,
                f"{label}\n(n={correlations[index]['n']})",
                ha="center",
                va="bottom" if plotted_values[index] >= 0 else "top",
                fontsize=8,
                color="#333333",
            )

    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    scope_title = f"{spec.page_title}\n{suite.overall_label} Correlation Summary"
    fig.suptitle(scope_title, x=0.01, y=0.99, ha="left", va="top", fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.94))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_suite_reports(
    *,
    output_root: Path,
    report_version: str,
    suite: ReportSuite,
    store: dict[str, dict[str, dict[str, list[tuple[float, float]]]]],
    topics: list[str],
) -> list[Path]:
    if plt is None or PdfPages is None:
        raise RuntimeError("matplotlib with PdfPages support is required to render scatter reports.")

    output_dir = output_root / suite.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_paths: list[Path] = []
    legend_order = build_display_order(suite.models)

    for spec in PLOT_SPECS:
        report_path = output_dir / f"report_{suite.report_prefix}_{spec.report_suffix}_scatter_{report_version}.pdf"
        with PdfPages(report_path) as pdf:
            page_index = 0
            overall_points = store[spec.key].get("overall", {})
            if any(overall_points.get(entity) for entity in legend_order):
                page_index += 1
                render_scatter_page(
                    pdf,
                    spec=spec,
                    suite=suite,
                    topic_name="overall",
                    entity_points=overall_points,
                    legend_order=legend_order,
                    page_index=page_index,
                )
                render_summary_page(
                    pdf,
                    spec=spec,
                    suite=suite,
                    entity_points=overall_points,
                    legend_order=legend_order,
                )

            if suite.include_topic_pages:
                for topic_name in topics:
                    entity_points = store[spec.key].get(topic_name, {})
                    if not any(entity_points.get(entity) for entity in legend_order):
                        continue
                    page_index += 1
                    render_scatter_page(
                        pdf,
                        spec=spec,
                        suite=suite,
                        topic_name=topic_name,
                        entity_points=entity_points,
                        legend_order=legend_order,
                        page_index=page_index,
                    )
        report_paths.append(report_path)
        print(f"Wrote: {report_path}")

    return report_paths


def cleanup_legacy_root_reports(output_root: Path, report_version: str) -> None:
    legacy_names = [
        f"report_depth_base_model_tweet1_vs_tweet_change_scatter_{report_version}.pdf",
        f"report_depth_base_model_partner_tweet_gap_vs_tweet_change_scatter_{report_version}.pdf",
        f"report_breadth_base_model_tweet1_vs_tweet_change_scatter_{report_version}.pdf",
        f"report_breadth_base_model_partner_tweet_gap_vs_tweet_change_scatter_{report_version}.pdf",
    ]
    for legacy_name in legacy_names:
        legacy_path = output_root / legacy_name
        replacement_path = output_root / "base_model" / legacy_name
        if legacy_path.exists() and replacement_path.exists():
            legacy_path.unlink()
            print(f"Removed legacy root report: {legacy_path}")
        base_model_path = output_root / "base_model" / legacy_name
        if base_model_path.exists():
            base_model_path.unlink()
            print(f"Removed legacy base-model report: {base_model_path}")


def cleanup_legacy_post_train_reports(output_root: Path, report_version: str) -> None:
    legacy_names = [f"report_breadth_post_train_{spec.report_suffix}_scatter_{report_version}.pdf" for spec in PLOT_SPECS]
    split_prefixes = [
        "breadth_topic_train_post_train",
        "breadth_topic_test_post_train",
        "breadth_group_train_post_train",
        "breadth_group_test_post_train",
        "depth_topic_train_post_train",
        "depth_topic_test_post_train",
        "depth_group_train_post_train",
        "depth_group_test_post_train",
    ]
    legacy_names.extend(
        f"report_{prefix}_{spec.report_suffix}_scatter_{report_version}.pdf"
        for prefix in split_prefixes
        for spec in PLOT_SPECS
    )
    for legacy_name in legacy_names:
        legacy_path = output_root / "post_train" / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()
            print(f"Removed legacy post-train report: {legacy_path}")


def correlation_summary_rows(
    *,
    entity_points: dict[str, list[tuple[float, float]]],
    legend_order: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for entity in [entity for entity in legend_order if entity_points.get(entity)]:
        points = entity_points[entity]
        x_vals = [x for x, _ in points]
        y_vals = [y for _, y in points]
        rows.append(
            {
                "entity": entity,
                "entity_display": get_model_display_label(entity),
                "model_dir": "" if entity == HUMAN_MODEL else entity,
                "n_points": len(points),
                "spearman_rho": spearman_rho(x_vals, y_vals),
                "kendall_tau_c": kendall_tau_c(x_vals, y_vals),
            }
        )
    return rows


def write_csv_rows(csv_path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> Path:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def write_merged_base_report_csv(
    csv_path: Path,
    *,
    report_path: Path,
    report_version: str,
    suite: ReportSuite,
    plot_pair: MergedPlotPair,
    store: dict[str, dict[str, dict[str, list[tuple[float, float]]]]],
    legend_order: list[str],
) -> Path:
    fieldnames = [
        "row_type",
        "report_relpath",
        "report_version",
        "suite_scope",
        "report_prefix",
        "plot_suffix",
        "subset",
        "scope_key",
        "scope_label",
        "panel_key",
        "panel_label",
        "entity",
        "entity_display",
        "model_dir",
        "point_index",
        "x_value",
        "y_value",
        "n_points",
        "spearman_rho",
        "kendall_tau_c",
    ]
    rows: list[dict[str, object]] = []
    report_relpath = str(report_path.resolve().relative_to(REPO_ROOT))
    panel_specs = [
        ("public", plot_pair.public_label, store.get(plot_pair.public_spec_key, {}).get("overall", {})),
        ("private", plot_pair.private_label, store.get(plot_pair.private_spec_key, {}).get("overall", {})),
    ]

    for panel_key, panel_label, entity_points in panel_specs:
        for entity in [entity for entity in legend_order if entity_points.get(entity)]:
            for point_index, (x_value, y_value) in enumerate(entity_points[entity], start=1):
                rows.append(
                    {
                        "row_type": "point",
                        "report_relpath": report_relpath,
                        "report_version": report_version,
                        "suite_scope": suite.overall_label,
                        "report_prefix": suite.report_prefix,
                        "plot_suffix": plot_pair.report_suffix,
                        "subset": "overall",
                        "scope_key": "overall",
                        "scope_label": suite.overall_label,
                        "panel_key": panel_key,
                        "panel_label": panel_label,
                        "entity": entity,
                        "entity_display": get_model_display_label(entity),
                        "model_dir": "" if entity == HUMAN_MODEL else entity,
                        "point_index": point_index,
                        "x_value": x_value,
                        "y_value": y_value,
                        "n_points": "",
                        "spearman_rho": "",
                        "kendall_tau_c": "",
                    }
                )
        for summary_row in correlation_summary_rows(entity_points=entity_points, legend_order=legend_order):
            rows.append(
                {
                    "row_type": "correlation_summary",
                    "report_relpath": report_relpath,
                    "report_version": report_version,
                    "suite_scope": suite.overall_label,
                    "report_prefix": suite.report_prefix,
                    "plot_suffix": plot_pair.report_suffix,
                    "subset": "overall",
                    "scope_key": "overall",
                    "scope_label": suite.overall_label,
                    "panel_key": panel_key,
                    "panel_label": panel_label,
                    "entity": summary_row["entity"],
                    "entity_display": summary_row["entity_display"],
                    "model_dir": summary_row["model_dir"],
                    "point_index": "",
                    "x_value": "",
                    "y_value": "",
                    "n_points": summary_row["n_points"],
                    "spearman_rho": summary_row["spearman_rho"],
                    "kendall_tau_c": summary_row["kendall_tau_c"],
                }
            )

    return write_csv_rows(csv_path, fieldnames, rows)


def write_merged_post_train_report_csv(
    csv_path: Path,
    *,
    report_path: Path,
    report_version: str,
    suite: MergedPostTrainSuite,
    plot_pair: MergedPlotPair,
    train_store: dict[str, dict[str, dict[str, list[tuple[float, float]]]]],
    test_store: dict[str, dict[str, dict[str, list[tuple[float, float]]]]],
    legend_order: list[str],
) -> Path:
    fieldnames = [
        "row_type",
        "report_relpath",
        "report_version",
        "suite_scope",
        "report_prefix",
        "plot_suffix",
        "subset",
        "scope_key",
        "scope_label",
        "panel_key",
        "panel_label",
        "entity",
        "entity_display",
        "model_dir",
        "point_index",
        "x_value",
        "y_value",
        "n_points",
        "spearman_rho",
        "kendall_tau_c",
    ]
    rows: list[dict[str, object]] = []
    report_relpath = str(report_path.resolve().relative_to(REPO_ROOT))

    for subset_key, scope_label, subset_store in (
        ("train", "Train Split Overall", train_store),
        ("test", "Test Split Overall", test_store),
    ):
        panel_specs = [
            ("public", plot_pair.public_label, subset_store.get(plot_pair.public_spec_key, {}).get("overall", {})),
            ("private", plot_pair.private_label, subset_store.get(plot_pair.private_spec_key, {}).get("overall", {})),
        ]
        for panel_key, panel_label, entity_points in panel_specs:
            for entity in [entity for entity in legend_order if entity_points.get(entity)]:
                for point_index, (x_value, y_value) in enumerate(entity_points[entity], start=1):
                    rows.append(
                        {
                            "row_type": "point",
                            "report_relpath": report_relpath,
                            "report_version": report_version,
                            "suite_scope": suite.scope_label,
                            "report_prefix": suite.report_prefix,
                            "plot_suffix": plot_pair.report_suffix,
                            "subset": subset_key,
                            "scope_key": "overall",
                            "scope_label": scope_label,
                            "panel_key": panel_key,
                            "panel_label": panel_label,
                            "entity": entity,
                            "entity_display": get_model_display_label(entity),
                            "model_dir": "" if entity == HUMAN_MODEL else entity,
                            "point_index": point_index,
                            "x_value": x_value,
                            "y_value": y_value,
                            "n_points": "",
                            "spearman_rho": "",
                            "kendall_tau_c": "",
                        }
                    )
            for summary_row in correlation_summary_rows(entity_points=entity_points, legend_order=legend_order):
                rows.append(
                    {
                        "row_type": "correlation_summary",
                        "report_relpath": report_relpath,
                        "report_version": report_version,
                        "suite_scope": suite.scope_label,
                        "report_prefix": suite.report_prefix,
                        "plot_suffix": plot_pair.report_suffix,
                        "subset": subset_key,
                        "scope_key": "overall",
                        "scope_label": scope_label,
                        "panel_key": panel_key,
                        "panel_label": panel_label,
                        "entity": summary_row["entity"],
                        "entity_display": summary_row["entity_display"],
                        "model_dir": summary_row["model_dir"],
                        "point_index": "",
                        "x_value": "",
                        "y_value": "",
                        "n_points": summary_row["n_points"],
                        "spearman_rho": summary_row["spearman_rho"],
                        "kendall_tau_c": summary_row["kendall_tau_c"],
                    }
                )

    return write_csv_rows(csv_path, fieldnames, rows)


def build_base_report_suites(base_models: list[str]) -> list[ReportSuite]:
    return [
        ReportSuite(
            report_prefix="depth_base_model",
            output_subdir="base_model",
            models=base_models,
            prefix_file=resolve_existing_or_script_relative(DEFAULT_DEPTH_MASK),
            overall_label="Overall Across All Depth Topics",
            topic_label="Depth Topic",
            include_topic_pages=True,
            progress_label="Depth base-model scatter plots",
        ),
        ReportSuite(
            report_prefix="breadth_base_model",
            output_subdir="base_model",
            models=base_models,
            prefix_file=resolve_existing_or_script_relative(DEFAULT_BREADTH_MASK),
            overall_label="Overall Across All Breadth Topics",
            topic_label="Breadth Topic",
            include_topic_pages=False,
            progress_label="Breadth base-model scatter plots",
        ),
    ]


def build_merged_post_train_suites(report_version: str, base_models: list[str] | None = None) -> list[MergedPostTrainSuite]:
    model_groups = post_train_model_groups(report_version)
    background_models = background_base_models(base_models or [])
    depth_topic_models = model_groups["depth_topic"]
    depth_group_models = model_groups["depth_group"]
    depth_topic_llama_models = [
        model_name for model_name in depth_topic_models if model_name not in {GPT_4O_MINI_BASE_MODEL, GPT_DEPTH_TOPIC_SFT_MODEL}
    ]
    depth_group_llama_models = [
        model_name for model_name in depth_group_models if model_name not in {GPT_4O_MINI_BASE_MODEL, GPT_DEPTH_GROUP_SFT_MODEL}
    ]
    suites: list[MergedPostTrainSuite] = []
    if not skip_v4_breadth_topic_post_train("breadth_topic", report_version):
        suites.append(
            MergedPostTrainSuite(
                report_prefix="breadth_topic_post_train",
                output_subdir="post_train",
                models=dedupe_preserving_order([
                    *model_groups["breadth_topic"],
                    *background_models,
                    FULL_FIT_MODEL_NAME,
                ])
                if report_version_family(report_version) == "v4"
                else model_groups["breadth_topic"],
                train_prefix_file=resolve_existing_or_script_relative(DEFAULT_BREADTH_TOPIC_TRAIN_MASK),
                test_prefix_file=resolve_existing_or_script_relative(DEFAULT_BREADTH_TOPIC_TEST_MASK),
                scope_label="Breadth Topic Post-Train",
                progress_label_train="Breadth topic train post-train scatter plots",
                progress_label_test="Breadth topic test post-train scatter plots",
            )
        )
    suites.extend([
        MergedPostTrainSuite(
            report_prefix="breadth_group_post_train",
            output_subdir="post_train",
            models=dedupe_preserving_order([
                *model_groups["breadth_group"],
                *background_models,
                FULL_FIT_MODEL_NAME,
            ])
            if report_version_family(report_version) == "v4"
            else model_groups["breadth_group"],
            train_prefix_file=resolve_existing_or_script_relative(DEFAULT_BREADTH_GROUP_TRAIN_MASK),
            test_prefix_file=resolve_existing_or_script_relative(DEFAULT_BREADTH_GROUP_TEST_MASK),
            scope_label="Breadth Group Post-Train",
            progress_label_train="Breadth group train post-train scatter plots",
            progress_label_test="Breadth group test post-train scatter plots",
        ),
        MergedPostTrainSuite(
            report_prefix="depth_topic_post_train",
            output_subdir="post_train",
            models=depth_topic_models,
            train_prefix_file=resolve_existing_or_script_relative(DEFAULT_DEPTH_TOPIC_TRAIN_MASK),
            test_prefix_file=resolve_existing_or_script_relative(DEFAULT_DEPTH_TOPIC_TEST_MASK),
            scope_label="Depth Topic Post-Train",
            progress_label_train="Depth topic train post-train scatter plots",
            progress_label_test="Depth topic test post-train scatter plots",
            scatter_panel_specs=(
                ("LLaMA", tuple(depth_topic_llama_models)),
                ("GPT", (GPT_4O_MINI_BASE_MODEL, GPT_DEPTH_TOPIC_SFT_MODEL)),
            ),
        ),
        MergedPostTrainSuite(
            report_prefix="depth_group_post_train",
            output_subdir="post_train",
            models=depth_group_models,
            train_prefix_file=resolve_existing_or_script_relative(DEFAULT_DEPTH_GROUP_TRAIN_MASK),
            test_prefix_file=resolve_existing_or_script_relative(DEFAULT_DEPTH_GROUP_TEST_MASK),
            scope_label="Depth Group Post-Train",
            progress_label_train="Depth group train post-train scatter plots",
            progress_label_test="Depth group test post-train scatter plots",
            scatter_panel_specs=(
                ("LLaMA", tuple(depth_group_llama_models)),
                ("GPT", (GPT_4O_MINI_BASE_MODEL, GPT_DEPTH_GROUP_SFT_MODEL)),
            ),
        ),
    ])
    return suites


def render_merged_post_train_reports(
    *,
    output_root: Path,
    report_version: str,
    suite: MergedPostTrainSuite,
    train_store: dict[str, dict[str, dict[str, list[tuple[float, float]]]]],
    test_store: dict[str, dict[str, dict[str, list[tuple[float, float]]]]],
    apply_v3_additional_group_masks: bool,
    write_pdf: bool = True,
) -> list[Path]:
    if plt is None or PdfPages is None:
        raise RuntimeError("matplotlib with PdfPages support is required to render scatter reports.")

    output_dir = output_root / suite.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_paths: list[Path] = []
    legend_order = build_display_order(suite.models)

    for plot_pair in MERGED_PLOT_PAIRS:
        report_path = output_dir / f"report_{suite.report_prefix}_{plot_pair.report_suffix}_scatter_{report_version}.pdf"
        if write_pdf:
            with PdfPages(report_path) as pdf:
                page_index = 1
                render_cover_page(
                    pdf,
                    suite_label=suite.scope_label,
                    plot_pair=plot_pair,
                    legend_order=legend_order,
                    scatter_panel_specs=suite.scatter_panel_specs,
                    apply_v3_additional_group_masks=apply_v3_additional_group_masks,
                )
                if suite.scatter_panel_specs:
                    for panel_title, panel_models in suite.scatter_panel_specs:
                        render_scatter_split_page(
                            pdf,
                            suite_label=suite.scope_label,
                            page_title=f"{panel_title} Train Scatter Panels",
                            plot_pair=plot_pair,
                            subset_store=train_store,
                            legend_order=legend_order,
                            page_index=page_index,
                            subset_label="train",
                            allowed_entities=panel_models,
                        )
                        page_index += 1
                        render_scatter_split_page(
                            pdf,
                            suite_label=suite.scope_label,
                            page_title=f"{panel_title} Test Scatter Panels",
                            plot_pair=plot_pair,
                            subset_store=test_store,
                            legend_order=legend_order,
                            page_index=page_index,
                            subset_label="test",
                            allowed_entities=panel_models,
                        )
                        page_index += 1
                else:
                    render_scatter_split_page(
                        pdf,
                        suite_label=suite.scope_label,
                        page_title="Train Scatter Panels",
                        plot_pair=plot_pair,
                        subset_store=train_store,
                        legend_order=legend_order,
                        page_index=page_index,
                        subset_label="train",
                    )
                    page_index += 1
                    render_scatter_split_page(
                        pdf,
                        suite_label=suite.scope_label,
                        page_title="Test Scatter Panels",
                        plot_pair=plot_pair,
                        subset_store=test_store,
                        legend_order=legend_order,
                        page_index=page_index,
                        subset_label="test",
                    )
                    page_index += 1
                render_summary_split_page(
                    pdf,
                    suite_label=suite.scope_label,
                    page_title="Train Correlation Summary",
                    plot_pair=plot_pair,
                    subset_store=train_store,
                    legend_order=legend_order,
                    subset_label="train",
                )
                render_summary_split_page(
                    pdf,
                    suite_label=suite.scope_label,
                    page_title="Test Correlation Summary",
                    plot_pair=plot_pair,
                    subset_store=test_store,
                    legend_order=legend_order,
                    subset_label="test",
                )
        write_merged_post_train_report_csv(
            report_path.with_suffix(".csv"),
            report_path=report_path,
            report_version=report_version,
            suite=suite,
            plot_pair=plot_pair,
            train_store=train_store,
            test_store=test_store,
            legend_order=legend_order,
        )
        report_paths.append(report_path)
        if write_pdf:
            print(f"Wrote: {report_path}")
        print(f"Wrote CSV: {report_path.with_suffix('.csv')}")

    return report_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate scatter plot reports.")
    parser.add_argument(
        "--data-root",
        action="append",
        default=["../../result/eval_v3/human_llm"],
        help="Data root(s) containing per-prefix eval CSVs.",
    )
    parser.add_argument(
        "--models-file",
        default="models.txt",
        help="Text file listing base models to include.",
    )
    parser.add_argument(
        "--human-filename",
        default=None,
        help="Human CSV filename. Defaults to the canonical file for --report-version.",
    )
    parser.add_argument(
        "--llm-filename",
        default=None,
        help="LLM CSV filename. Defaults to the canonical file for --report-version.",
    )
    parser.add_argument(
        "--likert-col",
        default="likert_pred",
        help="Likert column to read from the eval CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Root output directory for the report PDFs.",
    )
    parser.add_argument(
        "--report-version",
        default="v3",
        help="Version label to embed in output report filenames.",
    )
    parser.add_argument(
        "--initial-post-validity-sidecar-root",
        action="append",
        default=[DEFAULT_INITIAL_POST_SIDECAR_ROOT],
        help="Sidecar root(s) for Initial/Post Opinion rejudged validity labels.",
    )
    parser.add_argument(
        "--tweet-validity-sidecar-root",
        action="append",
        default=[DEFAULT_TWEET_SIDECAR_ROOT],
        help="Sidecar root(s) for tweet rejudged validity labels.",
    )
    parser.add_argument(
        "--respect-validity-labels",
        dest="respect_validity_labels",
        action="store_true",
        help="Apply inline validity labels and sidecar overrides when available.",
    )
    parser.add_argument(
        "--no-respect-validity-labels",
        dest="respect_validity_labels",
        action="store_false",
        help="Ignore validity labels and sidecars.",
    )
    parser.set_defaults(respect_validity_labels=True)
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress display.",
    )
    parser.add_argument(
        "--skip-v3-additional-group-masks",
        action="store_true",
        help="Do not apply the extra v3 tweet/opinion group masks from masks/groups_with_valid_tweets or masks/groups_with_valid_opinions.",
    )
    parser.add_argument(
        "--write-csv-only",
        action="store_true",
        help="Write the overall-only report CSVs without regenerating the PDFs.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if report_version_family(args.report_version) in {"v3", "v4"} and not args.skip_v3_additional_group_masks:
        enable_v3_complete_group_masks()
    output_report_version = args.report_version
    if args.skip_v3_additional_group_masks and not str(output_report_version).endswith("_unfiltered"):
        output_report_version = f"{output_report_version}_unfiltered"
    if args.human_filename is None:
        args.human_filename = (
            human_csv_filename_for_result(args.report_version)
            if has_canonical_result_namespace(args.report_version)
            else HUMAN_DEFAULT_FILENAME
        )
    if args.llm_filename is None:
        args.llm_filename = (
            llm_csv_filename_for_result(args.report_version)
            if has_canonical_result_namespace(args.report_version)
            else LLM_DEFAULT_FILENAME
        )

    roots = [resolve_existing_or_script_relative(path_text) for path_text in args.data_root]
    output_root = resolve_existing_or_script_relative(args.output_dir or default_output_dir(args.report_version))
    base_models = ordered_base_model_llms_for_report(
        args.report_version,
        read_lines(resolve_existing_or_script_relative(args.models_file)),
    )
    initial_post_sidecar_roots = [
        resolve_existing_or_script_relative(path_text) for path_text in args.initial_post_validity_sidecar_root
    ]
    tweet_sidecar_roots = [
        resolve_existing_or_script_relative(path_text) for path_text in args.tweet_validity_sidecar_root
    ]

    for suite in build_base_report_suites(base_models):
        prefixes = read_lines(suite.prefix_file)
        store, topics = collect_plot_data(
            roots=roots,
            prefixes=prefixes,
            models=suite.models,
            human_filename=args.human_filename,
            llm_filename=args.llm_filename,
            likert_col=args.likert_col,
            no_progress=args.no_progress,
            respect_validity_labels=args.respect_validity_labels,
            initial_post_sidecar_roots=initial_post_sidecar_roots,
            tweet_sidecar_roots=tweet_sidecar_roots,
            progress_label=suite.progress_label,
            apply_v3_additional_group_masks=not args.skip_v3_additional_group_masks,
        )
        render_merged_base_reports(
            output_root=output_root,
            report_version=output_report_version,
            suite=suite,
            store=store,
            topics=topics,
            apply_v3_additional_group_masks=not args.skip_v3_additional_group_masks,
            write_pdf=not args.write_csv_only,
        )

    extra_post_train_roots: dict[str, Path] = {}
    if report_version_family(args.report_version) == "v4":
        breadth_group_full_fit_root = build_post_train_full_fit_eval_root(
            family_key="breadth_group",
            roots=roots,
            human_filename=args.human_filename,
            llm_filename=args.llm_filename,
            likert_col=args.likert_col,
            initial_post_sidecar_roots=initial_post_sidecar_roots,
            tweet_sidecar_roots=tweet_sidecar_roots,
            output_root=output_root,
            report_version=args.report_version,
        )
        if breadth_group_full_fit_root is not None:
            extra_post_train_roots["breadth_group"] = breadth_group_full_fit_root

        breadth_topic_full_fit_root = build_post_train_full_fit_eval_root(
            family_key="breadth_topic",
            roots=roots,
            human_filename=args.human_filename,
            llm_filename=args.llm_filename,
            likert_col=args.likert_col,
            initial_post_sidecar_roots=initial_post_sidecar_roots,
            tweet_sidecar_roots=tweet_sidecar_roots,
            output_root=output_root,
            report_version=args.report_version,
        )
        if breadth_topic_full_fit_root is not None:
            extra_post_train_roots["breadth_topic"] = breadth_topic_full_fit_root

    for suite in build_merged_post_train_suites(args.report_version, base_models):
        family_key = suite.report_prefix.replace("_post_train", "")
        suite_roots = [extra_post_train_roots[family_key], *roots] if family_key in extra_post_train_roots else roots
        train_prefixes = read_lines(suite.train_prefix_file)
        train_store, _ = collect_plot_data(
            roots=suite_roots,
            prefixes=train_prefixes,
            models=suite.models,
            human_filename=args.human_filename,
            llm_filename=args.llm_filename,
            likert_col=args.likert_col,
            no_progress=args.no_progress,
            respect_validity_labels=args.respect_validity_labels,
            initial_post_sidecar_roots=initial_post_sidecar_roots,
            tweet_sidecar_roots=tweet_sidecar_roots,
            progress_label=suite.progress_label_train,
            apply_v3_additional_group_masks=not args.skip_v3_additional_group_masks,
        )
        test_prefixes = read_lines(suite.test_prefix_file)
        test_store, _ = collect_plot_data(
            roots=suite_roots,
            prefixes=test_prefixes,
            models=suite.models,
            human_filename=args.human_filename,
            llm_filename=args.llm_filename,
            likert_col=args.likert_col,
            no_progress=args.no_progress,
            respect_validity_labels=args.respect_validity_labels,
            initial_post_sidecar_roots=initial_post_sidecar_roots,
            tweet_sidecar_roots=tweet_sidecar_roots,
            progress_label=suite.progress_label_test,
            apply_v3_additional_group_masks=not args.skip_v3_additional_group_masks,
        )
        render_merged_post_train_reports(
            output_root=output_root,
            report_version=output_report_version,
            suite=suite,
            train_store=train_store,
            test_store=test_store,
            apply_v3_additional_group_masks=not args.skip_v3_additional_group_masks,
            write_pdf=not args.write_csv_only,
        )

    cleanup_legacy_root_reports(output_root, output_report_version)
    cleanup_legacy_post_train_reports(output_root, output_report_version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())