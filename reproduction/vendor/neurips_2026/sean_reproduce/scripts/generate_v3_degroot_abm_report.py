#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
SEAN_REPRODUCE_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SEAN_REPRODUCE_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CHENGTAO_RADAR_DIR = REPO_ROOT / "neurips_2026" / "chengtao" / "src" / "radar"
if str(CHENGTAO_RADAR_DIR) not in sys.path:
    sys.path.insert(0, str(CHENGTAO_RADAR_DIR))

from neurips_2026.sean_reproduce.base_model_v3_style import CANONICAL_V3_HUMAN_SOURCE_MODEL  # noqa: E402
from neurips_2026.sean_reproduce.v3.masks import (  # noqa: E402
    V3_OPINION_GROUP_MASK_ENV_VAR,
    V3_TWEET_GROUP_MASK_ENV_VAR,
    configured_private_mask_root,
    configured_public_mask_root,
    enable_v3_complete_group_masks,
    wrapped_path_lines,
)

from generate_plots import (  # noqa: E402
    BASE_MODEL_V3_LLM_ORDER,
    HUMAN_DEFAULT_FILENAME,
    LLM_DEFAULT_FILENAME,
    MODEL_COLOR_MAP,
    parse_likert,
    parse_round,
    participant_id,
    read_csv_rows,
    read_lines,
    resolve_file,
    save_csv,
    safe_mean,
    safe_pstdev,
    tqdm,
)
from group_opinion_convergence import (  # noqa: E402
    BREADTH_GOLDEN_MASK,
    DEPTH_GOLDEN_MASK,
    REQUIRED_COMPLETE_PARTICIPANTS,
    build_report_pdf,
    compute_consensus_floor_lists,
    compute_half_sd_reduction_lists,
    compute_profile_delta_lists,
    plot_convergence_rows,
)


DEFAULT_OUTPUT_DIR = (
    SEAN_REPRODUCE_ROOT / "v3" / "abm" / "group_opinion_convergence" / "base_model"
)
DEFAULT_DATA_ROOT = REPO_ROOT / "neurips_2026" / "chengtao" / "result" / "eval_v3" / "human_llm"
DEFAULT_INITIAL_POST_SIDECAR_ROOT = REPO_ROOT / "neurips_2026" / "rejudged_labels" / "eval_v3" / "initial_post"
DEFAULT_TWEET_SIDECAR_ROOT = REPO_ROOT / "neurips_2026" / "rejudged_labels" / "eval_v3" / "tweet"
DEFAULT_PREFIX_FILE = REPO_ROOT / "neurips_2026" / "chengtao" / "inputs" / "masks_train_test" / "golden_group_v0" / "aggregate_universe_mask_golden_v0.txt"
DEFAULT_MODELS_FILE = CHENGTAO_RADAR_DIR / "models.txt"
DEFAULT_ALPHAS = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
DEFAULT_REPORT_VERSION = "v3"
DEFAULT_MASK_NAMESPACE_ROOT = REPO_ROOT / "neurips_2026" / "sean_reproduce" / "v3" / "masks"
DEFAULT_TWEET_MASK_ROOT = DEFAULT_MASK_NAMESPACE_ROOT / "groups_with_valid_tweets"
DEFAULT_OPINION_MASK_ROOT = DEFAULT_MASK_NAMESPACE_ROOT / "groups_with_valid_opinions"
DEFAULT_POST_TRAIN_OUTPUT_DIR = DEFAULT_OUTPUT_DIR.parent / "post_train"
DEFAULT_BOOTSTRAP_REPS = 1000
DEFAULT_BOOTSTRAP_SEED = 1
V4_EXCLUDED_BASE_MODELS = {"Llama-3.1-Tulu-3-8B-SFT"}
FULL_FIT_COLOR = "#9A9A9A"
DEGROOT_COLOR_PALETTE = {
    "DeGroot alpha=0.00": "#2F6B8A",
    "DeGroot alpha=0.25": "#4D8BA8",
    "DeGroot alpha=0.50": "#74A9C2",
    "DeGroot alpha=0.75": "#A26E3D",
    "DeGroot alpha=0.90": "#C58B54",
    "DeGroot alpha=1.00": "#8C4C3A",
}


def report_version_family(report_version: str) -> str:
    return "v4" if str(report_version).strip().startswith("v4") else "v3"


def report_version_namespace(report_version: str) -> str:
    version = str(report_version).strip()
    if version.startswith("v4_b"):
        return "v4_b"
    if version.startswith("v4"):
        return "v4"
    if version.startswith("v3_b"):
        return "v3_b"
    return "v3"


def default_abm_output_dir(report_version: str, *parts: str) -> Path:
    return SEAN_REPRODUCE_ROOT / report_version_namespace(report_version) / "abm" / Path(*parts)


def filter_base_models_for_report_version(report_version: str, requested_models: list[str]) -> list[str]:
    if report_version_family(report_version) != "v4":
        return [model for model in BASE_MODEL_V3_LLM_ORDER if model in requested_models]
    return [
        model
        for model in BASE_MODEL_V3_LLM_ORDER
        if model in requested_models and model not in V4_EXCLUDED_BASE_MODELS
    ]


def alpha_model_name(alpha: float) -> str:
    return f"DeGroot alpha={alpha:.2f}"


FULL_FIT_MODEL_NAME = "DeGroot fit"
CONSENSUS_FLOOR_MODEL = "One-Step Toward First Partner"
HALF_SD_REDUCTION_MODEL = "50% SD Reduction"
REFERENCE_LINE_SPECS = [
    (CONSENSUS_FLOOR_MODEL, "#1F77B4", CONSENSUS_FLOOR_MODEL),
    (HALF_SD_REDUCTION_MODEL, "#7F7F7F", HALF_SD_REDUCTION_MODEL),
]


for _model_name, _color in DEGROOT_COLOR_PALETTE.items():
    MODEL_COLOR_MAP[_model_name] = _color
MODEL_COLOR_MAP[FULL_FIT_MODEL_NAME] = FULL_FIT_COLOR


def collect_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    return fieldnames


def clip_likert(value: float) -> float:
    return max(1.0, min(6.0, value))


def update_from_partner_state(
    current_state: dict[str, float],
    partner_state: dict[str, float],
    partner_map: dict[str, str],
    alpha: float,
    participants: list[str],
) -> dict[str, float]:
    next_state: dict[str, float] = {}
    for participant in participants:
        partner = partner_map[participant]
        next_state[participant] = clip_likert(
            alpha * current_state[participant] + (1.0 - alpha) * partner_state[partner]
        )
    return next_state


def routing_participant_key(row: dict[str, str]) -> str | None:
    return participant_id(row)


def build_degroot_start_states(
    rows: list[dict[str, str]],
    *,
    data_prefix: str,
    likert_col: str,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], dict[str, dict[str, str]]]:
    private_start: dict[str, dict[str, float]] = {}
    public_round1: dict[str, dict[str, float]] = {}
    raw_partner_map: dict[str, dict[str, str]] = {}
    aliases_by_group: dict[str, dict[str, str]] = {}

    for row in rows:
        participant = routing_participant_key(row)
        if not participant:
            continue
        group_id = (row.get("group_id") or "").strip() or data_prefix
        aliases = aliases_by_group.setdefault(group_id, {})
        for key in ("empirica_id", "worker_id", "sender_id"):
            value = (row.get(key) or "").strip()
            if not value:
                continue
            existing = aliases.setdefault(value, participant)
            if existing != participant:
                raise ValueError(
                    f"Ambiguous participant alias in group {group_id}: {value!r} maps to both {existing!r} and {participant!r}"
                )
        event_type = (row.get("event_type") or "").strip()
        stance_text = row.get(likert_col)
        round_text = str(row.get("chat_round_order") or "").strip()

        if event_type == "Initial Opinion":
            stance = None if stance_text is None else float(stance_text) if str(stance_text).strip().replace(".", "", 1).isdigit() else None
            if stance is None:
                stance = parse_likert(stance_text)
            if stance is not None:
                private_start.setdefault(group_id, {})[participant] = stance
            continue

        if event_type != "tweet" or round_text not in {"1", "1.0"}:
            continue

        stance = parse_likert(stance_text)
        if stance is None:
            continue
        public_group = public_round1.setdefault(group_id, {})
        if participant in public_group:
            raise ValueError(
                f"Duplicate valid round-1 tweet for participant {participant!r} in group {group_id}"
            )
        public_group[participant] = stance
        recipient = (row.get("recipient_id") or "").strip()
        if recipient:
            raw_partner_map.setdefault(group_id, {})[participant] = recipient

    first_partner_map: dict[str, dict[str, str]] = {}
    for group_id, participants in raw_partner_map.items():
        aliases = aliases_by_group.get(group_id, {})
        first_partner_map[group_id] = {
            participant: aliases.get(recipient, recipient)
            for participant, recipient in participants.items()
        }

    return private_start, public_round1, first_partner_map


def public_degroot_participants(
    *,
    public_round1: dict[str, dict[str, float]],
    first_partner_map: dict[str, dict[str, str]],
    group_id: str,
) -> list[str]:
    public_group = public_round1.get(group_id, {})
    partner_group = first_partner_map.get(group_id, {})
    participants = sorted(
        participant
        for participant in partner_group
        if participant in public_group and partner_group[participant] in public_group
    )
    if len(participants) != REQUIRED_COMPLETE_PARTICIPANTS:
        return []
    return participants


def private_degroot_participants(
    *,
    group_id: str,
    private_start: dict[str, dict[str, float]],
    public_round1: dict[str, dict[str, float]],
    first_partner_map: dict[str, dict[str, str]],
) -> list[str]:
    private_group = private_start.get(group_id, {})
    public_group = public_round1.get(group_id, {})
    partner_group = first_partner_map.get(group_id, {})
    participants = sorted(
        participant
        for participant in partner_group
        if participant in private_group
        and participant in public_group
        and partner_group[participant] in public_group
    )
    if len(participants) != REQUIRED_COMPLETE_PARTICIPANTS:
        return []
    return participants


def simulate_degroot_trajectories(
    rows: list[dict[str, str]],
    *,
    data_prefix: str,
    likert_col: str,
    alpha_public: float,
    alpha_private: float | None = None,
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    if alpha_private is None:
        alpha_private = alpha_public
    private_start, public_round1_by_group, first_partner_map = build_degroot_start_states(
        rows,
        data_prefix=data_prefix,
        likert_col=likert_col,
    )

    trajectories: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    group_ids = sorted(
        set(private_start.keys())
        | set(public_round1_by_group.keys())
        | set(first_partner_map.keys())
    )

    for group_id in group_ids:
        public_participants = public_degroot_participants(
            public_round1=public_round1_by_group,
            first_partner_map=first_partner_map,
            group_id=group_id,
        )
        private_participants = private_degroot_participants(
            group_id=group_id,
            private_start=private_start,
            public_round1=public_round1_by_group,
            first_partner_map=first_partner_map,
        )
        if not public_participants and not private_participants:
            continue

        partner_map = first_partner_map[group_id]
        public_states: dict[str, dict[str, float]] = {}
        public_round1: dict[str, float] = {}
        public_round2: dict[str, float] = {}
        public_round3: dict[str, float] = {}
        if public_participants:
            public_round1 = {
                participant: public_round1_by_group[group_id][participant]
                for participant in public_participants
            }
            public_round2 = update_from_partner_state(
                public_round1,
                public_round1,
                partner_map,
                alpha_public,
                public_participants,
            )
            public_round3 = update_from_partner_state(
                public_round2,
                public_round2,
                partner_map,
                alpha_public,
                public_participants,
            )
            public_states = {
                participant: {
                    "public_round1": public_round1[participant],
                    "public_round2": public_round2[participant],
                    "public_round3": public_round3[participant],
                }
                for participant in public_participants
            }

        private_states: dict[str, dict[str, float]] = {}
        if private_participants:
            private_initial = {
                participant: private_start[group_id][participant]
                for participant in private_participants
            }
            private_after_msg1 = update_from_partner_state(
                private_initial,
                public_round1,
                partner_map,
                alpha_private,
                private_participants,
            )
            private_after_msg2 = update_from_partner_state(
                private_after_msg1,
                public_round2,
                partner_map,
                alpha_private,
                private_participants,
            )
            private_after_msg3 = update_from_partner_state(
                private_after_msg2,
                public_round3,
                partner_map,
                alpha_private,
                private_participants,
            )
            private_states = {
                participant: {
                    "private_initial": private_initial[participant],
                    "private_final": private_after_msg3[participant],
                }
                for participant in private_participants
            }

        trajectories[group_id] = {
            "public": public_states,
            "private": private_states,
        }

    return trajectories


def build_observed_human_targets(
    rows: list[dict[str, str]],
    *,
    data_prefix: str,
    likert_col: str,
) -> tuple[dict[str, dict[str, dict[int, float]]], dict[str, dict[str, float]]]:
    public_round_lists: dict[str, dict[str, dict[int, list[float]]]] = {}
    private_final: dict[str, dict[str, float]] = {}

    for row in rows:
        participant = routing_participant_key(row)
        if not participant:
            continue
        stance = parse_likert(row.get(likert_col))
        if stance is None:
            continue
        group_id = (row.get("group_id") or "").strip() or data_prefix
        event_type = (row.get("event_type") or "").strip()

        if event_type == "Post Opinion":
            private_final.setdefault(group_id, {})[participant] = stance
            continue

        if event_type != "tweet":
            continue
        round_idx = parse_round(row.get("chat_round_order"))
        if round_idx not in {2, 3}:
            continue
        public_round_lists.setdefault(group_id, {}).setdefault(participant, {}).setdefault(round_idx, []).append(stance)

    public_round_targets: dict[str, dict[str, dict[int, float]]] = {}
    for group_id, participants in public_round_lists.items():
        public_round_targets[group_id] = {}
        for participant, rounds in participants.items():
            public_round_targets[group_id][participant] = {
                round_idx: safe_mean(values)
                for round_idx, values in rounds.items()
                if values
            }

    return public_round_targets, private_final


def compute_degroot_delta_lists(
    rows: list[dict[str, str]],
    *,
    data_prefix: str,
    likert_col: str,
    alpha_public: float,
    alpha_private: float | None = None,
) -> tuple[list[float], list[float]]:
    try:
        trajectories = simulate_degroot_trajectories(
            rows,
            data_prefix=data_prefix,
            likert_col=likert_col,
            alpha_public=alpha_public,
            alpha_private=alpha_private,
        )
    except ValueError:
        return [], []
    public_deltas: list[float] = []
    private_deltas: list[float] = []
    for group_states in trajectories.values():
        public_states = group_states["public"]
        if public_states:
            public_participants = sorted(public_states.keys())
            round1_sd = safe_pstdev([public_states[participant]["public_round1"] for participant in public_participants])
            round3_sd = safe_pstdev([public_states[participant]["public_round3"] for participant in public_participants])
            if not math.isnan(round1_sd) and not math.isnan(round3_sd):
                public_deltas.append(round3_sd - round1_sd)

        private_states = group_states["private"]
        if private_states:
            private_participants = sorted(private_states.keys())
            initial_sd = safe_pstdev([private_states[participant]["private_initial"] for participant in private_participants])
            final_sd = safe_pstdev([private_states[participant]["private_final"] for participant in private_participants])
            if not math.isnan(initial_sd) and not math.isnan(final_sd):
                private_deltas.append(final_sd - initial_sd)

    return public_deltas, private_deltas


def compute_degroot_prediction_error_lists(
    rows: list[dict[str, str]],
    *,
    data_prefix: str,
    likert_col: str,
    alpha_public: float,
    alpha_private: float | None = None,
) -> tuple[list[float], list[float], list[float]]:
    try:
        trajectories = simulate_degroot_trajectories(
            rows,
            data_prefix=data_prefix,
            likert_col=likert_col,
            alpha_public=alpha_public,
            alpha_private=alpha_private,
        )
    except ValueError:
        return [], [], []
    observed_public, observed_private_final = build_observed_human_targets(
        rows,
        data_prefix=data_prefix,
        likert_col=likert_col,
    )

    public_round2_errors: list[float] = []
    public_round3_errors: list[float] = []
    private_final_errors: list[float] = []
    for group_id, group_states in trajectories.items():
        observed_public_group = observed_public.get(group_id, {})
        observed_private_group = observed_private_final.get(group_id, {})
        for participant, states in group_states["public"].items():
            observed_rounds = observed_public_group.get(participant, {})
            if 2 in observed_rounds:
                public_round2_errors.append((states["public_round2"] - observed_rounds[2]) ** 2)
            if 3 in observed_rounds:
                public_round3_errors.append((states["public_round3"] - observed_rounds[3]) ** 2)
        for participant, states in group_states["private"].items():
            if participant in observed_private_group:
                private_final_errors.append((states["private_final"] - observed_private_group[participant]) ** 2)

    return public_round2_errors, public_round3_errors, private_final_errors


def bootstrap_standard_error(samples: list[float]) -> float:
    if len(samples) <= 1:
        return math.nan
    mean_value = sum(samples) / len(samples)
    variance = sum((sample - mean_value) ** 2 for sample in samples) / (len(samples) - 1)
    return math.sqrt(variance)


def resolve_human_path(
    *,
    roots: list[Path],
    data_prefix: str,
    models: list[str],
    human_filename: str,
) -> Path | None:
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


def collect_abm_rows_with_bootstrap(
    *,
    roots: list[Path],
    prefixes: list[str],
    models: list[str],
    alphas: list[float] | None,
    human_filename: str,
    llm_filename: str,
    likert_col: str,
    no_progress: bool,
    initial_post_sidecar_roots: list[Path],
    tweet_sidecar_roots: list[Path],
    bootstrap_reps: int,
    bootstrap_seed: int,
    abm_specs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    alpha_values = list(alphas or [])
    normalized_specs: list[dict[str, Any]] = []
    if abm_specs is None:
        normalized_specs = [
            {
                "model": alpha_model_name(alpha),
                "alpha_public": alpha,
                "alpha_private": alpha,
            }
            for alpha in alpha_values
        ]
    else:
        for spec in abm_specs:
            normalized_specs.append(
                {
                    "model": str(spec["model"]),
                    "alpha_public": float(spec["alpha_public"]),
                    "alpha_private": float(spec.get("alpha_private", spec["alpha_public"])),
                }
            )

    degroot_models = [spec["model"] for spec in normalized_specs]
    reference_models = [model_name for model_name, _, _ in REFERENCE_LINE_SPECS]
    model_names = ["human", *models, *reference_models, *degroot_models]
    prefix_public: dict[str, dict[str, list[float]]] = {
        prefix: {model_name: [] for model_name in model_names}
        for prefix in prefixes
    }
    prefix_private: dict[str, dict[str, list[float]]] = {
        prefix: {model_name: [] for model_name in model_names}
        for prefix in prefixes
    }

    prefix_iter: Any = prefixes
    if not no_progress and tqdm is not None:
        prefix_iter = tqdm(prefixes, total=len(prefixes), desc="ABM convergence", unit="prefix")
    elif not no_progress and tqdm is None:
        print("tqdm not installed; running without progress bar.")

    for data_prefix in prefix_iter:
        human_path = resolve_human_path(
            roots=roots,
            data_prefix=data_prefix,
            models=models,
            human_filename=human_filename,
        )
        if human_path is None:
            continue
        human_rows = read_csv_rows(
            human_path,
            respect_validity_labels=True,
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

        for spec in normalized_specs:
            model_name = spec["model"]
            abm_public_delta, abm_private_delta = compute_degroot_delta_lists(
                human_rows,
                data_prefix=data_prefix,
                likert_col=likert_col,
                alpha_public=spec["alpha_public"],
                alpha_private=spec["alpha_private"],
            )
            prefix_public[data_prefix][model_name].extend(abm_public_delta)
            prefix_private[data_prefix][model_name].extend(abm_private_delta)

        for model in models:
            llm_path = resolve_file(roots, data_prefix, model, llm_filename, allow_without_model=False)
            if llm_path is None:
                continue
            llm_rows = read_csv_rows(
                llm_path,
                respect_validity_labels=True,
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
            prefix_public[data_prefix][model].extend(llm_public_delta)
            prefix_private[data_prefix][model].extend(llm_private_delta)

    rows: list[dict[str, Any]] = []
    bootstrap_public: dict[str, list[float]] = {model_name: [] for model_name in model_names}
    bootstrap_private: dict[str, list[float]] = {model_name: [] for model_name in model_names}
    available_prefixes = [
        prefix
        for prefix in prefixes
        if any(prefix_public[prefix][model_name] or prefix_private[prefix][model_name] for model_name in model_names)
    ]

    if available_prefixes and bootstrap_reps > 0:
        rng = random.Random(bootstrap_seed)
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


def output_paths(output_dir: Path, report_version: str) -> tuple[Path, Path, Path]:
    csv_path = output_dir / "csv" / f"group_opinion_convergence_degroot_{report_version}.csv"
    png_path = output_dir / "img" / f"group_opinion_convergence_degroot_{report_version}.png"
    report_path = output_dir / f"report_group_opinion_convergence_degroot_{report_version}.pdf"
    return csv_path, png_path, report_path


def load_post_train_fit_results(post_train_output_dir: Path) -> list[str]:
    pair_order = (
        "breadth_topic",
        "breadth_group",
        "depth_topic",
        "depth_group",
    )
    csv_dir = post_train_output_dir / "csv"
    lines: list[str] = []
    for pair_key in pair_order:
        csv_path = csv_dir / f"group_opinion_convergence_models_vs_all_{pair_key}_train_degroot_fit_{DEFAULT_REPORT_VERSION}.csv"
        if not csv_path.exists():
            continue
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fit_row = next((row for row in reader if (row.get("fit_pair_key") or "") == pair_key), None)
        if fit_row is None:
            continue
        alpha_public = fit_row.get("fit_alpha_public") or ""
        alpha_private = fit_row.get("fit_alpha_private") or ""
        combined_mse = fit_row.get("fit_combined_mse") or ""
        if not alpha_public or not alpha_private:
            continue
        summary = (
            f"- {pair_key.replace('_', ' ')}: "
            f"alpha_public={float(alpha_public):.3f}, "
            f"alpha_private={float(alpha_private):.3f}"
        )
        if combined_mse:
            summary += f", combined MSE={float(combined_mse):.6f}"
        lines.append(summary)
    return lines


def load_human_fit_rows(
    *,
    roots: list[Path],
    prefixes: list[str],
    human_filename: str,
    likert_col: str,
    initial_post_sidecar_roots: list[Path],
    tweet_sidecar_roots: list[Path],
) -> dict[str, list[dict[str, str]]]:
    rows_by_prefix: dict[str, list[dict[str, str]]] = {}
    for data_prefix in prefixes:
        human_path = resolve_human_path(
            roots=roots,
            data_prefix=data_prefix,
            models=[],
            human_filename=human_filename,
        )
        if human_path is None:
            continue
        rows_by_prefix[data_prefix] = read_csv_rows(
            human_path,
            respect_validity_labels=True,
            likert_col=likert_col,
            data_roots=roots,
            initial_post_sidecar_roots=initial_post_sidecar_roots,
            tweet_sidecar_roots=tweet_sidecar_roots,
        )
    return rows_by_prefix


def collect_fit_error_lists(
    *,
    rows_by_prefix: dict[str, list[dict[str, str]]],
    likert_col: str,
    alpha_public: float,
    alpha_private: float,
) -> dict[str, list[float]]:
    public_tweet2_errors: list[float] = []
    public_tweet3_errors: list[float] = []
    private_final_errors: list[float] = []
    for data_prefix, human_rows in rows_by_prefix.items():
        tweet2_errors, tweet3_errors, private_errors = compute_degroot_prediction_error_lists(
            human_rows,
            data_prefix=data_prefix,
            likert_col=likert_col,
            alpha_public=alpha_public,
            alpha_private=alpha_private,
        )
        public_tweet2_errors.extend(tweet2_errors)
        public_tweet3_errors.extend(tweet3_errors)
        private_final_errors.extend(private_errors)
    return {
        "public_tweet2_errors": public_tweet2_errors,
        "public_tweet3_errors": public_tweet3_errors,
        "private_final_errors": private_final_errors,
    }


def pooled_mean(values: list[float]) -> float:
    return float(safe_mean(values))


def summarize_fit_errors(error_lists: dict[str, list[float]]) -> dict[str, float]:
    public_tweet2_errors = error_lists["public_tweet2_errors"]
    public_tweet3_errors = error_lists["public_tweet3_errors"]
    private_final_errors = error_lists["private_final_errors"]
    public_errors = [*public_tweet2_errors, *public_tweet3_errors]
    return {
        "public_tweet2_mse": pooled_mean(public_tweet2_errors),
        "public_tweet3_mse": pooled_mean(public_tweet3_errors),
        "public_pooled_mse": pooled_mean(public_errors),
        "private_final_mse": pooled_mean(private_final_errors),
        "combined_mse": pooled_mean([*public_errors, *private_final_errors]),
        "public_count": float(len(public_errors)),
        "private_count": float(len(private_final_errors)),
    }


def is_better_minimum(
    score: float,
    point: float,
    best_score: float,
    best_point: float,
    *,
    tie_tol: float = 1e-12,
) -> bool:
    if score < best_score - tie_tol:
        return True
    if abs(score - best_score) <= tie_tol and point < best_point:
        return True
    return False


def bounded_minimize_1d(
    objective: Any,
    *,
    lower: float = 0.0,
    upper: float = 1.0,
    coarse_steps: int = 20,
    tol: float = 1e-4,
    max_iter: int = 64,
    max_candidates: int = 4,
) -> tuple[float, float]:
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    grid = [lower + (upper - lower) * (idx / coarse_steps) for idx in range(coarse_steps + 1)]
    grid_scores = [(float(objective(point)), point) for point in grid]
    best_score, best_point = min(grid_scores, key=lambda item: item[0])
    candidate_indices: set[int] = {0, coarse_steps}
    for idx in range(1, coarse_steps):
        current_score = grid_scores[idx][0]
        if current_score <= grid_scores[idx - 1][0] and current_score <= grid_scores[idx + 1][0]:
            candidate_indices.add(idx)
    candidate_indices.update(
        idx for idx, _item in sorted(enumerate(grid_scores), key=lambda item: (item[1][0], item[1][1]))[:max_candidates]
    )
    for candidate_index in sorted(candidate_indices, key=lambda idx: (grid_scores[idx][0], grid_scores[idx][1]))[:max_candidates]:
        left = grid[max(0, candidate_index - 1)]
        right = grid[min(coarse_steps, candidate_index + 1)]
        if left == right:
            continue
        c = right - (right - left) / phi
        d = left + (right - left) / phi
        fc = float(objective(c))
        fd = float(objective(d))
        if is_better_minimum(fc, c, best_score, best_point):
            best_point, best_score = c, fc
        if is_better_minimum(fd, d, best_score, best_point):
            best_point, best_score = d, fd
        for _ in range(max_iter):
            if abs(right - left) <= tol:
                break
            if fc <= fd:
                right = d
                d = c
                fd = fc
                c = right - (right - left) / phi
                fc = float(objective(c))
                if is_better_minimum(fc, c, best_score, best_point):
                    best_point, best_score = c, fc
            else:
                left = c
                c = d
                fc = fd
                d = left + (right - left) / phi
                fd = float(objective(d))
                if is_better_minimum(fd, d, best_score, best_point):
                    best_point, best_score = d, fd
        for point in (left, right, (left + right) / 2.0, lower, upper):
            score = float(objective(point))
            if is_better_minimum(score, point, best_score, best_point):
                best_point, best_score = point, score
    return best_point, best_score


def fit_best_alphas_from_human_rows(
    *,
    rows_by_prefix: dict[str, list[dict[str, str]]],
    likert_col: str,
) -> dict[str, float]:
    """Fit the standard DeGroot public/private alphas from loaded human rows.

    The objective and optimizer are shared by the v3/v4 paper reports and
    newer result bundles whose human CSVs use a different directory layout.
    """

    def public_objective(alpha_public: float) -> float:
        return summarize_fit_errors(
            collect_fit_error_lists(
                rows_by_prefix=rows_by_prefix,
                likert_col=likert_col,
                alpha_public=alpha_public,
                alpha_private=1.0,
            )
        )["public_pooled_mse"]

    alpha_public, public_score = bounded_minimize_1d(public_objective)

    def private_objective(alpha_private: float) -> float:
        return summarize_fit_errors(
            collect_fit_error_lists(
                rows_by_prefix=rows_by_prefix,
                likert_col=likert_col,
                alpha_public=alpha_public,
                alpha_private=alpha_private,
            )
        )["private_final_mse"]

    alpha_private, private_score = bounded_minimize_1d(private_objective)
    summary = summarize_fit_errors(
        collect_fit_error_lists(
            rows_by_prefix=rows_by_prefix,
            likert_col=likert_col,
            alpha_public=alpha_public,
            alpha_private=alpha_private,
        )
    )
    return {
        "alpha_public": alpha_public,
        "alpha_private": alpha_private,
        "public_objective": public_score,
        "private_objective": private_score,
        **summary,
    }


def fit_best_alphas_for_prefixes(
    *,
    roots: list[Path],
    prefixes: list[str],
    human_filename: str,
    likert_col: str,
    initial_post_sidecar_roots: list[Path],
    tweet_sidecar_roots: list[Path],
) -> dict[str, float]:
    """Load canonical human files, then apply the standard DeGroot fit."""
    rows_by_prefix = load_human_fit_rows(
        roots=roots,
        prefixes=prefixes,
        human_filename=human_filename,
        likert_col=likert_col,
        initial_post_sidecar_roots=initial_post_sidecar_roots,
        tweet_sidecar_roots=tweet_sidecar_roots,
    )
    return fit_best_alphas_from_human_rows(
        rows_by_prefix=rows_by_prefix,
        likert_col=likert_col,
    )


def cover_body(alphas: list[float], models: list[str], full_fit_lines: list[str]) -> str:
    alpha_text = ", ".join(f"{alpha:.2f}" for alpha in alphas)
    lines = [
        "Pages: aggregate alpha grid, full breadth fit, full depth fit.",
        f"Alpha grid: {alpha_text}",
        f"Base models in comparison set: {len(models)}",
        "Masks: v3 complete-group tweet/opinion masks enabled.",
        "Full-set fitted alphas:",
    ]
    if full_fit_lines:
        lines.extend(full_fit_lines)
    else:
        lines.append("- Full breadth/depth fitted alpha results not found.")
    return "\n".join(lines)


def full_fit_cover_lines(fit_results: dict[str, dict[str, float]]) -> list[str]:
    lines: list[str] = []
    for key in ("aggregate", "breadth", "depth"):
        result = fit_results.get(key)
        if result is None:
            continue
        lines.append(
            f"- {key}: alpha_public={result['alpha_public']:.3f}, "
            f"alpha_private={result['alpha_private']:.3f}, "
            f"combined MSE={result['combined_mse']:.6f}"
        )
    return lines


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an aggregate DeGroot ABM baseline report for group opinion convergence."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--data-prefix-file", type=Path, default=DEFAULT_PREFIX_FILE)
    parser.add_argument("--models-file", type=Path, default=DEFAULT_MODELS_FILE)
    parser.add_argument("--human-filename", default=HUMAN_DEFAULT_FILENAME)
    parser.add_argument("--llm-filename", default=LLM_DEFAULT_FILENAME)
    parser.add_argument("--likert-col", default="likert_pred")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--alpha",
        dest="alphas",
        action="append",
        type=float,
        default=None,
        help="Repeat to override the default alpha grid.",
    )
    parser.add_argument(
        "--initial-post-validity-sidecar-root",
        action="append",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--tweet-validity-sidecar-root",
        action="append",
        type=Path,
        default=None,
    )
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--report-version", default=DEFAULT_REPORT_VERSION)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.alphas is None:
        args.alphas = list(DEFAULT_ALPHAS)
    if args.initial_post_validity_sidecar_root is None:
        args.initial_post_validity_sidecar_root = [DEFAULT_INITIAL_POST_SIDECAR_ROOT]
    if args.tweet_validity_sidecar_root is None:
        args.tweet_validity_sidecar_root = [DEFAULT_TWEET_SIDECAR_ROOT]
    enable_v3_complete_group_masks(DEFAULT_MASK_NAMESPACE_ROOT)
    os.environ[V3_TWEET_GROUP_MASK_ENV_VAR] = str(DEFAULT_TWEET_MASK_ROOT)
    os.environ[V3_OPINION_GROUP_MASK_ENV_VAR] = str(DEFAULT_OPINION_MASK_ROOT)

    roots = [args.data_root.resolve()]
    prefixes = read_lines(args.data_prefix_file.resolve())
    breadth_prefixes = read_lines(BREADTH_GOLDEN_MASK)
    depth_prefixes = read_lines(DEPTH_GOLDEN_MASK)
    requested_models = read_lines(args.models_file.resolve())
    model_order = filter_base_models_for_report_version(args.report_version, requested_models)
    for model in requested_models:
        if model in BASE_MODEL_V3_LLM_ORDER or model in model_order:
            continue
        model_order.append(model)
    initial_post_sidecar_roots = [path.resolve() for path in args.initial_post_validity_sidecar_root]
    tweet_sidecar_roots = [path.resolve() for path in args.tweet_validity_sidecar_root]

    rows = collect_abm_rows_with_bootstrap(
        roots=roots,
        prefixes=prefixes,
        models=model_order,
        alphas=list(args.alphas),
        human_filename=args.human_filename,
        llm_filename=args.llm_filename,
        likert_col=args.likert_col,
        no_progress=args.no_progress,
        initial_post_sidecar_roots=initial_post_sidecar_roots,
        tweet_sidecar_roots=tweet_sidecar_roots,
        bootstrap_reps=args.bootstrap_reps,
        bootstrap_seed=args.bootstrap_seed,
    )
    full_fit_results = {
        "aggregate": fit_best_alphas_for_prefixes(
            roots=roots,
            prefixes=prefixes,
            human_filename=args.human_filename,
            likert_col=args.likert_col,
            initial_post_sidecar_roots=initial_post_sidecar_roots,
            tweet_sidecar_roots=tweet_sidecar_roots,
        ),
        "breadth": fit_best_alphas_for_prefixes(
            roots=roots,
            prefixes=breadth_prefixes,
            human_filename=args.human_filename,
            likert_col=args.likert_col,
            initial_post_sidecar_roots=initial_post_sidecar_roots,
            tweet_sidecar_roots=tweet_sidecar_roots,
        ),
        "depth": fit_best_alphas_for_prefixes(
            roots=roots,
            prefixes=depth_prefixes,
            human_filename=args.human_filename,
            likert_col=args.likert_col,
            initial_post_sidecar_roots=initial_post_sidecar_roots,
            tweet_sidecar_roots=tweet_sidecar_roots,
        ),
    }
    aggregate_fit_rows = collect_abm_rows_with_bootstrap(
        roots=roots,
        prefixes=prefixes,
        models=[],
        alphas=None,
        human_filename=args.human_filename,
        llm_filename=args.llm_filename,
        likert_col=args.likert_col,
        no_progress=args.no_progress,
        initial_post_sidecar_roots=initial_post_sidecar_roots,
        tweet_sidecar_roots=tweet_sidecar_roots,
        bootstrap_reps=args.bootstrap_reps,
        bootstrap_seed=args.bootstrap_seed,
        abm_specs=[
            {
                "model": FULL_FIT_MODEL_NAME,
                "alpha_public": full_fit_results["aggregate"]["alpha_public"],
                "alpha_private": full_fit_results["aggregate"]["alpha_private"],
            }
        ],
    )
    aggregate_fit_row = next((row for row in aggregate_fit_rows if row.get("model") == FULL_FIT_MODEL_NAME), None)
    if aggregate_fit_row is not None:
        rows.append(aggregate_fit_row)
    breadth_rows = collect_abm_rows_with_bootstrap(
        roots=roots,
        prefixes=breadth_prefixes,
        models=model_order,
        alphas=None,
        human_filename=args.human_filename,
        llm_filename=args.llm_filename,
        likert_col=args.likert_col,
        no_progress=args.no_progress,
        initial_post_sidecar_roots=initial_post_sidecar_roots,
        tweet_sidecar_roots=tweet_sidecar_roots,
        bootstrap_reps=args.bootstrap_reps,
        bootstrap_seed=args.bootstrap_seed,
        abm_specs=[
            {
                "model": FULL_FIT_MODEL_NAME,
                "alpha_public": full_fit_results["breadth"]["alpha_public"],
                "alpha_private": full_fit_results["breadth"]["alpha_private"],
            }
        ],
    )
    depth_rows = collect_abm_rows_with_bootstrap(
        roots=roots,
        prefixes=depth_prefixes,
        models=model_order,
        alphas=None,
        human_filename=args.human_filename,
        llm_filename=args.llm_filename,
        likert_col=args.likert_col,
        no_progress=args.no_progress,
        initial_post_sidecar_roots=initial_post_sidecar_roots,
        tweet_sidecar_roots=tweet_sidecar_roots,
        bootstrap_reps=args.bootstrap_reps,
        bootstrap_seed=args.bootstrap_seed,
        abm_specs=[
            {
                "model": FULL_FIT_MODEL_NAME,
                "alpha_public": full_fit_results["depth"]["alpha_public"],
                "alpha_private": full_fit_results["depth"]["alpha_private"],
            }
        ],
    )

    output_dir = args.output_dir.resolve() if args.output_dir is not None else default_abm_output_dir(args.report_version, "group_opinion_convergence", "base_model").resolve()
    csv_path, png_path, report_path = output_paths(output_dir, args.report_version)
    breadth_csv_path = csv_path.parent / f"group_opinion_convergence_degroot_breadth_{args.report_version}.csv"
    breadth_png_path = png_path.parent / f"group_opinion_convergence_degroot_breadth_{args.report_version}.png"
    depth_csv_path = csv_path.parent / f"group_opinion_convergence_degroot_depth_{args.report_version}.csv"
    depth_png_path = png_path.parent / f"group_opinion_convergence_degroot_depth_{args.report_version}.png"
    if not args.report_only:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        save_csv(csv_path, rows, collect_fieldnames(rows))
        save_csv(breadth_csv_path, breadth_rows, collect_fieldnames(breadth_rows))
        save_csv(depth_csv_path, depth_rows, collect_fieldnames(depth_rows))
        print(f"Wrote: {csv_path}")
        print(f"Wrote: {breadth_csv_path}")
        print(f"Wrote: {depth_csv_path}")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plot_order = ["human", *model_order, *[alpha_model_name(alpha) for alpha in args.alphas], FULL_FIT_MODEL_NAME]
    plot_convergence_rows(
        rows,
        out_png=png_path,
        title=(
            "Aggregate Group Opinion Convergence: Base Models vs DeGroot ABM "
            f"(fit alpha_public={full_fit_results['aggregate']['alpha_public']:.3f}, "
            f"alpha_private={full_fit_results['aggregate']['alpha_private']:.3f})"
        ),
        model_order=plot_order,
        write_png=not args.report_only,
    )
    fit_plot_order = ["human", *model_order, FULL_FIT_MODEL_NAME]
    plot_convergence_rows(
        breadth_rows,
        out_png=breadth_png_path,
        title=(
            "Breadth Group Opinion Convergence: Base Models vs DeGroot fit "
            f"(alpha_public={full_fit_results['breadth']['alpha_public']:.3f}, "
            f"alpha_private={full_fit_results['breadth']['alpha_private']:.3f})"
        ),
        model_order=fit_plot_order,
        write_png=not args.report_only,
    )
    plot_convergence_rows(
        depth_rows,
        out_png=depth_png_path,
        title=(
            "Depth Group Opinion Convergence: Base Models vs DeGroot fit "
            f"(alpha_public={full_fit_results['depth']['alpha_public']:.3f}, "
            f"alpha_private={full_fit_results['depth']['alpha_private']:.3f})"
        ),
        model_order=fit_plot_order,
        write_png=not args.report_only,
    )
    if not args.report_only:
        print(f"Wrote: {png_path}")
        print(f"Wrote: {breadth_png_path}")
        print(f"Wrote: {depth_png_path}")
    print(f"Wrote: {png_path.with_suffix('.pdf')}")
    print(f"Wrote: {breadth_png_path.with_suffix('.pdf')}")
    print(f"Wrote: {depth_png_path.with_suffix('.pdf')}")

    report_result = build_report_pdf(
        output_dir,
        [
            ("Aggregate Group Opinion Convergence", png_path),
            ("Breadth Group Opinion Convergence", breadth_png_path),
            ("Depth Group Opinion Convergence", depth_png_path),
        ],
        report_name=report_path.name,
        cover_title=f"DeGroot ABM Convergence ({args.report_version})",
        cover_body=cover_body(
            list(args.alphas),
            model_order,
            full_fit_cover_lines(full_fit_results),
        ),
    )
    if report_result is not None:
        print(f"Wrote: {report_result}")


if __name__ == "__main__":
    main()
