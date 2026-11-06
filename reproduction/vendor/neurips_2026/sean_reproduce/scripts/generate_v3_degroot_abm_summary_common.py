#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
SEAN_REPRODUCE_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SEAN_REPRODUCE_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

CHENGTAO_RADAR_DIR = REPO_ROOT / "neurips_2026" / "chengtao" / "src" / "radar"
if str(CHENGTAO_RADAR_DIR) not in sys.path:
    sys.path.insert(0, str(CHENGTAO_RADAR_DIR))

from neurips_2026.sean_reproduce.v3.masks import (  # noqa: E402
    V3_OPINION_GROUP_MASK_ENV_VAR,
    V3_TWEET_GROUP_MASK_ENV_VAR,
    enable_v3_complete_group_masks,
)

from generate_plots import (  # noqa: E402
    DEFAULT_UTTERANCE_EVENT_TYPES,
    add_v2_bootstrap_cis_to_raw_rows,
    build_model_aggregates,
    build_raw_rows_from_aggregates,
    collect_fieldnames,
    compute_prefix_model_from_rows,
    is_depth_topic,
    merge_prefix_result_into_aggregate,
    parse_round,
    participant_id,
    read_csv_rows,
)

from generate_v3_degroot_abm_report import (  # noqa: E402
    DEFAULT_BOOTSTRAP_REPS,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_DATA_ROOT,
    DEFAULT_INITIAL_POST_SIDECAR_ROOT,
    DEFAULT_MASK_NAMESPACE_ROOT,
    DEFAULT_OPINION_MASK_ROOT,
    DEFAULT_REPORT_VERSION,
    DEFAULT_TWEET_MASK_ROOT,
    DEFAULT_TWEET_SIDECAR_ROOT,
    FULL_FIT_MODEL_NAME,
    alpha_model_name,
    simulate_degroot_trajectories,
)


def configure_v3_abm_environment() -> None:
    enable_v3_complete_group_masks(DEFAULT_MASK_NAMESPACE_ROOT)
    os.environ[V3_TWEET_GROUP_MASK_ENV_VAR] = str(DEFAULT_TWEET_MASK_ROOT)
    os.environ[V3_OPINION_GROUP_MASK_ENV_VAR] = str(DEFAULT_OPINION_MASK_ROOT)


def build_degroot_proxy_rows(
    human_rows: list[dict[str, str]],
    *,
    data_prefix: str,
    likert_col: str,
    alpha_public: float,
    alpha_private: float,
) -> list[dict[str, str]]:
    trajectories = simulate_degroot_trajectories(
        human_rows,
        data_prefix=data_prefix,
        likert_col=likert_col,
        alpha_public=alpha_public,
        alpha_private=alpha_private,
    )

    proxy_rows: list[dict[str, str]] = []
    for row in human_rows:
        proxy_row = dict(row)
        proxy_row[likert_col] = ""

        participant = participant_id(row)
        if not participant:
            proxy_rows.append(proxy_row)
            continue

        group_id = (row.get("group_id") or "").strip() or data_prefix
        group_states = trajectories.get(group_id, {})
        public_states = group_states.get("public", {})
        private_states = group_states.get("private", {})
        event_type = (row.get("event_type") or "").strip()
        round_idx = parse_round(row.get("chat_round_order"))

        if event_type == "tweet" and round_idx in {1, 2, 3}:
            state_name = {1: "public_round1", 2: "public_round2", 3: "public_round3"}[round_idx]
            state_value = public_states.get(participant, {}).get(state_name)
            if state_value is not None:
                proxy_row[likert_col] = str(state_value)
        elif event_type == "Initial Opinion":
            state_value = private_states.get(participant, {}).get("private_initial")
            if state_value is not None:
                proxy_row[likert_col] = str(state_value)
        elif event_type == "Post Opinion":
            state_value = private_states.get(participant, {}).get("private_final")
            if state_value is not None:
                proxy_row[likert_col] = str(state_value)

        proxy_rows.append(proxy_row)

    return proxy_rows


def load_human_rows_by_prefix(
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
        human_path = None
        for root in roots:
            candidate = root / data_prefix / human_filename
            if candidate.exists():
                human_path = candidate
                break
        if human_path is None:
            for root in roots:
                for child in root.glob(f"{data_prefix}/*/{human_filename}"):
                    human_path = child
                    break
                if human_path is not None:
                    break
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


def collect_abm_raw_rows_with_bootstrap(
    *,
    prefixes: list[str],
    rows_by_prefix: dict[str, list[dict[str, str]]],
    likert_col: str,
    abm_specs: list[dict[str, Any]],
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model_names = [str(spec["model"]) for spec in abm_specs]
    aggregates = build_model_aggregates(model_names)
    prefix_results_by_prefix: dict[str, dict[str, Any]] = {}
    status_rows: list[dict[str, Any]] = []

    for data_prefix in prefixes:
        human_rows = rows_by_prefix.get(data_prefix)
        if human_rows is None:
            continue
        prefix_results_by_prefix[data_prefix] = {}
        prefix_error: str | None = None
        for spec in abm_specs:
            model_name = str(spec["model"])
            alpha_public = float(spec["alpha_public"])
            alpha_private = float(spec.get("alpha_private", alpha_public))
            try:
                proxy_rows = build_degroot_proxy_rows(
                    human_rows,
                    data_prefix=data_prefix,
                    likert_col=likert_col,
                    alpha_public=alpha_public,
                    alpha_private=alpha_private,
                )
            except ValueError as exc:
                prefix_error = str(exc)
                status_rows.append(
                    {
                        "data_prefix": data_prefix,
                        "model": model_name,
                        "status": "degroot_skip",
                        "message": prefix_error,
                        "human_path": f"human::{data_prefix}",
                        "llm_path": "",
                        "alignment_mode": "",
                        "key_match_rate": "",
                        "aligned_pairs": "",
                    }
                )
                break
            result = compute_prefix_model_from_rows(
                data_prefix=data_prefix,
                model=model_name,
                human_rows=human_rows,
                llm_rows=proxy_rows,
                utterance_event_types=set(DEFAULT_UTTERANCE_EVENT_TYPES),
                likert_col=likert_col,
                min_key_match_rate=0.7,
                include_conditioned_alignment=False,
                human_path=f"human::{data_prefix}",
                llm_path=(
                    f"degroot::{data_prefix}::alpha_public={alpha_public:.6f}::alpha_private={alpha_private:.6f}"
                ),
            )
            prefix_results_by_prefix[data_prefix][model_name] = result
            merge_prefix_result_into_aggregate(aggregates[model_name], result)
            status_rows.append(
                {
                    "data_prefix": data_prefix,
                    "model": model_name,
                    "status": result.status,
                    "message": result.message,
                    "human_path": result.human_path,
                    "llm_path": result.llm_path,
                    "alignment_mode": result.alignment_mode,
                    "key_match_rate": f"{result.key_match_rate:.6f}" if result.status == "ok" else "",
                    "aligned_pairs": result.aligned_pairs if result.status == "ok" else "",
                }
            )

        if prefix_error is not None:
            prefix_results_by_prefix.pop(data_prefix, None)

    raw_rows = build_raw_rows_from_aggregates(aggregates, model_names, ub_mode="gaussian_likert")
    raw_rows = [
        row
        for row in raw_rows
        if str(row.get("model", "")).strip() in set(model_names)
    ]
    if bootstrap_reps > 0:
        raw_rows = add_v2_bootstrap_cis_to_raw_rows(
            raw_rows,
            models=model_names,
            prefixes=[data_prefix for data_prefix in prefixes if data_prefix in rows_by_prefix],
            prefix_results_by_prefix=prefix_results_by_prefix,
            bootstrap_reps=bootstrap_reps,
            seed=bootstrap_seed,
        )

    raw_by_model = {str(row.get("model", "")): row for row in raw_rows}
    for spec in abm_specs:
        row = raw_by_model.get(str(spec["model"]))
        if row is None:
            continue
        row["abm_alpha_public"] = float(spec["alpha_public"])
        row["abm_alpha_private"] = float(spec.get("alpha_private", spec["alpha_public"]))
        row["abm_fit_scope"] = str(spec.get("fit_scope", ""))

    return raw_rows, status_rows


def merge_reference_and_abm_rows(
    reference_rows: list[dict[str, Any]],
    abm_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    abm_model_names = {str(row.get("model", "")).strip() for row in abm_rows}
    merged_rows = [
        dict(row)
        for row in reference_rows
        if str(row.get("model", "")).strip() not in abm_model_names
    ]
    merged_rows.extend(dict(row) for row in abm_rows)
    return merged_rows


def scope_prefix_splits(prefixes: list[str]) -> dict[str, list[str]]:
    depth_prefixes = [data_prefix for data_prefix in prefixes if is_depth_topic(data_prefix)]
    breadth_prefixes = [data_prefix for data_prefix in prefixes if not is_depth_topic(data_prefix)]
    return {
        "all": prefixes,
        "depth": depth_prefixes,
        "breadth": breadth_prefixes,
    }


def raw_fieldnames_for_rows(rows: list[dict[str, Any]]) -> list[str]:
    return collect_fieldnames(rows)


__all__ = [
    "DEFAULT_BOOTSTRAP_REPS",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_INITIAL_POST_SIDECAR_ROOT",
    "DEFAULT_REPORT_VERSION",
    "DEFAULT_TWEET_SIDECAR_ROOT",
    "FULL_FIT_MODEL_NAME",
    "SEAN_REPRODUCE_ROOT",
    "alpha_model_name",
    "collect_abm_raw_rows_with_bootstrap",
    "configure_v3_abm_environment",
    "load_human_rows_by_prefix",
    "merge_reference_and_abm_rows",
    "raw_fieldnames_for_rows",
    "scope_prefix_splits",
]