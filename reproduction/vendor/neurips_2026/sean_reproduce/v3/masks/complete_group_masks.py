from __future__ import annotations

import hashlib
import os
import re
from functools import lru_cache
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
V3_COMPLETE_GROUP_MASK_NAMESPACE_ROOT = SCRIPT_PATH.parent
V3_COMPLETE_GROUP_MASK_VARIANT_DIR_NAMES = {
    "both": "groups_with_valid_tweets_AND_opinions",
    "tweets_only": "groups_with_valid_tweets",
    "opinions_only": "groups_with_valid_opinions",
}
V3_COMPLETE_GROUP_MASK_VARIANT_DIRS = {
    variant: V3_COMPLETE_GROUP_MASK_NAMESPACE_ROOT / dirname
    for variant, dirname in V3_COMPLETE_GROUP_MASK_VARIANT_DIR_NAMES.items()
}
V3_COMPLETE_GROUP_MASK_ROOT = V3_COMPLETE_GROUP_MASK_NAMESPACE_ROOT
V3_COMPLETE_GROUP_MASK_GROUP_DIR = V3_COMPLETE_GROUP_MASK_VARIANT_DIRS["both"]
V3_COMPLETE_GROUP_MASK_MANIFEST = V3_COMPLETE_GROUP_MASK_GROUP_DIR / "manifest.tsv"
V3_COMPLETE_GROUP_MASK_ENV_VAR = "V3_COMPLETE_GROUP_MASK_ROOT"
V3_TWEET_GROUP_MASK_ENV_VAR = "V3_TWEET_GROUP_MASK_ROOT"
V3_OPINION_GROUP_MASK_ENV_VAR = "V3_OPINION_GROUP_MASK_ROOT"

CANONICAL_V3_EVAL_ROOT = REPO_ROOT / "neurips_2026" / "chengtao" / "result" / "eval_v3" / "human_llm"
REQUIRED_COMPLETE_PARTICIPANTS = 4
REQUIRED_POSITION_KEYS = (
    "initial_opinion",
    "tweet_1",
    "tweet_2",
    "tweet_3",
    "post_opinion",
)
TWEET_ONLY_REQUIRED_POSITION_KEYS = (
    "tweet_1",
    "tweet_2",
    "tweet_3",
)
OPINION_ONLY_REQUIRED_POSITION_KEYS = (
    "initial_opinion",
    "post_opinion",
)
V3_COMPLETE_GROUP_MASK_VARIANT_REQUIREMENTS = {
    "both": REQUIRED_POSITION_KEYS,
    "tweets_only": TWEET_ONLY_REQUIRED_POSITION_KEYS,
    "opinions_only": OPINION_ONLY_REQUIRED_POSITION_KEYS,
}

BASE_MODEL_V3_SERIES_KEYS = [
    "gpt-5-mini",
    "gpt-4o-mini-2024-07-18",
    "gemini-2.5-flash-lite",
    "Mistral-7B-Instruct-v0.3",
    "Qwen2.5-32B-Instruct",
    "Llama-3.1-70B-Instruct",
    "Llama-3.1-8B-Instruct",
    "Llama-3.1-Tulu-3-8B-SFT",
]

POST_TRAIN_V3_SERIES_KEYS = [
    "ft:Llama-3.1-8B-SFT-20260210:breadth-topic-3epochs-unsloth",
    "ft:Llama-3.1-8B-Instruct-DPO-20260212:breadth-topic-5epochs",
    "ft:Llama-3.1-8B-SFT-20260210:breadth-group-3epochs-unsloth",
    "ft:Llama-3.1-8B-Instruct-DPO-20260211:breadth-group-5epochs",
    "ft:Llama-3.1-8B-Instruct-SFT-20251218:depth-topic-5epochs",
    "ft:gpt-4o-mini-2024-07-18:camer:sft-20251218-depth-topic-5epochs:CoFxFPcv",
    "ft:Llama-3.1-8B-Instruct-DPO-20260105:depth-topic-5epochs",
    "ft:Llama-3.1-8B-Instruct-SFT-20251218:depth-group-5epochs",
    "ft:gpt-4o-mini-2024-07-18:camer:sft-20251218-depth-group-5epochs:CoFzijg0",
    "ft:Llama-3.1-8B-Instruct-DPO-20260105:depth-group-5epochs",
]

POST_TRAIN_V4_SERIES_KEYS = [
    "ft:Llama-3.1-8B-Instruct-SFT-20260430:breadth-topic-3epochs-unsloth",
    "ft:Llama-3.1-8B-Instruct-SFT-20260430:breadth-group-3epochs-unsloth",
    "ft:Llama-3.1-8B-Instruct-SFT-20260501:depth-topic-3epochs-unsloth",
    "ft:Llama-3.1-8B-Instruct-SFT-20260501:depth-group-3epochs-unsloth",
]

KNOWN_V3_SERIES_KEYS = [
    "human",
    *BASE_MODEL_V3_SERIES_KEYS,
    *[
        series_key
        for series_key in [*POST_TRAIN_V3_SERIES_KEYS, *POST_TRAIN_V4_SERIES_KEYS]
        if series_key not in BASE_MODEL_V3_SERIES_KEYS
    ],
]


def enable_v3_complete_group_masks(mask_root: Path | None = None) -> Path:
    root = (mask_root or configured_v3_complete_group_mask_root() or V3_COMPLETE_GROUP_MASK_ROOT).resolve()
    os.environ[V3_COMPLETE_GROUP_MASK_ENV_VAR] = str(root)
    return root


def configured_v3_complete_group_mask_root() -> Path | None:
    configured_root = os.environ.get(V3_COMPLETE_GROUP_MASK_ENV_VAR)
    if not configured_root:
        return None
    return Path(configured_root).expanduser().resolve()


def _configured_mask_root(env_var: str) -> Path | None:
    configured_root = os.environ.get(env_var)
    if not configured_root:
        return None
    return Path(configured_root).expanduser().resolve()


def _looks_like_mask_namespace(root: Path) -> bool:
    return any((root / dirname).exists() for dirname in V3_COMPLETE_GROUP_MASK_VARIANT_DIR_NAMES.values())


def variant_mask_root(variant: str) -> Path:
    try:
        return V3_COMPLETE_GROUP_MASK_VARIANT_DIRS[variant]
    except KeyError as exc:
        supported = ", ".join(sorted(V3_COMPLETE_GROUP_MASK_VARIANT_DIRS))
        raise ValueError(f"Unsupported v3 complete-group mask variant={variant!r}; expected one of: {supported}") from exc


def configured_variant_mask_root(variant: str, mask_root: Path | None = None) -> Path:
    env_var = {
        "tweets_only": V3_TWEET_GROUP_MASK_ENV_VAR,
        "opinions_only": V3_OPINION_GROUP_MASK_ENV_VAR,
    }.get(variant)
    configured_specific_root = _configured_mask_root(env_var) if env_var is not None else None
    if configured_specific_root is not None:
        return configured_specific_root

    configured_root = mask_root or configured_v3_complete_group_mask_root()
    if configured_root is None:
        return variant_mask_root(variant)
    if _looks_like_mask_namespace(configured_root):
        return configured_root / V3_COMPLETE_GROUP_MASK_VARIANT_DIR_NAMES[variant]
    return configured_root


def persisted_group_mask_manifest_path(mask_root: Path | None = None) -> Path:
    root = (mask_root or V3_COMPLETE_GROUP_MASK_GROUP_DIR).resolve()
    return root / "manifest.tsv"


def _nonempty_text(value: object) -> str:
    text = str(value or "").strip()
    return text


def parse_round(value: object) -> int | None:
    text = _nonempty_text(value)
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def participant_key(row: dict[str, str]) -> str | None:
    for key in ("empirica_id", "worker_id", "sender_id"):
        value = _nonempty_text(row.get(key))
        if value:
            return value
    return None


def row_position_key(row: dict[str, str]) -> str | None:
    event_type = _nonempty_text(row.get("event_type"))
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


def canonical_group_key(data_prefix: str, group_id: str) -> str:
    normalized_prefix = _nonempty_text(data_prefix)
    normalized_group = _nonempty_text(group_id) or normalized_prefix
    if normalized_group == normalized_prefix:
        return normalized_prefix
    return f"{normalized_prefix}::{normalized_group}"


def data_prefix_for_group_key(group_key: str) -> str:
    return str(group_key).split("::", 1)[0]


def infer_eval_relative_path(csv_path: Path) -> Path | None:
    try:
        return csv_path.resolve().relative_to(CANONICAL_V3_EVAL_ROOT.resolve())
    except ValueError:
        return None


def infer_data_prefix_from_eval_csv_path(csv_path: Path) -> str | None:
    relative_path = infer_eval_relative_path(csv_path)
    if relative_path is None or len(relative_path.parts) < 2:
        return None
    return relative_path.parts[0]


def infer_series_key_from_eval_csv_path(csv_path: Path) -> str | None:
    relative_path = infer_eval_relative_path(csv_path)
    if relative_path is None or relative_path.suffix.lower() != ".csv":
        return None
    if len(relative_path.parts) >= 3 and relative_path.name.startswith("opinion_human"):
        return "human"
    if len(relative_path.parts) >= 3 and relative_path.name.startswith("opinion_llm"):
        return relative_path.parts[1]
    return None


def stable_series_mask_filename(series_key: str) -> str:
    normalized = _nonempty_text(series_key)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", normalized).strip("._") or "series"
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
    return f"{slug}__{digest}.txt"


def persisted_group_mask_path(series_key: str, mask_root: Path | None = None) -> Path:
    root = (mask_root or V3_COMPLETE_GROUP_MASK_GROUP_DIR).resolve()
    return root / stable_series_mask_filename(series_key)


@lru_cache(maxsize=None)
def _load_persisted_group_mask_cached(mask_path_text: str) -> frozenset[str]:
    mask_path = Path(mask_path_text)
    if not mask_path.exists():
        raise FileNotFoundError(f"Persisted complete-group mask not found: {mask_path}")
    with mask_path.open("r", encoding="utf-8") as handle:
        values = frozenset(line.strip() for line in handle if line.strip())
    return values


def load_persisted_group_mask(series_key: str, mask_root: Path | None = None) -> set[str]:
    mask_path = persisted_group_mask_path(series_key, mask_root)
    return set(_load_persisted_group_mask_cached(str(mask_path.resolve())))


def qualified_complete_group_keys(
    rows: list[dict[str, str]],
    *,
    data_prefix: str,
    likert_col: str = "likert_pred",
    required_position_keys: tuple[str, ...] = REQUIRED_POSITION_KEYS,
) -> list[str]:
    grouped_positions: dict[str, dict[str, set[str]]] = {}
    for row in rows:
        position_key = row_position_key(row)
        if position_key is None:
            continue
        if position_key not in required_position_keys:
            continue
        participant = participant_key(row)
        if participant is None:
            continue
        if _nonempty_text(row.get(likert_col)) == "":
            continue
        group_id = _nonempty_text(row.get("group_id")) or data_prefix
        grouped_positions.setdefault(group_id, {}).setdefault(position_key, set()).add(participant)

    qualified_keys: list[str] = []
    for group_id, position_store in grouped_positions.items():
        if any(position_key not in position_store for position_key in required_position_keys):
            continue
        participant_sets = [position_store[position_key] for position_key in required_position_keys]
        common_participants = set.intersection(*participant_sets)
        if len(common_participants) != REQUIRED_COMPLETE_PARTICIPANTS:
            continue
        if not all(participants == common_participants for participants in participant_sets):
            continue
        qualified_keys.append(canonical_group_key(data_prefix, group_id))
    return sorted(qualified_keys)


def filter_rows_to_group_mask(
    rows: list[dict[str, str]],
    *,
    data_prefix: str,
    allowed_group_keys: set[str],
) -> list[dict[str, str]]:
    if not allowed_group_keys:
        return []
    filtered_rows: list[dict[str, str]] = []
    for row in rows:
        group_id = _nonempty_text(row.get("group_id")) or data_prefix
        if canonical_group_key(data_prefix, group_id) in allowed_group_keys:
            filtered_rows.append(row)
    return filtered_rows


def filter_rows_for_v3_complete_group_mask(
    rows: list[dict[str, str]],
    *,
    csv_path: Path,
    mask_root: Path | None = None,
) -> list[dict[str, str]]:
    resolved_mask_root = mask_root or configured_v3_complete_group_mask_root()
    if (
        resolved_mask_root is None
        and _configured_mask_root(V3_TWEET_GROUP_MASK_ENV_VAR) is None
        and _configured_mask_root(V3_OPINION_GROUP_MASK_ENV_VAR) is None
    ):
        return rows
    series_key = infer_series_key_from_eval_csv_path(csv_path)
    data_prefix = infer_data_prefix_from_eval_csv_path(csv_path)
    if series_key is None or data_prefix is None:
        return rows
    tweet_group_keys = load_persisted_group_mask(
        series_key,
        configured_variant_mask_root("tweets_only", resolved_mask_root),
    )
    opinion_group_keys = load_persisted_group_mask(
        series_key,
        configured_variant_mask_root("opinions_only", resolved_mask_root),
    )
    fallback_group_keys = tweet_group_keys | opinion_group_keys

    filtered_rows: list[dict[str, str]] = []
    for row in rows:
        event_type = _nonempty_text(row.get("event_type"))
        if event_type == "tweet":
            allowed_group_keys = tweet_group_keys
        elif event_type in {"Initial Opinion", "Post Opinion"}:
            allowed_group_keys = opinion_group_keys
        else:
            allowed_group_keys = fallback_group_keys

        group_id = _nonempty_text(row.get("group_id")) or data_prefix
        if canonical_group_key(data_prefix, group_id) in allowed_group_keys:
            filtered_rows.append(row)
    return filtered_rows
