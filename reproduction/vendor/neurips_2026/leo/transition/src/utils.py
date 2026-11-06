from __future__ import annotations

import csv
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neurips_2026.sean_reproduce.v3.masks import filter_rows_for_v3_complete_group_mask  # noqa: E402


IGNORE_PREFIXES = {
    '20250401_200733_A_"body_cleanse,"_in_which_you_consume_only_particular_kinds_of_nutrients_over_1-3_days,_helps_your_body_to_eliminate_toxins_01JQSDNZ14DGZFDR8TA28272JK'
}
DEFAULT_PATTERN = r"2025(03|04|05|06|07|08).*\.csv"
INITIAL_POST_REJUDGE_SUFFIX = "__initial_post_rejudge.csv"
TWEET_REJUDGE_SUFFIX = "__tweet_rejudge.csv"
METRIC_VERSION_ALIASES = {"v0_fix": "v0_fixed"}
VALIDITY_FILTERING_METRIC_VERSIONS = {"v0_fixed", "v3", "v4", "v4_b"}
DEFAULT_TRANSITION_SIMULATION_VERSION = "v0"
RESULT_ROOT_OVERRIDE_ENV_VAR = "TRANSITION_RESULT_ROOT_OVERRIDE"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def chengtao_root() -> Path:
    return repo_root() / "neurips_2026" / "chengtao"


def result_root() -> Path:
    override = os.environ.get(RESULT_ROOT_OVERRIDE_ENV_VAR, "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return chengtao_root() / "result" / "eval_v3" / "human_llm"


def transition_simulation_version() -> str:
    return os.environ.get("TRANSITION_SIMULATION_VERSION", DEFAULT_TRANSITION_SIMULATION_VERSION)


def llm_transition_csv_filename() -> str:
    return f"opinion_llm_memory_gpt-5-mini_{transition_simulation_version()}.csv"


def human_transition_csv_filename() -> str:
    return f"opinion_human_memory_gpt-5-mini_{transition_simulation_version()}.csv"


def rejudged_labels_root() -> Path:
    return repo_root() / "neurips_2026" / "rejudged_labels" / "eval_v3"


def initial_post_rejudged_root() -> Path:
    return rejudged_labels_root() / "initial_post"


def tweet_rejudged_root() -> Path:
    return rejudged_labels_root() / "tweet"


def data_root() -> Path:
    return repo_root() / "data"


def output_root() -> Path:
    return Path(__file__).resolve().parent


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_sidecar_csv_path(csv_path: Path, sidecar_root: Path, suffix: str) -> Path | None:
    try:
        relative_path = csv_path.relative_to(result_root())
    except ValueError:
        return None
    sidecar_path = sidecar_root / relative_path
    return sidecar_path.with_name(sidecar_path.stem + suffix)


def _load_sidecar_validity_overrides(csv_path: Path, sidecar_root: Path, suffix: str) -> dict[int, str]:
    sidecar_path = _get_sidecar_csv_path(csv_path, sidecar_root, suffix)
    overrides: dict[int, str] = {}
    if sidecar_path is None or not sidecar_path.exists():
        return overrides

    with sidecar_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
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


def normalize_metric_version(metric_version: str) -> str:
    return METRIC_VERSION_ALIASES.get(metric_version, metric_version)


def metric_version_family(metric_version: str) -> str:
    normalized = normalize_metric_version(metric_version)
    return "v4" if normalized.startswith("v4") else normalized


def metric_version_filters_invalid(metric_version: str) -> bool:
    return metric_version_family(metric_version) in VALIDITY_FILTERING_METRIC_VERSIONS


def metric_version_uses_rejudged_sidecars(metric_version: str) -> bool:
    return metric_version_family(metric_version) in {"v3", "v4"}


def read_eval_csv_rows(
    csv_path: Path,
    *,
    respect_validity_labels: bool = False,
    apply_rejudged_sidecars: bool = False,
    likert_col: str = "likert_pred",
) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not respect_validity_labels:
        return filter_rows_for_v3_complete_group_mask(rows, csv_path=csv_path)

    initial_post_overrides: dict[int, str] = {}
    tweet_overrides: dict[int, str] = {}
    if apply_rejudged_sidecars:
        initial_post_overrides = _load_sidecar_validity_overrides(
            csv_path,
            initial_post_rejudged_root(),
            INITIAL_POST_REJUDGE_SUFFIX,
        )
        tweet_overrides = _load_sidecar_validity_overrides(
            csv_path,
            tweet_rejudged_root(),
            TWEET_REJUDGE_SUFFIX,
        )

    adjusted_rows: list[dict[str, str]] = []
    for row_index, row in enumerate(rows, start=2):
        event_type = str(row.get("event_type") or "").strip()
        validity_column = determine_validity_column(csv_path, row)
        effective_validity = ""
        if validity_column is not None:
            effective_validity = str(row.get(validity_column) or row.get("validity") or "").strip().upper()

        if event_type in {"Initial Opinion", "Post Opinion"} and row_index in initial_post_overrides:
            effective_validity = initial_post_overrides[row_index]
        elif event_type == "tweet" and row_index in tweet_overrides:
            effective_validity = tweet_overrides[row_index]

        adjusted_row = dict(row)
        adjusted_row["effective_validity"] = effective_validity
        if effective_validity == "INVALID":
            adjusted_row[likert_col] = ""
            for opinion_col in ("sliderValue", "agreement_level"):
                if opinion_col in adjusted_row:
                    adjusted_row[opinion_col] = ""
            adjusted_rows.append(adjusted_row)
            continue

        adjusted_rows.append(adjusted_row)

    return filter_rows_for_v3_complete_group_mask(adjusted_rows, csv_path=csv_path)


def load_eval_dataframe(
    csv_path: Path,
    *,
    respect_validity_labels: bool = False,
    apply_rejudged_sidecars: bool = False,
    likert_col: str = "likert_pred",
) -> pd.DataFrame:
    return pd.DataFrame(
        read_eval_csv_rows(
            csv_path,
            respect_validity_labels=respect_validity_labels,
            apply_rejudged_sidecars=apply_rejudged_sidecars,
            likert_col=likert_col,
        )
    )


def golden_group_mask_v0_path() -> Path:
    return (
        chengtao_root()
        / "inputs"
        / "masks_train_test"
        / "golden_group_v0"
        / "aggregate_universe_mask_golden_v0.txt"
    )


def collection_golden_mask_path(collection_type: str) -> Path:
    collection = collection_type.lower()
    if collection not in {"breadth", "depth"}:
        raise ValueError(f"Unsupported collection_type={collection_type}")
    return (
        chengtao_root()
        / "inputs"
        / "masks_train_test"
        / "golden_group_v0"
        / collection
        / f"{collection}_universe_mask_golden_v0.txt"
    )


def collection_split_mask_path(collection_type: str, split: str, partition: str) -> Path:
    collection = collection_type.lower()
    split_name = split.lower()
    partition_name = partition.lower()
    if collection not in {"breadth", "depth"}:
        raise ValueError(f"Unsupported collection_type={collection_type}")
    if split_name not in {"topic", "group"}:
        raise ValueError(f"Unsupported split={split}")
    if partition_name not in {"train", "test"}:
        raise ValueError(f"Unsupported partition={partition}")

    return (
        chengtao_root()
        / "inputs"
        / "masks_train_test"
        / "golden_group_v0"
        / collection
        / f"{collection}_{partition_name}_{split_name}_mask_golden_v0.txt"
    )


def get_topic(data_prefix: str) -> str:
    match = re.search(r"\d{8}_\d{6}_(.*)_.{26}$", str(data_prefix))
    if not match:
        return str(data_prefix)
    return match.group(1)


def get_valid_data_prefixes(base_path: str | Path | None = None, subset: bool = False) -> list[str]:
    del subset
    base_dir = Path(base_path) if base_path is not None else data_root()
    processed_dir = base_dir / "processed_data"
    pattern = re.compile(DEFAULT_PATTERN)
    invalid_files = {
        "20250307_005857_Regular_fasting_will_improve_your_health_01JNQ0VASHPVHHPWEMS9ZW45MK.csv",
        "20250221_162336_The_US_deficit_increased_after_President_Obama_was_elected_01JMMKSXQ28HBP1PQ0Z4FJ86V7.csv",
        "20250407_193709_The_position_of_the_planets_at_the_time_of_your_birth_can_influence_your_personality_01JR8T86G0GKQW3ZX1HY14N3D2.csv",
        "20250403_200510_The_US_deficit_increased_after_President_Obama_was_elected_01JQYJJF8919Z9P7QC2FXAZR1R.csv",
        "20250331_202343_The_United_States_has_the_highest_federal_income_tax_rate_of_any_Western_country_01JQPWA5XP98NBXZYWQRQ3TA73.csv",
        "20250401_040642_The_United_States_has_the_highest_federal_income_tax_rate_of_any_Western_country_01JQQPKVAARAHTCV41MJK27K5E.csv",
        "20250306_182733_Everything_that_happens_can_eventually_be_explained_by_science_01JNP8FT36R1JZ56F3ZP1H1V1T.csv",
        "20250403_191749_A_\"body_cleanse,\"_in_which_you_consume_only_particular_kinds_of_nutrients_over_1-3_days,_helps_your_body_to_eliminate_toxins_01JQYFRNFBG8RG0JRY47K2CGSP.csv",
        "20250404_195614_Everything_that_happens_can_eventually_be_explained_by_science_01JR144VYTRCJGQ76WYYCFCHXZ.csv",
        "20250403_212439_The_United_States_has_the_highest_federal_income_tax_rate_of_any_Western_country_01JQYNW3ZWDPFM7ZN7JAJEDSV0.csv",
        '20250401_200733_A_"body_cleanse,"_in_which_you_consume_only_particular_kinds_of_nutrients_over_1-3_days,_helps_your_body_to_eliminate_toxins_01JQSDNZ14DGZFDR8TA28272JK.csv',
    }
    prefixes: list[str] = []
    for entry in sorted(processed_dir.iterdir()):
        if not entry.is_file() or entry.name in invalid_files:
            continue
        if pattern.match(entry.name):
            prefixes.append(entry.stem)
    return prefixes


def get_typed_data_prefixes(data_prefixes: list[str], collection_type: str, base_path: str | Path | None = None) -> list[str]:
    base_dir = Path(base_path) if base_path is not None else data_root()
    dirname = "phase_2_breadth_topics" if collection_type == "breadth" else "phase_1_depth_topics"
    raw_dir = base_dir / "raw_data" / dirname
    typed: set[str] = set()
    for root, _, files in os.walk(raw_dir):
        for filename in files:
            if filename.endswith(".csv"):
                typed.add(filename.replace("_0.0.1", "").replace(".csv", ""))
    return [prefix for prefix in data_prefixes if prefix in typed]


def get_golden(filter_type: str, dp_type: str = "depth") -> set[str]:
    if filter_type.lower() not in {"golden_groups", "golden_group"}:
        raise ValueError(f"Unsupported filter_type={filter_type}; only golden_groups is needed here")
    mask_path = collection_golden_mask_path(dp_type)
    if not mask_path.exists():
        raise FileNotFoundError(f"Required golden universe mask not found: {mask_path}")
    with mask_path.open("r", encoding="utf-8") as handle:
        groups = {line.strip() for line in handle if line.strip()}
    return groups


def prepare_golden_dps(collection_type: str) -> list[str]:
    valid = get_valid_data_prefixes()
    typed = get_typed_data_prefixes(valid, collection_type)
    golden = get_golden("golden_groups", dp_type=collection_type)
    return [prefix for prefix in typed if prefix in golden and prefix not in IGNORE_PREFIXES]


def safe_model_dir_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", model)


def _processed_csv_names(collection_type: str) -> list[str]:
    valid_prefixes = prepare_golden_dps(collection_type)
    return [f"{prefix}.csv" for prefix in valid_prefixes]


@lru_cache(maxsize=None)
def split_membership(split: str, collection_type: str) -> dict[str, str]:
    if split not in {"topic", "group"}:
        raise ValueError(f"Unsupported split={split}")

    expected_prefixes = set(prepare_golden_dps(collection_type))
    train_mask_path = collection_split_mask_path(collection_type, split, "train")
    test_mask_path = collection_split_mask_path(collection_type, split, "test")

    missing_paths = [path for path in (train_mask_path, test_mask_path) if not path.exists()]
    if missing_paths:
        missing_display = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Required golden split mask(s) not found: {missing_display}")

    train_prefixes = _read_mask_prefixes(train_mask_path)
    test_prefixes = _read_mask_prefixes(test_mask_path)

    overlap = train_prefixes & test_prefixes
    if overlap:
        raise ValueError(
            f"Golden split masks overlap for collection={collection_type}, split={split}: {sorted(overlap)[:10]}"
        )

    covered_prefixes = train_prefixes | test_prefixes
    missing_prefixes = sorted(expected_prefixes - covered_prefixes)
    extra_prefixes = sorted(covered_prefixes - expected_prefixes)
    if missing_prefixes or extra_prefixes:
        problems: list[str] = []
        if missing_prefixes:
            problems.append(f"missing prefixes {missing_prefixes[:10]}")
        if extra_prefixes:
            problems.append(f"unexpected prefixes {extra_prefixes[:10]}")
        raise ValueError(
            f"Golden split masks inconsistent for collection={collection_type}, split={split}: " + "; ".join(problems)
        )

    membership = {prefix: "train" for prefix in train_prefixes}
    membership.update({prefix: "test" for prefix in test_prefixes})
    return membership


def _read_mask_prefixes(mask_path: Path) -> set[str]:
    with mask_path.open("r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def train_or_test(data_prefix: str, split: str, collection_type: str) -> str:
    membership = split_membership(split, collection_type)
    if data_prefix not in membership:
        raise KeyError(
            f"No golden split assignment found for data_prefix={data_prefix}, split={split}, collection_type={collection_type}"
        )
    return membership[data_prefix]