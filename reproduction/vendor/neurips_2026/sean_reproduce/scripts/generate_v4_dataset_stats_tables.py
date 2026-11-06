#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neurips_2026.chengtao.src.radar.generate_plots import (  # noqa: E402
    DEFAULT_UTTERANCE_EVENT_TYPES,
    extract_topic_from_data_prefix,
    read_csv_rows,
)
from neurips_2026.sean_reproduce.base_model_v3_style import (  # noqa: E402
    CANONICAL_V3_HUMAN_SOURCE_MODEL,
)
from neurips_2026.sean_reproduce.result_version_config import (  # noqa: E402
    human_csv_filename_for_result,
)


DEFAULT_RESULT_VERSION = "v4"
DEFAULT_EVAL_ROOT = REPO_ROOT / "neurips_2026" / "chengtao" / "result" / "eval_v3" / "human_llm"
DEFAULT_MASK_NAMESPACE_ROOT = REPO_ROOT / "neurips_2026" / "chengtao" / "inputs" / "masks_train_test"
DEFAULT_INITIAL_POST_SIDECAR_ROOT = (
    REPO_ROOT / "neurips_2026" / "rejudged_labels" / "eval_v3" / "initial_post"
)
DEFAULT_TWEET_SIDECAR_ROOT = (
    REPO_ROOT / "neurips_2026" / "rejudged_labels" / "eval_v3" / "tweet"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "neurips_2026" / "sean_reproduce" / "v4_pub" / "table_dataset_stats"
LIKERT_COL = "likert_pred"
UTTERANCE_EVENT_TYPES = frozenset(DEFAULT_UTTERANCE_EVENT_TYPES)
OPINION_EVENT_TYPES = frozenset(("Initial Opinion", "Post Opinion"))
TWEET_EVENT_TYPE = "tweet"
CHAT_MESSAGE_EVENT_TYPE = "message_sent"

COLLECTION_DISPLAY = {
    "depth": "Depth",
    "breadth": "Breadth",
    "combined": "Combined Topic Set (D+B)",
}

PARTITION_DISPLAY = {
    "train": "Train",
    "test": "Test",
}

MASK_VARIANT_DIR = {
    "unmasked": "raw_full_set",
    "masked": "golden_group_v0",
}

MASK_VARIANT_LABEL = {
    "unmasked": "Raw split membership before golden filtering",
    "masked": "Golden-group v0 masked membership",
}


def ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def percent_str(numerator: int, denominator: int) -> str:
    return f"{100.0 * ratio(numerator, denominator):.2f}"


@dataclass
class PrefixResolution:
    requested_prefixes: list[str]
    available_prefixes: list[str]
    missing_prefixes: list[str]

    @property
    def requested_count(self) -> int:
        return len(self.requested_prefixes)

    @property
    def available_count(self) -> int:
        return len(self.available_prefixes)

    @property
    def missing_count(self) -> int:
        return len(self.missing_prefixes)


@dataclass
class SliceStats:
    topics: set[str] = field(default_factory=set)
    groups: set[str] = field(default_factory=set)
    subjects: set[str] = field(default_factory=set)
    prefixes: set[str] = field(default_factory=set)
    message_count: int = 0
    on_topic_message_count: int = 0
    tweet_count: int = 0
    on_topic_tweet_count: int = 0
    opinion_count: int = 0
    on_topic_opinion_count: int = 0
    chat_message_count: int = 0
    on_topic_chat_message_count: int = 0

    @property
    def prefix_count(self) -> int:
        return len(self.prefixes)

    @property
    def topic_count(self) -> int:
        return len(self.topics)

    @property
    def subject_count(self) -> int:
        return len(self.subjects)

    @property
    def group_count(self) -> int:
        return len(self.groups)

    @property
    def groups_per_topic(self) -> float:
        return self.group_count / self.topic_count if self.topic_count else 0.0

    def update_from_prefix_rows(self, data_prefix: str, rows: list[dict[str, str]]) -> None:
        self.prefixes.add(data_prefix)
        self.topics.add(extract_topic_from_data_prefix(data_prefix))
        for row in rows:
            event_type = str(row.get("event_type") or "").strip()
            if event_type not in UTTERANCE_EVENT_TYPES:
                continue
            self.message_count += 1

            subject_id = first_nonempty(row, "worker_id", "sender_id", "empirica_id")
            if subject_id:
                self.subjects.add(subject_id)

            group_id = str(row.get("group_id") or "").strip() or data_prefix
            self.groups.add(group_id)

            is_on_topic = bool(str(row.get(LIKERT_COL) or "").strip())
            if is_on_topic:
                self.on_topic_message_count += 1

            if event_type == TWEET_EVENT_TYPE:
                self.tweet_count += 1
                if is_on_topic:
                    self.on_topic_tweet_count += 1
            elif event_type in OPINION_EVENT_TYPES:
                self.opinion_count += 1
                if is_on_topic:
                    self.on_topic_opinion_count += 1
            elif event_type == CHAT_MESSAGE_EVENT_TYPE:
                self.chat_message_count += 1
                if is_on_topic:
                    self.on_topic_chat_message_count += 1

def first_nonempty(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def read_mask_prefixes(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def mask_path(mask_namespace_root: Path, variant: str, collection: str, split_type: str, partition: str) -> Path:
    if variant == "masked":
        return mask_namespace_root / MASK_VARIANT_DIR[variant] / collection / f"{collection}_{partition}_{split_type}_mask_golden_v0.txt"
    if variant == "unmasked":
        return mask_namespace_root / MASK_VARIANT_DIR[variant] / collection / f"{collection}_{partition}_{split_type}_mask_raw.txt"
    raise ValueError(f"Unsupported mask variant: {variant}")


def resolve_human_csv_path(eval_root: Path, data_prefix: str, filename: str) -> Path:
    canonical_path = eval_root / data_prefix / CANONICAL_V3_HUMAN_SOURCE_MODEL / filename
    if canonical_path.exists():
        return canonical_path

    direct_path = eval_root / data_prefix / filename
    if direct_path.exists():
        return direct_path

    prefix_dir = eval_root / data_prefix
    if prefix_dir.exists():
        for child in sorted(prefix_dir.iterdir(), key=lambda item: item.name):
            if not child.is_dir():
                continue
            candidate = child / filename
            if candidate.exists():
                return candidate

    raise FileNotFoundError(f"Human CSV not found for data_prefix={data_prefix!r} filename={filename!r}")


def has_human_csv(eval_root: Path, data_prefix: str, filename: str) -> bool:
    try:
        resolve_human_csv_path(eval_root, data_prefix, filename)
    except FileNotFoundError:
        return False
    return True


def resolve_available_prefixes(
    prefixes: list[str],
    *,
    eval_root: Path,
    human_filename: str,
) -> PrefixResolution:
    available_prefixes: list[str] = []
    missing_prefixes: list[str] = []
    for data_prefix in prefixes:
        if has_human_csv(eval_root, data_prefix, human_filename):
            available_prefixes.append(data_prefix)
        else:
            missing_prefixes.append(data_prefix)
    return PrefixResolution(
        requested_prefixes=list(prefixes),
        available_prefixes=available_prefixes,
        missing_prefixes=missing_prefixes,
    )


def collect_stats_for_prefixes(
    prefixes: list[str],
    *,
    eval_root: Path,
    human_filename: str,
    initial_post_sidecar_root: Path,
    tweet_sidecar_root: Path,
) -> SliceStats:
    stats = SliceStats()
    for data_prefix in prefixes:
        human_path = resolve_human_csv_path(eval_root, data_prefix, human_filename)
        rows = read_csv_rows(
            human_path,
            respect_validity_labels=True,
            likert_col=LIKERT_COL,
            data_roots=[eval_root],
            initial_post_sidecar_roots=[initial_post_sidecar_root],
            tweet_sidecar_roots=[tweet_sidecar_root],
        )
        stats.update_from_prefix_rows(data_prefix, rows)
    return stats


def build_comparison_row(
    *,
    collection: str,
    partition: str,
    unmasked_resolution: PrefixResolution,
    masked_resolution: PrefixResolution,
    unmasked: SliceStats,
    masked: SliceStats,
) -> dict[str, str | int | float]:
    return {
        "collection": COLLECTION_DISPLAY[collection],
        "partition": PARTITION_DISPLAY[partition],
        "requested_unmasked_prefixes": unmasked_resolution.requested_count,
        "requested_masked_prefixes": masked_resolution.requested_count,
        "requested_masked_out_prefixes": unmasked_resolution.requested_count - masked_resolution.requested_count,
        "missing_unmasked_eval_prefixes": unmasked_resolution.missing_count,
        "missing_masked_eval_prefixes": masked_resolution.missing_count,
        "counted_unmasked_prefixes": unmasked_resolution.available_count,
        "counted_masked_prefixes": masked_resolution.available_count,
        "counted_masked_out_prefixes": unmasked_resolution.available_count - masked_resolution.available_count,
        "unmasked_topics": unmasked.topic_count,
        "masked_topics": masked.topic_count,
        "masked_out_topics": unmasked.topic_count - masked.topic_count,
        "unmasked_messages": unmasked.message_count,
        "masked_messages": masked.message_count,
        "masked_out_messages": unmasked.message_count - masked.message_count,
        "unmasked_on_topic_messages": unmasked.on_topic_message_count,
        "masked_on_topic_messages": masked.on_topic_message_count,
        "masked_out_on_topic_messages": unmasked.on_topic_message_count - masked.on_topic_message_count,
        "unmasked_tweets": unmasked.tweet_count,
        "masked_tweets": masked.tweet_count,
        "masked_out_tweets": unmasked.tweet_count - masked.tweet_count,
        "unmasked_on_topic_tweets": unmasked.on_topic_tweet_count,
        "masked_on_topic_tweets": masked.on_topic_tweet_count,
        "unmasked_initial_post_opinions": unmasked.opinion_count,
        "masked_initial_post_opinions": masked.opinion_count,
        "masked_out_initial_post_opinions": unmasked.opinion_count - masked.opinion_count,
        "unmasked_on_topic_initial_post_opinions": unmasked.on_topic_opinion_count,
        "masked_on_topic_initial_post_opinions": masked.on_topic_opinion_count,
        "unmasked_chat_messages": unmasked.chat_message_count,
        "masked_chat_messages": masked.chat_message_count,
        "masked_out_chat_messages": unmasked.chat_message_count - masked.chat_message_count,
        "unmasked_on_topic_chat_messages": unmasked.on_topic_chat_message_count,
        "masked_on_topic_chat_messages": masked.on_topic_chat_message_count,
        "unmasked_subjects": unmasked.subject_count,
        "masked_subjects": masked.subject_count,
        "masked_out_subjects": unmasked.subject_count - masked.subject_count,
        "unmasked_groups": unmasked.group_count,
        "masked_groups": masked.group_count,
        "masked_out_groups": unmasked.group_count - masked.group_count,
        "unmasked_groups_per_topic": f"{unmasked.groups_per_topic:.2f}",
        "masked_groups_per_topic": f"{masked.groups_per_topic:.2f}",
        "groups_per_topic_delta": f"{(unmasked.groups_per_topic - masked.groups_per_topic):.2f}",
    }


def build_prefix_coverage_rows(rows: list[dict[str, str | int | float]]) -> list[dict[str, str | int | float]]:
    return [
        {
            "collection": row["collection"],
            "partition": row["partition"],
            "raw_requested_prefixes": row["requested_unmasked_prefixes"],
            "masked_requested_prefixes": row["requested_masked_prefixes"],
            "masked_out_prefixes": row["requested_masked_out_prefixes"],
            "prefixes_masked_pct": percent_str(
                int(row["requested_masked_out_prefixes"]),
                int(row["requested_unmasked_prefixes"]),
            ),
            "raw_missing_eval_prefixes": row["missing_unmasked_eval_prefixes"],
            "masked_missing_eval_prefixes": row["missing_masked_eval_prefixes"],
            "raw_counted_prefixes": row["counted_unmasked_prefixes"],
            "masked_counted_prefixes": row["counted_masked_prefixes"],
            "counted_masked_out_prefixes": row["counted_masked_out_prefixes"],
            "counted_prefixes_masked_pct": percent_str(
                int(row["counted_masked_out_prefixes"]),
                int(row["counted_unmasked_prefixes"]),
            ),
        }
        for row in rows
    ]


def build_count_statistics_rows(rows: list[dict[str, str | int | float]]) -> list[dict[str, str | int | float]]:
    count_rows: list[dict[str, str | int | float]] = []
    for row in rows:
        masked_out_messages = int(row["masked_out_messages"])
        unmasked_messages = int(row["unmasked_messages"])
        masked_out_on_topic_messages = int(row["masked_out_on_topic_messages"])
        unmasked_on_topic_messages = int(row["unmasked_on_topic_messages"])
        masked_out_tweets = int(row["masked_out_tweets"])
        unmasked_tweets = int(row["unmasked_tweets"])
        masked_out_initial_post_opinions = int(row["masked_out_initial_post_opinions"])
        unmasked_initial_post_opinions = int(row["unmasked_initial_post_opinions"])
        masked_out_chat_messages = int(row["masked_out_chat_messages"])
        unmasked_chat_messages = int(row["unmasked_chat_messages"])
        masked_out_subjects = int(row["masked_out_subjects"])
        unmasked_subjects = int(row["unmasked_subjects"])
        masked_out_groups = int(row["masked_out_groups"])
        unmasked_groups = int(row["unmasked_groups"])
        count_rows.append(
            {
                "collection": row["collection"],
                "partition": row["partition"],
                "raw_topics": row["unmasked_topics"],
                "masked_topics": row["masked_topics"],
                "masked_out_topics": row["masked_out_topics"],
                "raw_messages": row["unmasked_messages"],
                "masked_messages": row["masked_messages"],
                "masked_out_messages": row["masked_out_messages"],
                "messages_masked_pct": percent_str(masked_out_messages, unmasked_messages),
                "raw_on_topic_messages": row["unmasked_on_topic_messages"],
                "masked_on_topic_messages": row["masked_on_topic_messages"],
                "masked_out_on_topic_messages": row["masked_out_on_topic_messages"],
                "on_topic_messages_masked_pct": percent_str(masked_out_on_topic_messages, unmasked_on_topic_messages),
                "raw_message_on_topic_rate": percent_str(int(row["unmasked_on_topic_messages"]), unmasked_messages),
                "masked_message_on_topic_rate": percent_str(
                    int(row["masked_on_topic_messages"]),
                    int(row["masked_messages"]),
                ),
                "raw_tweets": row["unmasked_tweets"],
                "masked_tweets": row["masked_tweets"],
                "masked_out_tweets": row["masked_out_tweets"],
                "tweets_masked_pct": percent_str(masked_out_tweets, unmasked_tweets),
                "raw_tweet_on_topic_rate": percent_str(int(row["unmasked_on_topic_tweets"]), unmasked_tweets),
                "masked_tweet_on_topic_rate": percent_str(int(row["masked_on_topic_tweets"]), int(row["masked_tweets"])),
                "raw_initial_post_opinions": row["unmasked_initial_post_opinions"],
                "masked_initial_post_opinions": row["masked_initial_post_opinions"],
                "masked_out_initial_post_opinions": row["masked_out_initial_post_opinions"],
                "initial_post_opinions_masked_pct": percent_str(masked_out_initial_post_opinions, unmasked_initial_post_opinions),
                "raw_initial_post_on_topic_rate": percent_str(
                    int(row["unmasked_on_topic_initial_post_opinions"]),
                    unmasked_initial_post_opinions,
                ),
                "masked_initial_post_on_topic_rate": percent_str(
                    int(row["masked_on_topic_initial_post_opinions"]),
                    int(row["masked_initial_post_opinions"]),
                ),
                "raw_chat_messages": row["unmasked_chat_messages"],
                "masked_chat_messages": row["masked_chat_messages"],
                "masked_out_chat_messages": row["masked_out_chat_messages"],
                "chat_messages_masked_pct": percent_str(masked_out_chat_messages, unmasked_chat_messages),
                "raw_chat_message_on_topic_rate": percent_str(
                    int(row["unmasked_on_topic_chat_messages"]),
                    unmasked_chat_messages,
                ),
                "masked_chat_message_on_topic_rate": percent_str(
                    int(row["masked_on_topic_chat_messages"]),
                    int(row["masked_chat_messages"]),
                ),
                "raw_subjects": row["unmasked_subjects"],
                "masked_subjects": row["masked_subjects"],
                "masked_out_subjects": row["masked_out_subjects"],
                "subjects_masked_pct": percent_str(masked_out_subjects, unmasked_subjects),
                "raw_groups": row["unmasked_groups"],
                "masked_groups": row["masked_groups"],
                "masked_out_groups": row["masked_out_groups"],
                "groups_masked_pct": percent_str(masked_out_groups, unmasked_groups),
                "raw_groups_per_topic": row["unmasked_groups_per_topic"],
                "masked_groups_per_topic": row["masked_groups_per_topic"],
                "groups_per_topic_delta": row["groups_per_topic_delta"],
            }
        )
    return count_rows


def write_csv(path: Path, rows: list[dict[str, str | int | float]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_prefix_coverage_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    fieldnames = [
        "collection",
        "partition",
        "raw_requested_prefixes",
        "masked_requested_prefixes",
        "masked_out_prefixes",
        "prefixes_masked_pct",
        "raw_missing_eval_prefixes",
        "masked_missing_eval_prefixes",
        "raw_counted_prefixes",
        "masked_counted_prefixes",
        "counted_masked_out_prefixes",
        "counted_prefixes_masked_pct",
    ]
    write_csv(path, rows, fieldnames)


def write_count_statistics_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    fieldnames = [
        "collection",
        "partition",
        "raw_topics",
        "masked_topics",
        "masked_out_topics",
        "raw_messages",
        "masked_messages",
        "masked_out_messages",
        "messages_masked_pct",
        "raw_on_topic_messages",
        "masked_on_topic_messages",
        "masked_out_on_topic_messages",
        "on_topic_messages_masked_pct",
        "raw_message_on_topic_rate",
        "masked_message_on_topic_rate",
        "raw_tweets",
        "masked_tweets",
        "masked_out_tweets",
        "tweets_masked_pct",
        "raw_tweet_on_topic_rate",
        "masked_tweet_on_topic_rate",
        "raw_initial_post_opinions",
        "masked_initial_post_opinions",
        "masked_out_initial_post_opinions",
        "initial_post_opinions_masked_pct",
        "raw_initial_post_on_topic_rate",
        "masked_initial_post_on_topic_rate",
        "raw_chat_messages",
        "masked_chat_messages",
        "masked_out_chat_messages",
        "chat_messages_masked_pct",
        "raw_chat_message_on_topic_rate",
        "masked_chat_message_on_topic_rate",
        "raw_subjects",
        "masked_subjects",
        "masked_out_subjects",
        "subjects_masked_pct",
        "raw_groups",
        "masked_groups",
        "masked_out_groups",
        "groups_masked_pct",
        "raw_groups_per_topic",
        "masked_groups_per_topic",
        "groups_per_topic_delta",
    ]
    write_csv(path, rows, fieldnames)


def format_prefix_table(section_title: str, rows: list[dict[str, str | int | float]]) -> str:
    lines = [
        f"### Prefix Coverage ({section_title})",
        "",
        "| Collection | Partition | Raw Requested | Masked Requested | Masked Out | % Masked | Raw Missing Eval | Masked Missing Eval | Raw Counted | Masked Counted | Counted Masked Out | % Counted Masked |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {collection} | {partition} | {raw_requested_prefixes} | {masked_requested_prefixes} | {masked_out_prefixes} | {prefixes_masked_pct}% | {raw_missing_eval_prefixes} | {masked_missing_eval_prefixes} | {raw_counted_prefixes} | {masked_counted_prefixes} | {counted_masked_out_prefixes} | {counted_prefixes_masked_pct}% |".format(
                **row,
            )
        )
    return "\n".join(lines)


def format_count_table(section_title: str, rows: list[dict[str, str | int | float]]) -> str:
    lines = [
        f"### Count Statistics ({section_title})",
        "",
        f"#### Overall Utterances ({section_title})",
        "",
        "| Collection | Partition | Raw Topics | Masked Topics | Masked Out Topics | Raw Messages | Masked Messages | Masked Out Messages | % Messages Masked | Raw On-topic Messages | Masked On-topic Messages | Masked Out On-topic Messages | % On-topic Messages Masked | Raw Message On-topic Rate | Masked Message On-topic Rate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {collection} | {partition} | {raw_topics} | {masked_topics} | {masked_out_topics} | {raw_messages} | {masked_messages} | {masked_out_messages} | {messages_masked_pct}% | {raw_on_topic_messages} | {masked_on_topic_messages} | {masked_out_on_topic_messages} | {on_topic_messages_masked_pct}% | {raw_message_on_topic_rate}% | {masked_message_on_topic_rate}% |".format(**row)
        )

    lines.extend(
        [
            "",
            f"#### Event-Type Breakdown ({section_title})",
            "",
            "| Collection | Partition | Raw Tweets | Masked Tweets | Masked Out Tweets | % Tweets Masked | Raw Tweet On-topic Rate | Masked Tweet On-topic Rate | Raw Initial/Post Opinions | Masked Initial/Post Opinions | Masked Out Initial/Post Opinions | % Initial/Post Masked | Raw Initial/Post On-topic Rate | Masked Initial/Post On-topic Rate | Raw Chat Messages | Masked Chat Messages | Masked Out Chat Messages | % Chat Messages Masked | Raw Chat On-topic Rate | Masked Chat On-topic Rate |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| {collection} | {partition} | {raw_tweets} | {masked_tweets} | {masked_out_tweets} | {tweets_masked_pct}% | {raw_tweet_on_topic_rate}% | {masked_tweet_on_topic_rate}% | {raw_initial_post_opinions} | {masked_initial_post_opinions} | {masked_out_initial_post_opinions} | {initial_post_opinions_masked_pct}% | {raw_initial_post_on_topic_rate}% | {masked_initial_post_on_topic_rate}% | {raw_chat_messages} | {masked_chat_messages} | {masked_out_chat_messages} | {chat_messages_masked_pct}% | {raw_chat_message_on_topic_rate}% | {masked_chat_message_on_topic_rate}% |".format(**row)
        )

    lines.extend(
        [
            "",
            f"#### Subjects And Groups ({section_title})",
            "",
            "| Collection | Partition | Raw Subjects | Masked Subjects | Masked Out Subjects | % Subjects Masked | Raw Groups | Masked Groups | Masked Out Groups | % Groups Masked | Raw Groups/Topic | Masked Groups/Topic | Delta Groups/Topic |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| {collection} | {partition} | {raw_subjects} | {masked_subjects} | {masked_out_subjects} | {subjects_masked_pct}% | {raw_groups} | {masked_groups} | {masked_out_groups} | {groups_masked_pct}% | {raw_groups_per_topic} | {masked_groups_per_topic} | {groups_per_topic_delta} |".format(**row)
        )
    return "\n".join(lines)


def format_markdown_section(title: str, rows: list[dict[str, str | int | float]]) -> str:
    prefix_rows = build_prefix_coverage_rows(rows)
    count_rows = build_count_statistics_rows(rows)
    return "\n".join(
        [
            f"## {title}",
            "",
            format_prefix_table(title, prefix_rows),
            "",
            format_count_table(title, count_rows),
        ]
    )


def build_markdown_summary(
    *,
    result_version: str,
    human_filename: str,
    group_rows: list[dict[str, str | int | float]],
    topic_rows: list[dict[str, str | int | float]],
) -> str:
    notes = [
        "# v4 Dataset Statistics",
        "",
        f"- Result namespace: {result_version}",
        f"- Human CSV filename: {human_filename}",
        f"- Unmasked prefixes: neurips_2026/chengtao/inputs/masks_train_test/raw_full_set",
        f"- Masked prefixes: neurips_2026/chengtao/inputs/masks_train_test/golden_group_v0",
        "- Output files: `group_split_prefix_coverage_v4.csv`, `group_split_count_statistics_v4.csv`, `topic_split_prefix_coverage_v4.csv`, `topic_split_count_statistics_v4.csv`.",
        f"- Message counts use utterance rows with event_type in: {', '.join(DEFAULT_UTTERANCE_EVENT_TYPES)}",
        "- On-topic counts and rates are computed separately for all utterances, tweets, initial/post opinions, and chat-message rows after applying the same rejudged validity sidecars used by the v4 report pipeline.",
        "- Prefix-mask differences are reported directly from the raw and golden mask files.",
        "- Message/topic/subject/group differences are computed on the countable eval surface only, because a small number of raw prefixes do not have a corresponding human eval CSV under neurips_2026/chengtao/result/eval_v3/human_llm.",
        "- `masked_out_* = unmasked_* - masked_*` for the counted eval surface metrics.",
        "",
        format_markdown_section("Group Split", group_rows),
        "",
        format_markdown_section("Topic Split", topic_rows),
        "",
    ]
    return "\n".join(notes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate v4 masked dataset-statistics tables for train/test topic and group splits.")
    parser.add_argument("--result-version", default=DEFAULT_RESULT_VERSION)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--mask-namespace-root", type=Path, default=DEFAULT_MASK_NAMESPACE_ROOT)
    parser.add_argument("--initial-post-sidecar-root", type=Path, default=DEFAULT_INITIAL_POST_SIDECAR_ROOT)
    parser.add_argument("--tweet-sidecar-root", type=Path, default=DEFAULT_TWEET_SIDECAR_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    human_filename = human_csv_filename_for_result(args.result_version)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prefix_sets: dict[str, dict[str, dict[str, dict[str, list[str]]]]] = {
        variant: {
            collection: {
                split_type: {
                    partition: read_mask_prefixes(mask_path(args.mask_namespace_root, variant, collection, split_type, partition))
                    for partition in ("train", "test")
                }
                for split_type in ("group", "topic")
            }
            for collection in ("breadth", "depth")
        }
        for variant in ("unmasked", "masked")
    }

    group_rows: list[dict[str, str | int | float]] = []
    topic_rows: list[dict[str, str | int | float]] = []

    for split_type, target_rows in (("group", group_rows), ("topic", topic_rows)):
        for partition in ("train", "test"):
            for collection in ("depth", "breadth"):
                unmasked_resolution = resolve_available_prefixes(
                    prefix_sets["unmasked"][collection][split_type][partition],
                    eval_root=args.eval_root,
                    human_filename=human_filename,
                )
                masked_resolution = resolve_available_prefixes(
                    prefix_sets["masked"][collection][split_type][partition],
                    eval_root=args.eval_root,
                    human_filename=human_filename,
                )
                unmasked_stats = collect_stats_for_prefixes(
                    unmasked_resolution.available_prefixes,
                    eval_root=args.eval_root,
                    human_filename=human_filename,
                    initial_post_sidecar_root=args.initial_post_sidecar_root,
                    tweet_sidecar_root=args.tweet_sidecar_root,
                )
                masked_stats = collect_stats_for_prefixes(
                    masked_resolution.available_prefixes,
                    eval_root=args.eval_root,
                    human_filename=human_filename,
                    initial_post_sidecar_root=args.initial_post_sidecar_root,
                    tweet_sidecar_root=args.tweet_sidecar_root,
                )
                target_rows.append(
                    build_comparison_row(
                        collection=collection,
                        partition=partition,
                        unmasked_resolution=unmasked_resolution,
                        masked_resolution=masked_resolution,
                        unmasked=unmasked_stats,
                        masked=masked_stats,
                    )
                )

            combined_unmasked_prefixes = sorted(
                set(prefix_sets["unmasked"]["breadth"][split_type][partition])
                | set(prefix_sets["unmasked"]["depth"][split_type][partition])
            )
            combined_masked_prefixes = sorted(
                set(prefix_sets["masked"]["breadth"][split_type][partition])
                | set(prefix_sets["masked"]["depth"][split_type][partition])
            )
            combined_unmasked_resolution = resolve_available_prefixes(
                combined_unmasked_prefixes,
                eval_root=args.eval_root,
                human_filename=human_filename,
            )
            combined_masked_resolution = resolve_available_prefixes(
                combined_masked_prefixes,
                eval_root=args.eval_root,
                human_filename=human_filename,
            )
            combined_unmasked_stats = collect_stats_for_prefixes(
                combined_unmasked_resolution.available_prefixes,
                eval_root=args.eval_root,
                human_filename=human_filename,
                initial_post_sidecar_root=args.initial_post_sidecar_root,
                tweet_sidecar_root=args.tweet_sidecar_root,
            )
            combined_masked_stats = collect_stats_for_prefixes(
                combined_masked_resolution.available_prefixes,
                eval_root=args.eval_root,
                human_filename=human_filename,
                initial_post_sidecar_root=args.initial_post_sidecar_root,
                tweet_sidecar_root=args.tweet_sidecar_root,
            )
            target_rows.append(
                build_comparison_row(
                    collection="combined",
                    partition=partition,
                    unmasked_resolution=combined_unmasked_resolution,
                    masked_resolution=combined_masked_resolution,
                    unmasked=combined_unmasked_stats,
                    masked=combined_masked_stats,
                )
            )

    group_prefix_rows = build_prefix_coverage_rows(group_rows)
    group_count_rows = build_count_statistics_rows(group_rows)
    topic_prefix_rows = build_prefix_coverage_rows(topic_rows)
    topic_count_rows = build_count_statistics_rows(topic_rows)

    write_prefix_coverage_csv(args.output_dir / "group_split_prefix_coverage_v4.csv", group_prefix_rows)
    write_count_statistics_csv(args.output_dir / "group_split_count_statistics_v4.csv", group_count_rows)
    write_prefix_coverage_csv(args.output_dir / "topic_split_prefix_coverage_v4.csv", topic_prefix_rows)
    write_count_statistics_csv(args.output_dir / "topic_split_count_statistics_v4.csv", topic_count_rows)

    for legacy_name in (
        "group_split_dataset_stats_v4.csv",
        "topic_split_dataset_stats_v4.csv",
    ):
        legacy_path = args.output_dir / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()

    summary_md = build_markdown_summary(
        result_version=args.result_version,
        human_filename=human_filename,
        group_rows=group_rows,
        topic_rows=topic_rows,
    )
    (args.output_dir / "README.md").write_text(summary_md, encoding="utf-8")


if __name__ == "__main__":
    main()