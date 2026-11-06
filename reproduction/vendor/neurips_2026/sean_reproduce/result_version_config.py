from __future__ import annotations

import os
from typing import Final


RESULT_NAMESPACE_TO_SIMULATION_VERSION: Final[dict[str, str]] = {
    "v3": "v0",
    "v3_b": "v1_5",
    "v4": "v0",
    "v4_b": "v1_5",
}

LLAMA_V4_BREADTH_TOPIC_SFT_MODEL: Final[str] = "ft:Llama-3.1-8B-Instruct-SFT-20260430:breadth-topic-3epochs-unsloth"
LLAMA_V4_BREADTH_GROUP_SFT_MODEL: Final[str] = "ft:Llama-3.1-8B-Instruct-SFT-20260430:breadth-group-3epochs-unsloth"
LLAMA_V4_DEPTH_TOPIC_SFT_MODEL: Final[str] = "ft:Llama-3.1-8B-Instruct-SFT-20260501:depth-topic-3epochs-unsloth"
LLAMA_V4_DEPTH_GROUP_SFT_MODEL: Final[str] = "ft:Llama-3.1-8B-Instruct-SFT-20260501:depth-group-3epochs-unsloth"

V4_FAMILY_SFT_MODELS: Final[dict[tuple[str, str], str]] = {
    ("breadth", "topic"): LLAMA_V4_BREADTH_TOPIC_SFT_MODEL,
    ("breadth", "group"): LLAMA_V4_BREADTH_GROUP_SFT_MODEL,
    ("depth", "topic"): LLAMA_V4_DEPTH_TOPIC_SFT_MODEL,
    ("depth", "group"): LLAMA_V4_DEPTH_GROUP_SFT_MODEL,
}


def canonical_result_namespace(result_version: str) -> str:
    namespace = str(result_version).strip()
    if namespace not in RESULT_NAMESPACE_TO_SIMULATION_VERSION:
        expected = ", ".join(sorted(RESULT_NAMESPACE_TO_SIMULATION_VERSION))
        raise ValueError(
            f"Unsupported result namespace {result_version!r}; expected one of: {expected}"
        )
    return namespace


def has_canonical_result_namespace(result_version: str) -> bool:
    return str(result_version).strip() in RESULT_NAMESPACE_TO_SIMULATION_VERSION


def report_namespace_for_version(report_version: str) -> str:
    namespace = str(report_version).strip()
    if namespace.startswith("v4_b"):
        return "v4_b"
    if namespace.startswith("v4"):
        return "v4"
    if namespace.startswith("v3_b"):
        return "v3_b"
    if namespace.startswith("v3"):
        return "v3"
    return canonical_result_namespace(namespace)


def report_family_for_version(report_version: str) -> str:
    return "v4" if report_namespace_for_version(report_version).startswith("v4") else "v3"


def include_close_far_transitions(result_version: str) -> bool:
    return report_namespace_for_version(result_version) not in {"v4", "v4_b"}


def simulation_version_for_result(result_version: str) -> str:
    return RESULT_NAMESPACE_TO_SIMULATION_VERSION[canonical_result_namespace(result_version)]


def human_csv_filename_for_result(result_version: str) -> str:
    return f"opinion_human_memory_gpt-5-mini_{simulation_version_for_result(result_version)}.csv"


def llm_csv_filename_for_result(result_version: str) -> str:
    return f"opinion_llm_memory_gpt-5-mini_{simulation_version_for_result(result_version)}.csv"


def apply_transition_simulation_version_for_result(result_version: str) -> str | None:
    if not has_canonical_result_namespace(result_version):
        return None
    simulation_version = simulation_version_for_result(result_version)
    os.environ["TRANSITION_SIMULATION_VERSION"] = simulation_version
    return simulation_version


def v4_family_sft_model(collection_type: str, split_type: str) -> str:
    key = (str(collection_type).strip(), str(split_type).strip())
    if key not in V4_FAMILY_SFT_MODELS:
        expected = ", ".join(f"{collection}/{split}" for collection, split in sorted(V4_FAMILY_SFT_MODELS))
        raise ValueError(
            f"Unsupported v4 family split {collection_type!r}/{split_type!r}; expected one of: {expected}"
        )
    return V4_FAMILY_SFT_MODELS[key]