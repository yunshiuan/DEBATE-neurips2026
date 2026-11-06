from __future__ import annotations

CANONICAL_V3_HUMAN_SOURCE_MODEL = "Llama-3.1-8B-Instruct"
TULU_BASE_MODEL = "Llama-3.1-Tulu-3-8B-SFT"

BASE_MODEL_V3_LLM_ORDER = [
    "gpt-5-mini",
    "gpt-4o-mini-2024-07-18",
    "gemini-2.5-flash-lite",
    "Mistral-7B-Instruct-v0.3",
    "Qwen2.5-32B-Instruct",
    "Llama-3.1-70B-Instruct",
    "Llama-3.1-8B-Instruct",
    TULU_BASE_MODEL,
]

BASE_MODEL_V4_LLM_ORDER = [model_name for model_name in BASE_MODEL_V3_LLM_ORDER if model_name != TULU_BASE_MODEL]

BASE_MODEL_V3_COLOR_MAP = {
    "gpt-4o-mini-2024-07-18": "#E89B93",
    "gpt-5-mini": "#D97F73",
    "Llama-3.1-Tulu-3-8B-SFT": "#DDE7F5",
    "Llama-3.1-8B-Instruct": "#C8D4E9",
    "Llama-3.1-70B-Instruct": "#9EAAD1",
    "Qwen2.5-32B-Instruct": "#B9A2E3",
    "Mistral-7B-Instruct-v0.3": "#E8D2AE",
    "gemini-2.5-flash-lite": "#9ECDBF",
    "human": "#222222",
}

BASE_MODEL_V3_DISPLAY_LABELS = {
    "gpt-5-mini": "gpt-5-mini",
    "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
    "gemini-2.5-flash-lite": "Gemini-2.5-Flash-Lite",
    "Mistral-7B-Instruct-v0.3": "Mistral-7B",
    "Qwen2.5-32B-Instruct": "Qwen2.5-32B",
    "Llama-3.1-70B-Instruct": "Llama-3.1-70B",
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "Llama-3.1-Tulu-3-8B-SFT": "Tulu-3-8B",
    "human": "Human",
}


def ordered_base_model_v3_llms(allowed_models: list[str] | None = None) -> list[str]:
    if allowed_models is None:
        return list(BASE_MODEL_V3_LLM_ORDER)
    allowed_set = set(allowed_models)
    return [model_name for model_name in BASE_MODEL_V3_LLM_ORDER if model_name in allowed_set]


def ordered_base_model_llms_for_report(report_version: str, allowed_models: list[str] | None = None) -> list[str]:
    base_order = BASE_MODEL_V4_LLM_ORDER if str(report_version).strip().startswith("v4") else BASE_MODEL_V3_LLM_ORDER
    if allowed_models is None:
        return list(base_order)
    allowed_set = set(allowed_models)
    ordered = [model_name for model_name in base_order if model_name in allowed_set]
    for model_name in allowed_models:
        if model_name in BASE_MODEL_V3_LLM_ORDER or model_name in ordered:
            continue
        ordered.append(model_name)
    return ordered
