import json
from types import SimpleNamespace

import pytest

from pipelines import annotate
from pipelines.common import read_csv
from .conftest import synthetic_rows, write_session


def test_cached_judgments_and_template_coverage(session, tmp_path, monkeypatch):
    cache_path = tmp_path / "judge.jsonl"
    output = tmp_path / "labels.csv"
    calls = []

    def create(**payload):
        calls.append(payload)
        content = "Probably agree" if len(payload["messages"]) == 1 else "VALID"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    import openai
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    args = ["--input", str(session), "--output", str(output), "--cache", str(cache_path), "--source", "human"]
    annotate.main(args)
    assert calls
    assert all(item["model"] == "gpt-5-mini" and item["reasoning_effort"] == "medium" for item in calls)
    _, rows = read_csv(output)
    labeled = [row for row in rows if row["event_type"] in annotate.EVENTS]
    assert {row["human_validity"] for row in labeled} == {"VALID"}
    assert {row["likert_pred"] for row in labeled} == {"5"}
    expected = output.read_bytes()
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: pytest.fail("Cache-only replay must not initialize a provider"))
    annotate.main(args + ["--cache-only"])
    assert output.read_bytes() == expected
    metadata = json.loads(output.with_suffix(".metadata.json").read_text())
    assert metadata["unparseable_judgments"] == 0


def test_cache_miss_does_not_guess(tmp_path):
    cache = annotate.JudgmentCache(tmp_path / "empty.jsonl")
    with pytest.raises(ValueError, match="missing from the cache"):
        cache.request("gpt-5-mini", [{"role": "user", "content": "test"}], "medium", None)


@pytest.mark.parametrize("protocol", ["generated-text", "paper-v5"])
def test_llm_post_validity_protocol_is_explicit(protocol, session, tmp_path, monkeypatch):
    rows = synthetic_rows()
    for row in rows:
        row["llm_text"] = "GENERATED_FINAL_STANCE" if row["event_type"] == "Post Opinion" else "GENERATED_TEXT"
    write_session(session, rows)
    calls = []

    def create(**payload):
        calls.append(payload)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="VALID"))])

    import openai
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    annotate.main(["--input", str(session), "--output", str(tmp_path / "generated_labels.csv"),
                   "--cache", str(tmp_path / "generated_cache.jsonl"), "--source", "llm", "--task", "validity",
                   "--validity-protocol", protocol])
    post_calls = [item for item in calls if "# Post Opinion Validity Judge" in item["messages"][0]["content"]]
    assert post_calls
    expected, excluded = ("GENERATED_FINAL_STANCE", "HUMAN_POST_") if protocol == "generated-text" else ("HUMAN_POST_", "GENERATED_FINAL_STANCE")
    assert all(expected in item["messages"][1]["content"] for item in post_calls)
    assert all(excluded not in item["messages"][1]["content"] for item in post_calls)
    metadata = json.loads((tmp_path / "generated_labels.metadata.json").read_text())
    assert metadata["validity_protocol"] == protocol


def test_protocols_cannot_cross_reuse_cache_entries(tmp_path):
    calls = []
    def create(**payload):
        calls.append(payload)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="VALID"))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    cache = annotate.JudgmentCache(tmp_path / "protocols.jsonl")
    messages = [{"role": "user", "content": "same input"}]
    cache.request("gpt-5-mini", messages, "medium", client, "paper-v5")
    with pytest.raises(ValueError, match="missing from the cache"):
        cache.request("gpt-5-mini", messages, "medium", None, "generated-text")
    cache.request("gpt-5-mini", messages, "medium", client, "generated-text")
    assert len(calls) == 2
    assert all("validity_protocol" not in item for item in calls)
