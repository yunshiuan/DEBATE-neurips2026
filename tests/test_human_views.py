import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from debate.data import load_release, sha256
from debate.human_views import (CSV_FIELDS, HUMAN_FILENAME, _opinion, _resolve_participant,
                                _restore_flag, export_human, opinion_pairs, select_submission)


def event(record="a", value=1, auto=False, text="A human opinion", stage="Initial Opinion", order=1):
    return {"record_id": record, "source_file": "human/raw/topic/session.csv", "source_row": order,
            "session_id": "01JTGYE0TGE8E600TWCDF7TWPE", "session_prefix": "session", "topic": "Topic",
            "participant_id": "67e147TWPE", "group_split": "train", "topic_split": "test",
            "event_type": stage, "human_text": text, "self_reported_opinion": value,
            "self_reported_opinion_label": "", "is_auto_submitted": auto}


@pytest.mark.parametrize("manual_order,auto_order", [(1, 2), (2, 1)])
def test_manual_preference_ignores_event_order(manual_order, auto_order):
    result = select_submission([event("auto", 6, True, order=auto_order), event("manual", 2, False, order=manual_order)])
    assert result["record_id"] == "manual"
    assert result["self_reported_opinion"] == 2
    assert result["selection_reason"] == "manual_preferred"
    assert result["conflicting_ratings"] is True
    assert result["submission_count"] == 2
    assert set(json.loads(result["candidate_record_ids"])) == {"auto", "manual"}


def test_auto_only_missing_and_unresolved_submissions():
    automatic = select_submission([event(auto=True)])
    assert automatic["is_auto_submitted"] is True
    assert automatic["selection_reason"] == "single_submission"
    missing = select_submission([])
    assert missing["selection_reason"] == "missing_submission"
    assert missing["self_reported_opinion"] is None
    assert missing["candidate_record_ids"] == "[]"
    conflicting = select_submission([event("a", 1), event("b", 4)])
    assert conflicting["selection_reason"] == "unresolved_multiple_submissions"
    assert conflicting["self_reported_opinion"] is None
    assert conflicting["record_id"] is None
    assert conflicting["text"] is None
    assert conflicting["conflicting_ratings"] is True


def test_same_rating_different_text_is_not_arbitrarily_selected():
    result = select_submission([event("a", 1, text="One explanation"), event("b", 1, text="Another explanation")])
    assert result["selection_reason"] == "unresolved_multiple_submissions"
    assert result["conflicting_ratings"] is False
    assert result["text"] is None
    identical = select_submission([event("a", order=1), event("b", order=2)])
    assert identical["selection_reason"] == "identical_submissions"
    assert identical["record_id"] == "a"
    assert identical["submission_count"] == 2


def test_unknown_flag_is_not_treated_as_automatic():
    result = select_submission([event("a", 1, False), event("b", 4, None), event("c", 6, True)])
    assert result["selection_reason"] == "unresolved_multiple_submissions"
    assert result["self_reported_opinion"] is None


def test_pairs_keep_missing_and_compute_delta_only_for_resolved_numbers():
    rows = [event("a", 1), event("b", 4, stage="Post Opinion", order=2)]
    pair = opinion_pairs(rows)[0]
    assert pair["self_reported_opinion_change"] == 3
    rows[1]["self_reported_opinion"] = None
    pair = opinion_pairs(rows)[0]
    assert pair["post_record_id"] == "b"
    assert pair["post_text"] == "A human opinion"
    assert pair["self_reported_opinion_change"] is None
    assert _opinion("") is None
    with pytest.raises(ValueError, match="six-point"):
        _opinion("0")


def test_opinion_flag_join_requires_same_key_text_and_rating():
    row = {"worker_id": "worker", "event_type": "Initial Opinion", "event_order": "1.0", "text": "Human", "sliderValue": "2"}
    key = ("worker", "Initial Opinion", 1)
    source = dict(row, isAutosubmitted="False", event_order="1")
    assert _restore_flag(row, {key: [source]}) is False
    for candidates in ([], [source, source], [dict(source, text="Different")], [dict(source, sliderValue="4")]):
        with pytest.raises(ValueError, match="unique and exact"):
            _restore_flag(row, {key: candidates})


def test_participant_mapping_is_scoped_unambiguous_and_checks_session_suffix():
    session = "01JTGYE0TGE8E600TWCDF7TWPE"
    assert _resolve_participant("67746.0", {"67746"}, session) == "67746"
    assert _resolve_participant("67746.0", {"677467TWPE"}, session) == "677467TWPE"
    assert _resolve_participant("67e14", {"67e147TWPE"}, session) == "67e147TWPE"
    assert _resolve_participant("67e147TWPE", {"67e14"}, session) == "67e14"
    with pytest.raises(ValueError, match="uniquely"):
        _resolve_participant("67e14", {"67e14AAAAA", "67e14BBBBB"}, session)
    with pytest.raises(ValueError, match="uniquely"):
        _resolve_participant("67e14XXXXX", {"67e14"}, session)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture
def release(tmp_path):
    root = tmp_path / "data"
    prefix = "20250505_220024_Topic_01JTGYE0TGE8E600TWCDF7TWPE"
    def row(order, stage, text="Human", slider="", **extra):
        return {"chat_round_order": "", "event_order": str(order), "event_type": stage,
                "worker_id": "67e147TWPE", "sender_id": "", "recipient_id": "", "field": "",
                "text": text, "sliderValue": slider, "agreement_level": "MODEL_RESPONSE_DO_NOT_EXPORT", "validity": "VALID", **extra}
    manual = row(1, "Initial Opinion", "Manual human text", "1.0")
    automatic = row(2, "Initial Opinion", "Automatic human text", "6.0")
    post = row(3, "Post Opinion", "Post human text", "4.0")
    tweet = row(4, "tweet", "Human tweet")
    survey = row(5, "exit_survey", "survey answer", field="example")
    # Third survey differs only in an omitted legacy column: it must not collapse.
    rows = [manual, automatic, post, tweet, survey, survey, dict(survey, agreement_level="DIFFERENT_MODEL_METADATA")]
    raw = root / "human/raw/breadth/Topic" / (prefix + ".csv")
    full = root / "reproduction/inputs/llama/data/sft_data/group_split_data_bd/train" / (prefix + ".csv")
    write_csv(raw, rows)
    write_csv(full, [dict(row, isAutosubmitted="True" if row["event_order"] == "2" else "False") for row in rows])
    masks = root / "reproduction/inputs/llama/data/sft_data/masks"
    masks.mkdir(parents=True)
    mask_paths = []
    for split in ("group", "topic"):
        for part in ("train", "test"):
            path = masks / f"{split}_{part}_prefixes.txt"
            path.write_text(prefix + "\n" if part == "train" else "")
            mask_paths.append(path)
    canonical = root / "reproduction/inputs/base/human_llm" / prefix / "Llama-3.1-8B-Instruct" / HUMAN_FILENAME
    write_csv(canonical, [dict(row, worker_id="67e14", human_validity="VALID", likert_pred=stance, label_pred=label,
                               isAutosubmitted="False", llm_text="MODEL_TEXT_DO_NOT_EXPORT", input_prompt="MODEL_PROMPT_DO_NOT_EXPORT")
                          for row, stance, label in ((manual, "1", "Certainly disagree"), (post, "0", "NA"), (tweet, "3", "Lean disagree"))])
    sidecar = root / "reproduction/inputs/validity/initial_post" / prefix / "Llama-3.1-8B-Instruct" / (Path(HUMAN_FILENAME).stem + "__initial_post_rejudge.csv")
    # Sidecar indexes include the header: post is second data row -> row_index3.
    write_csv(sidecar, [{"row_index": "3", "new_validity": "INVALID"}])
    meta = root / ".debate"
    (meta / "manifests").mkdir(parents=True)
    pin = load_release()
    (meta / "release.json").write_text(json.dumps({"dataset_repo": pin["dataset_repo"], "revision": pin["revision"], "release": "test-fixture", "source": "local"}))
    for component, paths in (("human_inputs", [raw, full, *mask_paths]), ("base_labels", [canonical]), ("validity_labels", [sidecar])):
        (meta / "manifests" / (component + ".json")).write_text(json.dumps([
            {"path": path.relative_to(root).as_posix(), "sha256": sha256(path), "size_bytes": path.stat().st_size} for path in paths]))
    return SimpleNamespace(root=root, raw=raw, canonical=canonical, sidecar=sidecar,
                           output=tmp_path / "views", args=SimpleNamespace(data_root=root, output_dir=tmp_path / "views", overwrite=False))


def test_export_human_content_labels_and_lineage(release):
    counts = export_human(release.args)
    assert counts["human_event_rows"] == 6
    assert counts["removed_exact_survey_copies"] == 1
    assert counts["paper_rows"] == 3
    rows = read_csv(release.output / "human_events.csv")
    surveys = [row for row in rows if row["event_type"] == "exit_survey"]
    assert [row["duplicate_count"] for row in surveys] == ["2", "1"]
    assert [row["source_row"] for row in surveys] == ["5", "7"]
    assert len([row for row in rows if row["event_type"] == "Initial Opinion"]) == 2
    assert rows[0]["is_auto_submitted"] == "false"
    assert rows[1]["is_auto_submitted"] == "true"
    for name, fields in CSV_FIELDS.items():
        csv_rows = read_csv(release.output / name)
        assert list(csv_rows[0]) == fields
        text = (release.output / name).read_text()
        assert "MODEL_RESPONSE_DO_NOT_EXPORT" not in text
        assert "MODEL_TEXT_DO_NOT_EXPORT" not in text
        assert "MODEL_PROMPT_DO_NOT_EXPORT" not in text
        assert str(release.root) not in text
    paper = read_csv(release.output / "paper_human_evaluation.csv")
    assert paper[1]["annotated_stance"] == "0"
    assert paper[1]["annotated_stance_label"] == "NA"
    assert paper[1]["human_validity"] == "INVALID"
    assert paper[1]["stance_in_paper_inventory"] == "false"
    assert paper[1]["self_reported_opinion"] == "4"
    assert paper[0]["participant_id"] == "67e147TWPE"
    assert counts["paper_inventory"]["group_train"]["opinions"] == {"rows": 2, "on_topic": 1}
    pair = read_csv(release.output / "human_opinion_pairs.csv")[0]
    assert pair["self_reported_opinion_change"] == "3"
    assert pair["pre_selection_reason"] == "manual_preferred"
    manifest = json.loads((release.output / "manifest.json").read_text())
    for item in manifest["files"]:
        assert sha256(release.output / item["path"]) == item["sha256"]
    assert any(item["path"].endswith("__initial_post_rejudge.csv") for item in manifest["source_files"])


def test_export_is_deterministic_and_explicit_overwrite_is_safe(release):
    export_human(release.args)
    original = {path.name: path.read_bytes() for path in release.output.iterdir()}
    with pytest.raises(ValueError, match="not empty"):
        export_human(release.args)
    release.args.overwrite = True
    export_human(release.args)
    assert {path.name: path.read_bytes() for path in release.output.iterdir()} == original
    (release.output / "personal_note.txt").write_text("Keep me")
    with pytest.raises(ValueError, match="no extra files"):
        export_human(release.args)
    assert (release.output / "personal_note.txt").read_text() == "Keep me"
    release.args.output_dir = release.root / "human_views"
    with pytest.raises(ValueError, match="separate"):
        export_human(release.args)


@pytest.mark.parametrize("which", ["canonical", "sidecar", "raw"])
def test_missing_manifest_listed_inputs_cannot_silently_change_population(release, which):
    getattr(release, which).unlink()
    with pytest.raises(ValueError, match="Missing or modified|coverage"):
        export_human(release.args)
    assert not release.output.exists()


def test_modified_source_and_wrong_revision_are_rejected(release):
    original = release.raw.read_bytes()
    release.raw.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="Missing or modified"):
        export_human(release.args)
    release.raw.write_bytes(original)
    meta = release.root / ".debate/release.json"
    document = json.loads(meta.read_text())
    document["revision"] = "PENDING_RELEASE_COMMIT"
    meta.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="pinned public"):
        export_human(release.args)


@pytest.mark.parametrize("variable", ["V3_COMPLETE_GROUP_MASK_ROOT", "V3_TWEET_GROUP_MASK_ROOT", "V3_OPINION_GROUP_MASK_ROOT"])
def test_external_group_mask_overrides_are_rejected(release, variable):
    with patch.dict(os.environ, {variable: "/arbitrary/private/masks"}):
        with pytest.raises(ValueError, match="optional"):
            export_human(release.args)
    assert not release.output.exists()
