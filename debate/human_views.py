"""Export clearly named human records from the verified public data release.

These additive views never replace the archived inputs used by the paper.
The pre/post selection policy is a convenience policy for new analyses.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

from .data import load_release, manifest_files, safe_path, sha256

POLICY_VERSION = "human-views-v1"
COMPONENTS = ("human_inputs", "base_labels", "validity_labels")
OPINIONS = {"Initial Opinion", "Post Opinion"}
LABELS = {1: "Certainly disagree", 2: "Probably disagree", 3: "Lean disagree",
          4: "Lean agree", 5: "Probably agree", 6: "Certainly agree"}
HUMAN_FILENAME = "opinion_human_memory_gpt-5-mini_v0.csv"
EVENT_FIELDS = ["record_id", "session_id", "session_prefix", "topic", "participant_id",
                "sender_id", "recipient_id", "event_type", "event_order", "chat_round_order",
                "field", "human_text", "self_reported_opinion", "self_reported_opinion_label",
                "is_auto_submitted", "group_split", "topic_split", "source_file", "source_row", "duplicate_count"]
PAPER_FIELDS = EVENT_FIELDS + ["annotated_stance", "annotated_stance_label", "human_validity",
                              "annotation_model", "stance_in_paper_inventory"]
PAIR_ID_FIELDS = ["session_id", "session_prefix", "topic", "participant_id", "group_split", "topic_split"]
STAGE_FIELDS = ["record_id", "text", "self_reported_opinion", "self_reported_opinion_label",
                "is_auto_submitted", "submission_count", "conflicting_ratings", "selection_reason", "candidate_record_ids"]
PAIR_FIELDS = PAIR_ID_FIELDS + [stage + "_" + field for stage in ("pre", "post") for field in STAGE_FIELDS] + ["self_reported_opinion_change"]
CSV_FIELDS = {"human_events.csv": EVENT_FIELDS, "paper_human_evaluation.csv": PAPER_FIELDS,
              "human_opinion_pairs.csv": PAIR_FIELDS}


def _read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value):
    if value is None or str(value).strip() == "":
        return None
    try:
        number = Decimal(str(value).strip())
    except InvalidOperation as exc:
        raise ValueError(f"Expected a number, got {value!r}") from exc
    if not number.is_finite():
        raise ValueError(f"Expected a finite number, got {value!r}")
    return int(number) if number == number.to_integral() else float(number)


def _opinion(value):
    number = _number(value)
    if number is not None and number not in LABELS:
        raise ValueError(f"Self-reported opinion outside the six-point scale: {value!r}")
    return number


def _boolean(value):
    value = str(value or "").strip().lower()
    if value == "":
        return None
    if value in {"true", "1"}:
        return True
    if value in {"false", "0"}:
        return False
    raise ValueError(f"Unexpected automatic-submission flag: {value!r}")


def _event(value):
    value = str(value or "").strip()
    return "message_received" if value == "message_recieved" else value


def _key(row, participant=None):
    return (row.get("worker_id", "") if participant is None else participant,
            _event(row.get("event_type")), _number(row.get("event_order")))


def _record_id(kind, source_file, source_row):
    value = f"{kind}\n{source_file}\n{source_row}".encode("utf-8")
    return kind + "-" + hashlib.sha256(value).hexdigest()[:32]


class VerifiedInputs:
    """Verify each consumed source against installed public component manifests."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.source = json.loads(safe_path(self.root, ".debate/release.json").read_text())
        pin = load_release()
        if (self.source.get("dataset_repo") != pin["dataset_repo"]
                or self.source.get("revision") != pin["revision"]
                or not re.fullmatch(r"[0-9a-f]{40}", self.source.get("revision", ""))):
            raise ValueError("Human views require the pinned public data revision. Run 'debate download' in a fresh data directory.")
        self.catalog = {}
        self.component_manifests = []
        for component in COMPONENTS:
            path = safe_path(self.root, f".debate/manifests/{component}.json")
            if not path.is_file():
                raise FileNotFoundError(f"Missing {component}; run 'debate download --components {' '.join(COMPONENTS)}'.")
            self.component_manifests.append({"component": component, "sha256": sha256(path)})
            for item in manifest_files(json.loads(path.read_text())):
                name = item["path"]
                safe_path(self.root, name)
                if name in self.catalog and self.catalog[name] != item:
                    raise ValueError(f"Conflicting installed manifests for {name}")
                self.catalog[name] = item
        self.consumed = {}

    def check(self, path):
        path = Path(path)
        relative = path.relative_to(self.root).as_posix()
        path = safe_path(self.root, relative)
        if relative not in self.catalog:
            raise ValueError(f"Source not in the installed release manifests: {relative}")
        spec = self.catalog[relative]
        if relative not in self.consumed:
            if (not path.is_file() or path.stat().st_size != spec["size_bytes"] or sha256(path) != spec["sha256"]):
                raise ValueError(f"Missing or modified release input: {relative}")
            self.consumed[relative] = {key: spec[key] for key in ("path", "sha256", "size_bytes")}
        return path

    def rows(self, path):
        return _read_csv(self.check(path))


def _helpers():
    # The same resolver and validity code used by reproduction.tables.inventory.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "debate-matplotlib"))
    vendor = Path(__file__).resolve().parents[1] / "reproduction/vendor"
    scripts = vendor / "neurips_2026/sean_reproduce/scripts"
    for path in (vendor, scripts):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    stats = importlib.import_module("generate_v4_dataset_stats_tables")
    radar = importlib.import_module("neurips_2026.chengtao.src.radar.generate_plots")
    return stats, radar


def _splits(inputs):
    masks = inputs.root / "reproduction/inputs/llama/data/sft_data/masks"
    splits = {}
    for split in ("group", "topic"):
        membership = {}
        for partition in ("train", "test"):
            path = inputs.check(masks / f"{split}_{partition}_prefixes.txt")
            for prefix in path.read_text().splitlines():
                prefix = prefix.strip()
                if not prefix:
                    continue
                if prefix in membership:
                    raise ValueError(f"Repeated split membership: {split}/{prefix}")
                membership[prefix] = partition
        splits[split] = membership
    if set(splits["group"]) != set(splits["topic"]):
        raise ValueError("Group and topic partitions do not cover the same sessions")
    return splits


def _resolve_participant(value, participants, session_id):
    value = str(value or "").strip()
    if not value or value in participants:
        return value
    # Some legacy CSVs serialize an all-digit shortened ID as a float.
    if re.fullmatch(r"[0-9]+\.0+", value):
        value = value.split(".", 1)[0]
        if value in participants:
            return value
    matches = [worker for worker in participants
               if (len(value) == 5 and worker[:5] == value)
               or (len(worker) == 5 and value == worker + session_id[-5:])]
    if len(matches) != 1:
        raise ValueError(f"Cannot map participant ID {value!r} uniquely within this session")
    return matches[0]


def _human_row(row, *, source, source_row, prefix, topic, splits, kind, participants=None):
    opinion = _opinion(row.get("sliderValue"))
    sid = prefix.rsplit("_", 1)[-1]
    if not re.fullmatch(r"[0-9A-HJKMNP-TV-Z]{26}", sid):
        raise ValueError(f"Invalid session ID in {prefix}")
    resolve = (lambda value: _resolve_participant(value, participants, sid)) if participants is not None else (lambda value: str(value or ""))
    return {"record_id": _record_id(kind, source, source_row), "session_id": sid,
            "session_prefix": prefix, "topic": topic, "participant_id": resolve(row.get("worker_id")),
            "sender_id": resolve(row.get("sender_id")), "recipient_id": resolve(row.get("recipient_id")),
            "event_type": _event(row.get("event_type")), "event_order": _number(row.get("event_order")),
            "chat_round_order": _number(row.get("chat_round_order")), "field": row.get("field", ""),
            "human_text": row.get("text", ""), "self_reported_opinion": opinion,
            "self_reported_opinion_label": LABELS.get(opinion, ""),
            "is_auto_submitted": _boolean(row.get("isAutosubmitted")),
            "group_split": splits["group"][prefix], "topic_split": splits["topic"][prefix],
            "source_file": source, "source_row": source_row, "duplicate_count": 1}


def _restore_flag(row, source_rows):
    matches = source_rows.get(_key(row), [])
    exact = [candidate for candidate in matches
             if candidate.get("text", "") == row.get("text", "")
             and _opinion(candidate.get("sliderValue")) == _opinion(row.get("sliderValue"))]
    if _event(row.get("event_type")) in OPINIONS and (len(matches) != 1 or len(exact) != 1):
        raise ValueError(f"Human opinion metadata join is not unique and exact: {_key(row)!r}")
    return _boolean(exact[0].get("isAutosubmitted")) if len(exact) == 1 else None


def _raw_events(inputs, splits):
    source_root = inputs.root / "reproduction/inputs/llama/data/sft_data/group_split_data_bd"
    source_paths = {}
    for path in sorted(source_root.glob("*/*.csv")):
        if path.stem in source_paths:
            raise ValueError(f"Duplicate full input session: {path.stem}")
        source_paths[path.stem] = path
    output, sessions = [], {}
    raw_rows_count = 0
    for path in sorted((inputs.root / "human/raw").glob("*/*/*.csv")):
        prefix = path.stem
        if prefix in sessions:
            raise ValueError(f"Duplicate raw session: {prefix}")
        if prefix not in splits["group"] or prefix not in source_paths:
            raise ValueError(f"Session lacks split or full public inputs: {prefix}")
        rows = inputs.rows(path)
        metadata = defaultdict(list)
        for source_row in inputs.rows(source_paths[prefix]):
            metadata[_key(source_row)].append(source_row)
        relative = path.relative_to(inputs.root).as_posix()
        session = {"topic": path.parent.name, "participants": {row["worker_id"] for row in rows if row.get("worker_id")}, "rows": rows}
        sessions[prefix] = session
        surveys = {}
        for index, row in enumerate(rows, 1):
            raw_rows_count += 1
            # Compare the entire original row before projecting away legacy fields.
            fingerprint = tuple(row.items())
            if row.get("event_type") == "exit_survey" and fingerprint in surveys:
                surveys[fingerprint]["duplicate_count"] += 1
                continue
            item = _human_row(row, source=relative, source_row=index, prefix=prefix, topic=session["topic"], splits=splits, kind="event")
            item["is_auto_submitted"] = _restore_flag(row, metadata)
            output.append(item)
            if row.get("event_type") == "exit_survey":
                surveys[fingerprint] = item
    if set(sessions) != set(splits["group"]):
        raise ValueError("Raw human sessions differ from released split coverage")
    return output, sessions, raw_rows_count


def _paper_events(inputs, splits, sessions):
    stats, radar = _helpers()
    base = inputs.root / "reproduction/inputs/base/human_llm"
    initial = inputs.root / "reproduction/inputs/validity/initial_post"
    tweet = inputs.root / "reproduction/inputs/validity/tweet"
    # A genuinely absent source is allowed, but a deleted installed source is not.
    for relative in sorted(inputs.catalog):
        if relative.startswith("reproduction/inputs/base/human_llm/") and relative.endswith("/" + HUMAN_FILENAME):
            inputs.check(inputs.root / relative)
    resolution = stats.resolve_available_prefixes(sorted(sessions), eval_root=base, human_filename=HUMAN_FILENAME)
    output = []
    for prefix in resolution.available_prefixes:
        path = stats.resolve_human_csv_path(base, prefix, HUMAN_FILENAME)
        rows = inputs.rows(path)
        overrides = {}
        for name, sidecar_root, suffix in (("opinion", initial, "__initial_post_rejudge.csv"), ("tweet", tweet, "__tweet_rejudge.csv")):
            sidecar = radar.get_sidecar_csv_path(path, [base], sidecar_root, suffix)
            if sidecar and (sidecar.exists() or sidecar.relative_to(inputs.root).as_posix() in inputs.catalog):
                inputs.check(sidecar)
            overrides[name] = radar.load_sidecar_validity_overrides(path, [base], [sidecar_root], suffix)
        adjusted = radar.apply_validity_overrides(rows, path=path, likert_col="likert_pred", data_roots=[base],
                                                initial_post_sidecar_roots=[initial], tweet_sidecar_roots=[tweet])
        marked = [dict(row, _human_view_source_row=index) for index, row in enumerate(adjusted, 1)]
        kept = radar.filter_rows_for_v3_complete_group_mask(marked, csv_path=path)
        retained = {row["_human_view_source_row"] for row in kept}
        session = sessions[prefix]
        raw_index = defaultdict(list)
        for raw in session["rows"]:
            raw_index[_key(raw)].append(raw)
        relative = path.relative_to(inputs.root).as_posix()
        for index, row in enumerate(rows, 1):
            if index not in retained:
                continue
            item = _human_row(row, source=relative, source_row=index, prefix=prefix, topic=session["topic"], splits=splits,
                              kind="paper", participants=session["participants"])
            matches = raw_index[_key(row, item["participant_id"])]
            if not matches or not any(_opinion(match.get("sliderValue")) == item["self_reported_opinion"] for match in matches):
                raise ValueError(f"Canonical human row cannot be traced to raw data: {relative}:{index}")
            column = radar.determine_validity_column(path, row)
            validity = str(row.get(column) or row.get("validity") or "").strip().upper() if column else ""
            event = row.get("event_type")
            if event in OPINIONS:
                validity = overrides["opinion"].get(index + 1, validity)
            elif event == "tweet":
                validity = overrides["tweet"].get(index + 1, validity)
            item.update(annotated_stance=_number(row.get("likert_pred")), annotated_stance_label=row.get("label_pred", ""),
                        human_validity=validity, annotation_model="gpt-5-mini",
                        stance_in_paper_inventory=bool(str(adjusted[index - 1].get("likert_pred") or "").strip()))
            output.append(item)
    return output, resolution


def select_submission(rows):
    """Prefer manual over automatic; leave competing nonidentical rows unresolved."""
    rows = sorted(rows, key=lambda row: (row["source_file"], row["source_row"]))
    result = {field: None for field in STAGE_FIELDS}
    result.update(submission_count=len(rows), conflicting_ratings=len({row["self_reported_opinion"] for row in rows if row["self_reported_opinion"] is not None}) > 1,
                  candidate_record_ids=json.dumps([row["record_id"] for row in rows], separators=(",", ":")))
    if not rows:
        result["selection_reason"] = "missing_submission"
        return result
    has_manual = any(row["is_auto_submitted"] is False for row in rows)
    has_auto = any(row["is_auto_submitted"] is True for row in rows)
    preferred = has_manual and has_auto
    candidates = [row for row in rows if row["is_auto_submitted"] is not True] if preferred else rows
    identities = {(row["human_text"], row["self_reported_opinion"], row["is_auto_submitted"]) for row in candidates}
    if len(identities) != 1:
        result["selection_reason"] = "unresolved_multiple_submissions"
        return result
    selected = candidates[0]
    result.update(record_id=selected["record_id"], text=selected["human_text"],
                  self_reported_opinion=selected["self_reported_opinion"], self_reported_opinion_label=selected["self_reported_opinion_label"],
                  is_auto_submitted=selected["is_auto_submitted"],
                  selection_reason="manual_preferred" if preferred else "single_submission" if len(rows) == 1 else "identical_submissions")
    return result


def opinion_pairs(events):
    groups = defaultdict(list)
    for row in events:
        if row["participant_id"]:
            groups[(row["session_id"], row["participant_id"])].append(row)
    pairs = []
    for key in sorted(groups):
        rows = groups[key]
        pair = {field: rows[0][field] for field in PAIR_ID_FIELDS}
        for stage, event_type in (("pre", "Initial Opinion"), ("post", "Post Opinion")):
            selection = select_submission([row for row in rows if row["event_type"] == event_type])
            pair.update({stage + "_" + field: value for field, value in selection.items()})
        pre, post = pair["pre_self_reported_opinion"], pair["post_self_reported_opinion"]
        pair["self_reported_opinion_change"] = post - pre if pre is not None and post is not None else None
        pairs.append(pair)
    return pairs


def schema_document():
    descriptions = {
        "record_id": "Stable ID derived from the view, relative source file and 1-based data row.",
        "session_id": "26-character session identifier; identifies the group interaction.",
        "session_prefix": "Original source filename stem used by the released split masks.",
        "topic": "Original topic statement from the raw source directory, without slug normalization.",
        "participant_id": "Pseudonymous participant ID scoped to this session; use with session_id.",
        "sender_id": "Pseudonymous message sender ID within the session, when present.",
        "recipient_id": "Pseudonymous message recipient ID within the session, when present.",
        "event_type": "Original event type; the legacy message_recieved spelling is normalized to message_received.",
        "event_order": "Numeric event order within the session; not a globally unique key.",
        "chat_round_order": "Recorded chat round, when present; canonical evaluation opinion rows retain rounds 0 and 4.",
        "field": "Original field name, including survey field names.",
        "human_text": "Human-authored text or survey response from the indicated source row. Canonical evaluation chat text may aggregate messages.",
        "self_reported_opinion": "Participant's own recorded slider response on the integer 1–6 agreement scale; blank if absent. Distinct from annotated stance.",
        "self_reported_opinion_label": "Label derived solely from self_reported_opinion: 1 Certainly disagree, 2 Probably disagree, 3 Lean disagree, 4 Lean agree, 5 Probably agree, 6 Certainly agree.",
        "is_auto_submitted": "True for automatic submission, false for explicit submission, blank when unknown. Event-view opinion flags are joined exactly from public full inputs.",
        "group_split": "train or test membership in the released group-generalization partition.",
        "topic_split": "train or test membership in the released topic-generalization partition.",
        "source_file": "Source CSV path relative to the extracted public data root.",
        "source_row": "1-based data-row number in source_file; excludes the CSV header.",
        "duplicate_count": "Number of identical original survey copies represented, including the retained first copy. Other events and all paper rows have 1.",
        "annotated_stance": "Cached text-classified stance (likert_pred), preserved numerically without validity filtering. Values 1–6 correspond to stance labels; cached 0 with NA is unassigned, not an additional response category.",
        "annotated_stance_label": "Unchanged cached text-classified stance label (label_pred); not a participant self-report.",
        "human_validity": "Effective cached validity for human text, using the paper's sidecar-over-inline rules: VALID, INVALID, or blank when absent.",
        "annotation_model": "Stance-annotation model associated with this cached human evaluation file (gpt-5-mini).",
        "stance_in_paper_inventory": "Whether a cached stance remains present after the paper's effective validity filtering; reproduces its inventory on-topic counts, without changing annotated_stance.",
        "self_reported_opinion_change": "Selected post self-report minus selected pre self-report; blank unless both are resolved numeric ratings.",
        "text": "Human text from the same selected source row as the selected rating; blank when unresolved or missing.",
        "submission_count": "Number of original submissions for this stage, before convenience selection.",
        "conflicting_ratings": "True when distinct nonmissing numeric ratings occur among all stage submissions, including a nonselected automatic submission.",
        "selection_reason": "Convenience selection outcome: missing_submission, manual_preferred, single_submission, identical_submissions, or unresolved_multiple_submissions. This is not the historical paper selection rule.",
        "candidate_record_ids": "JSON array of all stage record IDs in human_events.csv, including candidates not selected.",
    }
    integers = {"self_reported_opinion", "source_row", "duplicate_count", "submission_count", "self_reported_opinion_change"}
    numbers = {"event_order", "chat_round_order", "annotated_stance"}
    booleans = {"is_auto_submitted", "conflicting_ratings", "stance_in_paper_inventory"}
    required = {"record_id", "session_id", "session_prefix", "topic", "event_type", "group_split", "topic_split", "source_file", "source_row", "duplicate_count", "annotation_model", "stance_in_paper_inventory", "submission_count", "conflicting_ratings", "selection_reason", "candidate_record_ids"}
    views = {}
    for name, fields in CSV_FIELDS.items():
        specs = []
        for name_field in fields:
            base = name_field.split("_", 1)[1] if name_field.startswith(("pre_", "post_")) else name_field
            description = descriptions[base]
            if name_field.startswith(("pre_", "post_")):
                description = ("Initial Opinion: " if name_field.startswith("pre_") else "Post Opinion: ") + description
                if base == "record_id":
                    description = "Selected human_events.csv record ID; blank when missing or unresolved."
            nullable = base not in required or (name_field.startswith(("pre_", "post_")) and base == "record_id")
            specs.append({"name": name_field, "type": "integer" if base in integers else "number" if base in numbers else "boolean" if base in booleans else "string", "nullable": nullable, "description": description})
        views[name] = {"description": {"human_events.csv": "Human event records with exact duplicate survey copies removed; distinct opinion submissions retained.",
                                             "paper_human_evaluation.csv": "Canonical human evaluation inventory with distinct self-report and cached text-annotation fields.",
                                             "human_opinion_pairs.csv": "Convenience pre/post self-report pairs; manual submissions take precedence over automatic submissions and unresolved competing records remain missing."}[name],
                       "primary_key": ["session_id", "participant_id"] if name == "human_opinion_pairs.csv" else ["record_id"], "fields": specs}
    return {"schema_version": 1, "missing_values": "Empty CSV cells represent missing values; booleans use true/false.", "views": views}


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _output_location(data_root, output_dir, overwrite):
    target = Path(output_dir).absolute()
    if any(path.is_symlink() for path in [target, *target.parents]):
        raise ValueError("Refusing symlinks in the output path")
    target = target.resolve()
    root = Path(data_root).resolve()
    if target.is_relative_to(root) or root.is_relative_to(target):
        raise ValueError("Output must be separate from the extracted data directory")
    if target.exists() and not target.is_dir():
        raise ValueError("Output path is not a directory")
    if target.exists() and any(target.iterdir()):
        if not overwrite:
            raise ValueError("Output directory is not empty; use --overwrite for an earlier human-view export")
        manifest = target / "manifest.json"
        allowed = {*CSV_FIELDS, "schema.json", "manifest.json"}
        if (not manifest.is_file() or json.loads(manifest.read_text()).get("policy_version") != POLICY_VERSION
                or {path.name for path in target.iterdir()} != allowed
                or any(not path.is_file() or path.is_symlink() for path in target.iterdir())):
            raise ValueError("--overwrite only replaces a complete previous human-view export with no extra files")
    return target


def export_human(args):
    for variable in ("V3_COMPLETE_GROUP_MASK_ROOT", "V3_TWEET_GROUP_MASK_ROOT", "V3_OPINION_GROUP_MASK_ROOT"):
        if os.environ.get(variable, "").strip():
            raise ValueError(f"Human views require the canonical inventory; unset the optional {variable} override")
    target = _output_location(args.data_root, args.output_dir, args.overwrite)
    inputs = VerifiedInputs(args.data_root)
    splits = _splits(inputs)
    events, sessions, raw_rows = _raw_events(inputs, splits)
    paper, resolution = _paper_events(inputs, splits, sessions)
    pairs = opinion_pairs(events)
    views = {"human_events.csv": events, "paper_human_evaluation.csv": paper, "human_opinion_pairs.csv": pairs}
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".debate-human-", dir=target.parent) as scratch:
        staging = Path(scratch) / "export"
        staging.mkdir()
        files = []
        for name, rows in views.items():
            ids = [(row["session_id"], row["participant_id"]) if name == "human_opinion_pairs.csv" else row["record_id"] for row in rows]
            if len(set(ids)) != len(ids):
                raise ValueError(f"Duplicate output primary key in {name}")
            path = staging / name
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS[name], lineterminator="\n")
                writer.writeheader()
                for row in rows:
                    writer.writerow({key: str(value).lower() if isinstance(value, bool) else value for key, value in row.items()})
            files.append({"path": name, "sha256": sha256(path), "size_bytes": path.stat().st_size, "row_count": len(rows)})
        _write_json(staging / "schema.json", schema_document())
        files.append({"path": "schema.json", "sha256": sha256(staging / "schema.json"), "size_bytes": (staging / "schema.json").stat().st_size})
        counts = {"sessions": len(sessions), "raw_source_rows": raw_rows, "human_event_rows": len(events),
                  "removed_exact_survey_copies": raw_rows - len(events), "paper_sessions": len(resolution.available_prefixes),
                  "paper_missing_sessions": len(resolution.missing_prefixes), "paper_rows": len(paper), "opinion_pairs": len(pairs),
                  "complete_numeric_pairs": sum(pair["self_reported_opinion_change"] is not None for pair in pairs),
                  "opinion_auto_submission_flags": dict(Counter(str(row["is_auto_submitted"]).lower() for row in events if row["event_type"] in OPINIONS)),
                  "selection_reasons": {stage: dict(sorted(Counter(pair[stage + "_selection_reason"] for pair in pairs).items())) for stage in ("pre", "post")},
                  "paper_inventory": {f"{split}_{part}": {event: {"rows": sum(row[split + "_split"] == part and row["event_type"] in types for row in paper), "on_topic": sum(row[split + "_split"] == part and row["event_type"] in types and row["stance_in_paper_inventory"] for row in paper)} for event, types in (("tweets", {"tweet"}), ("opinions", OPINIONS), ("chats", {"message_sent"}))} for split in ("group", "topic") for part in ("train", "test")}}
        _write_json(staging / "manifest.json", {"schema_version": 1, "policy_version": POLICY_VERSION,
            "source": {key: inputs.source[key] for key in ("dataset_repo", "revision", "release")},
            "files": files, "counts": counts, "component_manifests": inputs.component_manifests,
            "source_files": [inputs.consumed[key] for key in sorted(inputs.consumed)],
            "selection_policy": {"scope": "New convenience export; does not replace historical paper preprocessing or archived caches.",
                "survey_duplicates": "Remove only exactly identical complete original exit_survey rows within one session; retain first source_row and count copies.",
                "opinions": "Keep every distinct submission in human_events. For pairs prefer manual over automatic when both exist. Unknown flags remain candidates. Resolve repeated candidates only when text, rating and flag all agree; retain first source row for identical candidates. Otherwise leave selected values missing. Never average or choose by text length or event order.",
                "paper": "Preserve canonical source text, cached stance and labels. Compute effective human validity and inventory inclusion with the original paper helpers."}})
        if target.exists():
            backup = Path(scratch) / "previous"
            os.replace(target, backup)
            try:
                os.replace(staging, target)
            except BaseException:
                os.replace(backup, target)
                raise
        else:
            os.replace(staging, target)
    print(f"Exported {len(events):,} human events, {len(paper):,} paper records and {len(pairs):,} opinion pairs to {target}")
    return counts
