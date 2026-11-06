"""Annotate human or generated event text with cached stance and validity judgments."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

from .common import PROMPTS, digest, read_csv, write_csv, write_json

LABELS = {"Certainly disagree": 1, "Probably disagree": 2, "Lean disagree": 3,
          "Lean agree": 4, "Probably agree": 5, "Certainly agree": 6}
EVENTS = {"Initial Opinion", "tweet", "message_sent", "Post Opinion"}


def stance_prompt(topic, text, player, history):
    suffix = 'Please use exactly one word from the following 6 categories to label it: "Certainly agree", "Probably agree", "Lean agree", "Lean disagree", "Probably disagree", and "Certainly disagree". Your predicted label: '
    if history:
        return f'Here is prior conversation around the topic "{topic}":\n{history}\nWhat is the stance of the following new message by {player} with respect to the topic "{topic}"? Here is the message: "{text}". ' + suffix
    return f'What is the stance of the following message with respect to the topic "{topic}"? Here is the message: "{text}". ' + suffix


def conversation(rows, index, text_column):
    row = rows[index]
    if row.get("event_type") in ("Initial Opinion", "Post Opinion"):
        return ""
    pair = {row.get("sender_id"), row.get("recipient_id")}
    return "\n".join(f"{earlier.get('sender_id')}: {earlier.get(text_column, '')}" for earlier in rows[:index]
                     if earlier.get("event_type") in ("tweet", "message_sent")
                     and earlier.get("chat_round_order") == row.get("chat_round_order")
                     and earlier.get("sender_id") in pair and earlier.get(text_column, "").strip())


def validity_messages(topic, text, event, history):
    if event in ("Initial Opinion", "Post Opinion"):
        slug = "initial" if event == "Initial Opinion" else "post"
        system = (PROMPTS / "validity" / f"llm_judge_system_message_{slug}_v3_recovered.md").read_text()
        user = (PROMPTS / "validity" / f"llm_judge_user_message_{slug}_v2_20260513.md").read_text()
        fields = {"TOPIC": topic, "INITIAL_OPINION": text, "POST_OPINION": text}
    elif event == "tweet":
        system = (PROMPTS / "llm_judge_v2/system_message_tweet.md").read_text()
        user = (PROMPTS / "llm_judge_v2/user_message_tweet.md").read_text()
        fields = {"TOPIC": topic, "TWEET": text}
    else:
        system = (PROMPTS / "llm_judge_v2/system_message_utterances.md").read_text()
        user = (PROMPTS / "llm_judge_v2/user_message_utterances.md").read_text()
        fields = {"TOPIC": topic, "CURRENT_OR_OTHER": "the" if topic in ("Regular fasting will improve your health", "Regular fasting will not improve your health") else "another",
                  "CHAT_HISTORY": history, "LAST_MESSAGE": text}
    return [{"role": "system", "content": system.format(**fields)}, {"role": "user", "content": user.format(**fields)}]


class JudgmentCache:
    def __init__(self, path):
        self.path = path
        self.values = {}
        if path.exists():
            for number, line in enumerate(path.read_text().splitlines(), 1):
                if line.strip():
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid cache record {path}:{number}; restore/truncate the incomplete record explicitly") from exc
                    self.values[item["key"]] = item["response"]

    def request(self, model, messages, reasoning_effort, client, validity_protocol="generated-text"):
        payload = {"model": model, "messages": messages}
        if model.startswith(("gpt-5", "o1", "o3", "o4")):
            payload["reasoning_effort"] = reasoning_effort
        identity = {"request": payload, "validity_protocol": validity_protocol, "cache_schema": 2}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if key in self.values:
            return self.values[key]
        if client is None:
            raise ValueError("A judgment is missing from the cache; supply API credentials or use --dry-run")
        for attempt in range(4):
            try:
                response = client.chat.completions.create(**payload).choices[0].message.content or ""
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, **identity, "response": response}, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.values[key] = response
        return response


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--source", choices=("human", "llm"), default="llm")
    parser.add_argument("--topic")
    parser.add_argument("--judge", default="gpt-5-mini")
    parser.add_argument("--endpoint", help="Optional OpenAI-compatible endpoint")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"), default="medium")
    parser.add_argument("--task", choices=("both", "stance", "validity"), default="both")
    parser.add_argument("--validity-protocol", choices=("generated-text", "paper-v5"), default="generated-text",
                        help="generated-text judges generated LLM post opinions; paper-v5 reproduces the historical post_opinion/text selector for cached-paper protocol studies")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    fields, rows = read_csv(args.input)
    text_column = "text" if args.source == "human" else "llm_text"
    if text_column not in fields:
        raise ValueError(f"Missing {text_column} in {args.input}")
    topic = args.topic
    if topic is None:
        match = re.search(r"\d{8}_\d{6}_(.*)_.{26}", str(args.input))
        if not match:
            parser.error("--topic is required when the input path does not contain a session prefix")
        topic = match.group(1).replace("_", " ")
    rows.sort(key=lambda row: float(row.get("event_order") or 0))
    jobs = []
    for index, row in enumerate(rows):
        if row.get("event_type") not in EVENTS:
            continue
        text = row.get(text_column, "").strip()
        if row["event_type"] == "Initial Opinion" and args.source == "llm":
            text = row.get("text", "").strip()  # Initial private opinion is conditioned human information.
        validity_text = text
        if row["event_type"] == "Post Opinion" and args.validity_protocol == "paper-v5":
            # Historical apply_validation() selected this field independently
            # of isLLM. Cached simulation CSVs have text but no post_opinion.
            validity_text = row.get("post_opinion", row.get("text", "")).strip()
        history = conversation(rows, index, text_column)
        if args.task in ("both", "stance") and text:
            jobs.append((index, "stance", [{"role": "user", "content": stance_prompt(topic, text, row.get("sender_id") or row.get("worker_id"), history)}]))
        if args.task in ("both", "validity") and validity_text:
            jobs.append((index, "validity", validity_messages(topic, validity_text, row["event_type"], history)))
    metadata = {"input_sha256": digest(args.input), "source": args.source, "judge": args.judge, "task": args.task,
                "validity_protocol": args.validity_protocol,
                "requests": len(jobs), "prompt_profile": "research stance prompts; recovered eval_v5 initial/post system prompts; retained tweet/chat validity templates",
                "historical_cache_identity": "New judgments are not guaranteed identical to the released historical labels."}
    if args.dry_run:
        write_json(args.output.with_suffix(".plan.json"), metadata)
        print(f"Prepared {len(jobs)} judgment requests; no API calls made.")
        return 0
    cache = JudgmentCache(args.cache)
    client = None
    if not args.cache_only:
        from openai import OpenAI
        client = OpenAI(base_url=args.endpoint) if args.endpoint else OpenAI()
    extras = ["likert_pred", "label_pred", f"{args.source}_validity", "validity_reason", "judge_model"]
    fields += [field for field in extras if field not in fields]
    invalid_answers = 0
    for index, task, messages in jobs:
        answer = cache.request(args.judge, messages, args.reasoning_effort, client, args.validity_protocol)
        normalized = answer.strip().strip('"').strip()
        row = rows[index]
        if task == "stance":
            row["label_pred"] = normalized if normalized in LABELS else ""
            row["likert_pred"] = LABELS.get(normalized, "")
            invalid_answers += normalized not in LABELS
        else:
            status = normalized.upper()
            row[f"{args.source}_validity"] = status if status in ("VALID", "INVALID") else "UNKNOWN"
            row["validity_reason"] = answer
            invalid_answers += status not in ("VALID", "INVALID")
        row["judge_model"] = args.judge
    write_csv(args.output, fields, rows)
    metadata["unparseable_judgments"] = invalid_answers
    metadata["output_sha256"] = digest(args.output)
    write_json(args.output.with_suffix(".metadata.json"), metadata)
    print(f"Wrote {args.output}; {invalid_answers} unparseable judgments retained as missing/UNKNOWN.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
