"""Reconstruct next-message training examples from the released event records."""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
from pathlib import Path
import tempfile

from .common import digest, select_csvs, write_json


def load_post_validity(root: Path | None, prefix: str, model_pattern: str):
    if root is None:
        return {}
    mixed = root / prefix / "simulation-human.csv"
    paths = [mixed] if mixed.exists() else sorted((root / prefix).glob(model_pattern + "/opinion_human_memory_gpt-5-mini_v0*.csv"))
    values = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row.get("event_type") != "Post Opinion":
                    continue
                player = str(row.get("worker_id") or row.get("sender_id") or "")
                status = (row.get("new_validity") or row.get("human_validity") or row.get("validity") or "").strip().upper()
                # A present but blank/unknown label is not VALID. The original
                # formatter likewise compared present new_validity to VALID.
                status = status if status in ("VALID", "INVALID") else "UNKNOWN"
                if path == mixed and player in values:
                    # Match get_post_opinion(), which selects the first
                    # post-opinion record for this participant in the CSV.
                    continue
                if player in values and values[player] != status:
                    raise ValueError(f"Conflicting post-opinion labels for {prefix}/{player}; narrow --label-model-pattern")
                values[player] = status
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=("group", "topic"), required=True)
    parser.add_argument("--partition", choices=("train", "test"), required=True)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--post-validity-root", type=Path, help="Per-session mixed simulation-human.csv validity cache or original post-opinion sidecars")
    parser.add_argument("--label-model-pattern", default="Llama-3.1-8B-Instruct")
    parser.add_argument("--allow-reconstructed", action="store_true", help="Allow missing post labels to use the original formatter's keep-by-default fallback, with an explicit audit")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Inspect split membership and label coverage without producing examples")
    args = parser.parse_args(argv)
    split_root = args.data_root / "reproduction/inputs/llama/data/sft_data"
    released_validity = args.data_root / "reproduction/inputs/llama/validity"
    if args.post_validity_root is None and released_validity.exists():
        args.post_validity_root = released_validity
    input_dir = args.input_dir or split_root / f"{args.split}_split_data_bd" / args.partition
    mask = args.mask or split_root / "masks" / f"{args.split}_{args.partition}_prefixes.txt"
    paths = select_csvs(input_dir, mask if mask.exists() else None, args.limit)
    if not args.allow_reconstructed and args.post_validity_root is None:
        parser.error("No post-opinion validity source found. Supply --post-validity-root or explicitly select --allow-reconstructed; reconstructed examples are not verified as the exact original training corpus.")
    opposite = "test" if args.partition == "train" else "train"
    opposite_mask = split_root / "masks" / f"{args.split}_{opposite}_prefixes.txt"
    if opposite_mask.exists():
        opposite_ids = set(opposite_mask.read_text().splitlines())
        overlap = opposite_ids & {path.stem for path in paths}
        if overlap:
            raise ValueError(f"Train/test session overlap: {sorted(overlap)[:3]}")
    coverage = {path.stem: load_post_validity(args.post_validity_root, path.stem, args.label_model_pattern) for path in paths}
    report = {
        "split": args.split, "partition": args.partition, "sessions": len(paths),
        "input_sha256": {path.name: digest(path) for path in paths},
        "training_corpus_status": "reconstructed",
        "historical_original_jsonl_available": False,
        "historical_expected_examples": {("group", "train"): 17057, ("group", "test"): 3804, ("topic", "train"): 14662, ("topic", "test"): 6199}[args.split, args.partition],
        "test_partition_usage": "Epoch validation and checkpoint selection; excluded from parameter updates.",
        "post_label_coverage": {prefix: len(labels) for prefix, labels in coverage.items()},
        "post_label_source": "mixed_eval_v5_human_cache" if args.post_validity_root and any((args.post_validity_root / p.stem / "simulation-human.csv").exists() for p in paths) else "sidecars_or_missing",
        "missing_post_labels": [],
    }
    if args.dry_run:
        write_json(args.output_dir / f"{args.split}_{args.partition}_plan.json", report)
        print(f"Validated {len(paths)} {args.split}/{args.partition} input sessions; no training or API calls.")
        return 0
    import pandas as pd
    from .vendor import format_sft as formatter
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.split}_{args.partition}.jsonl"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}; choose a new output directory")
    missing = set()

    def post_validity(prefix, player):
        key = str(player)
        status = coverage[prefix].get(key)
        if status is not None:
            return status == "VALID"
        if not args.allow_reconstructed:
            raise ValueError(f"Missing post-opinion validity for {prefix}/{key}")
        missing.add((prefix, key))
        return True  # Explicit reconstruction mode preserves the original formatter's missing-overlay fallback.

    counts = {}
    formatter.POST_VALIDITY_CHECK = post_validity
    with tempfile.TemporaryDirectory(prefix="debate-sft-") as tmp, (args.output_dir / f"{args.split}_{args.partition}_format.log").open("w") as log:
        formatter.DATA_DIR = tmp
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            for path in paths:
                frame = pd.read_csv(path, dtype={"worker_id": str, "empirica_id": str, "sender_id": str, "recipient_id": str})
                if "validity" not in frame:
                    raise ValueError(f"{path.name} has no message validity labels")
                frame["validity"] = frame["validity"].fillna("UNKNOWN").astype(str)
                formatter.format_data_for_model(frame, formatter.extract_topic_from_filename(path.name),
                                                "selected", {}, path.name, split_type=args.split,
                                                set_type=args.partition, filter_type="none", golden_filter=set())
        temporary_output = output.with_suffix(".jsonl.tmp")
        with temporary_output.open("w", encoding="utf-8") as handle:
            for path in paths:
                count = 0
                example_root = Path(tmp) / formatter.INTERMEDIATE_DIR / "selected" / path.stem
                for example in sorted(example_root.glob("*/message_*.jsonl"), key=lambda p: (p.parent.name, int(p.stem.split("_")[-1]))):
                    row = json.loads(example.read_text())
                    roles = [m["role"] for m in row["messages"]]
                    if roles != ["system", "user", "assistant"]:
                        raise ValueError(f"Unexpected message roles in {example}: {roles}")
                    row.update({"session_id": path.stem, "participant_id": example.parent.name,
                                "split": args.split, "partition": args.partition,
                                "training_corpus_status": report["training_corpus_status"]})
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
                counts[path.stem] = count
        temporary_output.replace(output)
    report["missing_post_labels"] = [{"session": prefix, "participant": player} for prefix, player in sorted(missing)]
    report["examples_by_session"] = counts
    report["examples"] = sum(counts.values())
    report["matches_historical_count"] = report["examples"] == report["historical_expected_examples"]
    report["jsonl_sha256"] = digest(output)
    write_json(args.output_dir / f"{args.split}_{args.partition}_manifest.json", report)
    print(f"Wrote {report['examples']} examples to {output}; corpus status: {report['training_corpus_status']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
