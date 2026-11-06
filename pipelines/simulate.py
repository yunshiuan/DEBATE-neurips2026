"""Run the research conversation simulator with portable inputs and outputs."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .common import VERSION_NAMES, digest, read_csv, select_csvs, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, help="Extracted DEBATE release root")
    parser.add_argument("--input-dir", type=Path, help="Override the default released group-test event CSV directory")
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=VERSION_NAMES, default="full")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--api", choices=("openai", "gemini", "huggingface"), default="huggingface")
    parser.add_argument("--endpoint", default="http://localhost:8000/v1")
    parser.add_argument("--model-family", choices=("llama", "mistral", "olmo", "tulu"))
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=258)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print a plan without importing model clients or contacting an API")
    args = parser.parse_args(argv)
    input_dir = args.input_dir
    if input_dir is None:
        if args.data_root is None:
            parser.error("provide --data-root or --input-dir")
        input_dir = args.data_root / "reproduction/inputs/llama/data/sft_data/group_split_data_bd/test"
    paths = select_csvs(input_dir, args.mask, args.limit)
    version = VERSION_NAMES[args.mode]
    inputs = []
    required = {"event_type", "event_order", "text", "worker_id", "empirica_id", "sender_id", "recipient_id", "chat_round_order", "field", "sliderValue"}
    for path in paths:
        columns, rows = read_csv(path)
        if required - set(columns):
            raise ValueError(f"{path.name}: missing columns {sorted(required - set(columns))}")
        inputs.append({"session": path.stem, "sha256": digest(path), "events": dict(Counter(row["event_type"] for row in rows))})
    config = {"mode": version, "model": args.model, "api": args.api, "temperature": args.temperature, "max_tokens": args.max_tokens, "seed": args.seed, "sessions": inputs}
    if args.dry_run:
        print(f"Validated {len(inputs)} sessions for {version}; no model calls made.")
        write_json(args.output_dir / "simulation_plan.json", config)
        return 0
    from .vendor import simulation
    model_archive = simulation.Agent.get_model_archive_name(args.model)
    for path in paths:
        result = args.output_dir / "simulation" / path.stem / model_archive / f"simulation-{version}.csv"
        run_identity = {key: config[key] for key in ("model", "api", "temperature", "max_tokens", "seed", "mode")}
        run_identity.update({"input_sha256": digest(path), "model_family": args.model_family})
        identity_path = result.with_suffix(".pipeline.json")
        if result.exists() and not args.overwrite:
            if not identity_path.exists() or json.loads(identity_path.read_text()) != run_identity:
                raise FileExistsError(f"Existing output has different or unknown inputs/config: {result}; use a new output directory or --overwrite")
            print(f"Skipping existing {result}")
            continue
        simulation.main(api_type=args.api, model_name=args.model, temperature=args.temperature,
                        seed=args.seed, max_tokens=args.max_tokens, data_prefix=path.stem,
                        version=version, endpoint_url=args.endpoint, input_path=path,
                        output_root=args.output_dir, model_family=args.model_family)
        write_json(identity_path, run_identity)
    write_json(args.output_dir / f"simulation_run_{version}.json", config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
