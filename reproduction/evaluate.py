"""Evaluate newly annotated human/model runs with the paper's stance metrics."""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path

from reproduction.run import load_renderer, read_csv, write_json

NUMERIC_KEYS = ("chat_round_order", "event_order", "webpage_order", "chat_order")
EVENTS = {"Initial Opinion", "tweet", "message_sent", "Post Opinion"}


def normalized_number(value):
    value = str(value or "").strip()
    if not value:
        return ""
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"Non-numeric event key {value!r}") from exc
    if not number.is_finite():
        return ""
    return format(number.normalize(), "f")


def load_labels(path, *, source, session, validity_policy, core):
    """Explicit source filtering avoids the legacy filename-dependent reader."""
    rows = read_csv(path)
    field = source + "_validity"
    if rows and field not in rows[0]:
        raise ValueError(f"{path}: missing source-specific {field}; annotate both stance and validity for this source")
    stats = Counter(total_rows=len(rows))
    output = []
    for original in rows:
        if original.get("event_type", "").strip() not in EVENTS:
            continue
        row = dict(original)
        row["event_type"] = row["event_type"].strip()
        for key in NUMERIC_KEYS:
            if key in row:
                row[key] = normalized_number(row[key])
        row["group_id"] = row.get("group_id", "").strip() or session
        if not core.participant_id(row):
            raise ValueError(f"{path}: evaluated row has no participant identifier")
        if not row.get("event_order") and not row.get("message_id"):
            raise ValueError(f"{path}: evaluated row requires event_order or message_id for a stable join")
        if row["event_type"] == "tweet" and row.get("chat_round_order") not in {"1","2","3"}:
            raise ValueError(f"{path}: tweet requires chat_round_order 1, 2, or 3")
        validity = (row.get(field) or "").strip().upper()
        stats["validity_" + (validity.lower() or "missing")] += 1
        if validity == "INVALID" or (validity_policy == "valid-only" and validity != "VALID"):
            row["likert_pred"] = ""
            stats["excluded_by_validity"] += 1
        if core.parse_likert(row.get("likert_pred")) is None:
            stats["missing_stance_after_validity"] += 1
        else:
            stats["usable_stance_rows"] += 1
        output.append(row)
    if not output:
        raise ValueError(f"{path}: no evaluated event rows")
    keys = [core.row_key(row,index) for index,row in enumerate(output)]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{path}: duplicate event keys make row alignment ambiguous")
    metadata_path = path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if metadata.get("source") not in {None,source}:
        raise ValueError(f"{path}: annotation metadata source does not match {source}")
    if metadata.get("output_sha256") and hashlib.sha256(path.read_bytes()).hexdigest() != metadata["output_sha256"]:
        raise ValueError(f"{path}: labels differ from their annotation metadata checksum")
    return output,dict(stats),metadata


def evaluate(args):
    r = load_renderer()
    # Imported by the vendored renderer and loaded from the same scientific core.
    import generate_plots as core
    if args.pairs:
        manifest = args.pairs.expanduser().resolve()
        specs = json.loads(manifest.read_text())
        if not isinstance(specs,list) or not specs:
            raise ValueError("--pairs must contain a nonempty JSON list")
        for item in specs:
            for key in ("human","llm"):
                item[key] = (manifest.parent/Path(item[key]).expanduser()).resolve()
    else:
        specs = [{"session":args.session,"model":args.model,"human":args.human.expanduser().resolve(),"llm":args.llm.expanduser().resolve()}]
    models = list(dict.fromkeys(str(item.get("model",args.model)) for item in specs))
    prefixes = list(dict.fromkeys(str(item["session"]) for item in specs))
    aggregates = core.build_model_aggregates(models)
    prefix_results = {prefix:{} for prefix in prefixes}
    statuses = []
    llm_protocols = set()
    for item in specs:
        session,model = str(item["session"]),str(item.get("model",args.model))
        if not session or model in prefix_results[session]:
            raise ValueError("Each session/model pair must be nonempty and unique")
        human,hstats,hmeta = load_labels(item["human"],source="human",session=session,validity_policy=args.validity_policy,core=core)
        llm,lstats,lmeta = load_labels(item["llm"],source="llm",session=session,validity_policy=args.validity_policy,core=core)
        protocol = lmeta.get("validity_protocol","unrecorded")
        llm_protocols.add(protocol)
        if len(llm_protocols)>1:
            raise ValueError("Input model labels use mixed or partially unrecorded validity protocols; evaluate them separately")
        pairs,mode,rate = core.align_rows(human,llm,min_key_match_rate=args.min_key_match_rate)
        if mode != "key":
            raise ValueError(f"{session}/{model}: only {rate:.1%} human rows match by key; refusing positional alignment")
        result = core.compute_prefix_model_from_rows(data_prefix=session,model=model,human_rows=human,llm_rows=llm,
            utterance_event_types=EVENTS,likert_col="likert_pred",min_key_match_rate=args.min_key_match_rate,
            include_conditioned_alignment=False,human_path=str(item["human"]),llm_path=str(item["llm"]))
        if result.public_alignment_count+result.private_alignment_count == 0:
            raise ValueError(f"{session}/{model}: no valid matched public/private predictions")
        core.merge_prefix_result_into_aggregate(aggregates[model],result)
        prefix_results[session][model]=result
        statuses.append({"session":session,"model":model,"alignment":"key","key_match_rate":rate,"aligned_rows":len(pairs),
                         "human_rows":hstats,"llm_rows":lstats,"human_judge":hmeta.get("judge","unrecorded"),
                         "llm_judge":lmeta.get("judge","unrecorded"),"validity_protocol":protocol})
    rows = core.build_raw_rows_from_aggregates(aggregates,models,ub_mode="gaussian_likert")
    rows = core.add_v2_bootstrap_cis_to_raw_rows(rows,models=models,prefixes=prefixes,prefix_results_by_prefix=prefix_results,
                                              bootstrap_reps=args.bootstrap_reps,seed=42)
    rows = [row for row in rows if row.get("model") in models]
    for row in rows:
        row.update(simulation_version=args.simulation_mode,validity_policy=args.validity_policy,validity_protocol=next(iter(llm_protocols)))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    r.write_csv(args.output_dir/"summary_metrics.csv",rows)
    write_json(args.output_dir/"evaluation_manifest.json",{
        "input_kind":"newly annotated runs; separate from fixed paper replay",
        "simulation_mode":args.simulation_mode,"validity_policy":args.validity_policy,
        "validity_protocols":sorted(llm_protocols),"bootstrap_reps":args.bootstrap_reps,"bootstrap_seed":42,
        "conditioned_alignment_included":False,"sessions":len(prefixes),"models":models,"inputs":statuses,
        "metric_scope":"Six stance prediction errors, associated counts/correlations, and bootstrap intervals from the same scientific core as paper replay. Text embedding metrics are not computed by this command.",
        "key_normalization":"Numeric event/round keys normalize 1 and 1.0; participant identifiers are unchanged. Positional fallback is rejected."
    })
    print(f"Evaluated {len(specs)} session/model pairs; wrote {args.output_dir/'summary_metrics.csv'}")
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs",type=Path,help="JSON list of session, human, llm and optional model paths, relative to the manifest")
    parser.add_argument("--human",type=Path,help="Human annotations CSV for a single session")
    parser.add_argument("--llm",type=Path,help="Generated-text annotations CSV for a single session")
    parser.add_argument("--session",help="Stable session identifier for a single pair")
    parser.add_argument("--model",default="generated-model")
    parser.add_argument("--simulation-mode",choices=("v0","v1_5"),required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--bootstrap-reps",type=int,default=1000)
    parser.add_argument("--min-key-match-rate",type=float,default=0.7)
    parser.add_argument("--validity-policy",choices=("valid-only","paper-inline"),default="valid-only",
                        help="valid-only excludes missing/UNKNOWN validity; paper-inline mirrors the historical INVALID-only exclusion")
    args=parser.parse_args(argv)
    if args.pairs and any(value is not None for value in (args.human,args.llm,args.session)):
        parser.error("Use --pairs or --human/--llm/--session, not both")
    if not args.pairs and not all((args.human,args.llm,args.session)):
        parser.error("Supply --pairs or all of --human, --llm, --session")
    if args.bootstrap_reps<0 or not 0<args.min_key_match_rate<=1:
        parser.error("Require nonnegative bootstrap count and 0 < key match rate <= 1")
    args.output_dir = args.output_dir.expanduser().resolve()
    return evaluate(args)


if __name__=="__main__":
    raise SystemExit(main())
