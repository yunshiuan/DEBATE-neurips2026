"""Recompute the recoverable historical memory ablation; expose other gaps."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import numpy as np

from reproduction.run import HERE, input_paths, load_renderer, read_csv, require, write_json


def summarize(path):
    rows = read_csv(path)
    buckets = defaultdict(list)
    for row in rows:
        key = (row["group_id"],row["topic"],row["event_type"],int(float(row["chat_round_order"])))
        buckets[key].append(row)
    group_rows=[]
    for (group,topic,event,round_number),items in sorted(buckets.items()):
        for role in ("human","llm"):
            values=[]
            for row in items:
                try:
                    value=float(row[role+"_likert_pred"])
                except (ValueError,TypeError):
                    continue
                if np.isfinite(value):
                    values.append(value)
            if not values:
                continue
            x=np.asarray(values)
            # Historical summaries use the available cached labels directly,
            # population SD (ddof=0), and no additional validity filter.
            group_rows.append({"group_id":group,"topic":topic,"event":event,"round":round_number,
                               "role":role,"n":len(x),"sd":float(np.std(x,ddof=0)),
                               "mean":float(np.mean(x)),"range":float(np.ptp(x))})
    summary=[]
    for role in ("human","llm"):
        for channel,initial,final in (("public",("tweet",1),("tweet",3)),("private",("Initial Opinion",0),("Post Opinion",4))):
            start=[row for row in group_rows if row["role"]==role and (row["event"],row["round"])==initial]
            end=[row for row in group_rows if row["role"]==role and (row["event"],row["round"])==final]
            summary.append({"role":role,"channel":channel,"initial_groups":len(start),"final_groups":len(end),
                            "initial_sd":float(np.mean([r["sd"] for r in start])),"final_sd":float(np.mean([r["sd"] for r in end])),
                            "delta_sd":float(np.mean([r["sd"] for r in end])-np.mean([r["sd"] for r in start]))})
    return group_rows,summary


def conversation_quality(args):
    path=input_paths(args.data_root)["inputs"]/"historical/conversation_quality_annotations.csv"
    require([path])
    rows=read_csv(path)
    paired=defaultdict(dict)
    for row in rows:
        key=(row["source_group_id"],row["round"],row["participant_pair"])
        if row["annotator"] in paired[key]:
            raise ValueError("Duplicate conversation/annotator record in quality annotations")
        paired[key][row["annotator"]]=row
    expected_annotators={"annotator_1","annotator_2"}
    if len(paired)!=120 or any(set(item)!=expected_annotators for item in paired.values()):
        raise ValueError("Quality audit requires exactly 120 paired conversations")
    criteria=("coherent","responsive","substantive")
    expected={"coherent":(100.0,100.0,100.0),"responsive":(99.2,100.0,99.2),"substantive":(97.5,98.3,99.2)}
    summaries=[]
    agreements=0
    for criterion in criteria:
        a=np.asarray([int(unit["annotator_1"][criterion]) for unit in paired.values()])
        b=np.asarray([int(unit["annotator_2"][criterion]) for unit in paired.values()])
        if not set(a).issubset({0,1}) or not set(b).issubset({0,1}):
            raise ValueError("Quality labels must be binary")
        agreements+=int(np.sum(a==b))
        actual=tuple(round(float(value),1) for value in (100*np.mean(a),100*np.mean(b),100*np.mean(a==b)))
        summaries.append({"criterion":criterion,"conversations":len(a),"annotator_1_positive":int(np.sum(a)),"annotator_2_positive":int(np.sum(b)),"agreements":int(np.sum(a==b)),"annotator_1_percent":actual[0],"annotator_2_percent":actual[1],"agreement_percent":actual[2],"match":actual==expected[criterion]})
    groups=len({row["source_group_id"] for row in rows})
    passed=groups==20 and agreements==358 and all(row["match"] for row in summaries)
    out=args.output_dir/"historical"
    load_renderer().write_csv(out/"conversation_quality.csv",summaries)
    write_json(out/"conversation_quality_verification.json",{"passed":passed,"groups":groups,"conversations":len(paired),"agreements":agreements,"judgments":len(paired)*len(criteria),"overall_agreement_percent":round(100*agreements/(len(paired)*len(criteria)),1),"criteria":summaries})
    if not passed:
        raise RuntimeError("Conversation quality audit differs from the manuscript")
    print("Conversation quality audit: all criterion rates and 358/360 agreement reproduced.",flush=True)


def run_historical(args):
    root=input_paths(args.data_root)["inputs"]/"historical/group_level_eval"
    files={"with_memory":"gpt_sampled_human_validation_rows_merged_with_tweet2.csv",
           "without_memory":"ablation_sampled_human_validation_rows_merged_with_tweet2.csv"}
    require([root/name for name in files.values()])
    out=args.output_dir/"historical"
    r=load_renderer()
    results=[]
    expected={("with_memory","public"):-0.28,("with_memory","private"):-0.32,
              ("without_memory","public"):-0.57,("without_memory","private"):-0.59}
    checks=[]
    for condition,filename in files.items():
        groups,summary=summarize(root/filename)
        r.write_csv(out/(condition+"_group_sd.csv"),groups)
        for row in summary:
            row["condition"]=condition
            if row["role"]=="llm":
                target=expected[condition,row["channel"]]
                checks.append({"condition":condition,"channel":row["channel"],"expected":target,
                               "actual":round(row["delta_sd"],2),"match":round(row["delta_sd"],2)==target})
        results.extend(summary)
    r.write_csv(out/"memory_ablation.csv",results)
    write_json(out/"memory_ablation_verification.json",{"passed":all(row["match"] for row in checks),"checks":checks})
    if not all(row["match"] for row in checks):
        raise RuntimeError("Historical memory-ablation results differ from paper reference values.")
    conversation_quality(args)
    claims=json.loads((HERE/"reference/historical_claims.json").read_text())
    for claim in claims:
        claim["source_audit_status"]=claim.pop("status")
        claim["reproduction_status"]="recomputed_from_cached_labels" if claim["claim"] in {"GPT-4o-mini memory ablation","Conversation-quality audit arithmetic"} else "reported_aggregate_only"
    write_json(out/"historical_coverage.json",{"claims":claims,"note":"Memory ablation and the conversation-quality audit are independently regenerated here. Available similarly named 39-group CSVs do not reproduce the separate Llama robustness analysis. See docs/paper_reproduction.md for missing inputs."})
    print("Historical memory ablation: all four paper values reproduced; quality audit verified; remaining auxiliary claims recorded as aggregate-only.",flush=True)
