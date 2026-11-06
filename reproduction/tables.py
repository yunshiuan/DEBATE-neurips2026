"""Recompute human inventories and cross-model tables from released rows."""
from __future__ import annotations

import importlib
import json
import math
from pathlib import Path

from reproduction.run import HERE, input_paths, load_renderer, read_csv, require, write_json


def inventory(args):
    r = load_renderer()
    g = importlib.import_module("generate_v4_dataset_stats_tables")
    inputs = input_paths(args.data_root)["inputs"]
    base, masks = inputs/"base/human_llm", inputs/"llama/data/sft_data/masks"
    initial, tweet = inputs/"validity/initial_post", inputs/"validity/tweet"
    # The canonical human records have no tweet override sidecars. The
    # historical resolver correctly falls back to inline validity when this
    # optional (and therefore unarchived empty) root does not exist.
    require([base,masks,initial])
    results = []
    coverage = []
    for split in ("group","topic"):
        for partition in ("train","test"):
            key = split+"_"+partition
            prefixes = g.read_mask_prefixes(masks/(key+"_prefixes.txt"))
            resolution = g.resolve_available_prefixes(prefixes,eval_root=base,human_filename="opinion_human_memory_gpt-5-mini_v0.csv")
            stats = g.collect_stats_for_prefixes(resolution.available_prefixes,eval_root=base,human_filename="opinion_human_memory_gpt-5-mini_v0.csv",initial_post_sidecar_root=initial,tweet_sidecar_root=tweet)
            row = {"split":split,"partition":partition,"requested_groups":len(prefixes),"available_groups":stats.group_count,"topics":stats.topic_count,"requested_topics":len({g.extract_topic_from_data_prefix(prefix) for prefix in prefixes}),"subjects":stats.subject_count,
                   "eval_rows":stats.message_count,"on_topic_rows":stats.on_topic_message_count,"tweets":stats.tweet_count,"on_topic_tweets":stats.on_topic_tweet_count,
                   "opinions":stats.opinion_count,"on_topic_opinions":stats.on_topic_opinion_count,"chats":stats.chat_message_count,"on_topic_chats":stats.on_topic_chat_message_count}
            for family in ("tweets","opinions","chats"):
                row[family+"_on_topic_pct"] = 100*row["on_topic_"+family]/row[family]
            results.append(row)
            spec = r.PARTITION_SPECS[key]
            cov = {"split":split,"partition":partition,"nominal_groups":len(prefixes),"human_available":len(resolution.available_prefixes),"human_missing":len(resolution.missing_prefixes)}
            for version in r.VERSION_SPECS:
                counts = {"usable":0,"missing_human":0,"missing_llm":0}
                for prefix in prefixes:
                    human = r.first_human_path(inputs/"llama/human_llm",prefix,version,str(spec["model"]))
                    llm = inputs/"llama/human_llm"/prefix/str(spec["model"])/r.VERSION_SPECS[version]["llm_filename"]
                    if human is None:
                        counts["missing_human"] += 1
                    elif not llm.is_file():
                        counts["missing_llm"] += 1
                    else:
                        counts["usable"] += 1
                cov.update({version+"_"+k:v for k,v in counts.items()})
            coverage.append(cov)
    out = args.output_dir/"tables"
    r.write_csv(out/"dataset_inventory.csv",results)
    r.write_csv(out/"dataset_coverage.csv",coverage)
    lines = [r"\begin{tabular}{llrrrrrrrrr}",r"\toprule",r"Split & Partition & Topics & Tweets & \% on-topic & Opinions & \% on-topic & Chats & \% on-topic & Subjects & Groups \\",r"\midrule"]
    for row in results:
        cells = [row["split"].title(),row["partition"].title(),str(row["topics"]),f'{row["tweets"]:,}',f'{row["tweets_on_topic_pct"]:.1f}',f'{row["opinions"]:,}',f'{row["opinions_on_topic_pct"]:.1f}',f'{row["chats"]:,}',f'{row["chats_on_topic_pct"]:.1f}',f'{row["subjects"]:,}',str(row["available_groups"])]
        lines.append(" & ".join(cells)+r" \\")
    lines.extend([r"\bottomrule",r"\end{tabular}"])
    (out/"dataset_inventory.tex").write_text("\n".join(lines)+"\n")
    expected = json.loads((HERE/"reference/dataset_inventory.json").read_text())
    checks=[]
    for row,cov,reference in zip(results,coverage,expected):
        actual={**row,"sft_fds":cov["v0_usable"],"sft_nmp":cov["v1_5_usable"]}
        for field,value in reference.items():
            checks.append({"split":row["split"],"partition":row["partition"],"field":field,"expected":value,"actual":actual[field],"match":actual[field]==value})
    write_json(out/"dataset_verification.json",{"passed":all(x["match"] for x in checks),"checks":checks})
    if not all(x["match"] for x in checks):
        raise RuntimeError("Dataset inventory differs from the paper; inspect tables/dataset_verification.json")
    total={key:sum(row[key] for row in results if row["split"]=="topic") for key in ("requested_groups","available_groups","subjects","topics","tweets","opinions","chats")}
    write_json(out/"dataset_totals.json",total)
    total_cells=["Total","",str(total["topics"]),f'{total["tweets"]:,}',"",f'{total["opinions"]:,}',"",f'{total["chats"]:,}',"",f'{total["subjects"]:,}',str(total["available_groups"])]
    lines.insert(-2,r"\midrule")
    lines.insert(-2," & ".join(total_cells)+r" \\")
    (out/"dataset_inventory.tex").write_text("\n".join(lines)+"\n")
    for name,data,cols in (
        ("dataset_count_details",results,("split","partition","requested_groups","available_groups","requested_topics","topics","eval_rows","on_topic_rows")),
        ("dataset_coverage",coverage,("split","partition","nominal_groups","human_available","human_missing","v0_usable","v1_5_usable")),
    ):
        body=[r"\begin{tabular}{"+"l"*len(cols)+"}",r"\toprule"," & ".join(k.replace("_",r"\_") for k in cols)+r" \\",r"\midrule"]
        body.extend(" & ".join(str(row[key]) for key in cols)+r" \\" for row in data)
        body.extend([r"\bottomrule",r"\end{tabular}"])
        (out/(name+".tex")).write_text("\n".join(body)+"\n")
    return results,coverage


def cross_model(args):
    r = load_renderer()
    inputs = input_paths(args.data_root)["inputs"]
    out = args.output_dir/"tables"
    outputs_path = args.output_dir/"figure_outputs.json"
    require([outputs_path])
    figure_paths = json.loads(outputs_path.read_text())
    convergence, errors = [], []
    llama_models = {"group_test": "ft:Llama-3.1-8B-Instruct-SFT-20260608-bd-real:bd-group-5epochs", "topic_test":"ft:Llama-3.1-8B-Instruct-SFT-20260608-bd-real:bd-topic-5epochs"}
    metrics = ("private_stance_alignment_abs","private_stance_change_error_abs","private_opinion_convergence_error")
    for partition in ("group_test","topic_test"):
        group = partition == "group_test"
        conv = read_csv(args.output_dir/figure_paths["M3" if group else "A5"])
        align = read_csv(args.output_dir/figure_paths["M1" if group else "A3"])
        models = {"human":"human","base":"Llama-3.1-8B-Instruct","SFT":llama_models[partition]}
        for channel in ("public","private"):
            for role,model in models.items():
                value = next(float(row["mean_delta_sd"]) for row in conv if row["panel"] == channel+"_convergence" and row["model"] == model)
                convergence.append({"family":"Llama","split":partition,"channel":channel,"role":role,"model":model,"mean_delta_sd":value})
        for role in ("base","SFT"):
            source = next(row for row in align if row["model"] == models[role] and row["simulation_version"] == "v0")
            errors.append({"family":"Llama","split":partition,"role":role,"model":models[role],**{metric:float(source[metric]) for metric in metrics}})

    q = importlib.import_module("generate_v5_leo_qwen4b_comparison")
    for family in ("Qwen","Mistral"):
        if family == "Qwen":
            root,mask = inputs/"qwen/human_llm",inputs/"qwen/masks"
            q.configure_renderer()
        else:
            m = importlib.import_module("generate_v5_ruixuan_mistral_sft_comparison")
            m.configure_shared_helpers()
            m.configure_renderer()
            root,mask = inputs/"mistral/human_llm",inputs/"llama/data/sft_data/masks"
        base,leo = inputs/"base/human_llm",inputs/"leo_llama/human_llm"
        require([root,mask,base,leo])
        all_raw, all_status = [], []
        for partition in ("group_test","topic_test"):
            print(f"Recompute {family} paper tables: {partition}",flush=True)
            alpha = q.fit_alphas(qwen_root=root,mask_root=mask,partition=partition.replace("_test","_train"),version="v0")
            raw,status = q.compute_partition_rows(qwen_root=root,llama_root=leo,llama_base_root=base,mask_root=mask,partition=partition,version="v0",alpha_spec=alpha,bootstrap_reps=0)
            for row in raw:
                row.update(partition=partition,simulation_version="v0")
            for row in status:
                row["partition"] = partition
            all_raw.extend(raw)
            all_status.extend(status)
            if family == "Qwen":
                series = q.profile_series(qwen_root=root,llama_root=leo,llama_base_root=base,mask_root=mask,partition=partition,version="v0",alpha_spec=alpha)
                models = {"human":"human","base":q.QWEN_BASE,"SFT":q.qwen_sft_model(partition)}
            else:
                series = m.profile_series(mistral_root=root,base_root=base,mask_root=mask,partition=partition,version="v0",alpha_spec=alpha)
                models = {"human":"human","base":m.MISTRAL_BASE,"SFT":m.sft_model(partition)}
            trajectories = {model:{kind:r.per_group_sd_trajectory(rows,kind=kind) for kind in ("public","private")} for model,rows in series.items()}
            trajectories = r.anchor_trajectories_to_human(trajectories)
            for channel in ("public","private"):
                for role,model in models.items():
                    means,_ = trajectories[model][channel]
                    convergence.append({"family":family,"split":partition,"channel":channel,"role":role,"model":model,"mean_delta_sd":means[-1]-means[0]})
            for role in ("base","SFT"):
                source = next(row for row in raw if row["model"] == models[role])
                errors.append({"family":family,"split":partition,"role":role,"model":models[role],**{metric:float(source[metric]) for metric in metrics}})
        r.write_csv(out/(family.lower()+"_raw_metrics.csv"),all_raw)
        r.write_csv(out/(family.lower()+"_coverage.csv"),all_status)
    r.write_csv(out/"cross_model_convergence.csv",convergence)
    r.write_csv(out/"cross_model_private_errors.csv",errors)
    expected = json.loads((HERE/"reference/cross_model_table_values.json").read_text())
    checks=[]
    for item in expected:
        source = convergence if item["metric"] == "mean_delta_sd" else errors
        found = [row for row in source if all(row[key]==item[key] for key in ("family","split","role")) and ("channel" not in row or row["channel"]==item.get("channel"))]
        assert len(found)==1,(item,found)
        value = float(found[0][item["metric"]])
        checks.append({"family":item["family"],"split":item["split"],"role":item["role"],"metric":item["metric"],"expected":item["displayed"],"actual":f"{value:.3f}","match":f"{value:.3f}"==item["displayed"]})
    write_json(out/"cross_model_verification.json",{"passed":all(x["match"] for x in checks),"checks":checks})
    if not all(x["match"] for x in checks):
        raise RuntimeError("Cross-model table values differ from the paper; inspect tables/cross_model_verification.json")
    for name, data, cols in [("cross_model_convergence",convergence,("family","split","channel","role","mean_delta_sd")),("cross_model_private_errors",errors,("family","split","role",*metrics))]:
        lines=[r"\begin{tabular}{"+"l"*len(cols)+"}",r"\toprule"," & ".join(key.replace("_",r"\_") for key in cols)+r" \\",r"\midrule"]
        for row in data:
            lines.append(" & ".join(f"{row[key]:.3f}" if isinstance(row[key],float) else str(row[key]).replace("_",r"\_") for key in cols)+r" \\")
        lines.extend([r"\bottomrule",r"\end{tabular}"])
        (out/(name+".tex")).write_text("\n".join(lines)+"\n")


def run_tables(args):
    args.output_dir.joinpath("tables").mkdir(parents=True,exist_ok=True)
    inventory(args)
    cross_model(args)
