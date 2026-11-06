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


ERROR_METRICS = {
    channel: tuple(channel + suffix for suffix in (
        "_stance_alignment_abs", "_stance_change_error_abs", "_opinion_convergence_error"
    ))
    for channel in ("public", "private")
}
LLAMA_MODELS = {
    "group_test": "ft:Llama-3.1-8B-Instruct-SFT-20260608-bd-real:bd-group-5epochs",
    "topic_test": "ft:Llama-3.1-8B-Instruct-SFT-20260608-bd-real:bd-topic-5epochs",
}


def convergence_rows(family, partition, models, trajectories, human_initial):
    """Return paper rows and the unchanged family-specific reference values.

    Each trajectory contains mean within-group SDs and their standard errors.
    Preserve its raw initial and final means; only the descriptive table's
    subtraction uses the main-paper human initial mean for the split/type.
    """
    paper, original = [], []
    for channel in ("public", "private"):
        family_initial = trajectories[models["human"]][channel][0][0]
        for role, model in models.items():
            means, _ = trajectories[model][channel]
            initial, final = means[0], means[-1]
            if not all(math.isfinite(x) for x in (initial, final, family_initial, human_initial[channel])):
                raise ValueError(f"Missing convergence endpoints: {family}/{partition}/{channel}/{role}")
            identity = {"family": family, "split": partition, "channel": channel, "role": role, "model": model}
            original.append({**identity, "raw_initial_mean_sd": initial, "final_mean_sd": final,
                             "family_human_initial_mean_sd": family_initial,
                             "mean_delta_sd": final - family_initial,
                             "paper_human_initial_mean_sd": human_initial[channel]})
            if role != "human" or family == "Llama":
                if role == "human":
                    identity["family"] = "Human"
                paper.append({**identity, "mean_delta_sd": final - human_initial[channel]})
    return paper, original


def prediction_error_rows(family, partition, models, raw):
    rows = {channel: [] for channel in ERROR_METRICS}
    for role in ("base", "SFT"):
        matches = [row for row in raw if row["model"] == models[role] and row["simulation_version"] == "v0"]
        if len(matches) != 1:
            raise ValueError(f"Expected one Full Dynamics row for {family}/{partition}/{role}, found {len(matches)}")
        source = matches[0]
        for channel, metrics in ERROR_METRICS.items():
            rows[channel].append({"family": family, "split": partition, "channel": channel,
                                  "role": role, "model": models[role],
                                  **{metric: float(source[metric]) for metric in metrics}})
    return rows


def primary_trajectories(rows, models):
    """Read raw trajectories, never model labels/values from a reference file."""
    steps = {"public": ("Tweet r1", "Tweet r2", "Tweet r3"), "private": ("Initial", "Post")}
    result = {}
    for model in models.values():
        result[model] = {}
        for channel, order in steps.items():
            selected = [row for row in rows if row["panel"] == channel + "_trajectory" and row["model"] == model]
            by_step = {row["step"]: row for row in selected}
            if len(selected) != len(order) or set(by_step) != set(order):
                raise ValueError(f"Incomplete primary trajectory: {model}/{channel}")
            result[model][channel] = (
                [float(by_step[step]["raw_mean_sd"]) for step in order],
                [float(by_step[step]["raw_se"]) for step in order],
            )
    return result


def write_cross_model_tex(out, convergence, errors):
    names = {"Llama": r"\texttt{Llama-3.1-8B}", "Qwen": r"\texttt{Qwen3.5-4B}", "Mistral": r"\texttt{Mistral-7B}"}
    lookup = {(row["family"], row["split"], row["channel"], row["role"]): row["mean_delta_sd"] for row in convergence}
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             "Split & Opinion type & Human & " + " & ".join(names.values()) + r" \\", r"\midrule"]
    for partition in ("group_test", "topic_test"):
        if partition == "topic_test":
            lines.append(r"\midrule")
        for channel in ("public", "private"):
            cells = [partition.split("_")[0].title(), channel.title(), f'${lookup[("Human", partition, channel, "human")]:.3f}$']
            cells.extend(f'${lookup[(family, partition, channel, "base")]:.3f} \\rightarrow {lookup[(family, partition, channel, "SFT")]:.3f}$' for family in names)
            lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (out / "cross_model_convergence.tex").write_text("\n".join(lines) + "\n")
    for channel, metrics in ERROR_METRICS.items():
        lookup = {(row["family"], row["split"], row["role"]): row for row in errors[channel]}
        lines = [r"\begin{tabular}{llrrr}", r"\toprule",
                 r"Model & Split & Single-opinion & Opinion-update & Opinion-convergence \\", r"\midrule"]
        for family in names:
            if family != "Llama":
                lines.append(r"\midrule")
            for partition in ("group_test", "topic_test"):
                base, sft = (lookup[(family, partition, role)] for role in ("base", "SFT"))
                cells = [names[family], partition.split("_")[0].title()]
                cells.extend(f'${base[metric]:.3f} \\rightarrow {sft[metric]:.3f}$' for metric in metrics)
                lines.append(" & ".join(cells) + r" \\")
        lines.extend([r"\bottomrule", r"\end{tabular}"])
        (out / f"cross_model_{channel}_errors.tex").write_text("\n".join(lines) + "\n")


def verify_cross_model_tables(convergence, errors, expected):
    """Check every table cell, rejecting incomplete or duplicate result sets."""
    fields = ("family", "split", "channel", "role", "metric")
    contract = set()
    for split in LLAMA_MODELS:
        for channel, metrics in ERROR_METRICS.items():
            contract.add(("Human", split, channel, "human", "mean_delta_sd"))
            for family in ("Llama", "Qwen", "Mistral"):
                for role in ("base", "SFT"):
                    contract.update((family, split, channel, role, metric) for metric in ("mean_delta_sd", *metrics))

    references = {}
    for item in expected:
        key = tuple(item[field] for field in fields)
        if key in references:
            raise ValueError(f"Duplicate cross-model reference cell: {key}")
        references[key] = item["displayed"]
    if set(references) != contract:
        raise ValueError(f"Cross-model reference must cover all 100 paper cells: missing={sorted(contract - set(references))}, extra={sorted(set(references) - contract)}")

    actual = {}
    for rows, metrics in [(convergence, ("mean_delta_sd",)), *[(errors[channel], metrics) for channel, metrics in ERROR_METRICS.items()]]:
        for row in rows:
            for metric in metrics:
                key = tuple(row[field] for field in fields[:-1]) + (metric,)
                if key in actual:
                    raise ValueError(f"Duplicate cross-model output cell: {key}")
                actual[key] = float(row[metric])
    if set(actual) != contract:
        raise ValueError(f"Cross-model output must cover all 100 paper cells: missing={sorted(contract - set(actual))}, extra={sorted(set(actual) - contract)}")
    return [{**dict(zip(fields, key)), "expected": references[key], "actual": f"{actual[key]:.3f}",
             "match": math.isfinite(actual[key]) and f"{actual[key]:.3f}" == references[key]} for key in references]


def cross_model(args):
    r = load_renderer()
    inputs = input_paths(args.data_root)["inputs"]
    out = args.output_dir / "tables"
    outputs_path = args.output_dir / "figure_outputs.json"
    require([outputs_path])
    figure_paths = json.loads(outputs_path.read_text())
    convergence, family_references, shared_initial = [], [], {}
    errors = {channel: [] for channel in ERROR_METRICS}
    for partition in ("group_test", "topic_test"):
        group = partition == "group_test"
        conv = read_csv(args.output_dir / figure_paths["M3" if group else "A5"])
        align = read_csv(args.output_dir / figure_paths["M1" if group else "A3"])
        models = {"human": "human", "base": "Llama-3.1-8B-Instruct", "SFT": LLAMA_MODELS[partition]}
        trajectories = primary_trajectories(conv, models)
        shared_initial[partition] = {channel: trajectories["human"][channel][0][0] for channel in ERROR_METRICS}
        paper, original = convergence_rows("Llama", partition, models, trajectories, shared_initial[partition])
        convergence.extend(paper)
        family_references.extend(original)
        computed_errors = prediction_error_rows("Llama", partition, models, align)
        for channel in errors:
            errors[channel].extend(computed_errors[channel])

    q = importlib.import_module("generate_v5_leo_qwen4b_comparison")
    for family in ("Qwen", "Mistral"):
        if family == "Qwen":
            root, mask = inputs / "qwen/human_llm", inputs / "qwen/masks"
            q.configure_renderer()
        else:
            m = importlib.import_module("generate_v5_ruixuan_mistral_sft_comparison")
            m.configure_shared_helpers()
            m.configure_renderer()
            root, mask = inputs / "mistral/human_llm", inputs / "llama/data/sft_data/masks"
        base, leo = inputs / "base/human_llm", inputs / "leo_llama/human_llm"
        require([root, mask, base, leo])
        all_raw, all_status = [], []
        for partition in ("group_test", "topic_test"):
            print(f"Recompute {family} paper tables: {partition}", flush=True)
            alpha = q.fit_alphas(qwen_root=root, mask_root=mask, partition=partition.replace("_test", "_train"), version="v0")
            raw, status = q.compute_partition_rows(qwen_root=root, llama_root=leo, llama_base_root=base, mask_root=mask, partition=partition, version="v0", alpha_spec=alpha, bootstrap_reps=0)
            for row in raw:
                row.update(partition=partition, simulation_version="v0")
            for row in status:
                row["partition"] = partition
            all_raw.extend(raw)
            all_status.extend(status)
            if family == "Qwen":
                series = q.profile_series(qwen_root=root, llama_root=leo, llama_base_root=base, mask_root=mask, partition=partition, version="v0", alpha_spec=alpha)
                models = {"human": "human", "base": q.QWEN_BASE, "SFT": q.qwen_sft_model(partition)}
            else:
                series = m.profile_series(mistral_root=root, base_root=base, mask_root=mask, partition=partition, version="v0", alpha_spec=alpha)
                models = {"human": "human", "base": m.MISTRAL_BASE, "SFT": m.sft_model(partition)}
            trajectories = {model: {kind: r.per_group_sd_trajectory(series[model], kind=kind) for kind in ERROR_METRICS} for model in models.values()}
            paper, original = convergence_rows(family, partition, models, trajectories, shared_initial[partition])
            convergence.extend(paper)
            family_references.extend(original)
            computed_errors = prediction_error_rows(family, partition, models, raw)
            for channel in errors:
                errors[channel].extend(computed_errors[channel])
        r.write_csv(out / (family.lower() + "_raw_metrics.csv"), all_raw)
        r.write_csv(out / (family.lower() + "_coverage.csv"), all_status)
    r.write_csv(out / "cross_model_convergence.csv", convergence)
    r.write_csv(out / "cross_model_convergence_family_references.csv", family_references)
    for channel, rows in errors.items():
        r.write_csv(out / f"cross_model_{channel}_errors.csv", rows)
    write_json(out / "cross_model_provenance.json", {
        "simulation": "v0",
        "convergence_formula": "model final mean within-group SD - main-paper human initial mean within-group SD",
        "human_initial_mean_sd": shared_initial,
        "human_table_rows": "One main-paper human change per split and public/private opinion type",
        "original_family_references": "cross_model_convergence_family_references.csv",
        "prediction_errors": "Original model-family inclusion and validity rules; no rescoring or common-reference adjustment",
        "llama_sft_models": LLAMA_MODELS,
        "llama_input_figures": {key: figure_paths[key] for key in ("M1", "M3", "A3", "A5")},
    })
    expected = json.loads((HERE / "reference/cross_model_table_values.json").read_text())
    checks = verify_cross_model_tables(convergence, errors, expected)
    write_json(out / "cross_model_verification.json", {"passed": all(x["match"] for x in checks), "checks": checks})
    if not all(x["match"] for x in checks):
        raise RuntimeError("Cross-model table values differ from the paper; inspect tables/cross_model_verification.json")
    write_cross_model_tex(out, convergence, errors)


def run_tables(args):
    args.output_dir.joinpath("tables").mkdir(parents=True, exist_ok=True)
    inventory(args)
    cross_model(args)
