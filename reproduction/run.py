"""Regenerate DEBATE camera-ready results from the released cached observations."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor"
SCRIPTS = VENDOR / "neurips_2026/sean_reproduce/scripts"


def load_renderer():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "debate-matplotlib"))
    for path in (VENDOR, SCRIPTS):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import generate_v5_ruixuan_paper_figures as renderer
    from reproduction.presentation import configure
    configure(renderer)
    from reproduction.numerics import configure as configure_numerics
    configure_numerics()
    return renderer


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def input_paths(root):
    inputs = root / "reproduction/inputs"
    return {
        "result_root": inputs / "llama/human_llm",
        "base_result_root": inputs / "base/human_llm",
        "data_root": inputs / "llama",
        "inputs": inputs,
    }


def require(paths):
    missing = [str(path) for path in paths if not Path(path).exists()]
    if missing:
        raise SystemExit("Missing released inputs. Download and extract the data bundles first:\n" + "\n".join(missing))


def primary(args):
    r = load_renderer()
    paths = input_paths(args.data_root)
    inputs = paths.pop("inputs")
    require([*paths.values(), inputs / "base/text_metrics/maintable.csv"])
    r.beyond_stance_style.MAINTABLE_FILES["breadth"] = inputs / "base/text_metrics/maintable.csv"
    out = args.output_dir / "primary"
    csv_dir = out / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    partitions = ["group_test", "topic_test"]
    if args.include_train:
        partitions = list(r.PARTITION_SPECS)
    summary_path = csv_dir / "summary_metrics.csv"
    alpha_path = csv_dir / "degroot_fit_alphas.csv"
    status_path = csv_dir / "per_prefix_status.csv"
    if args.reuse_summary:
        require([summary_path, alpha_path, status_path])
        rows = [row for row in read_csv(summary_path) if row.get("model") not in {"Mean Predictor", "Uniform LB"}]
        alphas = r.read_alpha_specs(alpha_path)
        if set(partitions) - {row["partition"] for row in rows}:
            raise SystemExit("Existing summary does not cover the requested partitions.")
    else:
        alphas, fit_cache, rows, statuses = {}, {}, [], []
        for partition in partitions:
            spec = r.PARTITION_SPECS[partition]
            fit_partition = str(spec["fit_partition"])
            for version in r.VERSION_SPECS:
                key = (fit_partition, version)
                if key not in fit_cache:
                    print(f"Fit DeGroot: {fit_partition}/{version}", flush=True)
                    fit_cache[key] = r.fit_alphas(result_root=paths["result_root"], data_root=paths["data_root"], partition=fit_partition, version=version)
                alphas[(partition, version)] = fit_cache[key]
                print(f"Recompute cached-label metrics: {partition}/{version}", flush=True)
                part_rows, part_statuses = r.compute_raw_rows(
                    sft_result_root=paths["result_root"], base_result_root=paths["base_result_root"],
                    prefixes=r.read_prefixes(paths["data_root"], partition), sft_model=str(spec["model"]),
                    version=version, alpha_spec=fit_cache[key], bootstrap_reps=args.bootstrap_reps,
                )
                # The paper assembled one cache per actual model; synthetic
                # mean/uniform diagnostics were not included in its figure CSVs.
                part_rows = [row for row in part_rows if row.get("model") in {*r.BASE_MODELS, str(spec["model"]), r.DEGROOT_MODEL}]
                for row in part_rows:
                    model = str(row.get("model", ""))
                    row.update(partition=partition, simulation_version=version, model_label=r.model_label(model),
                               fit_alpha_public=fit_cache[key]["alpha_public"], fit_alpha_private=fit_cache[key]["alpha_private"],
                               evaluation_source="v5 SFT result root" if model == spec["model"] else "reused eval_v3 base result root" if model in r.BASE_MODELS else "v5 human-surface DeGroot fit")
                for status in part_statuses:
                    status["partition"] = partition
                rows.extend(part_rows)
                statuses.extend(part_statuses)
                r.write_csv(summary_path, rows)
                r.write_csv(status_path, statuses)
                r.write_csv(alpha_path, [{"partition": p, "simulation_version": v, "fit_partition": r.PARTITION_SPECS[p]["fit_partition"], **values} for (p,v),values in alphas.items()])

    from reproduction.histograms import render_full_histogram
    outputs = {}
    for partition in partitions:
        if not partition.endswith("test"):
            continue
        group = partition.startswith("group")
        dest = out / (partition + "_set")
        dest.mkdir(parents=True, exist_ok=True)
        families = {"alignment": "M1" if group else "A3", "correlations": "M2" if group else "A4",
                    "convergence": "M3" if group else "A5", "transitions": "M4" if group else "A8",
                    "partner_gap": "M5" if group else "A9", "text": "A1" if group else "A2",
                    "histogram": "A6" if group else "A7"}
        print(f"Render paper figures: {partition}", flush=True)
        common = {**paths, "partition": partition, "alpha_spec": alphas[(partition,"v0")]}
        calls = [
            (families["alignment"], r.render_alignment_bars, {"rows":rows,"partition":partition}),
            (families["correlations"], r.render_correlations, {"rows":rows,"partition":partition}),
            (families["convergence"], r.render_convergence_composite, common),
            (families["transitions"], r.render_transition_panel, common),
            (families["partner_gap"], r.render_partner_gap_composite, {**common,"bootstrap_reps":args.bootstrap_reps}),
            (families["text"], r.render_beyond_stance, {"result_root":paths["result_root"],"partition":partition}),
        ]
        for family, function, kwargs in calls:
            # The publication correlation helper selects the topic panel scale
            # from its filename; preserve that original convention explicitly.
            target = dest / ("A4_topic_test" if family == "A4" else family)
            function(**kwargs, output_path=target)
            outputs[family] = str(target.with_suffix(".csv").relative_to(args.output_dir))
        for kind in ("public", "private"):
            family = families["histogram"] + "_" + kind
            target = dest / family
            render_full_histogram(r, **common, kind=kind, output_path=target)
            outputs[family] = str(target.with_suffix(".csv").relative_to(args.output_dir))
        r.PROFILE_CACHE.clear()
        gc.collect()
    write_json(args.output_dir / "figure_outputs.json", outputs)
    print(f"Generated {len(outputs)} paper figure families from cached rows/declared text-metric aggregates.", flush=True)


def verify(args):
    require([args.output_dir / "figure_outputs.json"])
    outputs = json.loads((args.output_dir / "figure_outputs.json").read_text())
    mapping = json.loads((HERE / "reference/figure_mapping.json").read_text())
    checks, mismatches = [], []
    # Old figure CSVs retain stale fit metadata in non-DeGroot model rows.
    # This metadata is not plotted. The actual fitted coefficients are checked
    # separately against the values reported in the paper.
    ignored = {"fit_alpha_public", "fit_alpha_private"}
    stable_keys = {"model", "panel", "stage", "opinion_type", "simulation_version", "partition", "comparison_version", "scope", "metric_x", "metric_y"}
    def number(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    for family, spec in mapping.items():
        expected = read_csv(HERE / spec["reference_csv"])
        actual = read_csv(args.output_dir / outputs[family])
        if len(expected) != len(actual):
            mismatches.append({"family":family,"problem":"row_count","expected":len(expected),"actual":len(actual)})
            continue
        numeric_cells = 0
        for index, (old,new) in enumerate(zip(expected,actual)):
            for key, value in old.items():
                if key in ignored:
                    continue
                if key not in new:
                    # Presentation-only recovery metadata may be newer than the
                    # manuscript CSV. Every numeric source column must remain.
                    mismatches.append({"family":family,"row":index,"field":key,"problem":"missing_column"})
                    continue
                a,b = number(value),number(new[key])
                if a is not None and b is not None:
                    numeric_cells += 1
                    same = (math.isnan(a) and math.isnan(b)) or math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-10)
                elif (a is None) != (b is None):
                    same = False
                elif key in stable_keys:
                    same = value == new[key]
                else:
                    same = True
                if not same:
                    mismatches.append({"family":family,"row":index,"field":key,"expected":value,"actual":new[key]})
        checks.append({"family":family,"rows":len(actual),"numeric_cells_compared":numeric_cells})
    expected_alphas = {
        ("group_test","v0"): (0.8570889621358622,0.9238267389807363),
        ("group_test","v1_5"): (0.8652389307362671,0.9211676121232439),
        ("topic_test","v0"): (0.8663798616405309,0.9292895773654417),
        ("topic_test","v1_5"): (0.8689010145244372,0.9198973607589799),
    }
    alpha_rows = read_csv(args.output_dir/"primary/csv/degroot_fit_alphas.csv")
    for row in alpha_rows:
        key = (row["partition"],row["simulation_version"])
        if key in expected_alphas:
            for channel,expected_value in zip(("public","private"),expected_alphas[key]):
                actual_value = float(row["alpha_"+channel])
                if not math.isclose(actual_value,expected_value,abs_tol=1e-10):
                    mismatches.append({"family":"DeGroot coefficients","partition":key[0],"version":key[1],"channel":channel,"expected":expected_value,"actual":actual_value})
    result = {"passed":not mismatches,"figures":checks,"mismatches":mismatches,"note":"Checks numeric outputs; PDF/PNG byte equality is not required across plotting-library or font versions."}
    write_json(args.output_dir / "verification.json", result)
    print(json.dumps({"passed":result["passed"],"figures":len(checks),"mismatches":len(mismatches)},indent=2))
    if mismatches:
        raise SystemExit("Numeric reproduction differs from paper references; inspect verification.json.")


def demographics(args):
    source = input_paths(args.data_root)["inputs"] / "llama/data/sft_data/group_split_data_bd"
    require([source])
    out = args.output_dir / "demographics"
    subprocess.run([sys.executable,str(HERE/"demographics/aggregate_demographics.py"),"--source-root",str(source),"--output-dir",str(out)],check=True)
    actual = json.loads((out/"aggregates.json").read_text())
    reference = json.loads((HERE/"reference/demographics.json").read_text())
    mismatches = [key for key,value in reference.items() if actual.get(key)!=value]
    write_json(out/"verification.json",{"passed":not mismatches,"fields_compared":len(reference),"mismatches":mismatches})
    if mismatches:
        raise RuntimeError("Demographic aggregates differ from the paper; inspect demographics/verification.json")
    subprocess.run([sys.executable,str(HERE/"demographics/render_demographics.py"),"--aggregates",str(out/"aggregates.json"),"--output-dir",str(out)],check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("all","primary","tables","demographics","historical","verify"))
    parser.add_argument("--data-root",type=Path,required=True,help="Root populated by the DEBATE data downloader.")
    parser.add_argument("--output-dir",type=Path,default=Path("outputs/paper"))
    parser.add_argument("--bootstrap-reps",type=int,default=1000)
    parser.add_argument("--reuse-summary",action="store_true",help="Reuse this output directory's previously computed primary metric summaries.")
    parser.add_argument("--include-train",action="store_true",help="Also compute training-partition metric summaries (paper plots use test partitions).")
    args = parser.parse_args(argv)
    if args.bootstrap_reps < 0:
        parser.error("--bootstrap-reps must be nonnegative")
    args.data_root = args.data_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.stage in ("all","primary"):
        primary(args)
    if args.stage in ("all","tables"):
        from reproduction.tables import run_tables
        run_tables(args)
    if args.stage in ("all","demographics"):
        demographics(args)
    if args.stage in ("all","historical"):
        from reproduction.historical import run_historical
        run_historical(args)
    if args.stage in ("all","verify"):
        verify(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
