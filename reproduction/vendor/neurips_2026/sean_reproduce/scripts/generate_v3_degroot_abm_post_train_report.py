#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
SEAN_REPRODUCE_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SEAN_REPRODUCE_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

CHENGTAO_RADAR_DIR = REPO_ROOT / "neurips_2026" / "chengtao" / "src" / "radar"
if str(CHENGTAO_RADAR_DIR) not in sys.path:
    sys.path.insert(0, str(CHENGTAO_RADAR_DIR))

from neurips_2026.sean_reproduce.v3.masks import (  # noqa: E402
    V3_OPINION_GROUP_MASK_ENV_VAR,
    V3_TWEET_GROUP_MASK_ENV_VAR,
    configured_private_mask_root,
    configured_public_mask_root,
    enable_v3_complete_group_masks,
    wrapped_path_lines,
)

from generate_plots import HUMAN_DEFAULT_FILENAME, LLM_DEFAULT_FILENAME, read_csv_rows, read_lines, save_csv, safe_mean  # noqa: E402
from group_opinion_convergence import (  # noqa: E402
    POST_TRAIN_PARTITION_SPECS,
    build_report_pdf,
    collect_fieldnames,
    plot_convergence_rows,
    resolve_post_train_partition_models,
)
from post_training_breadth_profiles import MODEL_SET_CONFIGS  # noqa: E402
from neurips_2026.sean_reproduce.result_version_config import (  # noqa: E402
    human_csv_filename_for_result,
    llm_csv_filename_for_result,
)

from generate_v3_degroot_abm_report import (  # noqa: E402
    DEFAULT_BOOTSTRAP_REPS,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_DATA_ROOT,
    DEFAULT_INITIAL_POST_SIDECAR_ROOT,
    DEFAULT_MASK_NAMESPACE_ROOT,
    DEFAULT_OPINION_MASK_ROOT,
    DEFAULT_REPORT_VERSION,
    DEFAULT_TWEET_MASK_ROOT,
    DEFAULT_TWEET_SIDECAR_ROOT,
    collect_abm_rows_with_bootstrap,
    compute_degroot_prediction_error_lists,
    default_abm_output_dir,
    resolve_human_path,
)


DEFAULT_OUTPUT_DIR = (
    SEAN_REPRODUCE_ROOT / "v3" / "abm" / "group_opinion_convergence" / "post_train"
)
MODELS_VS_ALL_KEY = "post-training-vs-all"
MODELS_V4_VS_ALL_KEY = "post-training-v4-vs-all"
PAIR_ORDER = (
    "breadth_topic",
    "breadth_group",
    "depth_topic",
    "depth_group",
)


def models_vs_all_key_for_report_version(report_version: str) -> str:
    if str(report_version).strip().startswith("v4"):
        return MODELS_V4_VS_ALL_KEY
    return MODELS_VS_ALL_KEY


def resolve_partition_prefix_file(prefix_file: str) -> Path:
    path = Path(prefix_file)
    if path.is_absolute():
        return path
    return (CHENGTAO_RADAR_DIR / path).resolve()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit DeGroot public/private alphas on post-train train splits and evaluate on matching test splits for the group-convergence report."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--human-filename", default=HUMAN_DEFAULT_FILENAME)
    parser.add_argument("--llm-filename", default=LLM_DEFAULT_FILENAME)
    parser.add_argument("--likert-col", default="likert_pred")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--alpha-search-coarse-steps", type=int, default=20)
    parser.add_argument("--alpha-search-tol", type=float, default=1e-4)
    parser.add_argument("--alpha-search-max-iter", type=int, default=64)
    parser.add_argument(
        "--initial-post-validity-sidecar-root",
        action="append",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--tweet-validity-sidecar-root",
        action="append",
        type=Path,
        default=None,
    )
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--report-version", default=DEFAULT_REPORT_VERSION)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    return parser


def partition_pair_key(suffix: str) -> str:
    if suffix.endswith("_train"):
        return suffix[: -len("_train")]
    if suffix.endswith("_test"):
        return suffix[: -len("_test")]
    return suffix


def fit_objective_value(human_row: dict[str, Any], degroot_row: dict[str, Any]) -> float:
    public_h = float(human_row.get("public_mean_delta_sd", math.nan))
    private_h = float(human_row.get("private_mean_delta_sd", math.nan))
    public_d = float(degroot_row.get("public_mean_delta_sd", math.nan))
    private_d = float(degroot_row.get("private_mean_delta_sd", math.nan))
    if any(math.isnan(value) for value in (public_h, private_h, public_d, private_d)):
        return math.inf
    return (public_d - public_h) ** 2 + (private_d - private_h) ** 2


def build_fit_spec(alpha_public: float, alpha_private: float, model_name: str) -> dict[str, Any]:
    return {
        "model": model_name,
        "alpha_public": float(alpha_public),
        "alpha_private": float(alpha_private),
    }


def load_human_fit_rows(
    *,
    roots: list[Path],
    prefixes: list[str],
    human_filename: str,
    likert_col: str,
    initial_post_sidecar_roots: list[Path],
    tweet_sidecar_roots: list[Path],
) -> dict[str, list[dict[str, str]]]:
    rows_by_prefix: dict[str, list[dict[str, str]]] = {}
    for data_prefix in prefixes:
        human_path = resolve_human_path(
            roots=roots,
            data_prefix=data_prefix,
            models=[],
            human_filename=human_filename,
        )
        if human_path is None:
            continue
        rows_by_prefix[data_prefix] = read_csv_rows(
            human_path,
            respect_validity_labels=True,
            likert_col=likert_col,
            data_roots=roots,
            initial_post_sidecar_roots=initial_post_sidecar_roots,
            tweet_sidecar_roots=tweet_sidecar_roots,
        )
    return rows_by_prefix


def collect_fit_error_lists(
    *,
    rows_by_prefix: dict[str, list[dict[str, str]]],
    likert_col: str,
    alpha_public: float,
    alpha_private: float,
) -> dict[str, list[float]]:
    public_tweet2_errors: list[float] = []
    public_tweet3_errors: list[float] = []
    private_final_errors: list[float] = []
    for data_prefix, human_rows in rows_by_prefix.items():
        tweet2_errors, tweet3_errors, private_errors = compute_degroot_prediction_error_lists(
            human_rows,
            data_prefix=data_prefix,
            likert_col=likert_col,
            alpha_public=alpha_public,
            alpha_private=alpha_private,
        )
        public_tweet2_errors.extend(tweet2_errors)
        public_tweet3_errors.extend(tweet3_errors)
        private_final_errors.extend(private_errors)
    return {
        "public_tweet2_errors": public_tweet2_errors,
        "public_tweet3_errors": public_tweet3_errors,
        "private_final_errors": private_final_errors,
    }


def pooled_mean(values: list[float]) -> float:
    return float(safe_mean(values))


def summarize_fit_errors(error_lists: dict[str, list[float]]) -> dict[str, float]:
    public_tweet2_errors = error_lists["public_tweet2_errors"]
    public_tweet3_errors = error_lists["public_tweet3_errors"]
    private_final_errors = error_lists["private_final_errors"]
    public_errors = [*public_tweet2_errors, *public_tweet3_errors]
    public_tweet2_mse = pooled_mean(public_tweet2_errors)
    public_tweet3_mse = pooled_mean(public_tweet3_errors)
    public_pooled_mse = pooled_mean(public_errors)
    private_final_mse = pooled_mean(private_final_errors)
    combined_mse = pooled_mean([*public_errors, *private_final_errors])
    return {
        "public_tweet2_mse": public_tweet2_mse,
        "public_tweet3_mse": public_tweet3_mse,
        "public_pooled_mse": public_pooled_mse,
        "private_final_mse": private_final_mse,
        "combined_mse": combined_mse,
        "public_tweet2_count": float(len(public_tweet2_errors)),
        "public_tweet3_count": float(len(public_tweet3_errors)),
        "public_count": float(len(public_errors)),
        "private_count": float(len(private_final_errors)),
        "combined_count": float(len(public_errors) + len(private_final_errors)),
    }


def is_better_minimum(
    score: float,
    point: float,
    best_score: float,
    best_point: float,
    *,
    tie_tol: float = 1e-12,
) -> bool:
    if score < best_score - tie_tol:
        return True
    if abs(score - best_score) <= tie_tol and point < best_point:
        return True
    return False


def bounded_minimize_1d(
    objective: Any,
    *,
    lower: float = 0.0,
    upper: float = 1.0,
    coarse_steps: int = 20,
    tol: float = 1e-4,
    max_iter: int = 64,
    max_candidates: int = 4,
) -> tuple[float, float]:
    if coarse_steps < 1:
        raise ValueError("coarse_steps must be >= 1")
    if lower > upper:
        raise ValueError("lower bound must be <= upper bound")

    phi = (1.0 + math.sqrt(5.0)) / 2.0
    grid = [lower + (upper - lower) * (idx / coarse_steps) for idx in range(coarse_steps + 1)]
    grid_scores = [(float(objective(point)), point) for point in grid]
    best_score, best_point = min(grid_scores, key=lambda item: item[0])
    candidate_indices: set[int] = {0, coarse_steps}
    for idx in range(1, coarse_steps):
        current_score = grid_scores[idx][0]
        if current_score <= grid_scores[idx - 1][0] and current_score <= grid_scores[idx + 1][0]:
            candidate_indices.add(idx)
    candidate_indices.update(
        idx for idx, _item in sorted(enumerate(grid_scores), key=lambda item: (item[1][0], item[1][1]))[:max_candidates]
    )

    for candidate_index in sorted(
        candidate_indices,
        key=lambda idx: (grid_scores[idx][0], grid_scores[idx][1]),
    )[:max_candidates]:
        left = grid[max(0, candidate_index - 1)]
        right = grid[min(coarse_steps, candidate_index + 1)]
        if left == right:
            continue

        c = right - (right - left) / phi
        d = left + (right - left) / phi
        fc = float(objective(c))
        fd = float(objective(d))
        if is_better_minimum(fc, c, best_score, best_point):
            best_point, best_score = c, fc
        if is_better_minimum(fd, d, best_score, best_point):
            best_point, best_score = d, fd

        for _ in range(max_iter):
            if abs(right - left) <= tol:
                break
            if fc <= fd:
                right = d
                d = c
                fd = fc
                c = right - (right - left) / phi
                fc = float(objective(c))
                if is_better_minimum(fc, c, best_score, best_point):
                    best_point, best_score = c, fc
            else:
                left = c
                c = d
                fc = fd
                d = left + (right - left) / phi
                fd = float(objective(d))
                if is_better_minimum(fd, d, best_score, best_point):
                    best_point, best_score = d, fd

        for point in (left, right, (left + right) / 2.0):
            score = float(objective(point))
            if is_better_minimum(score, point, best_score, best_point):
                best_point, best_score = point, score

    for point in (lower, upper):
        score = float(objective(point))
        if is_better_minimum(score, point, best_score, best_point):
            best_point, best_score = point, score
    return best_point, best_score


def fit_best_alphas(
    *,
    roots: list[Path],
    prefixes: list[str],
    human_filename: str,
    llm_filename: str,
    likert_col: str,
    initial_post_sidecar_roots: list[Path],
    tweet_sidecar_roots: list[Path],
    coarse_steps: int,
    tol: float,
    max_iter: int,
) -> dict[str, float]:
    rows_by_prefix = load_human_fit_rows(
        roots=roots,
        prefixes=prefixes,
        human_filename=human_filename,
        likert_col=likert_col,
        initial_post_sidecar_roots=initial_post_sidecar_roots,
        tweet_sidecar_roots=tweet_sidecar_roots,
    )

    def public_objective(alpha_public: float) -> float:
        summary = summarize_fit_errors(
            collect_fit_error_lists(
                rows_by_prefix=rows_by_prefix,
                likert_col=likert_col,
                alpha_public=alpha_public,
                alpha_private=1.0,
            )
        )
        return summary["public_pooled_mse"]

    alpha_public, public_score = bounded_minimize_1d(
        public_objective,
        coarse_steps=coarse_steps,
        tol=tol,
        max_iter=max_iter,
    )

    def private_objective(alpha_private: float) -> float:
        summary = summarize_fit_errors(
            collect_fit_error_lists(
                rows_by_prefix=rows_by_prefix,
                likert_col=likert_col,
                alpha_public=alpha_public,
                alpha_private=alpha_private,
            )
        )
        return summary["private_final_mse"]

    alpha_private, private_score = bounded_minimize_1d(
        private_objective,
        coarse_steps=coarse_steps,
        tol=tol,
        max_iter=max_iter,
    )

    final_summary = summarize_fit_errors(
        collect_fit_error_lists(
            rows_by_prefix=rows_by_prefix,
            likert_col=likert_col,
            alpha_public=alpha_public,
            alpha_private=alpha_private,
        )
    )
    return {
        "alpha_public": alpha_public,
        "alpha_private": alpha_private,
        "public_objective": public_score,
        "private_objective": private_score,
        "combined_objective": final_summary["combined_mse"],
        **final_summary,
    }


def fitted_row_name(alpha_public: float, alpha_private: float, pair_key: str) -> str:
    pair_label = pair_key.replace("_", " ")
    return (
        f"DeGroot fit on {pair_label} train "
        f"(alpha_public={alpha_public:.3f}, alpha_private={alpha_private:.3f})"
    )


def subset_title(partition_label: str, alpha_public: float, alpha_private: float) -> str:
    return (
        f"Post-Train Models vs All - {partition_label} "
        f"(fit alpha_public={alpha_public:.3f}, alpha_private={alpha_private:.3f})"
    )


def build_cover_body(fit_results: dict[str, dict[str, float]]) -> str:
    lines = [
        "This report fits separate DeGroot alpha_public and alpha_private values per train/test split family using the train partition only, then evaluates that fitted pair on both the matching train and test partitions.",
        "",
        "Fit objective:",
        "- Fit alpha_public on [0, 1] by minimizing pooled per-agent MSE over human tweet round 2 and tweet round 3 stances.",
        "- Fit alpha_private on [0, 1] by minimizing per-agent MSE on human final post-opinion stances, conditional on the fitted alpha_public public-message trajectory.",
        "- Report tweet-2, tweet-3, public pooled, private final, and combined per-agent MSE at the fitted alpha pair.",
        "",
        "Chosen alphas:",
    ]
    for pair_key in PAIR_ORDER:
        result = fit_results[pair_key]
        lines.append(
            "- "
            f"{pair_key.replace('_', ' ')}: "
            f"alpha_public={result['alpha_public']:.3f}, "
            f"alpha_private={result['alpha_private']:.3f}, "
            f"tweet2 MSE={result['public_tweet2_mse']:.6f}, "
            f"tweet3 MSE={result['public_tweet3_mse']:.6f}, "
            f"public pooled MSE={result['public_pooled_mse']:.6f}, "
            f"private final MSE={result['private_final_mse']:.6f}, "
            f"combined MSE={result['combined_mse']:.6f}"
        )
    lines.extend(
        [
            "",
            "Mask setup:",
            *wrapped_path_lines(configured_public_mask_root(), indent="  Public/tweet: "),
            *wrapped_path_lines(configured_private_mask_root(), indent="  Private/opinion: "),
        ]
    )
    return "\n".join(lines)


def report_output_name(report_version: str) -> str:
    return f"report_group_opinion_convergence_posttrain_models_vs_all_degroot_fit_{report_version}.pdf"


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.initial_post_validity_sidecar_root is None:
        args.initial_post_validity_sidecar_root = [DEFAULT_INITIAL_POST_SIDECAR_ROOT]
    if args.tweet_validity_sidecar_root is None:
        args.tweet_validity_sidecar_root = [DEFAULT_TWEET_SIDECAR_ROOT]

    if args.human_filename == HUMAN_DEFAULT_FILENAME:
        args.human_filename = human_csv_filename_for_result(args.report_version)
    if args.llm_filename == LLM_DEFAULT_FILENAME:
        args.llm_filename = llm_csv_filename_for_result(args.report_version)

    enable_v3_complete_group_masks(DEFAULT_MASK_NAMESPACE_ROOT)
    os.environ[V3_TWEET_GROUP_MASK_ENV_VAR] = str(DEFAULT_TWEET_MASK_ROOT)
    os.environ[V3_OPINION_GROUP_MASK_ENV_VAR] = str(DEFAULT_OPINION_MASK_ROOT)

    roots = [args.data_root.resolve()]
    initial_post_sidecar_roots = [path.resolve() for path in args.initial_post_validity_sidecar_root]
    tweet_sidecar_roots = [path.resolve() for path in args.tweet_validity_sidecar_root]
    output_dir = args.output_dir.resolve() if args.output_dir is not None else default_abm_output_dir(args.report_version, "group_opinion_convergence", "post_train").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    img_dir = output_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = output_dir / "csv"
    if not args.report_only:
        csv_dir.mkdir(parents=True, exist_ok=True)

    model_set_key = models_vs_all_key_for_report_version(args.report_version)
    model_set_cfg = MODEL_SET_CONFIGS[model_set_key]
    partition_specs: list[tuple[str, list[str], list[str], str]] = []
    fit_results: dict[str, dict[str, float]] = {}

    for suffix, prefix_file, model_attr, partition_label in POST_TRAIN_PARTITION_SPECS:
        prefixes = read_lines(resolve_partition_prefix_file(prefix_file))
        models = resolve_post_train_partition_models(model_set_key=model_set_key, suffix=suffix)
        if models is None:
            models = getattr(model_set_cfg, model_attr)
        partition_specs.append((suffix, prefixes, models, partition_label))
        pair_key = partition_pair_key(suffix)
        if suffix.endswith("_train"):
            fit_results[pair_key] = fit_best_alphas(
                roots=roots,
                prefixes=prefixes,
                human_filename=args.human_filename,
                llm_filename=args.llm_filename,
                likert_col=args.likert_col,
                initial_post_sidecar_roots=initial_post_sidecar_roots,
                tweet_sidecar_roots=tweet_sidecar_roots,
                coarse_steps=args.alpha_search_coarse_steps,
                tol=args.alpha_search_tol,
                max_iter=args.alpha_search_max_iter,
            )

    figures: list[tuple[str, Path]] = []
    for suffix, prefixes, models, partition_label in partition_specs:
        pair_key = partition_pair_key(suffix)
        fit_result = fit_results[pair_key]
        fit_alpha_public = fit_result["alpha_public"]
        fit_alpha_private = fit_result["alpha_private"]
        rows = collect_abm_rows_with_bootstrap(
            roots=roots,
            prefixes=prefixes,
            models=models,
            alphas=None,
            human_filename=args.human_filename,
            llm_filename=args.llm_filename,
            likert_col=args.likert_col,
            no_progress=args.no_progress,
            initial_post_sidecar_roots=initial_post_sidecar_roots,
            tweet_sidecar_roots=tweet_sidecar_roots,
            bootstrap_reps=args.bootstrap_reps,
            bootstrap_seed=args.bootstrap_seed,
            abm_specs=[
                build_fit_spec(
                    fit_alpha_public,
                    fit_alpha_private,
                    fitted_row_name(fit_alpha_public, fit_alpha_private, pair_key),
                )
            ],
        )
        for row in rows:
            if row.get("model") == fitted_row_name(fit_alpha_public, fit_alpha_private, pair_key):
                row["fit_alpha_public"] = fit_alpha_public
                row["fit_alpha_private"] = fit_alpha_private
                row["fit_pair_key"] = pair_key
                row["fit_public_tweet2_mse"] = fit_result["public_tweet2_mse"]
                row["fit_public_tweet3_mse"] = fit_result["public_tweet3_mse"]
                row["fit_public_pooled_mse"] = fit_result["public_pooled_mse"]
                row["fit_private_final_mse"] = fit_result["private_final_mse"]
                row["fit_combined_mse"] = fit_result["combined_mse"]
                row["fit_public_observations"] = int(fit_result["public_count"])
                row["fit_private_observations"] = int(fit_result["private_count"])
            else:
                row.setdefault("fit_alpha_public", "")
                row.setdefault("fit_alpha_private", "")
                row.setdefault("fit_pair_key", "")
                row.setdefault("fit_public_tweet2_mse", "")
                row.setdefault("fit_public_tweet3_mse", "")
                row.setdefault("fit_public_pooled_mse", "")
                row.setdefault("fit_private_final_mse", "")
                row.setdefault("fit_combined_mse", "")
                row.setdefault("fit_public_observations", "")
                row.setdefault("fit_private_observations", "")

        base_name = f"group_opinion_convergence_models_vs_all_{suffix}_degroot_fit_{args.report_version}"
        csv_path = csv_dir / f"{base_name}.csv"
        png_path = img_dir / f"{base_name}.png"
        title = subset_title(partition_label, fit_alpha_public, fit_alpha_private)
        if not args.report_only:
            save_csv(csv_path, rows, collect_fieldnames(rows))
            print(f"Wrote: {csv_path}")
        plot_convergence_rows(rows, out_png=png_path, title=title, model_order=models, write_png=not args.report_only)
        if not args.report_only:
            print(f"Wrote: {png_path}")
        print(f"Wrote: {png_path.with_suffix('.pdf')}")
        figures.append((title, png_path))

    report_path = build_report_pdf(
        output_dir,
        figures,
        report_name=report_output_name(args.report_version),
        cover_title=f"Post-Train DeGroot Fit Report ({args.report_version})",
        cover_body=build_cover_body(fit_results),
    )
    if report_path is not None:
        print(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()