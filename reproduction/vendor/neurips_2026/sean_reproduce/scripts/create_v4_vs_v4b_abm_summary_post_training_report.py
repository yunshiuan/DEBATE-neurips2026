#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
RADAR_DIR = REPO_ROOT / "neurips_2026" / "chengtao" / "src" / "radar"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(RADAR_DIR) not in sys.path:
    sys.path.insert(0, str(RADAR_DIR))

from generate_plots import get_v2_display_model_order  # noqa: E402
from post_training_breadth_profiles import maybe_adjust_bar_model_order  # noqa: E402

from neurips_2026.sean_reproduce.scripts.create_v4_vs_v4b_summary_base_model_report import (  # noqa: E402
    build_page_figure,
    read_csv_rows,
    write_comparison_csv,
)

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"matplotlib is required to render the v4 vs v4_b ABM summary post-training report: {exc}")


COMPARE_ROOT = REPO_ROOT / "neurips_2026" / "sean_reproduce" / "v4_vs_v4b" / "abm" / "summary_post_training"
SHARED_ABM_BASELINE_MODELS = {"DeGroot fit", "DeGroot fit topic", "DeGroot fit group"}

VERSION_SPECS = {
    "v4": {
        "summary_dir": REPO_ROOT / "neurips_2026" / "sean_reproduce" / "v4" / "abm" / "summary_post_training",
        "csv_dir": REPO_ROOT / "neurips_2026" / "sean_reproduce" / "v4" / "abm" / "summary_post_training" / "csv",
    },
    "v4_b": {
        "summary_dir": REPO_ROOT / "neurips_2026" / "sean_reproduce" / "v4_b" / "abm" / "summary_post_training",
        "csv_dir": REPO_ROOT / "neurips_2026" / "sean_reproduce" / "v4_b" / "abm" / "summary_post_training" / "csv",
    },
}

PARTITION_SPECS = [
    ("breadth_topic_train", "Breadth Topic Partition - Train"),
    ("breadth_topic_test", "Breadth Topic Partition - Test"),
    ("breadth_group_train", "Breadth Group Partition - Train"),
    ("breadth_group_test", "Breadth Group Partition - Test"),
    ("depth_topic_train", "Depth Topic Partition - Train"),
    ("depth_topic_test", "Depth Topic Partition - Test"),
    ("depth_group_train", "Depth Group Partition - Train"),
    ("depth_group_test", "Depth Group Partition - Test"),
]

FAMILY_TEMPLATES = [
    {
        "report_stem": "report_bar_posttrain_models_degroot",
        "csv_prefix": "",
        "model_set": "post-training-v4",
        "page_title_prefix": "Post-Training DeGroot ABM",
        "cover_subtitle": "ABM Summary Post-Training Comparison Bar Report",
        "source_description": "ABM post-training DeGroot summary reports",
    },
    {
        "report_stem": "report_bar_posttrain_models_vs_all_degroot",
        "csv_prefix": "models_vs_all_",
        "model_set": "post-training-v4-vs-all",
        "page_title_prefix": "Post-Train Models vs All DeGroot ABM",
        "cover_subtitle": "ABM Summary Post-Training vs-All Comparison Bar Report",
        "source_description": "ABM post-training-vs-all DeGroot summary reports",
    },
]

VARIANT_SPECS = [
    {
        "variant_suffix": "",
        "cover_variant_label": "Filtered standard panels",
    },
    {
        "variant_suffix": "_unfiltered_msg_and_ind",
        "cover_variant_label": "Unfiltered message/individual panels",
    },
]


def _family_specs() -> list[dict[str, str | list[dict[str, str]]]]:
    family_specs: list[dict[str, str | list[dict[str, str]]]] = []
    for family_template in FAMILY_TEMPLATES:
        for variant_spec in VARIANT_SPECS:
            variant_suffix = str(variant_spec["variant_suffix"])
            panels: list[dict[str, str]] = []
            for partition_name, partition_title in PARTITION_SPECS:
                source_stem = f"{family_template['csv_prefix']}{partition_name}"
                panels.append(
                    {
                        "partition_name": partition_name,
                        "csv_name_v4": f"raw_metrics_{source_stem}_v4{variant_suffix}.csv",
                        "csv_name_v4_b": f"raw_metrics_{source_stem}_v4_b{variant_suffix}.csv",
                        "comparison_csv": f"raw_metrics_{source_stem}_v4_vs_v4b{variant_suffix}.csv",
                        "png_name": f"bar_{source_stem}_v4_vs_v4b{variant_suffix}.png",
                        "page_title": f"{family_template['page_title_prefix']} - {partition_title}",
                    }
                )

            family_specs.append(
                {
                    "report_pdf_name": f"{family_template['report_stem']}_v4_vs_4b{variant_suffix}.pdf",
                    "report_stem": str(family_template["report_stem"]),
                    "model_set": str(family_template["model_set"]),
                    "variant_suffix": variant_suffix,
                    "cover_subtitle": str(family_template["cover_subtitle"]),
                    "cover_variant_label": str(variant_spec["cover_variant_label"]),
                    "source_description": str(family_template["source_description"]),
                    "panels": panels,
                }
            )
    return family_specs


def _source_report_paths(family_spec: dict[str, str | list[dict[str, str]]]) -> dict[str, Path]:
    report_stem = str(family_spec["report_stem"])
    variant_suffix = str(family_spec["variant_suffix"])
    return {
        version_key: Path(VERSION_SPECS[version_key]["summary_dir"]) / f"{report_stem}_{version_key}{variant_suffix}.pdf"
        for version_key in ("v4", "v4_b")
    }


def build_cover_figure(family_spec: dict[str, str | list[dict[str, str]]]) -> object:
    source_paths = _source_report_paths(family_spec)
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.5, 0.78, "v4 vs v4_b", ha="center", va="center", fontsize=24, fontweight="bold")
    ax.text(0.5, 0.69, str(family_spec["cover_subtitle"]), ha="center", va="center", fontsize=16)
    ax.text(0.5, 0.61, str(family_spec["cover_variant_label"]), ha="center", va="center", fontsize=12)
    ax.text(
        0.5,
        0.53,
        "Pages compare v4 and v4_b side by side for the same ABM post-training panel within each breadth/depth topic/group train/test split.",
        ha="center",
        va="center",
        fontsize=11,
    )
    ax.text(0.5, 0.47, "Solid bars = v4, hatched bars = v4_b.", ha="center", va="center", fontsize=11)
    ax.text(
        0.08,
        0.28,
        f"Source {family_spec['source_description']}:\n"
        f"- {source_paths['v4'].relative_to(REPO_ROOT)}\n"
        f"- {source_paths['v4_b'].relative_to(REPO_ROOT)}",
        ha="left",
        va="top",
        fontsize=10,
        family="monospace",
    )
    return fig


def render_report(family_spec: dict[str, str | list[dict[str, str]]], *, output_root: Path, hide_count_annotations: bool = False) -> Path:
    output_pdf = output_root / str(family_spec["report_pdf_name"])
    img_dir = output_root / "img"
    csv_dir = output_root / "csv"
    output_root.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_pdf) as pdf:
        cover = build_cover_figure(family_spec)
        pdf.savefig(cover, bbox_inches="tight")
        plt.close(cover)

        for panel_spec in family_spec["panels"]:
            rows_by_version = {
                "v4": read_csv_rows(Path(VERSION_SPECS["v4"]["csv_dir"]) / str(panel_spec["csv_name_v4"])),
                "v4_b": read_csv_rows(Path(VERSION_SPECS["v4_b"]["csv_dir"]) / str(panel_spec["csv_name_v4_b"])),
            }
            write_comparison_csv(csv_dir / str(panel_spec["comparison_csv"]), rows_by_version)
            combined_rows = rows_by_version["v4"] + rows_by_version["v4_b"]
            display_model_order = get_v2_display_model_order(combined_rows, include_human=False)
            model_order = maybe_adjust_bar_model_order(
                display_model_order,
                partition_name=str(panel_spec["partition_name"]),
                model_set=str(family_spec["model_set"]),
            )
            fig = build_page_figure(
                page_title=str(panel_spec["page_title"]),
                rows_by_version=rows_by_version,
                hide_count_annotations=hide_count_annotations,
                model_order=list(model_order),
                shared_models=SHARED_ABM_BASELINE_MODELS,
            )
            pdf.savefig(fig, bbox_inches="tight")
            fig.savefig(img_dir / str(panel_spec["png_name"]), dpi=220)
            plt.close(fig)

    return output_pdf


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build v4 vs v4_b ABM summary post-training comparison bar reports.")
    parser.add_argument("--output-dir", type=Path, default=COMPARE_ROOT)
    parser.add_argument(
        "--hide-count-annotations",
        action="store_true",
        help="Hide n=XXX count annotations in the v4-vs-v4_b ABM summary post-training comparison pages.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    output_dir = args.output_dir.resolve()
    for family_spec in _family_specs():
        print(render_report(family_spec, output_root=output_dir, hide_count_annotations=bool(args.hide_count_annotations)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())