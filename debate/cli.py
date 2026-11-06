from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys

from . import __version__
from .data import download, verify
from .human_views import export_human


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    modules = {"reproduce": "reproduction.run", "pipeline": "pipelines.run", "evaluate": "reproduction.evaluate"}
    if argv and argv[0] in modules:
        sys.argv = [f"debate {argv[0]}", *argv[1:]]
        runpy.run_module(modules[argv[0]], run_name="__main__")
        return
    parser = argparse.ArgumentParser(description="Download DEBATE data and reproduce the NeurIPS 2026 evaluation.")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("download", help="Download and verify cached data from Hugging Face")
    fetch.add_argument("--data-root", type=Path, default=Path("data"))
    fetch.add_argument("--source", type=Path, help="Use an existing local Hugging Face data checkout without network access")
    fetch.add_argument("--components", nargs="+", help="Component IDs, or all (default)")
    fetch.add_argument("--revision", help="Override the pinned dataset commit")
    fetch.add_argument("--repo-id", help="Override the dataset repository")
    fetch.add_argument("--list", action="store_true", help="Show component sizes without downloading")
    fetch.add_argument("--force", action="store_true", help="Replace locally modified release files")
    fetch.set_defaults(function=download)
    check = commands.add_parser("verify", help="Check every installed data file against its SHA-256 manifest")
    check.add_argument("--data-root", type=Path, default=Path("data"))
    check.set_defaults(function=verify)
    human = commands.add_parser("export-human", help="Export human-only event records, paper annotations and pre/post self-report pairs")
    human.add_argument("--data-root", type=Path, default=Path("data"))
    human.add_argument("--output-dir", type=Path, required=True)
    human.add_argument("--overwrite", action="store_true", help="Replace a previous human-view export; never replace input data")
    human.set_defaults(function=export_human)
    commands.add_parser("reproduce", help="Recompute paper figures and tables from cached labels (use --help)")
    commands.add_parser("pipeline", help="Simulate, annotate, prepare SFT data, train or serve (use --help)")
    commands.add_parser("evaluate", help="Evaluate newly annotated human/model records with the benchmark metrics")
    args = parser.parse_args(argv)
    try:
        args.function(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
