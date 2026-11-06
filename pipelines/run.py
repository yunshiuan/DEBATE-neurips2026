"""Command dispatcher used by `debate pipeline` and `python -m pipelines.run`."""
from __future__ import annotations

import importlib
import sys

COMMANDS = {"simulate": "simulate", "annotate": "annotate", "prepare-sft": "prepare_sft", "train": "train_sft", "serve": "serve"}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("Usage: debate pipeline {simulate,annotate,prepare-sft,train,serve} [arguments]\nUse COMMAND --help for options. Each command supports --dry-run.")
        return 0
    command = argv.pop(0)
    if command not in COMMANDS:
        raise SystemExit(f"Unknown pipeline {command!r}; choose {', '.join(COMMANDS)}")
    module = importlib.import_module("pipelines." + COMMANDS[command])
    return module.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
