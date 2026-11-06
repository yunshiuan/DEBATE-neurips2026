# Working on the public DEBATE release

## Local commits only; the user pushes

- Never run `git push` (including dry runs), `git send-pack`, `git lfs push`, or equivalent remote repository writes, publishing, or deployment actions. The user reviews local commits and pushes personally. Earlier push authorization is revoked.
- Never set or use `DEBATE_REVIEWED_PUSH`, bypass hooks with `--no-verify` or `core.hooksPath`, remove/disable the guard, or hide a push inside scripts, tools, or CI. Read-only remote checks remain allowed.
- Local edits, checks, and explicitly authorized local commits remain allowed. Stop at the local commit and report it for review.
- The local push guard is installed with `python3 .githooks/install.py`. See [.githooks/README.md](.githooks/README.md) for the user's manual-review workflow.

- Preserve the released split masks, stage-specific validity rules, and separate model-family references. Do not silently substitute another experiment or training corpus.
- `debate/release.json` pins the public data commit. Update it only after the corresponding dataset revision is published and verified.
- Cached reproduction must run without API credentials, model weights, private repositories, or author-machine paths.
- Keep data, generated figures, model checkpoints and credentials out of Git. The dataset has its own research-only license; code uses MIT.
- Run `python -m pytest -q` for pipeline changes. For numerical changes, run the appropriate `debate reproduce` stages and inspect their verification reports.
- Record what was actually tested. CPU dry runs do not establish that GPU training has been executed, and rerendering an aggregate does not establish regeneration of its original annotations.
- Do not modify paper results or fill missing experimental records to force a verification check to pass.
