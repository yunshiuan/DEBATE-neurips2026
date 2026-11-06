# Working on the public DEBATE release

- Preserve the released split masks, stage-specific validity rules, and separate model-family references. Do not silently substitute another experiment or training corpus.
- `debate/release.json` pins the public data commit. Update it only after the corresponding dataset revision is published and verified.
- Cached reproduction must run without API credentials, model weights, private repositories, or author-machine paths.
- Keep data, generated figures, model checkpoints and credentials out of Git. The dataset has its own research-only license; code uses MIT.
- Run `python -m pytest -q` for pipeline changes. For numerical changes, run the appropriate `debate reproduce` stages and inspect their verification reports.
- Record what was actually tested. CPU dry runs do not establish that GPU training has been executed, and rerendering an aggregate does not establish regeneration of its original annotations.
- Do not modify paper results or fill missing experimental records to force a verification check to pass.
