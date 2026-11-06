# Local commit review and manual pushes

Agents must never push. They may make local changes and, when authorized, local commits for the user to review. Earlier push authorization is revoked. This applies to Git pushes, Git LFS uploads, direct remote repository writes, and equivalent publishing or deployment actions.

The persistent local `pre-push` wrapper calls the tracked `pre-push-check` guard, which blocks pushes by default. The user alone may permit a single manual push after reviewing the commits:

```sh
git --no-pager log --oneline
git --no-pager show HEAD
DEBATE_REVIEWED_PUSH=1 git push
```

Run the final command yourself in your terminal. Do not export the override globally or put it in shell startup files. Agents must never set or use it, use `--no-verify`, change `core.hooksPath` to bypass the guard, remove/disable the guard, or invoke an equivalent remote publishing route.

After a fresh clone, install the guard locally:

```sh
python3 .githooks/install.py
```

Installation preserves the configured hooks directory and all other hooks. In a worktree hooks directory, the installed wrapper is excluded locally from Git so it survives branch changes; a missing tracked guard blocks the push. Any existing pre-push hook is preserved as `pre-push.before-debate-review-guard` and runs, with its arguments and input intact, only after the review guard allows the user's manual push. This keeps Git LFS uploads working for the user.

The tracked guard and installer travel with the repository; Git does not activate hooks automatically on a new clone. This is a local accidental-push safeguard, not server-side access control. The agent prohibition applies even if the hook is absent or bypassable.
