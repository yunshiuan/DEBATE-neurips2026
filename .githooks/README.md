# Local commit review and manual pushes

Agents must never push. They may make local changes and, when authorized, local commits for the user to review. Earlier push authorization is revoked. This applies to Git pushes, Git LFS uploads, direct remote repository writes, and equivalent publishing or deployment actions.

## For the user

Review the local commits, then use **Push in VS Code Source Control** or run `git push` in your own terminal. No environment override is needed. The old `DEBATE_REVIEWED_PUSH` flag is no longer used.

## Agent protection

The persistent local `pre-push` wrapper calls the tracked `pre-push-check` guard. It rejects commands with a nonempty `CODEX_THREAD_ID`, `CODEX_SESSION_ID`, `CODEX_TOOL_CALL_ID`, `CODEX_CI`, or `CODEX_SANDBOX` execution marker. Ordinary VS Code Git operations without these markers are allowed through to any existing hook.

Agents must never unset or spoof these markers, use `--no-verify`, change `core.hooksPath` to bypass the guard, remove/disable the guard, or invoke an equivalent remote publishing route. The prohibition applies regardless of whether a marker is present.

The hook distinguishes normal execution environments; it is an accidental-push safeguard, not an authentication boundary. A terminal launched by an agent can inherit its markers, so use your independently opened VS Code window or terminal for manual pushes.

Codex also supports forbidden-command rules in `~/.codex/rules/`. On the author's workstation, `debate-no-agent-push.rules` denies direct push/upload commands in Codex's command policy. These additional rules load on Codex startup and do not affect human VS Code commands. Prefix rules do not cover every possible scripted invocation; the hook and the independent agent prohibition still apply. See [Codex rules](https://learn.chatgpt.com/docs/agent-configuration/rules).

## Installation

After a fresh clone, install the guard locally:

```sh
python3 .githooks/install.py
```

Installation preserves the configured hooks directory and all other hooks. In a worktree hooks directory, the installed wrapper is excluded locally from Git so it survives branch changes; a missing tracked guard blocks the push. Any existing pre-push hook is preserved as `pre-push.before-debate-review-guard` and runs with its arguments and input intact after the guard permits a human push. This keeps Git LFS uploads working for the user.

Git does not activate hooks automatically on a new clone. The tracked guard and installer travel with the repository; the workstation's Codex rule file does not.
