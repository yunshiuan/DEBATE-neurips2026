#!/usr/bin/env python3
"""Install a persistent local push wrapper without changing other Git hooks."""
from pathlib import Path
import os
import subprocess

root = Path(__file__).resolve().parent.parent
source = root / '.githooks/pre-push-check'
marker = '# DEBATE local review-only push wrapper v1'

def git(*args, missing=False):
    result = subprocess.run(['git', '--no-pager', *args], cwd=root, text=True, capture_output=True)
    if missing and result.returncode == 1:
        return ''
    result.check_returncode()
    return result.stdout.strip()

if Path(git('rev-parse', '--show-toplevel')).resolve() != root:
    raise SystemExit('Run this installer from its own repository checkout.')
configured = git('config', '--path', '--get', 'core.hooksPath', missing=True)
hooks = Path(configured or git('rev-parse', '--git-path', 'hooks'))
if not hooks.is_absolute():
    hooks = root / hooks
hooks.mkdir(parents=True, exist_ok=True)
target = hooks / 'pre-push'
source.chmod(source.stat().st_mode | 0o111)
backup = hooks / 'pre-push.before-debate-review-guard'
had_hook = target.exists() or target.is_symlink()
managed = target.exists() and marker in target.read_text(errors='replace')
if not had_hook and (backup.exists() or backup.is_symlink()):
    raise SystemExit('A saved pre-push hook exists without its wrapper; inspect it before reinstalling.')

# A wrapper in a worktree hooks directory must survive branch changes.
if configured:
    try:
        relative = target.relative_to(root)
    except ValueError:
        relative = None
    if relative is not None:
        if git('ls-files', '--', str(relative)):
            raise SystemExit('The installed wrapper path is tracked; migrate it before installing.')
        exclude = Path(git('rev-parse', '--git-path', 'info/exclude'))
        if not exclude.is_absolute():
            exclude = root / exclude
        exclude.parent.mkdir(parents=True, exist_ok=True)
        current = exclude.read_text() if exclude.exists() else ''
        pattern = '/' + relative.as_posix()
        if pattern not in current.splitlines():
            exclude.write_text(current.rstrip('\n') + '\n' + pattern + '\n')

if managed:
    chain_previous = 'exec "$previous" "$@"' in target.read_text()
    if chain_previous and not os.access(backup, os.X_OK):
        raise SystemExit('The preserved pre-push hook is missing or not executable; inspect it manually.')
else:
    chain_previous = had_hook and os.access(target, os.X_OK)
    if had_hook:
        if backup.exists() or backup.is_symlink():
            raise SystemExit('Existing backup would be overwritten; inspect both hooks manually.')
        target.rename(backup)

wrapper = r'''#!/bin/sh
# DEBATE local review-only push wrapper v1
repo_root=$(git rev-parse --show-toplevel) || exit 1
guard="$repo_root/.githooks/pre-push-check"
if [ ! -x "$guard" ]; then
    printf '%s\n' 'Push blocked: the local review guard is missing or not executable.' >&2
    exit 1
fi
"$guard" "$@"
guard_status=$?
if [ "$guard_status" -ne 0 ]; then
    exit "$guard_status"
fi
'''
if chain_previous:
    wrapper += r'''hook_dir=$(CDPATH= cd "$(dirname "$0")" && pwd) || exit 1
previous="$hook_dir/pre-push.before-debate-review-guard"
if [ ! -x "$previous" ]; then
    printf '%s\n' 'Push blocked: the preserved pre-push hook is missing or not executable.' >&2
    exit 1
fi
exec "$previous" "$@"
'''
else:
    wrapper += 'exit 0\n'
if not target.exists() or target.read_text() != wrapper:
    target.write_text(wrapper)
target.chmod(0o755)
print('Push guard active: ' + str(target))
if had_hook and not managed:
    print('Preserved previous hook: ' + str(backup))
