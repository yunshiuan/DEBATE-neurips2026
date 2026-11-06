from __future__ import annotations

import os
from pathlib import Path

from .complete_group_masks import REPO_ROOT, V3_OPINION_GROUP_MASK_ENV_VAR, V3_TWEET_GROUP_MASK_ENV_VAR


DEFAULT_TWEET_MASK_ROOT = REPO_ROOT / "neurips_2026" / "sean_reproduce" / "v3" / "masks" / "groups_with_valid_tweets"
DEFAULT_OPINION_MASK_ROOT = REPO_ROOT / "neurips_2026" / "sean_reproduce" / "v3" / "masks" / "groups_with_valid_opinions"


def configured_public_mask_root() -> Path:
    configured_root = os.environ.get(V3_TWEET_GROUP_MASK_ENV_VAR)
    return Path(configured_root).expanduser().resolve() if configured_root else DEFAULT_TWEET_MASK_ROOT.resolve()


def configured_private_mask_root() -> Path:
    configured_root = os.environ.get(V3_OPINION_GROUP_MASK_ENV_VAR)
    return Path(configured_root).expanduser().resolve() if configured_root else DEFAULT_OPINION_MASK_ROOT.resolve()


def repo_relative_display_path(path: Path) -> str:
    resolved_path = Path(path).expanduser().resolve()
    try:
        return str(resolved_path.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved_path)


def wrapped_path_lines(
    path: Path,
    *,
    width: int = 92,
    indent: str = "",
    subsequent_indent: str = "    ",
) -> list[str]:
    display_path = repo_relative_display_path(path)
    parts = display_path.split("/")
    if not parts:
        return [indent.rstrip()]

    current = f"{indent}{parts[0]}"
    wrapped_lines: list[str] = []
    for part in parts[1:]:
        candidate = f"{current}/{part}"
        if len(candidate) <= width:
            current = candidate
            continue
        wrapped_lines.append(f"{current}/")
        current = f"{subsequent_indent}{part}"
    wrapped_lines.append(current)
    return wrapped_lines