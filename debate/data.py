"""Download and verify the versioned DEBATE data release.

The data are separately licensed. This module never executes downloaded code
or deserializes pickle files; archives contain ordinary data files only.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(name: str) -> PurePosixPath:
    if not isinstance(name, str):
        raise ValueError("Archive and manifest paths must be strings")
    relative = PurePosixPath(name)
    if (relative.is_absolute() or not relative.parts or ".." in relative.parts
            or "\\" in name or "\x00" in name or relative.as_posix() != name):
        raise ValueError(f"Unsafe archive or manifest path: {name!r}")
    return relative


def safe_path(root: Path, name: str) -> Path:
    relative = relative_path(name)
    target = root.joinpath(*relative.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes the data directory: {name!r}")
    for parent in [target, *target.parents]:
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError(f"Refusing a symlink in the data path: {name!r}")
    return target


def load_release() -> dict:
    return json.loads(Path(__file__).with_name("release.json").read_text())


def get_file(filename: str, repo: str, revision: str, source: Path | None) -> Path:
    relative_path(filename)
    if source is not None:
        path = safe_path(source, filename)
        if not path.is_file():
            raise FileNotFoundError(f"Missing release file: {path}")
        return path
    if revision.startswith("PENDING"):
        raise ValueError("This development checkout has no published data revision. Use --source with a local dataset checkout.")
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=repo, repo_type="dataset", filename=filename, revision=revision))


def resolve_remote_revision(repo: str, revision: str) -> str:
    """Resolve an explicit branch/tag once, so every file uses one snapshot."""
    if revision.startswith("PENDING"):
        raise ValueError("This development checkout has no published data revision. Use --source with a local dataset checkout.")
    if re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    from huggingface_hub import HfApi
    commit = HfApi().repo_info(repo_id=repo, repo_type="dataset", revision=revision).sha
    if not commit or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("The requested dataset revision did not resolve to an immutable commit")
    return commit


def manifest_files(document: dict | list) -> list[dict]:
    return document if isinstance(document, list) else document["files"]


def extract_verified(archive: Path, output: Path, file_manifest: list[dict], force: bool = False) -> None:
    """Check every archive member before promoting files into the data directory."""
    output.mkdir(parents=True, exist_ok=True)
    expected = {}
    for item in file_manifest:
        name = item["path"]
        if relative_path(name).parts[0].casefold() == ".debate":
            raise ValueError("Archive data cannot overwrite .debate installation metadata")
        safe_path(output, name)
        if name in expected:
            raise ValueError(f"Duplicate manifest entry: {name}")
        expected[name] = item
        current = safe_path(output, name)
        if current.exists() and not current.is_file():
            raise ValueError(f"Expected a regular file: {current}")
        if current.is_file() and not force and sha256(current) != item["sha256"]:
            raise ValueError(f"Existing file differs from the release: {current}. Use a fresh directory or --force.")
    with tempfile.TemporaryDirectory(prefix=".debate-extract-", dir=output) as scratch:
        staging = Path(scratch)
        seen = set()
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle:
                target = safe_path(staging, member.name.rstrip("/") if member.isdir() else member.name)
                if member.isdir():
                    continue
                if not member.isfile() or member.name not in expected or member.name in seen:
                    raise ValueError(f"Unexpected archive member: {member.name!r}")
                spec = expected[member.name]
                if member.size != spec["size_bytes"]:
                    raise ValueError(f"Size mismatch in archive member: {member.name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                stream = bundle.extractfile(member)
                if stream is None:
                    raise ValueError(f"Unreadable archive member: {member.name}")
                with stream, target.open("wb") as handle:
                    shutil.copyfileobj(stream, handle, length=8 * 1024 * 1024)
                if sha256(target) != spec["sha256"]:
                    raise ValueError(f"Checksum mismatch in archive member: {member.name}")
                seen.add(member.name)
        if seen != set(expected):
            raise ValueError(f"Archive lacks {len(set(expected) - seen)} manifest files")
        for name in expected:
            destination = safe_path(output, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(safe_path(staging, name), destination)


def download(args) -> None:
    locked = load_release()
    repo = args.repo_id or locked["dataset_repo"]
    revision = args.revision or locked["revision"]
    source = args.source.resolve() if args.source else None
    requested_revision = revision
    if source is None:
        if args.revision is None and not re.fullmatch(r"[0-9a-f]{40}", revision) and not revision.startswith("PENDING"):
            raise ValueError("The bundled release must pin a full dataset commit; use --revision for an explicit override")
        revision = resolve_remote_revision(repo, revision)
    document = json.loads(get_file(locked["manifest_filename"], repo, revision, source).read_text())
    if document.get("schema_version") != 1:
        raise ValueError("Unsupported data manifest version")
    archives = {}
    for item in document["archives"]:
        name = item["id"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
            raise ValueError(f"Unsafe component ID: {name!r}")
        if name in archives:
            raise ValueError(f"Duplicate component ID: {name}")
        archives[name] = item
    selected = list(archives) if not args.components or args.components == ["all"] else args.components
    unknown = set(selected) - archives.keys()
    if unknown:
        raise ValueError(f"Unknown components: {sorted(unknown)}; available: {list(archives)}")
    if args.list:
        for name in selected:
            item = archives[name]
            print(f"{name}: {item['size_bytes'] / 1e6:.1f} MB download; {item['unpacked_bytes'] / 1e9:.2f} GB extracted; {item['file_count']:,} files")
        return
    root = args.data_root.resolve()
    state = safe_path(root, ".debate")
    state.mkdir(parents=True, exist_ok=True)
    manifest_directory = safe_path(root, ".debate/manifests")
    manifest_directory.mkdir(exist_ok=True)
    for name in selected:
        spec = archives[name]
        manifest_name = spec.get("files_manifest") or spec.get("manifest_filename", f"neurips2026/manifests/{name}.json")
        manifest_path = get_file(manifest_name, repo, revision, source)
        if spec.get("manifest_sha256") and sha256(manifest_path) != spec["manifest_sha256"]:
            raise ValueError(f"File-manifest checksum mismatch for {name}")
        files = manifest_files(json.loads(manifest_path.read_text()))
        if len(files) != spec["file_count"]:
            raise ValueError(f"Manifest file count mismatch for {name}")
        print(f"Downloading {name} ({spec['size_bytes'] / 1e6:.1f} MB)...", flush=True)
        archive = get_file(spec["filename"], repo, revision, source)
        if archive.stat().st_size != spec["size_bytes"] or sha256(archive) != spec["sha256"]:
            raise ValueError(f"Archive checksum mismatch for {name}")
        extract_verified(archive, root, files, force=args.force)
        safe_path(root, f".debate/manifests/{name}.json").write_text(json.dumps(files, indent=2) + "\n")
        print(f"Verified {name}: {len(files):,} files", flush=True)
    safe_path(root, ".debate/release.json").write_text(json.dumps({"dataset_repo": repo, "revision": revision, "requested_revision": requested_revision,
        "release": document["release"], "source": "local" if source else "huggingface"}, indent=2) + "\n")
    print(f"Data ready at {root}")


def verify(args) -> None:
    root = args.data_root.resolve()
    manifests = sorted(safe_path(root, ".debate/manifests").glob("*.json"))
    if not manifests:
        raise ValueError("No installed component manifests. Run 'debate download' first.")
    failures = []
    count = 0
    for manifest in manifests:
        safe_path(root, manifest.relative_to(root).as_posix())
        for item in manifest_files(json.loads(manifest.read_text())):
            path = safe_path(root, item["path"])
            if not path.is_file() or path.stat().st_size != item["size_bytes"] or sha256(path) != item["sha256"]:
                failures.append(item["path"])
            count += 1
    if failures:
        raise ValueError(f"{len(failures)} missing or modified files; first examples: {failures[:10]}")
    print(f"Verified {count:,} files across {len(manifests)} components.")
