import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from debate.data import download, extract_verified, safe_path, sha256, verify


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.root = Path(self.scratch.name)

    def tearDown(self):
        self.scratch.cleanup()

    def archive(self, name="data/sample.csv", payload=b"a,b\n1,2\n", link=False):
        path = self.root / "bundle.tar.gz"
        with tarfile.open(path, "w:gz") as handle:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            if link:
                member.type = tarfile.SYMTYPE
                member.linkname = "../../outside"
                handle.addfile(member)
            else:
                handle.addfile(member, io.BytesIO(payload))
        spec = [{"path": name, "size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}]
        return path, spec

    def test_verified_extract_and_modified_file_protection(self):
        archive, spec = self.archive()
        output = self.root / "output"
        extract_verified(archive, output, spec)
        target = output / "data/sample.csv"
        self.assertEqual(target.read_bytes(), b"a,b\n1,2\n")
        target.write_text("local modification")
        with self.assertRaisesRegex(ValueError, "Existing file differs"):
            extract_verified(archive, output, spec)
        self.assertEqual(target.read_text(), "local modification")
        extract_verified(archive, output, spec, force=True)
        self.assertEqual(target.read_bytes(), b"a,b\n1,2\n")

    def test_bad_checksum_does_not_promote(self):
        archive, spec = self.archive()
        spec[0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            extract_verified(archive, self.root / "output", spec)
        self.assertFalse((self.root / "output/data/sample.csv").exists())

    def test_rejects_traversal_and_archive_symlinks(self):
        archive, spec = self.archive("../../outside")
        with self.assertRaises(ValueError):
            extract_verified(archive, self.root / "output", spec)
        archive, spec = self.archive("data/link", link=True)
        with self.assertRaisesRegex(ValueError, "Unexpected archive member"):
            extract_verified(archive, self.root / "output", spec)

    def test_rejects_existing_parent_symlink(self):
        output = self.root / "output"
        output.mkdir()
        (output / "data").symlink_to(self.root)
        with self.assertRaises(ValueError):
            safe_path(output, "data/sample.csv")

    def test_rejects_alias_paths_and_reserved_metadata(self):
        for name in ("data//sample.csv", "data/./sample.csv", "./data/sample.csv"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_path(self.root, name)
        for name in (".debate/release.json", ".DEBATE/release.json"):
            archive, spec = self.archive(name)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "installation metadata"):
                extract_verified(archive, self.root / "output", spec)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.root = Path(self.scratch.name)
        self.source = self.root / "dataset"
        self.source.mkdir()
        archive, files = ArchiveTests.archive(self)
        manifest_name = "neurips2026/custom/input_files.json"
        manifest = self.source / manifest_name
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps(files))
        destination = self.source / "neurips2026/archives/sample.tar.gz"
        destination.parent.mkdir()
        archive.rename(destination)
        self.document = {"schema_version": 1, "release": "test-release", "archives": [{
            "id": "sample", "filename": "neurips2026/archives/sample.tar.gz",
            "sha256": sha256(destination), "size_bytes": destination.stat().st_size,
            "unpacked_bytes": sum(item["size_bytes"] for item in files), "file_count": len(files),
            "files_manifest": manifest_name, "manifest_sha256": sha256(manifest),
        }]}
        self.manifest = self.source / "neurips2026/manifest.json"
        self.manifest.write_text(json.dumps(self.document))
        self.locked = {"dataset_repo": "example/data", "revision": "a" * 40,
                       "manifest_filename": "neurips2026/manifest.json"}
        self.args = SimpleNamespace(repo_id=None, revision=None, source=self.source,
                                    components=None, list=False, data_root=self.root / "output", force=False)

    def tearDown(self):
        self.scratch.cleanup()

    def test_local_download_is_offline_and_honors_files_manifest(self):
        with patch("debate.data.load_release", return_value=self.locked), \
                patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("network")), \
                patch("huggingface_hub.HfApi", side_effect=AssertionError("network")):
            download(self.args)
            verify(self.args)
        self.assertEqual((self.args.data_root / "data/sample.csv").read_bytes(), b"a,b\n1,2\n")
        (self.args.data_root / "data/sample.csv").write_text("changed")
        with self.assertRaisesRegex(ValueError, "modified files"):
            verify(self.args)

    def test_pinned_revision_never_resolves_a_mutable_branch(self):
        self.args.source = None
        fetched = []
        def fetch(**kwargs):
            fetched.append(kwargs)
            return str(self.source / kwargs["filename"])
        with patch("debate.data.load_release", return_value=self.locked), \
                patch("huggingface_hub.hf_hub_download", side_effect=fetch), \
                patch("huggingface_hub.HfApi", side_effect=AssertionError("unexpected revision lookup")):
            download(self.args)
        self.assertEqual({item["revision"] for item in fetched}, {"a" * 40})
        self.assertEqual({item["repo_type"] for item in fetched}, {"dataset"})

    def test_explicit_branch_override_resolves_once(self):
        self.args.source = None
        self.args.revision = "main"
        fetched = []
        def fetch(**kwargs):
            fetched.append(kwargs)
            return str(self.source / kwargs["filename"])
        with patch("debate.data.load_release", return_value=self.locked), \
                patch("huggingface_hub.hf_hub_download", side_effect=fetch), \
                patch("huggingface_hub.HfApi") as api:
            api.return_value.repo_info.return_value = SimpleNamespace(sha="b" * 40)
            download(self.args)
            api.return_value.repo_info.assert_called_once_with(repo_id="example/data", repo_type="dataset", revision="main")
        self.assertEqual({item["revision"] for item in fetched}, {"b" * 40})
        state = json.loads((self.args.data_root / ".debate/release.json").read_text())
        self.assertEqual(state["revision"], "b" * 40)
        self.assertEqual(state["requested_revision"], "main")

    def test_rejects_unpublished_remote_revision_and_mutable_default(self):
        self.args.source = None
        for revision in ("PENDING_RELEASE_COMMIT", "main"):
            with self.subTest(revision=revision), \
                    patch("debate.data.load_release", return_value={**self.locked, "revision": revision}), \
                    patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("network")), \
                    self.assertRaises(ValueError):
                download(self.args)

    def test_rejects_unsafe_and_duplicate_component_ids(self):
        for identifiers in (("../../outside",), ("sample", "sample")):
            document = {**self.document, "archives": [{**self.document["archives"][0], "id": name} for name in identifiers]}
            self.manifest.write_text(json.dumps(document))
            with self.subTest(identifiers=identifiers), patch("debate.data.load_release", return_value=self.locked), self.assertRaises(ValueError):
                download(self.args)
        self.assertFalse((self.root / "outside.json").exists())

    def test_rejects_metadata_symlink_before_writing(self):
        self.args.data_root.mkdir()
        outside = self.root / "outside"
        outside.mkdir()
        (self.args.data_root / ".debate").symlink_to(outside)
        with patch("debate.data.load_release", return_value=self.locked), self.assertRaises(ValueError):
            download(self.args)
        self.assertEqual(list(outside.iterdir()), [])

    def test_bad_archive_digest_does_not_extract(self):
        self.document["archives"][0]["sha256"] = "0" * 64
        self.manifest.write_text(json.dumps(self.document))
        with patch("debate.data.load_release", return_value=self.locked), self.assertRaisesRegex(ValueError, "Archive checksum mismatch"):
            download(self.args)
        self.assertFalse((self.args.data_root / "data/sample.csv").exists())


if __name__ == "__main__":
    unittest.main()
