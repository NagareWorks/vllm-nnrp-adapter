from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import urllib.error
from pathlib import Path
from types import ModuleType

import pytest


def _load_script(name: str) -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release_identity = _load_script("check_release_identity")
pypi_identity = _load_script("check_pypi_identity")


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        capture_output=True,
        check=True,
        text=True,
    )
    return result.stdout.strip()


def _repository(path: Path) -> Path:
    _git(path, "init")
    _git(path, "config", "user.email", "release-test@example.com")
    _git(path, "config", "user.name", "Release Test")
    (path / "version.txt").write_text("one\n", encoding="utf-8")
    _git(path, "add", "version.txt")
    _git(path, "commit", "-m", "initial")
    _git(path, "update-ref", "refs/remotes/origin/main", "HEAD")
    return path


def test_git_identity_accepts_main_and_same_commit_tag(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    _git(repository, "tag", "v0.1.0")

    source = release_identity.validate_git_identity(repository, "origin/main", "v0.1.0")

    assert source == _git(repository, "rev-parse", "HEAD")


def test_git_identity_rejects_source_or_tag_drift(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    _git(repository, "tag", "v0.1.0")
    (repository / "version.txt").write_text("two\n", encoding="utf-8")
    _git(repository, "commit", "-am", "second")

    with pytest.raises(release_identity.ReleaseIdentityError, match="does not match"):
        release_identity.validate_git_identity(repository, "origin/main", "unused")

    _git(repository, "update-ref", "refs/remotes/origin/main", "HEAD")
    with pytest.raises(release_identity.ReleaseIdentityError, match="already points"):
        release_identity.validate_git_identity(repository, "origin/main", "v0.1.0")


def test_distribution_identity_requires_absent_release_and_accepts_exact_rerun(tmp_path: Path) -> None:
    (tmp_path / "adapter.whl").write_bytes(b"wheel")
    (tmp_path / "adapter.tar.gz").write_bytes(b"sdist")
    local = pypi_identity.distribution_digests(tmp_path)

    assert pypi_identity.publication_required(local, None)
    assert not pypi_identity.publication_required(local, local)


def test_distribution_identity_rejects_missing_or_different_public_bytes(tmp_path: Path) -> None:
    (tmp_path / "adapter.whl").write_bytes(b"wheel")
    local = pypi_identity.distribution_digests(tmp_path)

    with pytest.raises(pypi_identity.PyPiIdentityError, match="not available"):
        pypi_identity.publication_required(local, None, require_published=True)
    with pytest.raises(pypi_identity.PyPiIdentityError, match="different distribution set"):
        pypi_identity.publication_required(local, {"adapter.whl": "0" * 64})


def test_pypi_lookup_reads_digests_and_handles_missing_or_failed_requests() -> None:
    def available(request: object, *, timeout: int) -> io.BytesIO:
        assert "vllm-nnrp-adapter/0.1.0" in request.full_url  # type: ignore[attr-defined]
        assert timeout == 20
        document = {"urls": [{"filename": "adapter.whl", "digests": {"sha256": "a" * 64}}]}
        return io.BytesIO(json.dumps(document).encode())

    assert pypi_identity.pypi_distribution_digests(
        "vllm-nnrp-adapter", "0.1.0", opener=available
    ) == {"adapter.whl": "a" * 64}

    def missing(*_args: object, **_kwargs: object) -> None:
        raise urllib.error.HTTPError("https://pypi.org", 404, "missing", {}, None)

    assert pypi_identity.pypi_distribution_digests("vllm-nnrp-adapter", "missing", opener=missing) is None

    def failed(*_args: object, **_kwargs: object) -> None:
        raise urllib.error.HTTPError("https://pypi.org", 503, "failed", {}, None)

    with pytest.raises(pypi_identity.PyPiIdentityError, match="HTTP 503"):
        pypi_identity.pypi_distribution_digests("vllm-nnrp-adapter", "failed", opener=failed)
