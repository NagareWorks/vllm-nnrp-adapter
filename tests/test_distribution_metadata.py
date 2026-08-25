from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from vllm_nnrp_adapter.distribution_metadata import (
    DistributionMetadataError,
    documented_public_symbols,
    validate_distribution,
    validate_metadata_text,
)


def _metadata(*, nnrp_range: str = ">=1.0.0rc4.post20,<1.0.0rc5") -> str:
    return "\n".join(
        (
            "Metadata-Version: 2.4",
            "Name: vllm-nnrp-adapter",
            "Version: 0.1.0",
            "Description-Content-Type: text/markdown",
            "License-File: LICENSE",
            f"Requires-Dist: nnrp-py{nnrp_range}",
            'Requires-Dist: vllm>=0.18.0,<0.27; extra == "vllm"',
            "",
            "# vLLM NNRP adapter",
        )
    )


def _write_valid_wheel(path: Path, *, extra_files: dict[str, bytes] | None = None) -> None:
    files = {
        "vllm_nnrp_adapter/__init__.py": b"",
        "vllm_nnrp_adapter/py.typed": b"",
        "adapter.dist-info/licenses/LICENSE": b"license",
        "adapter.dist-info/METADATA": _metadata().encode("utf-8"),
        "adapter.dist-info/entry_points.txt": (
            b"[console_scripts]\nvllm-nnrp-adapter = vllm_nnrp_adapter.cli:main\n"
        ),
    }
    files.update(extra_files or {})
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)


def _write_valid_sdist(path: Path, *, excluded_file: str | None = None) -> None:
    files = {
        "adapter/PKG-INFO": _metadata().encode("utf-8"),
        "adapter/LICENSE": b"license",
        "adapter/README.md": b"# adapter",
        "adapter/pyproject.toml": b"[project]",
        "adapter/src/vllm_nnrp_adapter/__init__.py": b"",
        "adapter/src/vllm_nnrp_adapter/py.typed": b"",
    }
    with tarfile.open(path, "w:gz") as archive:
        for name, content in files.items():
            if name == excluded_file:
                continue
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))


def test_distribution_metadata_accepts_frozen_preview4_ranges() -> None:
    validate_metadata_text(_metadata(), artifact="adapter.whl")


def test_distribution_metadata_rejects_preview3_artifact() -> None:
    with pytest.raises(DistributionMetadataError, match="superseded Preview3"):
        validate_metadata_text(_metadata(nnrp_range=">=1.0.0rc3.post1,<1.0.0rc4"), artifact="adapter.whl")


def test_distribution_metadata_rejects_wrong_preview4_range() -> None:
    with pytest.raises(DistributionMetadataError, match="frozen nnrp-py range"):
        validate_metadata_text(_metadata(nnrp_range=">=1.0.0rc4.post19,<1.0.0rc5"), artifact="adapter.whl")


def test_distribution_metadata_rejects_missing_vllm_extra() -> None:
    metadata = _metadata().replace('Requires-Dist: vllm>=0.18.0,<0.27; extra == "vllm"\n', "")

    with pytest.raises(DistributionMetadataError, match="frozen vLLM extra range"):
        validate_metadata_text(metadata, artifact="adapter.whl")


def test_distribution_metadata_rejects_unconditional_vllm_dependency() -> None:
    metadata = _metadata().replace(
        "\n\n# vLLM NNRP adapter",
        "\nRequires-Dist: vllm>=0.18.0,<0.27\n\n# vLLM NNRP adapter",
    )

    with pytest.raises(DistributionMetadataError, match="frozen vLLM extra range"):
        validate_metadata_text(metadata, artifact="adapter.whl")


def test_distribution_metadata_rejects_wrong_vllm_marker() -> None:
    metadata = _metadata().replace('extra == "vllm"', 'python_version >= "3.11"')

    with pytest.raises(DistributionMetadataError, match="frozen vLLM extra range"):
        validate_metadata_text(metadata, artifact="adapter.whl")


@pytest.mark.parametrize(
    ("removed_text", "message"),
    [
        ("Name: vllm-nnrp-adapter\n", "project identity"),
        ("Description-Content-Type: text/markdown\n", "README"),
        ("License-File: LICENSE\n", "LICENSE"),
        ("# vLLM NNRP adapter", "README"),
    ],
)
def test_distribution_metadata_rejects_missing_project_content(removed_text: str, message: str) -> None:
    with pytest.raises(DistributionMetadataError, match=message):
        validate_metadata_text(_metadata().replace(removed_text, ""), artifact="adapter.whl")


def test_validate_wheel_and_sdist_metadata(tmp_path) -> None:
    wheel = tmp_path / "adapter.whl"
    sdist = tmp_path / "adapter.tar.gz"
    _write_valid_wheel(wheel)
    _write_valid_sdist(sdist)

    assert validate_distribution(wheel) == validate_distribution(sdist)


def test_validate_distribution_rejects_unknown_artifact(tmp_path) -> None:
    artifact = tmp_path / "adapter.zip"
    artifact.write_bytes(b"")

    with pytest.raises(DistributionMetadataError, match="unsupported distribution artifact"):
        validate_distribution(artifact)


@pytest.mark.parametrize("forbidden_path", [".ci-venv/Lib/site.py", "artifacts/coverage.xml", "dist/old.whl"])
def test_validate_distribution_rejects_generated_paths(tmp_path, forbidden_path: str) -> None:
    artifact = tmp_path / "adapter.whl"
    _write_valid_wheel(artifact, extra_files={forbidden_path: b"generated"})

    with pytest.raises(DistributionMetadataError, match="forbidden generated paths"):
        validate_distribution(artifact)


@pytest.mark.parametrize(
    "missing_file",
    [
        "vllm_nnrp_adapter/__init__.py",
        "vllm_nnrp_adapter/py.typed",
        "adapter.dist-info/licenses/LICENSE",
        "adapter.dist-info/entry_points.txt",
    ],
)
def test_validate_wheel_rejects_missing_required_files(tmp_path, missing_file: str) -> None:
    artifact = tmp_path / "adapter.whl"
    _write_valid_wheel(artifact)
    with zipfile.ZipFile(artifact) as source:
        retained = {name: source.read(name) for name in source.namelist() if name != missing_file}
    with zipfile.ZipFile(artifact, "w") as archive:
        for name, content in retained.items():
            archive.writestr(name, content)

    with pytest.raises(DistributionMetadataError, match="missing required files|entry_points.txt"):
        validate_distribution(artifact)


def test_validate_wheel_rejects_wrong_console_entrypoint(tmp_path) -> None:
    artifact = tmp_path / "adapter.whl"
    _write_valid_wheel(
        artifact,
        extra_files={
            "adapter.dist-info/entry_points.txt": b"[console_scripts]\nvllm-nnrp-adapter = wrong.module:main\n"
        },
    )

    with pytest.raises(DistributionMetadataError, match="does not expose"):
        validate_distribution(artifact)


def test_validate_wheel_rejects_invalid_entrypoint_file(tmp_path) -> None:
    artifact = tmp_path / "adapter.whl"
    _write_valid_wheel(
        artifact,
        extra_files={"adapter.dist-info/entry_points.txt": b"not an ini document"},
    )

    with pytest.raises(DistributionMetadataError, match="entry_points.txt is invalid"):
        validate_distribution(artifact)


@pytest.mark.parametrize(
    "missing_file",
    [
        "adapter/LICENSE",
        "adapter/README.md",
        "adapter/pyproject.toml",
        "adapter/src/vllm_nnrp_adapter/__init__.py",
        "adapter/src/vllm_nnrp_adapter/py.typed",
    ],
)
def test_validate_sdist_rejects_missing_required_files(tmp_path, missing_file: str) -> None:
    artifact = tmp_path / "adapter.tar.gz"
    _write_valid_sdist(artifact, excluded_file=missing_file)

    with pytest.raises(DistributionMetadataError, match="missing required files"):
        validate_distribution(artifact)


def test_documented_public_symbols_collects_adapter_imports(tmp_path: Path) -> None:
    document = tmp_path / "usage.md"
    document.write_text(
        """# Usage

```python
from vllm_nnrp_adapter import (
    NnrpServerConfig,
    OpenAiNnrpAdapter as Adapter,
)

import vllm_nnrp_adapter as adapter_package

adapter_package.create_vllm_backend
```

```bash
from vllm_nnrp_adapter import NotPython
```
""",
        encoding="utf-8",
    )

    assert documented_public_symbols((document,)) == frozenset(
        {"NnrpServerConfig", "OpenAiNnrpAdapter", "create_vllm_backend"}
    )


def test_documented_public_symbols_rejects_invalid_python_example(tmp_path: Path) -> None:
    document = tmp_path / "usage.md"
    document.write_text("```python\nfrom vllm_nnrp_adapter import\n```\n", encoding="utf-8")

    with pytest.raises(DistributionMetadataError, match="not valid syntax"):
        documented_public_symbols((document,))


def test_documented_public_symbols_rejects_missing_or_empty_sources(tmp_path: Path) -> None:
    with pytest.raises(DistributionMetadataError, match="does not exist"):
        documented_public_symbols((tmp_path / "missing.md",))

    document = tmp_path / "usage.md"
    document.write_text("# No imports\n", encoding="utf-8")
    with pytest.raises(DistributionMetadataError, match="does not import"):
        documented_public_symbols((document,))
