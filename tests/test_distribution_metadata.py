from __future__ import annotations

import io
import tarfile
import zipfile

import pytest

from vllm_nnrp_adapter.distribution_metadata import (
    DistributionMetadataError,
    validate_distribution,
    validate_metadata_text,
)


def _metadata(*, nnrp_range: str = ">=1.0.0rc4.post15,<1.0.0rc5") -> str:
    return "\n".join(
        (
            "Metadata-Version: 2.4",
            "Name: vllm-nnrp-adapter",
            "Version: 0.1.0",
            f"Requires-Dist: nnrp-py{nnrp_range}",
            'Requires-Dist: vllm>=0.18.0,<0.27; extra == "vllm"',
            "",
        )
    )


def test_distribution_metadata_accepts_frozen_preview4_ranges() -> None:
    validate_metadata_text(_metadata(), artifact="adapter.whl")


def test_distribution_metadata_rejects_preview3_artifact() -> None:
    with pytest.raises(DistributionMetadataError, match="superseded Preview3"):
        validate_metadata_text(_metadata(nnrp_range=">=1.0.0rc3.post1,<1.0.0rc4"), artifact="adapter.whl")


def test_distribution_metadata_rejects_wrong_preview4_range() -> None:
    with pytest.raises(DistributionMetadataError, match="frozen nnrp-py range"):
        validate_metadata_text(_metadata(nnrp_range=">=1.0.0rc4.post13,<1.0.0rc5"), artifact="adapter.whl")


def test_distribution_metadata_rejects_missing_vllm_extra() -> None:
    metadata = _metadata().replace('Requires-Dist: vllm>=0.18.0,<0.27; extra == "vllm"\n', "")

    with pytest.raises(DistributionMetadataError, match="frozen vLLM extra range"):
        validate_metadata_text(metadata, artifact="adapter.whl")


def test_validate_wheel_and_sdist_metadata(tmp_path) -> None:
    metadata = _metadata().encode("utf-8")
    wheel = tmp_path / "adapter.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("adapter.dist-info/METADATA", metadata)

    sdist = tmp_path / "adapter.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        member = tarfile.TarInfo("adapter/PKG-INFO")
        member.size = len(metadata)
        archive.addfile(member, io.BytesIO(metadata))

    validate_distribution(wheel)
    validate_distribution(sdist)


def test_validate_distribution_rejects_unknown_artifact(tmp_path) -> None:
    artifact = tmp_path / "adapter.zip"
    artifact.write_bytes(b"")

    with pytest.raises(DistributionMetadataError, match="unsupported distribution artifact"):
        validate_distribution(artifact)


@pytest.mark.parametrize("forbidden_path", [".ci-venv/Lib/site.py", "artifacts/coverage.xml", "dist/old.whl"])
def test_validate_distribution_rejects_generated_paths(tmp_path, forbidden_path: str) -> None:
    artifact = tmp_path / "adapter.whl"
    with zipfile.ZipFile(artifact, "w") as archive:
        archive.writestr("adapter.dist-info/METADATA", _metadata())
        archive.writestr(forbidden_path, b"generated")

    with pytest.raises(DistributionMetadataError, match="forbidden generated paths"):
        validate_distribution(artifact)
