from __future__ import annotations

import tarfile
import zipfile
from collections.abc import Iterable
from email.parser import Parser
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

NNRP_PY_RANGE = SpecifierSet(">=1.0.0rc4.post18,<1.0.0rc5")
VLLM_RANGE = SpecifierSet(">=0.18.0,<0.27")
FORBIDDEN_DISTRIBUTION_PARTS = frozenset(
    {".ci-venv", ".venv", ".git", "artifacts", "build", "dist"}
)


class DistributionMetadataError(RuntimeError):
    pass


def validate_metadata_text(metadata_text: str, *, artifact: str) -> None:
    if "rc3" in metadata_text:
        raise DistributionMetadataError(f"{artifact} contains a superseded Preview3 dependency")

    metadata = Parser().parsestr(metadata_text)
    requirements = [Requirement(value) for value in metadata.get_all("Requires-Dist", [])]
    nnrp_requirement = next((item for item in requirements if item.name == "nnrp-py"), None)
    if nnrp_requirement is None or nnrp_requirement.specifier != NNRP_PY_RANGE:
        raise DistributionMetadataError(f"{artifact} does not carry the frozen nnrp-py range {NNRP_PY_RANGE}")

    vllm_requirement = next(
        (item for item in requirements if item.name == "vllm" and item.marker and "extra" in str(item.marker)),
        None,
    )
    if vllm_requirement is None or vllm_requirement.specifier != VLLM_RANGE:
        raise DistributionMetadataError(f"{artifact} does not carry the frozen vLLM extra range {VLLM_RANGE}")


def validate_distribution(path: Path) -> None:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            _validate_archive_names(names, artifact=path.name)
            metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
            if len(metadata_names) != 1:
                raise DistributionMetadataError(f"{path.name} must contain exactly one METADATA file")
            metadata_text = archive.read(metadata_names[0]).decode("utf-8")
    elif path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            members = archive.getmembers()
            _validate_archive_names((member.name for member in members), artifact=path.name)
            metadata_members = [member for member in members if member.name.endswith("/PKG-INFO")]
            if len(metadata_members) != 1:
                raise DistributionMetadataError(f"{path.name} must contain exactly one PKG-INFO file")
            extracted = archive.extractfile(metadata_members[0])
            if extracted is None:
                raise DistributionMetadataError(f"{path.name} PKG-INFO could not be read")
            metadata_text = extracted.read().decode("utf-8")
    else:
        raise DistributionMetadataError(f"unsupported distribution artifact: {path.name}")

    validate_metadata_text(metadata_text, artifact=path.name)


def _validate_archive_names(names: Iterable[str], *, artifact: str) -> None:
    for name in names:
        parts = tuple(part for part in str(name).replace("\\", "/").split("/") if part)
        forbidden = FORBIDDEN_DISTRIBUTION_PARTS.intersection(parts)
        if forbidden:
            joined = ", ".join(sorted(forbidden))
            raise DistributionMetadataError(f"{artifact} contains forbidden generated paths: {joined}")
