from __future__ import annotations

import tarfile
import zipfile
from collections.abc import Iterable
from configparser import ConfigParser
from configparser import Error as ConfigParserError
from dataclasses import dataclass
from email.parser import Parser
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

NNRP_PY_RANGE = SpecifierSet(">=1.0.0rc4.post19,<1.0.0rc5")
VLLM_RANGE = SpecifierSet(">=0.18.0,<0.27")
PROJECT_NAME = "vllm-nnrp-adapter"
CONSOLE_ENTRYPOINT = "vllm_nnrp_adapter.cli:main"
FORBIDDEN_DISTRIBUTION_PARTS = frozenset(
    {".ci-venv", ".venv", ".git", "artifacts", "build", "dist"}
)
WHEEL_REQUIRED_SUFFIXES = (
    "vllm_nnrp_adapter/__init__.py",
    "vllm_nnrp_adapter/py.typed",
    ".dist-info/licenses/LICENSE",
)
SDIST_REQUIRED_SUFFIXES = (
    "/LICENSE",
    "/README.md",
    "/pyproject.toml",
    "/src/vllm_nnrp_adapter/__init__.py",
    "/src/vllm_nnrp_adapter/py.typed",
)


class DistributionMetadataError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DistributionIdentity:
    name: str
    version: str


def validate_metadata_text(metadata_text: str, *, artifact: str) -> DistributionIdentity:
    if "rc3" in metadata_text:
        raise DistributionMetadataError(f"{artifact} contains a superseded Preview3 dependency")

    metadata = Parser().parsestr(metadata_text)
    name = metadata.get("Name")
    version = metadata.get("Version")
    description = metadata.get_payload()
    if name != PROJECT_NAME or not version:
        raise DistributionMetadataError(f"{artifact} does not carry the expected project identity")
    if (
        metadata.get("Description-Content-Type") != "text/markdown"
        or not isinstance(description, str)
        or not description.strip()
    ):
        raise DistributionMetadataError(f"{artifact} does not carry the README as Markdown metadata")
    if "LICENSE" not in metadata.get_all("License-File", []):
        raise DistributionMetadataError(f"{artifact} does not declare the packaged LICENSE file")

    requirements = [Requirement(value) for value in metadata.get_all("Requires-Dist", [])]
    nnrp_requirement = next((item for item in requirements if item.name == "nnrp-py"), None)
    if nnrp_requirement is None or nnrp_requirement.specifier != NNRP_PY_RANGE:
        raise DistributionMetadataError(f"{artifact} does not carry the frozen nnrp-py range {NNRP_PY_RANGE}")

    vllm_requirements = [item for item in requirements if item.name == "vllm"]
    if (
        len(vllm_requirements) != 1
        or vllm_requirements[0].specifier != VLLM_RANGE
        or vllm_requirements[0].marker is None
        or str(vllm_requirements[0].marker) != 'extra == "vllm"'
    ):
        raise DistributionMetadataError(f"{artifact} does not carry the frozen vLLM extra range {VLLM_RANGE}")

    return DistributionIdentity(name=name, version=version)


def validate_distribution(path: Path) -> DistributionIdentity:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            _validate_archive_names(names, artifact=path.name)
            _require_suffixes(names, WHEEL_REQUIRED_SUFFIXES, artifact=path.name)
            metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
            if len(metadata_names) != 1:
                raise DistributionMetadataError(f"{path.name} must contain exactly one METADATA file")
            entrypoint_names = [name for name in names if name.endswith(".dist-info/entry_points.txt")]
            if len(entrypoint_names) != 1:
                raise DistributionMetadataError(f"{path.name} must contain exactly one entry_points.txt file")
            metadata_text = archive.read(metadata_names[0]).decode("utf-8")
            _validate_entrypoints(
                archive.read(entrypoint_names[0]).decode("utf-8"),
                artifact=path.name,
            )
    elif path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            _validate_archive_names(names, artifact=path.name)
            _require_suffixes(names, SDIST_REQUIRED_SUFFIXES, artifact=path.name)
            metadata_members = [member for member in members if member.name.endswith("/PKG-INFO")]
            if len(metadata_members) != 1:
                raise DistributionMetadataError(f"{path.name} must contain exactly one PKG-INFO file")
            extracted = archive.extractfile(metadata_members[0])
            if extracted is None:
                raise DistributionMetadataError(f"{path.name} PKG-INFO could not be read")
            metadata_text = extracted.read().decode("utf-8")
    else:
        raise DistributionMetadataError(f"unsupported distribution artifact: {path.name}")

    return validate_metadata_text(metadata_text, artifact=path.name)


def _validate_archive_names(names: Iterable[str], *, artifact: str) -> None:
    for name in names:
        parts = tuple(part for part in str(name).replace("\\", "/").split("/") if part)
        forbidden = FORBIDDEN_DISTRIBUTION_PARTS.intersection(parts)
        if forbidden:
            joined = ", ".join(sorted(forbidden))
            raise DistributionMetadataError(f"{artifact} contains forbidden generated paths: {joined}")


def _require_suffixes(names: Iterable[str], suffixes: Iterable[str], *, artifact: str) -> None:
    normalized_names = tuple(str(name).replace("\\", "/") for name in names)
    missing = [suffix for suffix in suffixes if not any(name.endswith(suffix) for name in normalized_names)]
    if missing:
        raise DistributionMetadataError(f"{artifact} is missing required files: {', '.join(missing)}")


def _validate_entrypoints(entrypoint_text: str, *, artifact: str) -> None:
    parser = ConfigParser(interpolation=None)
    try:
        parser.read_string(entrypoint_text)
    except ConfigParserError as error:
        raise DistributionMetadataError(f"{artifact} entry_points.txt is invalid") from error
    if parser.get("console_scripts", PROJECT_NAME, fallback=None) != CONSOLE_ENTRYPOINT:
        raise DistributionMetadataError(
            f"{artifact} does not expose {PROJECT_NAME} = {CONSOLE_ENTRYPOINT}"
        )
