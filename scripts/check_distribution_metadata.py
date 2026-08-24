from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

from vllm_nnrp_adapter.distribution_metadata import (
    DistributionIdentity,
    DistributionMetadataError,
    validate_distribution,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist_dir", type=Path)
    args = parser.parse_args()
    artifacts = sorted((*args.dist_dir.glob("*.whl"), *args.dist_dir.glob("*.tar.gz")))
    if not artifacts:
        raise DistributionMetadataError(f"no distributions found in {args.dist_dir}")
    wheels = [artifact for artifact in artifacts if artifact.suffix == ".whl"]
    sdists = [artifact for artifact in artifacts if artifact.name.endswith(".tar.gz")]
    if len(wheels) != 1 or len(sdists) != 1:
        raise DistributionMetadataError(
            f"{args.dist_dir} must contain exactly one wheel and one sdist; "
            f"found {len(wheels)} wheel(s) and {len(sdists)} sdist(s)"
        )

    identities: set[DistributionIdentity] = set()
    for artifact in artifacts:
        identities.add(validate_distribution(artifact))
    if len(identities) != 1:
        raise DistributionMetadataError("wheel and sdist project identities do not match")

    _validate_clean_install(wheels[0], identities.pop())
    return 0


def _validate_clean_install(wheel: Path, identity: DistributionIdentity) -> None:
    wheel = wheel.resolve()
    with tempfile.TemporaryDirectory(prefix=".install-check-", dir=wheel.parent) as temporary_directory:
        environment = Path(temporary_directory) / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        console_script = environment / (
            "Scripts/vllm-nnrp-adapter.exe" if sys.platform == "win32" else "bin/vllm-nnrp-adapter"
        )
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-cache-dir",
                str(wheel),
            ],
            check=True,
        )
        subprocess.run(
            [
                str(python),
                "-c",
                (
                    "import importlib.metadata; import importlib.util; import vllm_nnrp_adapter; "
                    "assert importlib.metadata.version('vllm-nnrp-adapter') == "
                    f"{identity.version!r}; "
                    "assert importlib.util.find_spec('vllm') is None"
                ),
            ],
            check=True,
        )
        subprocess.run([str(console_script), "--help"], check=True)


if __name__ == "__main__":
    raise SystemExit(main())
