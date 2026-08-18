from __future__ import annotations

import argparse
from pathlib import Path

from vllm_nnrp_adapter.distribution_metadata import DistributionMetadataError, validate_distribution


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist_dir", type=Path)
    args = parser.parse_args()
    artifacts = sorted((*args.dist_dir.glob("*.whl"), *args.dist_dir.glob("*.tar.gz")))
    if not artifacts:
        raise DistributionMetadataError(f"no distributions found in {args.dist_dir}")
    for artifact in artifacts:
        validate_distribution(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
