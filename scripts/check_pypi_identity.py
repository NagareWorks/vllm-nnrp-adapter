from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


class PyPiIdentityError(RuntimeError):
    pass


def distribution_digests(directory: Path) -> dict[str, str]:
    distributions = sorted((*directory.glob("*.whl"), *directory.glob("*.tar.gz")))
    if not distributions:
        raise PyPiIdentityError(f"no wheel or source distribution found in {directory}")
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in distributions}


def pypi_distribution_digests(
    package_name: str,
    package_version: str,
    *,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, str] | None:
    package = urllib.parse.quote(package_name, safe="")
    version = urllib.parse.quote(package_version, safe="")
    request = urllib.request.Request(
        f"https://pypi.org/pypi/{package}/{version}/json",
        headers={"Accept": "application/json"},
    )
    try:
        with opener(request, timeout=20) as response:
            document = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise PyPiIdentityError(f"PyPI identity lookup failed with HTTP {error.code}") from error
    except (OSError, ValueError) as error:
        raise PyPiIdentityError(f"PyPI identity lookup failed: {error}") from error

    try:
        return {item["filename"]: item["digests"]["sha256"] for item in document["urls"]}
    except (KeyError, TypeError) as error:
        raise PyPiIdentityError("PyPI identity response is missing distribution digests") from error


def publication_required(
    local: Mapping[str, str],
    published: Mapping[str, str] | None,
    *,
    require_published: bool = False,
) -> bool:
    if published is None:
        if require_published:
            raise PyPiIdentityError("the requested package version is not available on PyPI")
        return True
    if dict(local) != dict(published):
        local_names = ", ".join(sorted(local))
        published_names = ", ".join(sorted(published))
        raise PyPiIdentityError(
            "PyPI already contains a different distribution set for this version "
            f"(local: {local_names}; published: {published_names})"
        )
    return False


def _write_github_output(path: Path, *, publish_required: bool) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"publish_required={'true' if publish_required else 'false'}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify immutable PyPI distribution identity.")
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--require-published", action="store_true")
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()

    local = distribution_digests(args.dist.resolve())
    published = pypi_distribution_digests(args.package, args.version)
    required = publication_required(local, published, require_published=args.require_published)
    if args.github_output is not None:
        _write_github_output(args.github_output, publish_required=required)
    state = "requires publication" if required else "matches published files"
    print(f"verified PyPI identity: {args.package} {args.version} {state}")


if __name__ == "__main__":
    main()
