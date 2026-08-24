from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


class ReleaseIdentityError(RuntimeError):
    pass


def git_commit(repository: Path, revision: str) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{revision}^{{commit}}"],
        cwd=repository,
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode == 1:
        return None
    if result.returncode != 0:
        raise ReleaseIdentityError(result.stderr.strip() or f"failed to resolve {revision}")
    return result.stdout.strip()


def validate_git_identity(repository: Path, expected_ref: str, tag_name: str) -> str:
    head = git_commit(repository, "HEAD")
    expected = git_commit(repository, expected_ref)
    if head is None or expected is None:
        raise ReleaseIdentityError(f"release source or expected ref is missing: HEAD, {expected_ref}")
    if head != expected:
        raise ReleaseIdentityError(f"release source {head} does not match {expected_ref} at {expected}")

    tag_commit = git_commit(repository, f"refs/tags/{tag_name}")
    if tag_commit is not None and tag_commit != head:
        raise ReleaseIdentityError(f"release tag {tag_name} already points to {tag_commit}, not {head}")
    return head


def main() -> None:
    parser = argparse.ArgumentParser(description="Reject ambiguous or reused adapter release identities.")
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--expected-ref", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    source_commit = validate_git_identity(args.repository.resolve(), args.expected_ref, args.tag)
    print(f"verified release identity: {args.tag} from {source_commit}")


if __name__ == "__main__":
    main()
