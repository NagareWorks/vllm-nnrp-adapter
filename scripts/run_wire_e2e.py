from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the adapter against the independent NNRP wire suite.")
    parser.add_argument("--conformance-root", type=Path, required=True)
    parser.add_argument("--artifact-directory", type=Path, required=True)
    args = parser.parse_args()

    conformance_root = args.conformance_root.resolve()
    artifacts = args.artifact_directory.resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    target_manifest = artifacts / "target.json"
    plan = artifacts / "plan.json"
    results = artifacts / "results.json"
    evidence = artifacts / "evidence"
    target_log = artifacts / "target.log"
    for path in (target_manifest, plan, results, target_log):
        path.unlink(missing_ok=True)

    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    target_command = [
        sys.executable,
        "-m",
        "vllm_nnrp_adapter.cli",
        "serve-wire-target",
        "--ready-output",
        str(target_manifest),
    ]
    with target_log.open("w", encoding="utf-8") as log:
        target = subprocess.Popen(
            target_command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            text=True,
        )
        try:
            _wait_for_ready(target, target_manifest)
            _run_runner(
                conformance_root,
                [
                    "wire-plan",
                    "--suite",
                    str(conformance_root / "wire-conformance/nnrp-1-preview4/manifest.json"),
                    "--target",
                    str(target_manifest),
                    "--scenarios",
                    str(conformance_root / "wire-conformance/nnrp-1-preview4/cases/openai-compatible-e2e.json"),
                    "--output",
                    str(plan),
                    "--results-path",
                    str(results),
                    "--evidence-dir",
                    str(evidence),
                ],
                environment,
            )
            _run_runner(
                conformance_root,
                ["wire-run", "--plan", str(plan), "--target", str(target_manifest), "--output", str(results)],
                environment,
            )
            _run_runner(
                conformance_root,
                ["validate-wire-results", "--plan", str(plan), "--results", str(results)],
                environment,
            )
        finally:
            _stop_target(target)
    return 0


def _wait_for_ready(target: subprocess.Popen[str], target_manifest: Path) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if target_manifest.is_file():
            return
        return_code = target.poll()
        if return_code is not None:
            raise RuntimeError(f"wire target exited before readiness with code {return_code}")
        time.sleep(0.05)
    raise TimeoutError("wire target did not publish readiness within 30 seconds")


def _run_runner(conformance_root: Path, arguments: list[str], environment: dict[str, str]) -> None:
    subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--manifest-path",
            str(conformance_root / "Cargo.toml"),
            "-p",
            "nnrp-conformance-runner",
            "--",
            *arguments,
        ],
        check=True,
        env=environment,
    )


def _stop_target(target: subprocess.Popen[str]) -> None:
    if target.poll() is not None:
        return
    target.terminate()
    try:
        target.wait(timeout=10)
    except subprocess.TimeoutExpired:
        target.kill()
        target.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
