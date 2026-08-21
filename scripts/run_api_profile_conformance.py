from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the suite-owned OpenAI NNRP API profile plans.")
    parser.add_argument("--conformance-root", type=Path, required=True)
    parser.add_argument("--artifact-directory", type=Path, required=True)
    parser.add_argument(
        "--capabilities",
        type=Path,
        default=Path("conformance/openai-api-capabilities.json"),
    )
    args = parser.parse_args()

    conformance_root = args.conformance_root.resolve()
    artifact_root = args.artifact_directory.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    capabilities = _load_object(args.capabilities.resolve())

    full_directory = artifact_root / "declared-capabilities"
    _reset_directory(full_directory, artifact_root)
    _run_profile(conformance_root, args.capabilities.resolve(), full_directory)

    baseline_directory = artifact_root / "level1-without-extensions"
    _reset_directory(baseline_directory, artifact_root)
    baseline_capabilities = dict(capabilities)
    baseline_capabilities["extensions"] = []
    baseline_capabilities_path = baseline_directory / "capabilities.json"
    _write_json(baseline_capabilities_path, baseline_capabilities)
    _run_profile(conformance_root, baseline_capabilities_path, baseline_directory)

    full_results = _load_object(full_directory / "results.json")
    baseline_results = _load_object(baseline_directory / "results.json")
    full_ids = _passed_case_ids(full_results)
    baseline_ids = _passed_case_ids(baseline_results)
    if full_ids != baseline_ids:
        raise RuntimeError("Level 1 case coverage changed when non-critical extensions were removed")

    _write_json(
        artifact_root / "summary.json",
        {
            "profile": full_results["profile"],
            "schema_version": full_results["schema_version"],
            "selected_cases": len(full_ids),
            "passed_cases": len(full_ids),
            "extension_independent": True,
            "case_ids": sorted(full_ids),
        },
    )
    return 0


def _run_profile(conformance_root: Path, capabilities_path: Path, output_directory: Path) -> None:
    plan = output_directory / "plan.json"
    results = output_directory / "results.json"
    evidence = output_directory / "evidence"
    _run_runner(
        conformance_root,
        [
            "api-profile-plan",
            "--protocol",
            str(conformance_root / "protocol/nnrp-1-preview4/manifest.json"),
            "--profile",
            str(conformance_root / "profiles/openai-compatible/1/manifest.json"),
            "--capabilities",
            str(capabilities_path),
            "--output",
            str(plan),
            "--results-path",
            str(results),
            "--evidence-dir",
            str(evidence),
        ],
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "vllm_nnrp_adapter.cli",
            "run-conformance-plan",
            "--plan",
            str(plan),
            "--output",
            str(results),
            "--backend",
            "mock",
        ],
        check=True,
    )
    _run_runner(
        conformance_root,
        ["validate-api-profile-results", "--plan", str(plan), "--results", str(results)],
    )
    report = _load_object(results)
    case_ids = _passed_case_ids(report)
    evidence_files = list(evidence.glob("*.json"))
    if len(evidence_files) != len(case_ids):
        raise RuntimeError(f"expected {len(case_ids)} case evidence files, found {len(evidence_files)}")


def _run_runner(conformance_root: Path, arguments: list[str]) -> None:
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
    )


def _passed_case_ids(report: Mapping[str, Any]) -> set[str]:
    results = report.get("results")
    if not isinstance(results, list) or not results:
        raise TypeError("results must be a non-empty JSON array")
    case_ids: set[str] = set()
    for result in results:
        if not isinstance(result, Mapping):
            raise TypeError("results[] must be a JSON object")
        case_id = result.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise TypeError("results[].id must be a non-empty string")
        if result.get("outcome") != "passed":
            raise RuntimeError(f"API profile case did not pass: {case_id}")
        case_ids.add(case_id)
    if len(case_ids) != len(results):
        raise RuntimeError("API profile results contain duplicate case ids")
    return case_ids


def _reset_directory(path: Path, root: Path) -> None:
    if path.parent != root:
        raise ValueError(f"refusing to reset artifact directory outside {root}: {path}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(value, indent=2, sort_keys=True)}\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
