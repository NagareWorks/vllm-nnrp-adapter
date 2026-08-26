from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from vllm_nnrp_adapter.capability_ledger import (
    RUNTIME_CAPABILITY_LEDGER,
    CapabilityClassification,
)


def validate_runtime_capability_evidence(repository_root: Path) -> None:
    root = repository_root.resolve()
    core_entries = tuple(
        entry
        for entry in RUNTIME_CAPABILITY_LEDGER
        if entry.classification is CapabilityClassification.CORE
    )
    if not core_entries:
        raise RuntimeError("runtime capability ledger does not define any core release claims")

    loaded_artifacts: dict[Path, dict[str, Mapping[str, object]]] = {}
    for entry in core_entries:
        if not entry.advertised_by_default or not entry.release_gate_ready:
            raise RuntimeError(f"core runtime capability is not release-gate ready: {entry.token}")
        assert entry.evidence_artifact is not None
        assert entry.independent_scenario is not None
        artifact = (root / entry.evidence_artifact).resolve()
        if not artifact.is_relative_to(root):
            raise RuntimeError(f"capability evidence escapes the repository root: {entry.token}")
        scenario_results = loaded_artifacts.get(artifact)
        if scenario_results is None:
            scenario_results = _load_scenario_results(artifact)
            loaded_artifacts[artifact] = scenario_results
        result = scenario_results.get(entry.independent_scenario)
        if result is None:
            raise RuntimeError(
                f"capability evidence does not contain {entry.independent_scenario}: {entry.token}"
            )
        if result.get("outcome") != "passed":
            raise RuntimeError(
                f"capability evidence scenario did not pass: {entry.independent_scenario}"
            )
        terminal = result.get("terminal")
        if not isinstance(terminal, str) or not terminal:
            raise RuntimeError(f"capability evidence scenario has no terminal: {entry.independent_scenario}")
        evidence_paths = result.get("evidence_paths")
        if not isinstance(evidence_paths, list) or not evidence_paths or not all(
            isinstance(path, str) and path for path in evidence_paths
        ):
            raise RuntimeError(
                f"capability evidence scenario has no reproducible evidence path: {entry.independent_scenario}"
            )


def _load_scenario_results(path: Path) -> dict[str, Mapping[str, object]]:
    if not path.is_file():
        raise RuntimeError(f"capability evidence artifact does not exist: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise TypeError(f"capability evidence artifact must contain a JSON object: {path}")
    results = document.get("results")
    if not isinstance(results, list) or not results:
        raise TypeError(f"capability evidence artifact must contain non-empty results: {path}")

    indexed: dict[str, Mapping[str, object]] = {}
    for index, result in enumerate(results):
        if not isinstance(result, Mapping):
            raise TypeError(f"capability evidence results[{index}] must be a JSON object: {path}")
        scenario_id = result.get("id")
        if not isinstance(scenario_id, str) or not scenario_id:
            raise TypeError(f"capability evidence results[{index}].id must be a non-empty string: {path}")
        if scenario_id in indexed:
            raise RuntimeError(f"capability evidence contains duplicate scenario id: {scenario_id}")
        indexed[scenario_id] = result
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate evidence for core runtime capability claims.")
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    args = parser.parse_args()
    validate_runtime_capability_evidence(args.repository_root)
    print("Core runtime capability evidence is complete and passing.")


if __name__ == "__main__":
    main()
