from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.check_runtime_capability_evidence import validate_runtime_capability_evidence


def test_capability_evidence_gate_accepts_all_core_release_claims(tmp_path: Path) -> None:
    _write_results(tmp_path)

    validate_runtime_capability_evidence(tmp_path)


@pytest.mark.parametrize(
    ("scenario_id", "field", "value", "message"),
    (
        (
            "wire.control.cancel-abort.client",
            "outcome",
            "failed",
            "did not pass",
        ),
        (
            "wire.control.deadline-before-submit.client",
            "terminal",
            "",
            "has no terminal",
        ),
        (
            "wire.profile.openai-compatible.level1",
            "evidence_paths",
            [],
            "has no reproducible evidence path",
        ),
    ),
)
def test_capability_evidence_gate_rejects_failed_or_incomplete_results(
    tmp_path: Path,
    scenario_id: str,
    field: str,
    value: object,
    message: str,
) -> None:
    results = _results()
    next(result for result in results if result["id"] == scenario_id)[field] = value
    _write_results(tmp_path, results)

    with pytest.raises(RuntimeError, match=message):
        validate_runtime_capability_evidence(tmp_path)


def test_capability_evidence_gate_rejects_missing_independent_scenario(tmp_path: Path) -> None:
    results = [
        result
        for result in _results()
        if result["id"] != "wire.control.deadline-before-submit.client"
    ]
    _write_results(tmp_path, results)

    with pytest.raises(RuntimeError, match="does not contain wire.control.deadline-before-submit.client"):
        validate_runtime_capability_evidence(tmp_path)


def _write_results(tmp_path: Path, results: list[dict[str, object]] | None = None) -> None:
    artifact = tmp_path / "artifacts" / "wire-e2e" / "results.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps({"results": results or _results()}), encoding="utf-8")


def _results() -> list[dict[str, object]]:
    return [
        {
            "id": scenario_id,
            "outcome": "passed",
            "terminal": "cancelled" if scenario_id == "wire.control.cancel-abort.client" else "success",
            "evidence_paths": [f"evidence/{scenario_id}.jsonl"],
        }
        for scenario_id in (
            "wire.profile.openai-compatible.level1",
            "wire.control.cancel-abort.client",
            "wire.control.deadline-before-submit.client",
        )
    ]
