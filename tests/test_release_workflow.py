from __future__ import annotations

from pathlib import Path

RELEASE_WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "release.yml"
CI_WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"


def test_release_is_manual_and_publication_is_opt_in() -> None:
    workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")

    assert "  push:\n" not in workflow
    assert "  workflow_dispatch:" in workflow
    assert "publish_github_release:" in workflow
    assert "publish_to_pypi:" in workflow
    assert workflow.count("default: false") == 3


def test_release_validates_immutable_git_and_registry_identity() -> None:
    workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")

    assert "Validate immutable release identity" in workflow
    assert "Validate immutable release identity\n        if: inputs.create_tag" in workflow
    assert "scripts/check_release_identity.py" in workflow
    assert "--expected-ref origin/main" in workflow
    assert "Resolve PyPI publication identity" in workflow
    assert "scripts/check_pypi_identity.py" in workflow
    assert "steps.pypi.outputs.publish_required == 'true'" in workflow
    assert "Verify public PyPI identity" in workflow
    assert "--require-published" in workflow


def test_release_creates_tag_only_after_validation_and_build() -> None:
    workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")

    assert workflow.index("Validate immutable release identity") < workflow.index("Verify release gates")
    assert workflow.index("Build distributions") < workflow.index("Create git tag")
    assert workflow.index("Resolve PyPI publication identity") < workflow.index("Create git tag")
    assert workflow.index("Create git tag") < workflow.index("Publish GitHub release")


def test_release_requires_preview4_todo_closure_before_validation() -> None:
    workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")

    assert "Validate Preview4 TODO closure\n        if: inputs.create_tag" in workflow
    assert "run: python scripts/check_preview4_todo.py" in workflow
    assert workflow.index("run: python scripts/check_preview4_todo.py") < workflow.index("Create git tag")


def test_release_reruns_pinned_api_and_wire_conformance_before_tagging() -> None:
    workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")

    assert "ref: 7ea6e30ab61a76efc9ce9cf2fe6aa93312edda81" in workflow
    assert "python scripts/run_api_profile_conformance.py" in workflow
    assert "python scripts/run_wire_e2e.py" in workflow
    assert "python scripts/check_runtime_capability_evidence.py" in workflow
    assert "artifacts/api-profile-conformance" in workflow
    assert "artifacts/wire-e2e" in workflow
    assert workflow.index("python scripts/run_wire_e2e.py") < workflow.index("Create git tag")
    assert workflow.index("python scripts/run_wire_e2e.py") < workflow.index(
        "python scripts/check_runtime_capability_evidence.py"
    )


def test_ci_validates_capability_evidence_after_wire_e2e() -> None:
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")

    assert "python scripts/check_runtime_capability_evidence.py" in workflow
    assert workflow.index("python scripts/run_wire_e2e.py") < workflow.index(
        "python scripts/check_runtime_capability_evidence.py"
    )
