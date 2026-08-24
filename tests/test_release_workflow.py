from __future__ import annotations

from pathlib import Path

RELEASE_WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "release.yml"


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
