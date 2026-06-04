# Contributing to vllm-nnrp-adapter

This repository publishes the vLLM adapter for the NNRP OpenAI-compatible API profile, so contribution flow needs to stay predictable.

## Branch Strategy

`main` is the stable branch for released or release-ready adapter state.

`develop` is the integration branch for active preview work. Preview feature, fix, documentation, and maintenance branches should merge into `develop` first.

Use short-lived topic branches:

- `feature/<scope>-<topic>` for new capabilities
- `fix/<scope>-<topic>` for bug fixes
- `docs/<scope>-<topic>` for documentation-only changes
- `chore/<scope>-<topic>` for maintenance and tooling updates
- `release/<version>` only after `develop` is ready to freeze into a public package release candidate

Rules:

- Branch from the latest `develop` for active preview work.
- Branch from `main` only for hotfixes against already released stable state.
- Keep topic branches focused on one slice of work.
- Merge normal preview work back to `develop` through a pull request.
- Do not push directly to `main` or `develop`.
- Do not publish packages directly from topic branches.

## Commit Message Convention

Use Conventional Commits.

Preferred forms:

- `feat: add chat completion profile mapper`
- `fix: preserve usage events from streaming chunks`
- `docs: clarify vllm support floor`
- `chore: tighten ci dependency bootstrap`
- `test: cover tool call delta mapping`

Normal PRs from feature, fix, docs, or chore branches must contain exactly one commit before review unless they target or originate from a necessary release branch.

## Pull Request Expectations

Every PR should:

- target `develop` for normal preview work, `main` for stable hotfixes, or `release/<version>` during an active release freeze
- explain the user-facing or engineering motivation
- summarize the main modules or flows changed
- list the validation performed
- mention release impact when package output changes
- pass the `required-checks` GitHub Actions job before merge

## Validation Expectations

Before opening or merging a PR, prefer the narrowest validation that proves the touched slice:

- `ruff check .`
- `pytest -q`
- `pytest --cov=src/vllm_nnrp_adapter --cov-report=xml:artifacts/coverage/coverage.xml --cov-fail-under=90 -q`
- `python -m build` when wheel or sdist output changed

Changed production lines under `src/vllm_nnrp_adapter/` must keep at least 90% line coverage in CI.

## Review Guidelines

Review for:

- profile compatibility risk
- vLLM version compatibility risk
- adapter boundary drift
- packaging and release regressions
- missing tests for changed behavior
- CI workflow correctness
- documentation drift when user-facing behavior changes

