from __future__ import annotations

import pytest

from vllm_nnrp_adapter.stale_work_workload import build_stale_work_schedule


def _manifest(*, ratio: float, sample_count: int = 100, random_seed: int = 17) -> dict[str, object]:
    return {
        "scenario": "stale_work",
        "stale_work_ratio": ratio,
        "model": "public-test-model",
        "engine": "vllm-0.26",
        "gpu": "test-gpu-class",
        "arrival_schedule": "seeded-fixed-interval",
        "arrival_interval_seconds": 0.025,
        "cancellation_schedule": "seeded-stale-selection",
        "prompt_tokens": 4096,
        "max_completion_tokens": 128,
        "warmup": 2,
        "random_seed": random_seed,
        "sample_count": sample_count,
        "max_in_flight": 4,
        "gpu_accounting": {
            "method": "device_active_time",
            "scope": "dedicated_device",
            "source": "test-device-counter",
        },
    }


@pytest.mark.parametrize("ratio", [0.1, 0.3, 0.5])
def test_schedule_is_deterministic_balanced_and_matches_ratio(ratio: float) -> None:
    manifest = _manifest(ratio=ratio)

    first = build_stale_work_schedule(manifest)
    second = build_stale_work_schedule(manifest)
    stale = [case for case in first if case.is_stale]

    assert first == second
    assert len(first) == 100
    assert len(stale) == int(100 * ratio)
    assert {case.control_kind for case in stale} == {"cancel", "abort", "deadline", "supersede"}
    assert first[3].scheduled_offset_seconds == pytest.approx(0.075)
    assert [case.sample_id for case in first] == [f"sample-{index:06d}" for index in range(100)]


def test_schedule_changes_stale_selection_with_seed_but_not_arrival_order() -> None:
    first = build_stale_work_schedule(_manifest(ratio=0.3, random_seed=1))
    second = build_stale_work_schedule(_manifest(ratio=0.3, random_seed=2))

    assert [case.ordinal for case in first] == [case.ordinal for case in second]
    assert [case.is_stale for case in first] != [case.is_stale for case in second]


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        (_manifest(ratio=0.1, sample_count=11), "represent stale_work_ratio exactly"),
        (_manifest(ratio=0.1, sample_count=10), "at least one cancel, abort, deadline, and supersede"),
    ],
)
def test_schedule_rejects_unrepresentable_or_incomplete_control_mix(
    manifest: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_stale_work_schedule(manifest)
