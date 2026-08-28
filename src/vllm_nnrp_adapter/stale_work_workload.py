from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .adoption_evidence import normalize_stale_work_manifest

_CONTROL_KINDS = ("cancel", "abort", "deadline", "supersede")


@dataclass(frozen=True)
class StaleWorkCase:
    sample_id: str
    ordinal: int
    scheduled_offset_seconds: float
    model: str
    prompt_tokens: int
    max_completion_tokens: int
    control_kind: str | None

    @property
    def is_stale(self) -> bool:
        return self.control_kind is not None


def build_stale_work_schedule(workload: object) -> tuple[StaleWorkCase, ...]:
    manifest = normalize_stale_work_manifest(workload)
    sample_count = int(manifest["sample_count"])
    stale_work_ratio = float(manifest["stale_work_ratio"])
    stale_count_float = sample_count * stale_work_ratio
    stale_count = round(stale_count_float)
    if not math.isclose(stale_count_float, stale_count, abs_tol=1e-12):
        raise ValueError("workload.sample_count must represent stale_work_ratio exactly")
    if stale_count < len(_CONTROL_KINDS):
        raise ValueError("workload must contain at least one cancel, abort, deadline, and supersede sample")

    random_seed = int(manifest["random_seed"])
    rng = random.Random(random_seed)
    stale_ordinals = set(rng.sample(range(sample_count), stale_count))
    controls = [_CONTROL_KINDS[index % len(_CONTROL_KINDS)] for index in range(stale_count)]
    rng.shuffle(controls)
    control_iterator = iter(controls)
    interval = float(manifest["arrival_interval_seconds"])
    model = str(manifest["model"])
    prompt_tokens = int(manifest["prompt_tokens"])
    max_completion_tokens = int(manifest["max_completion_tokens"])

    return tuple(
        StaleWorkCase(
            sample_id=f"sample-{ordinal:06d}",
            ordinal=ordinal,
            scheduled_offset_seconds=ordinal * interval,
            model=model,
            prompt_tokens=prompt_tokens,
            max_completion_tokens=max_completion_tokens,
            control_kind=next(control_iterator) if ordinal in stale_ordinals else None,
        )
        for ordinal in range(sample_count)
    )
