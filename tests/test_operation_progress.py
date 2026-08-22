from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from nnrp.runtime import ProgressMetadata

from vllm_nnrp_adapter.operation_progress import OperationProgressReporter, OperationProgressStage


@dataclass
class ProgressOperation:
    operation_id: int
    events: list[ProgressMetadata] = field(default_factory=list)

    async def send_progress(self, metadata: ProgressMetadata, body: bytes = b"") -> None:
        assert body == b""
        self.events.append(metadata)


@pytest.mark.asyncio
async def test_progress_reporter_emits_monotonic_frozen_stages_and_skips_duplicates() -> None:
    operation = ProgressOperation(44)
    observed: list[OperationProgressStage] = []
    reporter = OperationProgressReporter(operation, observer=observed.append)  # type: ignore[arg-type]

    await reporter.emit(OperationProgressStage.QUEUED)
    await reporter.emit(OperationProgressStage.QUEUED)
    await reporter.emit(OperationProgressStage.EXECUTING, percent_x100=2500)

    assert reporter.last_stage is OperationProgressStage.EXECUTING
    assert operation.events == [
        ProgressMetadata(44, 1, 0x0001, 0xFFFF, 0, 0),
        ProgressMetadata(44, 2, 0x0005, 2500, 0, 0),
    ]
    assert observed == [OperationProgressStage.QUEUED, OperationProgressStage.EXECUTING]
