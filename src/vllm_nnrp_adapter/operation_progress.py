from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum

from nnrp import NativeRuntimeServerOperation  # type: ignore[import-untyped]
from nnrp.runtime import ProgressMetadata  # type: ignore[import-untyped]


class OperationProgressStage(IntEnum):
    QUEUED = 0x0001
    ADMITTED = 0x0002
    INPUT_RECEIVED = 0x0003
    PREPROCESSING = 0x0004
    EXECUTING = 0x0005
    PRODUCING_PARTIAL = 0x0007
    FINALIZING = 0x0008
    COMPLETED = 0x0009
    DROPPED = 0x000A
    FAILED = 0x000B


class OperationProgressReporter:
    def __init__(
        self,
        operation: NativeRuntimeServerOperation,
        *,
        observer: Callable[[OperationProgressStage], None] | None = None,
    ) -> None:
        self._operation = operation
        self._observer = observer
        self._sequence = 0
        self._last_stage: OperationProgressStage | None = None

    @property
    def last_stage(self) -> OperationProgressStage | None:
        return self._last_stage

    async def emit(self, stage: OperationProgressStage, *, percent_x100: int = 0xFFFF) -> None:
        if stage is self._last_stage:
            return
        self._sequence += 1
        await self._operation.send_progress(
            ProgressMetadata(
                operation_id=self._operation.operation_id,
                progress_sequence=self._sequence,
                stage_code=int(stage),
                percent_x100=percent_x100,
                object_id=0,
                body_bytes=0,
            )
        )
        self._last_stage = stage
        if self._observer is not None:
            self._observer(stage)
