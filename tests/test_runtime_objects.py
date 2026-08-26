from __future__ import annotations

import pytest
from nnrp.core import CacheInvalidateMetadata, CacheInvalidateScope
from nnrp.runtime import (
    CacheMissMetadata,
    CacheMissReason,
    CacheReferenceMetadata,
    CacheReuseScope,
    MemoryLocationHint,
    ObjectDeltaMetadata,
    ObjectDescriptorMetadata,
    ObjectReferenceMetadata,
    ObjectReleaseMetadata,
    ObjectReleaseReason,
    OwnershipHint,
    RuntimeObjectKind,
    RuntimeRole,
)

from vllm_nnrp_adapter.runtime_objects import (
    RuntimeObjectError,
    RuntimeObjectFailureCode,
    RuntimeObjectRegistry,
)


def _descriptor(*, object_id: int = 9, byte_size: int = 16, metadata_bytes: int = 0) -> ObjectDescriptorMetadata:
    return ObjectDescriptorMetadata(
        object_id=object_id,
        object_kind=RuntimeObjectKind.IMAGE_TILE,
        producer_role=RuntimeRole.CLIENT,
        consumer_role=RuntimeRole.RUNTIME,
        session_id=7,
        byte_size=byte_size,
        compute_cost_units=4,
        memory_location_hint=MemoryLocationHint.HOST_MEMORY,
        ownership_hint=OwnershipHint.SESSION_OWNED,
        lifetime_hint_ms=1_000,
        metadata_bytes=metadata_bytes,
    )


def _reference(
    *,
    object_id: int = 9,
    operation_id: int = 21,
    object_version: int = 3,
    offset: int = 0,
    length: int = 0,
    flags: int = 0,
) -> ObjectReferenceMetadata:
    return ObjectReferenceMetadata(object_id, operation_id, object_version, offset, length, flags, 0)


def test_registry_keeps_object_version_explicit_and_validates_regions() -> None:
    registry = RuntimeObjectRegistry(session_id=7)
    registry.declare(_descriptor(metadata_bytes=4), b"meta")

    assert registry.object_snapshot(9).current_version is None
    registry.bind_version(9, 3)
    whole = registry.reference(_reference())
    region = registry.reference(_reference(operation_id=22, offset=4, length=8, flags=0x04))

    assert whole.metadata.operation_id == 21
    assert region.metadata.operation_id == 22
    assert registry.object_snapshot(9).current_version == 3
    assert registry.unresolved_references_for_operation(21) == (whole,)
    registry.bind_reference_resolution(21, 9, current_version=3)
    assert registry.unresolved_references_for_operation(21) == ()
    with pytest.raises(RuntimeObjectError, match="region-present"):
        registry.reference(_reference(operation_id=23, offset=1, length=1))
    with pytest.raises(RuntimeObjectError, match="exceeds"):
        registry.reference(_reference(operation_id=23, offset=12, length=8, flags=0x04))
    with pytest.raises(RuntimeObjectError) as mismatch:
        registry.reference(_reference(operation_id=23, object_version=2))
    assert mismatch.value.code is RuntimeObjectFailureCode.VERSION_MISMATCH


def test_object_descriptor_rejects_unspecified_and_reserved_registry_values() -> None:
    registry = RuntimeObjectRegistry(session_id=7)
    base = _descriptor()

    invalid_descriptors = (
        ObjectDescriptorMetadata(
            base.object_id,
            RuntimeObjectKind.UNSPECIFIED,
            base.producer_role,
            base.consumer_role,
            base.session_id,
            base.byte_size,
            base.compute_cost_units,
            base.memory_location_hint,
            base.ownership_hint,
            base.lifetime_hint_ms,
            base.metadata_bytes,
        ),
        ObjectDescriptorMetadata(
            base.object_id,
            base.object_kind,
            RuntimeRole.UNSPECIFIED,
            base.consumer_role,
            base.session_id,
            base.byte_size,
            base.compute_cost_units,
            base.memory_location_hint,
            base.ownership_hint,
            base.lifetime_hint_ms,
            base.metadata_bytes,
        ),
        ObjectDescriptorMetadata(
            base.object_id,
            base.object_kind,
            base.producer_role,
            base.consumer_role,
            base.session_id,
            base.byte_size,
            base.compute_cost_units,
            0x0007,
            base.ownership_hint,
            base.lifetime_hint_ms,
            base.metadata_bytes,
        ),
        ObjectDescriptorMetadata(
            base.object_id,
            base.object_kind,
            base.producer_role,
            base.consumer_role,
            base.session_id,
            base.byte_size,
            base.compute_cost_units,
            base.memory_location_hint,
            OwnershipHint.UNSPECIFIED,
            base.lifetime_hint_ms,
            base.metadata_bytes,
        ),
    )

    for descriptor in invalid_descriptors:
        with pytest.raises(RuntimeObjectError):
            registry.declare(descriptor)


def test_object_descriptor_accepts_frozen_private_registry_ranges() -> None:
    registry = RuntimeObjectRegistry(session_id=7)
    descriptor = ObjectDescriptorMetadata(
        9,
        0x8001,
        0x80,
        0x81,
        7,
        16,
        4,
        0x8002,
        0x8003,
        1_000,
        0,
    )

    registry.declare(descriptor)

    assert registry.object_snapshot(9).descriptor is descriptor


@pytest.mark.parametrize(
    "reference",
    [
        _reference(object_version=-1),
        _reference(offset=-1, length=1, flags=0x04),
        _reference(offset=0, length=-1, flags=0x04),
        _reference(flags=0x08),
    ],
)
def test_object_reference_rejects_out_of_range_fields(reference: ObjectReferenceMetadata) -> None:
    registry = RuntimeObjectRegistry(session_id=7)
    registry.declare(_descriptor())

    with pytest.raises(RuntimeObjectError):
        registry.reference(reference)


def test_delta_sequence_is_independent_from_object_version() -> None:
    registry = RuntimeObjectRegistry(session_id=7)
    registry.declare(_descriptor())
    registry.bind_version(9, 5)

    registry.apply_delta(ObjectDeltaMetadata(9, 10, 0, 4, 4, 0x01, 2), metadata_body=b"m1", delta=b"data")

    snapshot = registry.object_snapshot(9)
    assert snapshot.current_version == 5
    assert snapshot.last_delta_sequence == 10
    with pytest.raises(RuntimeObjectError, match="monotonically"):
        registry.apply_delta(ObjectDeltaMetadata(9, 10, 0, 4, 4, 0x01, 0), delta=b"next")


def test_final_release_invalidates_dependent_operations() -> None:
    registry = RuntimeObjectRegistry(session_id=7)
    registry.declare(_descriptor())
    registry.reference(_reference(operation_id=21, object_version=0))
    registry.reference(_reference(operation_id=22, object_version=0))

    affected = registry.release(
        ObjectReleaseMetadata(
            9,
            0,
            ObjectReleaseReason.INVALIDATED,
            RuntimeRole.CLIENT,
            0x03,
            4,
        ),
        b"gone",
    )

    assert affected == (21, 22)
    assert registry.object_snapshot(9).released is True
    assert registry.references_for_operation(21) == ()
    failure = registry.operation_failure(21)
    assert failure is not None
    assert failure.code is RuntimeObjectFailureCode.DEPENDENCY_INVALID
    with pytest.raises(RuntimeObjectError, match="released"):
        registry.reference(_reference(operation_id=23, object_version=0))


def test_non_final_release_only_retires_the_named_operation_reference() -> None:
    registry = RuntimeObjectRegistry(session_id=7)
    registry.declare(_descriptor())
    registry.reference(_reference(operation_id=21, object_version=0))
    registry.reference(_reference(operation_id=22, object_version=0))

    affected = registry.release(
        ObjectReleaseMetadata(9, 21, ObjectReleaseReason.COMPLETED, RuntimeRole.CLIENT, 0, 0)
    )

    assert affected == ()
    assert registry.references_for_operation(21) == ()
    assert len(registry.references_for_operation(22)) == 1
    assert registry.object_snapshot(9).released is False


def test_operation_failure_keeps_the_first_object_error() -> None:
    registry = RuntimeObjectRegistry(session_id=7)
    first = RuntimeObjectError(0, "invalid range", object_id=9, operation_id=21)
    later = RuntimeObjectError(
        RuntimeObjectFailureCode.VERSION_MISMATCH,
        "later mismatch",
        object_id=9,
        operation_id=21,
    )

    registry.record_operation_failure(first)
    registry.record_operation_failure(later)

    assert registry.operation_failure(21) is first


def test_cache_state_is_explicit_and_invalidation_respects_scope() -> None:
    registry = RuntimeObjectRegistry(session_id=7)
    first = CacheReferenceMetadata(3, 11, 12, 5, CacheReuseScope.SESSION, 8, 9, 100, 0, 0)
    second = CacheReferenceMetadata(4, 21, 22, 5, CacheReuseScope.SESSION, 0, 0, 0, 0, 0)
    first_key = registry.record_cache_reference(first)
    second_key = registry.record_cache_reference(second)

    assert registry.cache_snapshot(first_key).available is None
    registry.bind_cache_resolution(first_key, available=True, object_kind=7)
    registry.record_cache_miss(CacheMissMetadata(4, 21, 22, CacheMissReason.EXPIRED, 5, 4), b"miss")
    assert registry.cache_snapshot(first_key).available is True
    assert registry.cache_snapshot(second_key).available is False

    invalidated = registry.invalidate_cache(
        CacheInvalidateMetadata(CacheInvalidateScope.NAMESPACE, 3, 0, 0, 1)
    )
    assert invalidated == (first_key,)
    assert registry.cache_snapshot(first_key).available is False


def test_required_cache_lease_must_be_present_and_confirmed_by_provider() -> None:
    registry = RuntimeObjectRegistry(session_id=7)
    missing_lease = CacheReferenceMetadata(
        3,
        11,
        12,
        5,
        CacheReuseScope.SESSION,
        0,
        9,
        100,
        0,
        0x01,
    )
    with pytest.raises(RuntimeObjectError) as missing:
        registry.record_cache_reference(missing_lease)
    assert missing.value.code is RuntimeObjectFailureCode.CACHE_MISS

    reference = CacheReferenceMetadata(
        3,
        11,
        12,
        5,
        CacheReuseScope.SESSION,
        8,
        9,
        100,
        0,
        0x01,
    )
    key = registry.record_cache_reference(reference)
    with pytest.raises(RuntimeObjectError) as unconfirmed:
        registry.bind_cache_resolution(key, available=True)
    assert unconfirmed.value.code is RuntimeObjectFailureCode.LEASE_EXPIRED
    with pytest.raises(RuntimeObjectError) as mismatched:
        registry.bind_cache_resolution(key, available=True, lease_id=7, lease_live=True)
    assert mismatched.value.code is RuntimeObjectFailureCode.LEASE_EXPIRED
    with pytest.raises(RuntimeObjectError) as expired:
        registry.bind_cache_resolution(key, available=True, lease_id=8, lease_live=False)
    assert expired.value.code is RuntimeObjectFailureCode.LEASE_EXPIRED

    registry.bind_cache_resolution(key, available=True, lease_id=8, lease_live=True)

    assert registry.cache_snapshot(key).available is True


def test_object_kind_cache_invalidation_only_matches_resolved_kinds() -> None:
    registry = RuntimeObjectRegistry(session_id=7)
    resolved = registry.record_cache_reference(
        CacheReferenceMetadata(3, 11, 12, 5, CacheReuseScope.SESSION, 8, 9, 100, 0, 0)
    )
    unresolved = registry.record_cache_reference(
        CacheReferenceMetadata(3, 21, 22, 5, CacheReuseScope.SESSION, 0, 0, 0, 0, 0)
    )
    registry.bind_cache_resolution(resolved, available=True, object_kind=7)

    invalidated = registry.invalidate_cache(
        CacheInvalidateMetadata(CacheInvalidateScope.OBJECT_KIND, 3, 7, 0, 1)
    )

    assert invalidated == (resolved,)
    assert registry.cache_snapshot(unresolved).available is None
