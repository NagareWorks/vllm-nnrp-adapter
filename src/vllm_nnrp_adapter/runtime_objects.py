from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from nnrp.core import CacheInvalidateMetadata, CacheInvalidateScope  # type: ignore[import-untyped]
from nnrp.runtime import (  # type: ignore[import-untyped]
    CacheMissMetadata,
    CacheReferenceMetadata,
    CacheReuseScope,
    MemoryLocationHint,
    ObjectDeltaMetadata,
    ObjectDescriptorMetadata,
    ObjectReferenceMetadata,
    ObjectReleaseMetadata,
    OwnershipHint,
    RuntimeObjectKind,
    RuntimeRole,
)

_RUNTIME_OBJECT_KINDS = frozenset(int(value) for value in RuntimeObjectKind)
_RUNTIME_ROLES = frozenset(int(value) for value in RuntimeRole)
_MEMORY_LOCATION_HINTS = frozenset(int(value) for value in MemoryLocationHint)
_OWNERSHIP_HINTS = frozenset(int(value) for value in OwnershipHint)
_CACHE_REUSE_SCOPES = frozenset(int(value) for value in CacheReuseScope)


class RuntimeObjectFailureCode(IntEnum):
    CACHE_MISS = 0x0003_0001
    LEASE_EXPIRED = 0x0003_0002
    VERSION_MISMATCH = 0x0003_0003
    DEPENDENCY_INVALID = 0x0003_0004
    SCHEMA_MISMATCH = 0x0003_0005


class RuntimeObjectError(ValueError):
    def __init__(
        self,
        code: int,
        diagnostic: str,
        *,
        object_id: int = 0,
        operation_id: int = 0,
    ) -> None:
        super().__init__(diagnostic)
        self.code = code
        self.diagnostic = diagnostic
        self.object_id = object_id
        self.operation_id = operation_id


@dataclass(frozen=True, slots=True)
class RuntimeObjectReference:
    metadata: ObjectReferenceMetadata
    metadata_body: bytes


@dataclass(frozen=True, slots=True)
class RuntimeObjectSnapshot:
    descriptor: ObjectDescriptorMetadata
    metadata_body: bytes
    current_version: int | None
    last_delta_sequence: int | None
    released: bool


@dataclass(frozen=True, slots=True)
class RuntimeCacheKey:
    namespace: int
    key_hi: int
    key_lo: int
    profile_id: int

    @classmethod
    def from_reference(cls, metadata: CacheReferenceMetadata) -> RuntimeCacheKey:
        return cls(
            metadata.cache_namespace,
            metadata.cache_key_hi,
            metadata.cache_key_lo,
            metadata.profile_id,
        )

    @classmethod
    def from_miss(cls, metadata: CacheMissMetadata) -> RuntimeCacheKey:
        return cls(
            metadata.cache_namespace,
            metadata.cache_key_hi,
            metadata.cache_key_lo,
            metadata.profile_id,
        )


@dataclass(frozen=True, slots=True)
class RuntimeCacheSnapshot:
    metadata: CacheReferenceMetadata
    metadata_body: bytes
    available: bool | None
    miss_reason: int | None
    object_kind: int | None


@dataclass(slots=True)
class _RuntimeObjectEntry:
    descriptor: ObjectDescriptorMetadata
    metadata_body: bytes
    current_version: int | None = None
    last_delta_sequence: int | None = None
    released: bool = False

    def snapshot(self) -> RuntimeObjectSnapshot:
        return RuntimeObjectSnapshot(
            descriptor=self.descriptor,
            metadata_body=self.metadata_body,
            current_version=self.current_version,
            last_delta_sequence=self.last_delta_sequence,
            released=self.released,
        )


@dataclass(slots=True)
class _RuntimeCacheEntry:
    metadata: CacheReferenceMetadata
    metadata_body: bytes
    available: bool | None = None
    miss_reason: int | None = None
    object_kind: int | None = None

    def snapshot(self) -> RuntimeCacheSnapshot:
        return RuntimeCacheSnapshot(
            metadata=self.metadata,
            metadata_body=self.metadata_body,
            available=self.available,
            miss_reason=self.miss_reason,
            object_kind=self.object_kind,
        )


@dataclass(slots=True)
class RuntimeObjectRegistry:
    session_id: int
    _objects: dict[int, _RuntimeObjectEntry] = field(default_factory=dict, init=False, repr=False)
    _references: dict[int, dict[int, RuntimeObjectReference]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _operation_failures: dict[int, RuntimeObjectError] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _resolved_references: set[tuple[int, int]] = field(
        default_factory=set,
        init=False,
        repr=False,
    )
    _cache: dict[RuntimeCacheKey, _RuntimeCacheEntry] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if type(self.session_id) is not int or not 1 <= self.session_id <= 0xFFFF_FFFF:
            raise ValueError("session_id must be a non-zero unsigned 32-bit integer")

    def declare(self, metadata: ObjectDescriptorMetadata, metadata_body: bytes = b"") -> None:
        body = bytes(metadata_body)
        self._require_uint(metadata.object_id, 64, "object_id")
        self._require_registry_value(
            metadata.object_kind,
            "object_kind",
            _RUNTIME_OBJECT_KINDS,
            private_minimum=0x8000,
            maximum=0xFFFF,
            allow_unspecified=False,
        )
        self._require_registry_value(
            metadata.producer_role,
            "producer_role",
            _RUNTIME_ROLES,
            private_minimum=0x80,
            maximum=0xFF,
            allow_unspecified=False,
        )
        self._require_registry_value(
            metadata.consumer_role,
            "consumer_role",
            _RUNTIME_ROLES,
            private_minimum=0x80,
            maximum=0xFF,
            allow_unspecified=False,
        )
        self._require_uint(metadata.session_id, 32, "session_id")
        self._require_uint(metadata.byte_size, 64, "byte_size")
        self._require_uint(metadata.compute_cost_units, 32, "compute_cost_units")
        self._require_registry_value(
            metadata.memory_location_hint,
            "memory_location_hint",
            _MEMORY_LOCATION_HINTS,
            private_minimum=0x8000,
            maximum=0xFFFF,
        )
        self._require_registry_value(
            metadata.ownership_hint,
            "ownership_hint",
            _OWNERSHIP_HINTS,
            private_minimum=0x8000,
            maximum=0xFFFF,
            allow_unspecified=False,
        )
        self._require_uint(metadata.lifetime_hint_ms, 32, "lifetime_hint_ms")
        self._require_uint(metadata.metadata_bytes, 32, "metadata_bytes")
        self._require_declared_length(metadata.metadata_bytes, body, "object descriptor metadata")
        if metadata.object_id == 0:
            raise RuntimeObjectError(0, "object_id must be non-zero")
        if metadata.session_id != self.session_id:
            raise RuntimeObjectError(
                0,
                "object declaration belongs to another session",
                object_id=metadata.object_id,
            )
        existing = self._objects.get(metadata.object_id)
        if existing is not None and not existing.released:
            raise RuntimeObjectError(
                0,
                "object is already declared and active",
                object_id=metadata.object_id,
            )
        self._objects[metadata.object_id] = _RuntimeObjectEntry(metadata, body)

    def bind_version(self, object_id: int, object_version: int) -> None:
        entry = self._require_active_object(object_id)
        if type(object_version) is not int or not 0 <= object_version <= 0xFFFF_FFFF_FFFF_FFFF:
            raise ValueError("object_version must be an unsigned 64-bit integer")
        current = entry.current_version
        if current is not None and object_version < current:
            raise RuntimeObjectError(
                RuntimeObjectFailureCode.VERSION_MISMATCH,
                "object version cannot move backwards",
                object_id=object_id,
            )
        entry.current_version = object_version

    def reference(
        self,
        metadata: ObjectReferenceMetadata,
        metadata_body: bytes = b"",
    ) -> RuntimeObjectReference:
        body = bytes(metadata_body)
        self._require_uint(metadata.object_id, 64, "object_id")
        self._require_uint(metadata.operation_id, 64, "operation_id")
        self._require_uint(metadata.object_version, 64, "object_version")
        self._require_uint(metadata.offset, 64, "offset")
        self._require_uint(metadata.length, 64, "length")
        self._require_mask(metadata.flags, 0x0000_0007, "object_reference.flags")
        self._require_uint(metadata.metadata_bytes, 32, "metadata_bytes")
        self._require_declared_length(metadata.metadata_bytes, body, "object reference metadata")
        if metadata.operation_id == 0:
            raise RuntimeObjectError(
                0,
                "object reference requires an operation_id",
                object_id=metadata.object_id,
            )
        entry = self._require_active_object(metadata.object_id, operation_id=metadata.operation_id)
        region_present = bool(metadata.flags & 0x0000_0004)
        if not region_present and (metadata.offset != 0 or metadata.length != 0):
            raise RuntimeObjectError(
                0,
                "object reference offset and length require the region-present flag",
                object_id=metadata.object_id,
                operation_id=metadata.operation_id,
            )
        if region_present:
            self._require_region(
                entry.descriptor.byte_size,
                metadata.offset,
                metadata.length,
                object_id=metadata.object_id,
                operation_id=metadata.operation_id,
            )
        if entry.current_version is not None and metadata.object_version != entry.current_version:
            raise RuntimeObjectError(
                RuntimeObjectFailureCode.VERSION_MISMATCH,
                "object reference version does not match the available object version",
                object_id=metadata.object_id,
                operation_id=metadata.operation_id,
            )
        reference = RuntimeObjectReference(metadata, body)
        self._references.setdefault(metadata.operation_id, {})[metadata.object_id] = reference
        return reference

    def bind_reference_resolution(
        self,
        operation_id: int,
        object_id: int,
        *,
        current_version: int,
    ) -> None:
        references = self._references.get(operation_id)
        reference = None if references is None else references.get(object_id)
        if reference is None:
            raise RuntimeObjectError(
                RuntimeObjectFailureCode.CACHE_MISS,
                "operation does not reference the object",
                object_id=object_id,
                operation_id=operation_id,
            )
        self.bind_version(object_id, current_version)
        if reference.metadata.object_version != current_version:
            raise RuntimeObjectError(
                RuntimeObjectFailureCode.VERSION_MISMATCH,
                "resolved object version does not match the requested version",
                object_id=object_id,
                operation_id=operation_id,
            )
        self._resolved_references.add((operation_id, object_id))

    def apply_delta(
        self,
        metadata: ObjectDeltaMetadata,
        *,
        metadata_body: bytes = b"",
        delta: bytes = b"",
    ) -> None:
        metadata_bytes = bytes(metadata_body)
        delta_bytes = bytes(delta)
        self._require_uint(metadata.object_id, 64, "object_id")
        self._require_uint(metadata.delta_sequence, 64, "delta_sequence")
        self._require_uint(metadata.region_offset, 64, "region_offset")
        self._require_uint(metadata.region_bytes, 32, "region_bytes")
        self._require_uint(metadata.delta_bytes, 32, "delta_bytes")
        self._require_mask(metadata.flags, 0x0000_0007, "object_delta.flags")
        self._require_uint(metadata.metadata_bytes, 32, "metadata_bytes")
        self._require_declared_length(metadata.metadata_bytes, metadata_bytes, "object delta metadata")
        self._require_declared_length(metadata.delta_bytes, delta_bytes, "object delta payload")
        entry = self._require_active_object(metadata.object_id)
        if entry.last_delta_sequence is not None and metadata.delta_sequence <= entry.last_delta_sequence:
            raise RuntimeObjectError(
                RuntimeObjectFailureCode.VERSION_MISMATCH,
                "object delta sequence must increase monotonically",
                object_id=metadata.object_id,
            )
        self._require_region(
            entry.descriptor.byte_size,
            metadata.region_offset,
            metadata.region_bytes,
            object_id=metadata.object_id,
        )
        entry.last_delta_sequence = metadata.delta_sequence

    def release(self, metadata: ObjectReleaseMetadata, diagnostic: bytes = b"") -> tuple[int, ...]:
        diagnostic_bytes = bytes(diagnostic)
        self._require_declared_length(metadata.diagnostic_bytes, diagnostic_bytes, "object release diagnostic")
        entry = self._require_active_object(metadata.object_id, operation_id=metadata.operation_id)
        final_release = bool(metadata.flags & 0x01)
        invalidates_dependents = bool(metadata.flags & 0x02)
        if not final_release and metadata.operation_id == 0:
            raise RuntimeObjectError(
                0,
                "a non-final object release requires an operation_id",
                object_id=metadata.object_id,
            )

        affected = tuple(
            operation_id
            for operation_id, references in self._references.items()
            if metadata.object_id in references
        )
        if final_release:
            entry.released = True
            for released_references in self._references.values():
                released_references.pop(metadata.object_id, None)
            self._resolved_references = {
                item for item in self._resolved_references if item[1] != metadata.object_id
            }
        else:
            scoped_references = self._references.get(metadata.operation_id)
            if scoped_references is not None:
                scoped_references.pop(metadata.object_id, None)
            self._resolved_references.discard((metadata.operation_id, metadata.object_id))

        if invalidates_dependents:
            for operation_id in affected:
                self._operation_failures.setdefault(
                    operation_id,
                    RuntimeObjectError(
                        RuntimeObjectFailureCode.DEPENDENCY_INVALID,
                        "referenced object was released with dependent invalidation",
                        object_id=metadata.object_id,
                        operation_id=operation_id,
                    ),
                )
            return affected
        return ()

    def record_cache_reference(
        self,
        metadata: CacheReferenceMetadata,
        metadata_body: bytes = b"",
    ) -> RuntimeCacheKey:
        body = bytes(metadata_body)
        self._require_uint(metadata.cache_namespace, 32, "cache_namespace")
        self._require_uint(metadata.cache_key_hi, 64, "cache_key_hi")
        self._require_uint(metadata.cache_key_lo, 64, "cache_key_lo")
        self._require_uint(metadata.profile_id, 16, "profile_id")
        self._require_registry_value(
            metadata.reuse_scope,
            "reuse_scope",
            _CACHE_REUSE_SCOPES,
            private_minimum=0x8000,
            maximum=0xFFFF,
        )
        self._require_uint(metadata.lease_id, 64, "lease_id")
        self._require_uint(metadata.producer_trace_id, 64, "producer_trace_id")
        self._require_uint(metadata.expiration_hint_ms, 32, "expiration_hint_ms")
        self._require_uint(metadata.metadata_bytes, 32, "metadata_bytes")
        self._require_mask(metadata.flags, 0x0000_0003, "cache_reference.flags")
        self._require_declared_length(metadata.metadata_bytes, body, "cache reference metadata")
        if metadata.flags & 0x0000_0001 and metadata.lease_id == 0:
            raise RuntimeObjectError(
                RuntimeObjectFailureCode.CACHE_MISS,
                "cache reference requires a non-zero lease_id",
            )
        key = RuntimeCacheKey.from_reference(metadata)
        self._cache[key] = _RuntimeCacheEntry(metadata, body)
        return key

    def bind_cache_resolution(
        self,
        key: RuntimeCacheKey,
        *,
        available: bool,
        object_kind: int | None = None,
        lease_id: int | None = None,
        lease_live: bool | None = None,
    ) -> None:
        entry = self._cache.get(key)
        if entry is None:
            raise RuntimeObjectError(RuntimeObjectFailureCode.CACHE_MISS, "cache reference is unknown")
        if type(available) is not bool:
            raise ValueError("available must be a boolean")
        if object_kind is not None and (type(object_kind) is not int or not 0 <= object_kind <= 0xFFFF_FFFF):
            raise ValueError("object_kind must be an unsigned 32-bit integer")
        if lease_id is not None:
            self._require_uint(lease_id, 64, "lease_id")
        if lease_live is not None and type(lease_live) is not bool:
            raise ValueError("lease_live must be a boolean or None")
        if available:
            expected_lease_id = entry.metadata.lease_id
            lease_required = bool(entry.metadata.flags & 0x0000_0001)
            if lease_required and (lease_id != expected_lease_id or lease_live is not True):
                raise RuntimeObjectError(
                    RuntimeObjectFailureCode.LEASE_EXPIRED,
                    "cache provider did not confirm the required live lease",
                )
            if lease_id is not None and expected_lease_id != 0 and lease_id != expected_lease_id:
                raise RuntimeObjectError(
                    RuntimeObjectFailureCode.LEASE_EXPIRED,
                    "cache provider resolved a different lease_id",
                )
            if lease_live is False:
                raise RuntimeObjectError(
                    RuntimeObjectFailureCode.LEASE_EXPIRED,
                    "cache provider reported an expired lease",
                )
        entry.available = available
        entry.object_kind = object_kind
        if available:
            entry.miss_reason = None

    def record_cache_miss(self, metadata: CacheMissMetadata, diagnostic: bytes = b"") -> None:
        diagnostic_bytes = bytes(diagnostic)
        self._require_declared_length(metadata.diagnostic_bytes, diagnostic_bytes, "cache miss diagnostic")
        key = RuntimeCacheKey.from_miss(metadata)
        entry = self._cache.get(key)
        if entry is None:
            return
        entry.available = False
        entry.miss_reason = int(metadata.miss_reason)

    def invalidate_cache(self, metadata: CacheInvalidateMetadata) -> tuple[RuntimeCacheKey, ...]:
        invalidated = tuple(key for key, entry in self._cache.items() if self._cache_key_matches(key, entry, metadata))
        for key in invalidated:
            entry = self._cache[key]
            entry.available = False
            entry.miss_reason = None
        return invalidated

    def object_snapshot(self, object_id: int) -> RuntimeObjectSnapshot:
        entry = self._objects.get(object_id)
        if entry is None:
            raise RuntimeObjectError(RuntimeObjectFailureCode.CACHE_MISS, "object is not declared", object_id=object_id)
        return entry.snapshot()

    def cache_snapshot(self, key: RuntimeCacheKey) -> RuntimeCacheSnapshot:
        entry = self._cache.get(key)
        if entry is None:
            raise RuntimeObjectError(RuntimeObjectFailureCode.CACHE_MISS, "cache reference is unknown")
        return entry.snapshot()

    def references_for_operation(self, operation_id: int) -> tuple[RuntimeObjectReference, ...]:
        return tuple(self._references.get(operation_id, {}).values())

    def unresolved_references_for_operation(self, operation_id: int) -> tuple[RuntimeObjectReference, ...]:
        return tuple(
            reference
            for object_id, reference in self._references.get(operation_id, {}).items()
            if (operation_id, object_id) not in self._resolved_references
        )

    def operation_failure(self, operation_id: int) -> RuntimeObjectError | None:
        return self._operation_failures.get(operation_id)

    def record_operation_failure(self, error: RuntimeObjectError) -> None:
        if error.operation_id == 0:
            return
        self._operation_failures.setdefault(error.operation_id, error)

    def retire_operation(self, operation_id: int) -> None:
        self._references.pop(operation_id, None)
        self._operation_failures.pop(operation_id, None)
        self._resolved_references = {
            item for item in self._resolved_references if item[0] != operation_id
        }

    def clear(self) -> None:
        self._objects.clear()
        self._references.clear()
        self._operation_failures.clear()
        self._resolved_references.clear()
        self._cache.clear()

    def _require_active_object(self, object_id: int, *, operation_id: int = 0) -> _RuntimeObjectEntry:
        entry = self._objects.get(object_id)
        if entry is None:
            raise RuntimeObjectError(
                RuntimeObjectFailureCode.CACHE_MISS,
                "object is not declared",
                object_id=object_id,
                operation_id=operation_id,
            )
        if entry.released:
            raise RuntimeObjectError(
                RuntimeObjectFailureCode.DEPENDENCY_INVALID,
                "object has been released",
                object_id=object_id,
                operation_id=operation_id,
            )
        return entry

    @staticmethod
    def _require_declared_length(declared: int, value: bytes, name: str) -> None:
        if declared != len(value):
            raise RuntimeObjectError(0, f"{name} length does not match its declared byte count")

    @staticmethod
    def _require_uint(value: int, bits: int, name: str) -> None:
        if type(value) is bool or not isinstance(value, int) or not 0 <= int(value) < 1 << bits:
            raise RuntimeObjectError(0, f"{name} must be an unsigned {bits}-bit integer")

    @classmethod
    def _require_mask(cls, value: int, valid_mask: int, name: str) -> None:
        cls._require_uint(value, 32, name)
        if value & ~valid_mask:
            raise RuntimeObjectError(0, f"{name} contains reserved bits")

    @staticmethod
    def _require_registry_value(
        value: int,
        name: str,
        standard_values: frozenset[int],
        *,
        private_minimum: int,
        maximum: int,
        allow_unspecified: bool = True,
    ) -> None:
        if type(value) is bool or not isinstance(value, int):
            raise RuntimeObjectError(0, f"{name} must be an integer registry value")
        numeric = int(value)
        if not allow_unspecified and numeric == 0:
            raise RuntimeObjectError(0, f"{name} must not be unspecified")
        if numeric not in standard_values and not private_minimum <= numeric <= maximum:
            raise RuntimeObjectError(0, f"{name} uses a reserved registry value")

    @staticmethod
    def _require_region(
        object_bytes: int,
        offset: int,
        length: int,
        *,
        object_id: int,
        operation_id: int = 0,
    ) -> None:
        if offset > object_bytes or length > object_bytes - offset:
            raise RuntimeObjectError(
                0,
                "object region exceeds the declared object size",
                object_id=object_id,
                operation_id=operation_id,
            )

    @staticmethod
    def _cache_key_matches(
        key: RuntimeCacheKey,
        entry: _RuntimeCacheEntry,
        metadata: CacheInvalidateMetadata,
    ) -> bool:
        scope = int(metadata.invalidate_scope)
        if scope == int(CacheInvalidateScope.WHOLE_SESSION):
            return True
        if key.namespace != int(metadata.cache_namespace):
            return False
        if scope == int(CacheInvalidateScope.NAMESPACE):
            return True
        if scope == int(CacheInvalidateScope.OBJECT_KIND):
            return entry.object_kind == int(metadata.cache_key_hi)
        return key.key_hi == int(metadata.cache_key_hi) and key.key_lo == int(metadata.cache_key_lo)
