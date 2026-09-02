from __future__ import annotations

import math
import urllib.parse
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True)
class OpenAiHttpSseDriverConfig:
    endpoint: str
    api_key: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: float = 300.0
    sample_id_header: str = "X-NNRP-Benchmark-Sample-Id"

    def __post_init__(self) -> None:
        parsed = urllib.parse.urlsplit(self.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("endpoint must be an absolute http:// or https:// URL")
        if self.api_key is not None and not self.api_key:
            raise ValueError("api_key must be non-empty when provided")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        _validate_header(self.sample_id_header, "sample_id_header")
        copied_headers: dict[str, str] = {}
        for name, value in self.headers.items():
            _validate_header(name, "headers name")
            if not isinstance(value, str) or not value or "\r" in value or "\n" in value:
                raise ValueError("headers values must be non-empty single-line strings")
            copied_headers[name] = value
        object.__setattr__(self, "headers", MappingProxyType(copied_headers))


def _validate_header(value: object, location: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or ":" in value
        or "\r" in value
        or "\n" in value
        or value != value.strip()
    ):
        raise ValueError(f"{location} must be a non-empty HTTP header name")
