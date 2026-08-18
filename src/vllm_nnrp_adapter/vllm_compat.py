from __future__ import annotations

import importlib
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

VLLM_INSTALLATION_RANGE = ">=0.18.0,<0.27"


class VllmCompatibilityError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class VllmCompatibilityBinding:
    name: str
    anchor_version: str
    version_range: str
    request_type_path: str
    serving_method: str
    engine_direct_features: tuple[str, ...]

    def accepts(self, installed_version: Version) -> bool:
        return installed_version in SpecifierSet(self.version_range)


_ENGINE_DIRECT_FEATURES = (
    "render_chat_request",
    "_extract_prompt_components",
    "_extract_prompt_len",
    "engine_client",
)
_CHAT_COMPLETION_REQUEST_PATH = "vllm.entrypoints.openai.chat_completion.protocol:ChatCompletionRequest"

VLLM_COMPATIBILITY_BINDINGS = (
    VllmCompatibilityBinding(
        name="legacy-0.18",
        anchor_version="0.18.1",
        version_range=">=0.18.0,<0.19",
        request_type_path=_CHAT_COMPLETION_REQUEST_PATH,
        serving_method="create_chat_completion",
        engine_direct_features=_ENGINE_DIRECT_FEATURES,
    ),
    VllmCompatibilityBinding(
        name="transition-0.22",
        anchor_version="0.22.1",
        version_range=">=0.22.0,<0.23",
        request_type_path=_CHAT_COMPLETION_REQUEST_PATH,
        serving_method="create_chat_completion",
        engine_direct_features=_ENGINE_DIRECT_FEATURES,
    ),
    VllmCompatibilityBinding(
        name="current-0.26",
        anchor_version="0.26.0",
        version_range=">=0.26.0,<0.27",
        request_type_path=_CHAT_COMPLETION_REQUEST_PATH,
        serving_method="create_chat_completion",
        engine_direct_features=_ENGINE_DIRECT_FEATURES,
    ),
)


def resolve_vllm_compatibility(
    serving_chat: object | None = None,
    *,
    installed_version: str | None = None,
) -> tuple[str, VllmCompatibilityBinding, object]:
    detected_version = installed_version or _installed_vllm_version()
    parsed_version = _parse_version(detected_version)
    if parsed_version not in SpecifierSet(VLLM_INSTALLATION_RANGE):
        raise _compatibility_error(detected_version, f"installation range {VLLM_INSTALLATION_RANGE}")

    binding = next(
        (candidate for candidate in VLLM_COMPATIBILITY_BINDINGS if candidate.accepts(parsed_version)),
        None,
    )
    if binding is None:
        raise _compatibility_error(detected_version, "named compatibility binding for this minor line")

    if serving_chat is not None and not callable(getattr(serving_chat, binding.serving_method, None)):
        raise _compatibility_error(detected_version, f"serving_chat.{binding.serving_method}")

    request_type = _load_request_type(binding, detected_version)
    return detected_version, binding, request_type


def render_vllm_compatibility_table() -> str:
    rows = [
        "# vLLM Compatibility",
        "",
        f"The optional dependency accepts `{VLLM_INSTALLATION_RANGE}` for installation. Runtime support is",
        "limited to the named, feature-probed compatibility bindings below.",
        "",
        "| Binding | Tested anchor | Accepted minor family | Required serving method |",
        "| --- | --- | --- | --- |",
    ]
    rows.extend(
        f"| `{binding.name}` | `{binding.anchor_version}` | `{binding.version_range}` | "
        f"`{binding.serving_method}` |"
        for binding in VLLM_COMPATIBILITY_BINDINGS
    )
    rows.extend(
        (
            "",
            "Versions inside the installation range but outside these minor families are rejected at startup.",
            "Later patches in a listed family must still pass the same request-type and serving-feature probes.",
            "",
        )
    )
    return "\n".join(rows)


def _installed_vllm_version() -> str:
    try:
        return version("vllm")
    except PackageNotFoundError as error:
        raise _compatibility_error("not installed", "installed vLLM distribution") from error


def _parse_version(detected_version: str) -> Version:
    try:
        return Version(detected_version)
    except InvalidVersion as error:
        raise _compatibility_error(detected_version, "valid PEP 440 vLLM version") from error


def _load_request_type(binding: VllmCompatibilityBinding, detected_version: str) -> object:
    module_name, symbol_name = binding.request_type_path.split(":", maxsplit=1)
    try:
        module = importlib.import_module(module_name)
        return getattr(module, symbol_name)
    except (AttributeError, ModuleNotFoundError) as error:
        raise _compatibility_error(detected_version, binding.request_type_path) from error


def _compatibility_error(detected_version: str, missing_feature: str) -> VllmCompatibilityError:
    anchors = ", ".join(binding.anchor_version for binding in VLLM_COMPATIBILITY_BINDINGS)
    return VllmCompatibilityError(
        "vLLM compatibility check failed: "
        f"detected version {detected_version}; missing feature: {missing_feature}; tested anchors: {anchors}"
    )
