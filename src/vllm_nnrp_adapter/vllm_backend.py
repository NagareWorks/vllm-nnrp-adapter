from __future__ import annotations

import importlib
import inspect
import time
import uuid
from collections.abc import AsyncIterator, Mapping
from typing import Any, cast

CHAT_METHOD_CANDIDATES = (
    "create_chat_completion",
    "create_chat_completion_raw",
    "chat_completion",
)


class VllmBackend:
    supports_tool_calls = False

    def __init__(
        self,
        serving_chat: object,
        *,
        request_factory: object | None = None,
        prefer_engine_direct: bool = True,
    ) -> None:
        self._serving_chat = serving_chat
        self._chat_method_name = _resolve_chat_method(serving_chat)
        self._request_factory = request_factory
        self._prefer_engine_direct = prefer_engine_direct

    @property
    def compatibility_binding(self) -> str | None:
        value = getattr(self._request_factory, "compatibility_binding", None)
        return value if isinstance(value, str) else None

    @property
    def vllm_version(self) -> str | None:
        value = getattr(self._request_factory, "vllm_version", None)
        return value if isinstance(value, str) else None

    async def create_chat_completion(self, body: Mapping[str, Any]) -> Any:
        request = self._build_request(body)
        if body.get("stream", False):
            if not self._prefer_engine_direct:
                raise VllmProductionBoundaryError("production streaming requires the engine-direct backend")
            if not _supports_engine_direct(self._serving_chat, request, body):
                raise VllmProductionBoundaryError(
                    "request features or serving-object shape are not supported by the engine-direct backend"
                )
            try:
                return await _create_engine_direct_stream(self._serving_chat, request, body)
            except EngineDirectUnsupported as error:
                raise VllmProductionBoundaryError("vLLM rejected the engine-direct request shape") from error

        method = getattr(self._serving_chat, self._chat_method_name)
        result = _call_chat_method(method, request)
        if inspect.isawaitable(result):
            result = await result
        return _normalize_chat_result(result)

    def _build_request(self, body: Mapping[str, Any]) -> object:
        if self._request_factory is None:
            return dict(body)
        if not callable(self._request_factory):
            raise TypeError("request_factory must be callable")
        return self._request_factory(dict(body))


def _resolve_chat_method(serving_chat: object) -> str:
    for name in CHAT_METHOD_CANDIDATES:
        candidate = getattr(serving_chat, name, None)
        if callable(candidate):
            return name

    joined = ", ".join(CHAT_METHOD_CANDIDATES)
    raise TypeError(f"vLLM serving object does not expose a supported chat completion method: {joined}.")


def _call_chat_method(method: object, request: object) -> object:
    if not callable(method):
        raise TypeError("resolved vLLM chat completion method is not callable")

    try:
        return method(request)
    except TypeError as positional_error:
        try:
            return method(request, raw_request=None)
        except TypeError:
            try:
                return method(request=request, raw_request=None)
            except TypeError:
                raise positional_error from None


def _normalize_chat_result(result: object) -> object:
    if _is_async_iterator(result):
        raise VllmProductionBoundaryError("production vLLM backend received an HTTP/SSE-shaped stream")
    if isinstance(result, str):
        raise VllmProductionBoundaryError("production vLLM backend received an HTTP/SSE string")
    return _normalize_object(result)


def _normalize_object(value: object) -> object:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        if isinstance(dumped, Mapping):
            return dumped
    dict_method = getattr(value, "dict", None)
    if callable(dict_method):
        dumped = dict_method()
        if isinstance(dumped, Mapping):
            return dumped
    return value


def _is_async_iterator(value: object) -> bool:
    return hasattr(value, "__aiter__") and hasattr(value, "__anext__")


def _supports_engine_direct(serving_chat: object, request: object, body: Mapping[str, Any]) -> bool:
    if not body.get("stream", False):
        return False
    if body.get("tools") or body.get("tool_choice"):
        return False
    if body.get("logprobs") or body.get("top_logprobs") or body.get("echo") or body.get("return_token_ids"):
        return False
    if getattr(serving_chat, "reasoning_parser_cls", None) is not None:
        return False
    if getattr(serving_chat, "use_harmony", False):
        return False
    if bool(_getattr_default(request, "use_beam_search", False)):
        return False
    return all(
        callable(getattr(serving_chat, name, None))
        for name in ("render_chat_request", "_extract_prompt_components", "_extract_prompt_len")
    ) and hasattr(serving_chat, "engine_client")


async def _create_engine_direct_stream(
    serving_chat: object,
    request: object,
    body: Mapping[str, Any],
) -> AsyncIterator[Mapping[str, Any]]:
    render_chat_request = _required_callable(serving_chat, "render_chat_request")
    rendered = render_chat_request(request)
    if inspect.isawaitable(rendered):
        rendered = await rendered
    if _looks_like_error_response(rendered):
        raise EngineDirectUnsupported
    if not isinstance(rendered, tuple) or len(rendered) != 2:
        raise EngineDirectUnsupported

    _conversation, engine_prompts = rendered
    if not isinstance(engine_prompts, list) or len(engine_prompts) != 1:
        raise EngineDirectUnsupported
    engine_prompt = engine_prompts[0]

    request_id = _direct_request_id(request)
    lora_request = _call_optional(serving_chat, "_maybe_get_adapters", request, supports_default_mm_loras=True)
    model_name = _model_name(serving_chat, lora_request, body)
    sampling_params = _sampling_params(serving_chat, request, engine_prompt)
    trace_headers = None
    data_parallel_rank = _call_optional(serving_chat, "_get_data_parallel_rank", None)

    _call_optional(
        serving_chat,
        "_log_inputs",
        request_id,
        engine_prompt,
        params=sampling_params,
        lora_request=lora_request,
    )

    engine_client = cast(Any, _required_attr(serving_chat, "engine_client"))
    generator = engine_client.generate(
        engine_prompt,
        sampling_params,
        request_id,
        lora_request=lora_request,
        trace_headers=trace_headers,
        priority=_optional_int(_getattr_default(request, "priority", 0), default=0),
        data_parallel_rank=data_parallel_rank,
        reasoning_ended=None,
    )
    return EngineDirectChatStream(
        generator,
        engine_client=engine_client,
        request_id=request_id,
        model_name=model_name,
        include_usage=_include_usage(body),
    )


class EngineDirectChatStream:
    def __init__(
        self,
        generator: AsyncIterator[object],
        *,
        engine_client: object,
        request_id: str,
        model_name: str,
        include_usage: bool,
    ) -> None:
        self._generator = generator
        self._engine_client = engine_client
        self._request_id = request_id
        self._model_name = model_name
        self._include_usage = include_usage
        self._created = int(time.time())
        self._completion_tokens = 0
        self._prompt_tokens = 0
        self._final_usage_pending = False
        self._closed = False

    def __aiter__(self) -> EngineDirectChatStream:
        return self

    async def __anext__(self) -> Mapping[str, Any]:
        if self._final_usage_pending:
            self._final_usage_pending = False
            raise StopAsyncIteration

        result = await self._generator.__anext__()
        chunk = self._chunk_from_request_output(result)
        if _request_output_finished(result) and self._include_usage:
            self._final_usage_pending = True
        return chunk

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await _abort_request(self._engine_client, self._request_id)
        closer = getattr(self._generator, "aclose", None)
        if callable(closer):
            closed = closer()
            if inspect.isawaitable(closed):
                await closed

    def _chunk_from_request_output(self, result: object) -> Mapping[str, Any]:
        prompt_token_ids = _getattr_default(result, "prompt_token_ids", None)
        if prompt_token_ids is not None:
            self._prompt_tokens = len(cast(list[object], prompt_token_ids))
        encoder_prompt_token_ids = _getattr_default(result, "encoder_prompt_token_ids", None)
        if encoder_prompt_token_ids is not None:
            self._prompt_tokens += len(cast(list[object], encoder_prompt_token_ids))

        choices = []
        for output in cast(list[object], _getattr_default(result, "outputs", [])):
            token_ids = list(cast(list[object], _getattr_default(output, "token_ids", [])))
            self._completion_tokens += len(token_ids)
            delta_text = _getattr_default(output, "text", "")
            choice: dict[str, Any] = {
                "index": _optional_int(_getattr_default(output, "index", 0), default=0),
                "delta": {"content": delta_text} if isinstance(delta_text, str) and delta_text else {},
                "finish_reason": _getattr_default(output, "finish_reason", None),
            }
            stop_reason = _getattr_default(output, "stop_reason", None)
            if stop_reason is not None:
                choice["stop_reason"] = stop_reason
            choices.append(choice)

        chunk: dict[str, Any] = {
            "id": self._request_id,
            "object": "chat.completion.chunk",
            "created": self._created,
            "choices": choices,
            "model": self._model_name,
        }
        if self._include_usage and _request_output_finished(result):
            chunk["usage"] = {
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
                "total_tokens": self._prompt_tokens + self._completion_tokens,
            }
        return chunk


class EngineDirectUnsupported(Exception):
    pass


class VllmProductionBoundaryError(RuntimeError):
    pass


def _sampling_params(serving_chat: object, request: object, engine_prompt: object) -> object:
    try:
        utils = importlib.import_module("vllm.entrypoints.utils")
    except ModuleNotFoundError as error:
        raise EngineDirectUnsupported from error

    get_max_tokens = _required_callable(utils, "get_max_tokens")
    model_config = _required_attr(serving_chat, "model_config")
    default_sampling_params = _getattr_default(serving_chat, "default_sampling_params", {})
    max_model_len = _required_attr(model_config, "max_model_len")
    max_completion_tokens = _getattr_default(request, "max_completion_tokens", None)
    max_tokens = _getattr_default(request, "max_tokens", None)
    requested_max_tokens = max_completion_tokens if max_completion_tokens is not None else max_tokens
    prompt_len = _required_callable(serving_chat, "_extract_prompt_len")(engine_prompt)
    override_max_tokens = _getattr_default(serving_chat, "override_max_tokens", None)
    resolved_max_tokens = get_max_tokens(
        max_model_len,
        requested_max_tokens,
        prompt_len,
        default_sampling_params,
        override_max_tokens,
    )
    to_sampling_params = getattr(request, "to_sampling_params", None)
    if not callable(to_sampling_params):
        raise EngineDirectUnsupported
    return to_sampling_params(resolved_max_tokens, default_sampling_params)


def _direct_request_id(request: object) -> str:
    configured = _getattr_default(request, "request_id", None)
    suffix = configured if isinstance(configured, str) and configured else uuid.uuid4().hex
    return f"chatcmpl-nnrp-{suffix}"


def _model_name(serving_chat: object, lora_request: object, body: Mapping[str, Any]) -> str:
    models = _getattr_default(serving_chat, "models", None)
    model_name = getattr(models, "model_name", None)
    if callable(model_name):
        value = model_name(lora_request)
        if isinstance(value, str) and value:
            return value
    value = body.get("model")
    return value if isinstance(value, str) and value else "unknown"


def _include_usage(body: Mapping[str, Any]) -> bool:
    stream_options = body.get("stream_options")
    return isinstance(stream_options, Mapping) and stream_options.get("include_usage") is True


def _request_output_finished(result: object) -> bool:
    return bool(_getattr_default(result, "finished", False))


def _looks_like_error_response(value: object) -> bool:
    return hasattr(value, "error") or type(value).__name__ == "ErrorResponse"


def _call_optional(owner: object, name: str, *args: object, **kwargs: object) -> object | None:
    method = getattr(owner, name, None)
    if not callable(method):
        return None
    return cast(object | None, method(*args, **kwargs))


async def _abort_request(engine_client: object, request_id: str) -> None:
    abort = getattr(engine_client, "abort", None)
    if not callable(abort):
        return
    result = abort(request_id)
    if inspect.isawaitable(result):
        await result


def _getattr_default(value: object, name: str, default: object) -> object:
    return getattr(value, name, default)


def _required_attr(value: object, name: str) -> object:
    if not hasattr(value, name):
        raise EngineDirectUnsupported
    return getattr(value, name)


def _required_callable(value: object, name: str) -> Any:
    attr = _required_attr(value, name)
    if not callable(attr):
        raise EngineDirectUnsupported
    return attr


def _optional_int(value: object, *, default: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default
