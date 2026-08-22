from __future__ import annotations

import importlib
import inspect
import time
import uuid
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from .vllm_compat import VllmEngineDirectBinding

CHAT_METHOD_CANDIDATES = (
    "create_chat_completion",
    "create_chat_completion_raw",
    "chat_completion",
)


class VllmBackend:
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

    @property
    def supports_tool_calls(self) -> bool:
        binding = _engine_direct_binding(self._request_factory)
        return binding is not None and _serving_supports_tool_calls(self._serving_chat, binding)

    @property
    def supports_runtime_priority(self) -> bool:
        return callable(self._request_factory)

    @property
    def supports_live_runtime_priority(self) -> bool:
        binding = _engine_direct_binding(self._request_factory)
        engine_client = _getattr_default(self._serving_chat, "engine_client", None)
        return (
            binding is not None
            and binding.live_priority_method is not None
            and callable(getattr(engine_client, binding.live_priority_method, None))
        )

    async def update_runtime_priority(self, request_id: str, priority: int) -> bool:
        binding = _engine_direct_binding(self._request_factory)
        method_name = None if binding is None else binding.live_priority_method
        if method_name is None:
            return False
        engine_client = _getattr_default(self._serving_chat, "engine_client", None)
        method = getattr(engine_client, method_name, None)
        if not callable(method):
            return False
        result = method(request_id, priority)
        if inspect.isawaitable(result):
            result = await result
        return result if isinstance(result, bool) else True

    def benchmark_metadata(self) -> Mapping[str, object]:
        model_config = _getattr_default(self._serving_chat, "model_config", None)
        engine_configuration = {
            name: _metadata_value(_getattr_default(model_config, name, None))
            for name in ("max_model_len", "dtype", "quantization", "task")
            if _getattr_default(model_config, name, None) is not None
        }
        return {
            "vllm_version": self.vllm_version or "unknown",
            "compatibility_binding": self.compatibility_binding or "unknown",
            "engine_configuration": engine_configuration or {"status": "unknown"},
        }

    async def create_chat_completion(self, body: Mapping[str, Any]) -> Any:
        return await self._create_chat_completion(body, trace_headers=None, priority=None)

    async def create_chat_completion_with_context(
        self,
        body: Mapping[str, Any],
        *,
        trace_headers: Mapping[str, str] | None = None,
        priority: int | None = None,
    ) -> Any:
        return await self._create_chat_completion(
            body,
            trace_headers=trace_headers,
            priority=priority,
        )

    async def _create_chat_completion(
        self,
        body: Mapping[str, Any],
        *,
        trace_headers: Mapping[str, str] | None,
        priority: int | None,
    ) -> Any:
        request = self._build_request(body)
        if priority is not None:
            if not callable(self._request_factory):
                raise VllmProductionBoundaryError(
                    "runtime priority requires a versioned vLLM request binding"
                )
            try:
                _set_compat_attribute(request, "priority", priority)
            except (AttributeError, TypeError, ValueError) as error:
                raise VllmProductionBoundaryError(
                    "the selected vLLM request binding rejected runtime priority"
                ) from error
        if body.get("stream", False):
            if not self._prefer_engine_direct:
                raise VllmProductionBoundaryError("production streaming requires the engine-direct backend")
            engine_direct_binding = _engine_direct_binding(self._request_factory)
            if engine_direct_binding is None or not _supports_engine_direct(
                self._serving_chat,
                request,
                body,
                engine_direct_binding,
            ):
                raise VllmProductionBoundaryError(
                    "request features or serving-object shape are not supported by the engine-direct backend"
                )
            try:
                return await _create_engine_direct_stream(
                    self._serving_chat,
                    request,
                    body,
                    engine_direct_binding,
                    trace_headers=trace_headers,
                )
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
    normalized = _normalize_object(result)
    _raise_for_backend_error(normalized)
    return normalized


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


def _supports_engine_direct(
    serving_chat: object,
    request: object,
    body: Mapping[str, Any],
    binding: VllmEngineDirectBinding,
) -> bool:
    if not body.get("stream", False):
        return False
    if body.get("logprobs") or body.get("top_logprobs") or body.get("echo") or body.get("return_token_ids"):
        return False
    if getattr(serving_chat, "use_harmony", False):
        return False
    if bool(_getattr_default(request, "use_beam_search", False)):
        return False
    if body.get("tools") and not _tool_request_supported(serving_chat, request, binding):
        return False
    return all(
        callable(getattr(serving_chat, name, None))
        for name in binding.required_serving_features
    ) and hasattr(serving_chat, "engine_client")


async def _create_engine_direct_stream(
    serving_chat: object,
    request: object,
    body: Mapping[str, Any],
    binding: VllmEngineDirectBinding,
    *,
    trace_headers: Mapping[str, str] | None,
) -> AsyncIterator[Mapping[str, Any]]:
    render_chat_request = _required_callable(serving_chat, "render_chat_request")
    rendered = render_chat_request(request)
    if inspect.isawaitable(rendered):
        rendered = await rendered
    if _looks_like_error_response(rendered):
        _raise_for_backend_error(_normalize_object(rendered))
        raise EngineDirectUnsupported
    if not isinstance(rendered, tuple) or len(rendered) != 2:
        raise EngineDirectUnsupported

    conversation, engine_prompts = rendered
    if not isinstance(engine_prompts, list) or len(engine_prompts) != 1:
        raise EngineDirectUnsupported
    engine_prompt = engine_prompts[0]

    delta_parser = _create_engine_direct_delta_parser(
        serving_chat,
        request,
        binding,
        conversation=conversation,
    )

    request_id = _direct_request_id(request)
    lora_request = _call_optional(serving_chat, "_maybe_get_adapters", request, supports_default_mm_loras=True)
    model_name = _model_name(serving_chat, lora_request, body)
    sampling_params = _sampling_params(serving_chat, request, engine_prompt, binding)
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
    generate_kwargs: dict[str, object] = {
        "lora_request": lora_request,
        "trace_headers": trace_headers,
        "priority": _optional_int(_getattr_default(request, "priority", 0), default=0),
        "data_parallel_rank": data_parallel_rank,
        "reasoning_ended": _reasoning_ended(request, binding),
    }
    if binding.supports_reasoning_parser_kwargs:
        generate_kwargs["reasoning_parser_kwargs"] = None
    generator = engine_client.generate(engine_prompt, sampling_params, request_id, **generate_kwargs)
    return EngineDirectChatStream(
        generator,
        engine_client=engine_client,
        request_id=request_id,
        model_name=model_name,
        include_usage=_include_usage(body),
        delta_parser=delta_parser,
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
        delta_parser: EngineDirectDeltaParser | None = None,
    ) -> None:
        self._generator = generator
        self._engine_client = engine_client
        self._request_id = request_id
        self._model_name = model_name
        self._include_usage = include_usage
        self._delta_parser = delta_parser
        self._created = int(time.time())
        self._completion_tokens = 0
        self._prompt_tokens = 0
        self._final_usage_pending = False
        self._closed = False
        self._abort_accepted: bool | None = None

    def __aiter__(self) -> EngineDirectChatStream:
        return self

    async def __anext__(self) -> Mapping[str, Any]:
        if self._closed:
            raise StopAsyncIteration
        if self._final_usage_pending:
            self._final_usage_pending = False
            return self._usage_chunk()

        while True:
            result = await self._generator.__anext__()
            chunk = self._chunk_from_request_output(result)
            if _request_output_finished(result) and self._include_usage:
                self._final_usage_pending = True
            if _chunk_has_output(chunk) or self._final_usage_pending:
                return chunk

    async def aclose(self) -> bool:
        if self._closed:
            return self._abort_accepted is True
        self._closed = True
        self._abort_accepted = await _abort_request(self._engine_client, self._request_id)
        closer = getattr(self._generator, "aclose", None)
        if callable(closer):
            closed = closer()
            if inspect.isawaitable(closed):
                await closed
        return self._abort_accepted

    def _chunk_from_request_output(self, result: object) -> Mapping[str, Any]:
        observed_prompt_tokens = False
        prompt_tokens = 0
        prompt_token_ids = _getattr_default(result, "prompt_token_ids", None)
        if prompt_token_ids is not None:
            observed_prompt_tokens = True
            prompt_tokens += len(cast(list[object], prompt_token_ids))
        encoder_prompt_token_ids = _getattr_default(result, "encoder_prompt_token_ids", None)
        if encoder_prompt_token_ids is not None:
            observed_prompt_tokens = True
            prompt_tokens += len(cast(list[object], encoder_prompt_token_ids))
        if observed_prompt_tokens:
            self._prompt_tokens = prompt_tokens

        choices = []
        for output in cast(list[object], _getattr_default(result, "outputs", [])):
            token_ids = list(cast(list[object], _getattr_default(output, "token_ids", [])))
            self._completion_tokens += len(token_ids)
            delta_text = _getattr_default(output, "text", "")
            delta = (
                self._delta_parser.parse(output, prompt_token_ids=prompt_token_ids)
                if self._delta_parser is not None
                else {"content": delta_text}
                if isinstance(delta_text, str) and delta_text
                else {}
            )
            finish_reason = _getattr_default(output, "finish_reason", None)
            if self._delta_parser is not None:
                finish_reason = self._delta_parser.finish_reason(
                    _optional_int(_getattr_default(output, "index", 0), default=0),
                    finish_reason,
                )
            choice: dict[str, Any] = {
                "index": _optional_int(_getattr_default(output, "index", 0), default=0),
                "delta": delta,
                "finish_reason": finish_reason,
            }
            stop_reason = _getattr_default(output, "stop_reason", None)
            if stop_reason is not None:
                choice["stop_reason"] = stop_reason
            choices.append(choice)

        return {
            "id": self._request_id,
            "object": "chat.completion.chunk",
            "created": self._created,
            "choices": choices,
            "model": self._model_name,
        }

    def _usage_chunk(self) -> Mapping[str, Any]:
        return {
            "id": self._request_id,
            "object": "chat.completion.chunk",
            "created": self._created,
            "choices": [],
            "model": self._model_name,
            "usage": {
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
                "total_tokens": self._prompt_tokens + self._completion_tokens,
            },
        }


@dataclass
class _LegacyToolState:
    previous_text: str = ""
    previous_token_ids: list[object] = field(default_factory=list)
    function_name_returned: bool = False
    history_tool_call_count: int = 0
    call_id: str | None = None
    parser: object | None = None


class EngineDirectDeltaParser:
    def __init__(
        self,
        serving_chat: object,
        request: object,
        binding: VllmEngineDirectBinding,
        *,
        tokenizer: object,
        chat_template_kwargs: Mapping[str, Any] | None,
        history_tool_call_count: int,
    ) -> None:
        self._serving_chat = serving_chat
        self._request = request
        self._binding = binding
        self._tokenizer = tokenizer
        self._chat_template_kwargs = chat_template_kwargs
        self._history_tool_call_count = history_tool_call_count
        self._legacy_states: dict[int, _LegacyToolState] = {}
        self._parsers: dict[int, object] = {}
        self._tools_streamed: set[int] = set()
        self._named_tool = _named_tool_choice(request)
        self._legacy_mode = _legacy_tool_mode(serving_chat, request) if binding.parser_family == "legacy" else None

        if binding.parser_family != "legacy":
            for choice_index in range(_choice_count(request)):
                self._parsers[choice_index] = self._new_unified_parser()

    def parse(self, output: object, *, prompt_token_ids: object) -> dict[str, Any]:
        choice_index = _optional_int(_getattr_default(output, "index", 0), default=0)
        if self._binding.parser_family == "legacy":
            delta = self._parse_legacy(choice_index, output)
        else:
            parser = self._parsers.get(choice_index)
            if parser is None:
                parser = self._new_unified_parser()
                self._parsers[choice_index] = parser
            parse_delta = _required_callable(parser, "parse_delta")
            parse_kwargs: dict[str, object] = {
                "delta_text": _text_delta(output),
                "delta_token_ids": _token_ids(output),
                "request": self._request,
                "prompt_token_ids": prompt_token_ids,
            }
            if self._binding.parser_accepts_finished:
                parse_kwargs["finished"] = _getattr_default(output, "finish_reason", None) is not None
            delta = _normalize_delta(parse_delta(**parse_kwargs))

        if _delta_has_tool_calls(delta):
            self._tools_streamed.add(choice_index)
        return delta

    def finish_reason(self, choice_index: int, finish_reason: object) -> object:
        if finish_reason is None or choice_index not in self._tools_streamed:
            return finish_reason
        return "stop" if self._named_tool is not None else "tool_calls"

    def _new_unified_parser(self) -> object:
        parser_factory = _required_callable(self._serving_chat, self._binding.parser_attribute)
        parser_kwargs: dict[str, object] = {"chat_template_kwargs": self._chat_template_kwargs}
        if self._binding.parser_accepts_model_config:
            parser_kwargs["model_config"] = _required_attr(self._serving_chat, "model_config")
        parser = parser_factory(
            self._tokenizer,
            _getattr_default(self._request, "tools", None),
            **parser_kwargs,
        )
        if parser is None:
            raise EngineDirectUnsupported
        if self._binding.parser_family == "unified-0.22":
            stream_state = _getattr_default(parser, "_stream_state", None)
            if stream_state is not None:
                _set_compat_attribute(
                    stream_state,
                    "tool_call_id_type",
                    _getattr_default(self._serving_chat, "tool_call_id_type", "random"),
                )
                _set_compat_attribute(stream_state, "history_tool_call_cnt", self._history_tool_call_count)
        return parser

    def _parse_legacy(self, choice_index: int, output: object) -> dict[str, Any]:
        state = self._legacy_states.get(choice_index)
        if state is None:
            state = _LegacyToolState(history_tool_call_count=self._history_tool_call_count)
            self._legacy_states[choice_index] = state

        delta_text = _text_delta(output)
        delta_token_ids = _token_ids(output)
        current_text = state.previous_text + delta_text
        current_token_ids = state.previous_token_ids + delta_token_ids

        if self._legacy_mode == "auto":
            if state.parser is None:
                parser_factory = _required_callable(self._serving_chat, self._binding.parser_attribute)
                state.parser = parser_factory(self._tokenizer)
            parser = _required_callable(state.parser, "extract_tool_calls_streaming")
            parsed = parser(
                previous_text=state.previous_text,
                current_text=current_text,
                delta_text=delta_text,
                previous_token_ids=state.previous_token_ids,
                current_token_ids=current_token_ids,
                delta_token_ids=delta_token_ids,
                request=self._request,
            )
            delta = _normalize_delta(parsed)
        elif self._legacy_mode == "named":
            if state.call_id is None:
                state.call_id = f"call_{uuid.uuid4().hex}"
            function: dict[str, Any] = {"arguments": delta_text}
            tool_call: dict[str, Any] = {"index": 0, "function": function}
            if not state.function_name_returned:
                tool_call.update({"id": state.call_id, "type": "function"})
                function["name"] = cast(str, self._named_tool)
                state.function_name_returned = True
            delta = {"tool_calls": [tool_call]}
        elif self._legacy_mode == "required":
            parsed = _required_callable(self._serving_chat, "extract_tool_call_required_streaming")(
                previous_text=state.previous_text,
                current_text=current_text,
                delta_text=delta_text,
                function_name_returned=state.function_name_returned,
                tool_call_idx=state.history_tool_call_count,
            )
            if not isinstance(parsed, tuple) or len(parsed) != 2:
                raise EngineDirectUnsupported
            delta = _normalize_delta(parsed[0])
            state.function_name_returned = bool(parsed[1])
            if _delta_starts_tool_call(delta):
                state.history_tool_call_count += 1
        else:
            delta = {"content": delta_text} if delta_text else {}

        state.previous_text = current_text
        state.previous_token_ids = current_token_ids
        return delta


class EngineDirectUnsupported(Exception):
    pass


class VllmProductionBoundaryError(RuntimeError):
    pass


class _VllmBackendResponseError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        error_type: str | None,
        status_code: int | None,
        parameter: str | None,
    ) -> None:
        super().__init__(message)
        self.vllm_error_type = error_type
        self.vllm_status_code = status_code
        self.vllm_parameter = parameter


def _sampling_params(
    serving_chat: object,
    request: object,
    engine_prompt: object,
    binding: VllmEngineDirectBinding,
) -> object:
    try:
        module_name, symbol_name = binding.get_max_tokens_path.split(":", maxsplit=1)
        utils = importlib.import_module(module_name)
        get_max_tokens = _required_callable(utils, symbol_name)
    except (ModuleNotFoundError, ValueError) as error:
        raise EngineDirectUnsupported from error
    model_config = _required_attr(serving_chat, "model_config")
    default_sampling_params = _getattr_default(serving_chat, "default_sampling_params", {})
    max_model_len = _required_attr(model_config, "max_model_len")
    max_completion_tokens = _getattr_default(request, "max_completion_tokens", None)
    max_tokens = _getattr_default(request, "max_tokens", None)
    requested_max_tokens = max_completion_tokens if max_completion_tokens is not None else max_tokens
    prompt_len = _required_callable(serving_chat, "_extract_prompt_len")(engine_prompt)
    override_max_tokens = _getattr_default(serving_chat, "override_max_tokens", None)
    max_token_args = (
        max_model_len,
        requested_max_tokens,
        prompt_len,
        default_sampling_params,
        override_max_tokens,
    )
    if binding.supports_truncate_prompt_tokens:
        resolved_max_tokens = get_max_tokens(
            *max_token_args,
            truncate_prompt_tokens=_getattr_default(request, "truncate_prompt_tokens", None),
        )
    else:
        resolved_max_tokens = get_max_tokens(*max_token_args)
    to_sampling_params = getattr(request, "to_sampling_params", None)
    if not callable(to_sampling_params):
        raise EngineDirectUnsupported
    return to_sampling_params(resolved_max_tokens, default_sampling_params)


def _create_engine_direct_delta_parser(
    serving_chat: object,
    request: object,
    binding: VllmEngineDirectBinding,
    *,
    conversation: object,
) -> EngineDirectDeltaParser | None:
    has_tools = bool(_getattr_default(request, "tools", None))
    parser_factory = getattr(serving_chat, binding.parser_attribute, None)
    if binding.parser_family == "legacy":
        if not has_tools:
            if getattr(serving_chat, "reasoning_parser_cls", None) is not None:
                raise EngineDirectUnsupported
            return None
        if _legacy_tool_mode(serving_chat, request) is None:
            raise EngineDirectUnsupported
    elif not callable(parser_factory):
        if has_tools:
            raise EngineDirectUnsupported
        return None

    renderer = _required_attr(serving_chat, "renderer")
    tokenizer = _required_attr(renderer, "tokenizer")
    if tokenizer is None:
        raise EngineDirectUnsupported
    chat_template_kwargs = _chat_template_kwargs(serving_chat, request, binding)
    return EngineDirectDeltaParser(
        serving_chat,
        request,
        binding,
        tokenizer=tokenizer,
        chat_template_kwargs=chat_template_kwargs,
        history_tool_call_count=_history_tool_call_count(conversation),
    )


def _chat_template_kwargs(
    serving_chat: object,
    request: object,
    binding: VllmEngineDirectBinding,
) -> Mapping[str, Any] | None:
    method_name = binding.chat_template_kwargs_method
    if method_name is None:
        return None
    value = _required_callable(serving_chat, method_name)(request)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise EngineDirectUnsupported
    return value


def _serving_supports_tool_calls(serving_chat: object, binding: VllmEngineDirectBinding) -> bool:
    return bool(_getattr_default(serving_chat, "enable_auto_tools", False)) and callable(
        getattr(serving_chat, binding.parser_attribute, None)
    )


def _tool_request_supported(
    serving_chat: object,
    request: object,
    binding: VllmEngineDirectBinding,
) -> bool:
    renderer = _getattr_default(serving_chat, "renderer", None)
    if renderer is None or _getattr_default(renderer, "tokenizer", None) is None:
        return False
    if binding.parser_family != "legacy":
        method_name = binding.chat_template_kwargs_method
        parser_available = callable(getattr(serving_chat, binding.parser_attribute, None)) and (
            method_name is None or callable(getattr(serving_chat, method_name, None))
        )
        tool_choice = _getattr_default(request, "tool_choice", None)
        if tool_choice in (None, "auto"):
            return parser_available and bool(_getattr_default(serving_chat, "enable_auto_tools", False))
        return parser_available
    return _legacy_tool_mode(serving_chat, request) is not None


def _legacy_tool_mode(serving_chat: object, request: object) -> str | None:
    if not _getattr_default(request, "tools", None):
        return None
    if _named_tool_choice(request) is not None:
        return "named"
    tool_choice = _getattr_default(request, "tool_choice", None)
    if tool_choice == "required":
        return "required" if callable(getattr(serving_chat, "extract_tool_call_required_streaming", None)) else None
    if tool_choice in (None, "auto"):
        should_parse = getattr(serving_chat, "_should_stream_with_auto_tool_parsing", None)
        if callable(should_parse):
            try:
                if not bool(should_parse(request)):
                    return None
            except Exception:
                return None
        elif not bool(_getattr_default(serving_chat, "enable_auto_tools", False)):
            return None
        return "auto" if callable(getattr(serving_chat, "tool_parser", None)) else None
    return None


def _named_tool_choice(request: object) -> str | None:
    tool_choice = _getattr_default(request, "tool_choice", None)
    if isinstance(tool_choice, Mapping):
        function = tool_choice.get("function")
        name = function.get("name") if isinstance(function, Mapping) else None
    else:
        function = _getattr_default(tool_choice, "function", None)
        name = _getattr_default(function, "name", None)
    return name if isinstance(name, str) and name else None


def _choice_count(request: object) -> int:
    count = _optional_int(_getattr_default(request, "n", 1), default=1)
    return count if count > 0 else 1


def _history_tool_call_count(conversation: object) -> int:
    if not isinstance(conversation, list):
        return 0
    count = 0
    for message in conversation:
        if isinstance(message, Mapping):
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                count += len(tool_calls)
    return count


def _text_delta(output: object) -> str:
    value = _getattr_default(output, "text", "")
    return value if isinstance(value, str) else ""


def _token_ids(output: object) -> list[object]:
    value = _getattr_default(output, "token_ids", [])
    return list(value) if isinstance(value, (list, tuple)) else []


def _normalize_delta(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    normalized = _normalize_object(value)
    if not isinstance(normalized, Mapping):
        raise EngineDirectUnsupported
    return {str(key): item for key, item in normalized.items() if item is not None}


def _delta_has_tool_calls(delta: Mapping[str, Any]) -> bool:
    tool_calls = delta.get("tool_calls")
    return isinstance(tool_calls, list) and bool(tool_calls)


def _delta_starts_tool_call(delta: Mapping[str, Any]) -> bool:
    tool_calls = delta.get("tool_calls")
    if not isinstance(tool_calls, list):
        return False
    return any(isinstance(tool_call, Mapping) and bool(tool_call.get("id")) for tool_call in tool_calls)


def _chunk_has_output(chunk: Mapping[str, Any]) -> bool:
    if chunk.get("usage") is not None:
        return True
    choices = chunk.get("choices")
    if not isinstance(choices, list):
        return False
    return any(
        isinstance(choice, Mapping)
        and (bool(choice.get("delta")) or choice.get("finish_reason") is not None)
        for choice in choices
    )


def _engine_direct_binding(request_factory: object | None) -> VllmEngineDirectBinding | None:
    value = getattr(request_factory, "engine_direct_binding", None)
    return value if isinstance(value, VllmEngineDirectBinding) else None


def _reasoning_ended(request: object, binding: VllmEngineDirectBinding) -> bool | None:
    if binding.honors_include_reasoning and not bool(_getattr_default(request, "include_reasoning", True)):
        return True
    return None


def _direct_request_id(request: object) -> str:
    configured = _getattr_default(request, "request_id", None)
    suffix = configured if isinstance(configured, str) and configured else uuid.uuid4().hex
    return f"chatcmpl-{suffix}"


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
    return (
        (isinstance(value, Mapping) and "error" in value)
        or hasattr(value, "error")
        or type(value).__name__ == "ErrorResponse"
    )


def _raise_for_backend_error(value: object) -> None:
    error = value.get("error") if isinstance(value, Mapping) else getattr(value, "error", None)
    if error is None:
        return
    if isinstance(error, Mapping):
        message = error.get("message")
        error_type = error.get("type")
        status_code = error.get("code")
        parameter = error.get("param")
    else:
        message = getattr(error, "message", error)
        error_type = getattr(error, "type", None)
        status_code = getattr(error, "code", None)
        parameter = getattr(error, "param", None)
    raise _VllmBackendResponseError(
        message if isinstance(message, str) else str(message),
        error_type=error_type if isinstance(error_type, str) else None,
        status_code=status_code if type(status_code) is int else None,
        parameter=parameter if isinstance(parameter, str) else None,
    )


def _call_optional(owner: object, name: str, *args: object, **kwargs: object) -> object | None:
    method = getattr(owner, name, None)
    if not callable(method):
        return None
    return cast(object | None, method(*args, **kwargs))


async def _abort_request(engine_client: object, request_id: str) -> bool:
    abort = getattr(engine_client, "abort", None)
    if not callable(abort):
        return False
    try:
        result = abort(request_id)
        if inspect.isawaitable(result):
            result = await result
    except Exception:
        return False
    return result if isinstance(result, bool) else True


def _getattr_default(value: object, name: str, default: object) -> object:
    return getattr(value, name, default)


def _set_compat_attribute(value: object, name: str, attribute: object) -> None:
    setattr(value, name, attribute)


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


def _metadata_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)
