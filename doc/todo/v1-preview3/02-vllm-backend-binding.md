# vLLM Backend Binding

- [x] Add a vLLM serving-object wrapper with method probing for chat completion entry points.
- [x] Keep vLLM as an optional runtime extra so non-GPU CI can validate adapter logic without installing a serving stack.
- [x] Bind against the selected vLLM `0.18.x` OpenAI serving classes in an integration test environment.
- [x] Replace HTTP/SSE relay benchmarking with a true in-process vLLM NNRP serving path.
- [x] Add an in-process streaming bridge that consumes vLLM OpenAI serving chunks without starting the OpenAI HTTP server.
- [x] Add a vLLM process entrypoint or patch hook that registers the NNRP profile server beside the existing OpenAI serving stack.
- [x] Preserve vLLM streaming chunk usage and tool-call data without flattening them into text-only events.
- [x] Add cancellation propagation from NNRP frame cancellation into the active vLLM request path.
- [x] Propagate NNRP cancellation to the active vLLM request id through the serving engine abort path.
- [x] Add overload and scheduler-rejection mapping into OpenAI-compatible profile error bodies.
- [x] Keep HTTP/SSE relay mode as a compatibility smoke path only, not as the optimized Level 1 runtime.
