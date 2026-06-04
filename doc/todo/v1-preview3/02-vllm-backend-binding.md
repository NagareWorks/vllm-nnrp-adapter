# vLLM Backend Binding

- [x] Add a vLLM serving-object wrapper with method probing for chat completion entry points.
- [x] Keep vLLM as an optional runtime extra so non-GPU CI can validate adapter logic without installing a serving stack.
- [x] Bind against the selected vLLM `0.18.x` OpenAI serving classes in an integration test environment.
- [x] Preserve vLLM streaming chunk usage and tool-call data without flattening them into text-only events.
- [x] Add cancellation propagation from NNRP frame cancellation into the active vLLM request path.
- [x] Add overload and scheduler-rejection mapping into OpenAI-compatible profile error bodies.
