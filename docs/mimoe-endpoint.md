# The mimOE endpoint

What the agent relies on in mimOE Studio's OpenAI-compatible endpoint: the behaviour Studio
documents, and what I found by probing it on Studio 0.6.5 and 1.0.27.

## Documented behaviour I relied on

Studio 0.6.5 ships its own API reference under
`mimOE Studio.app/Contents/Resources/skills/mim-dev/` (`reference-inference-api.md` and
`reference-model-registry.md`). The agent leans on these documented facts:

- The inference API is OpenAI-compatible at `http://localhost:8083/mimik-ai/openai/v1`, takes a
  Bearer token whose default is `1234`, supports `stream: true` (SSE with `data:` lines and
  `[DONE]`) and honours `max_tokens`.
- Tool calls: "mILM parses `<tool_call>` tags from model output and returns structured tool calls",
  so a `tools` array in the request comes back as OpenAI-style `tool_calls` with `finish_reason:
  "tool_calls"`. The request-body table does not list `tools`, but the chat template injects them
  and both engines answered structured calls in every probe.
- Errors from the model routes use `{"message": ..., "statusCode": ...}` rather than OpenAI's
  `{"error": {...}}`, which the doc itself warns about; `friendly_error` reads both shapes.
- `GET /models` lists the loaded models with `info` (`max_context`, `n_params`, `model_size`) and
  `metrics` (`tokens_per_second`, `avg_tokens_per_second`); the banner, `/api/health` and the
  `mimoe_status` tool show those numbers. `POST /models {"model"}` loads with an SSE progress
  stream, `DELETE /models?modelId=` unloads, and a model auto-loads on its first completion.
- The model registry at `http://localhost:8083/mimik-ai/store/v1` provisions in two steps
  (`POST /models` metadata, then `POST /models/{id}/download {"url"}` with SSE progress), which is
  what `models pull` does on a 0.6 engine.

## Found by probing

Each of these is one curl away. `B=http://localhost:8083/mimik-ai/openai/v1`,
`H='-H "Authorization: Bearer 1234" -H "content-type: application/json"'`. Re-run on Studio 0.6.5
with `qwen3-4b` loaded; the 1.0.27 points come from the compatibility matrix.

1. `/no_think` works on 0.6 but leaves an empty think block in front of every answer:
   `curl -s $B/chat/completions $H -d '{"model":"qwen3-4b","messages":[{"role":"user","content":"Say hello in three words. /no_think"}],"max_tokens":24}'`
   answers `"content": "<think>\n\n</think>\n\nHello there!"`. The agent strips it before the
   checkpoint and before the UI sees it.
2. `enable_thinking` is ignored on 0.6 and honoured on 1.0: the same call with
   `"enable_thinking": false` and no `/no_think` still answers `<think>\nOkay, so I need to figure
   out what 2 plus 2 is...` on 0.6.5, while 1.0.27 answers clean content and, when thinking is on,
   puts the reasoning in `reasoning_content`. The preflight picks the control per engine generation.
3. `max_completion_tokens` is ignored, `max_tokens` is honoured: with `"max_completion_tokens": 5`
   the model kept going (128 tokens, `finish_reason: "stop"`); with `"max_tokens": 5` it wrote 5
   and finished with `length`. langchain-openai renames `max_tokens` to `max_completion_tokens`, so
   the agent sends the cap through `extra_body`.
4. `response_format` is not enforced: with `"response_format": {"type": "json_schema", ...}` the
   answer was `Hello! How can I assist you today?`. So no structured output; tools instead.
5. Tool-call ids are `tool_0`, `tool_1`, ... on 0.6 (every reply numbers from zero again) and
   `call_0_xxxxxxxx` on 1.0.27:
   `curl -s $B/chat/completions $H -d '{"model":"qwen3-4b","messages":[{"role":"user","content":"Call the ping tool now. /no_think"}],"max_tokens":48,"tools":[{"type":"function","function":{"name":"ping","description":"Reply with pong.","parameters":{"type":"object","properties":{}}}}]}'`
   answers `"tool_calls": [{"id": "tool_0", "type": "function", "function": {"name": "ping", "arguments": "{}"}}]`.
   The web UI keys its tool cards by id, so `stream.py` aliases a repeat as `tool_0#2`.
6. `GET /models` answers 200 without any key on both generations (`curl -s $B/models`), while a
   completion without a key answers `401 {"error":{"code":401,"message":"Unauthorized"}}` and a
   wrong key `403 {"error":{"code":403,"message":"Forbidden"}}`, in the OpenAI shape rather than
   the documented one. That is why the preflight validates the key with the warm-up completion
   instead of trusting `GET /models`.
7. A wrong base path answers 503 with an empty body:
   `curl -s -o /dev/null -w '%{http_code}' http://localhost:8083/openai/v1/models` prints `503`;
   `friendly_error` turns that into "the base URL path is wrong".
8. Per-request execution limit: 180 s on 0.6.5 and 300 s on 1.0.27 (its log says "Serverless
   Timeout"). Hence `timeout=200` and `max_retries=0` in `llm.py`: the engine's error arrives
   before ours, and a timed-out inference is never replayed.
9. Past the 12k context the engine answers 500 with `llama_decode() failed` in the body (seen at
   about 23k prompt tokens); the agent maps it to "the conversation exceeded the model's context
   window, start a new one with /new". Tool results are capped at 8 KB and old ones trimmed so a
   normal session does not get there.
10. What 1.0.27 adds: `GET /models` entries carry `family`, `supported_parameters`
    (`["tools","tool_choice","parallel_tool_calls"]` for Qwen3, `null` for SmolLM2) and a
    `reasoning` block whose `can_disable` is `false` for every model even though
    `enable_thinking: false` works, so the agent trusts the generation, not the flag; `usage` adds
    `ttft_ms`, `decode_token_per_second` and `prompt_tokens_details.cached_tokens` (it reuses the
    prompt prefix: 805 of 821 tokens cached on the second tool prompt, which made tool turns
    0.6 to 1.2 s instead of 3 s); load and unload are `PUT {store}/models {"id", "action":
    "load"|"unload"}` while the 0.6-style `POST`/`DELETE /models` answer 404; a tool call streams
    as one complete delta where 0.6 sends the name first and the arguments in fragments.
11. Memory: two multi-GB models loaded across the two engines on a 16 GB Mac ended in Metal
    `kIOGPUCommandBufferCallbackErrorOutOfMemory` and a SIGABRT of the 0.6.5 engine process.
12. Studio 0.6.5 holds one chat model: loading another evicts the first, and once in a while it
    answered a load request with 201 without actually loading (the agent re-reads `GET /models`
    after every load and says so when the model is not there).
13. An error in the middle of a 0.6.5 stream (for example `llama_decode` failing when the context
    fills up) is written without the blank line that ends an SSE event, so a standard client just
    sees the stream stop and returns a cut-off answer. The agent treats a stream that ends without
    a `finish_reason` as an error with a context/memory hint.
14. `GET /models` can also list models mimOE does not run locally: a configured cloud model on 0.6
    (`owned_by: "cloud"`) and provider models on 1.0 (`owned_by: "provider:<id>"`, `attached`).
    The agent never picks those on its own and warns if you name one with `--model`.
