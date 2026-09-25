# 05 mimoe-agent models list

- Date: 2026-09-25 02:22 PDT
- Engine: mimOE Studio 0.6.5 (node v3.22.8) at http://127.0.0.1:8083/mimik-ai/openai/v1, model qwen3-4b, Apple M1 Pro
- Exit code: 0
- Command: `uv run mimoe-agent models list`

## Raw output

```text
models on http://localhost:8083/mimik-ai/openai/v1 (0.6-generation engine)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ model                    ┃ size   ┃ state          ┃ context ┃ tok/s ┃ tools ┃ thinking ┃ notes                                          ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ qwen3-4b                 │ 2.5 GB │ loaded         │ 12,000  │ 36.0  │ ?     │ ?        │                                                │
│ smollm2-360m             │ 0.4 GB │ ready          │ 12,000  │ -     │ ?     │ ?        │ both engines; chat only (no tool calling on    │
│                          │        │                │         │       │       │          │ either engine), no thinking; fastest (135-150  │
│                          │        │                │         │       │       │          │ tok/s)                                         │
│ qwen3-4b-instruct-2507 * │ 2.5 GB │ ready          │ 12,000  │ -     │ ?     │ ?        │ recommended default: loads on 0.6 and 1.0      │
│                          │        │                │         │       │       │          │ engines, 6/6 structured tool calls, clean      │
│                          │        │                │         │       │       │          │ round trip, no thinking mode to manage         │
│ smollm3-3b               │ 1.9 GB │ ready          │ 12,000  │ -     │ ?     │ ?        │ both engines; tool calls are structured only   │
│                          │        │                │         │       │       │          │ on 1.0 (plain text on 0.6) and the round trip  │
│                          │        │                │         │       │       │          │ breaks either way; thinking leaks through      │
│                          │        │                │         │       │       │          │ /no_think on 0.6                               │
│ qwen3-8b                 │ 5.0 GB │ ready          │ 12,000  │ -     │ ?     │ ?        │ both engines, 6/6 structured tool calls;       │
│                          │        │                │         │       │       │          │ thinking model (keep it off for tools); 5 GB   │
│                          │        │                │         │       │       │          │ and about 40% slower per token than the 4B     │
│ qwen3.5-4b               │ 2.7 GB │ ready          │ 12,000  │ -     │ ?     │ ?        │ 1.0 engines only (fails to load on 0.6.5); 6/6 │
│                          │        │                │         │       │       │          │ structured tool calls; brief thinking (~60     │
│                          │        │                │         │       │       │          │ tokens) when enabled; does not load on this    │
│                          │        │                │         │       │       │          │ engine                                         │
│ qwen3.5-9b               │ 5.7 GB │ not registered │ -       │ -     │ ?     │ ?        │ 1.0 engines only; 6/6 structured tool calls;   │
│                          │        │                │         │       │       │          │ brief thinking when enabled; 5.7 GB, needs a   │
│                          │        │                │         │       │       │          │ 16 GB machine with nothing else open; does not │
│                          │        │                │         │       │       │          │ load on this engine                            │
└──────────────────────────┴────────┴────────────────┴─────────┴───────┴───────┴──────────┴────────────────────────────────────────────────┘
* recommended default. tools/thinking: from the engine's capability list on 1.0 engines, from the start-up probe on 0.6 engines, ? when
unknown. `mimoe-agent models pull ID` downloads a preset, `models use ID` loads it.
```
