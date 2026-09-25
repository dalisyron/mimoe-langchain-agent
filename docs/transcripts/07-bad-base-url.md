# 07 --base-url pointing at nothing: the friendly failure

- Date: 2026-09-25 02:22 PDT
- Engine: none reachable at that URL (port 9 refuses connections)
- Exit code: 1
- Command: `uv run mimoe-agent --base-url http://127.0.0.1:9/x`

The message and the hint come from `MimoeError`; the process exits with 1 and never starts the REPL.

## Raw output

```text
error: mimOE Studio is not reachable
Open mimOE Studio (menu-bar icon) and wait for the status dot to turn green, then retry; if Studio
listens on another port pass --base-url http://localhost:PORT/mimik-ai/openai/v1. Tried:
http://127.0.0.1:9/x: ConnectError.
```
