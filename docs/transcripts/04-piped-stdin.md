# 04 Piped stdin (non-tty) with --auto-approve

- Date: 2026-09-25 02:22 PDT
- Engine: mimOE Studio 0.6.5 (node v3.22.8) at http://127.0.0.1:8083/mimik-ai/openai/v1, model qwen3-4b, Apple M1 Pro
- Stdin: the printf pipe above (no terminal)
- Exit code: 0
- Command: `printf "What files are here?\n/quit\n" | uv run mimoe-agent --auto-approve`

The prompt is answered, `/quit` ends the session and the exit code is 0.

## Raw output

```text
╭─ mimoe-agent 0.1.0 ──────────────────────────────────────────────────────────────────────────────╮
│ model      qwen3-4b  (default: first loaded)                                                     │
│ speed      36.0 tokens/s (avg 33.4)                                                              │
│ context    12,000 tokens                                                                         │
│ engine     v3.22.8 (developer edition), 0.6-generation API                                       │
│ node       MacBookPro.lan                                                                        │
│ endpoint   http://localhost:8083/mimik-ai/openai/v1  (discovered)                                │
│ workspace  /Users/nibom/mimoe-assignment/workspace  (default)                                    │
│ approval   auto: model-written Python runs without asking  (flag)                                │
│ network    blocked for run_python  (default)                                                     │
│ thinking   off  (default)                                                                        │
│ tools      on                                                                                    │
│ probe: qwen3-4b answered with a structured ping call in 1.6 s                                    │
│ Try, in order:                                                                                   │
│   1. What files are in this workspace?                                                           │
│   2. Summarize notes.md                                                                          │
│   3. How many rows does sales.csv have, and what is the total revenue?                           │
│   4. Find every TODO in this workspace and tell me where they are.                               │
│   5. What model am I talking to, and how fast is it?                                             │
│                                                                                                  │
│ Commands: /new /status /verbose /model ID /models /help /quit                                    │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
you> What files are here?
-> list_files(path='.', max_depth=4)
   README.md  (321 B)
   notes.md  (307 B)
   sales.csv  (287 B)
   ... (+3 more lines; /verbose shows all)
The files in the workspace are:

 • README.md (321 B)
 • notes.md (307 B)
 • sales.csv (287 B)
 • src/app.py (916 B)
 • src/utils.py (109 B)
elapsed 9.6 s, 2 model calls, 96 tokens
you> /quit
```
