# Testing and performance

How fast the recommended model runs and where the agent was tested. The measurements for each
model I tried are in [compatibility.md](compatibility.md), recordings of real sessions are in
[transcripts/](transcripts/), and the README's [Development](../README.md#development) section shows
how to run the test suites.

## What to expect

On an Apple M1 Pro (16 GB) with Studio 0.6.5, `qwen3-4b-instruct-2507` decodes at
about 40 tokens/s and takes about 2.7 s to the first token on the full 8-tool prompt (36 tokens/s
on Studio 1.0.27); the first demo turn is about 10 s end to end from cold (two model calls). On a
CPU-only Windows Server 2025 VM (6 vCPU, Studio 1.0.27) the same model decodes at about 6
tokens/s. There the first prompt takes about a minute, because the engine reads the roughly
1,200-token system and tool prompt once (at about 25 tokens/s) and then caches it; after that each
demo prompt takes 25 to 30 s.

## Tested on

- macOS 26.6 on an Apple M1 Pro (16 GB) with Studio 0.6.5 and a Studio 1.0.27 runtime: the offline
  suite, the live smoke tests, the demo prompts in the terminal and the browser, and a real Ctrl-C
  under a terminal (the turn ends within 0.2 s).
- Windows Server 2025 (x64, CPU-only VM) with the Studio 1.0.27 runtime, installed from a ZIP of
  this repository: the offline suite (all pass; 7 POSIX-only tests skip), the five demo prompts,
  approve and deny, the web UI and API, a UTF-16 `.env`, and the Stop, memory and timeout kills
  (no `python.exe` left behind).
- Ubuntu through CI (offline tests only).
