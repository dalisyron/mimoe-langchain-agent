# mimoe-agent

A private, fully local dev/data assistant: a LangChain agent that runs on the mimOE Studio
on-device inference endpoint. Nothing leaves your machine.

I connected LangChain's `create_agent` to the [OpenAI-compatible endpoint](docs/mimoe-endpoint.md)
that mimOE Studio exposes on `localhost:8083`, gave the model eight tools over a sample workspace
folder, and put a human approval step in front of the one tool that can change anything (running
model-written Python). It has a terminal REPL and a small web UI, it works on both generations of
Studio I could get hold of (0.6.5 and 1.0.27), and the model choice is backed by a
[measured compatibility table](docs/compatibility.md) rather than a guess.
[docs/design.md](docs/design.md) explains how it works, why I chose each framework and tool, and
how the components connect.

CI: ruff + pytest on Ubuntu and Windows + vitest, green.

## What it does

You ask questions about a folder ("what files are here", "sum the revenue column", "where are the
TODOs") and a local model answers them by calling tools: list, read and search files, run Python,
a calculator, the clock, read-only git, and a status tool that reports on the engine itself. The
only tool that can change anything, `run_python` (also how the agent creates or edits files), stops
the run and asks you to approve or deny the code, in the terminal and in the browser, before anything
executes. Everything is on-device: the
model runs in mimOE Studio (the agent never picks a cloud or provider model that Studio may also
list), the agent's own tools never open a network connection, and the web server only listens on
`127.0.0.1`. The web UI keeps your conversations on this computer and lists them in a sidebar,
like ChatGPT's or Claude's, where you can search, rename and delete them; a reload or a server
restart reopens them, a pending approval included.

![The web UI: saved conversations on the left, the model menu at the top, the demo prompts in the middle](docs/img/web-empty-state.png)

![run_python asks for approval before the code runs](docs/img/web-approval-panel.png)

![After Approve: the tool call as one line (click it for the code and its output), the answer and the stats for the turn](docs/img/web-approved-answer.png)

![A saved conversation reopened from the sidebar in the dark theme, with the tool call's result opened](docs/img/web-history-dark.png)

## Quickstart

About 5 minutes, plus a 2.5 GB model download.

1. **Install mimOE Studio and start its engine.** Download
   [Studio 0.6.5](https://github.com/mimik-mimOE/mimOE-Studio/releases/tag/v0.6.5) for macOS on
   Apple Silicon or Windows x64; I also tested the agent with the
   [1.0.27 pre-release](https://github.com/mimik-mimOE/mimOE-Studio/releases/tag/1.0.27), the only
   version I ran on Windows.
   Studio 1.0.x asks on first launch to install the runtime and how it should run: choose
   **External** (a shared daemon on `localhost:8083`). **Built-in** opens no port, so no other
   program, this agent included, can reach it. The **API** button in Studio's model view shows the
   endpoint (`http://localhost:8083/mimik-ai/openai/v1`) and the key (`1234`); those are the
   agent's defaults.
2. **Install uv** (it installs Python 3.13 for you on first use), then open a new terminal so `uv`
   is on your PATH:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh        # macOS, Linux
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows
   ```
3. **Get the code.** On GitHub use Code > Download ZIP and unzip it (no git needed), or
   `git clone https://github.com/dalisyron/mimoe-langchain-agent.git`. `cd` into the folder.
4. **Get the recommended model**, `qwen3-4b-instruct-2507` (Qwen3-4B-Instruct-2507, Q4_K_M, 2.5 GB
   from `unsloth/Qwen3-4B-Instruct-2507-GGUF`). It scored 6/6 on tool selection on both engine
   generations, see the [compatibility matrix](docs/compatibility.md). The agent downloads it
   through Studio's own model registry, on either Studio generation:
   ```bash
   uv run mimoe-agent models pull qwen3-4b-instruct-2507
   uv run mimoe-agent models use qwen3-4b-instruct-2507
   ```
   The first `uv run` installs the locked dependencies (about 160 MB, 70 MB of it pandas and
   NumPy). Downloading and loading the model from Studio's own model page works just as well. A
   model that cannot produce tool calls, such as SmolLM2, still works for plain chat: the agent
   starts in chat-only mode and says so in its banner.
5. **Terminal REPL:**
   ```bash
   uv run mimoe-agent
   ```
   The agent finds the engine, picks the loaded model and checks that it can call tools: Studio
   1.0 says so in its model list; on Studio 0.6 the agent runs one warm-up completion with a tiny
   `ping` tool, which also validates the API key and absorbs the cold start. Then it prints a
   banner with the model, tokens/s, the workspace and the five demo prompts. `/help` lists the
   commands and `/quit` exits.
6. **Web UI:**
   ```bash
   uv run mimoe-agent serve
   ```
   then open http://127.0.0.1:8000. Same agent, same approval flow, plus a model picker. The built
   UI is committed, so no Node.js is needed.

For the speeds to expect and the platforms I tested on, see [docs/testing.md](docs/testing.md).

## Try it

Type these in order (the CLI banner lists them; they live in `workspace/README.md`). The answers
below are quoted from real sessions in `docs/transcripts/` (qwen3-4b on Studio 0.6.5); your
wording will differ, the facts should not.

| # | Prompt | Tool the model picks | You should see something like |
|---|---|---|---|
| 1 | `What files are in this workspace?` | `list_files` | "The files in the workspace are: README.md, notes.md, sales.csv, src/app.py, src/utils.py", each with its size. Five files, matching `ls -R workspace`. |
| 2 | `Summarize notes.md` | `read_file` | A short list: the CSV export shipped in week 38, a regression test is needed before the next release, the customer call went well and they want the dashboard by October, two ideas about caching the report and nightly versus on-demand runs. |
| 3 | `Use run_python to count the data rows of sales.csv and sum its revenue column.` | `run_python`, after your approval | The pandas code with line numbers, then the menu `Run this code?` with `1. Yes` and `2. No` (Esc cancels the turn). After Yes: "The sales.csv file has 8 data rows, and the sum of the revenue column is 1836.6." Check: `sales.csv` has 8 data rows below the header and the revenue column sums to 1836.6. After No: "The code was not executed as the user declined to run it." and the model stops. |
| 4 | `Find every TODO in this workspace and tell me where they are.` | `search_files` | "notes.md (line 5): TODO: add a regression test ...; notes.md (line 10): TODO: decide whether the report runs nightly or on demand" (plus README.md line 8, which is this prompt). The workspace has a third TODO in `src/app.py` line 25; in my runs the model searched `*.md` only and missed it, so if yours does too, ask it to search `*.py` as well. |
| 5 | `What model am I talking to, and how fast is it?` | `mimoe_status` | "You are talking to the "qwen3-4b" model, which has 4.0B parameters. It processes 31.8 tokens per second on average." Your model id and number will differ; they come from `GET /models`. |

The transcripts: `01-demo-prompts.md` (the five prompts with the approval), `02-denied-run-python.md`,
`03-model-switch.md`, `04-piped-stdin.md`, `05-models-list.md`, `06-ctrl-c-mid-turn.md`,
`07-bad-base-url.md`, `08-models-use-unload.md`, and `web-session.md` (the browser run with every
SSE frame the page received).

## Switching models

The [compatibility matrix](docs/compatibility.md) shows which models call tools reliably on each
Studio generation, and how fast they run on an Apple M1 Pro.

```bash
uv run mimoe-agent models list                        # registry, loaded models, presets, the recommended default
uv run mimoe-agent models pull qwen3-8b               # a preset id, or owner/repo:QUANT from Hugging Face
uv run mimoe-agent models use qwen3-8b                # load it (unloads the previous one unless --keep-loaded)
uv run mimoe-agent models unload qwen3-8b
```

## Framework and tooling choices

- **Agent:** LangChain 1.4 `create_agent` on LangGraph 1.2
- **Model client:** `ChatOpenAI` from langchain-openai, with a few adjustments for mimOE
- **Approval:** LangChain's built-in `HumanInTheLoopMiddleware` on `run_python`
- **Tools:** plain Python functions made into LangChain tools, with pandas available to `run_python`
- **Terminal:** typer and rich, with prompt_toolkit for the `you>` line and the approval menu
- **Web API:** FastAPI with Server-Sent Events (`sse-starlette`)
- **Web UI:** React, Vite and TypeScript with Tailwind CSS and shadcn/ui-style components (Radix
  menus, lucide icons), built into the committed `web/dist`
- **Tests:** pytest against an in-process fake engine (`httpx.MockTransport`), vitest for the UI
- **Python:** 3.13 or newer, with uv for the interpreter and the locked dependencies

The reasons for each choice, and the alternatives I rejected, are in
[docs/design.md](docs/design.md#framework-and-tooling-choices).

## Safety, honestly

`run_python` is not a sandbox. Code you approve runs as you, with your files, and can reach your
network: without `--allow-network` a socket guard stops only accidental network use. The approval
prompt is the only boundary, so read the code. `--auto-approve` means trusting the model, and file
contents are untrusted input: a file that tells the model to send your data somewhere can get that
done, and `--allow-network` makes it easy. The REPL's banner warns when you combine the two.

- `run_python` starts a fresh interpreter in the workspace with an allow-listed environment (none
  of your other variables) and kills it after 30 s, or once it uses more than 2 GB of memory or
  prints more than 32 MB. Before you approve, the terminal and the web UI flag common risky calls
  by name (network, subprocesses, deleting or writing files) and show invisible or terminal-control
  characters escaped. The terminal's approval menu drops keys typed before it appeared, so a stray
  Enter cannot approve code you have not seen.
- The other tools never touch the network (`mimoe_status` only asks the engine). The file tools
  are jailed to the workspace and refuse credential files; `git` is read-only and runs none of the
  repository's hooks.
- The web server listens on `127.0.0.1` only and accepts only loopback `Host` headers, so a
  DNS-rebinding page cannot drive it. It has no authentication: it is a single-user tool. Answers
  render without raw HTML or images (an image URL could leak file contents).
- The web UI saves conversations, file contents included, in a SQLite file only your user can read
  (created 0600 in a 0700 folder on macOS and Linux); `serve --history off` keeps nothing on disk.
  LangSmith tracing is forced off unless you pass `--trace`.

The full version, including what gets past these limits, is in [docs/safety.md](docs/safety.md).

## Configuration

Precedence: command-line flags, then `MIMOE_*` variables in the process environment, then a
`.env` file in the current directory (only the four string settings are read from it, and it never
walks up to parent directories), then the defaults. The banner shows where every value came from.
Booleans accept `1/true/yes/on` and `0/false/no/off`; anything else is an error rather than a
silent `false`. `.env.example` lists the keys.

| Setting | Flag | Environment | From `.env` | Default |
|---|---|---|---|---|
| Base URL | `--base-url` | `MIMOE_BASE_URL` | yes | auto-discover: `http://127.0.0.1:8083/mimik-ai/openai/v1`, then `/openai/v1` on the same port, then `localhost` (IPv4 first: Windows spends about 2 s on every refused IPv6 connect); an explicit URL is never replaced by a candidate |
| API key | `--api-key` | `MIMOE_API_KEY` | yes | `1234`, validated by the warm-up completion |
| Model | `--model` | `MIMOE_MODEL` | yes | the first loaded chat model; the id must be loaded (`models use` loads it) |
| Workspace | `--workspace` | `MIMOE_WORKSPACE` | yes | `./workspace`; an error with a hint if it does not exist |
| Thinking | `--think` | `MIMOE_THINK` | no | off; on, the token cap rises from 1024 to 4096 and the reasoning is shown collapsed (`/verbose` for all of it); experimental |
| Auto-approve | `--auto-approve` | `MIMOE_AUTO_APPROVE` | no | off; on, `run_python` runs without asking |
| Network for approved code | `--allow-network` | `MIMOE_ALLOW_NETWORK` | no | off; lifts only the child's socket guard |
| Force tools | `--force-tools` | `MIMOE_FORCE_TOOLS` | no | off; skips the probe and offers the tools anyway |
| Tracing | `--trace` | `MIMOE_TRACE` | no | off; keeps LangSmith variables as they are |
| Web port | `serve --port` | | | 8000, loopback only |
| Conversation history | `serve --history PATH` (`off` keeps it in memory) | `MIMOE_HISTORY` | no | `conversations.sqlite` in the per-user data folder: `~/Library/Application Support/mimoe-agent/` on macOS, `%LOCALAPPDATA%\mimoe-agent\` on Windows, `~/.local/share/mimoe-agent/` on Linux |

Exit codes of the REPL: 0 after `/quit` or EOF, 1 for a configuration or preflight failure (the
hint is on stderr), 130 for Ctrl-C at the prompt. In a terminal, an approval is a menu: arrows and
Enter, `1`/`2` or `y`/`n` answer it, and Esc or Ctrl-C cancels the turn (the code does not run and
the model is not asked again). Piped input works
(`printf "What files are here?\n/quit\n" | uv run mimoe-agent --auto-approve`); without
`--auto-approve` an approval question reads `y` or `n` from the next stdin line.

## Development

```bash
uv sync                                   # dependencies plus the dev group
uv run pytest -q                          # 748 offline tests against the fake engine, no Studio needed
MIMOE_LIVE=1 uv run pytest -m live        # 9 live tests against a running Studio (a handful of completions)
uvx ruff check . && uvx ruff format --check .
cd web && npm ci && npm test && npm run build   # 41 vitest tests; the build writes web/dist
```

The committed bundle: `web/dist` is in git so that you do not need Node.js. After a UI change
run `npm run build` and commit the new bundle in its own "build: refresh web bundle" commit; CI
rebuilds it and warns (without failing) when the committed one differs. For UI work without
inference, `uv run python web/dev/fake_backend.py` serves a scripted backend on `:8000` that plays
every event type, including the approval flow, and `npm run dev` proxies `/api` to it.

Tests: `tests/conftest.py` is an in-process fake of both engine generations behind
`httpx.MockTransport`, injected into ChatOpenAI's HTTP clients and the engine client, with a
scripted queue of replies and the model store. The suite covers the path jail, the `run_python`
kills and caps, the middleware, the stream mapping (sync and async), the agent's approve and reject
paths, the server's 409/422 and disconnect handling, scripted CLI sessions, and `models pull`/`use`
against the fake store. `tests/test_live.py` is the only file that talks to a real engine.

CI (`.github/workflows/ci.yml`): `lint` runs ruff on Ubuntu; `test` runs the offline suite plus a
`mimoe-agent --help` smoke on `ubuntu-latest` and `windows-latest` with Python 3.13 from
`uv sync --locked`; `web` runs `npm ci`, the vitest suite, the build and the bundle diff on Node 22.
Changes to only `README.md` or `docs/` do not trigger it.

## How I used AI coding assistants

I used Claude Code as the primary assistant, and I used it in an orchestrated way rather than as
an autocomplete. Before any code was written, research agents verified the LangChain 1.4 and
LangGraph 1.2 APIs against the installed source rather than the docs, probed the live endpoint on
both Studio generations and wrote down what they found; most of the probing results in
[docs/mimoe-endpoint.md](docs/mimoe-endpoint.md#found-by-probing) started there. Next it drafted a
plan, which I revised with it and approved before any code was written, and an interface contract
(`docs/CONTRACTS.md`: signatures, data shapes, the SSE event protocol); implementation agents then
worked in parallel on disjoint files against that contract. Each module then went to an
adversarial review agent whose job was to break it and add a regression test for whatever it
broke, and a final six-lens review of the whole repository found the last round of defects. The
[model compatibility matrix](docs/compatibility.md) was a separate, measured job, and the CLI
transcripts and the browser session are recordings of real runs, not prose written by hand.

## Limitations and next steps

- Only the web UI saves conversations; the REPL keeps them in memory and `/new` starts over. A
  conversation's title is its first message (rename it in the sidebar): a model-written title
  would cost an extra model call per conversation on a laptop CPU.
- No `fetch_url` tool yet. I cut it from this version; the design I want is a per-URL approval like
  `run_python`, IP pinning against SSRF (resolve once, refuse private ranges, connect to that
  address), a wall-clock limit and a size cap.
- Thinking is hidden unless you pass `--think`, and then only a one-line summary shows unless
  `/verbose` is on.
- Single-user loopback server: no authentication, per-thread locks kept for the process lifetime,
  no cap on the message size (a very large message is checkpointed and then hits the context error
  on every later turn; use New conversation).
- Long threads are trimmed, not summarised: tool results from earlier turns are shortened to 400
  characters in the request, every result is capped at 8 KB, and a long chat still reaches the 12k
  context eventually, at which point the agent tells you to start a new one.

## License

MIT, see [LICENSE](LICENSE).
