# mimoe-agent

A LangChain agent (Python 3.13, uv) that runs entirely against the local mimOE Studio
inference endpoint. `PLAN.md` (untracked) is the working spec.

## Commands
- `uv sync` then `uv run mimoe-agent` (REPL) or `uv run mimoe-agent serve` (web UI on 127.0.0.1:8000)
- `uv run pytest -q` (offline; `MIMOE_LIVE=1 uv run pytest -m live` needs Studio)
- `uvx ruff check . && uvx ruff format --check .`
- `cd web && npm ci && npm test && npm run build` (commit `web/dist` in a dedicated commit)

## Conventions
- Tools return strings, never raise, never return "". Every tool result is capped at 8 KB.
- Cross-platform first: pathlib and `sys.executable` subprocesses, no bash, no POSIX-only APIs
  without a Windows branch.
- `run_python` is not a sandbox; the approval step is the boundary. Say so honestly in docs.
- Never commit `email.md`, `*.dmg`, or `.env`. One commit per plan step.
