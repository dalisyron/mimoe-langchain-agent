# web/ — the chat UI

React 19 + Vite + TypeScript, no UI kit. `mimoe-agent serve` mounts the committed `dist/` at `/`,
so a reviewer without Node still gets the UI.

- `npm ci && npm test && npm run build` — unit tests (reducer, SSE parser, stream consumer, markdown safety) and the production bundle.
- `npm run dev` — Vite on http://localhost:5173 with `/api` proxied to the FastAPI server on :8000.
- `uv run python web/dev/fake_backend.py` — a scripted backend (no inference) that plays every
  event type including the `run_python` approval flow; see its docstring for the trigger words.

Reading order (about 700 lines): `events.ts` (wire contract) → `sse.ts` (parser) → `api.ts` →
`reducer.ts` (events → ordered blocks per assistant turn) → `useChat.ts` → `App.tsx` → `components/`.
