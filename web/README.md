# web/ — the chat UI

React 19 + Vite + TypeScript, styled with Tailwind CSS v4 and shadcn/ui-style components (Radix for
the menus, lucide icons, Prism for code). `mimoe-agent serve` mounts the committed `dist/` at `/`,
so a reviewer without Node still gets the UI. Nothing is fetched from the internet: system fonts,
no CDN.

- `npm ci && npm test && npm run build` — unit tests (reducer, SSE parser, stream consumer, tool
  steps, sidebar grouping, markdown safety) and the production bundle.
- `npm run dev` — Vite on http://localhost:5173 with `/api` proxied to the FastAPI server on :8000.
- `uv run python web/dev/fake_backend.py` — a scripted backend (no inference) that plays every
  event type including the `run_python` approval flow, and keeps conversations in memory for the
  sidebar; see its docstring for the trigger words.

Reading order: `events.ts` (wire contract) → `sse.ts` (parser) → `api.ts` → `reducer.ts` (events →
ordered blocks per assistant turn) → `useChat.ts`, `useThreads.ts`, `threads.ts` → `App.tsx` →
`components/` (`ui/` holds the button and menu primitives).
