import { SERVER_EVENT_TYPES, type Decision, type Health, type Json, type Models, type ServerEvent, type SwitchResult } from './events';
import { parseSSE } from './sse';
import type { ThreadDetail, ThreadSummary } from './threads';

/** A non-2xx response; `hint` comes from FastAPI's `{detail: {message, hint}}` shape when present. */
export class HttpError extends Error {
  readonly status: number;
  readonly hint?: string;
  constructor(status: number, message: string, hint?: string) {
    super(`HTTP ${status}: ${message}`);
    this.status = status;
    this.hint = hint;
  }
}

async function toHttpError(res: Response): Promise<HttpError> {
  const text = await res.text();
  let detail: unknown = text || res.statusText;
  try {
    detail = (JSON.parse(text) as { detail?: unknown }).detail ?? detail;
  } catch { /* not JSON */ }
  if (detail && typeof detail === 'object' && !Array.isArray(detail)) {
    const d = detail as { message?: string; hint?: string };
    return new HttpError(res.status, d.message ?? JSON.stringify(detail), d.hint);
  }
  return new HttpError(res.status, typeof detail === 'string' ? detail : JSON.stringify(detail));
}

/** POST a JSON body and yield the typed events of the SSE response; unknown event names are skipped. */
async function* stream(path: string, body: Json, signal: AbortSignal): AsyncGenerator<ServerEvent> {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'content-type': 'application/json', accept: 'text/event-stream' },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) throw await toHttpError(res);
  for await (const msg of parseSSE(res.body)) {
    if (SERVER_EVENT_TYPES.has(msg.event)) yield { type: msg.event, ...(JSON.parse(msg.data) as Json) } as ServerEvent;
  }
}

async function json<T>(path: string, body?: Json, method = body === undefined ? 'GET' : 'POST'): Promise<T> {
  const init: RequestInit = { method };
  if (body !== undefined) Object.assign(init, { headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
  const res = await fetch(path, init);
  if (!res.ok) throw await toHttpError(res);
  return (res.status === 204 ? undefined : await res.json()) as T;
}

const threadPath = (id: string) => `/api/threads/${encodeURIComponent(id)}`;

export const api = {
  chat: (threadId: string, message: string, signal: AbortSignal) =>
    stream('/api/chat', { thread_id: threadId, message }, signal),
  resume: (threadId: string, interruptId: string, decisions: Decision[], signal: AbortSignal) =>
    stream('/api/resume', { thread_id: threadId, interrupt_id: interruptId, decisions }, signal),
  health: () => json<Health>('/api/health'),
  models: () => json<Models>('/api/models'),
  switchModel: (model: string, unloadPrevious: boolean) =>
    json<SwitchResult>('/api/model', { model, unload_previous: unloadPrevious }),
  threads: () => json<{ threads: ThreadSummary[] }>('/api/threads'),
  thread: (id: string) => json<ThreadDetail>(threadPath(id)),
  renameThread: (id: string, title: string) => json<ThreadSummary>(threadPath(id), { title }, 'PATCH'),
  deleteThread: (id: string) => json<void>(threadPath(id), undefined, 'DELETE'),
};
