import { useCallback, useReducer, useRef } from 'react';
import { HttpError, api } from './api';
import type { Action, Decision, ServerEvent } from './events';
import { initialState, reducer } from './reducer';

// crypto.randomUUID needs a secure context; http://127.0.0.1 and localhost qualify.
const newThreadId = (): string =>
  typeof crypto.randomUUID === 'function' ? crypto.randomUUID() : Math.random().toString(36).slice(2);

const CLOSED_EARLY = 'The connection closed before the run finished.';
const CLOSED_HINT = 'Is `mimoe-agent serve` still running? Send the message again or start a new conversation.';
const NETWORK_HINT = 'Is `mimoe-agent serve` running?';

/** Feed one SSE response into the reducer. Every run must end with `done` or `error`; a stream that closes
 * without either (server restart, proxy drop) is reported as `stream_failed` so the composer is re-enabled.
 * Once `signal` is aborted (Stop / New conversation) nothing more is dispatched: the state was already updated. */
export async function consumeStream(events: AsyncIterable<ServerEvent>, dispatch: (action: Action) => void, signal: AbortSignal): Promise<void> {
  let finished = false;
  try {
    for await (const ev of events) {
      if (signal.aborted) return;
      dispatch(ev);
      if (ev.type === 'done' || ev.type === 'error') finished = true;
    }
  } catch (e) {
    if (signal.aborted || (e as Error).name === 'AbortError') return;
    const message = e instanceof Error ? e.message : String(e);
    const hint = e instanceof HttpError ? e.hint : e instanceof TypeError ? NETWORK_HINT : undefined; // fetch's network failure is a TypeError
    dispatch({ type: 'stream_failed', message, hint });
    return;
  }
  if (!finished && !signal.aborted) dispatch({ type: 'stream_failed', message: CLOSED_EARLY, hint: CLOSED_HINT });
}

/** Chat state plus the actions the UI can take; the reducer stays pure, ids are made here. */
export function useChat() {
  const [state, dispatch] = useReducer(reducer, undefined, () => initialState(newThreadId()));
  const abort = useRef<AbortController | null>(null);

  const newSignal = useCallback(() => {
    abort.current?.abort();
    abort.current = new AbortController();
    return abort.current.signal;
  }, []);

  const send = (text: string) => {
    dispatch({ type: 'send', text });
    const signal = newSignal();
    void consumeStream(api.chat(state.threadId, text, signal), dispatch, signal);
  };
  const decide = (decisions: Decision[]) => {
    if (!state.approval) return;
    const { interruptId } = state.approval;
    dispatch({ type: 'decided', decisions });
    const signal = newSignal();
    void consumeStream(api.resume(state.threadId, interruptId, decisions, signal), dispatch, signal);
  };
  const stop = () => { abort.current?.abort(); dispatch({ type: 'stopped' }); };
  const reset = () => { abort.current?.abort(); dispatch({ type: 'reset', threadId: newThreadId() }); };
  return { state, send, decide, stop, reset };
}
