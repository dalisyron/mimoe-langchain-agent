import type { Action, ActionRequest, Decision, Json, Usage } from './events';

export type ToolStatus = 'running' | 'awaiting_approval' | 'done' | 'error' | 'denied';
export type ToolBlock = { kind: 'tool'; id: string; name: string; args: Json; status: ToolStatus; result?: string };

/** An assistant turn is an ordered list of blocks, so thinking, tool cards and text keep their real order. */
export type Block =
  | { kind: 'thinking'; text: string }
  | { kind: 'text'; text: string }
  | ToolBlock
  | { kind: 'notice'; text: string }
  | { kind: 'error'; message: string; hint?: string };

export type Stats = { elapsed_s: number; model: string; usage?: Usage | null };
export type AssistantTurn = { role: 'assistant'; blocks: Block[]; stats?: Stats };
export type Turn = { role: 'user'; text: string } | AssistantTurn;
export type Approval = { interruptId: string; requests: ActionRequest[] };

export type ChatState = {
  threadId: string;
  turns: Turn[];
  streaming: boolean; // an SSE response is open
  approval: Approval | null; // the ApprovalPanel shows while set; the composer stays disabled
};

export const initialState = (threadId: string): ChatState => ({ threadId, turns: [], streaming: false, approval: null });

/** Apply `fn` to the current assistant turn (server events always belong to one). */
function withTurn(state: ChatState, fn: (turn: AssistantTurn) => AssistantTurn): ChatState {
  const last = state.turns[state.turns.length - 1];
  const turn: AssistantTurn = last?.role === 'assistant' ? last : { role: 'assistant', blocks: [] };
  const rest = last?.role === 'assistant' ? state.turns.slice(0, -1) : state.turns;
  return { ...state, turns: [...rest, fn(turn)] };
}
const withBlocks = (state: ChatState, fn: (blocks: Block[]) => Block[]): ChatState =>
  withTurn(state, (t) => ({ ...t, blocks: fn(t.blocks) }));

/** Append streamed text to a trailing block of the same kind, or open a new one. */
function appendText(blocks: Block[], kind: 'thinking' | 'text', text: string): Block[] {
  const last = blocks[blocks.length - 1];
  return last?.kind === kind ? [...blocks.slice(0, -1), { kind, text: last.text + text }] : [...blocks, { kind, text }];
}

/** The k-th action request is the k-th running tool of that name (HITL keeps the model's order). */
function markAwaiting(blocks: Block[], requests: ActionRequest[]): Block[] {
  const out = [...blocks];
  for (const r of requests) {
    const i = out.findIndex((b) => b.kind === 'tool' && b.status === 'running' && b.name === r.name);
    if (i >= 0) out[i] = { ...(out[i] as ToolBlock), status: 'awaiting_approval' };
  }
  return out;
}

/** The k-th decision answers the k-th tool that is awaiting approval. */
function applyDecisions(blocks: Block[], decisions: Decision[]): Block[] {
  let k = 0;
  return blocks.map((b) =>
    b.kind === 'tool' && b.status === 'awaiting_approval' ? { ...b, status: decisions[k++] === 'reject' ? 'denied' : 'running' } : b,
  );
}

export function reducer(state: ChatState, action: Action): ChatState {
  switch (action.type) {
    case 'send':
      return { ...state, streaming: true, turns: [...state.turns, { role: 'user', text: action.text }, { role: 'assistant', blocks: [] }] };
    case 'token':
    case 'thinking':
      if (!action.text) return state;
      return withBlocks(state, (b) => appendText(b, action.type === 'token' ? 'text' : 'thinking', action.text));
    case 'tool_call':
      return withBlocks(state, (b) => [...b, { kind: 'tool', id: action.id, name: action.name, args: action.args, status: 'running' }]);
    case 'tool_result': { // a denied tool stays denied even though the backend reports the rejection as an error
      const status: ToolStatus = action.is_error ? 'error' : 'done';
      return withBlocks(state, (blocks) => blocks.some((b) => b.kind === 'tool' && b.id === action.id)
        ? blocks.map((b) => (b.kind === 'tool' && b.id === action.id ? { ...b, result: action.content, status: b.status === 'denied' ? 'denied' : status } : b))
        // no tool_call announced it (out-of-order or dropped frame): show the result rather than lose it
        : [...blocks, { kind: 'tool', id: action.id, name: action.name, args: {}, status, result: action.content }]);
    }
    case 'approval_required':
      return {
        ...withBlocks(state, (b) => markAwaiting(b, action.action_requests)),
        approval: { interruptId: action.interrupt_id, requests: action.action_requests },
      };
    case 'decided':
      return { ...withBlocks(state, (b) => applyDecisions(b, action.decisions)), approval: null, streaming: true };
    case 'notice':
      return withBlocks(state, (b) => [...b, { kind: 'notice', text: action.text }]);
    case 'done':
      return {
        ...withTurn(state, (t) => ({ ...t, stats: { elapsed_s: action.elapsed_s, model: action.model, usage: action.usage_total } })),
        streaming: false,
      };
    case 'error':
    case 'stream_failed':
      return {
        ...withBlocks(state, (b) => [...b, { kind: 'error', message: action.message, hint: action.hint ?? undefined }]),
        streaming: false,
      };
    case 'stopped': // the fetch was aborted, so running tools cannot report back
      if (!state.streaming) return state;
      return {
        ...withBlocks(state, (blocks) => [
          ...blocks.map((b) => (b.kind === 'tool' && b.status === 'running' ? { ...b, status: 'error' as const, result: 'stopped' } : b)),
          { kind: 'notice', text: 'Stopped.' },
        ]),
        streaming: false,
      };
    case 'reset':
      return initialState(action.threadId);
  }
}
