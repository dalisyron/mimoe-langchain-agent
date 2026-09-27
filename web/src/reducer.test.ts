import { describe, expect, it } from 'vitest';
import type { Action } from './events';
import { RUN_ENDED, initialState, reducer, type AssistantTurn, type ChatState } from './reducer';

const replay = (actions: Action[], state: ChatState = initialState('t1')) => actions.reduce(reducer, state);
const assistant = (s: ChatState, i = 1): AssistantTurn => {
  const t = s.turns[i];
  if (t?.role !== 'assistant') throw new Error(`turn ${i} is not an assistant turn`);
  return t;
};
const done = (status: 'completed' | 'awaiting_approval' = 'completed'): Action => ({
  type: 'done', status, elapsed_s: 2.5, model: 'qwen3-4b',
  usage_total: { input_tokens: 200, output_tokens: 30, llm_calls: 2 },
});

describe('reducer: event sequence -> blocks', () => {
  it('keeps thinking, tool cards, text and notices in stream order inside one assistant turn', () => {
    const s = replay([
      { type: 'send', text: 'list the files' },
      { type: 'thinking', text: 'The user' }, { type: 'thinking', text: ' wants a listing.' },
      { type: 'tool_call', id: 'tool_0', name: 'list_files', args: { path: '.' } },
      { type: 'tool_result', id: 'tool_0', name: 'list_files', content: 'notes.md\nsales.csv', is_error: false },
      { type: 'thinking', text: 'Two files.' }, // Qwen thinks again after a tool result
      { type: 'token', text: '' }, // empty deltas open no block
      { type: 'token', text: 'Two files: ' }, { type: 'token', text: '**notes.md** and sales.csv' },
      { type: 'notice', text: 'Model call limit reached (8/8)' },
      done(),
    ]);
    expect(s.turns[0]).toEqual({ role: 'user', text: 'list the files' });
    expect(assistant(s).blocks).toEqual([
      { kind: 'thinking', text: 'The user wants a listing.' },
      { kind: 'tool', id: 'tool_0', name: 'list_files', args: { path: '.' }, status: 'done', result: 'notes.md\nsales.csv' },
      { kind: 'thinking', text: 'Two files.' },
      { kind: 'text', text: 'Two files: **notes.md** and sales.csv' },
      { kind: 'notice', text: 'Model call limit reached (8/8)' },
    ]);
    expect(assistant(s).stats).toEqual({
      elapsed_s: 2.5, model: 'qwen3-4b', usage: { input_tokens: 200, output_tokens: 30, llm_calls: 2 },
    });
    expect(s.streaming).toBe(false);
    expect(s.approval).toBeNull();
  });

  it('a tool_result without a preceding tool_call gets its own card instead of being dropped', () => {
    const s = replay([
      { type: 'send', text: 'x' },
      { type: 'tool_result', id: 'tool_7', name: 'calculator', content: '42', is_error: false },
      { type: 'tool_result', id: 'tool_7', name: 'calculator', content: '43', is_error: true }, // same id again: updates, no duplicate
    ]);
    expect(assistant(s).blocks).toEqual([{ kind: 'tool', id: 'tool_7', name: 'calculator', args: {}, status: 'error', result: '43' }]);
  });

  it('marks a failed tool as error', () => {
    const s = replay([
      { type: 'send', text: 'read it' },
      { type: 'tool_call', id: 'tool_0', name: 'read_file', args: { path: 'nope.txt' } },
      { type: 'tool_result', id: 'tool_0', name: 'read_file', content: 'ERROR: no such file', is_error: true },
      done(),
    ]);
    expect(assistant(s).blocks[0]).toMatchObject({ kind: 'tool', status: 'error', result: 'ERROR: no such file' });
  });
});

describe('reducer: approval flow', () => {
  const untilApproval: Action[] = [
    { type: 'send', text: 'total revenue?' },
    { type: 'tool_call', id: 'tool_0', name: 'run_python', args: { code: 'print(1)' } },
    { type: 'approval_required', interrupt_id: 'int-1', action_requests: [{ name: 'run_python', args: { code: 'print(1)' } }] },
    done('awaiting_approval'),
  ];

  it('parks the tool as awaiting_approval and closes the stream without clearing the approval', () => {
    const s = replay(untilApproval);
    expect(s.streaming).toBe(false);
    expect(s.approval).toEqual({ interruptId: 'int-1', requests: [{ name: 'run_python', args: { code: 'print(1)' } }] });
    expect(assistant(s).blocks[0]).toMatchObject({ kind: 'tool', status: 'awaiting_approval' });
  });

  it('approve: the tool runs again, its result lands on the same card, the answer streams into the same turn', () => {
    const s1 = replay(untilApproval);
    const s2 = replay([{ type: 'decided', decisions: ['approve'] }], s1);
    expect(s2.approval).toBeNull();
    expect(s2.streaming).toBe(true);
    expect(assistant(s2).blocks[0]).toMatchObject({ kind: 'tool', status: 'running' });
    const s3 = replay([
      { type: 'tool_result', id: 'tool_0', name: 'run_python', content: '1', is_error: false },
      { type: 'token', text: 'It prints **1**.' },
      done(),
    ], s2);
    expect(s3.turns).toHaveLength(2);
    expect(assistant(s3).blocks).toEqual([
      { kind: 'tool', id: 'tool_0', name: 'run_python', args: { code: 'print(1)' }, status: 'done', result: '1' },
      { kind: 'text', text: 'It prints **1**.' },
    ]);
    expect(s3.streaming).toBe(false);
  });

  it('the stats of a turn add up its runs: the chat that asked for approval plus the resume', () => {
    const s = replay([
      { type: 'decided', decisions: ['approve'] },
      { type: 'tool_result', id: 'tool_0', name: 'run_python', content: '1', is_error: false },
      { type: 'token', text: 'It prints 1.' },
      { type: 'done', status: 'completed', elapsed_s: 1.5, model: 'qwen3-4b', usage_total: { input_tokens: 100, output_tokens: 20, llm_calls: 1 } },
    ], replay(untilApproval));
    expect(assistant(s).stats).toEqual({ elapsed_s: 4, model: 'qwen3-4b', usage: { input_tokens: 300, output_tokens: 50, llm_calls: 3 } });
    // a run without usage keeps the numbers it has
    const s2 = replay([{ type: 'done', status: 'completed', elapsed_s: 0.5, model: 'm2' }], { ...s, streaming: true });
    expect(assistant(s2).stats).toEqual({ elapsed_s: 4.5, model: 'm2', usage: { input_tokens: 300, output_tokens: 50, llm_calls: 3 } });
  });

  it('a resume that fails (409 stale or nothing pending) marks the approved tool instead of leaving it running', () => {
    const s = replay([
      { type: 'decided', decisions: ['approve'] },
      { type: 'stream_failed', message: 'HTTP 409: nothing to resume on this thread', hint: 'Send a message with POST /api/chat' },
    ], replay(untilApproval));
    expect(assistant(s).blocks).toEqual([
      { kind: 'tool', id: 'tool_0', name: 'run_python', args: { code: 'print(1)' }, status: 'error', result: RUN_ENDED },
      { kind: 'error', message: 'HTTP 409: nothing to resume on this thread', hint: 'Send a message with POST /api/chat' },
    ]);
    expect(s.streaming).toBe(false);
    expect(s.approval).toBeNull();
  });

  it('deny: the card stays denied even though the backend reports the rejection as an error result', () => {
    const s = replay([
      { type: 'decided', decisions: ['reject'] },
      { type: 'tool_result', id: 'tool_0', name: 'run_python', content: 'User rejected the tool call', is_error: true },
      { type: 'token', text: 'Not executed.' },
      done(),
    ], replay(untilApproval));
    expect(assistant(s).blocks[0]).toMatchObject({ kind: 'tool', status: 'denied', result: 'User rejected the tool call' });
  });

  it('several requests: only the named tools wait; the k-th decision answers the k-th request', () => {
    const s = replay([
      { type: 'send', text: 'do both' },
      { type: 'tool_call', id: 'tool_0', name: 'list_files', args: { path: '.' } }, // not gated by HITL
      { type: 'tool_call', id: 'tool_1', name: 'run_python', args: { code: 'a' } },
      { type: 'tool_call', id: 'tool_2', name: 'run_python', args: { code: 'b' } },
      { type: 'approval_required', interrupt_id: 'int-2', action_requests: [
        { name: 'run_python', args: { code: 'a' } }, { name: 'run_python', args: { code: 'b' } },
      ] },
      done('awaiting_approval'),
    ]);
    expect(assistant(s).blocks.map((b) => b.kind === 'tool' && b.status)).toEqual(['running', 'awaiting_approval', 'awaiting_approval']);
    const s2 = replay([
      { type: 'decided', decisions: ['approve', 'reject'] },
      { type: 'tool_result', id: 'tool_0', name: 'list_files', content: 'x', is_error: false },
      { type: 'tool_result', id: 'tool_1', name: 'run_python', content: 'ok', is_error: false },
      { type: 'tool_result', id: 'tool_2', name: 'run_python', content: 'User rejected', is_error: true },
      done(),
    ], s);
    expect(assistant(s2).blocks.map((b) => b.kind === 'tool' && b.status)).toEqual(['done', 'done', 'denied']);
  });
});

describe('reducer: errors, stop, reset', () => {
  it('an error event ends the stream with an error block that carries the hint', () => {
    const s = replay([
      { type: 'send', text: 'hi' },
      { type: 'token', text: 'partial' },
      { type: 'error', message: 'mimOE returned 500', hint: 'start a new conversation' },
    ]);
    expect(s.streaming).toBe(false);
    expect(assistant(s).blocks).toEqual([
      { kind: 'text', text: 'partial' },
      { kind: 'error', message: 'mimOE returned 500', hint: 'start a new conversation' },
    ]);
  });

  it('an error event after a tool_call ends the tool card too: nothing will report back', () => {
    const s = replay([
      { type: 'send', text: 'go' },
      { type: 'tool_call', id: 'tool_0', name: 'read_file', args: { path: 'notes.md' } },
      { type: 'error', message: 'mimOE returned 500' },
    ]);
    expect(assistant(s).blocks[0]).toMatchObject({ kind: 'tool', status: 'error', result: RUN_ENDED });
    expect(assistant(s).blocks[1]).toEqual({ kind: 'error', message: 'mimOE returned 500', hint: undefined });
  });

  it('an HTTP failure (409/422) shows as an error block in the turn it belongs to', () => {
    const s = replay([
      { type: 'send', text: 'again' },
      { type: 'stream_failed', message: 'HTTP 409: run in progress on this thread' },
    ]);
    expect(assistant(s).blocks).toEqual([{ kind: 'error', message: 'HTTP 409: run in progress on this thread', hint: undefined }]);
    expect(s.streaming).toBe(false);
  });

  it('stop marks running tools, adds a notice and stops streaming; reset starts a fresh thread', () => {
    const s = replay([
      { type: 'send', text: 'go' },
      { type: 'tool_call', id: 'tool_0', name: 'search_files', args: { pattern: 'TODO' } },
      { type: 'stopped' },
    ]);
    expect(assistant(s).blocks).toEqual([
      { kind: 'tool', id: 'tool_0', name: 'search_files', args: { pattern: 'TODO' }, status: 'error', result: 'stopped' },
      { kind: 'notice', text: 'Stopped.' },
    ]);
    expect(s.streaming).toBe(false);
    expect(reducer(s, { type: 'stopped' })).toBe(s); // idempotent when nothing streams
    expect(reducer(s, { type: 'reset', threadId: 't2' })).toEqual(initialState('t2'));
  });

  it('load replaces the conversation with a saved one, its pending approval included', () => {
    const streaming = replay([{ type: 'send', text: 'first' }]);
    const turns = [
      { role: 'user' as const, text: 'Use run_python to print 6*7' },
      {
        role: 'assistant' as const,
        blocks: [{ kind: 'tool' as const, id: 'tool_0', name: 'run_python', args: { code: 'print(6*7)' }, status: 'awaiting_approval' as const }],
        stats: { model: null, usage: { input_tokens: 10, output_tokens: 5, llm_calls: 1 } },
      },
    ];
    const approval = { interruptId: 'abc', requests: [{ name: 'run_python', args: { code: 'print(6*7)' } }] };
    const s = reducer(streaming, { type: 'load', threadId: 'saved', turns, approval });
    expect(s).toEqual({ threadId: 'saved', turns, streaming: false, approval });
    // the resume's events continue the restored turn: the result lands on the restored call
    const resumed = replay([
      { type: 'decided', decisions: ['approve'] },
      { type: 'tool_result', id: 'tool_0', name: 'run_python', content: '42', is_error: false },
      done(),
    ], s);
    expect(assistant(resumed).blocks).toEqual([{ ...turns[1].blocks![0], status: 'done', result: '42' }]);
    expect(assistant(resumed).stats?.elapsed_s).toBe(2.5); // restored stats have no elapsed time
    expect(assistant(resumed).stats?.usage?.llm_calls).toBe(3);
  });
});
