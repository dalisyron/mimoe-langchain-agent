import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { ToolCard } from './components/ToolCard';
import { RUN_ENDED, type Block, type ToolBlock } from './reducer';
import { isRetrying, phrase, stepLabel } from './steps';

const PRIMES = 'sum(1 for n in range(400, 6601) if all(n % i != 0 for i in range(2, int(n**0.5) + 1)))';
const REFUSED =
  'ERROR: the calculator evaluates one arithmetic expression; it cannot run a generator expression (loops, ' +
  'comprehensions and conditions are Python code). For counting, loops or anything beyond one arithmetic ' +
  'expression, call run_python with code that prints the result.';
const step = (over: Partial<ToolBlock> = {}): ToolBlock => ({
  kind: 'tool', id: 'tool_0', name: 'calculator', args: { expression: PRIMES }, status: 'running', ...over,
});

describe('tool steps: one plain line per call', () => {
  it('names what each tool does, while it runs and after', () => {
    expect(phrase('run_python', { code: 'print(1)' })).toEqual({ doing: 'Running Python', did: 'Ran Python' });
    expect(phrase('calculator', { expression: '1 + 1' })).toEqual({ doing: 'Calculating', did: 'Calculated' });
    expect(phrase('list_files', { path: '.', max_depth: 4 }).doing).toBe('Listing files');
    expect(phrase('list_files', { path: 'data' }).did).toBe('Listed files in data');
    expect(phrase('read_file', { path: 'notes.md', offset: 1 }).doing).toBe('Reading notes.md');
    expect(phrase('search_files', { pattern: 'TODO', glob: '*.md' }).did).toBe('Searched files for “TODO”');
    expect(phrase('now', {}).doing).toBe('Checking the time');
    expect(phrase('mimoe_status', {}).did).toBe('Checked the mimOE engine');
    expect(phrase('git', { command: 'log' }).doing).toBe('Checking git log');
    expect(phrase('git', {}).doing).toBe('Checking git');
    expect(phrase('fetch_url', { url: 'http://example.com' })).toEqual({ doing: 'Using fetch_url', did: 'Used fetch_url' });
  });

  it('shows model-written arguments short, on one line, with hidden characters escaped', () => {
    expect(phrase('read_file', { path: 'a\n  b.md' }).doing).toBe('Reading a b.md');
    expect(phrase('read_file', { path: 'x'.repeat(80) }).doing).toBe(`Reading ${'x'.repeat(47)}…`);
    expect(phrase('read_file', { path: 'evil‮txt.md' }).doing).toBe('Reading evil\\u202etxt.md');
    expect(phrase('read_file', { path: 42 }).doing).toBe('Reading a file');
  });

  it('says a failed call failed and that the agent is trying another way, not what the error was', () => {
    const failed = step({ status: 'error', result: REFUSED });
    expect(stepLabel(failed, true)).toBe('Calculating failed, trying a different approach');
    expect(stepLabel(failed, false)).toBe('Calculating failed');
    expect(stepLabel(step(), false)).toBe('Calculating…');
    expect(stepLabel(step({ status: 'done', result: '775' }), false)).toBe('Calculated');
  });

  it('covers approval, denial and a run that ended before the tool answered', () => {
    const py = step({ name: 'run_python', args: { code: 'print(1)' } });
    expect(stepLabel({ ...py, status: 'awaiting_approval' }, false)).toBe('Waiting for your approval to run this code');
    expect(stepLabel({ ...py, status: 'denied', result: 'User rejected the tool call' }, false)).toBe('You declined; the code did not run');
    expect(stepLabel({ ...py, status: 'error', result: RUN_ENDED }, true)).toBe('Running Python stopped before it finished');
    expect(stepLabel({ ...py, status: 'error', result: 'stopped' }, false)).toBe('Running Python stopped before it finished');
  });

  it('treats a failed call as retried while the turn runs or once another tool call followed', () => {
    const failed = step({ status: 'error', result: REFUSED });
    const text: Block = { kind: 'text', text: 'I will count them with Python.' };
    const next = step({ id: 'tool_1', name: 'run_python', args: { code: 'print(775)' }, status: 'awaiting_approval' });
    expect(isRetrying([failed, text], 0, true)).toBe(true);
    expect(isRetrying([failed, text], 0, false)).toBe(false);
    expect(isRetrying([failed, text, next], 0, false)).toBe(true);
  });

  it('renders the line collapsed; the arguments and the error open on click', () => {
    const html = renderToStaticMarkup(createElement(ToolCard, { block: step({ status: 'error', result: REFUSED }), retrying: true }));
    expect(html).toContain('<details class="step step-error"><summary>Calculating failed, trying a different approach</summary>');
    expect(html).not.toContain('open=');
    expect(html).toContain('ERROR: the calculator evaluates one arithmetic expression'); // one click away
  });
});
