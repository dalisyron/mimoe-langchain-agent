import { describe, expect, it } from 'vitest';
import { HttpError } from './api';
import type { Action, ServerEvent } from './events';
import { consumeStream } from './useChat';

const DONE: ServerEvent = { type: 'done', status: 'completed', elapsed_s: 1, model: 'm' };
async function* events(list: ServerEvent[], fail?: Error): AsyncGenerator<ServerEvent> {
  for (const ev of list) yield ev;
  if (fail) throw fail;
}
const run = async (list: ServerEvent[], fail?: Error, controller = new AbortController()) => {
  const out: Action[] = [];
  await consumeStream(events(list, fail), (a) => out.push(a), controller.signal);
  return out;
};

describe('consumeStream', () => {
  it('dispatches every event of a run that ends with done, and nothing else', async () => {
    const list: ServerEvent[] = [{ type: 'token', text: 'hi' }, DONE];
    expect(await run(list)).toEqual(list);
  });

  it('an error event is a valid end of stream (no stream_failed after it)', async () => {
    const list: ServerEvent[] = [{ type: 'token', text: 'partial' }, { type: 'error', message: 'boom', hint: 'h' }];
    expect(await run(list)).toEqual(list);
  });

  it('a stream that closes without done or error is reported, so the composer is re-enabled', async () => {
    const out = await run([{ type: 'token', text: 'partial' }]);
    expect(out[1]).toMatchObject({ type: 'stream_failed', message: expect.stringContaining('closed') });
  });

  it('an HTTP failure carries its status and hint; a network failure gets the serve hint', async () => {
    const [http] = await run([], new HttpError(409, 'run in progress', 'wait'));
    expect(http).toEqual({ type: 'stream_failed', message: 'HTTP 409: run in progress', hint: 'wait' });
    const [net] = await run([], new TypeError('Failed to fetch'));
    expect(net).toMatchObject({ type: 'stream_failed', message: 'Failed to fetch', hint: expect.stringContaining('serve') });
  });

  it('after an abort nothing is dispatched: not the AbortError, not events already buffered', async () => {
    const c = new AbortController();
    const out: Action[] = [];
    async function* aborting(): AsyncGenerator<ServerEvent> {
      yield { type: 'token', text: 'a' };
      c.abort(); // Stop was pressed while a chunk with several frames was still being replayed
      yield { type: 'token', text: 'b' };
      throw Object.assign(new Error('aborted'), { name: 'AbortError' });
    }
    await consumeStream(aborting(), (a) => out.push(a), c.signal);
    expect(out).toEqual([{ type: 'token', text: 'a' }]);
    expect(await run([{ type: 'token', text: 'x' }], undefined, c)).toEqual([]); // and no "closed early" either
  });
});
