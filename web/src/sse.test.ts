import { describe, expect, it } from 'vitest';
import { parseSSE } from './sse';

const streamOf = (chunks: (string | Uint8Array)[]) =>
  new ReadableStream<Uint8Array>({
    start(c) {
      for (const ch of chunks) c.enqueue(typeof ch === 'string' ? new TextEncoder().encode(ch) : ch);
      c.close();
    },
  });
const collect = async (chunks: (string | Uint8Array)[]) => {
  const out = [];
  for await (const m of parseSSE(streamOf(chunks))) out.push(m);
  return out;
};

describe('parseSSE', () => {
  it('handles CRLF frames, comments, multi-line data and chunk boundaries (incl. a split CRLF)', async () => {
    const out = await collect([
      ': ping - 2026-09-25\r\n\r\nevent: token\r\ndata: {"text": "a"}\r', // chunk ends with "\r" ...
      '\n\r\nevent: token\r\ndata: line1\r\ndata: line2\r\n\r\n', // ... and "\n" completes the pair
      'data: {"tail": tr', 'ue}\n\n', // a JSON payload split across chunks, LF framing
      'event: done\ndata: {}', // no trailing blank line
    ]);
    expect(out).toEqual([
      { event: 'token', data: '{"text": "a"}' },
      { event: 'token', data: 'line1\nline2' },
      { event: 'message', data: '{"tail": true}' },
      { event: 'done', data: '{}' },
    ]);
  });

  it('does not leave a "\\r" in the event name (a naive split on "\\n" would)', async () => {
    const out = await collect(['event: thinking\r\ndata: {"text": "x"}\r\n\r\n']);
    expect(out[0].event).toBe('thinking');
  });

  it('decodes a multi-byte character split across two chunks', async () => {
    const bytes = new TextEncoder().encode('event: token\ndata: café\n\n');
    const cut = bytes.indexOf(0xc3) + 1; // inside the two-byte "é"
    const out = await collect([bytes.slice(0, cut), bytes.slice(cut)]);
    expect(out).toEqual([{ event: 'token', data: 'café' }]);
  });

  it('ignores keep-alive comments, retry/id fields and frames without data; a bare CR ends a line', async () => {
    const out = await collect([': ping\n\nevent: nothing\n\nid: 7\nretry: 1000\rdata: x\r\r']);
    expect(out).toEqual([{ event: 'message', data: 'x' }]);
  });

  it('strips one leading space of a value and keeps the rest verbatim', async () => {
    const out = await collect(['data:  two spaces\n\ndata:none\n\n']);
    expect(out.map((m) => m.data)).toEqual([' two spaces', 'none']);
  });
});
