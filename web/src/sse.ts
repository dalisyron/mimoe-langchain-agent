/** Minimal text/event-stream parser over fetch's ReadableStream (EventSource cannot POST a body). */
export type SseMessage = { event: string; data: string };

/** One message per frame. Handles CRLF/LF/CR line ends (sse-starlette writes CRLF), a CRLF split across
 * chunks, multi-line `data:` (joined with "\n"), `: ping` comments and a stream closing without a blank line. */
export async function* parseSSE(body: ReadableStream<Uint8Array>): AsyncGenerator<SseMessage> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let pendingLF = false; // the last chunk ended with "\r": swallow a "\n" that starts the next one
  let event = '';
  let data: string[] = [];

  const flush = (): SseMessage | null => {
    const msg = data.length ? { event: event || 'message', data: data.join('\n') } : null;
    event = '';
    data = [];
    return msg;
  };
  const handleLine = (line: string): SseMessage | null => {
    if (line === '') return flush(); // blank line ends a frame
    if (line.startsWith(':')) return null; // comment / keep-alive
    const i = line.indexOf(':');
    const field = i < 0 ? line : line.slice(0, i);
    let value = i < 0 ? '' : line.slice(i + 1);
    if (value.startsWith(' ')) value = value.slice(1); // the spec strips one leading space
    if (field === 'event') event = value;
    else if (field === 'data') data.push(value);
    return null; // `id`, `retry` and unknown fields are ignored
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      let text = done ? decoder.decode() : decoder.decode(value, { stream: true });
      if (pendingLF && text.startsWith('\n')) text = text.slice(1);
      pendingLF = text.endsWith('\r');
      buffer += text;
      const lines = buffer.split(/\r\n|\r|\n/);
      buffer = done ? '' : (lines.pop() ?? ''); // keep the trailing partial line
      if (done && lines[lines.length - 1] === '') lines.pop();
      for (const line of lines) {
        const msg = handleLine(line);
        if (msg) yield msg;
      }
      if (done) {
        const msg = flush(); // closed without a final blank line
        if (msg) yield msg;
        return;
      }
    }
  } finally {
    reader.releaseLock();
  }
}
