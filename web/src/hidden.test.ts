import { describe, expect, it } from 'vitest';
import { revealHidden } from './hidden';

describe('revealHidden', () => {
  it('escapes terminal controls, bidi overrides and zero-width characters', () => {
    const { text, count } = revealHidden('print(1)\u001b[2K\n# \u202eevil\u200b');
    expect(text).toBe('print(1)\\x1b[2K\n# \\u202eevil\\u200b');
    expect(count).toBe(3);
  });

  it('leaves ordinary code, tabs and newlines alone', () => {
    const code = 'for i in range(3):\n\tprint("ok \u00e9 \u2713")\r\n';
    expect(revealHidden(code)).toEqual({ text: 'for i in range(3):\n\tprint("ok \u00e9 \u2713")\n', count: 0 });
  });
});
