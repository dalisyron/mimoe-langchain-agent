/**
 * Characters that hide or reorder code: C0/C1 controls, zero-width characters, bidirectional
 * overrides and line separators. Model-written code containing them can look different from
 * what runs ("Trojan Source"), so the approval panel shows them escaped and warns.
 */
const HIDDEN = /[\u0000-\u0008\u000b-\u001f\u007f-\u009f\u200b-\u200f\u2028\u2029\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff]/g;

/** `text` with every hidden character written as a visible escape, and how many there were. */
export function revealHidden(text: string): { text: string; count: number } {
  let count = 0;
  const shown = text.replace(/\r\n/g, '\n').replace(HIDDEN, (ch) => {
    count += 1;
    const code = ch.codePointAt(0) ?? 0;
    return code < 0x100 ? `\\x${code.toString(16).padStart(2, '0')}` : `\\u${code.toString(16).padStart(4, '0')}`;
  });
  return { text: shown, count };
}
