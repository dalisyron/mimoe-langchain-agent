/** The model's reasoning, collapsed by default (it can appear anywhere in a turn, also after a tool result). */
export function Thinking({ text, live }: { text: string; live: boolean }) {
  return (
    <details className="thinking">
      <summary>{live ? 'Thinking…' : 'Thought'} ({text.length} chars)</summary>
      <pre>{text}</pre>
    </details>
  );
}
