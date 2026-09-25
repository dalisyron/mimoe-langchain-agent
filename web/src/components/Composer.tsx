import { useState, type KeyboardEvent } from 'react';

/** Textarea: Enter sends, Shift+Enter adds a line; Stop aborts the open stream. */
export function Composer({ disabled, streaming, onSend, onStop }: {
  disabled: boolean;
  streaming: boolean;
  onSend: (text: string) => void;
  onStop: () => void;
}) {
  const [text, setText] = useState('');
  const submit = () => {
    if (text.trim() && !disabled) { onSend(text.trim()); setText(''); }
  };
  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); submit(); }
  };
  return (
    <footer>
      <form className="composer" onSubmit={(e) => { e.preventDefault(); submit(); }}>
        <textarea
          rows={2}
          value={text}
          placeholder={disabled ? 'Waiting for the agent…' : 'Ask about the workspace… (Enter to send, Shift+Enter for a new line)'}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKey}
          disabled={disabled}
        />
        {streaming
          ? <button type="button" onClick={onStop}>Stop</button>
          : <button type="submit" className="primary" disabled={disabled || !text.trim()}>Send</button>}
      </form>
    </footer>
  );
}
