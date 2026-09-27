import { ArrowUp, Square } from 'lucide-react';
import { useLayoutEffect, useRef, useState, type KeyboardEvent } from 'react';
import { Button } from './ui/button';

const MAX_HEIGHT = 208; // px: about eight lines, then the textarea scrolls

/** Enter sends, Shift+Enter adds a line; the box grows with the text; Stop aborts the open stream. */
export function Composer({ disabled, streaming, placeholder, onSend, onStop }: {
  disabled: boolean;
  streaming: boolean;
  placeholder: string;
  onSend: (text: string) => void;
  onStop: () => void;
}) {
  const [text, setText] = useState('');
  const box = useRef<HTMLTextAreaElement>(null);
  useLayoutEffect(() => {
    const el = box.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, MAX_HEIGHT)}px`;
  }, [text]);

  const submit = () => {
    if (text.trim() && !disabled) { onSend(text.trim()); setText(''); }
  };
  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); submit(); }
  };
  return (
    <form
      className="flex items-end rounded-[26px] border border-zinc-300 bg-white shadow-sm transition-colors focus-within:border-zinc-400 dark:border-zinc-700 dark:bg-zinc-900 dark:focus-within:border-zinc-500"
      onSubmit={(e) => { e.preventDefault(); submit(); }}
    >
      <textarea
        ref={box}
        rows={1}
        value={text}
        placeholder={placeholder}
        aria-label="Message"
        autoFocus
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKey}
        disabled={disabled}
        className="min-h-[52px] flex-1 resize-none bg-transparent py-3.5 pl-5 pr-2 text-[15px] leading-6 outline-none placeholder:text-zinc-400 disabled:cursor-not-allowed dark:placeholder:text-zinc-500"
      />
      <div className="p-2">
        {streaming
          ? <Button variant="primary" size="icon" className="rounded-full" onClick={onStop} aria-label="Stop"><Square className="size-3.5 fill-current" /></Button>
          : <Button type="submit" variant="primary" size="icon" className="rounded-full" disabled={disabled || !text.trim()} aria-label="Send"><ArrowUp /></Button>}
      </div>
    </form>
  );
}
