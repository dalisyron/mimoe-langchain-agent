import { useEffect, useRef } from 'react';
import type { AssistantTurn, Turn } from '../reducer';
import { Markdown } from './Markdown';
import { Thinking } from './Thinking';
import { ToolCard } from './ToolCard';

const DEMO_PROMPTS = [
  'What files are in this workspace?',
  'Summarize notes.md',
  'Use run_python to count the data rows of sales.csv and sum its revenue column.',
  'Find every TODO in this workspace and tell me where they are.',
  'What model am I talking to, and how fast is it?',
];

export function Transcript({ turns, streaming, disabled, onPrompt }: {
  turns: Turn[];
  streaming: boolean;
  disabled: boolean; // a model switch is in flight
  onPrompt: (text: string) => void;
}) {
  const stick = useRef(true); // follow the stream unless the reader scrolled up
  useEffect(() => {
    const onScroll = () => { stick.current = window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 80; };
    window.addEventListener('scroll', onScroll);
    return () => window.removeEventListener('scroll', onScroll);
  }, []);
  // scrollTo the document end, not scrollIntoView on a marker: the sticky footer would cover the last lines.
  useEffect(() => { if (stick.current) window.scrollTo({ top: document.documentElement.scrollHeight }); }, [turns]);

  return (
    <main className="transcript">
      {turns.length === 0 && (
        <div className="empty">
          <p>Ask about the sample workspace. Try one of these:</p>
          <ul>{DEMO_PROMPTS.map((p) => <li key={p}><button type="button" className="link" disabled={disabled} onClick={() => onPrompt(p)}>{p}</button></li>)}</ul>
        </div>
      )}
      {turns.map((turn, i) => turn.role === 'user'
        ? <div key={i} className="turn user">{turn.text}</div>
        : <Assistant key={i} turn={turn} live={streaming && i === turns.length - 1} />)}
    </main>
  );
}

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? '' : 's'}`;

function Assistant({ turn, live }: { turn: AssistantTurn; live: boolean }) {
  const last = turn.blocks.length - 1;
  const s = turn.stats;
  return (
    <div className="turn assistant">
      {turn.blocks.map((b, j) => {
        switch (b.kind) {
          case 'thinking': return <Thinking key={j} text={b.text} live={live && j === last} />;
          case 'text': return <Markdown key={j} text={b.text} />;
          case 'tool': return <ToolCard key={j} block={b} />;
          case 'notice': return <div key={j} className="notice">{b.text}</div>;
          case 'error': return <div key={j} className="error">{b.message}{b.hint && <div className="hint">{b.hint}</div>}</div>;
        }
      })}
      {live && <span className="cursor" />}
      {s && (
        <div className="stats">
          {s.usage && `${plural(s.usage.llm_calls, 'model call')} · ${plural(s.usage.output_tokens, 'token')} · `}{s.elapsed_s.toFixed(1)} s · {s.model}
        </div>
      )}
    </div>
  );
}
