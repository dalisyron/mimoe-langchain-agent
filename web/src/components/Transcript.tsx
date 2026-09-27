import { CircleAlert, Info } from 'lucide-react';
import type { AssistantTurn, Stats, Turn } from '../reducer';
import { isRetrying } from '../steps';
import { Logo } from './Logo';
import { Markdown } from './Markdown';
import { Thinking } from './Thinking';
import { ToolCard } from './ToolCard';

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? '' : 's'}`;

export function Transcript({ turns, streaming }: { turns: Turn[]; streaming: boolean }) {
  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-8 px-4 py-6">
      {turns.map((turn, i) =>
        turn.role === 'user'
          ? <UserMessage key={i} text={turn.text} />
          : <AssistantMessage key={i} turn={turn} live={streaming && i === turns.length - 1} />,
      )}
    </div>
  );
}

function UserMessage({ text }: { text: string }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[85%] whitespace-pre-wrap break-words rounded-3xl bg-zinc-100 px-4 py-2.5 leading-relaxed dark:bg-zinc-800">
        {text}
      </div>
    </div>
  );
}

function AssistantMessage({ turn, live }: { turn: AssistantTurn; live: boolean }) {
  const last = turn.blocks.length - 1;
  // Dots while nothing visibly moves: before the first token and while a tool runs.
  const waiting = live && (last < 0 || turn.blocks[last].kind === 'tool');
  return (
    <div className="flex gap-3.5">
      <Logo className="mt-0.5 size-7 rounded-full text-[13px]" />
      <div className="flex min-w-0 flex-1 flex-col gap-3 pt-0.5">
        {turn.blocks.map((b, j) => {
          switch (b.kind) {
            case 'thinking': return <Thinking key={j} text={b.text} live={live && j === last} />;
            case 'text': return <Markdown key={j} text={b.text} />;
            case 'tool': return <ToolCard key={j} block={b} retrying={isRetrying(turn.blocks, j, live)} />;
            case 'notice':
              return (
                <div key={j} className="flex items-start gap-2 rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-200">
                  <Info className="mt-0.5 size-4 shrink-0" />{b.text}
                </div>
              );
            case 'error':
              return (
                <div key={j} role="alert" className="flex items-start gap-2 rounded-xl border border-red-200 bg-red-50 px-3 py-2.5 text-sm text-red-800 dark:border-red-500/30 dark:bg-red-500/10 dark:text-red-200">
                  <CircleAlert className="mt-0.5 size-4 shrink-0" />
                  <div className="min-w-0 whitespace-pre-wrap break-words">
                    {b.message}
                    {b.hint && <div className="mt-1 text-red-700/80 dark:text-red-300/80">{b.hint}</div>}
                  </div>
                </div>
              );
          }
        })}
        {waiting && <TypingDots />}
        {turn.stats && <StatsLine stats={turn.stats} />}
      </div>
    </div>
  );
}

function TypingDots() {
  return (
    <div className="flex h-6 items-center gap-1" aria-label="The agent is working">
      {[0, 150, 300].map((delay) => (
        <span key={delay} className="size-1.5 animate-bounce rounded-full bg-zinc-400 dark:bg-zinc-500" style={{ animationDelay: `${delay}ms` }} />
      ))}
    </div>
  );
}

function StatsLine({ stats }: { stats: Stats }) {
  const parts = [
    stats.usage && plural(stats.usage.llm_calls, 'model call'),
    stats.usage && plural(stats.usage.output_tokens, 'token'),
    stats.elapsed_s !== undefined && `${stats.elapsed_s.toFixed(1)} s`,
    stats.model,
  ].filter(Boolean);
  return <div className="stats text-xs text-zinc-400 dark:text-zinc-500">{parts.join(' · ')}</div>;
}
