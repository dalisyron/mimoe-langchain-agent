import { Brain, ChevronRight } from 'lucide-react';
import { cn } from '../lib/utils';

/** The model's reasoning, collapsed by default (it can appear anywhere in a turn, also after a tool result). */
export function Thinking({ text, live }: { text: string; live: boolean }) {
  return (
    <details className="disclosure group/think">
      <summary className="flex w-fit cursor-pointer select-none items-center gap-2 text-sm text-zinc-500 transition-colors hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100 [&_svg]:size-3.5">
        <Brain className={cn(live && 'animate-pulse')} />
        {live ? 'Thinking…' : 'Thought process'}
        <ChevronRight className="text-zinc-400 transition-transform group-open/think:rotate-90" />
      </summary>
      <div className="ml-[7px] mt-2 whitespace-pre-wrap border-l-2 border-zinc-200 pl-4 text-sm leading-relaxed text-zinc-500 dark:border-zinc-800 dark:text-zinc-400">
        {text}
      </div>
    </details>
  );
}
