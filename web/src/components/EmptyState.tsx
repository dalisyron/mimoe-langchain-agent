import { Cpu, FileText, FolderTree, ListTodo, Sheet } from 'lucide-react';
import { cn } from '../lib/utils';
import { Logo } from './Logo';

/** The demo prompts of workspace/README.md, as cards. */
const SUGGESTIONS = [
  { icon: FolderTree, title: 'Look around', prompt: 'What files are in this workspace?' },
  { icon: FileText, title: 'Summarize a file', prompt: 'Summarize notes.md' },
  { icon: Sheet, title: 'Crunch the numbers', prompt: 'Use run_python to count the data rows of sales.csv and sum its revenue column.' },
  { icon: ListTodo, title: 'Find loose ends', prompt: 'Find every TODO in this workspace and tell me where they are.' },
  { icon: Cpu, title: 'Check the engine', prompt: 'What model am I talking to, and how fast is it?' },
];

const folderName = (path: string) => path.split(/[\\/]/).filter(Boolean).pop() ?? path;

export function EmptyState({ workspace, model, disabled, onPrompt }: {
  workspace: string | null;
  model: string | null;
  disabled: boolean;
  onPrompt: (text: string) => void;
}) {
  return (
    <div className="mx-auto flex min-h-full w-full max-w-2xl flex-col items-center justify-center px-4 py-10">
      <Logo className="size-11 rounded-xl text-2xl" />
      <h2 className="mt-5 text-center text-2xl font-semibold tracking-tight sm:text-3xl">What can I help with?</h2>
      <p className="mt-2 max-w-lg text-center text-sm leading-relaxed text-zinc-500 dark:text-zinc-400">
        Ask about the files in{' '}
        <span className="font-medium text-zinc-700 dark:text-zinc-300" title={workspace ?? undefined}>
          {workspace ? folderName(workspace) : 'the workspace'}
        </span>
        . {model ? <>Answers come from <span className="whitespace-nowrap font-medium text-zinc-700 dark:text-zinc-300">{model}</span>, running on this computer.</> : 'Everything runs on this computer.'}
      </p>
      <div className="mt-8 grid w-full gap-3 sm:grid-cols-2">
        {SUGGESTIONS.map(({ icon: Icon, title, prompt }, i) => (
          <button
            key={prompt}
            type="button"
            disabled={disabled}
            onClick={() => onPrompt(prompt)}
            className={cn(
              'group rounded-2xl border border-zinc-200 p-4 text-left transition-colors hover:border-zinc-300 hover:bg-zinc-50',
              'outline-none focus-visible:ring-2 focus-visible:ring-zinc-400 disabled:pointer-events-none disabled:opacity-50',
              'dark:border-zinc-800 dark:hover:border-zinc-700 dark:hover:bg-zinc-900',
              i === SUGGESTIONS.length - 1 && 'sm:col-span-2',
            )}
          >
            <Icon className="size-5 text-zinc-400 transition-colors group-hover:text-zinc-700 dark:group-hover:text-zinc-300" />
            <div className="mt-2.5 text-sm font-medium">{title}</div>
            <div className="mt-0.5 text-sm text-zinc-500 dark:text-zinc-400">{prompt}</div>
          </button>
        ))}
      </div>
    </div>
  );
}
