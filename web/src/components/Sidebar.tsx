import { FolderOpen, Monitor, Moon, MoreHorizontal, PanelLeftClose, Pencil, Search, Server, SquarePen, Sun, Trash2 } from 'lucide-react';
import { useMemo, useRef, useState, type FormEvent } from 'react';
import type { Health } from '../events';
import { cn } from '../lib/utils';
import type { ThemeChoice } from '../theme';
import { filterThreads, groupByDate, type ThreadSummary } from '../threads';
import { Logo } from './Logo';
import { Button } from './ui/button';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from './ui/menu';

export function Sidebar({ open, threads, error, currentId, health, theme, onTheme, onClose, onNew, onSelect, onRename, onDelete }: {
  open: boolean;
  threads: ThreadSummary[] | null; // null while the first list loads
  error: string | null;
  currentId: string;
  health: Health | null | undefined;
  theme: ThemeChoice;
  onTheme: (choice: ThemeChoice) => void;
  onClose: () => void;
  onNew: () => void;
  onSelect: (id: string) => void;
  onRename: (id: string, title: string) => Promise<void>;
  onDelete: (id: string) => Promise<void>;
}) {
  const [query, setQuery] = useState('');
  const groups = useMemo(() => groupByDate(filterThreads(threads ?? [], query)), [threads, query]);
  return (
    <>
      <aside
        aria-label="Conversations"
        className={cn(
          'fixed inset-y-0 left-0 z-40 flex w-72 flex-col border-r border-zinc-200 bg-zinc-50 transition-transform duration-200 dark:border-zinc-800 dark:bg-zinc-900',
          'md:static md:z-auto md:transition-none',
          open ? 'translate-x-0' : '-translate-x-full md:hidden',
        )}
      >
        <div className="flex h-14 shrink-0 items-center justify-between pl-4 pr-2">
          <div className="flex items-center gap-2.5 font-semibold tracking-tight">
            <Logo />
            mimoe-agent
          </div>
          <Button variant="ghost" size="iconSm" onClick={onClose} aria-label="Close sidebar" title="Close sidebar">
            <PanelLeftClose />
          </Button>
        </div>
        <div className="space-y-2 px-3 pb-2">
          <Button variant="outline" className="w-full justify-start rounded-xl" onClick={onNew}>
            <SquarePen /> New chat
          </Button>
          <label className="relative block">
            <span className="sr-only">Search conversations</span>
            <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-zinc-400" />
            <input
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search conversations"
              className="h-9 w-full rounded-xl border border-transparent bg-zinc-200/60 pl-9 pr-3 text-sm outline-none placeholder:text-zinc-500 focus:border-zinc-300 focus:bg-white dark:bg-zinc-800/70 dark:focus:border-zinc-700 dark:focus:bg-zinc-950"
            />
          </label>
        </div>
        <nav className="min-h-0 flex-1 overflow-y-auto px-2 pb-3" aria-label="Conversation history">
          {threads === null && !error && <p className="px-3 py-2 text-sm text-zinc-500">Loading…</p>}
          {error && <p className="px-3 py-2 text-sm text-red-600 dark:text-red-400">Could not load conversations: {error}</p>}
          {threads?.length === 0 && <p className="px-3 py-2 text-sm text-zinc-500">No conversations yet.</p>}
          {threads && threads.length > 0 && groups.length === 0 && <p className="px-3 py-2 text-sm text-zinc-500">No matches.</p>}
          {groups.map((group) => (
            <section key={group.label}>
              <h2 className="px-3 pb-1 pt-4 text-xs font-medium text-zinc-500 dark:text-zinc-400">{group.label}</h2>
              <ul className="space-y-0.5">
                {group.threads.map((thread) => (
                  <ConversationItem
                    key={thread.id}
                    thread={thread}
                    active={thread.id === currentId}
                    onSelect={onSelect}
                    onRename={onRename}
                    onDelete={onDelete}
                  />
                ))}
              </ul>
            </section>
          ))}
        </nav>
        <footer className="shrink-0 space-y-3 border-t border-zinc-200 p-3 dark:border-zinc-800">
          <EngineInfo health={health} />
          <ThemeSwitch theme={theme} onTheme={onTheme} />
        </footer>
      </aside>
      {open && <div className="fixed inset-0 z-30 bg-zinc-950/40 md:hidden" onClick={onClose} aria-hidden />}
    </>
  );
}

function ConversationItem({ thread, active, onSelect, onRename, onDelete }: {
  thread: ThreadSummary;
  active: boolean;
  onSelect: (id: string) => void;
  onRename: (id: string, title: string) => Promise<void>;
  onDelete: (id: string) => Promise<void>;
}) {
  const [mode, setMode] = useState<'view' | 'rename' | 'confirm'>('view');
  const [draft, setDraft] = useState(thread.title);
  const [problem, setProblem] = useState<string | null>(null);
  const editing = useRef(false); // Enter and blur both save; Escape cancels: exactly one of them acts
  const attempt = async (action: () => Promise<void>) => {
    try { await action(); setMode('view'); setProblem(null); } catch (e) { setProblem(e instanceof Error ? e.message : String(e)); }
  };
  const save = (e?: FormEvent) => {
    e?.preventDefault();
    if (!editing.current) return;
    editing.current = false;
    const title = draft.trim();
    if (!title || title === thread.title) { setMode('view'); return; }
    onRename(thread.id, title).then(
      () => { setMode('view'); setProblem(null); },
      (err: unknown) => { editing.current = true; setProblem(err instanceof Error ? err.message : String(err)); },
    );
  };

  if (mode === 'rename') {
    return (
      <li>
        <form onSubmit={save}>
          <input
            autoFocus
            value={draft}
            maxLength={200}
            aria-label="Conversation title"
            onChange={(e) => setDraft(e.target.value)}
            onBlur={() => save()}
            onKeyDown={(e) => { if (e.key === 'Escape') { editing.current = false; setDraft(thread.title); setMode('view'); } }}
            className="h-9 w-full rounded-lg border border-zinc-300 bg-white px-2.5 text-sm outline-none focus:border-zinc-400 dark:border-zinc-700 dark:bg-zinc-950"
          />
        </form>
        {problem && <p className="px-1 pt-1 text-xs text-red-700 dark:text-red-300">{problem}</p>}
      </li>
    );
  }
  if (mode === 'confirm') {
    return (
      <li className="rounded-lg border border-red-200 bg-red-50 px-3 py-2.5 text-sm dark:border-red-500/30 dark:bg-red-500/10">
        <p className="line-clamp-2">Delete “{thread.title}”?</p>
        <p className="mt-0.5 text-xs text-zinc-500 dark:text-zinc-400">The conversation and its history are removed from this computer.</p>
        {problem && <p className="mt-1 text-xs text-red-700 dark:text-red-300">{problem}</p>}
        <div className="mt-2 flex gap-2">
          <Button size="sm" variant="danger" onClick={() => void attempt(() => onDelete(thread.id))}>Delete</Button>
          <Button size="sm" variant="ghost" onClick={() => { setMode('view'); setProblem(null); }}>Cancel</Button>
        </div>
      </li>
    );
  }
  return (
    <li className="group relative">
      <button
        type="button"
        onClick={() => onSelect(thread.id)}
        aria-current={active ? 'page' : undefined}
        title={thread.title}
        className={cn(
          'flex h-9 w-full items-center rounded-lg px-3 text-left text-sm outline-none transition-colors focus-visible:ring-2 focus-visible:ring-zinc-400',
          active ? 'bg-zinc-200/80 font-medium dark:bg-zinc-800' : 'text-zinc-700 hover:bg-zinc-200/50 dark:text-zinc-300 dark:hover:bg-zinc-800/60',
        )}
      >
        <span className="truncate pr-6">{thread.title}</span>
      </button>
      <DropdownMenu>
        <DropdownMenuTrigger
          aria-label={`Options for ${thread.title}`}
          className={cn(
            'absolute right-1 top-1/2 flex size-7 -translate-y-1/2 items-center justify-center rounded-md text-zinc-500 outline-none transition-opacity',
            'hover:bg-zinc-300/60 hover:text-zinc-900 focus-visible:opacity-100 focus-visible:ring-2 focus-visible:ring-zinc-400 data-[state=open]:opacity-100',
            'dark:hover:bg-zinc-700 dark:hover:text-zinc-100',
            active ? 'opacity-100' : 'opacity-0 group-hover:opacity-100',
          )}
        >
          <MoreHorizontal className="size-4" />
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start">
          <DropdownMenuItem onSelect={() => { editing.current = true; setDraft(thread.title); setMode('rename'); }}><Pencil /> Rename</DropdownMenuItem>
          <DropdownMenuItem className="text-red-600 dark:text-red-400" onSelect={() => setMode('confirm')}><Trash2 /> Delete</DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      {problem && <p className="px-3 pb-1 text-xs text-red-700 dark:text-red-300">{problem}</p>}
    </li>
  );
}

function Chip({ children, tone, title }: { children: React.ReactNode; tone?: 'ok' | 'warn'; title?: string }) {
  return (
    <span
      title={title}
      className={cn(
        'rounded-full px-2 py-0.5 text-[11px] font-medium',
        tone === 'ok' && 'bg-emerald-100 text-emerald-800 dark:bg-emerald-500/15 dark:text-emerald-300',
        tone === 'warn' && 'bg-amber-100 text-amber-900 dark:bg-amber-500/15 dark:text-amber-200',
        !tone && 'bg-zinc-200/80 text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300',
      )}
    >
      {children}
    </span>
  );
}

/** "approval: manual", "network: off": the server sends a label or a boolean. */
const setting = (v: string | boolean | null | undefined) => (typeof v === 'boolean' ? (v ? 'on' : 'off') : (v ?? '?'));

function EngineInfo({ health }: { health: Health | null | undefined }) {
  if (health === undefined) return <p className="text-xs text-zinc-500">Connecting to the server…</p>;
  if (health === null) return <p className="text-xs text-red-600 dark:text-red-400">Server unreachable: is `mimoe-agent serve` running?</p>;
  const tools = health.mode === 'tools';
  const studio = health.generation && health.generation !== 'unknown' ? `mimOE Studio ${health.generation}` : 'mimOE Studio';
  return (
    <div className="space-y-2 text-xs text-zinc-500 dark:text-zinc-400">
      <div className="flex items-center gap-2" title={[health.engine_version, health.node].filter(Boolean).join(' · ')}>
        <Server className="size-3.5 shrink-0" />
        <span className="truncate">{studio}{health.node ? ` · ${health.node}` : ''}</span>
      </div>
      <div className="flex items-center gap-2" title={health.workspace}>
        <FolderOpen className="size-3.5 shrink-0" />
        <span className="truncate">{health.workspace}</span>
      </div>
      <div className="flex flex-wrap gap-1.5">
        <Chip tone={tools ? 'ok' : 'warn'} title={tools ? 'the model passed the tool probe' : 'the model failed the tool probe, so the agent answers without tools (--force-tools overrides)'}>
          {tools ? 'tools on' : 'chat only'}
        </Chip>
        <Chip title="--auto-approve runs model-written code without asking">approval: {setting(health.approval)}</Chip>
        <Chip title="--allow-network lifts run_python's socket guard">network: {setting(health.network)}</Chip>
      </div>
    </div>
  );
}

const THEMES: [ThemeChoice, typeof Sun, string][] = [
  ['light', Sun, 'Light'],
  ['dark', Moon, 'Dark'],
  ['system', Monitor, 'System'],
];

function ThemeSwitch({ theme, onTheme }: { theme: ThemeChoice; onTheme: (choice: ThemeChoice) => void }) {
  return (
    <div className="flex rounded-lg bg-zinc-200/70 p-0.5 dark:bg-zinc-800" role="group" aria-label="Theme">
      {THEMES.map(([choice, Icon, label]) => (
        <button
          key={choice}
          type="button"
          aria-pressed={theme === choice}
          onClick={() => onTheme(choice)}
          className={cn(
            'flex flex-1 items-center justify-center gap-1.5 rounded-md py-1 text-xs font-medium outline-none transition-colors focus-visible:ring-2 focus-visible:ring-zinc-400',
            theme === choice ? 'bg-white text-zinc-900 shadow-sm dark:bg-zinc-950 dark:text-zinc-100' : 'text-zinc-500 hover:text-zinc-800 dark:text-zinc-400 dark:hover:text-zinc-200',
          )}
        >
          <Icon className="size-3.5" /> {label}
        </button>
      ))}
    </div>
  );
}
