import { TriangleAlert } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { api } from './api';
import { ApprovalPanel } from './components/ApprovalPanel';
import { Composer } from './components/Composer';
import { EmptyState } from './components/EmptyState';
import { ModelMenu } from './components/ModelMenu';
import { Sidebar } from './components/Sidebar';
import { TopBar } from './components/TopBar';
import { Transcript } from './components/Transcript';
import type { Health } from './events';
import { useTheme } from './theme';
import { titleFrom } from './threads';
import { useChat } from './useChat';
import { useThreads } from './useThreads';

const WIDE = '(min-width: 768px)'; // Tailwind's md: the sidebar sits beside the chat from here on
const isWide = () => matchMedia(WIDE).matches;

/** The open conversation lives in the address bar (`?c=<id>`), so a reload reopens it. */
const threadInUrl = () => new URLSearchParams(location.search).get('c');
const showThreadInUrl = (id: string | null) =>
  history.replaceState(null, '', id ? `?c=${encodeURIComponent(id)}` : location.pathname);

export default function App() {
  const { state, send, decide, stop, reset, open } = useChat();
  const { threads, error: threadsError, refresh, upsert, rename, remove } = useThreads();
  const [theme, setTheme] = useTheme();
  const [health, setHealth] = useState<Health | null | undefined>(undefined); // undefined = not fetched yet
  const [tick, setTick] = useState(0); // bumped after a model switch
  const [switching, setSwitching] = useState(false); // POST /api/model in flight: the agent is being rebuilt
  const [sidebar, setSidebar] = useState(isWide);
  const [openError, setOpenError] = useState<string | null>(null);
  const scroller = useRef<HTMLElement>(null);
  const stick = useRef(true); // follow the stream unless the reader scrolled up
  const busy = state.streaming || state.approval !== null || switching;

  const firstMessage = state.turns.find((t) => t.role === 'user');
  const title =
    threads?.find((t) => t.id === state.threadId)?.title ??
    (firstMessage?.role === 'user' ? titleFrom(firstMessage.text) : 'New chat');

  // Health: poll every 10 s, and right after a turn ends (tokens/s changes per inference) or a model switch
  // finishes. No polling during the switch: the server reports the engine as unreachable meanwhile.
  useEffect(() => {
    if (switching) return;
    let alive = true;
    const poll = () => api.health().then((h) => alive && setHealth(h)).catch(() => alive && setHealth(null));
    void poll();
    const timer = setInterval(poll, 10_000);
    return () => { alive = false; clearInterval(timer); };
  }, [state.streaming, tick, switching]);

  // The conversation list changes when a turn starts (a new conversation) and ends (it moves to the top).
  useEffect(() => { void refresh(); }, [state.streaming, refresh]);

  useEffect(() => { document.title = state.turns.length ? `${title} · mimoe-agent` : 'mimoe-agent'; }, [title, state.turns.length]);

  useEffect(() => {
    const el = scroller.current;
    if (!el) return;
    if (state.turns.length === 0) el.scrollTo({ top: 0 }); // a new chat starts at its greeting
    else if (stick.current) el.scrollTo({ top: el.scrollHeight });
  }, [state.turns, state.approval]);
  const onScroll = () => {
    const el = scroller.current;
    if (el) stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
  };

  useEffect(() => {
    if (!sidebar || isWide()) return;
    const close = (e: KeyboardEvent) => { if (e.key === 'Escape') setSidebar(false); };
    window.addEventListener('keydown', close);
    return () => window.removeEventListener('keydown', close);
  }, [sidebar]);

  const closeOnNarrow = () => { if (!isWide()) setSidebar(false); };

  const select = async (id: string) => {
    setOpenError(null);
    stick.current = true;
    closeOnNarrow();
    try {
      if (await open(id)) showThreadInUrl(id);
    } catch (e) {
      showThreadInUrl(null);
      reset();
      setOpenError(`Could not open that conversation (${e instanceof Error ? e.message : String(e)}).`);
    }
  };

  // Reopen the conversation named in the address bar, once.
  const restored = useRef(false);
  useEffect(() => {
    if (restored.current) return;
    restored.current = true;
    const id = threadInUrl();
    if (id) void select(id);
  });

  const newChat = () => { reset(); showThreadInUrl(null); setOpenError(null); closeOnNarrow(); };
  const sendMessage = (text: string) => {
    if (state.turns.length === 0) {
      const now = new Date().toISOString();
      upsert({ id: state.threadId, title: titleFrom(text), created_at: now, updated_at: now });
      showThreadInUrl(state.threadId);
    }
    stick.current = true;
    setOpenError(null);
    send(text);
  };
  const removeThread = async (id: string) => {
    await remove(id);
    if (id === state.threadId) newChat();
  };

  const problem = health === null ? 'The server is unreachable: is `mimoe-agent serve` running?' : (health?.error ?? openError);
  const placeholder = state.approval
    ? 'Approve or deny the code above to continue…'
    : busy ? 'Waiting for the agent…' : 'Ask about your workspace…';

  return (
    <div className="flex h-dvh overflow-hidden">
      <Sidebar
        open={sidebar}
        threads={threads}
        error={threadsError}
        currentId={state.threadId}
        health={health}
        theme={theme}
        onTheme={setTheme}
        onClose={() => setSidebar(false)}
        onNew={newChat}
        onSelect={(id) => void select(id)}
        onRename={rename}
        onDelete={removeThread}
      />
      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar title={title} sidebarOpen={sidebar} onOpenSidebar={() => setSidebar(true)} onNewChat={newChat}>
          <ModelMenu health={health} disabled={busy} onSwitching={setSwitching} onSwitched={() => setTick((n) => n + 1)} />
        </TopBar>
        {problem && (
          <div className="border-b border-amber-200 bg-amber-50 px-4 py-2 text-sm text-amber-900 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-200">
            <div className="mx-auto flex max-w-3xl items-start gap-2">
              <TriangleAlert className="mt-0.5 size-4 shrink-0" />
              <span className="whitespace-pre-wrap">{problem}</span>
            </div>
          </div>
        )}
        <main ref={scroller} onScroll={onScroll} className="min-h-0 flex-1 overflow-y-auto">
          {state.turns.length === 0
            ? <EmptyState workspace={health?.workspace ?? null} model={health?.model ?? null} disabled={busy} onPrompt={sendMessage} />
            : <Transcript turns={state.turns} streaming={state.streaming} />}
        </main>
        <div className="mx-auto w-full max-w-3xl shrink-0 px-4 pb-3 pt-2">
          {state.approval && (
            <ApprovalPanel key={state.approval.interruptId} requests={state.approval.requests} onDecide={decide} />
          )}
          <Composer disabled={busy} streaming={state.streaming} placeholder={placeholder} onSend={sendMessage} onStop={stop} />
          <p className="mt-2 text-center text-xs text-zinc-400 dark:text-zinc-500">
            Runs on this computer. The model can be wrong: numbers are safest when a tool computed them.
          </p>
        </div>
      </div>
    </div>
  );
}
