import { useCallback, useEffect, useState } from 'react';
import { api } from './api';
import type { ThreadSummary } from './threads';

/** The sidebar's conversations: loaded once, refreshed when a turn starts or ends, renamed and deleted in place. */
export function useThreads() {
  const [threads, setThreads] = useState<ThreadSummary[] | null>(null); // null until the first answer
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setThreads((await api.threads()).threads);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);
  useEffect(() => { void refresh(); }, [refresh]);

  /** Show a conversation the server is creating right now, before the next refresh confirms it. */
  const upsert = useCallback((thread: ThreadSummary) => {
    setThreads((list) => [thread, ...(list ?? []).filter((t) => t.id !== thread.id)]);
  }, []);
  const rename = useCallback(async (id: string, title: string) => {
    const updated = await api.renameThread(id, title);
    setThreads((list) => list?.map((t) => (t.id === id ? updated : t)) ?? null);
  }, []);
  const remove = useCallback(async (id: string) => {
    await api.deleteThread(id);
    setThreads((list) => list?.filter((t) => t.id !== id) ?? null);
  }, []);
  return { threads, error, refresh, upsert, rename, remove };
}
