import type { ActionRequest } from './events';
import type { Approval, Turn } from './reducer';

/** One row of the sidebar (`GET /api/threads`); times are ISO 8601, null for a thread without a row. */
export type ThreadSummary = { id: string; title: string; created_at: string | null; updated_at: string | null };

/** `GET /api/threads/{id}`: the turns as the stream drew them, and the approval still waiting, if any. */
export type ThreadDetail = ThreadSummary & {
  turns: Turn[];
  approval: { interrupt_id: string; action_requests: ActionRequest[] } | null;
  busy: boolean;
};

export const toApproval = (detail: ThreadDetail): Approval | null =>
  detail.approval ? { interruptId: detail.approval.interrupt_id, requests: detail.approval.action_requests } : null;

/** The title the server gives a conversation that starts with `message` (history.py `title_from`). */
export function titleFrom(message: string): string {
  const text = message.split(/\s+/).filter(Boolean).join(' ');
  if (!text) return 'New conversation';
  return text.length <= 60 ? text : `${text.slice(0, 59).trimEnd()}…`;
}

export type ThreadGroup = { label: string; threads: ThreadSummary[] };

const DAY = 86_400_000;

/** Sidebar sections by last use: Today, Yesterday, Previous 7 days, Previous 30 days, Older (order kept). */
export function groupByDate(threads: ThreadSummary[], now: Date = new Date()): ThreadGroup[] {
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const bounds: [string, number][] = [
    ['Today', today],
    ['Yesterday', today - DAY],
    ['Previous 7 days', today - 7 * DAY],
    ['Previous 30 days', today - 30 * DAY],
  ];
  const groups = new Map<string, ThreadSummary[]>();
  for (const thread of threads) {
    const time = Date.parse(thread.updated_at ?? thread.created_at ?? '');
    const label = Number.isNaN(time) ? 'Older' : (bounds.find(([, start]) => time >= start)?.[0] ?? 'Older');
    groups.set(label, [...(groups.get(label) ?? []), thread]);
  }
  return [...bounds.map(([label]) => label), 'Older']
    .filter((label) => groups.has(label))
    .map((label) => ({ label, threads: groups.get(label)! }));
}

/** Titles containing every word of `query`, ignoring case. */
export function filterThreads(threads: ThreadSummary[], query: string): ThreadSummary[] {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  return words.length ? threads.filter((t) => words.every((w) => t.title.toLowerCase().includes(w))) : threads;
}
