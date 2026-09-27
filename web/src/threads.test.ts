import { describe, expect, it } from 'vitest';
import { filterThreads, groupByDate, titleFrom, toApproval, type ThreadDetail, type ThreadSummary } from './threads';

const NOW = new Date(2026, 8, 26, 15, 30); // Saturday 26 September 2026, 15:30 local time
const at = (daysAgo: number, hour = 12): string => new Date(2026, 8, 26 - daysAgo, hour).toISOString();
const thread = (id: string, updated: string | null, title = id): ThreadSummary => ({ id, title, created_at: updated, updated_at: updated });

describe('sidebar grouping and search', () => {
  it('groups by last use, keeping the server order inside each group', () => {
    const threads = [
      thread('a', at(0, 9)), thread('b', at(0, 0)), thread('c', at(1, 23)), thread('d', at(3)),
      thread('e', at(20)), thread('f', at(90)), thread('g', null), thread('h', 'not a date'),
    ];
    expect(groupByDate(threads, NOW).map((g) => [g.label, g.threads.map((t) => t.id)])).toEqual([
      ['Today', ['a', 'b']],
      ['Yesterday', ['c']],
      ['Previous 7 days', ['d']],
      ['Previous 30 days', ['e']],
      ['Older', ['f', 'g', 'h']],
    ]);
    expect(groupByDate([], NOW)).toEqual([]);
  });

  it('finds titles containing every word, ignoring case', () => {
    const threads = [thread('1', at(0), 'Sum of sales.csv revenue'), thread('2', at(0), 'Summarize notes.md')];
    expect(filterThreads(threads, '  SUM  ').map((t) => t.id)).toEqual(['1', '2']);
    expect(filterThreads(threads, 'sum revenue').map((t) => t.id)).toEqual(['1']);
    expect(filterThreads(threads, 'nothing')).toEqual([]);
    expect(filterThreads(threads, '')).toBe(threads);
  });
});

describe('titles and reopened conversations', () => {
  it('titles a conversation like the server does (one line, 60 characters)', () => {
    expect(titleFrom('  What is the sum\n of 10 to 20?  ')).toBe('What is the sum of 10 to 20?');
    const long = titleFrom('word '.repeat(40));
    expect(long).toHaveLength(60);
    expect(long.endsWith('…')).toBe(true);
    expect(titleFrom('   ')).toBe('New conversation');
  });

  it('turns the pending approval of a reopened conversation into the panel state', () => {
    const detail: ThreadDetail = {
      ...thread('t1', at(0)),
      turns: [],
      busy: false,
      approval: { interrupt_id: 'abc', action_requests: [{ name: 'run_python', args: { code: 'print(1)' } }] },
    };
    expect(toApproval(detail)).toEqual({ interruptId: 'abc', requests: [{ name: 'run_python', args: { code: 'print(1)' } }] });
    expect(toApproval({ ...detail, approval: null })).toBeNull();
  });
});
