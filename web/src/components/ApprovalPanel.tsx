import { EyeOff, Play, ShieldAlert, TriangleAlert, X } from 'lucide-react';
import { useState } from 'react';
import type { ActionRequest, Decision } from '../events';
import { revealHidden } from '../hidden';
import { cn } from '../lib/utils';
import { splitCode } from './ToolCard';
import { Button } from './ui/button';

/** Same heuristic as the CLI: code that reaches the network, spawns processes or writes/deletes files. */
const RED_FLAG = /socket|urllib|requests|httpx|subprocess|os\.system|shutil\.rmtree|os\.remove|open\(.*["']w/;

/** The code exactly as it will run: numbered lines, no highlighting, hidden characters already escaped. */
function CodeLines({ code }: { code: string }) {
  return (
    <pre className="max-h-80 overflow-auto bg-zinc-50 py-2.5 font-mono text-[13px] leading-relaxed dark:bg-zinc-900/60">
      <code>
        {code.split('\n').map((line, i) => (
          <div key={i} className="flex">
            <span className="w-11 shrink-0 select-none pr-3 text-right text-zinc-400 dark:text-zinc-600">{i + 1}</span>
            <span className="whitespace-pre pr-4">{line || ' '}</span>
          </div>
        ))}
      </code>
    </pre>
  );
}

/** One Approve/Deny per action request; a single request sends at once, several need Continue. */
export function ApprovalPanel({ requests, onDecide }: {
  requests: ActionRequest[];
  onDecide: (decisions: Decision[]) => void;
}) {
  const [choices, setChoices] = useState<(Decision | null)[]>(() => requests.map(() => null));
  const single = requests.length === 1;
  const pick = (i: number, d: Decision) =>
    single ? onDecide([d]) : setChoices((c) => c.map((x, j) => (j === i ? d : x)));
  const decided = choices.filter(Boolean).length;

  return (
    <section
      role="alertdialog"
      aria-label="Tool approval"
      className="approval mb-3 rounded-2xl border border-amber-300/80 bg-amber-50/70 p-4 shadow-sm dark:border-amber-500/30 dark:bg-amber-500/[0.07]"
    >
      <div className="flex gap-3">
        <ShieldAlert className="mt-0.5 size-5 shrink-0 text-amber-600 dark:text-amber-400" />
        <div className="min-w-0">
          <p className="font-medium">The agent wants to run {single ? 'this code' : `${requests.length} pieces of code`}</p>
          <p className="mt-0.5 text-sm text-zinc-600 dark:text-zinc-400">Approved code runs as you, with your files. Nothing runs until you decide.</p>
        </div>
      </div>
      {requests.map((r, i) => {
        const [code, rest] = splitCode(r.args);
        const hidden = code !== null ? revealHidden(code) : { text: '', count: 0 };
        return (
          <div
            key={i}
            className={cn(
              'mt-3 overflow-hidden rounded-xl border bg-white dark:bg-zinc-950',
              choices[i] === 'approve' ? 'border-emerald-400' : choices[i] === 'reject' ? 'border-red-400' : 'border-zinc-200 dark:border-zinc-800',
            )}
          >
            <div className="flex flex-wrap items-center gap-2 border-b border-zinc-200 px-3 py-2 text-xs dark:border-zinc-800">
              <code className="font-mono font-medium">{r.name}</code>
              {code !== null && RED_FLAG.test(code) && (
                <span className="inline-flex items-center gap-1 rounded-full bg-amber-100 px-2 py-0.5 text-amber-900 dark:bg-amber-500/15 dark:text-amber-200">
                  <TriangleAlert className="size-3" /> red flag: touches the network, processes or files — read it first
                </span>
              )}
              {hidden.count > 0 && (
                <span className="inline-flex items-center gap-1 rounded-full bg-red-100 px-2 py-0.5 text-red-800 dark:bg-red-500/15 dark:text-red-200">
                  <EyeOff className="size-3" />
                  {hidden.count} invisible or control character{hidden.count === 1 ? '' : 's'}, shown escaped: they can make code look different from what runs
                </span>
              )}
            </div>
            {code !== null && <CodeLines code={hidden.text} />}
            {Object.keys(rest).length > 0 && (
              <pre className="overflow-x-auto border-t border-zinc-200 p-3 font-mono text-xs dark:border-zinc-800">{JSON.stringify(rest, null, 2)}</pre>
            )}
            <div className="flex items-center gap-2 border-t border-zinc-200 px-3 py-2 dark:border-zinc-800">
              <Button size="sm" variant={single || choices[i] === 'approve' ? 'primary' : 'outline'} onClick={() => pick(i, 'approve')}>
                <Play /> Approve
              </Button>
              <Button size="sm" variant={choices[i] === 'reject' ? 'danger' : 'outline'} onClick={() => pick(i, 'reject')}>
                <X /> Deny
              </Button>
            </div>
          </div>
        );
      })}
      {!single && (
        <div className="mt-3 flex items-center gap-3">
          <Button variant="primary" size="sm" disabled={decided < requests.length} onClick={() => onDecide(choices as Decision[])}>
            Continue
          </Button>
          <span className="text-xs text-zinc-500">{decided}/{requests.length} decided</span>
        </div>
      )}
    </section>
  );
}
