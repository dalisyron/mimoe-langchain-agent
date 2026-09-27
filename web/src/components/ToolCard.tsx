import { Ban, Check, ChevronRight, CircleX, Loader2, RotateCcw, ShieldQuestion } from 'lucide-react';
import type { Json } from '../events';
import { cn } from '../lib/utils';
import type { ToolBlock } from '../reducer';
import { stepLabel } from '../steps';
import { CodeBlock } from './CodeBlock';

/** run_python's `code` is shown as code; every other argument as JSON. */
export function splitCode(args: Json): [string | null, Json] {
  const { code, ...rest } = args;
  return typeof code === 'string' ? [code, rest] : [null, args];
}

export function Args({ args }: { args: Json }) {
  const [code, rest] = splitCode(args);
  return (
    <>
      {code !== null && <CodeBlock code={code} language="python" />}
      {Object.keys(rest).length > 0 && (
        <pre className="overflow-x-auto rounded-xl bg-zinc-50 p-3 font-mono text-xs text-zinc-700 dark:bg-zinc-900 dark:text-zinc-300">
          {JSON.stringify(rest, null, 2)}
        </pre>
      )}
    </>
  );
}

function StatusIcon({ block, retrying }: { block: ToolBlock; retrying: boolean }) {
  switch (block.status) {
    case 'running':
      return <Loader2 className="animate-spin" />;
    case 'awaiting_approval':
      return <ShieldQuestion className="text-amber-600 dark:text-amber-400" />;
    case 'done':
      return <Check className="text-emerald-600 dark:text-emerald-400" />;
    case 'denied':
      return <Ban />;
    case 'error':
      return retrying ? <RotateCcw /> : <CircleX />;
  }
}

/**
 * A tool call as one muted line ("Reading notes.md…", "Calculating failed, trying a different approach").
 * The arguments and the result, errors included, stay one click away; the approval panel shows code in full.
 */
export function ToolCard({ block, retrying }: { block: ToolBlock; retrying: boolean }) {
  const waiting = block.status === 'awaiting_approval';
  return (
    <details className="disclosure group/step" data-status={block.status}>
      <summary
        className={cn(
          'flex w-fit max-w-full cursor-pointer select-none items-center gap-2 rounded-md text-sm transition-colors [&_svg]:size-3.5 [&_svg]:shrink-0',
          waiting ? 'text-zinc-800 dark:text-zinc-200' : 'text-zinc-500 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100',
        )}
      >
        <StatusIcon block={block} retrying={retrying} />
        <span className="truncate">{stepLabel(block, retrying)}</span>
        <ChevronRight className="text-zinc-400 transition-transform group-open/step:rotate-90" />
      </summary>
      <div className="ml-[7px] mt-2 space-y-2 border-l-2 border-zinc-200 pl-4 dark:border-zinc-800">
        <Args args={block.args} />
        {block.result !== undefined && (
          <pre className="max-h-72 overflow-auto whitespace-pre-wrap break-words rounded-xl bg-zinc-50 p-3 font-mono text-xs text-zinc-700 dark:bg-zinc-900 dark:text-zinc-300">
            {block.result}
          </pre>
        )}
      </div>
    </details>
  );
}
