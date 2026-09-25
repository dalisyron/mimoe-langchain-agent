import type { Json } from '../events';
import type { ToolBlock, ToolStatus } from '../reducer';

const LABEL: Record<ToolStatus, string> = {
  running: 'running…', awaiting_approval: 'needs approval', done: 'done', error: 'error', denied: 'denied',
};

/** run_python's `code` is shown as code; every other argument as JSON. */
export function splitCode(args: Json): [string | null, Json] {
  const { code, ...rest } = args;
  return typeof code === 'string' ? [code, rest] : [null, args];
}

export function Args({ args }: { args: Json }) {
  const [code, rest] = splitCode(args);
  return (
    <>
      {code !== null && <pre className="code"><code>{code}</code></pre>}
      {Object.keys(rest).length > 0 && <pre className="args">{JSON.stringify(rest, null, 2)}</pre>}
    </>
  );
}

export function ToolCard({ block }: { block: ToolBlock }) {
  return (
    <div className={`tool tool-${block.status}`}>
      <div className="tool-head">
        <code>{block.name}</code>
        <span className="chip">{LABEL[block.status]}</span>
      </div>
      <Args args={block.args} />
      {block.result !== undefined && (
        <details open={block.status === 'error' || block.status === 'denied'}>
          <summary>result ({block.result.length} chars)</summary>
          <pre className="result">{block.result}</pre>
        </details>
      )}
    </div>
  );
}
