import type { Json } from '../events';
import type { ToolBlock } from '../reducer';
import { stepLabel } from '../steps';

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

/**
 * A tool call as one muted line ("Reading notes.md…", "Calculating failed, trying a different approach").
 * The arguments and the result, errors included, stay one click away; the approval panel shows code in full.
 */
export function ToolCard({ block, retrying }: { block: ToolBlock; retrying: boolean }) {
  return (
    <details className={`step step-${block.status}`}>
      <summary>{stepLabel(block, retrying)}</summary>
      <div className="step-detail">
        <Args args={block.args} />
        {block.result !== undefined && <pre className="result">{block.result}</pre>}
      </div>
    </details>
  );
}
