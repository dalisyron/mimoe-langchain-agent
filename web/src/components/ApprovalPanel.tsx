import { useState } from 'react';
import type { ActionRequest, Decision } from '../events';
import { revealHidden } from '../hidden';
import { Args, splitCode } from './ToolCard';

/** Same heuristic as the CLI: code that reaches the network, spawns processes or writes/deletes files. */
const RED_FLAG = /socket|urllib|requests|httpx|subprocess|os\.system|shutil\.rmtree|os\.remove|open\(.*["']w/;

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
    <section className="approval" role="alertdialog" aria-label="Tool approval">
      <p>
        <strong>The agent wants to run {single ? 'this code' : `${requests.length} pieces of code`}.</strong>{' '}
        Approved code runs as you, with your files; nothing runs until you decide.
      </p>
      {requests.map((r, i) => {
        const [code] = splitCode(r.args);
        const hidden = code !== null ? revealHidden(code) : { text: '', count: 0 };
        const shownArgs = hidden.count ? { ...r.args, code: hidden.text } : r.args;
        return (
          <div key={i} className={`request ${choices[i] ?? ''}`}>
            <div className="tool-head">
              <code>{r.name}</code>
              {code !== null && RED_FLAG.test(code) && (
                <span className="chip warn">red flag: touches the network, processes or files — read it first</span>
              )}
              {hidden.count > 0 && (
                <span className="chip danger">
                  {hidden.count} invisible or control character{hidden.count === 1 ? '' : 's'}, shown escaped: they can make code look different from what runs
                </span>
              )}
            </div>
            <Args args={shownArgs} />
            <div className="actions">
              <button type="button" className={choices[i] === 'approve' ? 'primary' : ''} onClick={() => pick(i, 'approve')}>Approve</button>
              <button type="button" className={choices[i] === 'reject' ? 'danger' : ''} onClick={() => pick(i, 'reject')}>Deny</button>
            </div>
          </div>
        );
      })}
      {!single && (
        <div className="actions">
          <button type="button" className="primary" disabled={decided < requests.length} onClick={() => onDecide(choices as Decision[])}>
            Continue
          </button>
          <span className="muted">{decided}/{requests.length} decided</span>
        </div>
      )}
    </section>
  );
}
