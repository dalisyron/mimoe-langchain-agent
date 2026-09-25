import type { Health } from '../events';
import { ModelPicker } from './ModelPicker';

/** "approval: manual", "network: off" — the server sends a label or a boolean. */
const flag = (label: string, v: string | boolean | null | undefined) =>
  v == null ? null : <span className="chip">{label}: {typeof v === 'boolean' ? (v ? 'on' : 'off') : v}</span>;

export function Header({ health, busy, onNew, onSwitching, onSwitched }: {
  health: Health | null | undefined; // undefined = not fetched yet, null = server unreachable
  busy: boolean;
  onNew: () => void;
  onSwitching: (inFlight: boolean) => void;
  onSwitched: () => void;
}) {
  const tools = health?.mode === 'tools';
  return (
    <header>
      <div className="row">
        <h1>mimoe-agent</h1>
        <div className="badge" title={health ? `${health.engine_version ?? ''} ${health.node ?? ''}`.trim() : ''}>
          <span className={`dot ${health?.mimoe_reachable ? 'ok' : 'bad'}`} />
          {health === undefined && <span>connecting…</span>}
          {health === null && <span>server unreachable — is `mimoe-agent serve` running?</span>}
          {health && (
            <>
              <span>{health.model ?? 'no model'}</span>
              {health.tokens_per_second != null && <span>{health.tokens_per_second.toFixed(1)} tok/s</span>}
              {health.generation && health.generation !== 'unknown' && <span>Studio {health.generation}</span>}
              <span
                className={`chip ${tools ? 'ok' : 'warn'}`}
                title={tools ? 'the model passed the tool probe' : 'the model failed the tool probe, so the agent answers without tools (--force-tools overrides)'}
              >
                {tools ? 'tools' : 'chat only'}
              </span>
              {flag('approval', health.approval)}
              {flag('network', health.network)}
              <span className="path" title={health.workspace}>{health.workspace}</span>
            </>
          )}
        </div>
      </div>
      <div className="row controls">
        <ModelPicker enabled={health !== undefined} current={health?.model ?? null} disabled={busy || !health} onSwitching={onSwitching} onSwitched={onSwitched} />
        <button type="button" onClick={onNew}>New conversation</button>
      </div>
      {health?.error && <div className="hint">{health.error}</div>}
    </header>
  );
}
