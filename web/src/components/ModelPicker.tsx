import { useEffect, useState } from 'react';
import { api } from '../api';
import type { Models } from '../events';

type Status = { kind: 'busy' | 'ok' | 'bad'; text: string };
const gb = (bytes: number | null | undefined) => (bytes ? ` (${(bytes / 1e9).toFixed(1)} GB)` : '');

/** Lists loaded and registry models; POST /api/model loads the chosen one and re-runs the tool probe. */
export function ModelPicker({ current, disabled, onSwitching, onSwitched }: {
  current: string | null;
  disabled: boolean;
  onSwitching: (inFlight: boolean) => void; // App disables the composer meanwhile
  onSwitched: () => void;
}) {
  const [models, setModels] = useState<Models | null>(null);
  const [choice, setChoice] = useState('');
  const [unload, setUnload] = useState(true);
  const [status, setStatus] = useState<Status | null>(null);
  const busy = status?.kind === 'busy';

  useEffect(() => {
    let alive = true;
    api.models().then((m) => alive && setModels(m)).catch(() => alive && setModels(null));
    return () => { alive = false; };
  }, [current]);

  if (!models) return null;
  const loaded = new Set(models.loaded.map((m) => m.id));
  const selected = choice || models.current || current || '';

  const load = async () => {
    setStatus({ kind: 'busy', text: `loading ${selected}… a cold load can take a minute` });
    onSwitching(true);
    try {
      const { model, probe } = await api.switchModel(selected, unload);
      setStatus({ kind: probe && !probe.tools_ok ? 'bad' : 'ok', text: probe ? `${model.id}: ${probe.detail}` : `${model.id} loaded` });
    } catch (e) {
      setStatus({ kind: 'bad', text: e instanceof Error ? e.message : String(e) });
    }
    setChoice(''); onSwitching(false); onSwitched();
  };

  return (
    <div className="picker">
      <select value={selected} disabled={disabled || busy} onChange={(e) => setChoice(e.target.value)} aria-label="Model">
        <optgroup label="Loaded">
          {models.loaded.map((m) => (
            <option key={m.id} value={m.id}>{m.id}{m.max_context ? ` · ${m.max_context} ctx` : ''}</option>
          ))}
        </optgroup>
        <optgroup label="Registry">
          {models.registry.filter((m) => !loaded.has(m.id)).map((m) => (
            <option key={m.id} value={m.id}>{m.id}{gb(m.size_bytes)}{m.ready === false ? ' — not downloaded' : ''}</option>
          ))}
        </optgroup>
      </select>
      <label title="Unload the current model first to free memory">
        <input type="checkbox" checked={unload} onChange={(e) => setUnload(e.target.checked)} /> unload current
      </label>
      <button type="button" onClick={load} disabled={disabled || busy || !selected || selected === current}>Load</button>
      {status && <span className={`status ${status.kind}`}>{status.text}</span>}
    </div>
  );
}
