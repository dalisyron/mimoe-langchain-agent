import { useEffect, useState } from 'react';
import { api } from './api';
import { ApprovalPanel } from './components/ApprovalPanel';
import { Composer } from './components/Composer';
import { Header } from './components/Header';
import { Transcript } from './components/Transcript';
import type { Health } from './events';
import { useChat } from './useChat';

export default function App() {
  const { state, send, decide, stop, reset } = useChat();
  const [health, setHealth] = useState<Health | null | undefined>(undefined); // undefined = not fetched yet
  const [tick, setTick] = useState(0); // bumped after a model switch
  const [switching, setSwitching] = useState(false); // POST /api/model in flight: the agent is being rebuilt
  const busy = state.streaming || state.approval !== null || switching;

  // Badge: poll every 10 s, and refresh right after a turn ends (tokens/s changes per inference) or a model
  // switch finishes. No polling during the switch: the picker shows its progress, and the server's health
  // reports the engine as unreachable meanwhile, which would paint the dot red for a normal load.
  useEffect(() => {
    if (switching) return;
    let alive = true;
    const refresh = () => api.health().then((h) => alive && setHealth(h)).catch(() => alive && setHealth(null));
    void refresh();
    const timer = setInterval(refresh, 10_000);
    return () => { alive = false; clearInterval(timer); };
  }, [state.streaming, tick, switching]);

  return (
    <div className="app">
      <Header health={health} busy={busy} onNew={reset} onSwitching={setSwitching} onSwitched={() => setTick((n) => n + 1)} />
      <Transcript turns={state.turns} streaming={state.streaming} disabled={busy} onPrompt={send} />
      {state.approval && (
        <ApprovalPanel key={state.approval.interruptId} requests={state.approval.requests} onDecide={decide} />
      )}
      <Composer disabled={busy} streaming={state.streaming} onSend={send} onStop={stop} />
    </div>
  );
}
