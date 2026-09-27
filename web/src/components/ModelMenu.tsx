import { Check, ChevronDown, Loader2 } from 'lucide-react';
import { useCallback, useEffect, useState } from 'react';
import { api } from '../api';
import type { Health, Models } from '../events';
import { cn } from '../lib/utils';
import { Button } from './ui/button';
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from './ui/menu';

type Status = { ok: boolean; text: string };
const gb = (bytes: number | null | undefined) => (bytes ? `${(bytes / 1e9).toFixed(1)} GB` : '');
const ctx = (tokens: number | null | undefined) => (tokens ? `${Math.round(tokens / 1000)}k ctx` : '');

/** The current model and a menu of loaded and registered ones; choosing one loads it (POST /api/model), which
 * re-runs the tool probe and rebuilds the agent. The composer is disabled meanwhile (App's `onSwitching`). */
export function ModelMenu({ health, disabled, onSwitching, onSwitched }: {
  health: Health | null | undefined;
  disabled: boolean;
  onSwitching: (inFlight: boolean) => void;
  onSwitched: () => void;
}) {
  const [models, setModels] = useState<Models | null>(null);
  const [failed, setFailed] = useState(false);
  const [unload, setUnload] = useState(true);
  const [loading, setLoading] = useState<string | null>(null);
  const [status, setStatus] = useState<Status | null>(null);
  const current = health?.model ?? null;
  const known = health !== undefined && health !== null; // also when no model is loaded: the registry can be

  const refresh = useCallback(() => {
    api.models().then((m) => { setModels(m); setFailed(false); }).catch(() => setFailed(true));
  }, []);
  useEffect(() => { if (known) refresh(); }, [known, current, refresh]);

  const load = async (id: string) => {
    setLoading(id);
    setStatus(null);
    onSwitching(true);
    try {
      const { model, probe } = await api.switchModel(id, unload);
      setStatus({ ok: !probe || probe.tools_ok, text: probe ? `${model.id}: ${probe.detail}` : `${model.id} loaded` });
    } catch (e) {
      setStatus({ ok: false, text: e instanceof Error ? e.message : String(e) });
    }
    setLoading(null);
    onSwitching(false);
    onSwitched();
  };

  const loaded = new Set(models?.loaded.map((m) => m.id));
  const registry = models?.registry.filter((m) => !loaded.has(m.id)) ?? [];
  return (
    <div className="flex min-w-0 items-center gap-2">
      {status && (
        <span className={cn('hidden max-w-xs truncate text-xs lg:block', status.ok ? 'text-emerald-700 dark:text-emerald-400' : 'text-red-700 dark:text-red-400')} title={status.text}>
          {status.text}
        </span>
      )}
      <DropdownMenu onOpenChange={(open) => open && refresh()}>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" className="min-w-0 gap-2 px-2.5" disabled={disabled || loading !== null || !known} aria-label="Model">
            {loading
              ? <Loader2 className="animate-spin" />
              : <span className={cn('size-2 shrink-0 rounded-full', health?.mimoe_reachable ? 'bg-emerald-500' : 'bg-red-500')} />}
            <span className="truncate">{loading ? `Loading ${loading}…` : (current ?? 'No model loaded')}</span>
            {!loading && health?.tokens_per_second != null && (
              <span className="hidden text-xs font-normal text-zinc-500 sm:inline">{health.tokens_per_second.toFixed(1)} tok/s</span>
            )}
            <ChevronDown className="text-zinc-400" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-80 max-w-[calc(100vw-1rem)]">
          <DropdownMenuLabel>Loaded in mimOE Studio</DropdownMenuLabel>
          {failed && <p className="px-2.5 py-1.5 text-xs text-red-600">Could not list the models.</p>}
          {models?.loaded.length === 0 && <p className="px-2.5 py-1.5 text-xs text-zinc-500">None: pick one below to load it.</p>}
          {models?.loaded.map((m) => (
            <DropdownMenuItem key={m.id} onSelect={() => { if (m.id !== current) void load(m.id); }}>
              <Check className={cn(m.id === current ? 'opacity-100' : 'opacity-0')} />
              <span className="flex-1 truncate">{m.id}</span>
              <span className="text-xs text-zinc-500">{ctx(m.max_context)}</span>
            </DropdownMenuItem>
          ))}
          {registry.length > 0 && (
            <>
              <DropdownMenuSeparator />
              <DropdownMenuLabel>Downloaded, not loaded</DropdownMenuLabel>
              {registry.map((m) => (
                <DropdownMenuItem key={m.id} disabled={m.ready === false} onSelect={() => void load(m.id)}>
                  <span className="w-4" />
                  <span className="flex-1 truncate">{m.id}</span>
                  <span className="text-xs text-zinc-500">{m.ready === false ? 'not downloaded' : gb(m.size_bytes)}</span>
                </DropdownMenuItem>
              ))}
            </>
          )}
          <DropdownMenuSeparator />
          <DropdownMenuCheckboxItem checked={unload} onCheckedChange={(v) => setUnload(v === true)} onSelect={(e) => e.preventDefault()}>
            Unload the current model first
          </DropdownMenuCheckboxItem>
          <p className="px-2.5 pb-1.5 pt-1 text-xs leading-relaxed text-zinc-500">
            Loading re-runs the tool probe; a cold load can take a minute. Unloading frees memory for the new model.
          </p>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
}
