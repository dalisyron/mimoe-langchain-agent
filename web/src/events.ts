// Wire contract with the FastAPI backend (docs/CONTRACTS.md, "server.py").
// Each SSE frame is `event: <name>` + `data: <json>`; api.ts folds them into `{ type: <name>, ...json }`.

export type Json = Record<string, unknown>;
export type Decision = 'approve' | 'reject';
export type ActionRequest = { name: string; args: Json; description?: string };
export type ReviewConfig = { action_name: string; allowed_decisions: string[] };
export type Usage = { input_tokens: number; output_tokens: number; llm_calls: number };

/** Events the server streams; `error` ends a stream without a `done`. */
export type ServerEvent =
  | { type: 'token'; text: string }
  | { type: 'thinking'; text: string }
  | { type: 'tool_call'; id: string; name: string; args: Json }
  | { type: 'tool_result'; id: string; name: string; content: string; is_error: boolean }
  | { type: 'approval_required'; interrupt_id: string; action_requests: ActionRequest[]; review_configs?: ReviewConfig[] }
  | { type: 'notice'; text: string }
  | { type: 'done'; status: 'completed' | 'awaiting_approval'; elapsed_s: number; model: string; usage_total?: Usage | null }
  | { type: 'error'; message: string; hint?: string | null };

export const SERVER_EVENT_TYPES: ReadonlySet<string> = new Set<ServerEvent['type']>([
  'token', 'thinking', 'tool_call', 'tool_result', 'approval_required', 'notice', 'done', 'error',
]);

/** Things that happen in the browser. */
export type ClientAction =
  | { type: 'send'; text: string }
  | { type: 'decided'; decisions: Decision[] }
  | { type: 'stream_failed'; message: string; hint?: string } // HTTP 409/422 or a network failure
  | { type: 'stopped' }
  | { type: 'reset'; threadId: string };

export type Action = ServerEvent | ClientAction;

/** GET /api/health; most fields are null while Studio is unreachable (then `error` says why). */
export type Health = {
  mimoe_reachable: boolean;
  model: string | null;
  tokens_per_second: number | null;
  max_context: number | null;
  node: string | null;
  engine_version: string | null;
  generation: string | null; // "0.6" | "1.0" | "unknown"
  workspace: string;
  approval: string | boolean | null;
  network: string | boolean | null;
  mode: 'tools' | 'chat_only';
  error: string | null;
};

/** GET /api/models and POST /api/model. */
export type LoadedModel = { id: string; max_context?: number | null; supports_tools?: boolean | null };
export type RegistryModel = { id: string; ready?: boolean; size_bytes?: number | null };
export type Models = { loaded: LoadedModel[]; registry: RegistryModel[]; current: string | null };
export type Probe = { tools_ok: boolean; latency_s: number; detail: string };
export type SwitchResult = { model: LoadedModel; probe: Probe | null };
