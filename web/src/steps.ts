import type { Json } from './events';
import { revealHidden } from './hidden';
import { RUN_ENDED, type Block, type ToolBlock } from './reducer';

/** What a tool call does, in the words a reader takes in at a glance: `doing` while it runs, `did` after. */
export type Phrase = { doing: string; did: string };

const ARG_CHARS = 48;

/** A model-written argument as one short, visible line (hidden characters escaped, whitespace folded). */
function short(value: unknown, max = ARG_CHARS): string {
  if (typeof value !== 'string') return '';
  const text = revealHidden(value).text.replace(/\s+/g, ' ').trim();
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

export function phrase(name: string, args: Json): Phrase {
  switch (name) {
    case 'run_python':
      return { doing: 'Running Python', did: 'Ran Python' };
    case 'calculator':
      return { doing: 'Calculating', did: 'Calculated' };
    case 'list_files': {
      const path = short(args.path);
      const where = path && path !== '.' ? ` in ${path}` : '';
      return { doing: `Listing files${where}`, did: `Listed files${where}` };
    }
    case 'read_file': {
      const path = short(args.path) || 'a file';
      return { doing: `Reading ${path}`, did: `Read ${path}` };
    }
    case 'search_files': {
      const pattern = short(args.pattern, 32);
      const what = pattern ? ` for “${pattern}”` : '';
      return { doing: `Searching files${what}`, did: `Searched files${what}` };
    }
    case 'now':
      return { doing: 'Checking the time', did: 'Checked the time' };
    case 'mimoe_status':
      return { doing: 'Checking the mimOE engine', did: 'Checked the mimOE engine' };
    case 'git': {
      const command = short(args.command, 16);
      return command ? { doing: `Checking git ${command}`, did: `Checked git ${command}` } : { doing: 'Checking git', did: 'Checked git' };
    }
    default: {
      const tool = short(name, 32) || 'a tool';
      return { doing: `Using ${tool}`, did: `Used ${tool}` };
    }
  }
}

/**
 * The one line a tool call shows in the transcript. A failed call is the agent's business, not the reader's:
 * it reads the error and adjusts, so the line says so without the error (which opens on click). `retrying`
 * is true while the turn runs and once another tool call followed.
 */
export function stepLabel(block: ToolBlock, retrying: boolean): string {
  const p = phrase(block.name, block.args);
  switch (block.status) {
    case 'running':
      return `${p.doing}…`;
    case 'awaiting_approval':
      return block.name === 'run_python' ? 'Waiting for your approval to run this code' : `Waiting for your approval: ${p.doing}`;
    case 'done':
      return p.did;
    case 'denied':
      return block.name === 'run_python' ? 'You declined; the code did not run' : `You declined: ${p.doing}`;
    case 'error':
      if (block.result === RUN_ENDED || block.result === 'stopped') return `${p.doing} stopped before it finished`;
      return retrying ? `${p.doing} failed, trying a different approach` : `${p.doing} failed`;
  }
}

/** Whether the tool block at `index` is followed by another attempt: the turn is still running, or a later tool call exists. */
export function isRetrying(blocks: Block[], index: number, live: boolean): boolean {
  return live || blocks.slice(index + 1).some((b) => b.kind === 'tool');
}
