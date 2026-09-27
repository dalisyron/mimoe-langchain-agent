/// <reference types="react-syntax-highlighter" />
import { Check, Copy } from 'lucide-react';
import { useState } from 'react';
import SyntaxHighlighter from 'react-syntax-highlighter/dist/esm/prism-light';
import bash from 'react-syntax-highlighter/dist/esm/languages/prism/bash';
import css from 'react-syntax-highlighter/dist/esm/languages/prism/css';
import diff from 'react-syntax-highlighter/dist/esm/languages/prism/diff';
import javascript from 'react-syntax-highlighter/dist/esm/languages/prism/javascript';
import json from 'react-syntax-highlighter/dist/esm/languages/prism/json';
import markup from 'react-syntax-highlighter/dist/esm/languages/prism/markup';
import python from 'react-syntax-highlighter/dist/esm/languages/prism/python';
import sql from 'react-syntax-highlighter/dist/esm/languages/prism/sql';
import typescript from 'react-syntax-highlighter/dist/esm/languages/prism/typescript';
import yaml from 'react-syntax-highlighter/dist/esm/languages/prism/yaml';
import { cn } from '../lib/utils';

// A few grammars instead of Prism's full set; the colours come from the .code-block rules in styles.css.
const GRAMMARS = { bash, css, diff, javascript, json, markup, python, sql, typescript, yaml };
for (const [name, grammar] of Object.entries(GRAMMARS)) SyntaxHighlighter.registerLanguage(name, grammar);
const ALIASES: Record<string, keyof typeof GRAMMARS> = {
  py: 'python', sh: 'bash', shell: 'bash', zsh: 'bash', console: 'bash', js: 'javascript', ts: 'typescript',
  yml: 'yaml', html: 'markup', xml: 'markup', svg: 'markup', patch: 'diff',
};

/** Highlighted code (rendered as React elements, never as HTML) with its language and a Copy button. */
export function CodeBlock({ code, language, className }: { code: string; language?: string; className?: string }) {
  const [copied, setCopied] = useState(false);
  const label = language?.toLowerCase();
  const grammar = label && (label in GRAMMARS ? label : ALIASES[label]);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch { /* the clipboard needs a secure context; 127.0.0.1 and localhost are one */ }
  };
  const body = 'overflow-x-auto p-3.5 font-mono text-[13px] leading-relaxed text-zinc-800 dark:text-zinc-200';
  return (
    <div className={cn('code-block overflow-hidden rounded-xl border border-zinc-200 bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-900/70', className)}>
      <div className="flex items-center justify-between border-b border-zinc-200 py-1 pl-3.5 pr-1.5 text-xs text-zinc-500 dark:border-zinc-800 dark:text-zinc-400">
        <span className="font-mono">{label ?? 'text'}</span>
        <button
          type="button"
          onClick={copy}
          className="inline-flex items-center gap-1 rounded-md px-2 py-1 transition-colors hover:bg-zinc-200/70 hover:text-zinc-800 dark:hover:bg-zinc-800 dark:hover:text-zinc-200"
        >
          {copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      {grammar ? (
        <SyntaxHighlighter language={grammar} useInlineStyles={false} PreTag="pre" CodeTag="code" className={body}>
          {code}
        </SyntaxHighlighter>
      ) : (
        <pre className={body}><code>{code}</code></pre>
      )}
    </div>
  );
}
