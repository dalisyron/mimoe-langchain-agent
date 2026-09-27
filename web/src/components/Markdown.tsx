import { Children, isValidElement, type ReactNode } from 'react';
import ReactMarkdown, { type Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { CodeBlock } from './CodeBlock';

const plugins = [remarkGfm];
// An <img src> in model output would beacon file contents to any URL through your browser.
const disallowed = ['img'];
type CodeProps = { className?: string; children?: ReactNode };
const components: Components = {
  a: ({ href, children }) => <a href={href} target="_blank" rel="noreferrer">{children}</a>,
  // A fenced block arrives as <pre><code class="language-x">; it becomes one highlighted CodeBlock.
  pre: ({ children }) => {
    const code = Children.toArray(children)[0];
    if (!isValidElement<CodeProps>(code)) return <pre>{children}</pre>;
    const language = /language-([\w+#-]+)/.exec(code.props.className ?? '')?.[1];
    const text = String(code.props.children ?? '').replace(/\n$/, '');
    return <CodeBlock className="not-prose my-3" code={text} language={language} />;
  },
  code: ({ children }) => (
    <code className="rounded-md bg-zinc-100 px-1.5 py-0.5 font-mono text-[0.85em] font-normal dark:bg-zinc-800">{children}</code>
  ),
};

const prose =
  'answer prose prose-zinc max-w-none leading-relaxed dark:prose-invert ' +
  'prose-p:my-2 prose-headings:mb-2 prose-headings:mt-5 prose-headings:font-semibold prose-ul:my-2 prose-ol:my-2 prose-li:my-0.5 ' +
  'prose-a:text-blue-600 prose-a:no-underline hover:prose-a:underline dark:prose-a:text-blue-400 ' +
  'prose-code:before:content-none prose-code:after:content-none prose-th:px-3 prose-td:px-3';

/** Assistant text as React elements: raw HTML stays text (no rehype-raw), images are dropped (unwrapDisallowed
 * keeps only children, and an image has none), unsafe link protocols are blanked by react-markdown's default. */
export function Markdown({ text }: { text: string }) {
  return (
    <div className={prose}>
      <ReactMarkdown remarkPlugins={plugins} disallowedElements={disallowed} unwrapDisallowed components={components}>
        {text}
      </ReactMarkdown>
    </div>
  );
}
