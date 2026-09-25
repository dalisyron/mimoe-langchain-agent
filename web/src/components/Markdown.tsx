import ReactMarkdown, { type Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';

const plugins = [remarkGfm];
// An <img src> in model output would beacon file contents to any URL through your browser.
const disallowed = ['img'];
const components: Components = {
  a: ({ href, children }) => <a href={href} target="_blank" rel="noreferrer">{children}</a>,
};

/** Assistant text as React elements: raw HTML stays text (no rehype-raw), images are dropped (unwrapDisallowed
 * keeps only children, and an image has none), unsafe link protocols are blanked by react-markdown's default. */
export function Markdown({ text }: { text: string }) {
  return (
    <ReactMarkdown remarkPlugins={plugins} disallowedElements={disallowed} unwrapDisallowed components={components}>
      {text}
    </ReactMarkdown>
  );
}
