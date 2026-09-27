import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { Markdown } from './components/Markdown';

/** The rendered Markdown without its styled wrapper <div>. */
const render = (text: string) =>
  renderToStaticMarkup(createElement(Markdown, { text })).replace(/^<div class="answer [^"]*">/, '').replace(/<\/div>$/, '');

describe('Markdown (model output is untrusted)', () => {
  it('keeps raw HTML as text, never as elements', () => {
    expect(render('hi <script>alert(1)</script>')).toBe('<p>hi &lt;script&gt;alert(1)&lt;/script&gt;</p>');
    expect(render('<img src="http://127.0.0.1:9/x" onerror="alert(1)">')).not.toContain('<img');
  });

  it('drops markdown images (an <img src> would beacon file contents to any URL)', () => {
    expect(render('![beacon](http://127.0.0.1:9/x?secret)')).toBe('<p></p>');
  });

  it('blanks javascript:/data:/vbscript: links and opens real ones in a new tab without a referrer', () => {
    for (const bad of ['javascript:alert(1)', 'JavaScript:alert(1)', 'data:text/html,x', 'vbscript:msgbox']) {
      expect(render(`[x](${bad})`)).toBe('<p><a href="" target="_blank" rel="noreferrer">x</a></p>');
    }
    expect(render('[ok](http://example.com/a?b=c#d)')).toBe('<p><a href="http://example.com/a?b=c#d" target="_blank" rel="noreferrer">ok</a></p>');
  });

  it('renders a fenced block as highlighted text, never as markup', () => {
    const html = render('```python\nprint("<b>hi</b>")  # comment\n```');
    expect(html).toContain('code-block');
    expect(html).toContain('<span class="token comment"># comment</span>');
    expect(html).toContain('&lt;b&gt;hi&lt;/b&gt;');
    expect(render('```html\n<script>alert(1)</script>\n```')).not.toContain('<script>');
    expect(render('```\nplain\n```')).toContain('<span class="font-mono">text</span>'); // no language: plain text
    expect(render('use `ls -la` here')).toContain('<code class="rounded-md');
  });

  it('renders GFM tables (the answer format the fake backend and the model use)', () => {
    expect(render('| a | b |\n|---|---|\n| 1 | 2 |')).toContain('<table>');
  });
});
