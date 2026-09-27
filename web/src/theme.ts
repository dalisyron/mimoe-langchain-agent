import { useEffect, useState } from 'react';

export type ThemeChoice = 'light' | 'dark' | 'system';

/** The same key the inline script in index.html reads before the first paint. */
const KEY = 'mimoe-theme';
const DARK_QUERY = '(prefers-color-scheme: dark)';

export function storedTheme(): ThemeChoice {
  try {
    const value = localStorage.getItem(KEY);
    return value === 'light' || value === 'dark' ? value : 'system';
  } catch {
    return 'system'; // storage blocked (private window, disabled site data)
  }
}

function apply(choice: ThemeChoice): void {
  const dark = choice === 'dark' || (choice === 'system' && matchMedia(DARK_QUERY).matches);
  document.documentElement.classList.toggle('dark', dark);
}

/** Light, dark or the system setting; remembered per browser. */
export function useTheme(): [ThemeChoice, (choice: ThemeChoice) => void] {
  const [choice, setChoice] = useState<ThemeChoice>(storedTheme);
  useEffect(() => {
    apply(choice);
    try {
      if (choice === 'system') localStorage.removeItem(KEY);
      else localStorage.setItem(KEY, choice);
    } catch { /* not remembered, still applied */ }
    if (choice !== 'system') return;
    const query = matchMedia(DARK_QUERY);
    const follow = () => apply('system');
    query.addEventListener('change', follow);
    return () => query.removeEventListener('change', follow);
  }, [choice]);
  return [choice, setChoice];
}
