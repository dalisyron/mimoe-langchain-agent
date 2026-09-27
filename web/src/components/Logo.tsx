import { cn } from '../lib/utils';

/** The app mark: a rounded square with an "m" (also the favicon in index.html). */
export function Logo({ className }: { className?: string }) {
  return (
    <div
      aria-hidden
      className={cn(
        'flex size-7 shrink-0 select-none items-center justify-center rounded-lg bg-zinc-900 text-[15px] font-bold leading-none text-white dark:bg-zinc-100 dark:text-zinc-900',
        className,
      )}
    >
      m
    </div>
  );
}
