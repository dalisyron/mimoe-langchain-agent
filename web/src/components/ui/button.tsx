import { cva, type VariantProps } from 'class-variance-authority';
import type { ButtonHTMLAttributes } from 'react';
import { cn } from '../../lib/utils';

const buttonVariants = cva(
  'inline-flex shrink-0 items-center justify-center gap-2 whitespace-nowrap rounded-lg text-sm font-medium transition-colors ' +
    'outline-none focus-visible:ring-2 focus-visible:ring-zinc-400 focus-visible:ring-offset-1 dark:focus-visible:ring-zinc-500 dark:focus-visible:ring-offset-zinc-950 ' +
    'disabled:pointer-events-none disabled:opacity-40 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0',
  {
    variants: {
      variant: {
        primary: 'bg-zinc-900 text-white hover:bg-zinc-700 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-300',
        outline:
          'border border-zinc-300 bg-white text-zinc-800 hover:bg-zinc-100 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100 dark:hover:bg-zinc-800',
        ghost: 'text-zinc-700 hover:bg-zinc-200/70 dark:text-zinc-300 dark:hover:bg-zinc-800',
        danger: 'bg-red-600 text-white hover:bg-red-500',
      },
      size: {
        sm: 'h-8 px-3',
        md: 'h-9 px-4',
        icon: 'size-9',
        iconSm: 'size-8',
      },
    },
    defaultVariants: { variant: 'ghost', size: 'md' },
  },
);

export type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & VariantProps<typeof buttonVariants>;

/** The shadcn/ui button: `type="button"` unless a form needs `submit`. */
export function Button({ className, variant, size, type = 'button', ...props }: ButtonProps) {
  return <button type={type} className={cn(buttonVariants({ variant, size }), className)} {...props} />;
}
