import * as Menu from '@radix-ui/react-dropdown-menu';
import { Check } from 'lucide-react';
import type { ComponentProps } from 'react';
import { cn } from '../../lib/utils';

/** Radix dropdown menu with the shadcn/ui look: keyboard navigation, focus handling and Escape come from Radix. */
export const DropdownMenu = Menu.Root;
export const DropdownMenuTrigger = Menu.Trigger;

export function DropdownMenuContent({ className, sideOffset = 6, ...props }: ComponentProps<typeof Menu.Content>) {
  return (
    <Menu.Portal>
      <Menu.Content
        sideOffset={sideOffset}
        className={cn(
          'z-50 min-w-44 overflow-hidden rounded-xl border border-zinc-200 bg-white p-1 text-sm text-zinc-800 shadow-lg shadow-zinc-900/5',
          'dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-100 dark:shadow-black/40',
          className,
        )}
        {...props}
      />
    </Menu.Portal>
  );
}

const itemClass =
  'relative flex cursor-default select-none items-center gap-2 rounded-lg px-2.5 py-1.5 outline-none ' +
  'data-[highlighted]:bg-zinc-100 data-[disabled]:pointer-events-none data-[disabled]:opacity-45 dark:data-[highlighted]:bg-zinc-800 ' +
  '[&_svg]:size-4 [&_svg]:shrink-0';

export function DropdownMenuItem({ className, ...props }: ComponentProps<typeof Menu.Item>) {
  return <Menu.Item className={cn(itemClass, className)} {...props} />;
}

export function DropdownMenuCheckboxItem({ className, children, ...props }: ComponentProps<typeof Menu.CheckboxItem>) {
  return (
    <Menu.CheckboxItem className={cn(itemClass, 'pl-8', className)} {...props}>
      <span className="absolute left-2.5 flex size-4 items-center justify-center">
        <Menu.ItemIndicator><Check /></Menu.ItemIndicator>
      </span>
      {children}
    </Menu.CheckboxItem>
  );
}

export function DropdownMenuLabel({ className, ...props }: ComponentProps<typeof Menu.Label>) {
  return <Menu.Label className={cn('px-2.5 pb-1 pt-2 text-xs font-medium text-zinc-500 dark:text-zinc-400', className)} {...props} />;
}

export function DropdownMenuSeparator({ className, ...props }: ComponentProps<typeof Menu.Separator>) {
  return <Menu.Separator className={cn('-mx-1 my-1 h-px bg-zinc-200 dark:bg-zinc-800', className)} {...props} />;
}
