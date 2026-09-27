import { PanelLeft, SquarePen } from 'lucide-react';
import type { ReactNode } from 'react';
import { cn } from '../lib/utils';
import { Button } from './ui/button';

/** Sidebar and New chat buttons (when the sidebar is closed), the conversation title, and the model menu. */
export function TopBar({ title, sidebarOpen, onOpenSidebar, onNewChat, children }: {
  title: string;
  sidebarOpen: boolean;
  onOpenSidebar: () => void;
  onNewChat: () => void;
  children: ReactNode;
}) {
  return (
    <header className="flex h-14 shrink-0 items-center gap-1 border-b border-zinc-200/70 px-2 sm:px-3 dark:border-zinc-800/70">
      <div className={cn('flex items-center gap-0.5', sidebarOpen && 'md:hidden')}>
        <Button variant="ghost" size="icon" onClick={onOpenSidebar} aria-label="Open sidebar" title="Open sidebar">
          <PanelLeft />
        </Button>
        <Button variant="ghost" size="icon" onClick={onNewChat} aria-label="New chat" title="New chat">
          <SquarePen />
        </Button>
      </div>
      {/* narrow screens give the room to the model menu; the sidebar lists the titles */}
      <h1 className="hidden min-w-0 flex-1 truncate px-1.5 text-[15px] font-medium sm:block">{title}</h1>
      <div className="flex-1 sm:hidden" />
      {children}
    </header>
  );
}
