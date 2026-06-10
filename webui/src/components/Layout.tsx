import {
  Activity,
  AudioLines,
  FileText,
  Languages,
  MessageSquare,
  Sparkles,
} from "lucide-react";
import type { ReactNode } from "react";
import type { Health } from "../lib/api";
import { StatusDot, cn } from "./ui";

export type PageId = "chat" | "translate" | "transcribe" | "documents" | "status";

const NAV: { id: PageId; label: string; icon: typeof MessageSquare }[] = [
  { id: "chat", label: "Chat", icon: MessageSquare },
  { id: "translate", label: "Translate", icon: Languages },
  { id: "transcribe", label: "Transcribe", icon: AudioLines },
  { id: "documents", label: "Documents", icon: FileText },
  { id: "status", label: "Status", icon: Activity },
];

export function Layout({
  page,
  onNavigate,
  health,
  children,
}: {
  page: PageId;
  onNavigate: (page: PageId) => void;
  health: Health | null;
  children: ReactNode;
}) {
  return (
    <div className="flex h-full">
      <aside className="flex w-60 shrink-0 flex-col border-r border-edge bg-panel">
        <div className="flex items-center gap-2.5 px-5 py-5">
          <div className="flex size-9 items-center justify-center rounded-xl bg-gradient-to-br from-accent to-accent-2 shadow-lg shadow-accent/30">
            <Sparkles className="size-5 text-white" />
          </div>
          <div>
            <div className="text-sm font-semibold tracking-wide">LocalLLM</div>
            <div className="text-[11px] text-ink-faint">private · offline · yours</div>
          </div>
        </div>

        <nav className="flex flex-col gap-1 px-3 py-2">
          {NAV.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => onNavigate(id)}
              className={cn(
                "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium",
                "transition-colors duration-150 cursor-pointer",
                page === id
                  ? "bg-accent/15 text-accent"
                  : "text-ink-dim hover:bg-panel-2 hover:text-ink",
              )}
            >
              <Icon className="size-4.5" />
              {label}
            </button>
          ))}
        </nav>

        <div className="mt-auto flex flex-col gap-2 border-t border-edge px-5 py-4">
          <StatusDot
            ok={health === null ? null : health.gateway_ready}
            label={
              health === null
                ? "checking gateway…"
                : health.gateway_ready
                  ? `${health.model}`
                  : "gateway offline — run localllm-serve"
            }
          />
          {health && (
            <StatusDot
              ok={health.tts_available}
              label={health.tts_available ? "Piper TTS ready" : "Piper TTS missing"}
            />
          )}
        </div>
      </aside>

      <main className="flex-1 overflow-y-auto">
        <div className="mx-auto flex h-full max-w-5xl flex-col gap-5 px-8 py-8">{children}</div>
      </main>
    </div>
  );
}
