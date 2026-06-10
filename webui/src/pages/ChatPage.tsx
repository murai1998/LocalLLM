import { SendHorizonal, Trash2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Badge, Button, ErrorNote, Textarea, cn } from "../components/ui";
import { api, fmtSeconds, type Health } from "../lib/api";

interface Message {
  role: "user" | "assistant";
  content: string;
  elapsedSec?: number;
}

export function ChatPage({ health }: { health: Health | null }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, busy]);

  const send = async () => {
    const text = input.trim();
    if (!text || busy) return;
    setError(null);
    const next: Message[] = [...messages, { role: "user", content: text }];
    setMessages(next);
    setInput("");
    setBusy(true);
    try {
      const res = await api.chat(next.map(({ role, content }) => ({ role, content })));
      setMessages([...next, { role: "assistant", content: res.reply, elapsedSec: res.elapsed_sec }]);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setMessages(messages); // roll back the optimistic user message
      setInput(text);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <header className="flex items-center justify-between fade-up">
        <div>
          <h1 className="text-xl font-semibold">Chat</h1>
          <p className="mt-1 text-sm text-ink-dim">
            Talking to <span className="text-ink">{health?.model ?? "…"}</span> on your own GPU.
          </p>
        </div>
        {messages.length > 0 && (
          <Button variant="ghost" onClick={() => setMessages([])}>
            <Trash2 className="size-4" />
            Clear
          </Button>
        )}
      </header>

      <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto pr-1">
        {messages.length === 0 && (
          <div className="flex flex-1 items-center justify-center text-sm text-ink-faint">
            Ask anything — responses never leave this machine.
          </div>
        )}
        {messages.map((m, i) => (
          <div
            key={i}
            className={cn("flex fade-up", m.role === "user" ? "justify-end" : "justify-start")}
          >
            <div
              className={cn(
                "max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap",
                m.role === "user"
                  ? "bg-accent/20 text-ink rounded-br-md"
                  : "border border-edge bg-panel text-ink rounded-bl-md",
              )}
            >
              {m.content}
              {m.elapsedSec !== undefined && (
                <div className="mt-2">
                  <Badge>{fmtSeconds(m.elapsedSec)}</Badge>
                </div>
              )}
            </div>
          </div>
        ))}
        {busy && (
          <div className="flex justify-start">
            <div className="rounded-2xl border border-edge bg-panel px-4 py-3 text-sm text-ink-faint pulse-soft">
              thinking…
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {error && <ErrorNote message={error} />}

      <div className="flex items-end gap-3">
        <Textarea
          rows={2}
          value={input}
          placeholder={
            health?.gateway_ready === false
              ? "Gateway offline — start localllm-serve first"
              : "Type a message… (Enter to send, Shift+Enter for newline)"
          }
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void send();
            }
          }}
        />
        <Button onClick={send} busy={busy} disabled={!input.trim()} className="mb-0.5">
          <SendHorizonal className="size-4" />
        </Button>
      </div>
    </>
  );
}
