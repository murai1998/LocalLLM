import {
  Bot,
  ChevronDown,
  FileAudio,
  FileImage,
  FileText,
  Paperclip,
  SendHorizonal,
  Trash2,
  Wrench,
  X,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Badge, Button, ErrorNote, Textarea, cn } from "../components/ui";
import {
  api,
  fmtSeconds,
  type AgentStep,
  type Health,
  type SkillInfo,
  type UploadedAttachment,
} from "../lib/api";

type Mode = "chat" | "agent";

interface Message {
  role: "user" | "assistant";
  content: string;
  elapsedSec?: number;
  attachments?: { name: string; kind: string }[];
  steps?: AgentStep[];
}

function kindIcon(kind: string) {
  if (kind === "image") return <FileImage className="size-3.5" />;
  if (kind === "audio") return <FileAudio className="size-3.5" />;
  return <FileText className="size-3.5" />;
}

function fmtSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function AttachmentChip({
  att,
  onRemove,
}: {
  att: UploadedAttachment & { previewUrl?: string };
  onRemove?: () => void;
}) {
  return (
    <span className="inline-flex items-center gap-2 rounded-lg border border-edge bg-panel-2 py-1 pl-2 pr-1 text-xs text-ink-dim fade-up">
      {att.previewUrl ? (
        <img src={att.previewUrl} alt={att.name} className="size-6 rounded object-cover" />
      ) : (
        kindIcon(att.kind)
      )}
      <span className="max-w-44 truncate text-ink">{att.name}</span>
      <span className="text-ink-faint">{fmtSize(att.size)}</span>
      {onRemove && (
        <button
          onClick={onRemove}
          className="rounded p-0.5 text-ink-faint transition-colors hover:bg-bad/20 hover:text-bad cursor-pointer"
          title="Detach"
        >
          <X className="size-3.5" />
        </button>
      )}
    </span>
  );
}

function AgentSteps({ steps }: { steps: AgentStep[] }) {
  const [open, setOpen] = useState(false);
  const calls = steps.filter((s) => s.kind === "call").length;
  return (
    <div className="mt-2 border-t border-edge pt-2">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 text-xs text-ink-faint transition-colors hover:text-ink cursor-pointer"
      >
        <Wrench className="size-3.5" />
        {calls} tool call{calls === 1 ? "" : "s"}
        <ChevronDown className={cn("size-3.5 transition-transform", open && "rotate-180")} />
      </button>
      {open && (
        <div className="mt-2 flex flex-col gap-1.5">
          {steps.map((s, i) => (
            <div key={i} className="rounded-md bg-surface/60 px-2.5 py-1.5 text-xs">
              <span className={cn("font-medium", s.kind === "call" ? "text-accent" : "text-good")}>
                {s.kind === "call" ? "→ " : "← "}
                {s.title}
              </span>
              <pre className="mt-1 max-h-32 overflow-auto whitespace-pre-wrap break-all font-mono text-[11px] text-ink-dim">
                {s.body}
              </pre>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function ChatPage({ health }: { health: Health | null }) {
  const [mode, setMode] = useState<Mode>("chat");
  const [skills, setSkills] = useState<SkillInfo[]>([]);
  const [enabledSkills, setEnabledSkills] = useState<Set<string>>(new Set());
  const [attachments, setAttachments] = useState<(UploadedAttachment & { previewUrl?: string })[]>(
    [],
  );
  const [uploading, setUploading] = useState(false);

  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    api
      .skills()
      .then(setSkills)
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, busy]);

  const toggleSkill = (name: string) => {
    setEnabledSkills((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  };

  const attachFiles = async (files: FileList | null) => {
    if (!files?.length) return;
    setError(null);
    setUploading(true);
    try {
      const list = Array.from(files);
      const stored = await api.uploadAttachments(list);
      const withPreviews = stored.map((s, i) => ({
        ...s,
        previewUrl: s.kind === "image" ? URL.createObjectURL(list[i]) : undefined,
      }));
      setAttachments((prev) => [...prev, ...withPreviews]);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const detach = (id: string) => {
    setAttachments((prev) => {
      const found = prev.find((a) => a.id === id);
      if (found?.previewUrl) URL.revokeObjectURL(found.previewUrl);
      return prev.filter((a) => a.id !== id);
    });
    void api.deleteAttachment(id).catch(() => undefined);
  };

  const send = async () => {
    const text = input.trim();
    if ((!text && attachments.length === 0) || busy) return;
    setError(null);
    const sent = attachments;
    const userMessage: Message = {
      role: "user",
      content: text || "(see attachments)",
      attachments: sent.map(({ name, kind }) => ({ name, kind })),
    };
    const next = [...messages, userMessage];
    setMessages(next);
    setInput("");
    setAttachments([]);
    setBusy(true);
    try {
      const res = await api.chat(
        next.map(({ role, content }) => ({ role, content })),
        {
          mode,
          skills: mode === "agent" ? [...enabledSkills] : [],
          attachmentIds: sent.map((a) => a.id),
        },
      );
      setMessages([
        ...next,
        { role: "assistant", content: res.reply, elapsedSec: res.elapsed_sec, steps: res.steps },
      ]);
      // Attachments were consumed by this message — release them server-side.
      for (const a of sent) {
        if (a.previewUrl) URL.revokeObjectURL(a.previewUrl);
        void api.deleteAttachment(a.id).catch(() => undefined);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setMessages(messages); // roll back; keep attachments so the user can retry
      setAttachments(sent);
      setInput(text);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <header className="flex items-center justify-between fade-up">
        <div>
          <h1 className="text-xl font-semibold">{mode === "chat" ? "Chat" : "Agent"}</h1>
          <p className="mt-1 text-sm text-ink-dim">
            {mode === "chat" ? (
              <>
                Talking to <span className="text-ink">{health?.model ?? "…"}</span> on your own
                GPU. Attach images, audio, or documents.
              </>
            ) : (
              <>
                Tool-using agent — reads project files, writes notes, and uses the skills you
                enable below.
              </>
            )}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1 rounded-lg bg-panel-2 p-1">
            {(["chat", "agent"] as Mode[]).map((m) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                className={cn(
                  "flex items-center gap-1.5 rounded-md px-3.5 py-1.5 text-sm font-medium transition-colors cursor-pointer",
                  mode === m ? "bg-accent/20 text-accent" : "text-ink-dim hover:text-ink",
                )}
              >
                {m === "agent" && <Bot className="size-4" />}
                {m === "chat" ? "Chat" : "Agent"}
              </button>
            ))}
          </div>
          {messages.length > 0 && (
            <Button variant="ghost" onClick={() => setMessages([])}>
              <Trash2 className="size-4" />
            </Button>
          )}
        </div>
      </header>

      {mode === "agent" && (
        <div className="rounded-xl border border-edge bg-panel px-4 py-3 fade-up">
          <div className="mb-2 flex items-center justify-between">
            <span className="text-xs font-medium uppercase tracking-wider text-ink-faint">
              Skills · {enabledSkills.size} of {skills.length} enabled
            </span>
            <div className="flex gap-2 text-xs">
              <button
                className="text-ink-faint transition-colors hover:text-ink cursor-pointer"
                onClick={() => setEnabledSkills(new Set(skills.map((s) => s.name)))}
              >
                all
              </button>
              <button
                className="text-ink-faint transition-colors hover:text-ink cursor-pointer"
                onClick={() => setEnabledSkills(new Set())}
              >
                none
              </button>
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            {skills.length === 0 && (
              <span className="text-xs text-ink-faint">No skills installed in skills/</span>
            )}
            {skills.map((skill) => (
              <button
                key={skill.name}
                onClick={() => toggleSkill(skill.name)}
                title={skill.description}
                className={cn(
                  "rounded-full border px-3 py-1 text-xs font-medium transition-colors cursor-pointer",
                  enabledSkills.has(skill.name)
                    ? "border-accent/50 bg-accent/15 text-accent"
                    : "border-edge bg-panel-2 text-ink-dim hover:text-ink",
                )}
              >
                {skill.name}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto pr-1">
        {messages.length === 0 && (
          <div className="flex flex-1 items-center justify-center text-sm text-ink-faint">
            {mode === "chat"
              ? "Ask anything, or attach files to analyze and compare — nothing leaves this machine."
              : "Give the agent a task — it can read project files, take notes, and use enabled skills."}
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
              {m.attachments && m.attachments.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {m.attachments.map((a, j) => (
                    <span
                      key={j}
                      className="inline-flex items-center gap-1.5 rounded-md bg-surface/50 px-2 py-1 text-xs text-ink-dim"
                    >
                      {kindIcon(a.kind)}
                      <span className="max-w-40 truncate">{a.name}</span>
                    </span>
                  ))}
                </div>
              )}
              {m.elapsedSec !== undefined && (
                <div className="mt-2">
                  <Badge>{fmtSeconds(m.elapsedSec)}</Badge>
                </div>
              )}
              {m.steps && m.steps.length > 0 && <AgentSteps steps={m.steps} />}
            </div>
          </div>
        ))}
        {busy && (
          <div className="flex justify-start">
            <div className="rounded-2xl border border-edge bg-panel px-4 py-3 text-sm text-ink-faint pulse-soft">
              {mode === "agent" ? "working with tools…" : "thinking…"}
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {error && <ErrorNote message={error} />}

      {attachments.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {attachments.map((a) => (
            <AttachmentChip key={a.id} att={a} onRemove={() => detach(a.id)} />
          ))}
        </div>
      )}

      <div className="flex items-end gap-2">
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".png,.jpg,.jpeg,.webp,.gif,.pdf,.txt,.md,.docx,.csv,.json,.yaml,.yml,.wav,.mp3,.m4a,.ogg,.flac,.webm,.html,.htm,.log,.xml"
          className="hidden"
          onChange={(e) => void attachFiles(e.target.files)}
        />
        <Button
          variant="secondary"
          busy={uploading}
          onClick={() => fileInputRef.current?.click()}
          title="Attach files — multiple allowed (e.g. two images to compare)"
          className="mb-0.5 px-3"
        >
          <Paperclip className="size-4" />
        </Button>
        <Textarea
          rows={2}
          value={input}
          placeholder={
            health?.gateway_ready === false
              ? "Gateway offline — start localllm-serve first"
              : mode === "chat"
                ? "Type a message… (Enter to send, Shift+Enter for newline)"
                : "Describe a task for the agent…"
          }
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void send();
            }
          }}
        />
        <Button
          onClick={send}
          busy={busy}
          disabled={!input.trim() && attachments.length === 0}
          className="mb-0.5"
        >
          <SendHorizonal className="size-4" />
        </Button>
      </div>
    </>
  );
}
