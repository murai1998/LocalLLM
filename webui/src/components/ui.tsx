import { Loader2 } from "lucide-react";
import type { ButtonHTMLAttributes, ReactNode, SelectHTMLAttributes, TextareaHTMLAttributes } from "react";

export function cn(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(" ");
}

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";

const buttonStyles: Record<ButtonVariant, string> = {
  primary:
    "bg-accent hover:bg-accent-2 text-white shadow-lg shadow-accent/20 disabled:bg-edge disabled:text-ink-faint disabled:shadow-none",
  secondary:
    "bg-panel-2 hover:bg-edge text-ink border border-edge disabled:text-ink-faint",
  ghost: "hover:bg-panel-2 text-ink-dim hover:text-ink",
  danger: "bg-bad/15 hover:bg-bad/25 text-bad border border-bad/30",
};

export function Button({
  variant = "primary",
  busy = false,
  className,
  children,
  disabled,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: ButtonVariant;
  busy?: boolean;
}) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2 text-sm font-medium",
        "transition-colors duration-150 cursor-pointer disabled:cursor-not-allowed",
        buttonStyles[variant],
        className,
      )}
      disabled={disabled || busy}
      {...rest}
    >
      {busy && <Loader2 className="size-4 animate-spin" />}
      {children}
    </button>
  );
}

export function Card({
  title,
  actions,
  className,
  children,
}: {
  title?: ReactNode;
  actions?: ReactNode;
  className?: string;
  children: ReactNode;
}) {
  return (
    <section
      className={cn(
        "rounded-xl border border-edge bg-panel p-5 shadow-sm fade-up",
        className,
      )}
    >
      {(title || actions) && (
        <header className="mb-4 flex items-center justify-between gap-3">
          {title && <h2 className="text-sm font-semibold tracking-wide text-ink">{title}</h2>}
          {actions}
        </header>
      )}
      {children}
    </section>
  );
}

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-xs font-medium uppercase tracking-wider text-ink-faint">{label}</span>
      {children}
    </label>
  );
}

export function Select({
  className,
  children,
  ...rest
}: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      className={cn(
        "rounded-lg border border-edge bg-panel-2 px-3 py-2 text-sm text-ink",
        "outline-none transition-colors focus:border-accent/60 cursor-pointer",
        className,
      )}
      {...rest}
    >
      {children}
    </select>
  );
}

export function Textarea({ className, ...rest }: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return (
    <textarea
      className={cn(
        "w-full resize-y rounded-lg border border-edge bg-panel-2 px-3 py-2.5 text-sm",
        "leading-relaxed text-ink placeholder:text-ink-faint outline-none",
        "transition-colors focus:border-accent/60",
        className,
      )}
      {...rest}
    />
  );
}

export function Badge({
  tone = "neutral",
  children,
}: {
  tone?: "neutral" | "good" | "warn" | "bad" | "accent";
  children: ReactNode;
}) {
  const tones = {
    neutral: "bg-panel-2 text-ink-dim border-edge",
    good: "bg-good/10 text-good border-good/30",
    warn: "bg-warn/10 text-warn border-warn/30",
    bad: "bg-bad/10 text-bad border-bad/30",
    accent: "bg-accent/10 text-accent border-accent/30",
  };
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-medium",
        tones[tone],
      )}
    >
      {children}
    </span>
  );
}

export function StatusDot({ ok, label }: { ok: boolean | null; label: string }) {
  return (
    <span className="inline-flex items-center gap-2 text-xs text-ink-dim">
      <span
        className={cn(
          "size-2 rounded-full",
          ok === null && "bg-ink-faint pulse-soft",
          ok === true && "bg-good shadow-[0_0_8px] shadow-good/60",
          ok === false && "bg-bad",
        )}
      />
      {label}
    </span>
  );
}

export function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-edge bg-panel-2 px-4 py-3">
      <div className="text-[11px] font-medium uppercase tracking-wider text-ink-faint">{label}</div>
      <div className="mt-1 text-lg font-semibold tabular-nums text-ink">{value}</div>
    </div>
  );
}

export function ErrorNote({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-bad/30 bg-bad/10 px-4 py-3 text-sm text-bad fade-up">
      {message}
    </div>
  );
}
