import { Cpu, Mic2, Radio, Server } from "lucide-react";
import type { ReactNode } from "react";
import { Badge, Card } from "../components/ui";
import type { Health } from "../lib/api";

function Row({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="flex items-center justify-between border-b border-edge py-2.5 text-sm last:border-0">
      <span className="text-ink-dim">{label}</span>
      <span className="text-ink">{value}</span>
    </div>
  );
}

export function StatusPage({ health }: { health: Health | null }) {
  return (
    <>
      <header className="fade-up">
        <h1 className="text-xl font-semibold">Status</h1>
        <p className="mt-1 text-sm text-ink-dim">
          What's running on this machine right now. Refreshes every 15 seconds.
        </p>
      </header>

      <div className="grid gap-5 md:grid-cols-2">
        <Card
          title={
            <span className="flex items-center gap-2">
              <Server className="size-4 text-accent" /> Inference gateway
            </span>
          }
        >
          {health === null ? (
            <p className="text-sm text-ink-faint pulse-soft">Checking…</p>
          ) : (
            <>
              <Row
                label="State"
                value={
                  health.gateway_ready ? (
                    <Badge tone="good">ready</Badge>
                  ) : (
                    <Badge tone="bad">offline — run localllm-serve</Badge>
                  )
                }
              />
              <Row label="Endpoint" value={<code className="text-xs">{health.gateway_url}</code>} />
              <Row label="Model" value={health.model} />
              <Row label="Provider" value={health.provider} />
            </>
          )}
        </Card>

        <Card
          title={
            <span className="flex items-center gap-2">
              <Cpu className="size-4 text-accent" /> Platform
            </span>
          }
        >
          {health && (
            <>
              <Row label="Acceleration" value={<Badge tone="accent">{health.platform}</Badge>} />
              <Row
                label="Text-to-speech"
                value={
                  health.tts_available ? (
                    <Badge tone="good">Piper (offline)</Badge>
                  ) : (
                    <Badge tone="warn">pip install piper-tts</Badge>
                  )
                }
              />
              <Row label="Translate pipeline" value={health.translate_pipeline} />
            </>
          )}
        </Card>

        <Card
          title={
            <span className="flex items-center gap-2">
              <Mic2 className="size-4 text-accent" /> Live chunking
            </span>
          }
        >
          {health && (
            <>
              <Row
                label="Window"
                value={`${health.live_chunking.min_chunk_seconds.toFixed(0)}–${health.live_chunking.max_chunk_seconds.toFixed(0)} s`}
              />
              <Row
                label="Overlap"
                value={`${health.live_chunking.overlap_seconds.toFixed(1)} s`}
              />
            </>
          )}
        </Card>

        <Card
          title={
            <span className="flex items-center gap-2">
              <Radio className="size-4 text-accent" /> Coming next
            </span>
          }
        >
          <p className="text-sm leading-relaxed text-ink-dim">
            Streaming voice-to-voice translation: continuous microphone capture, silence-aware
            segmentation, and pipelined STT → translate → TTS with a 5–8&nbsp;s steady-state lag.
            The microphone plumbing on the Translate page is the foundation for it.
          </p>
        </Card>
      </div>
    </>
  );
}
