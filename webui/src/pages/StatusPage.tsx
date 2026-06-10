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
              <Radio className="size-4 text-accent" /> Live voice-to-voice
            </span>
          }
        >
          {health && (
            <>
              <Row
                label="Segment ends after"
                value={`${(health.live_stream?.hangover_ms ?? 600) / 1000} s silence`}
              />
              <Row
                label="Segment length"
                value={`${health.live_stream?.min_segment_seconds ?? 1.2}–${health.live_stream?.max_segment_seconds ?? 12} s`}
              />
              <Row label="Target lag" value={<Badge tone="good">≤ 8 s behind speaker</Badge>} />
            </>
          )}
          <p className="mt-3 text-xs leading-relaxed text-ink-faint">
            Translate → Live: continuous microphone capture, silence-aware segmentation, and a
            pipelined STT → translate → TTS session. Benchmark offline with
            <code className="mx-1">localllm-live-bench</code>.
          </p>
        </Card>
      </div>
    </>
  );
}
