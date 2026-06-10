import { Headphones, Mic, Square } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { fmtSeconds } from "../lib/api";
import { Badge, Button, Card, ErrorNote, cn } from "./ui";

type LiveStatus = "idle" | "connecting" | "live" | "stopping";

interface SegmentRow {
  index: number;
  startSec?: number;
  endSec?: number;
  transcript?: string;
  translation?: string;
  sttSec?: number;
  mtSec?: number;
  ttsSec?: number;
  lagSec?: number;
  error?: string;
}

const WORKLET_CODE = `
class PCMTap extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (channel) this.port.postMessage(channel.slice(0));
    return true;
  }
}
registerProcessor("pcm-tap", PCMTap);
`;

const TARGET_RATE = 16000;
const SEND_CHUNK = 4096; // ~256 ms at 16 kHz

function downsample(input: Float32Array, fromRate: number): Float32Array {
  if (fromRate === TARGET_RATE) return input;
  const ratio = fromRate / TARGET_RATE;
  const outLen = Math.floor(input.length / ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const pos = i * ratio;
    const left = Math.floor(pos);
    const frac = pos - left;
    out[i] = input[left] * (1 - frac) + (input[Math.min(left + 1, input.length - 1)] ?? 0) * frac;
  }
  return out;
}

function toInt16(input: Float32Array): ArrayBuffer {
  const out = new Int16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    out[i] = Math.max(-32768, Math.min(32767, Math.round(input[i] * 32767)));
  }
  return out.buffer;
}

export function LiveTranslate({
  sourceLang,
  targetLang,
  tone,
  voiceId,
  ttsSupported,
  gatewayReady,
}: {
  sourceLang: string;
  targetLang: string;
  tone: string;
  voiceId: string | null;
  ttsSupported: boolean;
  gatewayReady: boolean | undefined;
}) {
  const [status, setStatus] = useState<LiveStatus>("idle");
  const [rows, setRows] = useState<SegmentRow[]>([]);
  const [halfDuplex, setHalfDuplex] = useState(true);
  const [speaking, setSpeaking] = useState(false);
  const [micPaused, setMicPaused] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const captureCtxRef = useRef<AudioContext | null>(null);
  const playbackCtxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const bufferRef = useRef<Float32Array[]>([]);
  const bufferedRef = useRef(0);
  const nextPlayTimeRef = useRef(0);
  const activeSourcesRef = useRef(0);
  const speakingRef = useRef(false);
  const halfDuplexRef = useRef(true);
  const rafRef = useRef(0);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  halfDuplexRef.current = halfDuplex;

  useEffect(() => () => void teardown(), []);

  const updateRow = (index: number, patch: Partial<SegmentRow>) => {
    setRows((prev) => {
      const existing = prev.find((r) => r.index === index);
      if (existing) {
        return prev.map((r) => (r.index === index ? { ...r, ...patch } : r));
      }
      return [...prev, { index, ...patch }];
    });
  };

  const teardown = async () => {
    cancelAnimationFrame(rafRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    await captureCtxRef.current?.close().catch(() => undefined);
    captureCtxRef.current = null;
    await playbackCtxRef.current?.close().catch(() => undefined);
    playbackCtxRef.current = null;
    wsRef.current?.close();
    wsRef.current = null;
    bufferRef.current = [];
    bufferedRef.current = 0;
    activeSourcesRef.current = 0;
    speakingRef.current = false;
    setSpeaking(false);
    setMicPaused(false);
  };

  const playWav = async (base64: string) => {
    const ctx =
      playbackCtxRef.current ?? (playbackCtxRef.current = new AudioContext());
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    const audioBuffer = await ctx.decodeAudioData(bytes.buffer);
    const source = ctx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(ctx.destination);
    const startAt = Math.max(ctx.currentTime, nextPlayTimeRef.current);
    nextPlayTimeRef.current = startAt + audioBuffer.duration;
    activeSourcesRef.current += 1;
    speakingRef.current = true;
    setSpeaking(true);
    if (halfDuplexRef.current) setMicPaused(true);
    source.onended = () => {
      activeSourcesRef.current -= 1;
      if (activeSourcesRef.current <= 0) {
        speakingRef.current = false;
        setSpeaking(false);
        setMicPaused(false);
      }
    };
    source.start(startAt);
  };

  const drawMeter = (analyser: AnalyserNode) => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    const data = new Uint8Array(analyser.frequencyBinCount);
    const render = () => {
      analyser.getByteTimeDomainData(data);
      const { width, height } = canvas;
      ctx.clearRect(0, 0, width, height);
      const bars = 40;
      const step = Math.floor(data.length / bars);
      const barWidth = width / bars;
      for (let i = 0; i < bars; i++) {
        let peak = 0;
        for (let j = 0; j < step; j++) {
          peak = Math.max(peak, Math.abs(data[i * step + j] - 128) / 128);
        }
        const barHeight = Math.max(2, peak * height * 0.9);
        ctx.fillStyle = speakingRef.current && halfDuplexRef.current
          ? `rgba(120,128,150,${0.3 + peak * 0.4})`
          : `rgba(139,92,246,${0.35 + peak * 0.65})`;
        ctx.beginPath();
        ctx.roundRect(i * barWidth + 1, (height - barHeight) / 2, barWidth - 2, barHeight, 2);
        ctx.fill();
      }
      rafRef.current = requestAnimationFrame(render);
    };
    rafRef.current = requestAnimationFrame(render);
  };

  const start = async () => {
    setError(null);
    setRows([]);
    setStatus("connecting");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true },
      });
      streamRef.current = stream;

      let ctx: AudioContext;
      try {
        ctx = new AudioContext({ sampleRate: TARGET_RATE });
      } catch {
        ctx = new AudioContext();
      }
      captureCtxRef.current = ctx;
      const workletUrl = URL.createObjectURL(
        new Blob([WORKLET_CODE], { type: "application/javascript" }),
      );
      await ctx.audioWorklet.addModule(workletUrl);
      URL.revokeObjectURL(workletUrl);

      const sourceNode = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 1024;
      sourceNode.connect(analyser);
      drawMeter(analyser);

      const tap = new AudioWorkletNode(ctx, "pcm-tap");
      sourceNode.connect(tap);

      const protocol = location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(`${protocol}://${location.host}/ws/translate`);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      ws.onopen = () => {
        ws.send(
          JSON.stringify({
            type: "start",
            sample_rate: TARGET_RATE,
            source_lang: sourceLang,
            target_lang: targetLang,
            tone,
            voice_id: voiceId ?? "",
          }),
        );
        setStatus("live");
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data as string);
        switch (data.type) {
          case "segment":
            updateRow(data.index, { startSec: data.start_sec, endSec: data.end_sec });
            break;
          case "transcript":
            updateRow(data.index, { transcript: data.text, sttSec: data.elapsed_sec });
            break;
          case "translation":
            updateRow(data.index, { translation: data.text, mtSec: data.elapsed_sec });
            break;
          case "audio":
            updateRow(data.index, { ttsSec: data.elapsed_sec, lagSec: data.lag_sec });
            void playWav(data.wav_base64);
            break;
          case "error":
            if (data.index !== undefined) updateRow(data.index, { error: data.message });
            else setError(data.message);
            break;
          case "done":
            setStatus("idle");
            void teardown();
            break;
        }
      };
      ws.onerror = () => setError("Live connection failed.");
      ws.onclose = () => {
        setStatus((s) => (s === "live" || s === "connecting" ? "idle" : s));
      };

      tap.port.onmessage = (e: MessageEvent<Float32Array>) => {
        if (ws.readyState !== WebSocket.OPEN) return;
        if (halfDuplexRef.current && speakingRef.current) return; // half-duplex gate
        bufferRef.current.push(e.data);
        bufferedRef.current += e.data.length;
        if (bufferedRef.current >= SEND_CHUNK) {
          const merged = new Float32Array(bufferedRef.current);
          let off = 0;
          for (const part of bufferRef.current) {
            merged.set(part, off);
            off += part.length;
          }
          bufferRef.current = [];
          bufferedRef.current = 0;
          ws.send(toInt16(downsample(merged, ctx.sampleRate)));
        }
      };
    } catch (e) {
      await teardown();
      setStatus("idle");
      setError(
        e instanceof DOMException && e.name === "NotAllowedError"
          ? "Microphone access was denied. Allow it in the browser and retry."
          : `Could not start: ${e instanceof Error ? e.message : e}`,
      );
    }
  };

  const stop = () => {
    setStatus("stopping");
    cancelAnimationFrame(rafRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "stop" }));
      // teardown happens when the `done` event arrives
    } else {
      void teardown();
      setStatus("idle");
    }
  };

  const lastLag = [...rows].reverse().find((r) => r.lagSec !== undefined)?.lagSec;

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-4">
        {status === "idle" ? (
          <Button onClick={start} disabled={gatewayReady === false}>
            <Mic className="size-4" />
            Go live
          </Button>
        ) : (
          <Button onClick={stop} variant="danger" busy={status !== "live"}>
            <Square className="size-3.5 fill-current" />
            {status === "live" ? "Stop" : "Finishing…"}
          </Button>
        )}

        {status === "live" && (
          <span className="inline-flex items-center gap-2 text-sm text-ink-dim">
            <span
              className={cn(
                "size-2.5 rounded-full",
                speaking ? "bg-warn" : "bg-good pulse-soft",
              )}
            />
            {speaking
              ? micPaused
                ? "Speaking translation — mic paused"
                : "Speaking translation"
              : "Listening…"}
          </span>
        )}

        {lastLag !== undefined && (
          <Badge tone={lastLag <= 8 ? "good" : "warn"}>lag {fmtSeconds(lastLag)}</Badge>
        )}

        <label
          className="ml-auto flex cursor-pointer items-center gap-2 text-sm text-ink-dim"
          title="Pauses the microphone while translated speech plays, so it doesn't translate itself. Turn off when wearing headphones."
        >
          <input
            type="checkbox"
            checked={halfDuplex}
            onChange={(e) => setHalfDuplex(e.target.checked)}
            className="size-4 accent-(--color-accent)"
          />
          <Headphones className="size-4" />
          Pause mic while speaking
        </label>
      </div>

      {status !== "idle" && (
        <canvas
          ref={canvasRef}
          width={680}
          height={48}
          className="h-12 w-full rounded-lg border border-edge bg-panel-2"
        />
      )}

      {!ttsSupported && (
        <p className="text-xs text-warn">
          No local voice for this target language — you'll see translations but won't hear them.
        </p>
      )}

      {rows.length > 0 && (
        <Card title="Live transcript">
          <div className="flex max-h-96 flex-col gap-2 overflow-y-auto">
            {rows.map((row) => (
              <div
                key={row.index}
                className="rounded-lg border border-edge bg-panel-2 px-4 py-3 text-sm fade-up"
              >
                <div className="mb-1.5 flex flex-wrap items-center gap-2 text-xs text-ink-faint">
                  <Badge tone="accent">#{row.index + 1}</Badge>
                  {row.startSec !== undefined && (
                    <span>
                      {row.startSec.toFixed(1)}–{row.endSec?.toFixed(1)}s
                    </span>
                  )}
                  {row.sttSec !== undefined && <span>stt {fmtSeconds(row.sttSec)}</span>}
                  {row.mtSec !== undefined && <span>mt {fmtSeconds(row.mtSec)}</span>}
                  {row.ttsSec !== undefined && <span>tts {fmtSeconds(row.ttsSec)}</span>}
                  {row.lagSec !== undefined && (
                    <Badge tone={row.lagSec <= 8 ? "good" : "warn"}>
                      lag {fmtSeconds(row.lagSec)}
                    </Badge>
                  )}
                </div>
                <p className="text-ink-dim">{row.transcript ?? "…"}</p>
                <p className="mt-1 text-ink">{row.translation ?? ""}</p>
                {row.error && <p className="mt-1 text-xs text-bad">{row.error}</p>}
              </div>
            ))}
          </div>
        </Card>
      )}

      {error && <ErrorNote message={error} />}
    </div>
  );
}
