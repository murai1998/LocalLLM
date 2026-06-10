import { Mic, Square, Trash2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Button } from "./ui";

export interface Recording {
  blob: Blob;
  filename: string;
  durationSec: number;
}

/** Pick a MediaRecorder mime type the backend's ffmpeg path can decode. */
function pickMimeType(): string {
  const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];
  for (const c of candidates) {
    if (typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported(c)) return c;
  }
  return "";
}

function extensionFor(mime: string): string {
  if (mime.includes("ogg")) return "ogg";
  if (mime.includes("mp4")) return "m4a";
  return "webm";
}

export function Recorder({
  recording,
  onChange,
  disabled,
}: {
  recording: Recording | null;
  onChange: (rec: Recording | null) => void;
  disabled?: boolean;
}) {
  const [isRecording, setIsRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const rafRef = useRef<number>(0);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const startedAtRef = useRef(0);
  const timerRef = useRef<number>(0);

  const cleanup = () => {
    cancelAnimationFrame(rafRef.current);
    window.clearInterval(timerRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    void audioCtxRef.current?.close().catch(() => undefined);
    audioCtxRef.current = null;
    recorderRef.current = null;
  };

  useEffect(() => cleanup, []);

  const drawMeter = (analyser: AnalyserNode) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const data = new Uint8Array(analyser.frequencyBinCount);

    const render = () => {
      analyser.getByteTimeDomainData(data);
      const { width, height } = canvas;
      ctx.clearRect(0, 0, width, height);
      const bars = 48;
      const step = Math.floor(data.length / bars);
      const barWidth = width / bars;
      for (let i = 0; i < bars; i++) {
        let peak = 0;
        for (let j = 0; j < step; j++) {
          peak = Math.max(peak, Math.abs(data[i * step + j] - 128) / 128);
        }
        const barHeight = Math.max(2, peak * height * 0.95);
        const x = i * barWidth;
        ctx.fillStyle = `rgba(139, 92, 246, ${0.35 + peak * 0.65})`;
        ctx.beginPath();
        ctx.roundRect(x + 1, (height - barHeight) / 2, barWidth - 2, barHeight, 2);
        ctx.fill();
      }
      rafRef.current = requestAnimationFrame(render);
    };
    rafRef.current = requestAnimationFrame(render);
  };

  const start = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true },
      });
      streamRef.current = stream;

      const audioCtx = new AudioContext();
      audioCtxRef.current = audioCtx;
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 1024;
      source.connect(analyser);
      drawMeter(analyser);

      const mime = pickMimeType();
      const recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
      recorderRef.current = recorder;
      const parts: BlobPart[] = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) parts.push(e.data);
      };
      recorder.onstop = () => {
        const durationSec = (Date.now() - startedAtRef.current) / 1000;
        const type = recorder.mimeType || "audio/webm";
        const blob = new Blob(parts, { type });
        onChange({
          blob,
          filename: `recording.${extensionFor(type)}`,
          durationSec,
        });
        cleanup();
      };

      startedAtRef.current = Date.now();
      setElapsed(0);
      timerRef.current = window.setInterval(
        () => setElapsed((Date.now() - startedAtRef.current) / 1000),
        200,
      );
      recorder.start();
      setIsRecording(true);
    } catch (e) {
      cleanup();
      setError(
        e instanceof DOMException && e.name === "NotAllowedError"
          ? "Microphone access was denied. Allow it in the browser and retry."
          : `Could not start recording: ${e instanceof Error ? e.message : e}`,
      );
    }
  };

  const stop = () => {
    setIsRecording(false);
    recorderRef.current?.stop();
  };

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-3">
        {!isRecording ? (
          <Button onClick={start} disabled={disabled} variant="secondary">
            <Mic className="size-4 text-accent" />
            Record
          </Button>
        ) : (
          <Button onClick={stop} variant="danger">
            <Square className="size-3.5 fill-current" />
            Stop · {elapsed.toFixed(0)}s
          </Button>
        )}
        {recording && !isRecording && (
          <>
            <audio controls src={URL.createObjectURL(recording.blob)} className="h-9 max-w-72" />
            <span className="text-xs text-ink-faint">{recording.durationSec.toFixed(1)}s</span>
            <Button variant="ghost" onClick={() => onChange(null)} title="Discard recording">
              <Trash2 className="size-4" />
            </Button>
          </>
        )}
      </div>
      {isRecording && (
        <canvas
          ref={canvasRef}
          width={560}
          height={56}
          className="h-14 w-full max-w-xl rounded-lg border border-edge bg-panel-2"
        />
      )}
      {error && <p className="text-sm text-bad">{error}</p>}
    </div>
  );
}
