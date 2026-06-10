import { ArrowRight, Play, RefreshCw, Upload } from "lucide-react";
import { useMemo, useRef, useState } from "react";
import { Recorder, type Recording } from "../components/Recorder";
import {
  Badge,
  Button,
  Card,
  ErrorNote,
  Field,
  Metric,
  Select,
  Textarea,
  cn,
} from "../components/ui";
import {
  api,
  fmtSeconds,
  type ChunkEvent,
  type Health,
  type Meta,
  type ResultEvent,
} from "../lib/api";

type InputMode = "record" | "upload";

export function TranslatePage({ meta, health }: { meta: Meta | null; health: Health | null }) {
  const [mode, setMode] = useState<InputMode>("record");
  const [recording, setRecording] = useState<Recording | null>(null);
  const [uploadFile, setUploadFile] = useState<File | null>(null);

  const [sourceLang, setSourceLang] = useState("");
  const [targetLang, setTargetLang] = useState("");
  const [tone, setTone] = useState("professional");
  const [voiceId, setVoiceId] = useState<string | null>(null);
  const [autoSpeak, setAutoSpeak] = useState(true);

  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [chunks, setChunks] = useState<ChunkEvent[]>([]);
  const [transcript, setTranscript] = useState("");
  const [translation, setTranslation] = useState("");
  const [result, setResult] = useState<ResultEvent | null>(null);
  const [ttsElapsed, setTtsElapsed] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [ttsUrl, setTtsUrl] = useState<string | null>(null);

  const effectiveTarget = targetLang || meta?.default_target || "es";
  const voices = meta?.voices[effectiveTarget] ?? [];
  const ttsSupported = voices.length > 0;

  const languages = useMemo(
    () => Object.entries(meta?.languages ?? {}).sort((a, b) => a[1].localeCompare(b[1])),
    [meta],
  );

  const audioSource: { blob: Blob; filename: string } | null =
    mode === "record"
      ? recording && { blob: recording.blob, filename: recording.filename }
      : uploadFile && { blob: uploadFile, filename: uploadFile.name };

  const speak = async (text: string) => {
    if (!ttsSupported || !text.trim()) return;
    const started = performance.now();
    try {
      const blob = await api.tts(text, effectiveTarget, voiceId);
      setTtsElapsed((performance.now() - started) / 1000);
      const url = URL.createObjectURL(blob);
      setTtsUrl((old) => {
        if (old) URL.revokeObjectURL(old);
        return url;
      });
      requestAnimationFrame(() => audioRef.current?.play().catch(() => undefined));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const run = async () => {
    if (!audioSource) return;
    setBusy(true);
    setError(null);
    setChunks([]);
    setResult(null);
    setTranscript("");
    setTranslation("");
    setTtsElapsed(null);
    setProgress(null);

    const collectedT: string[] = [];
    const collectedX: string[] = [];
    try {
      await api.translateAudio(
        audioSource.blob,
        audioSource.filename,
        { source_lang: sourceLang, target_lang: effectiveTarget, tone },
        (event) => {
          if (event.type === "chunk") {
            setChunks((prev) => [...prev, event]);
            setProgress({ done: event.done, total: event.total });
            collectedT.push(event.transcript);
            collectedX.push(event.translation);
            setTranscript(collectedT.join(" "));
            setTranslation(collectedX.join(" "));
          } else if (event.type === "result") {
            setResult(event);
            setTranscript(event.transcript);
            setTranslation(event.translation);
            if (autoSpeak) void speak(event.translation);
          } else {
            setError(event.message);
          }
        },
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
      setProgress(null);
    }
  };

  const retranslate = async () => {
    if (!transcript.trim()) return;
    setBusy(true);
    setError(null);
    try {
      const res = await api.translateText({
        transcript,
        source_lang: sourceLang || null,
        target_lang: effectiveTarget,
        tone,
      });
      setTranslation(res.translation);
      setResult((prev) =>
        prev
          ? { ...prev, translation: res.translation, llm_elapsed_sec: res.llm_elapsed_sec }
          : prev,
      );
      if (autoSpeak) void speak(res.translation);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <header className="fade-up">
        <h1 className="text-xl font-semibold">Voice Translator</h1>
        <p className="mt-1 text-sm text-ink-dim">
          Speak or upload audio — Gemma transcribes and translates locally, Piper speaks the
          result. Nothing leaves this machine.
        </p>
      </header>

      <Card>
        <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
          <Field label="From">
            <Select value={sourceLang} onChange={(e) => setSourceLang(e.target.value)}>
              <option value="">Auto-detect</option>
              {languages.map(([code, label]) => (
                <option key={code} value={code}>
                  {label}
                </option>
              ))}
            </Select>
          </Field>
          <Field label="To">
            <Select value={effectiveTarget} onChange={(e) => setTargetLang(e.target.value)}>
              {languages.map(([code, label]) => (
                <option key={code} value={code}>
                  {label}
                </option>
              ))}
            </Select>
          </Field>
          <Field label="Tone">
            <Select value={tone} onChange={(e) => setTone(e.target.value)}>
              {(meta?.tones ?? []).map((t) => (
                <option key={t.id} value={t.id} title={t.hint}>
                  {t.label}
                </option>
              ))}
            </Select>
          </Field>
          <Field label="Voice">
            {ttsSupported ? (
              <Select value={voiceId ?? voices[0]?.id} onChange={(e) => setVoiceId(e.target.value)}>
                {voices.map((v) => (
                  <option key={v.id} value={v.id}>
                    {v.label}
                  </option>
                ))}
              </Select>
            ) : (
              <span className="rounded-lg border border-warn/30 bg-warn/10 px-3 py-2 text-xs text-warn">
                No local voice for this language
              </span>
            )}
          </Field>
        </div>

        <div className="mt-5 flex items-center gap-1 rounded-lg bg-panel-2 p-1 w-fit">
          {(["record", "upload"] as InputMode[]).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={cn(
                "rounded-md px-4 py-1.5 text-sm font-medium transition-colors cursor-pointer",
                mode === m ? "bg-accent/20 text-accent" : "text-ink-dim hover:text-ink",
              )}
            >
              {m === "record" ? "Microphone" : "Upload file"}
            </button>
          ))}
        </div>

        <div className="mt-4">
          {mode === "record" ? (
            <Recorder recording={recording} onChange={setRecording} disabled={busy} />
          ) : (
            <label className="flex w-fit cursor-pointer items-center gap-3 rounded-lg border border-dashed border-edge-2 px-5 py-3 text-sm text-ink-dim transition-colors hover:border-accent/50 hover:text-ink">
              <Upload className="size-4" />
              {uploadFile ? uploadFile.name : "Choose audio (wav, mp3, m4a, ogg, webm…)"}
              <input
                type="file"
                accept=".wav,.mp3,.m4a,.ogg,.flac,.webm,.aac,audio/*"
                className="hidden"
                onChange={(e) => setUploadFile(e.target.files?.[0] ?? null)}
              />
            </label>
          )}
        </div>

        <div className="mt-5 flex flex-wrap items-center gap-4">
          <Button onClick={run} busy={busy} disabled={!audioSource || health?.gateway_ready === false}>
            Translate
            <ArrowRight className="size-4" />
          </Button>
          {transcript && (
            <Button variant="secondary" onClick={retranslate} disabled={busy}>
              <RefreshCw className="size-4" />
              Re-translate
            </Button>
          )}
          <label className="flex cursor-pointer items-center gap-2 text-sm text-ink-dim">
            <input
              type="checkbox"
              checked={autoSpeak}
              onChange={(e) => setAutoSpeak(e.target.checked)}
              className="size-4 accent-(--color-accent)"
            />
            Speak translation automatically
          </label>
          {health?.gateway_ready === false && (
            <Badge tone="warn">Gateway offline — start localllm-serve</Badge>
          )}
        </div>

        {progress && (
          <div className="mt-5">
            <div className="mb-1.5 flex justify-between text-xs text-ink-faint">
              <span>
                Chunk {progress.done} / {progress.total}
              </span>
              <span>{Math.round((progress.done / progress.total) * 100)}%</span>
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-panel-2">
              <div
                className="h-full rounded-full bg-gradient-to-r from-accent-2 to-accent transition-all duration-300"
                style={{ width: `${(progress.done / progress.total) * 100}%` }}
              />
            </div>
          </div>
        )}
      </Card>

      {(transcript || translation) && (
        <div className="grid gap-5 md:grid-cols-2">
          <Card title="Transcript">
            <Textarea
              rows={9}
              value={transcript}
              onChange={(e) => setTranscript(e.target.value)}
              placeholder="Transcript appears here…"
            />
          </Card>
          <Card
            title="Translation"
            actions={
              ttsSupported &&
              translation && (
                <Button variant="ghost" onClick={() => void speak(translation)} title="Speak">
                  <Play className="size-4" />
                </Button>
              )
            }
          >
            <Textarea
              rows={9}
              value={translation}
              readOnly
              placeholder="Translation appears here…"
            />
            {ttsUrl && (
              <audio ref={audioRef} controls src={ttsUrl} className="mt-3 h-9 w-full" />
            )}
          </Card>
        </div>
      )}

      {result && (
        <Card title="Timing">
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Metric label="Chunks" value={String(result.chunk_count)} />
            <Metric label="STT + translate" value={fmtSeconds(result.llm_elapsed_sec)} />
            {ttsElapsed !== null && <Metric label="TTS" value={fmtSeconds(ttsElapsed)} />}
            <Metric
              label="Avg per chunk"
              value={
                chunks.length
                  ? fmtSeconds(chunks.reduce((s, c) => s + c.elapsed_sec, 0) / chunks.length)
                  : "—"
              }
            />
          </div>
          {chunks.length > 0 && (
            <details className="mt-4">
              <summary className="cursor-pointer text-sm text-ink-dim hover:text-ink">
                Chunk details
              </summary>
              <div className="mt-3 flex flex-col gap-2">
                {chunks.map((c) => (
                  <div key={c.index} className="rounded-lg border border-edge bg-panel-2 px-4 py-3 text-sm">
                    <div className="mb-1 flex gap-2 text-xs text-ink-faint">
                      <Badge tone="accent">#{c.index + 1}</Badge>
                      <span>
                        {c.start_sec.toFixed(1)}–{c.end_sec.toFixed(1)}s · {fmtSeconds(c.elapsed_sec)}
                      </span>
                    </div>
                    <p className="text-ink-dim">{c.transcript}</p>
                    <p className="mt-1 text-ink">{c.translation}</p>
                  </div>
                ))}
              </div>
            </details>
          )}
        </Card>
      )}

      {error && <ErrorNote message={error} />}
    </>
  );
}
