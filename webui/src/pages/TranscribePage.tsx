import { ClipboardCopy, Download, Upload } from "lucide-react";
import { useMemo, useState } from "react";
import { Recorder, type Recording } from "../components/Recorder";
import { Button, Card, ErrorNote, Field, Metric, Select, Textarea, cn } from "../components/ui";
import { api, fmtSeconds, type Meta } from "../lib/api";

type InputMode = "record" | "upload";

export function TranscribePage({ meta }: { meta: Meta | null }) {
  const [mode, setMode] = useState<InputMode>("upload");
  const [recording, setRecording] = useState<Recording | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [language, setLanguage] = useState("");
  const [busy, setBusy] = useState(false);
  const [text, setText] = useState("");
  const [elapsed, setElapsed] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const languages = useMemo(
    () => Object.entries(meta?.languages ?? {}).sort((a, b) => a[1].localeCompare(b[1])),
    [meta],
  );

  const source =
    mode === "record"
      ? recording && { blob: recording.blob, filename: recording.filename }
      : file && { blob: file as Blob, filename: file.name };

  const run = async () => {
    if (!source) return;
    setBusy(true);
    setError(null);
    setText("");
    setElapsed(null);
    try {
      const hint = language ? (meta?.languages[language] ?? language) : "";
      const res = await api.transcribe(source.blob, source.filename, hint);
      setText(res.text);
      setElapsed(res.elapsed_sec);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const download = () => {
    const url = URL.createObjectURL(new Blob([text], { type: "text/plain" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = "transcript.txt";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <>
      <header className="fade-up">
        <h1 className="text-xl font-semibold">Transcribe</h1>
        <p className="mt-1 text-sm text-ink-dim">
          Speech → text with Gemma's native audio understanding. Long audio is chunked
          automatically.
        </p>
      </header>

      <Card>
        <div className="flex flex-wrap items-end gap-4">
          <div className="flex items-center gap-1 rounded-lg bg-panel-2 p-1">
            {(["upload", "record"] as InputMode[]).map((m) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                className={cn(
                  "rounded-md px-4 py-1.5 text-sm font-medium transition-colors cursor-pointer",
                  mode === m ? "bg-accent/20 text-accent" : "text-ink-dim hover:text-ink",
                )}
              >
                {m === "upload" ? "Upload file" : "Microphone"}
              </button>
            ))}
          </div>
          <Field label="Language hint">
            <Select value={language} onChange={(e) => setLanguage(e.target.value)}>
              <option value="">Auto</option>
              {languages.map(([code, label]) => (
                <option key={code} value={code}>
                  {label}
                </option>
              ))}
            </Select>
          </Field>
        </div>

        <div className="mt-4">
          {mode === "record" ? (
            <Recorder recording={recording} onChange={setRecording} disabled={busy} />
          ) : (
            <label className="flex w-fit cursor-pointer items-center gap-3 rounded-lg border border-dashed border-edge-2 px-5 py-3 text-sm text-ink-dim transition-colors hover:border-accent/50 hover:text-ink">
              <Upload className="size-4" />
              {file ? file.name : "Choose audio (wav, mp3, m4a, ogg, webm…)"}
              <input
                type="file"
                accept=".wav,.mp3,.m4a,.ogg,.flac,.webm,.aac,audio/*"
                className="hidden"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              />
            </label>
          )}
        </div>

        <div className="mt-5">
          <Button onClick={run} busy={busy} disabled={!source}>
            Transcribe
          </Button>
        </div>
      </Card>

      {text && (
        <Card
          title="Transcript"
          actions={
            <div className="flex gap-2">
              <Button
                variant="ghost"
                onClick={() => {
                  void navigator.clipboard.writeText(text);
                  setCopied(true);
                  window.setTimeout(() => setCopied(false), 1500);
                }}
              >
                <ClipboardCopy className="size-4" />
                {copied ? "Copied" : "Copy"}
              </Button>
              <Button variant="ghost" onClick={download}>
                <Download className="size-4" />
                .txt
              </Button>
            </div>
          }
        >
          <Textarea rows={12} value={text} onChange={(e) => setText(e.target.value)} />
          {elapsed !== null && (
            <div className="mt-3 w-44">
              <Metric label="Elapsed" value={fmtSeconds(elapsed)} />
            </div>
          )}
        </Card>
      )}

      {error && <ErrorNote message={error} />}
    </>
  );
}
