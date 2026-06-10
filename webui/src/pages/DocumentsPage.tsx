import { ClipboardCopy, FileScan, Upload } from "lucide-react";
import { useState } from "react";
import { Badge, Button, Card, ErrorNote, Field, Metric, Textarea } from "../components/ui";
import { api, fmtSeconds } from "../lib/api";

export function DocumentsPage() {
  const [file, setFile] = useState<File | null>(null);
  const [instructions, setInstructions] = useState("");
  const [busy, setBusy] = useState(false);
  const [text, setText] = useState("");
  const [mode, setMode] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const run = async () => {
    if (!file) return;
    setBusy(true);
    setError(null);
    setText("");
    setMode(null);
    try {
      const res = await api.ocr(file, instructions);
      setText(res.result.full_text ?? JSON.stringify(res.result, null, 2));
      setMode(res.mode);
      setElapsed(res.elapsed_sec);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const modeLabel: Record<string, string> = {
    pymupdf_text: "PDF text layer (fast, exact)",
    vision_ocr: "Gemma vision OCR",
    vision_ocr_pages: "Gemma vision OCR · per page",
  };

  return (
    <>
      <header className="fade-up">
        <h1 className="text-xl font-semibold">Documents</h1>
        <p className="mt-1 text-sm text-ink-dim">
          Extract text from PDFs and images. Digital PDFs use the text layer directly; scans and
          photos go through Gemma's local vision OCR.
        </p>
      </header>

      <Card>
        <div className="flex flex-col gap-4">
          <label className="flex w-fit cursor-pointer items-center gap-3 rounded-lg border border-dashed border-edge-2 px-5 py-3 text-sm text-ink-dim transition-colors hover:border-accent/50 hover:text-ink">
            <Upload className="size-4" />
            {file ? file.name : "Choose a PDF or image (png, jpg, webp…)"}
            <input
              type="file"
              accept=".pdf,.png,.jpg,.jpeg,.webp,.bmp,.tiff,image/*,application/pdf"
              className="hidden"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
          </label>

          <Field label="Instructions (optional)">
            <Textarea
              rows={2}
              value={instructions}
              onChange={(e) => setInstructions(e.target.value)}
              placeholder="e.g. extract only the table of invoice line items"
            />
          </Field>

          <div>
            <Button onClick={run} busy={busy} disabled={!file}>
              <FileScan className="size-4" />
              Extract text
            </Button>
          </div>
        </div>
      </Card>

      {text && (
        <Card
          title="Extracted text"
          actions={
            <div className="flex items-center gap-3">
              {mode && <Badge tone="accent">{modeLabel[mode] ?? mode}</Badge>}
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
            </div>
          }
        >
          <Textarea rows={14} value={text} readOnly className="font-mono text-xs" />
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
