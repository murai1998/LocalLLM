export interface Health {
  gateway_ready: boolean;
  gateway_url: string;
  model: string;
  provider: string;
  platform: string;
  tts_available: boolean;
  translate_pipeline: string;
  live_chunking: {
    min_chunk_seconds: number;
    max_chunk_seconds: number;
    overlap_seconds: number;
  };
  live_stream?: {
    hangover_ms: number;
    min_segment_seconds: number;
    max_segment_seconds: number;
  };
}

export interface Meta {
  languages: Record<string, string>;
  tones: { id: string; label: string; hint: string }[];
  voices: Record<string, { id: string; label: string }[]>;
  default_target: string;
}

export interface ChunkEvent {
  type: "chunk";
  done: number;
  total: number;
  index: number;
  transcript: string;
  translation: string;
  elapsed_sec: number;
  start_sec: number;
  end_sec: number;
}

export interface ResultEvent {
  type: "result";
  transcript: string;
  translation: string;
  source_language: string;
  target_language: string;
  chunk_count: number;
  llm_elapsed_sec: number;
  tone: string;
}

export interface ErrorEvent {
  type: "error";
  message: string;
}

export type TranslateEvent = ChunkEvent | ResultEvent | ErrorEvent;

async function jsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body.detail) detail = body.detail;
    } catch {
      /* keep status text */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export interface SkillInfo {
  name: string;
  description: string;
}

export interface UploadedAttachment {
  id: string;
  name: string;
  kind: string;
  size: number;
}

export interface AgentStep {
  kind: "call" | "result";
  title: string;
  body: string;
}

export interface ChatResponse {
  reply: string;
  elapsed_sec: number;
  mode: "chat" | "agent";
  steps: AgentStep[];
}

export const api = {
  health: (): Promise<Health> => fetch("/api/health").then((r) => jsonOrThrow<Health>(r)),

  meta: (): Promise<Meta> => fetch("/api/meta").then((r) => jsonOrThrow<Meta>(r)),

  skills: (): Promise<SkillInfo[]> =>
    fetch("/api/skills").then((r) => jsonOrThrow<SkillInfo[]>(r)),

  uploadAttachments: (files: File[]): Promise<UploadedAttachment[]> => {
    const form = new FormData();
    for (const f of files) form.append("files", f, f.name);
    return fetch("/api/uploads", { method: "POST", body: form }).then((r) =>
      jsonOrThrow<UploadedAttachment[]>(r),
    );
  },

  deleteAttachment: (id: string): Promise<void> =>
    fetch(`/api/uploads/${id}`, { method: "DELETE" }).then(() => undefined),

  chat: (
    messages: { role: string; content: string }[],
    options?: { mode?: "chat" | "agent"; skills?: string[]; attachmentIds?: string[] },
  ) =>
    fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages,
        mode: options?.mode ?? "chat",
        skills: options?.skills ?? [],
        attachment_ids: options?.attachmentIds ?? [],
      }),
    }).then((r) => jsonOrThrow<ChatResponse>(r)),

  translateText: (body: {
    transcript: string;
    source_lang: string | null;
    target_lang: string;
    tone: string;
  }) =>
    fetch("/api/translate/text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(
      (r) =>
        jsonOrThrow<{
          transcript: string;
          translation: string;
          target_language: string;
          llm_elapsed_sec: number;
        }>(r),
    ),

  tts: async (text: string, language: string, voiceId: string | null): Promise<Blob> => {
    const res = await fetch("/api/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, language, voice_id: voiceId }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => null);
      throw new Error(body?.detail ?? `TTS failed (${res.status})`);
    }
    return res.blob();
  },

  transcribe: (file: Blob, filename: string, languageHint: string) => {
    const form = new FormData();
    form.append("file", file, filename);
    form.append("language_hint", languageHint);
    return fetch("/api/transcribe", { method: "POST", body: form }).then((r) =>
      jsonOrThrow<{ text: string; elapsed_sec: number }>(r),
    );
  },

  ocr: (file: File, instructions: string) => {
    const form = new FormData();
    form.append("file", file, file.name);
    form.append("instructions", instructions);
    return fetch("/api/ocr", { method: "POST", body: form }).then((r) =>
      jsonOrThrow<{
        source: string;
        mode: string;
        result: { full_text?: string; pages?: unknown[] };
        elapsed_sec: number;
      }>(r),
    );
  },

  /** POST audio and consume the NDJSON event stream. */
  translateAudio: async (
    file: Blob,
    filename: string,
    params: { source_lang: string; target_lang: string; tone: string },
    onEvent: (event: TranslateEvent) => void,
  ): Promise<void> => {
    const form = new FormData();
    form.append("file", file, filename);
    form.append("source_lang", params.source_lang);
    form.append("target_lang", params.target_lang);
    form.append("tone", params.tone);

    const res = await fetch("/api/translate/audio", { method: "POST", body: form });
    if (!res.ok || !res.body) {
      const body = await res.json().catch(() => null);
      throw new Error(body?.detail ?? `Translate failed (${res.status})`);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let newline = buffer.indexOf("\n");
      while (newline >= 0) {
        const line = buffer.slice(0, newline).trim();
        buffer = buffer.slice(newline + 1);
        if (line) onEvent(JSON.parse(line) as TranslateEvent);
        newline = buffer.indexOf("\n");
      }
    }
  },
};

export function fmtSeconds(s: number): string {
  return `${s.toFixed(1)}s`;
}
