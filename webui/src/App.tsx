import { useEffect, useState } from "react";
import { Layout, type PageId } from "./components/Layout";
import { api, type Health, type Meta } from "./lib/api";
import { ChatPage } from "./pages/ChatPage";
import { DocumentsPage } from "./pages/DocumentsPage";
import { StatusPage } from "./pages/StatusPage";
import { TranscribePage } from "./pages/TranscribePage";
import { TranslatePage } from "./pages/TranslatePage";

export function App() {
  const [page, setPage] = useState<PageId>("translate");
  const [health, setHealth] = useState<Health | null>(null);
  const [meta, setMeta] = useState<Meta | null>(null);

  useEffect(() => {
    let cancelled = false;
    const poll = () =>
      api
        .health()
        .then((h) => !cancelled && setHealth(h))
        .catch(() => !cancelled && setHealth(null));
    poll();
    api
      .meta()
      .then((m) => !cancelled && setMeta(m))
      .catch(() => undefined);
    const interval = window.setInterval(poll, 15_000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, []);

  return (
    <Layout page={page} onNavigate={setPage} health={health}>
      {page === "chat" && <ChatPage health={health} />}
      {page === "translate" && <TranslatePage meta={meta} health={health} />}
      {page === "transcribe" && <TranscribePage meta={meta} />}
      {page === "documents" && <DocumentsPage />}
      {page === "status" && <StatusPage health={health} />}
    </Layout>
  );
}
