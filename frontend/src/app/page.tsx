"use client";

import Link from "next/link";
import {
  ChangeEvent,
  DragEvent,
  FormEvent,
  KeyboardEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import {
  AlertIcon,
  ArrowRightIcon,
  ChartIcon,
  FileIcon,
  GlobeIcon,
  PaperclipIcon,
  PulseIcon,
  RefreshIcon,
  ShieldIcon,
  SparkIcon,
  TrashIcon,
  UploadIcon,
  XIcon,
} from "@/components/icons";
import {
  API_URL,
  EMPTY_SUMMARY,
  ObservabilityData,
  RAG_ACTIVITY_EVENT,
  RAG_NEW_QUESTION_EVENT,
  emit,
  fetchObservability,
  timeAgo,
} from "@/lib/api";

type DocumentItem = {
  document_id: string;
  file_name: string;
  file_type?: string;
  source?: string;
  chunk_count?: number;
};

type Source = {
  type: string;
  file_name?: string;
  source?: string;
  document_id?: string;
  title?: string;
  url?: string;
};

type ChatResponse = {
  answer?: string;
  generation?: string;
  response?: string;
  route?: string;
  sources?: Source[];
  retrieved_documents?: number;
  relevant_documents?: number;
  grounded?: boolean;
  answers_question?: boolean;
  retry_count?: number;
  latency_seconds?: number;
  hitl_status?: string;
  hitl_reason?: string;
  security_status?: string;
  security_reason?: string;
  security_redactions?: number | unknown[];
  web_search?: boolean;
  [key: string]: unknown;
};

type DocFilter = "all" | "pdf" | "docx" | "text";

const FILTERS: { id: DocFilter; label: string }[] = [
  { id: "all", label: "All" },
  { id: "pdf", label: "PDF" },
  { id: "docx", label: "DOCX" },
  { id: "text", label: "Text" },
];

// Warm poster gradients for document cards (reference: project thumbnails).
const POSTERS = [
  "from-[#8a3b12] via-[#c4652a] to-[#2a1408]",
  "from-[#5b4636] via-[#a58b72] to-[#1d1712]",
  "from-[#7a2e1a] via-[#d48a4a] to-[#24110a]",
  "from-[#3e2a22] via-[#8f5a3c] to-[#140d0a]",
  "from-[#a2461a] via-[#e3a15e] to-[#311707]",
  "from-[#4a3b33] via-[#6f5546] to-[#120e0c]",
];

function hashIndex(value: string, size: number) {
  let hash = 0;
  for (let i = 0; i < value.length; i += 1) {
    hash = (hash * 31 + value.charCodeAt(i)) | 0;
  }
  return Math.abs(hash) % size;
}

function fileKind(doc: DocumentItem): DocFilter {
  const ext = (doc.file_type || doc.file_name.split(".").pop() || "")
    .replace(".", "")
    .toLowerCase();
  if (ext === "pdf") return "pdf";
  if (ext === "docx") return "docx";
  return "text";
}

function fileExt(name: string) {
  return (name.split(".").pop() || "doc").toUpperCase();
}

function greeting() {
  const hour = new Date().getHours();
  if (hour < 12) return "Good morning";
  if (hour < 18) return "Good afternoon";
  return "Good evening";
}

export default function Home() {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [question, setQuestion] = useState("");
  const [submittedQuestion, setSubmittedQuestion] = useState("");
  const [response, setResponse] = useState<ChatResponse | null>(null);

  const [loadingDocuments, setLoadingDocuments] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [asking, setAsking] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const [pendingApproval, setPendingApproval] = useState(false);
  const [approvalReason, setApprovalReason] = useState("");
  const [threadId, setThreadId] = useState<string | null>(null);

  const [filter, setFilter] = useState<DocFilter>("all");
  const [dragging, setDragging] = useState(false);
  const [activity, setActivity] = useState<ObservabilityData>({
    runs: [],
    summary: EMPTY_SUMMARY,
  });
  const [hello, setHello] = useState("Welcome back");

  const fileInputRef = useRef<HTMLInputElement>(null);
  const questionRef = useRef<HTMLTextAreaElement>(null);

  const loadActivity = useCallback(async () => {
    try {
      setActivity(await fetchObservability());
    } catch {
      /* activity widgets are best-effort */
    }
  }, []);

  useEffect(() => {
    const timeoutId = window.setTimeout(() => {
      setHello(greeting());
      loadDocuments();
      loadActivity();
    }, 0);

    function onNewQuestion() {
      setResponse(null);
      setSubmittedQuestion("");
      setPendingApproval(false);
      setThreadId(null);
      setQuestion("");
      questionRef.current?.focus();
    }

    window.addEventListener(RAG_NEW_QUESTION_EVENT, onNewQuestion);
    return () => {
      window.clearTimeout(timeoutId);
      window.removeEventListener(RAG_NEW_QUESTION_EVENT, onNewQuestion);
    };
  }, [loadActivity]);

  async function loadDocuments() {
    try {
      setLoadingDocuments(true);
      setError("");

      const res = await fetch(`${API_URL}/documents`, {
        method: "GET",
        cache: "no-store",
      });

      const data = await res.json().catch(() => ({}));

      if (!res.ok) {
        throw new Error(data.detail || "Failed to load documents.");
      }

      setDocuments(Array.isArray(data) ? data : data.documents || []);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Unable to connect to the backend."
      );
    } finally {
      setLoadingDocuments(false);
    }
  }

  function addFiles(files: File[]) {
    if (!files.length) return;
    const allowed = files.filter((file) =>
      /\.(pdf|docx|txt|md)$/i.test(file.name)
    );
    if (allowed.length < files.length) {
      setError("Some files were skipped. Supported: PDF, DOCX, TXT, Markdown.");
    } else {
      setError("");
    }
    setSelectedFiles((current) => [...current, ...allowed].slice(0, 5));
    setSuccess("");
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    addFiles(Array.from(event.target.files || []));
    event.target.value = "";
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    addFiles(Array.from(event.dataTransfer.files || []));
  }

  function removeSelectedFile(index: number) {
    setSelectedFiles((current) =>
      current.filter((_, fileIndex) => fileIndex !== index)
    );
  }

  function clearFileSelection() {
    setSelectedFiles([]);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  async function uploadFiles() {
    if (!selectedFiles.length) {
      setError("Please select at least one document.");
      return;
    }

    try {
      setUploading(true);
      setError("");
      setSuccess("");

      const formData = new FormData();
      selectedFiles.forEach((file) => {
        formData.append("files", file);
      });

      // Do NOT set Content-Type: the browser adds the multipart boundary.
      const res = await fetch(`${API_URL}/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json().catch(() => ({}));

      if (!res.ok) {
        throw new Error(
          data.detail || data.message || `Upload failed with status ${res.status}.`
        );
      }

      const rejected: { file_name: string; reason: string }[] =
        Array.isArray(data.rejected) ? data.rejected : [];

      setSuccess(
        [
          data.message ||
            `${selectedFiles.length} document${selectedFiles.length > 1 ? "s" : ""} uploaded.`,
          ...rejected.map((item) => `${item.file_name}: ${item.reason}`),
        ].join(" ")
      );

      clearFileSelection();
      await loadDocuments();
      emit(RAG_ACTIVITY_EVENT);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to upload documents.");
    } finally {
      setUploading(false);
    }
  }

  async function deleteDocument(documentId: string) {
    try {
      setDeletingId(documentId);
      setError("");
      setSuccess("");

      const res = await fetch(`${API_URL}/documents/${documentId}`, {
        method: "DELETE",
      });

      const data = await res.json().catch(() => ({}));

      if (!res.ok) {
        throw new Error(data.detail || "Failed to delete document.");
      }

      setSuccess(data.message || "Document deleted successfully.");
      await loadDocuments();
      emit(RAG_ACTIVITY_EVENT);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to delete document.");
    } finally {
      setDeletingId(null);
    }
  }

  async function askQuestion(event?: FormEvent) {
    event?.preventDefault();

    const trimmedQuestion = question.trim();

    if (!trimmedQuestion) {
      setError("Please enter a question.");
      return;
    }

    try {
      setAsking(true);
      setError("");
      setSuccess("");
      setResponse(null);
      setPendingApproval(false);
      setApprovalReason("");
      setThreadId(null);
      setSubmittedQuestion(trimmedQuestion);

      const res = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmedQuestion }),
      });

      const data: ChatResponse = await res.json().catch(() => ({}));

      if (!res.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "Failed to process the question."
        );
      }

      setQuestion("");
      setResponse(data);

      if (typeof data.thread_id === "string") {
        setThreadId(data.thread_id);
      }

      // Human-in-the-loop: backend may request approval before web search.
      const hitlStatus =
        typeof data.hitl_status === "string" ? data.hitl_status.toLowerCase() : "";

      if (
        hitlStatus === "pending" ||
        hitlStatus === "awaiting_approval" ||
        hitlStatus === "approval_required"
      ) {
        setPendingApproval(true);
        setApprovalReason(
          typeof data.hitl_reason === "string"
            ? data.hitl_reason
            : "Web search approval is required."
        );
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to process your question.");
    } finally {
      setAsking(false);
      loadActivity();
      emit(RAG_ACTIVITY_EVENT);
    }
  }

  async function submitApproval(approved: boolean) {
    if (!threadId) {
      setError("Approval session expired. Please ask the question again.");
      return;
    }

    try {
      setAsking(true);
      setError("");
      setSuccess("");

      const res = await fetch(`${API_URL}/chat/approval`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ thread_id: threadId, approved }),
      });

      const data: ChatResponse = await res.json().catch(() => ({}));

      if (!res.ok) {
        throw new Error(
          typeof data.detail === "string" ? data.detail : "Failed to submit approval."
        );
      }

      setResponse(data);
      setPendingApproval(false);
      setApprovalReason("");

      if (typeof data.thread_id === "string") {
        setThreadId(data.thread_id);
      }
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Unable to submit approval decision."
      );
    } finally {
      setAsking(false);
      loadActivity();
      emit(RAG_ACTIVITY_EVENT);
    }
  }

  function handleQuestionKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      askQuestion();
    }
  }

  function getAnswer() {
    if (!response) return "";
    return (
      response.answer ||
      response.generation ||
      response.response ||
      "No answer was returned."
    );
  }

  const filteredDocuments = useMemo(
    () =>
      filter === "all"
        ? documents
        : documents.filter((doc) => fileKind(doc) === filter),
    [documents, filter]
  );

  const redactions = Array.isArray(response?.security_redactions)
    ? response.security_redactions.length
    : typeof response?.security_redactions === "number"
      ? response.security_redactions
      : 0;

  const showAnswer = Boolean(response) && !pendingApproval;

  return (
    <div className="grid gap-3 xl:grid-cols-[minmax(0,1fr)_340px]">
      {/* ================= CENTER ================= */}
      <main className="min-w-0 px-1 pb-10 sm:px-4 lg:px-6">
        {/* Top bar */}
        <div className="flex items-center justify-between py-3">
          <p className="text-[13px] text-faint">
            Workspace <span className="mx-1.5">/</span>
            <span className="text-muted">Ask</span>
          </p>
          <div className="flex items-center gap-2">
            <span className="glass inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs">
              <SparkIcon size={13} className="text-accent" />
              <span className="tabular-nums">{documents.length}</span>
              <span className="text-faint">docs</span>
            </span>
            <button
              type="button"
              onClick={() => {
                loadDocuments();
                loadActivity();
              }}
              aria-label="Refresh"
              className="glass flex h-8 w-8 items-center justify-center rounded-full text-muted transition hover:text-ink"
            >
              <RefreshIcon size={14} />
            </button>
          </div>
        </div>

        {/* Hero */}
        <section className="pt-8 sm:pt-12">
          <span className="glass inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs text-muted">
            <span className="h-1.5 w-1.5 rounded-full bg-accent" />
            {hello}
            {documents.length > 0 && (
              <span className="text-faint">· {documents.length} in knowledge base</span>
            )}
          </span>
          <h1 className="mt-5 max-w-xl text-4xl font-normal leading-[1.08] tracking-tight sm:text-[3.4rem]">
            What do you want to know today?
          </h1>
        </section>

        {/* Prompt */}
        <form onSubmit={askQuestion} className="mt-8">
          <div className="glass rounded-2xl shadow-2xl shadow-black/40">
            <textarea
              ref={questionRef}
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={handleQuestionKeyDown}
              placeholder="Ask about your documents, or anything that may need the web…"
              rows={4}
              disabled={asking}
              aria-label="Question"
              className="w-full resize-none bg-transparent px-5 pt-5 pb-2 text-[15px] leading-6 text-ink outline-none placeholder:text-faint disabled:opacity-50"
            />
            <div className="flex flex-wrap items-center gap-2 px-4 pb-4">
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                aria-label="Attach documents"
                className="flex h-8 w-8 items-center justify-center rounded-lg border border-line bg-surface-raised text-muted transition hover:text-ink"
              >
                <PaperclipIcon size={14} />
              </button>
              <Chip>Adaptive routing</Chip>
              <Chip>Grounding check</Chip>
              <Chip accent>HITL web search</Chip>
              <div className="ml-auto flex items-center gap-3">
              <span className="hidden text-[11px] text-faint 2xl:block">
                Enter to ask · Shift+Enter for new line
              </span>
              <button
                type="submit"
                disabled={asking || !question.trim()}
                className="inline-flex items-center gap-2 rounded-xl bg-[#ece6e0] px-4 py-2 text-sm font-medium text-[#1a1412] shadow-lg shadow-black/30 transition hover:bg-white disabled:cursor-not-allowed disabled:opacity-40"
              >
                <SparkIcon size={14} />
                {asking ? "Thinking…" : "Ask agent"}
              </button>
              </div>
            </div>
            {asking && <ProgressBar />}
          </div>
        </form>

        {/* Quick actions */}
        <div className="mt-4 grid grid-cols-2 gap-2.5 2xl:grid-cols-4">
          <QuickAction
            icon={<SparkIcon size={16} />}
            title="Ask documents"
            subtitle="Grounded answers"
            onClick={() => questionRef.current?.focus()}
          />
          <QuickAction
            icon={<UploadIcon size={16} />}
            title="Upload files"
            subtitle="PDF, DOCX, TXT, MD"
            onClick={() => fileInputRef.current?.click()}
          />
          <QuickAction
            icon={<ChartIcon size={16} />}
            title="Evaluation"
            subtitle="Quality over time"
            href="/evaluation"
          />
          <QuickAction
            icon={<PulseIcon size={16} />}
            title="Observability"
            subtitle="Every run, traced"
            href="/observability"
          />
        </div>

        {/* Alerts */}
        {error && (
          <div role="alert" className="mt-5 flex items-start gap-2.5 rounded-xl border border-rose-500/25 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
            <AlertIcon size={16} className="mt-0.5 shrink-0" />
            {error}
          </div>
        )}
        {success && (
          <div role="status" className="mt-5 rounded-xl border border-emerald-500/20 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-200">
            {success}
          </div>
        )}

        {/* HITL approval */}
        {pendingApproval && (
          <section className="mt-6 rounded-2xl border border-accent/30 bg-gradient-to-br from-accent/15 to-transparent p-5">
            <div className="flex items-start gap-3">
              <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-accent-soft text-accent">
                <GlobeIcon size={17} />
              </span>
              <div className="flex-1">
                <h3 className="font-medium">Web search needs your approval</h3>
                {submittedQuestion && (
                  <p className="mt-1 text-sm text-muted">“{submittedQuestion}”</p>
                )}
                <p className="mt-2 text-sm leading-6 text-muted">
                  {approvalReason ||
                    "Local evidence looks insufficient, so the agent wants to search the web."}
                </p>
                <div className="mt-4 flex gap-2.5">
                  <button
                    type="button"
                    onClick={() => submitApproval(true)}
                    disabled={asking}
                    className="rounded-xl bg-accent px-4 py-2 text-sm font-medium text-[#1a0d05] transition hover:bg-accent-strong disabled:opacity-50"
                  >
                    Approve web search
                  </button>
                  <button
                    type="button"
                    onClick={() => submitApproval(false)}
                    disabled={asking}
                    className="rounded-xl border border-line-strong px-4 py-2 text-sm text-muted transition hover:text-ink disabled:opacity-50"
                  >
                    Answer without it
                  </button>
                </div>
              </div>
            </div>
          </section>
        )}

        {/* Answer */}
        {showAnswer && response && (
          <section className="mt-10">
            <SectionHeader
              title="Latest answer"
              right={
                response.route ? (
                  <span className="rounded-full border border-line bg-surface-raised px-2.5 py-1 text-[10px] uppercase tracking-wider text-muted">
                    {String(response.route)}
                  </span>
                ) : null
              }
            />

            <div className="glass rounded-2xl p-6">
              {submittedQuestion && (
                <p className="mb-4 border-b border-line pb-4 text-sm text-muted">
                  {submittedQuestion}
                </p>
              )}
              <div className="whitespace-pre-wrap text-[15px] leading-7 text-ink/90">
                {getAnswer()}
              </div>

              <div className="mt-6 flex flex-wrap gap-2">
                <Flag ok={response.grounded} label="Grounded" />
                <Flag ok={response.answers_question} label="Answers question" />
                {response.security_status && (
                  <span
                    title={response.security_reason || undefined}
                    className="inline-flex items-center gap-1.5 rounded-full border border-line bg-surface-raised px-2.5 py-1 text-xs text-muted"
                  >
                    <ShieldIcon size={12} />
                    {response.security_status}
                    {redactions > 0 && ` · ${redactions} redacted`}
                  </span>
                )}
              </div>
            </div>

            <div className="mt-3 grid grid-cols-2 gap-2.5 md:grid-cols-4">
              <Metric label="Retrieved" value={response.retrieved_documents ?? 0} />
              <Metric label="Relevant" value={response.relevant_documents ?? 0} />
              <Metric label="Retries" value={response.retry_count ?? 0} />
              <Metric
                label="Latency"
                value={
                  typeof response.latency_seconds === "number"
                    ? `${response.latency_seconds.toFixed(2)}s`
                    : "—"
                }
              />
            </div>

            {Array.isArray(response.sources) && response.sources.length > 0 && (
              <div className="mt-6">
                <p className="mb-3 text-xs font-medium uppercase tracking-[0.16em] text-faint">
                  Sources
                </p>
                <div className="grid gap-2.5 sm:grid-cols-2">
                  {response.sources.map((source, index) => (
                    <SourceCard key={`${source.url || source.source || index}-${index}`} source={source} index={index} />
                  ))}
                </div>
              </div>
            )}
          </section>
        )}

        {/* Knowledge base */}
        <section className="mt-12">
          <SectionHeader
            title="Knowledge base"
            left={
              <div className="ml-3 flex gap-1" role="tablist" aria-label="Filter documents">
                {FILTERS.map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    role="tab"
                    aria-selected={filter === item.id}
                    onClick={() => setFilter(item.id)}
                    className={`rounded-lg px-2.5 py-1 text-xs transition ${
                      filter === item.id
                        ? "bg-ink text-bg"
                        : "text-muted hover:bg-surface-raised hover:text-ink"
                    }`}
                  >
                    {item.label}
                  </button>
                ))}
              </div>
            }
            right={
              <button
                type="button"
                onClick={loadDocuments}
                disabled={loadingDocuments}
                className="inline-flex items-center gap-1.5 text-xs text-muted transition hover:text-ink"
              >
                {loadingDocuments ? "Loading…" : "Refresh"}
                <ArrowRightIcon size={12} />
              </button>
            }
          />

          {loadingDocuments && documents.length === 0 ? (
            <div className="grid grid-cols-2 gap-3 md:grid-cols-3 2xl:grid-cols-4">
              {[0, 1, 2].map((i) => (
                <div key={i} className="aspect-[4/5] animate-pulse rounded-2xl bg-surface-raised" />
              ))}
            </div>
          ) : filteredDocuments.length === 0 ? (
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="flex w-full flex-col items-center justify-center rounded-2xl border border-dashed border-line-strong px-6 py-14 text-center transition hover:border-accent/50"
            >
              <UploadIcon size={22} className="text-accent" />
              <p className="mt-3 text-sm">
                {documents.length === 0 ? "No documents yet" : "No documents of this type"}
              </p>
              <p className="mt-1 text-xs text-faint">
                Upload PDF, DOCX, TXT or Markdown to build your knowledge base.
              </p>
            </button>
          ) : (
            <div className="grid grid-cols-2 gap-3 md:grid-cols-3 2xl:grid-cols-4">
              {filteredDocuments.map((doc) => (
                <DocumentCard
                  key={doc.document_id}
                  doc={doc}
                  deleting={deletingId === doc.document_id}
                  onDelete={() => deleteDocument(doc.document_id)}
                />
              ))}
            </div>
          )}
        </section>
      </main>

      {/* ================= RIGHT RAIL ================= */}
      <aside className="space-y-3 px-1 pb-6 sm:px-4 xl:px-0 xl:pt-3 xl:pr-0">
        {/* Ingestion queue (reference: render queue) */}
        <div className="glass rounded-2xl p-4">
          <div className="mb-3 flex items-center justify-between">
            <p className="text-sm font-medium">Ingestion queue</p>
            <span className="text-xs text-accent">
              {selectedFiles.length ? `${selectedFiles.length} ready` : "idle"}
            </span>
          </div>

          <div
            onDragOver={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                fileInputRef.current?.click();
              }
            }}
            role="button"
            tabIndex={0}
            aria-label="Choose or drop files to upload"
            className={`relative flex h-32 cursor-pointer flex-col items-center justify-center overflow-hidden rounded-xl bg-gradient-to-br from-[#a44a17] via-[#d9894a] to-[#3a1a0a] text-center transition ${
              dragging ? "ring-2 ring-accent" : ""
            }`}
          >
            <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent" />
            <span className="absolute top-2.5 left-2.5 inline-flex items-center gap-1.5 rounded-full bg-black/40 px-2 py-0.5 text-[10px] backdrop-blur">
              <span className={`h-1.5 w-1.5 rounded-full ${uploading ? "bg-accent" : "bg-emerald-400"}`} />
              {uploading ? "Indexing" : "Ready"}
            </span>
            <UploadIcon size={20} className="relative" />
            <p className="relative mt-2 text-sm font-medium">
              {dragging ? "Drop to add" : "Drop files or browse"}
            </p>
            <p className="relative text-[11px] text-white/70">Up to 5 files per upload</p>
          </div>

          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf,.docx,.txt,.md"
            onChange={handleFileChange}
            className="hidden"
          />

          {uploading && (
            <div className="mt-3">
              <div className="flex justify-between text-xs text-muted">
                <span>Parsing, chunking & embedding…</span>
              </div>
              <ProgressBar className="mt-2 rounded-full" />
            </div>
          )}

          {selectedFiles.length > 0 && (
            <ul className="mt-3 space-y-1.5">
              {selectedFiles.map((file, index) => (
                <li
                  key={`${file.name}-${index}`}
                  className="flex items-center gap-2.5 rounded-lg px-1 py-1"
                >
                  <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-surface-raised text-[9px] font-semibold text-muted">
                    {fileExt(file.name)}
                  </span>
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-[13px]">{file.name}</p>
                    <p className="text-[11px] text-faint">{(file.size / 1024).toFixed(1)} KB</p>
                  </div>
                  <button
                    type="button"
                    onClick={() => removeSelectedFile(index)}
                    disabled={uploading}
                    aria-label={`Remove ${file.name}`}
                    className="text-faint transition hover:text-rose-300"
                  >
                    <XIcon size={14} />
                  </button>
                </li>
              ))}
            </ul>
          )}

          <button
            type="button"
            onClick={uploadFiles}
            disabled={uploading || selectedFiles.length === 0}
            className="mt-3 flex w-full items-center justify-center gap-2 rounded-xl border border-line-strong bg-surface-raised py-2.5 text-sm transition hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {uploading ? "Indexing…" : "Upload & index"}
            <ArrowRightIcon size={14} />
          </button>
        </div>

        <ActivityCard data={activity} />
        <QualityCard data={activity} />
        <RecentRuns data={activity} />
      </aside>
    </div>
  );
}

/* ============================== pieces ============================== */

function Chip({ children, accent = false }: { children: React.ReactNode; accent?: boolean }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs ${
        accent
          ? "border-accent/25 bg-accent-soft text-accent-strong"
          : "border-line bg-surface-raised text-muted"
      }`}
    >
      {accent && <span className="h-1.5 w-1.5 rounded-full bg-accent" />}
      {children}
    </span>
  );
}

function ProgressBar({ className = "" }: { className?: string }) {
  return (
    <div className={`h-0.5 overflow-hidden bg-line ${className}`} role="progressbar" aria-label="Working">
      <div className="progress-indeterminate h-full w-2/5 bg-gradient-to-r from-transparent via-accent to-transparent" />
    </div>
  );
}

function QuickAction({
  icon,
  title,
  subtitle,
  href,
  onClick,
}: {
  icon: React.ReactNode;
  title: string;
  subtitle: string;
  href?: string;
  onClick?: () => void;
}) {
  const body = (
    <>
      <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-surface-raised text-ink">
        {icon}
      </span>
      <span className="min-w-0 flex-1 text-left">
        <span className="block truncate text-[13px] font-medium">{title}</span>
        <span className="block truncate text-[11px] text-faint">{subtitle}</span>
      </span>
      <ArrowRightIcon size={14} className="shrink-0 text-faint transition group-hover:translate-x-0.5 group-hover:text-ink" />
    </>
  );
  const cls =
    "glass group flex items-center gap-3 rounded-xl p-3 transition hover:border-line-strong";
  return href ? (
    <Link href={href} className={cls}>
      {body}
    </Link>
  ) : (
    <button type="button" onClick={onClick} className={cls}>
      {body}
    </button>
  );
}

function SectionHeader({
  title,
  left,
  right,
}: {
  title: string;
  left?: React.ReactNode;
  right?: React.ReactNode;
}) {
  return (
    <div className="mb-4 flex items-center justify-between gap-3">
      <div className="flex items-center">
        <h2 className="text-lg font-medium">{title}</h2>
        {left}
      </div>
      {right}
    </div>
  );
}

function Flag({ ok, label }: { ok?: boolean; label: string }) {
  const known = typeof ok === "boolean";
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs ${
        !known
          ? "border-line text-faint"
          : ok
            ? "border-emerald-500/25 bg-emerald-500/10 text-emerald-300"
            : "border-amber-500/25 bg-amber-500/10 text-amber-200"
      }`}
    >
      <span
        className={`h-1.5 w-1.5 rounded-full ${
          !known ? "bg-faint" : ok ? "bg-emerald-400" : "bg-amber-300"
        }`}
      />
      {label}
      {known ? (ok ? "" : " · no") : ""}
    </span>
  );
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="glass rounded-xl px-4 py-3">
      <p className="text-[10px] uppercase tracking-[0.16em] text-faint">{label}</p>
      <p className="mt-1 text-xl font-medium tabular-nums">{value}</p>
    </div>
  );
}

function SourceCard({ source, index }: { source: Source; index: number }) {
  const web = source.type === "web";
  const name = web
    ? source.title || source.url || "Web source"
    : source.file_name || source.source || "Local document";
  return (
    <div className="glass flex items-start gap-3 rounded-xl p-3">
      <span
        className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ${
          web ? "bg-amber-400/15 text-amber-300" : "bg-accent-soft text-accent"
        }`}
      >
        {web ? <GlobeIcon size={15} /> : <FileIcon size={15} />}
      </span>
      <div className="min-w-0">
        <p className="truncate text-[13px]">
          <span className="mr-1.5 text-faint">{index + 1}.</span>
          {name}
        </p>
        {web && source.url ? (
          <a
            href={source.url}
            target="_blank"
            rel="noopener noreferrer"
            className="block truncate text-[11px] text-faint hover:text-muted"
          >
            {source.url}
          </a>
        ) : (
          <p className="text-[11px] text-faint">Knowledge base</p>
        )}
      </div>
    </div>
  );
}

function DocumentCard({
  doc,
  deleting,
  onDelete,
}: {
  doc: DocumentItem;
  deleting: boolean;
  onDelete: () => void;
}) {
  const poster = POSTERS[hashIndex(doc.document_id, POSTERS.length)];
  const ext = fileExt(doc.file_type || doc.file_name);
  return (
    <div
      className={`group relative aspect-[4/5] overflow-hidden rounded-2xl bg-gradient-to-br ${poster} ring-1 ring-white/5`}
    >
      {/* soft light + texture */}
      <div className="absolute inset-0 bg-[radial-gradient(70%_50%_at_50%_25%,rgba(255,220,180,0.25),transparent_70%)]" />
      <div className="absolute inset-0 bg-gradient-to-t from-black/85 via-black/20 to-transparent" />

      <span className="absolute top-3 left-3 inline-flex items-center gap-1.5 rounded-full bg-black/40 px-2 py-0.5 text-[10px] backdrop-blur">
        <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
        Indexed
      </span>
      <span className="absolute top-3 right-3 rounded-full bg-black/40 px-2 py-0.5 text-[10px] text-white/80 backdrop-blur">
        {ext.replace(".", "")}
      </span>

      <span className="absolute inset-x-0 top-[30%] text-center text-5xl font-semibold tracking-tighter text-white/15 select-none">
        {ext.replace(".", "")}
      </span>

      <div className="absolute inset-x-0 bottom-0 p-3.5">
        <p className="truncate text-sm font-medium" title={doc.file_name}>
          {doc.file_name}
        </p>
        <div className="mt-1 flex items-center justify-between">
          <p className="truncate font-mono text-[10px] text-white/50">
            {doc.document_id.slice(0, 10)}
          </p>
          <button
            type="button"
            onClick={onDelete}
            disabled={deleting}
            aria-label={`Delete ${doc.file_name}`}
            className="rounded-md p-1 text-white/60 opacity-100 transition hover:bg-black/40 hover:text-rose-300 disabled:opacity-50 sm:opacity-0 sm:group-hover:opacity-100 sm:focus:opacity-100"
          >
            {deleting ? <span className="text-[10px]">…</span> : <TrashIcon size={14} />}
          </button>
        </div>
      </div>
    </div>
  );
}

/* ---------------- right-rail widgets (from /observability) ---------------- */

function ActivityCard({ data }: { data: ObservabilityData }) {
  const runs = data.runs.slice(-14);
  const max = Math.max(1, ...runs.map((run) => Number(run.latency_seconds) || 0));
  const last = runs[runs.length - 1];
  return (
    <div className="glass rounded-2xl p-4">
      <div className="flex items-center justify-between">
        <p className="text-sm font-medium">Pipeline activity</p>
        <Link href="/observability" className="text-xs text-muted hover:text-ink">
          Details
        </Link>
      </div>
      <div className="mt-3 flex items-baseline gap-2">
        <p className="text-4xl font-light tabular-nums">{data.summary.total_runs}</p>
        <p className="text-xs text-faint">
          runs · avg {data.summary.average_latency_seconds.toFixed(1)}s
        </p>
      </div>

      {runs.length === 0 ? (
        <p className="mt-6 text-xs text-faint">Ask a question to see latency per run.</p>
      ) : (
        <>
          <div className="mt-5 flex h-20 items-end gap-1.5" aria-label="Latency of recent runs">
            {/* Pad to a fixed 14 slots so a few runs don't render as wide blocks. */}
            {Array.from({ length: 14 - runs.length }, (_, i) => (
              <div key={`pad-${i}`} className="h-[8%] flex-1 rounded-sm bg-white/5" />
            ))}
            {runs.map((run, index) => {
              const latency = Number(run.latency_seconds) || 0;
              const isLast = index === runs.length - 1;
              return (
                <div key={`${run.timestamp}-${index}`} className="relative flex h-full flex-1 items-end">
                  {isLast && (
                    <span className="absolute -top-6 left-1/2 -translate-x-1/2 rounded-md bg-ink px-1.5 py-0.5 text-[10px] whitespace-nowrap text-bg">
                      {latency.toFixed(1)}s
                    </span>
                  )}
                  <div
                    title={`${latency.toFixed(2)}s · ${run.route || "unknown"}`}
                    className={`w-full rounded-sm ${
                      isLast ? "bg-accent" : run.success === false ? "bg-rose-400/40" : "bg-white/12"
                    }`}
                    style={{ height: `${Math.max(8, (latency / max) * 100)}%` }}
                  />
                </div>
              );
            })}
          </div>
          <div className="mt-2 flex justify-between text-[10px] text-faint">
            <span>{timeAgo(runs[0]?.timestamp)} ago</span>
            <span>{last ? "latest" : ""}</span>
          </div>
        </>
      )}
    </div>
  );
}

function Ring({ value, label }: { value: number; label: string }) {
  const pct = Math.max(0, Math.min(1, value || 0));
  const r = 20;
  const c = 2 * Math.PI * r;
  return (
    <div className="flex flex-col items-center gap-1.5">
      <div className="relative h-12 w-12">
        <svg viewBox="0 0 48 48" className="h-12 w-12 -rotate-90">
          <circle cx="24" cy="24" r={r} fill="none" stroke="var(--line-strong)" strokeWidth="4" />
          <circle
            cx="24"
            cy="24"
            r={r}
            fill="none"
            stroke="var(--accent)"
            strokeWidth="4"
            strokeLinecap="round"
            strokeDasharray={`${pct * c} ${c}`}
          />
        </svg>
        <span className="absolute inset-0 flex items-center justify-center text-[11px] tabular-nums">
          {Math.round(pct * 100)}
        </span>
      </div>
      <span className="text-[11px] text-muted">{label}</span>
    </div>
  );
}

function QualityCard({ data }: { data: ObservabilityData }) {
  const { summary } = data;
  return (
    <div className="glass rounded-2xl p-4">
      <div className="flex items-center justify-between">
        <p className="text-sm font-medium">Answer quality</p>
        <Link href="/evaluation" className="text-xs text-muted hover:text-ink">
          View all
        </Link>
      </div>
      <div className="mt-4 grid grid-cols-3">
        <Ring value={summary.success_rate} label="Success" />
        <Ring value={summary.grounded_rate} label="Grounded" />
        <Ring value={summary.answer_quality_rate} label="Useful" />
      </div>
    </div>
  );
}

function RecentRuns({ data }: { data: ObservabilityData }) {
  const runs = data.runs.slice(-4).reverse();
  return (
    <div className="glass rounded-2xl p-4">
      <p className="text-sm font-medium">Recent runs</p>
      {runs.length === 0 ? (
        <p className="mt-3 text-xs text-faint">No runs yet.</p>
      ) : (
        <ul className="mt-3 space-y-2.5">
          {runs.map((run, index) => {
            const route = String(run.route || "").toLowerCase();
            const dot =
              run.success === false
                ? "bg-rose-400"
                : route.includes("web")
                  ? "bg-amber-300"
                  : route.includes("block")
                    ? "bg-rose-400"
                    : "bg-accent";
            return (
              <li key={`${run.timestamp}-${index}`} className="flex items-start gap-2.5">
                <span className={`mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full ${dot}`} />
                <p className="min-w-0 flex-1 truncate text-[13px] text-muted" title={run.question}>
                  {run.question || "Untitled question"}
                </p>
                <span className="shrink-0 text-[11px] text-faint tabular-nums">
                  {timeAgo(run.timestamp)}
                </span>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
