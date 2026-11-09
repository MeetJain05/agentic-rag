export const API_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ||
  "http://localhost:8000";

export type ObservabilityRun = {
  timestamp?: string;
  question?: string;
  route?: string;
  latency_seconds?: number;
  success?: boolean;
  grounded?: boolean;
  answers_question?: boolean;
  retry_count?: number;
  hitl_status?: string;
};

export type ObservabilitySummary = {
  total_runs: number;
  successful_runs: number;
  success_rate: number;
  average_latency_seconds: number;
  grounded_rate: number;
  answer_quality_rate: number;
  local_route_count: number;
  web_route_count: number;
  hitl_approved_count: number;
  hitl_rejected_count: number;
};

export type ObservabilityData = {
  runs: ObservabilityRun[];
  summary: ObservabilitySummary;
};

export const EMPTY_SUMMARY: ObservabilitySummary = {
  total_runs: 0,
  successful_runs: 0,
  success_rate: 0,
  average_latency_seconds: 0,
  grounded_rate: 0,
  answer_quality_rate: 0,
  local_route_count: 0,
  web_route_count: 0,
  hitl_approved_count: 0,
  hitl_rejected_count: 0,
};

export async function fetchObservability(): Promise<ObservabilityData> {
  const res = await fetch(`${API_URL}/observability`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Observability request failed (${res.status})`);
  const data = await res.json();
  return {
    runs: Array.isArray(data?.runs) ? data.runs : [],
    summary: { ...EMPTY_SUMMARY, ...(data?.summary || {}) },
  };
}

/** Fired after a question/upload so sidebar widgets can refresh. */
export const RAG_ACTIVITY_EVENT = "rag:activity";
/** Fired by the sidebar's "New question" button. */
export const RAG_NEW_QUESTION_EVENT = "rag:new-question";

export function emit(name: string) {
  if (typeof window !== "undefined") window.dispatchEvent(new Event(name));
}

export function timeAgo(value?: string): string {
  if (!value) return "";
  const then = new Date(value).getTime();
  if (Number.isNaN(then)) return "";
  const seconds = Math.max(0, Math.round((Date.now() - then) / 1000));
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h`;
  return `${Math.round(hours / 24)}d`;
}
