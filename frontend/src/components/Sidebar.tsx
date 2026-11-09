"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import {
  API_URL,
  EMPTY_SUMMARY,
  ObservabilitySummary,
  RAG_ACTIVITY_EVENT,
  RAG_NEW_QUESTION_EVENT,
  emit,
  fetchObservability,
} from "@/lib/api";
import {
  ChartIcon,
  DatabaseIcon,
  HomeIcon,
  PlusIcon,
  PulseIcon,
} from "./icons";

const NAV = [
  { href: "/", label: "Ask", icon: HomeIcon },
  { href: "/evaluation", label: "Evaluation", icon: ChartIcon },
  { href: "/observability", label: "Observability", icon: PulseIcon },
];

type Health = {
  status: "healthy" | "degraded" | "offline" | "loading";
  documents: number;
};

export default function Sidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const [summary, setSummary] = useState<ObservabilitySummary>(EMPTY_SUMMARY);
  const [health, setHealth] = useState<Health>({ status: "loading", documents: 0 });

  const refresh = useCallback(async () => {
    try {
      const data = await fetchObservability();
      setSummary(data.summary);
    } catch {
      /* sidebar stats are best-effort */
    }
    try {
      const res = await fetch(`${API_URL}/health`, { cache: "no-store" });
      const data = await res.json();
      setHealth({
        status: data?.status === "healthy" ? "healthy" : "degraded",
        documents: Number(data?.documents) || 0,
      });
    } catch {
      setHealth({ status: "offline", documents: 0 });
    }
  }, []);

  useEffect(() => {
    const timeoutId = window.setTimeout(refresh, 0);
    window.addEventListener(RAG_ACTIVITY_EVENT, refresh);
    return () => {
      window.clearTimeout(timeoutId);
      window.removeEventListener(RAG_ACTIVITY_EVENT, refresh);
    };
  }, [refresh]);

  function newQuestion() {
    if (pathname === "/") emit(RAG_NEW_QUESTION_EVENT);
    else router.push("/");
  }

  const routes = [
    { label: "Local RAG", color: "bg-accent", count: summary.local_route_count },
    { label: "Web search", color: "bg-amber-300", count: summary.web_route_count },
    { label: "HITL approved", color: "bg-emerald-400", count: summary.hitl_approved_count },
    { label: "HITL rejected", color: "bg-rose-400", count: summary.hitl_rejected_count },
  ];

  const statusColor =
    health.status === "healthy"
      ? "bg-emerald-400"
      : health.status === "loading"
        ? "bg-faint"
        : "bg-rose-400";

  return (
    <>
      {/* Desktop sidebar */}
      <aside className="glass relative z-10 hidden w-[248px] shrink-0 flex-col rounded-2xl lg:flex">
        <div className="flex items-center gap-3 px-4 pt-4 pb-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-ink text-bg">
            <span className="h-3.5 w-3.5 rounded-full border-[3px] border-bg" />
          </div>
          <div className="min-w-0">
            <p className="truncate text-sm font-semibold">Agentic RAG</p>
            <p className="truncate text-[11px] text-faint">Adaptive workspace</p>
          </div>
        </div>

        <div className="px-3">
          <button
            type="button"
            onClick={newQuestion}
            className="flex w-full items-center justify-between rounded-xl bg-ink px-3.5 py-2.5 text-sm font-medium text-bg transition hover:bg-white"
          >
            New question
            <PlusIcon size={15} />
          </button>
        </div>

        <nav className="scroll-thin mt-5 flex-1 overflow-y-auto px-3">
          <SectionLabel>Workspace</SectionLabel>
          <ul className="space-y-0.5">
            {NAV.map(({ href, label, icon: Icon }) => {
              const active = pathname === href;
              return (
                <li key={href}>
                  <Link
                    href={href}
                    aria-current={active ? "page" : undefined}
                    className={`flex items-center gap-2.5 rounded-lg px-3 py-2 text-[13px] transition ${
                      active
                        ? "bg-surface-raised text-ink ring-1 ring-line-strong"
                        : "text-muted hover:bg-surface-raised hover:text-ink"
                    }`}
                  >
                    <Icon size={15} />
                    {label}
                  </Link>
                </li>
              );
            })}
          </ul>

          <SectionLabel className="mt-6">Routing</SectionLabel>
          <ul className="space-y-0.5">
            {routes.map((route) => (
              <li
                key={route.label}
                className="flex items-center gap-2.5 px-3 py-1.5 text-[13px] text-muted"
              >
                <span className={`h-2 w-2 rounded-full ${route.color}`} />
                <span className="flex-1">{route.label}</span>
                <span className="text-[11px] tabular-nums text-faint">
                  {route.count}
                </span>
              </li>
            ))}
          </ul>
        </nav>

        {/* Index status card (reference: plan card) */}
        <div className="m-3 rounded-xl border border-accent/25 bg-gradient-to-br from-accent/15 via-accent/5 to-transparent p-3.5">
          <div className="flex items-center justify-between">
            <p className="flex items-center gap-2 text-[13px] font-medium">
              <DatabaseIcon size={14} className="text-accent" />
              Pinecone index
            </p>
            <span className={`h-2 w-2 rounded-full ${statusColor}`} />
          </div>
          <p className="mt-1.5 text-[11px] leading-4 text-muted">
            {health.status === "offline"
              ? "Backend unreachable"
              : health.status === "loading"
                ? "Checking connection…"
                : `${health.documents} document${health.documents === 1 ? "" : "s"} indexed · ${health.status}`}
          </p>
        </div>
      </aside>

      {/* Mobile top bar */}
      <header className="glass relative z-10 flex items-center justify-between rounded-2xl px-4 py-3 lg:hidden">
        <p className="text-sm font-semibold">Agentic RAG</p>
        <nav className="flex gap-1">
          {NAV.map(({ href, label }) => (
            <Link
              key={href}
              href={href}
              className={`rounded-lg px-2.5 py-1.5 text-xs ${
                pathname === href ? "bg-surface-raised text-ink" : "text-muted"
              }`}
            >
              {label}
            </Link>
          ))}
        </nav>
      </header>
    </>
  );
}

function SectionLabel({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <p
      className={`mb-1.5 px-3 text-[10px] font-semibold uppercase tracking-[0.18em] text-faint ${className}`}
    >
      {children}
    </p>
  );
}
