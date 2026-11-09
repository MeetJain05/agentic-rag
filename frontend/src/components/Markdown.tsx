"use client";

import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

// Styled renderers for model answers. Raw HTML in the answer is not
// rendered (react-markdown escapes it by default), which keeps
// untrusted retrieved content from injecting markup.
const components: Components = {
  p: ({ children }) => <p className="my-3 first:mt-0 last:mb-0">{children}</p>,
  h1: ({ children }) => (
    <h3 className="mb-2 mt-6 text-lg font-semibold text-ink first:mt-0">{children}</h3>
  ),
  h2: ({ children }) => (
    <h3 className="mb-2 mt-6 text-base font-semibold text-ink first:mt-0">{children}</h3>
  ),
  h3: ({ children }) => (
    <h4 className="mb-2 mt-5 text-[15px] font-semibold text-ink first:mt-0">{children}</h4>
  ),
  h4: ({ children }) => (
    <h5 className="mb-1 mt-4 text-sm font-semibold text-ink first:mt-0">{children}</h5>
  ),
  ul: ({ children }) => (
    <ul className="my-3 list-disc space-y-1.5 pl-5 marker:text-accent">{children}</ul>
  ),
  ol: ({ children }) => (
    <ol className="my-3 list-decimal space-y-1.5 pl-5 marker:text-muted">{children}</ol>
  ),
  li: ({ children }) => <li className="pl-1">{children}</li>,
  strong: ({ children }) => <strong className="font-semibold text-ink">{children}</strong>,
  em: ({ children }) => <em className="italic">{children}</em>,
  a: ({ href, children }) => (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent"
    >
      {children}
    </a>
  ),
  blockquote: ({ children }) => (
    <blockquote className="my-3 border-l-2 border-accent/60 pl-4 text-muted">
      {children}
    </blockquote>
  ),
  hr: () => <hr className="my-5 border-line" />,
  code: ({ className, children }) =>
    className ? (
      <code className={`${className} font-mono text-[13px]`}>{children}</code>
    ) : (
      <code className="rounded bg-surface-raised px-1.5 py-0.5 font-mono text-[13px] text-ink">
        {children}
      </code>
    ),
  pre: ({ children }) => (
    <pre className="scroll-thin my-3 overflow-x-auto rounded-xl border border-line bg-bg/60 p-4 text-[13px] leading-6">
      {children}
    </pre>
  ),
  table: ({ children }) => (
    <div className="scroll-thin my-4 overflow-x-auto rounded-xl border border-line">
      <table className="w-full border-collapse text-left text-sm">{children}</table>
    </div>
  ),
  thead: ({ children }) => <thead className="bg-surface-raised text-ink">{children}</thead>,
  th: ({ children }) => (
    <th className="border-b border-line px-3 py-2 font-semibold">{children}</th>
  ),
  td: ({ children }) => (
    <td className="border-b border-line/60 px-3 py-2 align-top">{children}</td>
  ),
};

export default function Markdown({ children }: { children: string }) {
  return (
    <div className="break-words">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {children}
      </ReactMarkdown>
    </div>
  );
}
