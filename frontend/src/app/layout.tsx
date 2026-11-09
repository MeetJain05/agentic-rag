import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";

import Sidebar from "@/components/Sidebar";

import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Agentic Adaptive RAG",
  description:
    "Ask questions over your documents with adaptive routing, grounding checks and human-approved web search.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body suppressHydrationWarning className="min-h-full bg-bg text-ink">
        <div className="ambient" aria-hidden />
        <div className="relative flex min-h-screen flex-col gap-3 p-3 lg:h-screen lg:flex-row">
          <Sidebar />
          <div className="scroll-thin relative z-10 min-w-0 flex-1 lg:overflow-y-auto">
            {children}
          </div>
        </div>
      </body>
    </html>
  );
}
