import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "EvidenceGuard",
  description: "Conflict-aware RAG research prototype",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
