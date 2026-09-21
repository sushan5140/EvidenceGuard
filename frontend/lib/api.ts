import type {
  AttackResponse,
  BenchmarkResponse,
  DocumentRecord,
  Health,
  QueryResponse,
  ResearchMode,
} from "./types";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  // A multipart upload needs the browser to supply its boundary.
  if (!(init?.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(`${API}${path}`, { ...init, headers });

  if (!response.ok) {
    const body = await response.text();
    let message = body || `Request failed: ${response.status}`;
    try {
      const parsed = JSON.parse(body);
      if (typeof parsed.detail === "string") message = parsed.detail;
      else if (Array.isArray(parsed.detail)) {
        message = parsed.detail.map((item: { msg?: string }) => item.msg || "Invalid input").join("; ");
      }
    } catch {
      // Keep the original response body if it is not JSON.
    }
    throw new Error(message);
  }

  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<Health>("/api/health"),
  documents: () => request<DocumentRecord[]>("/api/documents"),
  loadDemo: () =>
    request<{ added: number; documents: number }>("/api/demo/load", {
      method: "POST",
    }),
  addDocument: (body: {
    title: string;
    text: string;
    source_reliability: number;
    tags: string[];
  }) =>
    request<DocumentRecord>("/api/documents", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  uploadDocument: (file: File, reliability: number) => {
    const form = new FormData();
    form.append("file", file);
    return request<DocumentRecord>(
      `/api/documents/file?source_reliability=${encodeURIComponent(reliability)}`,
      { method: "POST", body: form },
    );
  },
  deleteDocument: (id: string) =>
    request<void>(`/api/documents/${encodeURIComponent(id)}`, { method: "DELETE" }),
  query: (question: string, useNli = true, mode: ResearchMode = "evidenceguard") =>
    request<QueryResponse>("/api/query", {
      method: "POST",
      body: JSON.stringify({ question, top_k: 8, use_nli: useNli, mode }),
    }),
  attack: (
    question: string,
    injectedClaim: string,
    mode: ResearchMode = "evidenceguard"
  ) =>
    request<AttackResponse>("/api/experiments/attack", {
      method: "POST",
      body: JSON.stringify({
        question,
        injected_claims: [injectedClaim],
        top_k: 8,
        mode,
      }),
    }),
  controlledBenchmark: () =>
    request<BenchmarkResponse>("/api/benchmarks/controlled", {
      method: "POST",
      body: JSON.stringify({
        modes: ["basic_rag", "hybrid_rag", "conflict_aware", "consensus_rag", "evidenceguard"],
        conflict_ratios: [0, 0.1, 0.25, 0.5, 0.75],
        use_nli: false,
        use_local_models: false,
      }),
    }),
};
