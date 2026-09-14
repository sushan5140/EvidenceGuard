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
  const response = await fetch(`${API}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed: ${response.status}`);
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
