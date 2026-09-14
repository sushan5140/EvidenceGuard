export type ResearchMode =
  | "basic_rag"
  | "hybrid_rag"
  | "conflict_aware"
  | "consensus_rag"
  | "evidenceguard";

export type EvidenceItem = {
  id: string;
  document_id: string;
  document_title: string;
  text: string;
  claim: string;
  retrieval_score: number;
  source_reliability: number;
  agreement_score: number;
  evidence_score: number;
  injected: boolean;
};

export type ConflictEdge = {
  source: string;
  target: string;
  relation: "supports" | "contradicts" | "neutral";
  confidence: number;
};

export type QueryResponse = {
  question: string;
  answer: string;
  confidence: number;
  abstained: boolean;
  conflict_rate: number;
  evidence: EvidenceItem[];
  graph: ConflictEdge[];
  generator: "extractive" | "llm";
  model_status: Record<string, string>;
  mode: ResearchMode;
};

export type DocumentRecord = {
  id: string;
  title: string;
  text: string;
  source_url?: string | null;
  source_reliability: number;
  tags: string[];
  created_at: string;
  injected: boolean;
};

export type Health = {
  status: string;
  documents: number;
  embedding_engine: string;
  nli_engine: string;
  llm_configured: boolean;
};

export type AttackResponse = {
  baseline: QueryResponse;
  attacked: QueryResponse;
  injected_document_ids: string[];
};

export type BenchmarkSummaryRow = {
  mode: ResearchMode;
  conflict_ratio: number;
  samples: number;
  accuracy: number;
  selective_accuracy: number;
  coverage: number;
  abstention_rate: number;
  mean_confidence: number;
  ece: number;
  conflict_precision: number;
  conflict_recall: number;
  conflict_f1: number;
};

export type BenchmarkResponse = {
  benchmark: string;
  cases: number;
  rows: BenchmarkSummaryRow[];
  notes: string[];
};
