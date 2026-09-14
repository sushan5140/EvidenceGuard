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
