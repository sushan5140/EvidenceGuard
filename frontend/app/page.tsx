"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import type {
  AttackResponse,
  BenchmarkResponse,
  ConflictEdge,
  DocumentRecord,
  EvidenceItem,
  Health,
  QueryResponse,
  ResearchMode,
} from "@/lib/types";

function pct(value: number) {
  return `${Math.round(value * 100)}%`;
}

const MODE_LABELS: Record<ResearchMode, string> = {
  basic_rag: "Basic RAG",
  hybrid_rag: "Hybrid RAG",
  conflict_aware: "Conflict-aware (no abstention)",
  evidenceguard: "Full EvidenceGuard",
};

function Metric({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{hint}</small>
    </div>
  );
}

function EvidenceCard({ item, index }: { item: EvidenceItem; index: number }) {
  return (
    <article className="evidence-card">
      <div className="evidence-top">
        <span className="source-index">[{index + 1}]</span>
        <div>
          <h4>{item.document_title}</h4>
          <p className="claim">{item.claim}</p>
        </div>
        <span className={`score ${item.evidence_score >= 0.65 ? "good" : "mid"}`}>
          {pct(item.evidence_score)}
        </span>
      </div>
      <div className="score-row">
        <span>retrieval {pct(item.retrieval_score)}</span>
        <span>reliability {pct(item.source_reliability)}</span>
        <span>agreement {pct(item.agreement_score)}</span>
        {item.injected && <span className="danger-tag">synthetic</span>}
      </div>
    </article>
  );
}

function ConflictGraph({ evidence, edges }: { evidence: EvidenceItem[]; edges: ConflictEdge[] }) {
  const visible = evidence.slice(0, 8);
  const nodes = useMemo(() => {
    const radius = 125;
    return visible.map((item, i) => {
      const angle = (Math.PI * 2 * i) / Math.max(visible.length, 1) - Math.PI / 2;
      return {
        ...item,
        x: 180 + Math.cos(angle) * radius,
        y: 165 + Math.sin(angle) * radius,
      };
    });
  }, [visible]);
  const byId = new Map(nodes.map((node) => [node.id, node]));

  if (nodes.length === 0) {
    return <div className="empty">Run an analysis to build the evidence graph.</div>;
  }

  return (
    <div className="graph-wrap">
      <svg viewBox="0 0 360 330" role="img" aria-label="Evidence conflict graph">
        {edges.map((edge, index) => {
          const a = byId.get(edge.source);
          const b = byId.get(edge.target);
          if (!a || !b || edge.relation === "neutral") return null;
          return (
            <line
              key={`${edge.source}-${edge.target}-${index}`}
              x1={a.x}
              y1={a.y}
              x2={b.x}
              y2={b.y}
              className={edge.relation === "contradicts" ? "edge conflict" : "edge support"}
              strokeWidth={1 + edge.confidence * 2}
            />
          );
        })}
        {nodes.map((node, i) => (
          <g key={node.id}>
            <circle
              cx={node.x}
              cy={node.y}
              r="24"
              className={node.injected ? "node injected" : "node"}
            />
            <text x={node.x} y={node.y + 4} textAnchor="middle" className="node-label">
              {i + 1}
            </text>
          </g>
        ))}
      </svg>
      <div className="legend">
        <span><i className="support-dot" /> supports</span>
        <span><i className="conflict-dot" /> contradicts</span>
      </div>
    </div>
  );
}

function BenchmarkMatrix({ data }: { data: BenchmarkResponse }) {
  return (
    <div className="benchmark-wrap">
      <div className="benchmark-meta">
        <span>{data.benchmark}</span>
        <span>{data.cases} cases · {data.rows.length} aggregate cells</span>
      </div>
      <div className="table-scroll">
        <table className="benchmark-table">
          <thead>
            <tr>
              <th>Mode</th>
              <th>Conflict</th>
              <th>Accuracy</th>
              <th>Selective</th>
              <th>Coverage</th>
              <th>ECE ↓</th>
              <th>Conflict F1</th>
            </tr>
          </thead>
          <tbody>
            {data.rows.map((row) => (
              <tr key={`${row.mode}-${row.conflict_ratio}`}>
                <td><b>{MODE_LABELS[row.mode]}</b></td>
                <td>{pct(row.conflict_ratio)}</td>
                <td>
                  <div className="bar-cell">
                    <span className="bar-fill" style={{ width: pct(row.accuracy) }} />
                    <b>{pct(row.accuracy)}</b>
                  </div>
                </td>
                <td>{pct(row.selective_accuracy)}</td>
                <td>{pct(row.coverage)}</td>
                <td>{row.ece.toFixed(3)}</td>
                <td>{row.conflict_f1.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="benchmark-notes">
        {data.notes.map((note) => <span key={note}>{note}</span>)}
      </div>
    </div>
  );
}

export default function Home() {
  const [health, setHealth] = useState<Health | null>(null);
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [question, setQuestion] = useState("When did the Eiffel Tower open to the public?");
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [mode, setMode] = useState<ResearchMode>("evidenceguard");
  const [useNli, setUseNli] = useState(true);
  const [busy, setBusy] = useState(false);
  const [benchmarkBusy, setBenchmarkBusy] = useState(false);
  const [error, setError] = useState("");
  const [attackClaim, setAttackClaim] = useState(
    "The Eiffel Tower opened to the public in 1905, not 1889."
  );
  const [attackResult, setAttackResult] = useState<AttackResponse | null>(null);
  const [benchmark, setBenchmark] = useState<BenchmarkResponse | null>(null);
  const [newSource, setNewSource] = useState({
    title: "",
    text: "",
    reliability: 0.7,
  });

  const conflictMode = mode === "conflict_aware" || mode === "evidenceguard";

  const refresh = async () => {
    const [h, docs] = await Promise.all([api.health(), api.documents()]);
    setHealth(h);
    setDocuments(docs);
  };

  useEffect(() => {
    refresh().catch((err) => setError(err.message));
  }, []);

  const loadDemo = async () => {
    setBusy(true);
    setError("");
    try {
      await api.loadDemo();
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load demo");
    } finally {
      setBusy(false);
    }
  };

  const analyze = async (event?: FormEvent) => {
    event?.preventDefault();
    setBusy(true);
    setError("");
    try {
      const data = await api.query(question, useNli, mode);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed");
    } finally {
      setBusy(false);
    }
  };

  const runAttack = async () => {
    setBusy(true);
    setError("");
    try {
      setAttackResult(await api.attack(question, attackClaim, mode));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Experiment failed");
    } finally {
      setBusy(false);
    }
  };

  const runBenchmark = async () => {
    setBenchmarkBusy(true);
    setError("");
    try {
      setBenchmark(await api.controlledBenchmark());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Benchmark failed");
    } finally {
      setBenchmarkBusy(false);
    }
  };

  const addSource = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      await api.addDocument({
        title: newSource.title,
        text: newSource.text,
        source_reliability: newSource.reliability,
        tags: ["manual"],
      });
      setNewSource({ title: "", text: "", reliability: 0.7 });
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not add source");
    } finally {
      setBusy(false);
    }
  };

  return (
    <main>
      <header className="topbar">
        <a className="brand" href="#">
          <span className="brand-mark">EG</span>
          <span><b>EvidenceGuard</b><small>conflict-aware RAG</small></span>
        </a>
        <div className="status">
          <span className={health?.status === "ok" ? "live-dot" : "offline-dot"} />
          API {health?.status || "connecting"}
        </div>
      </header>

      <section className="hero">
        <div>
          <p className="eyebrow">AI / COMPUTER SCIENCE · FINAL-YEAR RESEARCH PROTOTYPE</p>
          <h1>Answers are easy.<br /><em>Knowing when not to answer</em> is harder.</h1>
          <p className="lede">
            EvidenceGuard retrieves multiple sources, finds claims that agree or conflict,
            scores their reliability, and abstains when the evidence cannot support a safe answer.
          </p>
          <div className="hero-actions">
            <button className="primary" onClick={loadDemo} disabled={busy}>Load demo evidence</button>
            <span>{documents.length} sources indexed</span>
          </div>
        </div>
        <aside className="architecture">
          <span>QUESTION</span><b>↓</b>
          <div>Hybrid retrieval <small>BM25 + semantic</small></div><b>↓</b>
          <div>Claim extraction <small>atomic evidence</small></div><b>↓</b>
          <div className="accent-box">Conflict graph <small>NLI comparison</small></div><b>↓</b>
          <div>Score + abstain <small>uncertainty aware</small></div>
        </aside>
      </section>

      {error && <div className="error">{error}</div>}

      <section className="workspace">
        <div className="main-column">
          <form className="ask-card" onSubmit={analyze}>
            <div className="section-title">
              <div><span>01</span><h2>Ask / compare systems</h2></div>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={useNli}
                  disabled={!conflictMode}
                  onChange={(e) => setUseNli(e.target.checked)}
                />
                <span /> NLI model
              </label>
            </div>
            <div className="mode-row">
              <label>
                Research mode
                <select value={mode} onChange={(e) => setMode(e.target.value as ResearchMode)}>
                  {Object.entries(MODE_LABELS).map(([value, label]) => (
                    <option value={value} key={value}>{label}</option>
                  ))}
                </select>
              </label>
              <p>
                {mode === "basic_rag" && "BM25 retrieval; no explicit conflict reasoning or abstention."}
                {mode === "hybrid_rag" && "Hybrid retrieval; no explicit conflict reasoning or abstention."}
                {mode === "conflict_aware" && "Hybrid retrieval + conflict scoring; forced to answer."}
                {mode === "evidenceguard" && "Full system with conflict scoring and uncertainty-aware abstention."}
              </p>
            </div>
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask a question grounded in your evidence…"
              rows={3}
            />
            <div className="ask-footer">
              <p>Same corpus, same question, controlled system ablations.</p>
              <button className="primary" disabled={busy || question.trim().length < 3}>
                {busy ? "Analyzing…" : "Analyze evidence →"}
              </button>
            </div>
          </form>

          {result && (
            <>
              <section className="answer-card">
                <div className="answer-head">
                  <div>
                    <p className="eyebrow">{MODE_LABELS[result.mode].toUpperCase()}</p>
                    <h2>{result.abstained ? "Abstained" : "Answer"}</h2>
                  </div>
                  <span className={result.abstained ? "decision abstain" : "decision answer"}>
                    {result.abstained ? "INSUFFICIENT" : "SUPPORTED"}
                  </span>
                </div>
                <p className="answer-text">{result.answer}</p>
                <div className="metrics">
                  <Metric label="confidence" value={pct(result.confidence)} hint="system confidence" />
                  <Metric label="conflict rate" value={pct(result.conflict_rate)} hint="meaningful graph edges" />
                  <Metric label="evidence" value={String(result.evidence.length)} hint="atomic claims ranked" />
                  <Metric label="mode" value={result.mode.replaceAll("_", " ")} hint="ablation configuration" />
                </div>
              </section>

              <section className="panel">
                <div className="section-title">
                  <div><span>02</span><h2>Evidence inspector</h2></div>
                  <small>ranked by active scoring mode</small>
                </div>
                <div className="evidence-list">
                  {result.evidence.map((item, index) => (
                    <EvidenceCard item={item} index={index} key={item.id} />
                  ))}
                </div>
              </section>
            </>
          )}
        </div>

        <aside className="side-column">
          <section className="panel sticky">
            <div className="section-title">
              <div><span>03</span><h2>Conflict graph</h2></div>
            </div>
            <ConflictGraph evidence={result?.evidence || []} edges={result?.graph || []} />
            {result && (
              <div className="model-status">
                {Object.entries(result.model_status).map(([name, value]) => (
                  <div key={name}><span>{name}</span><b>{value}</b></div>
                ))}
              </div>
            )}
          </section>
        </aside>
      </section>

      <section className="lab-grid">
        <section className="panel lab">
          <div className="section-title">
            <div><span>04</span><h2>Adversarial research lab</h2></div>
            <small>controlled misinformation injection</small>
          </div>
          <p className="muted">
            Inject a conflicting claim temporarily and measure the same selected research mode
            before and after the attack.
          </p>
          <textarea value={attackClaim} onChange={(e) => setAttackClaim(e.target.value)} rows={3} />
          <button className="secondary" onClick={runAttack} disabled={busy || !attackClaim.trim()}>
            Run conflict attack
          </button>
          {attackResult && (
            <div className="comparison">
              <div><span>Clean</span><strong>{pct(attackResult.baseline.confidence)}</strong>
                <small>{attackResult.baseline.abstained ? "abstained" : "answered"}</small></div>
              <div className="arrow">→</div>
              <div><span>After attack</span><strong>{pct(attackResult.attacked.confidence)}</strong>
                <small>{attackResult.attacked.abstained ? "abstained" : "answered"}</small></div>
            </div>
          )}
        </section>

        <section className="panel lab">
          <div className="section-title">
            <div><span>05</span><h2>Add evidence source</h2></div>
            <small>build your own corpus</small>
          </div>
          <form className="source-form" onSubmit={addSource}>
            <input
              placeholder="Source title"
              value={newSource.title}
              onChange={(e) => setNewSource({ ...newSource, title: e.target.value })}
              required
            />
            <textarea
              placeholder="Paste source text…"
              rows={4}
              value={newSource.text}
              onChange={(e) => setNewSource({ ...newSource, text: e.target.value })}
              required
            />
            <label>
              Source reliability <b>{pct(newSource.reliability)}</b>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={newSource.reliability}
                onChange={(e) => setNewSource({ ...newSource, reliability: Number(e.target.value) })}
              />
            </label>
            <button className="secondary" disabled={busy}>Index source</button>
          </form>
        </section>
      </section>

      <section className="sources-panel benchmark-panel">
        <div className="section-title">
          <div><span>06</span><h2>Controlled benchmark matrix</h2></div>
          <button className="secondary" onClick={runBenchmark} disabled={benchmarkBusy}>
            {benchmarkBusy ? "Running benchmark…" : "Run 4 × 5 benchmark"}
          </button>
        </div>
        <p className="benchmark-intro">
          Six controlled QA cases × four system modes × five conflict levels. The quick dashboard
          run uses deterministic fallbacks; the CLI can enable local embedding/NLI models for the
          final experiment.
        </p>
        {benchmark ? (
          <BenchmarkMatrix data={benchmark} />
        ) : (
          <div className="empty">Run the benchmark to generate the comparison matrix.</div>
        )}
      </section>

      <section className="sources-panel">
        <div className="section-title">
          <div><span>07</span><h2>Indexed corpus</h2></div>
          <small>{documents.length} documents</small>
        </div>
        <div className="source-grid">
          {documents.length === 0 && <div className="empty">Load the demo or add a source to begin.</div>}
          {documents.map((doc) => (
            <article key={doc.id}>
              <div><h4>{doc.title}</h4><span>{pct(doc.source_reliability)}</span></div>
              <p>{doc.text.slice(0, 155)}{doc.text.length > 155 ? "…" : ""}</p>
              <small>{doc.tags.join(" · ") || "untagged"}</small>
            </article>
          ))}
        </div>
      </section>

      <footer>
        <b>EvidenceGuard v2</b>
        <span>Baselines · conflict sweeps · calibration · selective answering</span>
      </footer>
    </main>
  );
}
