import { useEffect, useRef, useState } from "react";

const API_URL = "http://localhost:8000";

const FAMILIES = {
  original: { label: "Original", color: "#1a1a1a" },
  rule_based: { label: "Rules", color: "#6a7179" },
  lm: { label: "LM pruning", color: "#c9622a" },
  lm_safe: { label: "LM pruning, risky words kept", color: "#2f9e44" },
  query_sentence: { label: "Query-aware sentences", color: "#1c7ed6" },
  random_drop: { label: "Random drop (control)", color: "#adb5bd" },
};

const MODELS = {
  squad: ["distilbert-base-cased-distilled-squad", "deepset/roberta-base-squad2"],
  gsm8k: ["Qwen/Qwen2.5-0.5B-Instruct"],
};

const pct = (v) => (v === null || v === undefined ? "-" : `${v}%`);

function Chart({ methods, metric }) {
  const W = 720;
  const H = 340;
  const m = { l: 56, r: 20, t: 16, b: 44 };
  const entries = Object.entries(methods);
  const xMax = Math.max(10, Math.ceil(Math.max(...entries.map(([, r]) => r.tokens_saved_pct)) / 10) * 10);
  const yMax = 1.1;
  const x = (v) => m.l + (v / xMax) * (W - m.l - m.r);
  const y = (v) => H - m.b - (Math.min(v, yMax) / yMax) * (H - m.t - m.b);

  const byFamily = {};
  entries.forEach(([name, r]) => {
    (byFamily[r.family] = byFamily[r.family] || []).push({ name, ...r });
  });

  return (
    <div>
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Quality retained versus tokens saved">
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line x1={m.l} x2={W - m.r} y1={y(t)} y2={y(t)} stroke="#e3e6e4" />
            <text x={m.l - 8} y={y(t) + 4} textAnchor="end" fontSize="12" fill="#6a7179">
              {Math.round(t * 100)}%
            </text>
          </g>
        ))}
        {Array.from({ length: xMax / 10 + 1 }, (_, i) => i * 10).map((t) => (
          <text key={t} x={x(t)} y={H - m.b + 18} textAnchor="middle" fontSize="12" fill="#6a7179">
            {t}%
          </text>
        ))}
        <text x={(m.l + W - m.r) / 2} y={H - 6} textAnchor="middle" fontSize="13" fill="#1a1a1a">
          Tokens saved
        </text>
        <text transform={`translate(14 ${H / 2}) rotate(-90)`} textAnchor="middle" fontSize="13" fill="#1a1a1a">
          {metric} retained
        </text>
        {Object.entries(byFamily).map(([fam, pts]) => {
          const sorted = [...pts].sort((a, b) => a.tokens_saved_pct - b.tokens_saved_pct);
          const color = (FAMILIES[fam] || {}).color || "#868e96";
          return (
            <g key={fam}>
              {sorted.length > 1 && (
                <polyline
                  fill="none"
                  stroke={color}
                  strokeWidth="1.5"
                  points={sorted.map((p) => `${x(p.tokens_saved_pct)},${y(p.quality_retained)}`).join(" ")}
                />
              )}
              {sorted.map((p) => (
                <g key={p.name}>
                  {p.quality_retained_ci95 && (
                    <line
                      x1={x(p.tokens_saved_pct)}
                      x2={x(p.tokens_saved_pct)}
                      y1={y(p.quality_retained_ci95[0])}
                      y2={y(p.quality_retained_ci95[1])}
                      stroke={color}
                      strokeWidth="1.5"
                      opacity="0.5"
                    />
                  )}
                  <circle cx={x(p.tokens_saved_pct)} cy={y(p.quality_retained)} r="5" fill={color}>
                    <title>
                      {p.name}: {p.tokens_saved_pct}% saved, {Math.round(p.quality_retained * 100)}% retained
                    </title>
                  </circle>
                </g>
              ))}
            </g>
          );
        })}
      </svg>
      <div className="legend">
        {Object.entries(byFamily).map(([fam]) => (
          <span key={fam} style={{ "--c": (FAMILIES[fam] || {}).color || "#868e96" }}>
            {(FAMILIES[fam] || {}).label || fam}
          </span>
        ))}
      </div>
    </div>
  );
}

function Security({ data }) {
  const rows = Object.entries(data.methods);
  return (
    <div className="result-card">
      <h2>
        Security: guardrails and injected text
        <span className="count">
          {data.meta.n} synthetic prompts, seed {data.meta.seed}
        </span>
      </h2>
      <p className="hint">
        Each prompt has {data.meta.rules_per_prompt} safety rules plus SQuAD reference text. "Rules kept" is the share
        of rules whose key terms all survive. "Injection intact" is the share of prompts where the injected sentence survives word for word, and
        "injection working" where its canary word and at least 75% of its words survive in order (a heuristic,
        text survival only; no model is asked to obey it).
      </p>
      <table className="data">
        <thead>
          <tr>
            <th>Method</th>
            <th>Tokens saved</th>
            <th>Rules kept</th>
            <th>Rules kept, locked</th>
            <th>Tokens saved, locked</th>
            <th>Injection intact</th>
            <th>Injection working</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([name, r]) => (
            <tr key={name}>
              <td>{name}</td>
              <td className="num">{pct(r.guardrails.unprotected.tokens_saved_pct)}</td>
              <td className="num">{pct(r.guardrails.unprotected.rules_retained_pct)}</td>
              <td className="num">{pct(r.guardrails.protected.rules_retained_pct)}</td>
              <td className="num">{pct(r.guardrails.protected.tokens_saved_pct)}</td>
              <td className="num">{pct(r.injection.verbatim_survives_pct)}</td>
              <td className="num">{pct(r.injection.functional_survives_pct)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="hint">{data.meta.notes}</p>
    </div>
  );
}

export default function Benchmark() {
  const [list, setList] = useState([]);
  const [selected, setSelected] = useState("");
  const [data, setData] = useState(null);
  const [securityName, setSecurityName] = useState("");
  const [security, setSecurity] = useState(null);
  const [example, setExample] = useState("");
  const [error, setError] = useState(null);
  const [task, setTask] = useState("squad");
  const [n, setN] = useState(50);
  const [model, setModel] = useState(MODELS.squad[0]);
  const [job, setJob] = useState(null);
  const timer = useRef(null);

  const load = async (pick) => {
    try {
      const res = await fetch(`${API_URL}/benchmark/results`);
      const body = await res.json();
      setList(body.results);
      const bench = body.results.filter((r) => r.kind === "benchmark");
      const sec = body.results.find((r) => r.kind === "security");
      setSelected((cur) => pick || cur || (bench[0] ? bench[0].name : ""));
      setSecurityName(sec ? sec.name : "");
    } catch {
      setError("Could not reach the API at " + API_URL);
    }
  };

  useEffect(() => {
    load();
    return () => clearInterval(timer.current);
  }, []);

  useEffect(() => {
    if (!selected) return;
    fetch(`${API_URL}/benchmark/results/${selected}`)
      .then((r) => r.json())
      .then((d) => {
        setData(d);
        setExample("");
      });
  }, [selected]);

  useEffect(() => {
    if (!securityName) return setSecurity(null);
    fetch(`${API_URL}/benchmark/results/${securityName}`)
      .then((r) => r.json())
      .then(setSecurity);
  }, [securityName]);

  const run = async () => {
    setError(null);
    const res = await fetch(`${API_URL}/benchmark`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task, n: Number(n), seed: 0, model: task === "security" ? null : model }),
    });
    const body = await res.json();
    if (!res.ok) return setError(body.detail || "Could not start the run");
    setJob({ status: "running", stage: "starting", done: 0, total: Number(n) });
    clearInterval(timer.current);
    timer.current = setInterval(async () => {
      const r = await fetch(`${API_URL}/benchmark/runs/${body.id}`);
      const j = await r.json();
      setJob(j);
      if (j.status !== "running") {
        clearInterval(timer.current);
        if (j.status === "done") load(task === "security" ? undefined : j.result_name);
        else setError(j.error);
      }
    }, 1500);
  };

  const benchmarks = list.filter((r) => r.kind === "benchmark");
  const label = (r) =>
    `${r.meta.task} · ${r.meta.answer_model} · n=${r.meta.n} · ${r.meta.date}`;
  const names = data ? Object.keys(data.methods).filter((k) => k !== "original") : [];

  return (
    <div>
      <div className="result-card">
        <h2>Run a benchmark</h2>
        <div className="bench-controls">
          <label>
            Task
            <select
              value={task}
              onChange={(e) => {
                setTask(e.target.value);
                if (MODELS[e.target.value]) setModel(MODELS[e.target.value][0]);
              }}
            >
              <option value="squad">SQuAD (question answering)</option>
              <option value="gsm8k">GSM8K (math reasoning)</option>
              <option value="security">Security (guardrails and injection)</option>
            </select>
          </label>
          {task !== "security" && (
            <label>
              Answer model
              <select value={model} onChange={(e) => setModel(e.target.value)}>
                {MODELS[task].map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </label>
          )}
          <label>
            Examples
            <input type="number" min="5" max="500" value={n} onChange={(e) => setN(e.target.value)} />
          </label>
          <button onClick={run} disabled={job && job.status === "running"}>
            {job && job.status === "running" ? "Running..." : "Run"}
          </button>
        </div>
        {job && job.status === "running" && (
          <div>
            <div className="progress">
              <div style={{ width: `${job.total ? (100 * job.done) / job.total : 0}%` }} />
            </div>
            <span className="hint">
              {job.stage}: {job.done} / {job.total}
            </span>
          </div>
        )}
        <p className="hint">Seed 0. Larger runs take minutes; the first run of a model downloads it.</p>
        {error && <p className="error">{error}</p>}
      </div>

      {benchmarks.length === 0 && !error && <p className="hint">No results yet. Run a benchmark above.</p>}

      {benchmarks.length > 0 && (
        <div className="bench-controls" style={{ marginTop: 24 }}>
          <label>
            Result
            <select value={selected} onChange={(e) => setSelected(e.target.value)}>
              {benchmarks.map((r) => (
                <option key={r.name} value={r.name}>
                  {label(r)}
                </option>
              ))}
            </select>
          </label>
        </div>
      )}

      {data && (
        <div className="results">
          <div className="result-card">
            <h2>
              Savings versus quality
              <span className="count">
                {data.meta.metric}, n={data.meta.n}, seed {data.meta.seed}, 95% bootstrap interval
              </span>
            </h2>
            <Chart methods={data.methods} metric={data.meta.metric} />
            <p className="hint">{data.meta.notes}</p>
          </div>

          <div className="result-card">
            <h2>Results</h2>
            <table className="data">
              <thead>
                <tr>
                  <th>Method</th>
                  <th>Tokens saved</th>
                  <th>{data.meta.metric}</th>
                  <th>Quality retained (95% interval)</th>
                  <th>Examples hurt</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(data.methods).map(([name, r]) => (
                  <tr key={name}>
                    <td>{name}</td>
                    <td className="num">{r.tokens_saved_pct}%</td>
                    <td className="num">{r.score}</td>
                    <td className="num">
                      {Math.round(r.quality_retained * 100)}%
                      {r.quality_retained_ci95 &&
                        ` (${Math.round(r.quality_retained_ci95[0] * 100)}-${Math.round(r.quality_retained_ci95[1] * 100)}%)`}
                    </td>
                    <td className="num">{r.examples_hurt}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="hint">
              Answer model: {data.meta.answer_model}. LM scorer: {data.meta.scorer_lm}. Tokens: {data.meta.token_counter}.
            </p>
          </div>

          <div className="result-card">
            <h2>Failure examples</h2>
            <div className="bench-controls">
              <label>
                Method
                <select value={example} onChange={(e) => setExample(e.target.value)}>
                  <option value="">Choose a method</option>
                  {names.map((nm) => (
                    <option key={nm} value={nm}>
                      {nm} ({data.methods[nm].examples_hurt} hurt)
                    </option>
                  ))}
                </select>
              </label>
            </div>
            {example &&
              (data.methods[example].worst_examples.length === 0 ? (
                <p className="hint">No example got worse.</p>
              ) : (
                data.methods[example].worst_examples.map((ex, i) => (
                  <div key={i} className="example">
                    <div>
                      <strong>{ex.question}</strong>
                    </div>
                    <div>
                      Gold: <code>{ex.gold}</code>
                    </div>
                    <div>
                      Original answer: <code>{ex.original_answer}</code>
                    </div>
                    <div>
                      After compression: <code>{ex.method_answer}</code>
                    </div>
                    <div className="hint">Compressed input (first 600 characters): {ex.compressed_text}</div>
                  </div>
                ))
              ))}
          </div>
        </div>
      )}

      {security && (
        <div className="results" style={{ marginTop: 24 }}>
          <Security data={security} />
        </div>
      )}
    </div>
  );
}
