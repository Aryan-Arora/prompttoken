import { useRef, useState } from "react";

const API_URL = "http://localhost:8000";

const TOKENIZER_LABELS = {
  "gpt-4o": "GPT-4o",
  "gpt-3.5-4": "GPT-3.5 / 4",
  llama: "Llama-style",
};

function preview(s) {
  const t = s.replace(/\s+/g, " ").trim();
  return t.length > 40 ? t.slice(0, 37) + "..." : t;
}

export default function Compress({ text, setText }) {
  const textareaRef = useRef(null);
  const [spans, setSpans] = useState([]);
  const [selection, setSelection] = useState({ start: 0, end: 0 });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [copied, setCopied] = useState(false);

  const onSelect = (e) => {
    setSelection({ start: e.target.selectionStart, end: e.target.selectionEnd });
  };

  const lockSelection = () => {
    if (selection.end <= selection.start) return;
    setSpans((prev) =>
      prev.some((s) => s.start === selection.start && s.end === selection.end)
        ? prev
        : [...prev, selection].sort((a, b) => a.start - b.start)
    );
  };

  const onChange = (e) => {
    setText(e.target.value);
    setSpans([]);
    setResult(null);
  };

  const compress = async () => {
    setLoading(true);
    setError(null);
    setCopied(false);
    try {
      const res = await fetch(`${API_URL}/compress`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, protected_spans: spans }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Request failed: ${res.status}`);
      }
      setResult(await res.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const copy = async () => {
    await navigator.clipboard.writeText(result.compressed);
    setCopied(true);
  };

  const hasSelection = selection.end > selection.start;

  return (
    <div>
      <textarea
        ref={textareaRef}
        value={text}
        onChange={onChange}
        onSelect={onSelect}
        rows={8}
        placeholder="Paste a prompt..."
      />

      <div className="protect-row">
        <button className="secondary" onClick={lockSelection} disabled={!hasSelection}>
          Lock selection as protected
        </button>
        <span className="hint">
          Select text above, then lock it. Locked text is never changed.
        </span>
      </div>

      {spans.length > 0 && (
        <div className="chips">
          {spans.map((s, i) => (
            <span key={i} className="chip">
              Locked: "{preview(text.slice(s.start, s.end))}"
              <button
                className="chip-x"
                aria-label="Remove protected span"
                onClick={() => setSpans(spans.filter((_, j) => j !== i))}
              >
                x
              </button>
            </span>
          ))}
        </div>
      )}

      <button onClick={compress} disabled={loading || !text.trim()}>
        {loading ? "Compressing..." : "Compress"}
      </button>

      {error && <p className="error">{error}</p>}

      {result && (
        <div className="results">
          <div className="stats">
            {Object.entries(result.tokens).map(([id, t]) => (
              <div key={id} className="stat">
                <div className="stat-label">{TOKENIZER_LABELS[id] || id}</div>
                <div className="stat-main">
                  {t.before} to {t.after} tokens
                </div>
                <div className="stat-sub">
                  {t.saved} saved ({t.saved_pct}%)
                </div>
                <div className="stat-sub">
                  ${t.cost_saved_per_1k_calls_usd} saved per 1,000 calls
                </div>
              </div>
            ))}
            <div className="stat">
              <div className="stat-label">Protected spans</div>
              <div className="stat-main">
                {result.protected.intact} / {result.protected.total} intact
              </div>
              <div className="stat-sub">Compression took {result.latency_ms} ms</div>
            </div>
          </div>

          <p className="hint">
            Cost uses ${result.usd_per_mtok} per 1M input tokens. {result.price_note} Quality
            retained is not measured yet; that comes from the Week 4 benchmark.
          </p>

          {result.lost.length > 0 ? (
            <div className="risk risk-warn">
              <strong>Meaning-risk check:</strong> {result.lost.length} risky item
              {result.lost.length === 1 ? "" : "s"} removed and not present anywhere in the
              compressed prompt.
              <ul>
                {result.lost.map((item, i) => (
                  <li key={i}>
                    <code>{item.token}</code> ({item.reason})
                  </li>
                ))}
              </ul>
            </div>
          ) : (
            <div className="risk risk-ok">
              <strong>Meaning-risk check:</strong> no numbers, names, negations, or constraint
              words were lost. This is a heuristic, not a guarantee.
            </div>
          )}

          <div className="result-card">
            <h2>
              Diff
              <span className="count">removed text struck through</span>
            </h2>
            <div className="diff">
              {result.segments.map((s, i) => (
                <span
                  key={i}
                  className={`seg seg-${s.kind}${s.risky ? " seg-risky" : ""}`}
                  title={s.risky ? "Lost: " + s.lost.map((l) => l.token).join(", ") : undefined}
                >
                  {s.text}
                </span>
              ))}
            </div>
          </div>

          <div className="result-card">
            <h2>
              Compressed prompt
              <button className="secondary small" onClick={copy}>
                {copied ? "Copied" : "Copy"}
              </button>
            </h2>
            <pre className="compressed">{result.compressed}</pre>
          </div>
        </div>
      )}
    </div>
  );
}
