import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

function App() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function ask() {
    if (!question.trim()) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const response = await fetch(`${API_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question.trim() }),
      });
      if (!response.ok) throw new Error(`API ${response.status}`);
      setResult(await response.json());
    } catch (err) {
      setError(`Impossible d'interroger DataTalk : ${err.message}`);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="page">
      <section className="card">
        <header>
          <h1>DataTalk</h1>
          <p>Interrogez vos données SQLite et MongoDB en langage naturel.</p>
        </header>
        <div className="query-row">
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                ask();
              }
            }}
            placeholder="Ex. Quels clients ont laissé un avis ?"
            rows={3}
          />
          <button onClick={ask} disabled={loading || !question.trim()}>
            {loading ? "Analyse…" : "Interroger"}
          </button>
        </div>
        {error && <p className="error">{error}</p>}
        {result && (
          <section className="result">
            <h2>Résultat</h2>
            {result.answer && <p>{result.answer}</p>}
            {result.data && <pre>{JSON.stringify(result.data, null, 2)}</pre>}
            {result.execution && (
              <details>
                <summary>Traçabilité</summary>
                <pre>{JSON.stringify(result.execution, null, 2)}</pre>
              </details>
            )}
          </section>
        )}
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
