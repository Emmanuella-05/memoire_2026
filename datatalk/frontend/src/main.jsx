import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

function UploadBox({ target, label, accept, onUploaded }) {
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");

  async function upload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setBusy(true);
    setMessage("");
    try {
      const body = new FormData();
      body.append("file", file);
      const response = await fetch(`${API_URL}/upload/${target}`, { method: "POST", body });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || `API ${response.status}`);
      setMessage(`✓ ${file.name}`);
      onUploaded?.(data);
    } catch (err) {
      setMessage(`Erreur : ${err.message}`);
    } finally {
      setBusy(false);
      event.target.value = "";
    }
  }

  return (
    <label className="upload-box">
      <span>{label}</span>
      <input type="file" accept={accept} onChange={upload} disabled={busy} />
      <small>{busy ? "Chargement…" : message || "Choisir un fichier"}</small>
    </label>
  );
}

function App() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [workspace, setWorkspace] = useState(null);

  async function ask() {
    if (!question.trim()) return;
    setLoading(true); setError(""); setResult(null);
    try {
      const response = await fetch(`${API_URL}/query`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question.trim() }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || `API ${response.status}`);
      setResult(data);
    } catch (err) {
      setError(`Impossible d'interroger DataTalk : ${err.message}`);
    } finally { setLoading(false); }
  }

  return (
    <main className="page">
      <section className="card">
        <header>
          <h1>DataTalk</h1>
          <p>Interrogez vos données SQLite et MongoDB en langage naturel.</p>
        </header>

        <section className="uploads">
          <h2>Configurer les données</h2>
          <p className="hint">Chargez la base SQLite et les documentations JSON utilisées par le RAG.</p>
          <div className="upload-grid">
            <UploadBox target="sqlite-db" label="Base SQLite" accept=".db,.sqlite,.sqlite3" onUploaded={setWorkspace} />
            <UploadBox target="sqlite-docs" label="Documentation SQLite" accept=".json" onUploaded={setWorkspace} />
            <UploadBox target="mongodb-docs" label="Documentation MongoDB" accept=".json" onUploaded={setWorkspace} />
            <UploadBox target="mappings" label="Correspondances SQL ↔ MongoDB" accept=".json" onUploaded={setWorkspace} />
          </div>
          {workspace?.workspace && <small className="status">RAG : {workspace.workspace.rag_documents} documents indexés.</small>}
        </section>

        <div className="query-row">
          <textarea value={question} onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); } }}
            placeholder="Ex. Quels clients ont laissé un avis ?" rows={3} />
          <button onClick={ask} disabled={loading || !question.trim()}>{loading ? "Analyse…" : "Interroger"}</button>
        </div>
        {error && <p className="error">{error}</p>}
        {result && (
          <section className="result">
            <h2>Résultat</h2>
            {result.answer && <p>{result.answer}</p>}
            {result.data && <pre>{JSON.stringify(result.data, null, 2)}</pre>}
            {result.execution && <details><summary>Traçabilité</summary><pre>{JSON.stringify(result.execution, null, 2)}</pre></details>}
          </section>
        )}
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
