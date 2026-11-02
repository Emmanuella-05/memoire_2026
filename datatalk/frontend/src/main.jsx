import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// Appel API commun : renvoie le JSON ou lève une Error au message lisible
// (backend injoignable, 409 source absente, 422 entrée invalide, 503 LLM...).
async function api(path, options) {
  let response;
  try {
    response = await fetch(`${API_URL}${path}`, options);
  } catch {
    throw new Error("Backend injoignable. Vérifiez qu'il est lancé et que VITE_API_URL est correct.");
  }
  let data = null;
  try {
    data = await response.json();
  } catch {
    // réponse non JSON
  }
  if (!response.ok) {
    let detail = data?.detail;
    if (Array.isArray(detail)) detail = detail.map((d) => d.msg).join(" ; ");
    throw new Error(detail || `Erreur API ${response.status}`);
  }
  return data;
}

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
      const data = await api(`/upload/${target}`, { method: "POST", body });
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
      <small>{busy ? "Analyse automatique…" : message || "Choisir un fichier"}</small>
    </label>
  );
}

function MongoConnect({ connection, onChanged }) {
  // Connexion directe à une base MongoDB existante (sans fichier). Lecture seule.
  const [uri, setUri] = useState("mongodb://localhost:27017");
  const [database, setDatabase] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");

  async function connect(event) {
    event.preventDefault();
    setBusy(true);
    setMessage("");
    try {
      const data = await api("/connect/mongodb", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ uri: uri.trim(), database: database.trim() }),
      });
      const names = (data.collections || []).map((c) => `${c.name} (${c.documents ?? "?"})`).join(", ");
      setMessage(`✓ ${data.message} Collections : ${names}`);
      onChanged?.();
    } catch (err) {
      setMessage(`Erreur : ${err.message}`);
    } finally {
      setBusy(false);
    }
  }

  async function disconnect() {
    setBusy(true);
    setMessage("");
    try {
      await api("/connect/mongodb", { method: "DELETE" });
      onChanged?.();
    } catch (err) {
      setMessage(`Erreur : ${err.message}`);
    } finally {
      setBusy(false);
    }
  }

  if (connection?.source === "connected") {
    return (
      <div className="upload-box mongo-connect">
        <span>MongoDB connecté (lecture seule)</span>
        <small>{connection.server} · base <b>{connection.database}</b></small>
        <button type="button" onClick={disconnect} disabled={busy}>Déconnecter</button>
        {message && <small>{message}</small>}
      </div>
    );
  }

  return (
    <form className="upload-box mongo-connect" onSubmit={connect}>
      <span>Se connecter à MongoDB</span>
      <input type="text" value={uri} onChange={(e) => setUri(e.target.value)} placeholder="mongodb://localhost:27017"
        aria-label="URI MongoDB" disabled={busy} />
      <input type="text" value={database} onChange={(e) => setDatabase(e.target.value)} placeholder="Nom de la base"
        aria-label="Nom de la base MongoDB" disabled={busy} />
      <button type="submit" disabled={busy || !uri.trim() || !database.trim()}>
        {busy ? "Connexion et analyse…" : "Connecter"}
      </button>
      {message && <small>{message}</small>}
    </form>
  );
}

function DataTable({ rows }) {
  if (!rows?.length) return null;
  const shown = rows.slice(0, 50);
  const columns = [...new Set(shown.flatMap((row) => Object.keys(row)))];
  const cell = (v) => (v !== null && typeof v === "object" ? JSON.stringify(v) : String(v ?? ""));
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>{columns.map((c) => <th key={c}>{c}</th>)}</tr>
        </thead>
        <tbody>
          {shown.map((row, i) => (
            <tr key={i}>{columns.map((c) => <td key={c}>{cell(row[c])}</td>)}</tr>
          ))}
        </tbody>
      </table>
      {rows.length > shown.length && <small>{shown.length} lignes affichées sur {rows.length}.</small>}
    </div>
  );
}

function Catalog({ schemas, mappings, onSetStatus }) {
  const tables = schemas?.sqlite?.tables || [];
  const collections = schemas?.mongodb?.collections || [];
  if (!tables.length && !collections.length) return null;
  const both = tables.length > 0 && collections.length > 0;
  return (
    <details className="analysis" open>
      <summary>Analyse automatique</summary>
      <div className="catalog">
        {tables.length > 0 && (
          <div>
            <h3>SQLite</h3>
            <ul>{tables.map((t) => <li key={t.name}><b>{t.name}</b> · {t.columns.length} colonnes · {t.row_count ?? "?"} lignes</li>)}</ul>
          </div>
        )}
        {collections.length > 0 && (
          <div>
            <h3>MongoDB</h3>
            <ul>{collections.map((c) => <li key={c.name}><b>{c.name}</b> · {c.fields.length} champs · {c.document_count ?? "?"} documents</li>)}</ul>
          </div>
        )}
      </div>

      {both && (
        <>
          <h3>Correspondances SQL ↔ MongoDB</h3>
          {mappings.length ? (
            <ul className="mappings">
              {mappings.map((m, i) => (
                <li key={i} className={`mapping ${m.status}`}>
                  <span>
                    <b>{m.entity}</b> : {m.sqlite.table}.{m.sqlite.key} ↔ {m.mongodb.collection}.{m.mongodb.key}
                    {m.confidence != null && <small> · confiance {m.confidence}</small>}
                    <small> · {m.status === "suggested" ? "suggérée" : m.status === "confirmed" ? "confirmée" : "rejetée"}</small>
                    {m.status === "suggested" && m.usable === false && <small> · non utilisée tant qu'elle n'est pas confirmée</small>}
                  </span>
                  <span className="actions">
                    {m.status !== "confirmed" && <button type="button" onClick={() => onSetStatus(i, "confirmed")}>Confirmer</button>}
                    {m.status !== "rejected" && <button type="button" onClick={() => onSetStatus(i, "rejected")}>Rejeter</button>}
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="hint">
              Aucune correspondance détectée. Les questions qui combinent les deux bases ne pourront pas être traitées ;
              celles qui concernent une seule base fonctionnent normalement.
            </p>
          )}
        </>
      )}
    </details>
  );
}

function Trace({ execution, source }) {
  if (!execution) return null;
  const { sql_query, mongo_collection, mongo_pipeline, join_plan, join_stats, join_error, attempts, classifier_reason } = execution;
  return (
    <details>
      <summary>Traçabilité</summary>
      <ul className="trace">
        {source && <li><b>Source :</b> {source}{classifier_reason ? ` — ${classifier_reason}` : ""}</li>}
        {sql_query && <li><b>SQL :</b> <code>{sql_query}</code></li>}
        {mongo_pipeline && <li><b>MongoDB :</b> {mongo_collection} <code>{JSON.stringify(mongo_pipeline)}</code></li>}
        {join_plan && (
          <li>
            <b>Jointure :</b> {join_plan.sqlite_table}.{join_plan.sqlite_key} ↔ {join_plan.mongo_collection}.{join_plan.mongo_key} ({join_plan.how})
          </li>
        )}
        {execution.translated_question && <li><b>Question traduite pour les modèles ouverts :</b> {execution.translated_question}</li>}
        {join_plan?.decomposed && (
          <li>
            <b>Plan en deux étapes :</b>
            <ol>
              {[join_plan.first, join_plan.first === "sql" ? "mongo" : "sql"].map((src, i) => (
                <li key={src}>
                  Étape {i + 1} ({src === "sql" ? "SQL" : "MongoDB"}) : {join_plan[src === "sql" ? "sql_question" : "mongo_question"]}
                  {i === 0 && execution.hybrid_key_count != null ? ` → ${execution.hybrid_key_count} identifiant(s) transmis à l'étape 2` : ""}
                </li>
              ))}
            </ol>
          </li>
        )}
        {join_plan && join_plan.decomposed === false && (
          <li>
            <b>Décomposition :</b> non établie ({join_plan.plan_note || "raison inconnue"}) — la question entière est envoyée aux deux bases.
          </li>
        )}
        {join_error && <li><b>Relation :</b> {join_error}</li>}
        {join_stats && (
          <li>
            <b>Fusion :</b> {join_stats.joined_rows} lignes · {join_stats.unmatched_left_rows} SQL sans correspondance · {join_stats.unmatched_right_rows} MongoDB sans correspondance
            {join_stats.one_to_many ? " · relation un-à-plusieurs" : ""}
          </li>
        )}
        {(attempts?.sql || attempts?.mongo) && (
          <li><b>Tentatives :</b> {attempts.sql ? `SQL ${attempts.sql}` : ""} {attempts.mongo ? `MongoDB ${attempts.mongo}` : ""}</li>
        )}
      </ul>
      <details>
        <summary>Données brutes</summary>
        <pre>{JSON.stringify(execution, null, 2)}</pre>
      </details>
    </details>
  );
}

function Result({ result }) {
  const status = result.status || "ok";
  return (
    <section className={`result ${status}`}>
      <h2>
        Résultat{" "}
        <span className={`badge ${status}`}>
          {status === "ok" ? "OK" : status === "no_data" ? "Aucune donnée" : "Erreur"}
        </span>
      </h2>
      {result.answer && <p className="answer">{result.answer}</p>}
      <DataTable rows={result.data} />
      <Trace execution={result.execution} source={result.source} />
    </section>
  );
}

function App() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [workspace, setWorkspace] = useState(null);
  const [schemas, setSchemas] = useState(null);
  const [mappings, setMappings] = useState([]);

  async function refreshCatalog() {
    try {
      const [ws, sc, mp] = await Promise.all([api("/workspace"), api("/docs/schemas"), api("/mappings")]);
      setWorkspace(ws);
      setSchemas(sc);
      setMappings(mp.mappings || []);
    } catch (err) {
      setError(err.message);
    }
  }

  useEffect(() => {
    refreshCatalog();
  }, []);

  async function ask() {
    if (!question.trim()) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      setResult(await api("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question.trim() }),
      }));
    } catch (err) {
      setError(`Impossible d'interroger DataTalk : ${err.message}`);
    } finally {
      setLoading(false);
    }
  }

  async function reanalyze() {
    setError("");
    try {
      await api("/analyze", { method: "POST" });
      await refreshCatalog();
    } catch (err) {
      setError(`Analyse impossible : ${err.message}`);
    }
  }

  async function setMappingStatus(index, status) {
    setError("");
    const updated = mappings.map((m, i) => (i === index ? { ...m, status } : m));
    try {
      const data = await api("/mappings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mappings: updated }),
      });
      setMappings(data.mappings);
    } catch (err) {
      setError(`Correspondance non enregistrée : ${err.message}`);
    }
  }

  const status = workspace
    ? `MongoDB : ${workspace.mongodb_connection ? `serveur ${workspace.mongodb_connection.database}` : "moteur embarqué"} · RAG : ${workspace.rag_documents ?? "?"} documents indexés${workspace.generation === "open" ? " · requêtes générées par des modèles ouverts" : workspace.claude_configured ? "" : " · clé Claude absente"}`
    : "Connexion au workspace…";

  return (
    <main className="page">
      <section className="card">
        <header>
          <h1>DataTalk</h1>
          <p>Interrogez vos données SQLite et MongoDB en langage naturel.</p>
        </header>

        <section className="uploads">
          <h2>Configurer les données</h2>
          <p className="hint">
            Chargez la base dont vous avez besoin : SQLite, MongoDB, ou les deux. DataTalk inspecte
            automatiquement les schémas et génère la documentation ; si les deux bases sont chargées,
            il propose aussi les correspondances SQL ↔ MongoDB.
          </p>

          <div className="upload-grid">
            <UploadBox target="sqlite-db" label="Base SQLite" accept=".db,.sqlite,.sqlite3"
              onUploaded={() => { setError(""); refreshCatalog(); }} />
            <MongoConnect connection={workspace?.mongodb_connection}
              onChanged={() => { setError(""); refreshCatalog(); }} />
            {workspace?.mongodb_connection?.source !== "connected" && (
              <UploadBox target="mongodb-data" label="Ou données MongoDB (JSON)" accept=".json"
                onUploaded={() => { setError(""); refreshCatalog(); }} />
            )}
            <UploadBox target="business-rules" label="Règles métier (optionnel)" accept=".json,.txt,.md"
              onUploaded={() => { setError(""); refreshCatalog(); }} />
          </div>

          <div className="workspace-status">
            <span>{status}</span>
            <button type="button" onClick={reanalyze}>Réanalyser</button>
          </div>

          <Catalog schemas={schemas} mappings={mappings} onSetStatus={setMappingStatus} />
        </section>

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
        {result && <Result result={result} />}
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);