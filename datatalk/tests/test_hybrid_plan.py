"""Tests du plan hybride en deux étapes dépendantes (faux LLM, vraies bases SQLite + mongomock).

Scénario : « Quelles villes comptent le plus de clients ayant signalé un produit endommagé ? »
Les tickets n'existent que dans MongoDB, les villes que dans SQLite.

Lancer :  python -m pytest tests -q      (depuis le dossier datatalk/)
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
from collections import Counter
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="datatalk-hybrid-test-")
os.environ["DATATALK_DATA_DIR"] = _TMP          # avant tout import de backend
os.environ.pop("MONGO_URI", None)
os.environ.pop("MONGO_DB", None)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from backend.agents import classifier, join_planner, mongo_agent, result_merger, sql_agent  # noqa: E402
from backend.tools import catalog_tools, hybrid_tools  # noqa: E402
from backend.tools.schema_tools import mongo_schema_summary, sql_schema_summary  # noqa: E402

QUESTION = "Quelles villes comptent le plus de clients ayant signalé un produit endommagé ?"
MONGO_SUBQ = "Identifiants distincts (customer_id) des clients ayant un ticket de catégorie produit_endommage"
SQL_SUBQ = "Nombre de clients par ville, du plus grand au plus petit"
CITIES = ["Cotonou", "Porto-Novo", "Parakou"]
CATEGORIES = ["produit_endommage", "retard_livraison", "remboursement"]

CUSTOMERS = [(i, f"Client {i}", CITIES[i % 3]) for i in range(1, 31)]
TICKETS = [{"customer_id": i, "category": CATEGORIES[(i + j) % 3], "message": f"ticket {i}-{j}"}
           for i in range(1, 31) for j in range(2)]
DAMAGED = sorted({t["customer_id"] for t in TICKETS if t["category"] == "produit_endommage"})
EXPECTED_BY_CITY = Counter(city for cid, _, city in CUSTOMERS if cid in DAMAGED)


@pytest.fixture(scope="module", autouse=True)
def workspace():
    from backend.tools.catalog_tools import SQLITE_DB
    from backend.tools.ingest_tools import ingest_mongo_payload, ingest_sqlite_file

    staging = Path(_TMP) / "upload.db"
    conn = sqlite3.connect(staging)
    conn.execute("CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT, city TEXT)")
    conn.executemany("INSERT INTO customers VALUES (?,?,?)", CUSTOMERS)
    conn.commit(); conn.close()
    ingest_sqlite_file(staging)
    ingest_mongo_payload({"support_tickets": TICKETS})
    assert SQLITE_DB.exists() and catalog_tools.load_mappings(), "la correspondance customers <-> support_tickets doit être détectée"


class FakeLLM:
    """Remplace claude_json / claude_generate ; enregistre chaque appel."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []     # (rôle, prompt)
        self.responses: dict[str, list] = {}

    def on(self, role: str, *answers) -> None:
        self.responses[role] = list(answers)

    def count(self, role: str) -> int:
        return sum(1 for r, _ in self.calls if r == role)

    def prompts(self, role: str) -> list[str]:
        return [p for r, p in self.calls if r == role]

    def _role(self, system: str) -> str:
        for marker, role in (("classifieur", "classifier"), ("planificateur de requêtes hybrides", "planner"),
                             ("agent SQL", "sql"), ("validateur SQL", "sql_validate"),
                             ("agent MongoDB", "mongo"), ("validateur MongoDB", "mongo_validate"),
                             ("agent de synthèse", "merger")):
            if marker in system:
                return role
        raise AssertionError(f"appel LLM inattendu : {system[:80]!r}")

    def json(self, prompt: str, system: str = "", max_tokens: int = 0):
        role = self._role(system)
        self.calls.append((role, prompt))
        if role.endswith("_validate"):
            return {"valid": True}
        queue = self.responses.get(role)
        assert queue, f"aucune réponse prévue pour le rôle {role!r}"
        answer = queue[0] if len(queue) == 1 else queue.pop(0)
        return answer

    def generate(self, prompt: str, system: str = "", max_tokens: int = 0):
        self.calls.append((self._role(system), prompt))
        return "Réponse rédigée."


@pytest.fixture()
def llm(monkeypatch):
    fake = FakeLLM()
    for module in (classifier, join_planner, sql_agent, mongo_agent):
        monkeypatch.setattr(module, "claude_json", fake.json)
    monkeypatch.setattr(result_merger, "claude_generate", fake.generate)
    monkeypatch.setattr(sql_agent, "rag_context", lambda *a, **k: "")
    monkeypatch.setattr(mongo_agent, "rag_context", lambda *a, **k: "")
    fake.on("classifier", {"source": "hybrid", "reason": "test"})
    return fake


def run_graph(question: str = QUESTION) -> dict:
    import importlib
    from backend import graph
    importlib.reload(graph)       # reconstruit le graphe avec les agents patchés
    return graph.run(question)


MONGO_STEP1 = {"collection": "support_tickets", "pipeline": [
    {"$match": {"category": "produit_endommage"}},
    {"$group": {"_id": "$customer_id"}},
    {"$project": {"customer_id": "$_id", "_id": 0}},
]}
SQL_STEP2 = {"sql": "SELECT city, COUNT(*) AS clients FROM customers "
                    "WHERE customer_id IN (SELECT id FROM selected_ids) GROUP BY city ORDER BY clients DESC"}


# --------------------------------------------------------------------------
# Valeurs exactes dans les prompts
# --------------------------------------------------------------------------

def test_categorical_values_are_documented_exactly():
    docs = catalog_tools.load_database_docs("mongodb")
    fields = {f["name"]: f for c in docs["collections"] for f in c["fields"]}
    assert sorted(fields["category"]["distinct_values"]) == sorted(CATEGORIES)
    assert "distinct_values" not in fields["customer_id"] and "distinct_values" not in fields["message"]
    sql_cols = {c["name"]: c for t in catalog_tools.load_database_docs("sqlite")["tables"] for c in t["columns"]}
    assert sorted(sql_cols["city"]["distinct_values"]) == sorted(CITIES)
    assert "distinct_values" not in sql_cols["name"] and "distinct_values" not in sql_cols["customer_id"]


def test_prompts_show_exact_values():
    assert '"produit_endommage"' in mongo_schema_summary()          # sans accent, avec tiret bas
    assert '"Porto-Novo"' in sql_schema_summary()


# --------------------------------------------------------------------------
# Plan en deux étapes
# --------------------------------------------------------------------------

def test_mongo_first_then_sql_with_injected_ids(llm):
    llm.on("planner", {"mapping": 1, "join_type": "inner", "first": "mongo",
                       "mongo_question": MONGO_SUBQ, "sql_question": SQL_SUBQ})
    llm.on("mongo", MONGO_STEP1)
    llm.on("sql", SQL_STEP2)
    state = run_graph()

    assert state["status"] == "ok", state.get("error")
    assert {r["city"]: r["clients"] for r in state["merged_results"]} == dict(EXPECTED_BY_CITY)
    assert state["hybrid_key_count"] == len(DAMAGED)
    assert state["join_plan"]["decomposed"] and state["join_plan"]["first"] == "mongo"

    # chaque agent ne reçoit que SA sous-question, jamais la question d'origine
    for prompt in llm.prompts("mongo"):
        assert MONGO_SUBQ in prompt and QUESTION not in prompt and SQL_SUBQ not in prompt
    for prompt in llm.prompts("sql"):
        assert SQL_SUBQ in prompt and QUESTION not in prompt and "selected_ids" in prompt
    # le validateur juge chaque requête par rapport à sa sous-question
    assert all(SQL_SUBQ in p and QUESTION not in p for p in llm.prompts("sql_validate"))
    assert all(MONGO_SUBQ in p and QUESTION not in p for p in llm.prompts("mongo_validate"))
    # le planificateur voit les schémas, avec les valeurs exactes
    assert '"produit_endommage"' in llm.prompts("planner")[0] and "support_tickets" in llm.prompts("planner")[0]
    # la requête affichée indique d'où viennent les identifiants ; ils ne sont pas tapés par le LLM
    assert state["sql_query"].startswith(f"-- selected_ids = {len(DAMAGED)} identifiant(s) issus de l'étape 1")
    import re
    assert not any(re.search(r"IN\s*\(\s*\d+\s*,", p) for p in llm.prompts("sql"))   # aucune liste d'identifiants dans le prompt


def test_sql_first_then_mongo_with_injected_match(llm):
    llm.on("planner", {"mapping": 1, "join_type": "inner", "first": "sql",
                       "sql_question": "Identifiants distincts (customer_id) des clients de Parakou",
                       "mongo_question": "Nombre de tickets par catégorie"})
    llm.on("sql", {"sql": "SELECT DISTINCT customer_id FROM customers WHERE city = 'Parakou'"})
    llm.on("mongo", {"collection": "support_tickets", "pipeline": [
        {"$group": {"_id": "$category", "tickets": {"$sum": 1}}}, {"$sort": {"_id": 1}}]})
    state = run_graph("Combien de tickets par catégorie pour les clients de Parakou ?")

    parakou = {cid for cid, _, city in CUSTOMERS if city == "Parakou"}
    expected = Counter(t["category"] for t in TICKETS if t["customer_id"] in parakou)
    assert state["status"] == "ok", state.get("error")
    assert {r["_id"]: r["tickets"] for r in state["merged_results"]} == dict(expected)
    assert state["mongo_pipeline"][0]["$match"]["customer_id"]["$in"][:3] == sorted(parakou)[:3]
    assert state["hybrid_key_count"] == len(parakou)


def test_first_step_without_result_stops_before_second(llm):
    llm.on("planner", {"mapping": 1, "first": "mongo", "mongo_question": MONGO_SUBQ, "sql_question": SQL_SUBQ})
    llm.on("mongo", {"collection": "support_tickets", "pipeline": [
        {"$match": {"category": "inexistante"}}, {"$group": {"_id": "$customer_id"}},
        {"$project": {"customer_id": "$_id", "_id": 0}}]})
    llm.on("sql", SQL_STEP2)
    state = run_graph()
    assert state["status"] == "no_data" and "première étape" in state["answer"]
    assert llm.count("sql") == 0                      # la seconde source n'est jamais interrogée


def test_too_many_ids_is_an_explicit_error(llm, monkeypatch):
    from backend import graph
    monkeypatch.setattr(hybrid_tools, "MAX_JOIN_KEYS", 3)
    monkeypatch.setattr(graph, "MAX_JOIN_KEYS", 3)
    llm.on("planner", {"mapping": 1, "first": "mongo", "mongo_question": MONGO_SUBQ, "sql_question": SQL_SUBQ})
    llm.on("mongo", MONGO_STEP1)
    llm.on("sql", SQL_STEP2)
    import importlib
    importlib.reload(graph)
    monkeypatch.setattr(graph, "MAX_JOIN_KEYS", 3)
    state = graph.run(QUESTION)
    assert state["status"] == "error" and "limite" in state["error"] and llm.count("sql") == 0


def test_left_join_forces_sql_first(llm):
    llm.on("planner", {"mapping": 1, "join_type": "left", "first": "mongo",
                       "mongo_question": "Tickets par client", "sql_question": "Tous les clients"})
    plan, error = join_planner.plan_join(QUESTION)
    assert error is None and plan["how"] == "left" and plan["first"] == "sql" and plan["decomposed"]


def test_unusable_plan_falls_back_to_the_old_behaviour(llm):
    llm.on("planner", {"mapping": 1, "first": "mongo"})          # sous-questions manquantes
    llm.on("sql", {"sql": "SELECT customer_id, city FROM customers"})
    llm.on("mongo", {"collection": "support_tickets", "pipeline": [
        {"$match": {"category": "produit_endommage"}}, {"$project": {"customer_id": 1, "category": 1, "_id": 0}}]})
    state = run_graph()
    assert state["join_plan"]["decomposed"] is False and state["status"] == "ok"
    assert all(QUESTION in p for p in llm.prompts("sql") + llm.prompts("mongo"))   # question entière, comme avant
    assert state["join_stats"] is not None                                        # jointure en mémoire


def test_second_step_must_use_the_mapped_collection(llm):
    llm.on("planner", {"mapping": 1, "first": "sql", "sql_question": "Identifiants distincts des clients de Parakou",
                       "mongo_question": "Nombre de tickets par catégorie"})
    llm.on("sql", {"sql": "SELECT DISTINCT customer_id FROM customers WHERE city = 'Parakou'"})
    llm.on("mongo", {"collection": "customers", "pipeline": [{"$match": {}}]},
           {"collection": "support_tickets", "pipeline": [{"$group": {"_id": "$category", "n": {"$sum": 1}}}]})
    state = run_graph("Combien de tickets par catégorie pour les clients de Parakou ?")
    # 1re tentative rejetée (mauvaise collection, message explicite renvoyé au modèle), 2e acceptée
    assert state["status"] == "ok" and state["mongo_attempts"] == 2
    assert "support_tickets" in llm.prompts("mongo")[1]


# --------------------------------------------------------------------------
# Outils d'injection
# --------------------------------------------------------------------------

def test_with_selected_ids_wraps_plain_and_cte_queries():
    assert hybrid_tools.with_selected_ids("SELECT 1;", [1, 2]) == "WITH selected_ids(id) AS (VALUES (1), (2)) SELECT 1"
    wrapped = hybrid_tools.with_selected_ids("WITH a AS (SELECT 1) SELECT * FROM a", ["x'y"])
    assert wrapped == "WITH selected_ids(id) AS (VALUES ('x''y')), a AS (SELECT 1) SELECT * FROM a"
    assert hybrid_tools.with_selected_ids("with recursive r(n) AS (SELECT 1) SELECT n FROM r", [1]).startswith(
        "WITH recursive selected_ids(id) AS (VALUES (1)), ")


def test_keys_are_coerced_to_the_column_types():
    assert hybrid_tools.coerce_for_sql(["3", 4.0, "x"], "customers", "customer_id") == [3, 4, "x"]
    assert hybrid_tools.coerce_for_mongo(["3"], "support_tickets", "customer_id") == [3]
    assert hybrid_tools.extract_keys([{"k": 1}, {"k": "1"}, {"k": None}, {"k": 2}], "k") == [1, 2]
    pipeline = hybrid_tools.display_pipeline(hybrid_tools.with_selected_keys([], "k", list(range(100))), list(range(100)))
    assert pipeline[0]["$match"]["k"]["$in"][-1] == "… (+80 identifiants)"


# --------------------------------------------------------------------------
# Les questions à une seule source ne changent pas
# --------------------------------------------------------------------------

def test_single_source_questions_are_unchanged(llm):
    llm.on("classifier", {"source": "sql", "reason": "villes"})
    llm.on("sql", {"sql": "SELECT city, COUNT(*) AS n FROM customers GROUP BY city"})
    state = run_graph("Combien de clients par ville ?")
    assert state["status"] == "ok" and state["sql_query"].startswith("SELECT city")
    assert "selected_ids" not in state["sql_query"] and llm.count("planner") == 0

    llm.on("classifier", {"source": "mongo", "reason": "tickets"})
    llm.on("mongo", {"collection": "support_tickets", "pipeline": [{"$group": {"_id": "$category", "n": {"$sum": 1}}}]})
    state = run_graph("Combien de tickets par catégorie ?")
    assert state["status"] == "ok" and len(state["merged_results"]) == 3
    assert state["mongo_pipeline"][0].get("$group") is not None       # aucun $match injecté


# --------------------------------------------------------------------------
# Planificateur tolérant : variantes d'écriture, nouvel essai, raison du repli visible
# --------------------------------------------------------------------------

def test_planner_accepts_natural_spellings(llm):
    llm.on("planner", {"mapping": "1", "join_type": "INNER", "first": "MongoDB",
                       "mongo_question": MONGO_SUBQ, "sql_question": SQL_SUBQ})
    plan, error = join_planner.plan_join(QUESTION)
    assert error is None and plan["decomposed"] is True and plan["first"] == "mongo" and plan["how"] == "inner"
    llm.on("planner", {"mapping": 1, "first": "SQLite", "mongo_question": MONGO_SUBQ, "sql_question": SQL_SUBQ})
    assert join_planner.plan_join(QUESTION)[0]["first"] == "sql"


def test_planner_retries_once_and_tells_the_model_what_was_missing(llm):
    llm.on("planner", {"mapping": 1, "first": "mongo"},       # sous-questions manquantes
           {"mapping": 1, "first": "mongo", "mongo_question": MONGO_SUBQ, "sql_question": SQL_SUBQ})
    plan, _ = join_planner.plan_join(QUESTION)
    assert plan["decomposed"] is True and llm.count("planner") == 2
    assert "sql_question" in llm.prompts("planner")[1] and "inutilisable" in llm.prompts("planner")[1]


def test_fallback_keeps_the_reason_and_the_failure_message_explains_it(llm):
    llm.on("planner", {"mapping": 1, "first": "mongo"})       # inutilisable aux deux essais
    llm.on("sql", {"sql": "SELECT city FROM customers"})      # pas de colonne customer_id : rejeté à chaque tentative
    llm.on("mongo", {"collection": "support_tickets", "pipeline": [{"$match": {}}]})
    state = run_graph()
    plan = state["join_plan"]
    assert plan["decomposed"] is False and "sql_question" in plan["plan_note"] and llm.count("planner") == 2
    assert state["status"] == "error"
    assert "n'a pas pu être décomposée" in state["answer"] and "sql_question" in state["answer"]
