"""Agent join planner : s'exécute seulement quand le classifieur a choisi
"hybrid". Il fait deux choses :

1. choisit la correspondance du catalogue (tools/catalog_tools.py) qui relie
   SQLite et MongoDB (la clé commune) ;
2. DÉCOMPOSE la question en deux sous-questions dépendantes, une par source, et
   fixe l'ordre : la source qui porte les conditions de filtrage est interrogée
   en premier et retourne les identifiants ; la seconde répond ensuite pour ces
   seuls identifiants (injectés par le code, voir tools/hybrid_tools.py).

Une question « hybride » n'est exécutée que si l'on sait comment relier les
données. Sinon, aucune requête n'est lancée : join_error explique pourquoi.
Si le modèle ne produit pas de décomposition exploitable, on retombe sur
l'ancien comportement (la question entière envoyée aux deux agents, jointure
en mémoire) : `decomposed` vaut alors False.
"""

from __future__ import annotations

from typing import Any

from ..llm import claude_json
from ..tools.catalog_tools import load_mappings
from ..tools.schema_tools import mongo_schema_summary, sql_schema_summary

MAX_SUBQUESTION_CHARS = 600

SYSTEM_PROMPT = """Tu es le planificateur de requêtes hybrides de DataTalk.
Une question hybride a besoin de DEUX sources : SQLite (tables) et MongoDB
(collections). On te donne le schéma des deux sources (avec les valeurs exactes
des colonnes catégorielles), la liste numérotée des correspondances connues
(clé commune entre une table SQLite et une collection MongoDB) et la question.

Décompose la question en DEUX étapes dépendantes, chacune traitée par UNE seule
source :
- "first" : la source interrogée en premier, celle qui contient les CONDITIONS
  qui restreignent les entités (ex. « tickets de telle catégorie », « clients de
  telle ville »). Sa sous-question demande uniquement les identifiants (la clé de
  la correspondance choisie) des entités concernées ; n'écris pas « distincts » :
  les doublons sont éliminés automatiquement.
- l'autre source répond ensuite au reste de la question, pour les seules entités
  sélectionnées par la première. Les identifiants lui sont fournis automatiquement :
  ne les mentionne pas et n'écris aucun identifiant dans les sous-questions.

Règles pour chaque sous-question :
- autonome et complète ; elle ne parle que de tables/collections/colonnes/champs
  de SA source, jamais de ceux de l'autre source ;
- elle reprend les valeurs exactes de la question, orthographiées comme dans le schéma.
- Si join_type vaut "left" (toutes les entités SQLite, même sans données MongoDB),
  "first" doit valoir "sql".
- Choisis la correspondance dont l'entité (la clé) et les tables sont celles dont parle la question ;
  entre deux candidates, préfère la confirmée ou celle de plus forte confiance. Ne choisis jamais une
  correspondance dont la clé désigne une autre entité que celle reliée par la question.
- join_type : "inner" (par défaut, seules les entités présentes dans les deux
  sources) ou "left".

Exemple. Correspondance : 1. clients: SQLite clients(client_id) -> MongoDB incidents(client_id).
Question : « Combien de clients par ville ont signalé un colis perdu ? »
Réponse : {"mapping": 1, "join_type": "inner", "first": "mongo",
"mongo_question": "Identifiants (client_id) des clients ayant au moins un incident de type perdu",
"sql_question": "Nombre de clients par ville, du plus grand au plus petit"}

Autre exemple, dans l'ordre inverse. Correspondance : 1. produits: SQLite articles(ref) -> MongoDB retours(ref).
Question : « Combien de retours par motif pour les articles de la catégorie jouets ? »
Réponse : {"mapping": 1, "join_type": "inner", "first": "sql",
"sql_question": "Références (ref) des articles de la catégorie jouets",
"mongo_question": "Nombre de retours par motif"}
Ici le filtre (catégorie) est dans SQLite, donc SQLite passe en premier.

Réponds STRICTEMENT avec un JSON de la forme :
{"mapping": <numéro>, "join_type": "inner" | "left", "first": "sql" | "mongo",
 "sql_question": "<sous-question SQL>", "mongo_question": "<sous-question MongoDB>"}
Aucun texte hors de ce JSON."""


def _plan(mapping: dict[str, Any], question: str, how: str = "inner", first: str = "sql",
          sql_question: str | None = None, mongo_question: str | None = None,
          decomposed: bool = False, plan_note: str | None = None,
          planner_output: Any = None) -> dict[str, Any]:
    plan = {
        "entity": mapping.get("entity"),
        "sqlite_table": mapping["sqlite"]["table"],
        "sqlite_key": mapping["sqlite"]["key"],
        "mongo_collection": mapping["mongodb"]["collection"],
        "mongo_key": mapping["mongodb"]["key"],
        "how": how,
        "status": mapping.get("status"),
        "confidence": mapping.get("confidence"),
        "decomposed": decomposed,
        "first": first,
        "sql_question": sql_question or question,
        "mongo_question": mongo_question or question,
    }
    if plan_note:   # pourquoi la décomposition n'a pas pu être établie (affiché dans la traçabilité)
        plan["plan_note"] = plan_note
        plan["planner_output"] = str(planner_output)[:500]
    return plan


def _text(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()[:MAX_SUBQUESTION_CHARS]
    return None


def _source(value: Any) -> str | None:
    """Accepte les variantes que le modèle écrit naturellement : « MongoDB », « SQLite »..."""
    if not isinstance(value, str):
        return None
    name = value.strip().lower()
    if name in ("sql", "sqlite", "sql_agent"):
        return "sql"
    if name in ("mongo", "mongodb", "mongo_agent"):
        return "mongo"
    return None


def _interpret(parsed: dict[str, Any], count: int) -> dict[str, Any]:
    """Extrait la décomposition d'une réponse du modèle ; `problems` liste ce qui manque."""
    problems: list[str] = []
    try:
        number: int | None = int(parsed.get("mapping", 1 if count == 1 else 0))
    except (TypeError, ValueError):
        number = None
    if number is None or not 1 <= number <= count:
        if count == 1:
            number = 1
        else:
            number = None
            problems.append("numéro de correspondance (« mapping ») invalide")
    how = parsed.get("join_type")
    how = how.strip().lower() if isinstance(how, str) and how.strip().lower() in ("inner", "left") else "inner"
    first = _source(parsed.get("first"))
    if not first:
        problems.append("champ « first » absent ou invalide (attendu : \"sql\" ou \"mongo\")")
    sql_q, mongo_q = _text(parsed.get("sql_question")), _text(parsed.get("mongo_question"))
    if not sql_q:
        problems.append("« sql_question » manquante")
    if not mongo_q:
        problems.append("« mongo_question » manquante")
    return {"number": number, "how": how, "first": first, "sql_q": sql_q, "mongo_q": mongo_q, "problems": problems}


def plan_join(question: str) -> tuple[dict[str, Any] | None, str | None]:
    """Retourne (plan, None) si la relation est identifiée, sinon (None, explication)."""
    mappings = load_mappings()
    if not mappings:
        return None, (
            "Cette question nécessite de combiner SQLite et MongoDB, mais aucune correspondance "
            "n'est définie entre les deux bases (ni détectée automatiquement, ni ajoutée manuellement). "
            "Ajoutez ou confirmez une correspondance (ex. clients.id ↔ avis.customer_id) puis reposez la question."
        )

    def reliability(m: dict[str, Any]) -> str:
        if m.get("status") == "confirmed" or m.get("source") == "manual":
            return " [confirmée]"
        return f" [détectée automatiquement, confiance {m['confidence']}]" if m.get("confidence") is not None else ""

    listing = "\n".join(
        f"{i}. {m.get('entity')}: SQLite {m['sqlite']['table']}({m['sqlite']['key']}) "
        f"-> MongoDB {m['mongodb']['collection']}({m['mongodb']['key']}){reliability(m)}"
        for i, m in enumerate(mappings, start=1)
    )
    prompt = (
        f"Schéma SQLite :\n{sql_schema_summary()}\n\n"
        f"Schéma MongoDB :\n{mongo_schema_summary()}\n\n"
        f"Correspondances disponibles :\n{listing}\n\n"
        f"Question : {question}"
    )
    parsed: Any = {}
    found: dict[str, Any] = {"problems": ["aucune réponse exploitable"], "number": None}
    current = prompt
    for _ in range(2):   # un nouvel essai, en disant au modèle ce qui manquait
        try:
            parsed = claude_json(current, system=SYSTEM_PROMPT, max_tokens=500)
        except ValueError as exc:   # JSON illisible (une panne du LLM, elle, remonte : LLMError)
            parsed = {}
            found = {"problems": [f"réponse illisible ({exc})"], "number": None}
        else:
            found = _interpret(parsed, len(mappings))
        if not found["problems"]:
            break
        current = (f"{prompt}\n\nTa réponse précédente était inutilisable : {'; '.join(found['problems'])}. "
                   "Réponds avec le JSON complet demandé, sans autre texte.")

    if found["number"] is None:
        return None, "Plusieurs correspondances sont possibles et aucune n'a pu être choisie de façon fiable."
    mapping = mappings[found["number"] - 1]
    how = found["how"]
    if not found["problems"]:
        first = "sql" if how == "left" else found["first"]   # « toutes les entités SQLite » : SQLite est la base de départ
        return _plan(mapping, question, how, first, found["sql_q"], found["mongo_q"], decomposed=True), None
    # Pas de décomposition exploitable : ancien comportement (question entière, jointure en mémoire),
    # mais la raison est conservée et affichée.
    note = "; ".join(found["problems"])
    return _plan(mapping, question, how, plan_note=note, planner_output=parsed), None


def join_planner_node(state: dict[str, Any]) -> dict[str, Any]:
    """Nœud LangGraph : écrit join_plan, ou join_error si la relation est inconnue."""
    plan, error = plan_join(state["question"])
    return {"join_plan": plan, "join_error": error}
