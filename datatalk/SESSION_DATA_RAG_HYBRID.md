# DataTalk — Suivi de travail / passation

> Document de passation pour l'intégration LangGraph. Le travail ci-dessous concerne principalement les parties **Données / RAG / Hybrid / Interface** et est directement sur `main`.

## 1. Contexte

Dépôt : `Octave2MK/memoire_2026`

Branche utilisée : `main`

Le principe retenu est maintenant : **l'utilisateur fournit les données, pas la documentation technique des bases**. DataTalk inspecte automatiquement les bases disponibles, génère une documentation structurée pour le RAG et propose les correspondances SQL ↔ MongoDB.

Les règles métier restent optionnelles : elles peuvent être fournies par l'utilisateur pour compléter la connaissance technique par des contraintes métier explicites.

## 2. Ce qui est en place

### Backend — `datatalk/backend/tools.py`

Le fichier contient les services réutilisables par les agents LangGraph :

- adaptateur Claude API via `anthropic` ;
- configuration par variables d'environnement : `CLAUDE_API_KEY`, `CLAUDE_MODEL` ;
- analyse automatique du schéma SQLite ;
- analyse automatique du schéma MongoDB à partir des collections et d'un échantillon de documents ;
- génération automatique de `data/sqlite/database_docs.json` ;
- génération automatique de `data/mongodb/database_docs.json` ;
- génération automatique de `data/mappings.json` lorsque des correspondances de clés sont détectées ;
- stockage optionnel des règles métier dans `data/business_rules.json` ;
- RAG local TF-IDF + similarité cosinus sur documentation générée + règles métier ;
- cache du RAG et fonction de réinitialisation ;
- exécution SQL en lecture seule (`SELECT` / `WITH`) ;
- exécution MongoDB par pipeline d'agrégation ;
- fusion déterministe de résultats avec pandas ;
- fonctions de consultation du catalogue de correspondances ;
- fonction `analyze_workspace()` pour relancer toute l'analyse.

**Principe important :** le LLM décide quoi demander et comment formuler la requête ; l'inspection des bases, la génération de métadonnées, l'accès aux données et la fusion pandas restent déterministes.

## 3. Analyse automatique des bases

### SQLite

`analyze_sqlite_schema()` inspecte :

- tables ;
- colonnes ;
- types SQLite ;
- nullabilité ;
- clés primaires ;
- clés étrangères ;
- index.

La documentation générée conserve les informations utiles au RAG et aux Schema Analysts.

### MongoDB

`analyze_mongodb_schema()` inspecte :

- collections ;
- nombre de documents ;
- échantillon de documents ;
- chemins de champs imbriqués ;
- types observés.

L'analyse est volontairement basée sur un échantillon pour éviter de parcourir inutilement toute la base.

### Correspondances SQL ↔ MongoDB

`analyze_correspondences()` compare les champs susceptibles d'être des clés et génère un catalogue explicite.

Le premier mécanisme est volontairement conservateur :

- champ de type identifiant/clé ;
- nom normalisé identique ;
- compatibilité de type ;
- confiance et méthode conservées pour l'explicabilité.

Le catalogue généré reste une source structurée pour le **Join Planner**. Le RAG ne doit pas être utilisé pour inventer une clé de jointure.

Exemple de contrat :

```json
{
  "mappings": [
    {
      "entity": "customer",
      "sqlite": {"table": "customers", "key": "customer_id"},
      "mongodb": {"collection": "reviews", "key": "customer_id"},
      "relation": "same_normalized_key",
      "confidence": 0.98,
      "method": "deterministic_name_match"
    }
  ]
}
```

## 4. RAG

Le RAG est une **infrastructure partagée**, pas un dixième agent.

Il indexe automatiquement :

- `data/sqlite/database_docs.json` ;
- `data/mongodb/database_docs.json` ;
- `data/business_rules.json` si des règles métier ont été fournies.

Le retriever retourne au maximum quelques documents pertinents (par défaut `k=3`) afin de limiter la taille des prompts.

Les métadonnées permettent de distinguer `sqlite.<table>`, `mongodb.<collection>` et `business.rule_X`.

Pour le Join Planner, utiliser directement le catalogue structuré via `mapping_context()`, `find_mapping()` ou `hybrid_correspondences()`.

## 5. Uploads via l'interface

L'API accepte maintenant uniquement les entrées nécessaires :

- base SQLite : `.db`, `.sqlite`, `.sqlite3` ;
- données MongoDB au format JSON ;
- règles métier optionnelles : `.json`, `.txt`, `.md`.

Les documentations JSON et `mappings.json` ne sont plus des uploads obligatoires : ils sont générés automatiquement.

Les données MongoDB JSON suivent ce format :

```json
{
  "collection_name": [
    {"field": "value"},
    {"field": "value"}
  ]
}
```

Après import MongoDB, l'analyse du workspace est automatiquement relancée.

## 6. FastAPI

`datatalk/backend/interface.py` expose :

- `GET /health` ;
- `GET /workspace` ;
- `POST /analyze` pour relancer explicitement l'analyse ;
- `POST /query` ;
- `POST /upload/sqlite-db` ;
- `POST /upload/business-rules` ;
- `POST /upload/mongodb-data` ;
- `POST /claude/test`.

Le endpoint `/query` reste découplé de LangGraph : il appelle `build_graph()` depuis `backend.graph` et récupère ensuite les champs de l'état.

## 7. Frontend

`datatalk/frontend/src/main.jsx` propose maintenant :

- upload de la base SQLite ;
- upload des données MongoDB JSON ;
- upload optionnel des règles métier ;
- affichage du statut RAG ;
- affichage du statut d'analyse automatique ;
- bouton de réanalyse ;
- question en langage naturel ;
- résultat ;
- traçabilité avec route, sources et correspondances.

L'utilisateur n'a donc plus à connaître ni fournir le format interne des documentations générées.

## 8. Contrat attendu côté LangGraph

L'API attend que `build_graph()` soit disponible dans `backend/graph.py`.

L'état retourné devrait idéalement contenir :

```text
question
query_type
routing_reason
sql_subquestion
sql_schema_context
sql_query
sql_result
sql_error
sql_attempts
mongo_subquestion
mongo_schema_context
mongo_query
mongo_result
mongo_error
join_plan
merged_result
retrieved_context
sources_used
correspondences_used
final_answer
```

Le `/query` lit notamment :

- `final_answer` (ou `answer`) ;
- `merged_result`, sinon `sql_result`, sinon `mongo_result` ;
- `query_type` ;
- `sources_used` ;
- `correspondences_used` (ou `join_plan`).

## 9. Architecture d'intégration visée

```text
React
  ↓
FastAPI
  ↓
LangGraph
  ↓
Orchestrator / Router
  ├── SQL Schema Analyst → SQL Generator → SQL Evaluator
  ├── Mongo Schema Analyst → Mongo Generator → Mongo Evaluator
  └── HYBRID
        ├── branche SQL
        └── branche MongoDB
                ↓
           Join Planner
                ↓
          Result Merger
                ↓
             pandas
                ↓
           réponse finale
```

Les 9 composants/agents retenus par le cahier des charges restent :

1. Orchestrator / Router
2. SQL Schema Analyst
3. SQL Generator
4. SQL Evaluator
5. MongoDB Schema Analyst
6. MongoDB Generator
7. MongoDB Evaluator
8. Join Planner
9. Result Merger

Le RAG, l'analyse automatique des bases et les accès DB sont des outils/services partagés, pas des agents supplémentaires.

## 10. Fonctions utiles pour la collègue

RAG :

```python
from .tools import get_rag
context = get_rag().context(question, database="sqlite", k=3)
```

SQLite :

```python
from .tools import execute_sql
rows = execute_sql(sql_query)
```

MongoDB :

```python
from .tools import execute_mongo
rows = execute_mongo(collection, pipeline)
```

Catalogue hybride :

```python
from .tools import mapping_context, find_mapping, hybrid_correspondences
```

Fusion :

```python
from .tools import merge_on_key
merged = merge_on_key(sql_rows, mongo_rows, "customer_id", "customer_id")
```

Claude :

```python
from .tools import claude_generate
text = claude_generate(prompt)
```

## 11. Structure de données

```text
datatalk/
├── backend/
│   ├── agent.py
│   ├── graph.py
│   ├── interface.py
│   ├── main.py
│   └── tools.py
├── data/
│   ├── sqlite/
│   │   └── database_docs.json      # généré automatiquement
│   ├── mongodb/
│   │   └── database_docs.json      # généré automatiquement
│   ├── mappings.json                # généré automatiquement
│   ├── business_rules.json          # optionnel
│   ├── datatalk.db
│   └── README.md
├── frontend/
├── requirements.txt
├── .env.example
└── .gitignore
```

## 12. Configuration locale

Créer `.env` à partir de `.env.example` et renseigner notamment :

```text
CLAUDE_API_KEY=...
CLAUDE_MODEL=...
MONGO_URI=mongodb://localhost:27017
MONGO_DB=datatalk
FRONTEND_ORIGIN=http://localhost:5173
VITE_API_URL=http://localhost:8000
```

La clé Claude ne doit jamais être commitée.

## 13. État actuel / prochaines priorités

### En place

- stockage des données ;
- analyse automatique SQLite/MongoDB ;
- génération des documentations ;
- génération prudente des correspondances ;
- RAG local ;
- règles métier optionnelles ;
- outils SQL/Mongo/pandas ;
- adaptateur Claude ;
- uploads FastAPI ;
- endpoint de réanalyse ;
- interface React minimale ;
- contrat d'intégration LangGraph documenté.

### Priorités restantes

1. Implémenter/intégrer les 9 agents dans `agent.py` et la coordination dans `graph.py`.
2. Brancher les agents sur `get_rag()`, `mapping_context()`, `execute_sql()`, `execute_mongo()` et `merge_on_key()`.
3. Tester SQL simple, Mongo simple et plusieurs cas hybrides.
4. Ajouter l'évaluation des résultats et de la pertinence du plan.
5. Tester le frontend avec le backend réel.

## 14. Niveau de validation

Les modifications ont été écrites et poussées sur GitHub, directement sur `main`. **L'exécution complète de l'application n'a pas été effectuée dans cette session** ; il faut donc encore lancer les tests locaux avant de considérer le parcours end-to-end comme validé.

## 15. Historique récent

- passage explicite du workflow sur `main` ;
- remplacement du modèle « l'utilisateur fournit les docs/mappings » par « l'utilisateur fournit les données, DataTalk analyse automatiquement » ;
- ajout de l'analyse automatique SQLite ;
- ajout de l'analyse automatique MongoDB ;
- génération automatique des documentations JSON ;
- génération automatique des correspondances SQL ↔ MongoDB ;
- ajout du support optionnel des règles métier ;
- extension du RAG aux règles métier ;
- ajout de `GET /workspace` et `POST /analyze` ;
- simplification de l'interface React pour supprimer les uploads de documentation/mappings.


## 16. Réparation 1 — intégration LangGraph

La fausse implémentation `SimpleGraph` a été remplacée par un vrai `StateGraph` LangGraph.

Flux actuellement branché :

```text
START → classifier
  ├── SQL    → sql_agent → result_merger → END
  ├── Mongo  → mongo_agent → result_merger → END
  └── Hybrid → hybrid_sql → hybrid_mongo → join_planner → result_merger → END
```

Le graphe expose `build_graph()`, `graph` et `run(question)`.

Le chemin hybride reste volontairement séquentiel pour cette première réparation.

Point restant : les agents SQL/Mongo encapsulent encore génération, validation, exécution et correction. La décomposition stricte en composants séparés du cahier des charges sera traitée ensuite.

Validation : code poussé sur `main` ; exécution locale end-to-end encore à faire.


## 17. Réparation 2 — FastAPI

`backend/main.py` a été reconnecté au vrai graphe via `from .graph import run`.

Routes restaurées pour compatibilité avec le frontend : `GET /health`, `GET /workspace`, `GET /status`, `POST /analyze`, `POST /query`, `POST /ask`, `POST /upload/sqlite-db`, `POST /upload/mongodb-data`, `POST /upload/business-rules`, `POST /claude/test`.

Les réponses `/query` exposent `answer`, `data` et `execution` avec les informations SQL/Mongo/join utiles au frontend. Les résultats non JSON natifs sont convertis en chaînes avant réponse HTTP.

Validation : modification poussée sur `main`. Le démarrage FastAPI et les appels réels restent à tester localement.


## 18. Réparation 3 — cohérence des imports Python

Le backend est lancé depuis la racine `datatalk/` avec `python -m uvicorn backend.main:app --reload`. Dans cette configuration, les imports internes devaient être cohérents avec le package `backend`.

Corrections appliquées sur `main` :

- `backend/graph.py` utilise maintenant les imports relatifs `.agents...`.
- les agents `classifier`, `sql_agent`, `mongo_agent`, `join_planner` et `result_merger` utilisent les imports relatifs vers `backend.llm` et `backend.tools`.
- `backend/tools/rag_tools.py` utilise `..rag`.
- `backend/rag/retriever.py` utilise `..tools.catalog_tools`.

Objectif : éviter les `ModuleNotFoundError` liés aux anciens imports `agents`, `tools` ou `llm` lorsque FastAPI importe `backend.main` comme package.

Validation effectuée : relecture des fichiers modifiés sur `main`. Le démarrage Python réel reste à exécuter localement, car aucune exécution end-to-end n'a encore été réalisée dans cette session.


## 19. Réparation 4 — contrat d'état HYBRID + environnement virtuel

Corrections appliquées sur `main` :

### Contrat HYBRID
- Le `Join Planner` est maintenant exécuté avant les agents SQL/Mongo en mode hybride.
- Le `join_plan` est donc disponible dès la génération des deux requêtes.
- L'agent SQL reçoit la clé SQLite de jointure et doit la retourner dans ses résultats.
- L'agent MongoDB reçoit la clé MongoDB de jointure et doit la retourner dans ses résultats.
- Si une source hybride retourne des lignes sans sa clé de jointure, l'agent déclenche son mécanisme de correction au lieu de produire une fusion incohérente.
- Le `Result Merger` ne concatène plus arbitrairement les résultats SQL et Mongo en cas de problème de jointure. Il retourne une erreur explicite.
- Une absence de résultat d'un côté hybride produit une jointure vide, et non une fausse concaténation.

### Environnement virtuel
Le `.gitignore` contenait déjà les règles `venv/`, `.venv/`, etc., mais cela ne suffisait pas car `datatalk/venv` était déjà suivi par Git.

Les **855 entrées** du répertoire virtuel suivi ont été retirées de l'arbre Git. Le répertoire local n'est pas supprimé par cette opération ; il est simplement retiré du dépôt et restera ignoré grâce au `.gitignore`.

Commit de nettoyage : `ca6a4fcf0aeed0cf6b2bf1170f7b5a03f743d84f`.

La prochaine étape est maintenant la validation réelle du backend et des contrats, avant d'ajouter de nouvelles fonctionnalités.
