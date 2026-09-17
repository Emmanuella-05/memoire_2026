# DataTalk — Suivi de travail / passation

> Document de passation pour la collègue chargée de l'intégration LangGraph. Il décrit ce qui est actuellement présent sur la branche `feature/data-rag-hybrid-interface`, ce qui reste à intégrer et les contrats à respecter.

## 1. Contexte

Le travail réalisé ici couvre principalement la partie **Données / RAG / Hybrid / Interface** du projet DataTalk.

La branche de travail est :

`feature/data-rag-hybrid-interface`

Le dépôt est : `Octave2MK/memoire_2026`

## 2. Ce qui est en place

### Backend — `datatalk/backend/tools.py`

Le fichier contient les services réutilisables par les agents LangGraph :

- adaptateur Claude API via `anthropic` ;
- configuration par variables d'environnement : `CLAUDE_API_KEY`, `CLAUDE_MODEL` ;
- chargement des documentations JSON SQLite/MongoDB ;
- chargement du catalogue `mappings.json` ;
- recherche d'une correspondance SQL ↔ MongoDB ;
- RAG local sur la documentation JSON avec TF-IDF + similarité cosinus ;
- cache du RAG et fonction de réinitialisation après upload ;
- validation des fichiers JSON de documentation et du catalogue de correspondances ;
- validation basique d'une base SQLite uploadée ;
- exécution SQL en lecture seule (`SELECT` / `WITH`) ;
- exécution MongoDB par pipeline d'agrégation ;
- inspection du schéma SQLite avec `sqlite_schema()` ;
- fusion déterministe de deux résultats avec pandas (`merge_on_key`) ;
- récupération des correspondances hybrides.

**Principe important :** le LLM décide quoi demander et comment formuler la requête ; l'accès aux bases, la récupération documentaire et la fusion pandas restent des opérations déterministes.

### RAG

Le RAG est conçu comme une **infrastructure partagée**, pas comme un dixième agent.

Les documents indexés sont les documentations JSON de :

- `data/sqlite/database_docs.json`
- `data/mongodb/database_docs.json`

Le retriever retourne au maximum quelques documents pertinents (par défaut `k=3`) afin de limiter la taille des prompts.

Les métadonnées conservées permettent de savoir si le contexte vient de `sqlite.<table>` ou `mongodb.<collection>`.

### Hybrid

Le catalogue explicite des correspondances est :

`data/mappings.json`

Exemple de contrat :

```json
{
  "mappings": [
    {
      "entity": "customer",
      "sqlite": {"table": "customers", "key": "customer_id"},
      "mongodb": {"collection": "reviews", "key": "customer_id"},
      "relation": "same_customer_id"
    }
  ]
}
```

Le **Join Planner** doit lire ce catalogue directement. Il ne faut pas demander au RAG de deviner une clé de jointure uniquement par similarité sémantique.

La fusion finale doit être faite avec `merge_on_key()` / pandas et non par un LLM.

### Uploads via l'interface

L'API accepte maintenant :

- une base SQLite (`.db`, `.sqlite`, `.sqlite3`) ;
- les données MongoDB sous forme JSON ;
- la documentation SQLite JSON ;
- la documentation MongoDB JSON ;
- le catalogue de correspondances JSON.

Les documentations et mappings sont validés avant remplacement des fichiers actifs.

Les données MongoDB JSON suivent ce format :

```json
{
  "collection_name": [
    {"field": "value"},
    {"field": "value"}
  ]
}
```

L'import écrit ces documents dans la base MongoDB configurée par `MONGO_URI` et `MONGO_DB`.

### FastAPI

`datatalk/backend/interface.py` expose :

- `GET /health`
- `POST /query`
- `POST /upload/{target}` pour SQLite/docs/mappings
- `POST /upload/mongodb-data`
- `POST /claude/test`

`datatalk/backend/main.py` expose l'application FastAPI afin de pouvoir lancer :

```bash
uvicorn backend.main:app --reload
```

Le endpoint `/query` est volontairement découplé de l'implémentation LangGraph : il appelle `build_graph()` depuis `backend.graph` et récupère ensuite les champs de l'état.

### Frontend

Une interface React/Vite minimale est présente dans `datatalk/frontend/`.

Elle permet :

- d'uploader la base SQLite ;
- d'uploader les données MongoDB JSON ;
- d'uploader les documentations SQLite/MongoDB ;
- d'uploader `mappings.json` ;
- de poser une question en langage naturel ;
- d'afficher la réponse ;
- d'afficher les données retournées ;
- d'afficher une section de traçabilité contenant la route, les sources et les correspondances.

L'URL du backend est configurable avec `VITE_API_URL`.

## 3. Structure des données actuellement prévue

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
│   │   └── database_docs.json
│   ├── mongodb/
│   │   └── database_docs.json
│   ├── mappings.json
│   ├── datatalk.db
│   └── README.md
├── frontend/
├── requirements.txt
├── .env.example
└── .gitignore
```

Les fichiers de documentation et `mappings.json` présents dans la branche sont actuellement des **templates vides** : les vrais schémas et vraies correspondances doivent encore être fournis par l'équipe. Aucune structure métier n'a été inventée.

## 4. Contrat attendu côté LangGraph

L'API attend que `build_graph()` soit disponible dans `backend/graph.py`.

L'état retourné devrait idéalement contenir les champs suivants :

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
mongo_attempts
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

## 5. Architecture d'intégration visée

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

Le RAG et les accès DB sont des outils/services partagés, pas des agents supplémentaires.

## 6. Points importants pour la collègue

### Utiliser les outils existants

Les agents peuvent appeler les fonctions de `tools.py` plutôt que de réimplémenter l'accès aux données.

Pour le RAG :

```python
from .tools import get_rag
context = get_rag().context(question, database="sqlite", k=3)
```

Pour SQLite :

```python
from .tools import execute_sql
rows = execute_sql(sql_query)
```

Pour MongoDB :

```python
from .tools import execute_mongo
rows = execute_mongo(collection, pipeline)
```

Pour le catalogue hybride :

```python
from .tools import find_mapping, hybrid_correspondences
```

Pour la fusion :

```python
from .tools import merge_on_key
merged = merge_on_key(sql_rows, mongo_rows, "customer_id", "customer_id")
```

Pour Claude :

```python
from .tools import claude_generate
text = claude_generate(prompt)
```

### Ne pas ajouter d'agent RAG

Le RAG est déjà un service. Il peut être appelé depuis le Router, les Schema Analysts, les Generators et éventuellement le Join Planner.

### Ne pas ajouter d'agent de fusion LLM

La fusion hybride est déterministe et doit rester dans pandas.

### Ne pas dépendre du schéma réel avant son import

Les templates JSON sont volontairement vides. Dès que les vrais fichiers sont disponibles, ils peuvent être envoyés depuis l'interface ou placés dans `data/`.

## 7. Configuration locale

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

## 8. Dépendances ajoutées

`datatalk/requirements.txt` contient notamment :

- FastAPI / Uvicorn
- LangGraph / LangChain Core
- Pydantic
- pandas
- pymongo
- scikit-learn
- python-dotenv
- anthropic

## 9. État actuel / limites

### En place

- architecture de stockage des données ;
- contrat JSON des documentations ;
- catalogue JSON des correspondances ;
- RAG local fonctionnel au niveau du code ;
- outils SQL/Mongo/pandas ;
- adaptateur Claude ;
- upload depuis l'interface ;
- endpoints FastAPI ;
- point d'entrée `main.py` ;
- interface React minimale ;
- contrat d'intégration avec LangGraph documenté.

### À faire

1. Fournir/importer les **vraies documentations JSON** SQLite et MongoDB.
2. Fournir/importer le **vrai `mappings.json`**.
3. Fournir/importer la **vraie base SQLite** et les données MongoDB de démonstration.
4. Implémenter les 9 agents dans `agent.py` et la coordination dans `graph.py`.
5. Connecter les agents au RAG et aux outils existants.
6. Tester au minimum : SQL simple, Mongo simple et plusieurs cas hybrides.
7. Ajouter l'évaluation des résultats et de la pertinence du plan.
8. Tester le frontend avec le backend réel.

## 10. Important — niveau de validation

Les fichiers ont été écrits et poussés sur GitHub, mais **l'exécution complète de l'application n'a pas encore été effectuée dans cette session**. Il faut donc considérer les points ci-dessus comme une implémentation à intégrer/tester, et non comme une démonstration end-to-end déjà validée.

## 11. Historique de cette session

- reprise de la branche `feature/data-rag-hybrid-interface` ;
- vérification de l'état du backend ;
- correction de l'ordre des routes FastAPI afin que `/upload/mongodb-data` soit effectivement accessible ;
- ajout de validation des documentations JSON, mappings et bases SQLite uploadées ;
- ajout de `sqlite_schema()` pour fournir une inspection déterministe du schéma ;
- ajout de `backend/main.py` comme point d'entrée FastAPI ;
- ajout de ce document de passation.
