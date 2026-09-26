# DataTalk — données, RAG, Claude et interface

## Configuration

1. Copier `.env.example` vers `.env` et renseigner `CLAUDE_API_KEY`.
2. Installer `requirements.txt`.
3. Pour MongoDB, lancer MongoDB et renseigner `MONGO_URI` / `MONGO_DB`.

## Claude

Les agents peuvent utiliser `backend.tools.claude_generate()`. Le modèle par défaut est `claude-sonnet-5`; il peut être changé avec `CLAUDE_MODEL` sans modifier le code.

## Uploads depuis React

L'interface propose :
- une base SQLite `.db/.sqlite/.sqlite3` ;
- un export JSON MongoDB de la forme `{ "collection": [{...}, {...}] }` ;
- la documentation SQLite JSON ;
- la documentation MongoDB JSON ;
- `mappings.json` pour les correspondances SQL ↔ MongoDB.

La documentation et les mappings sont rechargés/indexés après upload. Les correspondances restent explicites et ne sont pas déduites uniquement par le RAG.

## Interface FastAPI

`backend/interface.py` contient l'API `/query`, `/upload/...`, `/upload/mongodb-data`, `/health` et `/claude/test`. Le `main.py` de l'équipe peut simplement exposer cette `app` ou reprendre ses routes lors de l'intégration finale.

## Important

Ne jamais committer `.env` ni la clé Claude. Le `.gitignore` les exclut.

## Description du travail effectué

Ce dépôt contient une preuve de concept d'un pipeline RAG (Retrieval-Augmented Generation) et une interface minimale. Travaux réalisés :

- **Ingestion et catalogage :** outils pour charger et indexer des sources (SQLite, export MongoDB, JSON de documentation) et utiliser `mappings.json` pour les correspondances entre schémas.
- **Génération d'embeddings & retrieval :** pipeline d'`embeddings` et `retriever` dans `backend/rag/` pour vectoriser les documents et interroger l'index.
- **Composants RAG :** composants backend qui orchestrent la récupération d'informations et la génération assistée par LLM (`llm.py`, `rag/retriever.py`).
- **Agents spécialisés :** agents modulaires pour gérer différentes sources et tâches : `sql_agent.py`, `mongo_agent.py`, `classifier.py`, `join_planner.py`, `result_merger.py` — chacun encapsule la logique d'interrogation, planification et fusion des résultats.
- **Outils utilitaires :** fonctions d'assistance pour manipuler le catalogue, interagir avec MongoDB/SQL et faciliter les workflows RAG (`tools/`).
- **Interface front-end minimale :** une UI React/HTML dans `frontend/` pour téléverser des sources, lancer l'indexation et envoyer des requêtes vers l'API FastAPI backend.
- **Exemples et configuration :** fichiers d'exemple et instructions pour configurer les clés (Claude), MongoDB et l'environnement Python.

Le travail se concentre sur la démonstration d'un flux de bout en bout : ingestion → indexation → récupération → fusion + génération. Si vous le souhaitez, je peux :

- détailler chaque module (`backend/agents`, `backend/rag`) dans le README;
- ajouter un schéma d'architecture et des exemples d'utilisation;
- fournir des commandes pour lancer et tester localement.

### Détails des modules

- **`backend/` :** point d'entrée et orchestration backend. Contient la logique FastAPI, l'initialisation des agents et la configuration des services.

- **`backend/rag/embeddings.py` :** génération d'embeddings pour documents et enregistrements. Normalise les textes, gère les batchs et produit des vecteurs utilisés par l'index.

- **`backend/rag/retriever.py` :** logique de recherche vectorielle et ranking des passages. Encapsule les requêtes vers l'index vectoriel et la conversion score → passages.

- **`backend/llm.py` :** wrapper pour les appels LLM (Claude par défaut). Centralise les templates de prompt, la gestion des tokens et des paramètres de génération.

- **`backend/graph.py` :** utilitaires pour représentation/visualisation des relations entre entités ou pipelines (optionnel selon usage).

- **`backend/main.py` / `backend/interface.py` :** expose l'API HTTP (`/query`, `/upload/*`, `/health`, `/claude/test`) et orchestre le pipeline RAG et les agents.

- **`backend/agents/` :** agents spécialisés :
	- `sql_agent.py` : extraction et transformation depuis bases SQLite/SQL.
	- `mongo_agent.py` : ingestion et interrogation d'exports MongoDB.
	- `classifier.py` : routage des requêtes/document classification.
	- `join_planner.py` : planification et exécution de jointures multi-sources.
	- `result_merger.py` : fusion et post-traitement des résultats multi-agents.

- **`backend/tools/` :** helpers pour ingestion, mapping, interaction avec Mongo/SQL et utilitaires RAG (`catalog_tools.py`, `mongo_tools.py`, `sql_tools.py`, `rag_tools.py`).

- **`catalog/` & `ingestion/` :** scripts et artefacts pour construire le catalogue de sources et pipelines d'ingestion.

- **`data/` :** exemples d'exports (`sqlite/database_docs.json`, `mongodb/database_docs.json`) et `mappings.json` pour correspondances entre schémas.

- **`frontend/` :** UI minimale (React + HTML) pour téléverser des fichiers, déclencher l'indexation et consulter l'API.

### Schéma d'architecture (Mermaid)

```mermaid
flowchart TD
	User[Utilisateur / Navigateur] -->|Téléversement / Requête| Frontend[Frontend UI]
	Frontend -->|HTTP| API[FastAPI Backend]
	API -->|orchestrer| Agents[Agents (SQL / Mongo / Classifier / Planner)]
	Agents -->|ingestion/requête| Ingest[Ingestion & Catalogue]
	Ingest -->|indexer| Emb[Embeddings & Index]
	API -->|query| Retriever[Retriever]
	Retriever -->|documents| Merger[Result Merger]
	Merger -->|prompt| LLM[LLM (Claude)]
	LLM -->|réponse| API
	Data[(SQLite / MongoDB / JSON / mappings.json)] -->|sources| Ingest
	style User fill:#f9f,stroke:#333,stroke-width:1px
```

Souhaitez-vous que j'exporte ce diagramme en PNG/SVG et que j'ajoute une section `Exemples d'utilisation` avec commandes de démarrage ?
