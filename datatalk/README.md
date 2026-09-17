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
