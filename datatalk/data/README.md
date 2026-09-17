# DataTalk — données, RAG et correspondances

## Documentation SQLite / MongoDB

Les fichiers `sqlite/database_docs.json` et `mongodb/database_docs.json` sont la source du RAG. Ils doivent être remplis à partir du schéma réel des bases.

SQLite :
```json
{
  "database": "sqlite",
  "tables": [
    {
      "name": "customers",
      "description": "...",
      "columns": [
        {"name": "customer_id", "type": "INTEGER", "description": "..."}
      ]
    }
  ]
}
```

MongoDB :
```json
{
  "database": "mongodb",
  "collections": [
    {
      "name": "reviews",
      "description": "...",
      "fields": [
        {"name": "customer_id", "type": "string", "description": "..."}
      ]
    }
  ]
}
```

## Correspondances hybrides

`mappings.json` contient uniquement les relations explicites entre sources. Le Join Planner doit le consulter directement plutôt que de déduire une clé uniquement par similarité sémantique.

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
