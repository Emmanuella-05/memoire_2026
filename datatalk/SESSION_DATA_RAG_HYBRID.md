# DataTalk — Suivi de travail / passation

## Réparations d'intégration

### Réparation 1 — backend/graph.py

La fausse implémentation SimpleGraph a été remplacée par un vrai StateGraph LangGraph.

Flux actuellement branché :

START → classifier
- SQL → sql_agent → result_merger → END
- Mongo → mongo_agent → result_merger → END
- Hybrid → hybrid_sql → hybrid_mongo → join_planner → result_merger → END

Le graphe expose build_graph(), graph et run(question).

Le chemin hybride reste volontairement séquentiel pour cette première réparation.

Point restant : les agents SQL/Mongo encapsulent encore génération, validation, exécution et correction. La décomposition stricte en composants séparés du cahier des charges sera traitée ensuite.

Validation : code poussé sur main ; exécution locale end-to-end encore à faire.
