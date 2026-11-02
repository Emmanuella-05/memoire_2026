"""Vérification déterministe des champs d'un pipeline MongoDB.

Complète `mongo_tools.pipeline_rejection_reason` (collections connues, aucun
opérateur d'écriture / JavaScript) : ici on détecte les CHAMPS inventés.
MongoDB, lui, ne dit rien d'un champ inexistant : il renvoie simplement zéro
résultat, ce qui ferait croire à « aucune donnée » alors que la requête est fausse.

Le contrôle suit l'évolution des champs disponibles d'étape en étape
($group ne laisse que ses sorties, $lookup ajoute son `as`, etc.). Dès qu'une
étape rend la forme des documents imprévisible ($facet, $replaceRoot...), le
contrôle se désactive pour la suite plutôt que de risquer un faux rejet.
"""

from __future__ import annotations

from typing import Any

_PERMISSIVE_STAGES = {
    "$replaceRoot", "$replaceWith", "$facet", "$bucket", "$bucketAuto", "$graphLookup",
    "$unionWith", "$setWindowFields", "$redact", "$geoNear", "$densify", "$fill",
}


def _root(path: str) -> str:
    return path.split(".", 1)[0]


def _expr_refs(value: Any, out: list[str]) -> None:
    """Collecte les références « $champ » d'une expression d'agrégation."""
    if isinstance(value, str):
        if value.startswith("$") and not value.startswith("$$") and len(value) > 1:
            out.append(value[1:])
    elif isinstance(value, dict):
        for child in value.values():
            _expr_refs(child, out)
    elif isinstance(value, list):
        for child in value:
            _expr_refs(child, out)


def _match_refs(match: Any, out: list[str]) -> None:
    if not isinstance(match, dict):
        return
    for key, value in match.items():
        if key in ("$and", "$or", "$nor") and isinstance(value, list):
            for item in value:
                _match_refs(item, out)
        elif key == "$expr":
            _expr_refs(value, out)
        elif not key.startswith("$"):
            out.append(key)


def referenced_fields(pipeline: list[dict[str, Any]]) -> set[str]:
    """Noms de champs (racines) cités par un pipeline : sert à retrouver la
    collection quand le modèle n'en donne pas."""
    refs: list[str] = []
    for stage in pipeline:
        for op, body in stage.items():
            if op == "$match":
                _match_refs(body, refs)
            elif op == "$sort" and isinstance(body, dict):
                refs.extend(body.keys())
            elif op in ("$group", "$project", "$addFields", "$set", "$sortByCount", "$unwind"):
                _expr_refs(body, refs)
                if op == "$project" and isinstance(body, dict):
                    refs.extend(k for k, v in body.items() if v in (1, True))
            elif op == "$lookup" and isinstance(body, dict) and isinstance(body.get("localField"), str):
                refs.append(body["localField"])
    return {_root(r) for r in refs if r}


def infer_collection(pipeline: list[dict[str, Any]],
                     fields_by_collection: dict[str, list[str]]) -> str | None:
    """La seule collection qui possède TOUS les champs cités, sinon None."""
    refs = referenced_fields(pipeline) - {"_id"}
    if not refs:
        return None
    matches = [
        name for name, fields in fields_by_collection.items()
        if refs <= {_root(f) for f in fields}
    ]
    return matches[0] if len(matches) == 1 else None


def pipeline_field_issues(collection: str, pipeline: list[dict[str, Any]],
                          fields_by_collection: dict[str, list[str]]) -> str | None:
    """None si tous les champs cités existent, sinon une explication précise
    (réinjectée au modèle pour la correction)."""
    schema_paths = set(fields_by_collection.get(collection, []))
    if not schema_paths:
        return None  # collection non documentée : rien à vérifier
    original_roots = {_root(p) for p in schema_paths}
    avail: set[str] | None = original_roots | {"_id"}
    intact: set[str] = set(original_roots)  # racines dont la structure interne est celle du schéma

    def unknown(root: str) -> str:
        shown = ", ".join(sorted(avail or original_roots)[:25])
        return (f"Le champ '{root}' n'existe pas dans la collection '{collection}' à cette étape "
                f"(champs disponibles : {shown}). N'utilisez que des champs du schéma.")

    def check(path: str) -> str | None:
        if avail is None or not path:
            return None
        root = _root(path)
        if root == "_id" and root in avail:
            return None
        if root not in avail:
            return unknown(root)
        if "." in path and root in intact:
            children = {p for p in schema_paths if p.startswith(root + ".")}
            if children and not any(p == path or p.startswith(path + ".") or path.startswith(p + ".") for p in children):
                nested = ", ".join(sorted(children)[:15])
                return f"Le champ '{path}' n'existe pas (champs de '{root}' : {nested})."
        return None

    def check_refs(value: Any) -> str | None:
        refs: list[str] = []
        _expr_refs(value, refs)
        for ref in refs:
            problem = check(ref)
            if problem:
                return problem
        return None

    for stage in pipeline:
        op, body = next(iter(stage.items()))
        problem: str | None = None

        if op == "$match":
            refs: list[str] = []
            _match_refs(body, refs)
            problem = next((p for p in (check(r) for r in refs) if p), None)
        elif op == "$sort" and isinstance(body, dict):
            problem = next((p for p in (check(k) for k in body) if p), None)
        elif op == "$unwind":
            path = body if isinstance(body, str) else (body.get("path") if isinstance(body, dict) else None)
            if isinstance(path, str) and path.startswith("$"):
                problem = check(path[1:])
        elif op == "$group" and isinstance(body, dict):
            problem = check_refs(body)
            if not problem and avail is not None:
                avail = {"_id"} | {k for k in body if k != "_id"}
                intact = set()
        elif op == "$project" and isinstance(body, dict):
            included = {k for k, v in body.items() if v not in (0, False) and k != "_id"}
            excluded = {k for k, v in body.items() if v in (0, False)}
            problem = check_refs({k: v for k, v in body.items() if not isinstance(v, (int, bool))})
            if not problem:
                plain = [k for k, v in body.items() if v in (1, True)]
                problem = next((p for p in (check(k) for k in plain) if p), None)
            if not problem and avail is not None:
                if included:
                    keep_id = "_id" not in excluded
                    avail = {_root(k) for k in included} | ({"_id"} if keep_id else set())
                else:
                    avail = avail - {_root(k) for k in excluded}
                intact -= {_root(k) for k in body}
        elif op in ("$addFields", "$set") and isinstance(body, dict):
            problem = check_refs(body)
            if not problem and avail is not None:
                avail = avail | {_root(k) for k in body}
                intact -= {_root(k) for k in body}
        elif op == "$unset":
            names = [body] if isinstance(body, str) else body if isinstance(body, list) else []
            if avail is not None:
                avail = avail - {_root(n) for n in names if isinstance(n, str)}
        elif op == "$lookup" and isinstance(body, dict):
            local = body.get("localField")
            if isinstance(local, str):
                problem = check(local)
            foreign, foreign_field = body.get("from"), body.get("foreignField")
            if not problem and isinstance(foreign, str) and isinstance(foreign_field, str) and fields_by_collection.get(foreign):
                if _root(foreign_field) not in {_root(f) for f in fields_by_collection[foreign]} | {"_id"}:
                    problem = f"Le champ '{foreign_field}' n'existe pas dans la collection '{foreign}'."
            if not problem and avail is not None and isinstance(body.get("as"), str):
                avail = avail | {_root(body["as"])}
                intact.discard(_root(body["as"]))
        elif op == "$count" and isinstance(body, str):
            if avail is not None:
                avail = {body}
                intact = set()
        elif op == "$sortByCount":
            problem = check_refs(body)
            if not problem and avail is not None:
                avail = {"_id", "count"}
                intact = set()
        elif op in _PERMISSIVE_STAGES:
            avail = None
        elif op not in ("$limit", "$skip", "$sample"):
            avail = None  # étape inconnue : on ne devine pas

        if problem:
            return problem
    return None
