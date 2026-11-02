"""Convertisseur de la sortie d'un petit modèle vers un pipeline MongoDB exécutable.

Les modèles de 7B milliards de paramètres produisent rarement le format exact
attendu. On accepte donc, et on convertit en `(collection, pipeline)` :

  - la forme canonique  {"collection": "...", "pipeline": [...]}
  - une commande shell  db.clients.find({...}).sort({...}).limit(5)
                        db.clients.aggregate([...])   countDocuments / distinct
  - un `find(...)` sans le préfixe db.<collection>.
  - une liste d'étapes  [{"$match": ...}, {"$group": ...}]
  - UNE étape seule     {"$match": {...}}
  - un simple filtre    {"age": {"$gt": 30}}   (il manque le find / le $match)
  - la syntaxe JavaScript : clés sans guillemets, guillemets simples, virgules
    finales, commentaires, ISODate("..."), ObjectId("..."), NumberInt(5), /regex/i
  - une étape qui contient plusieurs opérateurs (coupée en étapes séparées)

La conversion n'exécute rien et n'interprète aucun code : c'est un analyseur
syntaxique, pas un eval. Elle ne décide pas de ce qui est autorisé : les
garde-fous (mongo_tools.pipeline_rejection_reason) s'appliquent ensuite.
"""

from __future__ import annotations

import re
from typing import Any

from .mongo_fields import infer_collection

PIPELINE_STAGES = {
    "$match", "$group", "$project", "$sort", "$limit", "$skip", "$unwind", "$lookup", "$count",
    "$addFields", "$set", "$unset", "$replaceRoot", "$replaceWith", "$facet", "$bucket",
    "$bucketAuto", "$sample", "$sortByCount", "$unionWith", "$graphLookup", "$redact",
    "$geoNear", "$densify", "$fill", "$setWindowFields", "$out", "$merge", "$collStats",
    "$indexStats", "$currentOp", "$listSessions", "$planCacheStats",
}

_VALUE_WRAPPERS = {
    "ISODate", "Date", "ObjectId", "NumberInt", "NumberLong", "NumberDecimal", "Int32",
    "Long", "Double", "Decimal128", "UUID", "Timestamp", "String", "Number",
}


class MongoParseError(ValueError):
    """La sortie du modèle n'a pas pu être convertie en pipeline."""


# --- analyseur de littéraux JavaScript / JSON tolérant ---------------------

class _Parser:
    def __init__(self, text: str, pos: int = 0, extended: bool = False) -> None:
        self.s = text
        self.i = pos
        # extended=True (vrai serveur MongoDB) : ISODate("...") -> {"$date": "..."} et ObjectId("...") ->
        # {"$oid": "..."}, convertis ensuite en vrais types BSON ; sinon on garde le texte (JSON importé).
        self.extended = extended

    def error(self, message: str) -> MongoParseError:
        return MongoParseError(f"{message} (position {self.i}, près de {self.s[self.i:self.i + 25]!r})")

    def skip(self) -> None:
        s = self.s
        while self.i < len(s):
            if s[self.i].isspace():
                self.i += 1
            elif s.startswith("//", self.i):
                end = s.find("\n", self.i)
                self.i = len(s) if end == -1 else end + 1
            elif s.startswith("/*", self.i):
                end = s.find("*/", self.i + 2)
                self.i = len(s) if end == -1 else end + 2
            else:
                break

    def peek(self) -> str:
        self.skip()
        return self.s[self.i] if self.i < len(self.s) else ""

    def expect(self, char: str) -> None:
        if self.peek() != char:
            raise self.error(f"'{char}' attendu")
        self.i += 1

    def value(self) -> Any:
        c = self.peek()
        if c == "{":
            return self.obj()
        if c == "[":
            return self.arr()
        if c in "\"'`":
            return self.string()
        if c == "/":
            return self.regex()
        if c and (c.isdigit() or c in "+-."):
            return self.number()
        if c and (c.isalpha() or c in "_$"):
            return self.word()
        raise self.error("valeur attendue")

    def obj(self) -> dict[str, Any]:
        self.expect("{")
        result: dict[str, Any] = {}
        while True:
            c = self.peek()
            if c == "}":
                self.i += 1
                return result
            if c == ",":
                self.i += 1
                continue
            if not c:
                raise self.error("objet non fermé")
            key = self.key()
            self.expect(":")
            result[key] = self.value()

    def key(self) -> str:
        c = self.peek()
        if c in "\"'`":
            return self.string()
        m = re.compile(r"[A-Za-z_$0-9][\w$.\-]*").match(self.s, self.i)
        if not m:
            raise self.error("nom de champ attendu")
        self.i = m.end()
        return m.group(0)

    def arr(self) -> list[Any]:
        self.expect("[")
        result: list[Any] = []
        while True:
            c = self.peek()
            if c == "]":
                self.i += 1
                return result
            if c == ",":
                self.i += 1
                continue
            if not c:
                raise self.error("liste non fermée")
            result.append(self.value())

    def string(self) -> str:
        quote = self.s[self.i]
        self.i += 1
        out: list[str] = []
        escapes = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "0": "\0"}
        while self.i < len(self.s):
            c = self.s[self.i]
            if c == "\\" and self.i + 1 < len(self.s):
                nxt = self.s[self.i + 1]
                if nxt == "u" and re.fullmatch(r"[0-9a-fA-F]{4}", self.s[self.i + 2:self.i + 6]):
                    out.append(chr(int(self.s[self.i + 2:self.i + 6], 16)))
                    self.i += 6
                    continue
                out.append(escapes.get(nxt, nxt))
                self.i += 2
                continue
            if c == quote:
                self.i += 1
                return "".join(out)
            out.append(c)
            self.i += 1
        raise self.error("chaîne non fermée")

    def regex(self) -> dict[str, str]:
        start = self.i
        self.i += 1
        body: list[str] = []
        while self.i < len(self.s):
            c = self.s[self.i]
            if c == "\\" and self.i + 1 < len(self.s):
                body.append(self.s[self.i:self.i + 2])
                self.i += 2
                continue
            if c == "/":
                self.i += 1
                flags = re.compile(r"[a-z]*").match(self.s, self.i).group(0)  # type: ignore[union-attr]
                self.i += len(flags)
                result = {"$regex": "".join(body)}
                if flags:
                    result["$options"] = flags
                return result
            body.append(c)
            self.i += 1
        self.i = start
        raise self.error("expression régulière non fermée")

    def number(self) -> int | float:
        m = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?").match(self.s, self.i)
        if not m:
            raise self.error("nombre invalide")
        self.i = m.end()
        text = m.group(0)
        return float(text) if re.search(r"[.eE]", text) else int(text)

    def word(self) -> Any:
        m = re.compile(r"[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*").match(self.s, self.i)
        assert m
        name = m.group(0)
        self.i = m.end()
        if name == "new":
            return self.word()
        if self.peek() == "(":
            args = self.args()
            if name in _VALUE_WRAPPERS:
                arg = args[0] if args else None
                if self.extended and isinstance(arg, str):
                    if name in ("ISODate", "Date"):
                        return {"$date": arg}
                    if name == "ObjectId":
                        return {"$oid": arg}
                return arg
            raise self.error(f"fonction non supportée : {name}()")
        lowered = {"true": True, "false": False, "null": None, "undefined": None}
        if name in lowered:
            return lowered[name]
        if name in ("NaN", "Infinity"):
            raise self.error(f"valeur non supportée : {name}")
        raise self.error(f"mot inattendu sans guillemets : {name}")

    def args(self) -> list[Any]:
        self.expect("(")
        result: list[Any] = []
        while True:
            c = self.peek()
            if c == ")":
                self.i += 1
                return result
            if c == ",":
                self.i += 1
                continue
            if not c:
                raise self.error("parenthèse non fermée")
            result.append(self.value())


# --- extraction du texte ---------------------------------------------------

_FENCE = re.compile(r"```[ \t]*[\w+-]*[ \t]*\n?(.*?)```", re.DOTALL)
_SHELL = re.compile(
    r"(?:\bdb\s*(?:\.\s*getCollection\(\s*(?P<q1>['\"])(?P<gc>[^'\"]+)(?P=q1)\s*\)"
    r"|\[\s*(?P<q2>['\"])(?P<br>[^'\"]+)(?P=q2)\s*\]"
    r"|\.\s*(?P<c>[A-Za-z_][\w\-]*))\s*\.\s*)?"
    r"\b(?P<m>findOne|find|aggregate|countDocuments|estimatedDocumentCount|count|distinct)\s*\("
)


def _strip_fences(raw: str) -> str:
    blocks = _FENCE.findall(raw)
    text = blocks[0] if blocks else raw
    return text.strip().rstrip(";").strip()


def _sort_spec(spec: Any) -> dict[str, int]:
    if not isinstance(spec, dict) or not spec:
        raise MongoParseError("sort() attend un objet {champ: 1 ou -1}.")
    words = {"asc": 1, "ascending": 1, "desc": -1, "descending": -1}
    out: dict[str, int] = {}
    for field, direction in spec.items():
        if isinstance(direction, str):
            direction = words.get(direction.lower(), direction)
        if isinstance(direction, float) and direction in (1.0, -1.0):
            direction = int(direction)
        if direction not in (1, -1):
            raise MongoParseError(f"Sens de tri invalide pour '{field}' : {direction!r} (1 ou -1 attendu).")
        out[field] = direction
    return out


def _int_arg(value: Any, method: str) -> int:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise MongoParseError(f"{method}() attend un entier positif, reçu {value!r}.")
    return value


def _find_to_pipeline(method: str, args: list[Any], chain: list[tuple[str, list[Any]]]) -> list[dict[str, Any]]:
    flt = args[0] if args else {}
    if flt is None:
        flt = {}
    if not isinstance(flt, dict):
        raise MongoParseError(f"{method}() attend un filtre (objet), reçu {type(flt).__name__}.")
    projection = args[1] if len(args) > 1 and isinstance(args[1], dict) else None
    sort = skip = limit = None
    count = False
    for name, cargs in chain:
        if name == "sort" and cargs:
            sort = _sort_spec(cargs[0])
        elif name == "limit" and cargs:
            limit = _int_arg(cargs[0], "limit")
        elif name == "skip" and cargs:
            skip = _int_arg(cargs[0], "skip")
        elif name in ("project", "projection") and cargs and isinstance(cargs[0], dict):
            projection = cargs[0]
        elif name in ("count", "itcount", "countDocuments"):
            count = True
    if method == "findOne":
        limit = 1
    pipeline: list[dict[str, Any]] = [{"$match": flt}]
    if sort:
        pipeline.append({"$sort": sort})
    if skip:
        pipeline.append({"$skip": skip})
    if limit:
        pipeline.append({"$limit": limit})
    if projection:
        pipeline.append({"$project": projection})
    if count:
        pipeline.append({"$count": "count"})
    return pipeline


def _split_stages(stages: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for stage in stages:
        if not isinstance(stage, dict):
            raise MongoParseError(f"Étape de pipeline invalide : {stage!r}.")
        for op, body in stage.items():
            if op == "$sort":
                body = _sort_spec(body)
            elif op in ("$limit", "$skip"):
                body = _int_arg(body, op)
            out.append({op: body})
    return out


def _value_to_pipeline(value: Any) -> tuple[str | None, list[dict[str, Any]]]:
    if isinstance(value, list):
        return None, _split_stages(value)
    if not isinstance(value, dict):
        raise MongoParseError("Une requête MongoDB (objet ou liste d'étapes) est attendue.")
    collection = value.get("collection") if isinstance(value.get("collection"), str) else None
    if isinstance(value.get("pipeline"), list):
        return collection, _split_stages(value["pipeline"])
    for key in ("filter", "query", "find"):
        if isinstance(value.get(key), dict):
            sort = value.get("sort")
            chain = []
            if isinstance(sort, dict):
                chain.append(("sort", [sort]))
            if isinstance(value.get("limit"), int):
                chain.append(("limit", [value["limit"]]))
            return collection, _find_to_pipeline("find", [value[key], value.get("projection")], chain)
    if any(k in PIPELINE_STAGES for k in value):
        return collection, _split_stages([value])
    # Aucun opérateur d'étape : c'est un filtre auquel il manque le find / $match.
    return collection, [{"$match": value}]


def _shell_to_pipeline(text: str, extended: bool = False) -> tuple[str | None, list[dict[str, Any]]] | None:
    m = _SHELL.search(text)
    if not m:
        return None
    collection = m.group("gc") or m.group("br") or m.group("c")
    method = m.group("m")
    parser = _Parser(text, m.end() - 1, extended)
    args = parser.args()
    chain: list[tuple[str, list[Any]]] = []
    while parser.peek() == ".":
        parser.i += 1
        name_match = re.compile(r"\s*(\w+)\s*").match(text, parser.i)
        if not name_match:
            break
        parser.i = name_match.end()
        if parser.peek() != "(":
            break
        chain.append((name_match.group(1), parser.args()))

    if method == "aggregate":
        if args and isinstance(args[0], list):
            return collection, _split_stages(args[0])
        if args and all(isinstance(a, dict) for a in args):
            return collection, _split_stages(args)
        raise MongoParseError("aggregate() attend une liste d'étapes.")
    if method in ("countDocuments", "count", "estimatedDocumentCount"):
        flt = args[0] if args and isinstance(args[0], dict) else {}
        return collection, [{"$match": flt}, {"$count": "count"}]
    if method == "distinct":
        if not args or not isinstance(args[0], str):
            raise MongoParseError("distinct() attend le nom d'un champ.")
        pipeline: list[dict[str, Any]] = []
        if len(args) > 1 and isinstance(args[1], dict) and args[1]:
            pipeline.append({"$match": args[1]})
        pipeline.append({"$group": {"_id": f"${args[0]}"}})
        return collection, pipeline
    return collection, _find_to_pipeline(method, args, chain)


# --- résolution de la collection ------------------------------------------

def _squash(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def resolve_collection(name: str | None, known: list[str] | None, default: str | None = None) -> str | None:
    """Fait correspondre le nom donné par le modèle à une collection connue
    (casse, singulier/pluriel, tirets). Un nom réellement inconnu est renvoyé
    tel quel : la validation le rejettera avec la liste des collections."""
    known = known or []
    if not name:
        if default:
            return default
        return known[0] if len(known) == 1 else None
    if name in known:
        return name
    wanted = _squash(name)
    variants = {wanted, wanted.rstrip("s"), wanted + "s"}
    for candidate in known:
        if _squash(candidate) in variants or _squash(candidate).rstrip("s") in variants:
            return candidate
    return name


# --- point d'entrée --------------------------------------------------------

def normalize_mongo_output(raw: str, known_collections: list[str] | None = None,
                           default_collection: str | None = None,
                           fields_by_collection: dict[str, list[str]] | None = None,
                           extended_json: bool = False,
                           ) -> tuple[str, list[dict[str, Any]]]:
    """Convertit la réponse brute du modèle en (collection, pipeline).
    Lève MongoParseError (ValueError) si rien d'exploitable.

    Quand le modèle ne nomme aucune collection : `default_collection` (ex. celle
    du plan de jointure), sinon la seule collection possédant tous les champs
    cités (`fields_by_collection`), sinon la seule collection existante."""
    try:
        return _normalize(raw, known_collections, default_collection, fields_by_collection, extended_json)
    except (IndexError, RecursionError) as exc:  # entrée tronquée ou trop imbriquée
        raise MongoParseError(f"Réponse du modèle illisible ou tronquée : {raw[:150]!r}") from exc


def _normalize(raw: str, known_collections: list[str] | None, default_collection: str | None,
               fields_by_collection: dict[str, list[str]] | None,
               extended: bool = False) -> tuple[str, list[dict[str, Any]]]:
    text = _strip_fences(raw or "")
    if not text:
        raise MongoParseError("Réponse vide du modèle.")

    result: tuple[str | None, list[dict[str, Any]]] | None = None
    if text[0] not in "{[":
        result = _shell_to_pipeline(text, extended)
    if result is None:
        start = min((i for i in (text.find("{"), text.find("[")) if i != -1), default=-1)
        if start == -1:
            raise MongoParseError(f"Aucune requête MongoDB trouvée dans la réponse : {raw[:150]!r}")
        result = _value_to_pipeline(_Parser(text, start, extended).value())

    collection, pipeline = result
    if not pipeline:
        raise MongoParseError("Pipeline vide.")
    for stage in pipeline:
        if len(stage) != 1 or not next(iter(stage)).startswith("$"):
            raise MongoParseError(f"Étape de pipeline invalide : {stage!r}.")
    resolved = resolve_collection(collection, known_collections, default_collection)
    if not resolved and fields_by_collection:
        resolved = infer_collection(pipeline, fields_by_collection)
    if not resolved:
        raise MongoParseError(
            "Collection non identifiée : écrire db.<collection>.find(...) ou aggregate(...)."
            + (f" Collections disponibles : {', '.join(known_collections)}." if known_collections else "")
        )
    return resolved, pipeline
