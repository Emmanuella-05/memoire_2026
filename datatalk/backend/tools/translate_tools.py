"""Traduction français -> anglais de la question, pour les modèles ouverts de
génération de requêtes (SQLCoder, Qwen-Coder), meilleurs en anglais.

Principes :
  - uniquement pour la GÉNÉRATION de requêtes ; le RAG, la classification et la
    réponse finale gardent le texte d'origine (français) ;
  - les valeurs entre guillemets, les nombres et les dates ne doivent pas
    changer : si la traduction les altère, on revient à la question d'origine ;
  - tout échec de traduction est sans gravité (repli sur l'original), sauf
    l'indisponibilité du serveur LLM, qui remonte comme LLMError ;
  - désactivable : DATATALK_TRANSLATE=0 (pour comparer avec/sans traduction).
"""

from __future__ import annotations

import os
import re
from functools import lru_cache

from ..llm import open_chat, open_models_enabled

SYSTEM_PROMPT = (
    "You are a translation engine. Translate the user's French text into English. "
    "Rules: output ONLY the translation, no comments. Keep unchanged: text inside quotes, "
    "proper nouns, product/category/city names, numbers, dates, identifiers and column-like names. "
    "Do not answer the question."
)

_FRENCH_WORDS = re.compile(
    r"\b(le|la|les|des|du|de|un|une|quel|quels|quelle|quelles|combien|est|sont|dans|pour|avec|"
    r"par|sur|qui|que|où|quand|chaque|tous|toutes|plus|moins|liste|donne|montre|clients?|"
    r"commandes?|produits?|ventes?|avis|mois|année|montant|total)\b",
    re.IGNORECASE,
)
_QUOTED = re.compile(r"\"([^\"]+)\"|'([^']+)'|«\s*([^»]+?)\s*»|“([^”]+)”")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def translation_enabled() -> bool:
    """Utile seulement pour les modèles ouverts (Claude comprend le français) ; désactivable
    (DATATALK_TRANSLATE=0) pour comparer avec et sans traduction."""
    return open_models_enabled() and os.getenv("DATATALK_TRANSLATE", "1").strip().lower() not in ("0", "false", "off", "no")


def looks_french(text: str) -> bool:
    """Heuristique bon marché (pas d'appel LLM pour une question déjà anglaise)."""
    if re.search(r"[àâçéèêëîïôùûüœ]", text, re.IGNORECASE):
        return True
    return len(_FRENCH_WORDS.findall(text)) >= 2


def _protected_tokens(text: str) -> list[str]:
    tokens = [next(g for g in m.groups() if g) for m in _QUOTED.finditer(text)]
    tokens += _NUMBER.findall(text)
    return tokens


def _preserves_values(original: str, translated: str) -> bool:
    low = translated.lower()
    return all(token.lower() in low for token in _protected_tokens(original))


@lru_cache(maxsize=256)
def _translate_cached(text: str) -> str:
    raw = open_chat(text, system=SYSTEM_PROMPT, role="general", max_tokens=300)
    out = raw.strip().strip("`").strip()
    if out.lower().startswith("translation:"):
        out = out[len("translation:"):].strip()
    if not out or len(out) > 3 * len(text) + 40 or "\n\n" in out:
        return text
    if not _preserves_values(text, out):
        return text
    return out


def to_english(text: str) -> str:
    """Traduit `text` en anglais s'il semble français ; sinon le renvoie tel quel."""
    text = (text or "").strip()
    if not text or not translation_enabled() or not looks_french(text):
        return text
    return _translate_cached(text)


def translate_context_enabled() -> bool:
    """Option : traduire aussi les règles métier récupérées par le RAG
    (DATATALK_TRANSLATE_CONTEXT=1). Désactivée par défaut."""
    return translation_enabled() and os.getenv("DATATALK_TRANSLATE_CONTEXT", "0").strip().lower() in ("1", "true", "on", "yes")
