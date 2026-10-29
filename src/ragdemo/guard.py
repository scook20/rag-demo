"""Second fail-closed guard: specific terms in the question that the retrieved excerpts never mention.

Embedding similarity is a topical signal. A question such as "uptime of Kestrel-4 in June" is on topic
for a corpus that only has January to March, so the best cosine similarity stays high. This guard
looks for the specific things the question asks about (numbers, capitalised names, acronyms, months)
and refuses when any of them is absent from every excerpt the model would see.

It is a heuristic with known limits: it checks surface forms, so a question that rephrases a number
("ten ppm" for "10 ppm") is refused, and lowercase names are not recognised as names. Refusing
wrongly is the cheaper error for a demo; the eval suite measures the false-refusal rate.
"""

from __future__ import annotations

import re
from typing import Sequence

from ragdemo.models import RetrievedChunk

_MONTHS = {
    name: number
    for number, name in enumerate(
        "january february march april may june july august september october november december".split(), start=1
    )
}
_AMBIGUOUS_MONTHS = frozenset({"may", "march"})
_STOPWORDS = frozenset(
    "a an and the of in on at to for by with from about into than then there these those their they it its "
    "this that i we you your our my me us as or not no yes".split()
)
_EDGE = re.compile(r"^[^\w]+|[^\w]+$")


def _clean(word: str) -> str:
    word = word.replace("’", "'")
    if word.endswith("'s"):
        word = word[:-2]
    return _EDGE.sub("", word)


def specific_terms(question: str) -> list[str]:
    """Terms a supporting excerpt must contain: tokens with digits, acronyms, capitalised words that are
    not sentence-initial question words, and month names. Order preserved, duplicates dropped."""
    terms: list[str] = []
    for position, raw in enumerate(question.split()):
        word = _clean(raw)
        if not word:
            continue
        lower = word.lower()
        if lower in _STOPWORDS:
            continue
        if lower in _MONTHS:
            # "may" and "march" are also ordinary verbs: only count them when capitalised mid-sentence.
            specific = lower not in _AMBIGUOUS_MONTHS or (word[0].isupper() and position > 0)
        elif any(ch.isdigit() for ch in word):
            specific = True
        elif len(word) >= 2 and word.isupper():
            specific = True  # acronym such as ESP or CEO
        elif word[0].isupper():
            specific = position > 0  # the first word is capitalised anyway: only digits/acronyms count there
        else:
            specific = False
        if specific and lower not in terms:
            terms.append(lower)
    return terms


def _supported(term: str, haystack: str) -> bool:
    if term in _MONTHS:
        number = _MONTHS[term]
        # A month is supported by its name or by an ISO date such as 2026-06.
        return term in haystack or re.search(rf"\d-0?{number}(?!\d)", haystack) is not None
    if term in haystack:
        return True
    return term.endswith("s") and len(term) > 3 and term[:-1] in haystack  # crude plural


def unsupported_terms(question: str, chunks: Sequence[RetrievedChunk]) -> list[str]:
    """Specific question terms that appear in none of the chunks (lowercased)."""
    haystack = " ".join(c.content for c in chunks).lower()
    return [t for t in specific_terms(question) if not _supported(t, haystack)]
