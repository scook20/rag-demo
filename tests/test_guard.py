from ragdemo.guard import specific_terms, unsupported_terms
from ragdemo.models import RetrievedChunk


def chunk(text):
    return RetrievedChunk(1, "s.csv", None, 0, text, {})


def test_specific_terms_pick_names_numbers_acronyms_and_months():
    assert specific_terms("What are the 2P reserves of Kestrel Field?") == ["2p", "kestrel", "field"]
    assert specific_terms("Which well had the ESP failure?") == ["esp"]
    assert specific_terms("What was the uptime of Kestrel-4 in June 2026?") == ["kestrel-4", "june", "2026"]
    assert specific_terms("what is the uptime") == []
    # "may" and "march" are verbs unless capitalised mid-sentence
    assert specific_terms("What may happen?") == []
    assert specific_terms("Output in March 2026") == ["march", "2026"]
    assert specific_terms("Kestrel-4's pump") == ["kestrel-4"]


def test_unsupported_terms_flags_the_missing_month_but_accepts_iso_dates():
    rows = [chunk("month: 2026-01; well: Kestrel-4; uptime_pct: 96.4"), chunk("month: 2026-03; well: Kestrel-4")]
    assert unsupported_terms("What was the uptime of Kestrel-4 in June?", rows) == ["june"]
    assert unsupported_terms("What was the uptime of Kestrel-4 in March?", rows) == []
    assert unsupported_terms("What was the uptime of Kestrel-4 in January 2026?", rows) == []
    assert unsupported_terms("What was the uptime of Kestrel-5 in January?", rows) == ["kestrel-5"]


def test_unsupported_terms_accepts_plurals_and_ignores_plain_words():
    c = [chunk("The recovery factor is 28 percent. Wells: Kestrel-2.")]
    assert unsupported_terms("What recovery factors does Kestrel-2 have?", c) == []
    assert unsupported_terms("Who is the CEO?", c) == ["ceo"]
