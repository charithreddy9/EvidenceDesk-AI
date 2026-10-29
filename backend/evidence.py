"""Conservative selection and lossless short answers for local semantic retrieval.

Similarity scores are cosine similarity, not likelihood of factual correctness.
These thresholds are provisional for BGE-small on illustrative policies; review
abstentions and false positives on new data before promotion.
"""
from __future__ import annotations

MIN_BGE_SIMILARITY = 0.55
MAX_BGE_GAP = 0.14


def select_evidence(passages: list[dict], retriever: str) -> list[dict]:
    """Preserve ranking but drop weak BGE matches and remote topic drift.

    The absolute floor rejects low semantic matches. The relative band excludes
    off-topic passages when a much stronger result exists. For other backends,
    leave existing ranking semantics alone.
    """
    if retriever != "fastembed" or not passages:
        return passages
    best = float(passages[0]["score"])
    floor = max(MIN_BGE_SIMILARITY, best - MAX_BGE_GAP)
    return [passage for passage in passages if float(passage["score"]) >= floor]


def cited_source_sentence(passage: dict, reference: str) -> str:
    """Return only verbatim evidence plus marker; never paraphrase or generate."""
    text = passage["text"].strip()
    return f"{text} {reference}" if text else ""
