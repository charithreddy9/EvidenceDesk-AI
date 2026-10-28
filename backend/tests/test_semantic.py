"""Tests true vector similarity plumbing without downloading a model in unit CI."""
import numpy as np
from backend.retrieval import EvidenceIndex

DOCS = [
    {"id": "weather", "title": "Forecast", "text": "Heavy precipitation is likely this afternoon."},
    {"id": "cooking", "title": "Recipe", "text": "Add chopped onions to the pan and cook until tender."},
]
class MockEmbeddings:
    def encode(self, texts):
        return np.array([[1.0, 0.0] if "precipitation" in s.lower() else [0.0, 1.0] for s in texts], dtype=np.float32)
    def encode_query(self, texts):
        return np.array([[1.0, 0.0] if "rain" in s.lower() else [0.0, 1.0] for s in texts], dtype=np.float32)

def test_semantic_query_vectors_and_nearest_neighbor():
    index=EvidenceIndex(DOCS, provider="fastembed", encoder=MockEmbeddings())
    results=index.search("Will it rain this afternoon?", top_k=2)
    assert results[0]["doc_id"] == "weather"
    assert results[0]["score"] == 1.0

def test_real_model_optional():
    """Real BGE model is opt-in because downloads/ML memory exceed lightweight CI."""
    import os, pytest
    if os.getenv("RUN_REAL_EMBEDDING_TEST") != "true":
        pytest.skip("Enable RUN_REAL_EMBEDDING_TEST=true with fastembed installed")
    from backend.retrieval import FastEmbedEncoder
    enc=FastEmbedEncoder()
    vectors=enc.encode(["The daily meal allowance is sixty-five dollars.", "Keep your secrets safe."])
    assert vectors.shape[0] == 2 and vectors.shape[1] > 100
    assert np.isfinite(vectors).all()

def test_paid_embedding_requires_explicit_enable(monkeypatch):
    import pytest
    from backend.retrieval import OpenAIEncoder
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-not-used")
    monkeypatch.delenv("ENABLE_PAID_EMBEDDINGS", raising=False)
    with pytest.raises(ValueError, match="Paid embeddings disabled"):
        OpenAIEncoder()
