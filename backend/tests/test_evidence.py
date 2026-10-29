import numpy as np
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.evidence import select_evidence, cited_source_sentence, support_score
from backend.retrieval import sentence_chunks, DEMO_DOCUMENTS
from backend.evaluate import measure, PARAPHRASE_CASES


def test_sentence_chunking_preserves_factual_sentence():
    text = DEMO_DOCUMENTS[0]["text"]
    found = sentence_chunks(text)
    assert "The meal allowance is $65 per day." in found
    assert "Approved business travel expenses are reimbursable." in found
    assert all(sentence in text for sentence in found)


def test_filters_irrelevant_semantic_matches():
    found = [
        {"doc_id": "travel", "score": 0.773},
        {"doc_id": "incident", "score": 0.485},
        {"doc_id": "security", "score": 0.484},
    ]
    assert [x["doc_id"] for x in select_evidence(found, "fastembed")] == ["travel"]
    assert select_evidence([{"score": 0.49}], "fastembed") == []
    assert len(select_evidence([{"score": .77}, {"score": .69}], "fastembed")) == 2
    assert select_evidence(found, "tfidf") == found


def test_short_answer_is_strictly_supported_by_source():
    passage = {"text": "The meal allowance is $65 per day."}
    assert cited_source_sentence(passage, "[1]") == "The meal allowance is $65 per day. [1]"


class PredictableEncoder:
    """No model download; ranks one source sentence for an answer."""
    def encode(self, texts):
        return np.asarray([[1., 0.] if "meal allowance" in text.lower()
                           else [0., 1.] for text in texts], dtype=np.float32)

    def encode_query(self, texts):
        return np.asarray([[0., 0.] if "unanswerable" in text.lower()
                           else [1., 0.] for text in texts], dtype=np.float32)


def test_fastembed_api_filters_and_extracts(tmp_path, monkeypatch):
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    monkeypatch.setenv("ENABLE_PAID_LLM", "false")
    app = create_app(db_path=str(tmp_path / "semantic.sqlite3"),
                     provider="fastembed", encoder=PredictableEncoder())
    client = TestClient(app)
    body = client.post("/api/ask",
                       json={"question": "How much food spending is allowed?"}).json()
    assert body["mode"] == "sentence_extract"
    assert body["abstained"] is False
    assert body["answer"] == "The meal allowance is $65 per day. [1]"
    assert len(body["citations"]) == 1
    assert body["citations"][0]["text"] == "The meal allowance is $65 per day."
    assert body["citations"][0]["doc_id"] == "policy-travel"
    empty = client.post("/api/ask", json={"question": "Unanswerable question please"}).json()
    assert empty["abstained"] and not empty["citations"]
    assert client.post("/api/ask", json={"question": "How much food spending is allowed?",
                                        "generate": True}).status_code == 403
    assert client.post("/api/documents", json={
        "title": "Cannot write", "content": "a" * 120}).status_code == 403


def test_evaluation_counts_and_mrr():
    class FakeIndex:
        provider = "fastembed"
        def search(self, query, top_k):
            return ([{"doc_id": "policy-travel", "score": 0.82},
                     {"doc_id": "policy-security", "score": 0.40}]
                    if "travel" in query.lower() else [])
    scores = measure(FakeIndex(), positives=[
        ("What about travel?", "policy-travel"),
        ("What about travel?", "policy-security"),
    ], negatives=["How many bridges are in London?"])
    assert scores["hit_at_1"] == 0.5
    assert scores["recall_at_3"] == 0.5
    assert scores["mrr"] == 0.5
    assert scores["abstention_accuracy"] == 1.0
    assert len(PARAPHRASE_CASES) >= 5

def test_verifier_rejects_unrelated_benefits_without_keyword_hacks():
    passages=[
        {"doc_id":"d1","title":"Travel Policy","text":"Travel expenses are reimbursable.","score":.78,"chunk":0},
        {"doc_id":"d2","title":"Travel Policy","text":"The meal allowance is $65 per day.","score":.75,"chunk":1},
    ]
    assert select_evidence(passages,"fastembed","Does the company reimburse gym memberships?")==[]
    assert select_evidence(passages,"fastembed","What is the meal allowance?")[0]["text"].startswith("The meal")
    assert not select_evidence(passages,"fastembed","How many vacation days are paid?")
    assert support_score("Where must production credentials be stored?",{"text":"Production credentials must be stored in a secrets manager.","title":"Security Handbook"})[0] > .67
