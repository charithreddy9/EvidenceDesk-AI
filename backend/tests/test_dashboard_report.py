"""Pure unit tests exercise metrics, labels, false positives and JSON output."""
import json
from pathlib import Path

from backend.evaluation_report import (
    LABELED_CASES, build_report, percentile, summarize, write_report,
)
from backend.retrieval import DEMO_DOCUMENTS


class FakeIndex:
    provider = "fastembed"
    documents = DEMO_DOCUMENTS

    def search(self, question, top_k=3):
        if "correct" in question:
            return [{"doc_id": "policy-travel", "title": "Travel",
                     "text": "The meal allowance is $65 per day.", "score": 0.81}]
        if "unknown" in question:
            return [{"doc_id": "policy-security", "title": "Security",
                     "text": "Multi-factor authentication is mandatory.", "score": 0.80}]
        return []


def test_labeled_corpus_is_diverse_and_has_valid_labels():
    assert len(LABELED_CASES) >= 25
    assert {x["category"] for x in LABELED_CASES} == {
        "direct", "paraphrase", "unanswerable"
    }
    assert len({x["id"] for x in LABELED_CASES}) == len(LABELED_CASES)
    allowed = {doc["id"] for doc in DEMO_DOCUMENTS}
    for case in LABELED_CASES:
        assert case["expected_document"] in allowed or case["expected_document"] is None
        assert (case["expected_fragment"] is None) == (case["expected_document"] is None)


def test_metric_computation_counts_all_failures():
    rows = [
        {"expected_document": "travel", "hit_at_1": True, "hit_at_3": True,
         "reciprocal_rank": 1., "answer_supported": True, "abstained": False,
         "passed": True, "latency_ms": 10.},
        {"expected_document": "security", "hit_at_1": False, "hit_at_3": True,
         "reciprocal_rank": .5, "answer_supported": False, "abstained": False,
         "passed": False, "latency_ms": 20.},
        {"expected_document": None, "hit_at_1": False, "hit_at_3": False,
         "reciprocal_rank": 0., "answer_supported": False, "abstained": False,
         "passed": False, "latency_ms": 30.},
        {"expected_document": None, "hit_at_1": False, "hit_at_3": False,
         "reciprocal_rank": 0., "answer_supported": False, "abstained": True,
         "passed": True, "latency_ms": 40.},
    ]
    s = summarize(rows)
    assert s["hit_at_1"] == .5
    assert s["recall_at_3"] == 1.
    assert s["mrr"] == .75
    assert s["answer_exact_support_rate"] == .5
    assert s["abstention_accuracy"] == .5
    assert s["case_pass_rate"] == .5
    assert s["latency_ms"]["p50"] == 25.
    assert s["latency_ms"]["p95"] == 38.5
    assert percentile([], 50) == 0.


def test_report_is_grounded_in_actual_index_results(tmp_path):
    labels = [
        {"id": "a", "category": "direct", "question": "correct answer?",
         "expected_document": "policy-travel", "expected_fragment": "$65"},
        {"id": "b", "category": "paraphrase", "question": "unknown outcome?",
         "expected_document": "policy-travel", "expected_fragment": "$65"},
        {"id": "c", "category": "unanswerable", "question": "no match?",
         "expected_document": None, "expected_fragment": None},
        {"id": "d", "category": "unanswerable", "question": "unknown question?",
         "expected_document": None, "expected_fragment": None},
    ]
    r = build_report(FakeIndex(), labels, generated_at="2026-10-28T22:00:00+00:00")
    assert r["schema_version"] == 1
    assert r["generated_at_utc"] == "2026-10-28T22:00:00+00:00"
    assert r["summary"]["cases"] == 4
    assert r["summary"]["answerable"] == 2
    assert r["summary"]["unanswerable"] == 2
    assert r["summary"]["hit_at_1"] == .5
    assert r["summary"]["recall_at_3"] == .5
    assert r["summary"]["mrr"] == .5
    assert r["summary"]["abstention_accuracy"] == .5
    assert r["cases"][0]["answer_supported"]
    assert not r["cases"][1]["answer_supported"]
    assert r["cases"][2]["abstained"]
    assert not r["cases"][3]["passed"]  # unanswerable false positive
    assert r["model"]["retriever"] == "fastembed"
    out = Path(tmp_path) / "evaluation.json"
    write_report(r, out)
    restored = json.loads(out.read_text())
    assert restored["summary"] == r["summary"]
    assert restored["cases"][3]["predicted_document"] == "policy-security"


def test_no_vacuous_success_when_no_cases():
    result = summarize([])
    assert result["cases"] == 0
    assert result["recall_at_3"] == 0
    assert result["abstention_accuracy"] == 0
