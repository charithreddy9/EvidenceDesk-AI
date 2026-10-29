from backend.holdout import HOLDOUT_CASES,compare_holdout
from backend.retrieval import DEMO_DOCUMENTS

def test_holdout_has_separate_labels():
    assert len(HOLDOUT_CASES)>=12
    assert len({x["id"] for x in HOLDOUT_CASES})==len(HOLDOUT_CASES)
    assert all(x["id"].startswith("h") for x in HOLDOUT_CASES)
    assert {x["category"] for x in HOLDOUT_CASES}=={"direct","paraphrase","unanswerable"}
    ids={d["id"] for d in DEMO_DOCUMENTS}
    assert all(c["expected_document"] in ids or c["expected_document"] is None for c in HOLDOUT_CASES)
