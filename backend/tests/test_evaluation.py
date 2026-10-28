from backend.evaluate import CASES
from backend.retrieval import EvidenceIndex

def test_retrieval_recall_k1_k3():
    index = EvidenceIndex()
    hit1 = sum(any(x["doc_id"] == label for x in index.search(q, top_k=1)) for q, label in CASES)
    hit3 = sum(any(x["doc_id"] == label for x in index.search(q, top_k=3)) for q, label in CASES)
    assert hit3 >= hit1
    assert hit3 >= 4
