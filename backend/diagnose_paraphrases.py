"""Print real local retrieval diagnostics for two paraphrase regression questions.

Never runs on public HTTP requests; useful for CI troubleshooting.
"""
from .retrieval import EvidenceIndex
from .evidence import select_evidence, support_score
from .evaluation_report import LABELED_CASES

def main():
    index=EvidenceIndex(provider="fastembed")
    for case in LABELED_CASES:
        if case["id"] not in {"p04","p05"}:
            continue
        q=case["question"]
        print("DIAGNOSTIC",case["id"],q,flush=True)
        found=index.search_hybrid(q,top_k=12)
        accepted=select_evidence(found,"fastembed",q,3)
        print("ACCEPTED",[(x["doc_id"],x["text"]) for x in accepted],flush=True)
        for hit in found[:8]:
            score,details=support_score(q,hit)
            print("CANDIDATE",repr(hit["text"]),hit["doc_id"],
                  "BGE",hit["score"],"TFIDF",hit["lexical_score"],
                  "agreement",score,"matched",details["matched"],
                  "missing",details["missing"],flush=True)

if __name__=="__main__":
    main()
