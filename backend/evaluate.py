"""Small, manually authored QA evaluation: smoke/regression only, not a benchmark."""
import argparse
import os
import json
from .evaluation_report import build_report, write_report
from .retrieval import EvidenceIndex
from .evidence import select_evidence

# Retain the original five-case interface for existing CI/tests.
CASES = [
    ("When must receipts be submitted?", "policy-travel"),
    ("What is the daily meal allowance?", "policy-travel"),
    ("Where must production credentials be stored?", "policy-security"),
    ("How quickly must P1 incidents be acknowledged?", "policy-incident"),
    ("How long are diagnostic logs kept?", "policy-data"),
]
PARAPHRASE_CASES = [
    ("How much can I spend per day on meals?", "policy-travel"),
    ("How soon do I need to submit a business trip receipt?", "policy-travel"),
    ("How fast must we respond to a critical P1 incident?", "policy-incident"),
    ("For how many days are application diagnostic records retained?", "policy-data"),
    ("Where should passwords and production keys be kept?", "policy-security"),
]
UNANSWERABLE_CASES = [
    "What is the weather in Tokyo?",
    "What is the company's parental leave policy?",
    "How much does employee parking cost?",
]


def measure(index, positives=CASES, negatives=UNANSWERABLE_CASES):
    """Report filtered hit@1, recall@3, MRR, and unanswerable abstention."""
    hit1 = hit3 = reciprocal_rank = 0.0
    for query, expected in positives:
        chosen = select_evidence(index.search(query, top_k=3), index.provider)
        matching = next((rank for rank, item in enumerate(chosen, start=1)
                         if item["doc_id"] == expected), None)
        hit1 += bool(matching == 1)
        hit3 += bool(matching is not None)
        reciprocal_rank += 1.0 / matching if matching else 0.0
    abstained = sum(not select_evidence(index.search(query, top_k=3), index.provider)
                    for query in negatives)
    return {
        "hit_at_1": hit1 / len(positives),
        "recall_at_3": hit3 / len(positives),
        "mrr": reciprocal_rank / len(positives),
        "abstention_accuracy": abstained / len(negatives),
        "positives": len(positives),
        "negatives": len(negatives),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate local fictional-policy retrieval")
    parser.add_argument("--report", help="Write actual labeled offline evaluation results as JSON")
    args = parser.parse_args()
    provider = os.getenv("RETRIEVAL_BACKEND", "tfidf")
    index = EvidenceIndex(provider=provider)
    if args.report:
        report = build_report(index)
        baseline = build_report(index, verifier=False)
        report['baseline_summary'] = baseline['summary']
        report['comparison'] = {metric: round(report['summary'][metric]-baseline['summary'][metric],4) for metric in ['hit_at_1','recall_at_3','mrr','abstention_accuracy','case_pass_rate']}
        from .holdout import compare_holdout
        report["holdout"] = compare_holdout(index)
        write_report(report, args.report)
        print(json.dumps({"model": report["model"], "summary": report["summary"]}, indent=2))
        print(f"Built static evaluation report at {args.report}; no paid APIs used.")
        return
    for title, positives in [("literal queries", CASES), ("paraphrases", PARAPHRASE_CASES)]:
        scores = measure(index, positives=positives)
        print(f"{provider} {title}: hit@1={scores['hit_at_1']:.2f} "
              f"recall@3={scores['recall_at_3']:.2f} MRR={scores['mrr']:.2f} "
              f"unanswerable abstention={scores['abstention_accuracy']:.2f}")
    print("Warning: small manually authored fictional-policy evaluation, not a "
          "held-out benchmark; scores and thresholds do not prove reliability.")


if __name__ == "__main__":
    main()
