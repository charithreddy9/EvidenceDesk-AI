"""Reproducible, offline, labeled QA evaluation for the fictional policy corpus.

A static report is produced at image build time. The public UI reads this
snapshot; visitors cannot trigger expensive model evaluations. The labels are
small, manually curated regression fixtures, NOT a held-out benchmark.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import mean
from time import perf_counter

from .evidence import MIN_BGE_SIMILARITY, MAX_BGE_GAP, select_evidence, baseline_evidence, cited_source_sentence

DATASET_VERSION = "2026-10-28-v1"

# Expected answer fragments must occur verbatim in the cited sentence.
# These are small curated examples, not diverse real-world documents.
LABELED_CASES = [
    {"id":"d01","category":"direct","question":"What is the daily meal allowance?","expected_document":"policy-travel","expected_fragment":"$65 per day"},
    {"id":"d02","category":"direct","question":"When must itemized receipts be submitted?","expected_document":"policy-travel","expected_fragment":"30 days"},
    {"id":"d03","category":"direct","question":"Which airfare costs require manager preapproval?","expected_document":"policy-travel","expected_fragment":"$800"},
    {"id":"d04","category":"direct","question":"Where must travel be booked?","expected_document":"policy-travel","expected_fragment":"company travel portal"},
    {"id":"d05","category":"direct","question":"Where must production credentials be stored?","expected_document":"policy-security","expected_fragment":"secrets manager"},
    {"id":"d06","category":"direct","question":"Is multi-factor authentication required?","expected_document":"policy-security","expected_fragment":"Multi-factor authentication"},
    {"id":"d07","category":"direct","question":"How soon must suspected security incidents be reported?","expected_document":"policy-security","expected_fragment":"one hour"},
    {"id":"d08","category":"direct","question":"How quickly must P1 incidents be acknowledged?","expected_document":"policy-incident","expected_fragment":"ten minutes"},
    {"id":"d09","category":"direct","question":"When is a post-incident report published?","expected_document":"policy-incident","expected_fragment":"five business days"},
    {"id":"d10","category":"direct","question":"How long are application diagnostic logs retained?","expected_document":"policy-data","expected_fragment":"90 days"},
    {"id":"d11","category":"direct","question":"When are customer support exports deleted?","expected_document":"policy-data","expected_fragment":"14 days"},
    {"id":"d12","category":"direct","question":"What limits personal data access?","expected_document":"policy-data","expected_fragment":"least-privilege"},
    {"id":"p01","category":"paraphrase","question":"How much can I spend on meals during a work trip?","expected_document":"policy-travel","expected_fragment":"$65 per day"},
    {"id":"p02","category":"paraphrase","question":"How long do I have to turn in a travel receipt?","expected_document":"policy-travel","expected_fragment":"30 days"},
    {"id":"p03","category":"paraphrase","question":"Do expensive flights need approval from my manager?","expected_document":"policy-travel","expected_fragment":"$800"},
    {"id":"p04","category":"paraphrase","question":"Can plane tickets be booked outside the corporate tool?","expected_document":"policy-travel","expected_fragment":"company travel portal"},
    {"id":"p05","category":"paraphrase","question":"Where should live system passwords be saved?","expected_document":"policy-security","expected_fragment":"secrets manager"},
    {"id":"p06","category":"paraphrase","question":"How fast must we alert security if we suspect a breach?","expected_document":"policy-security","expected_fragment":"one hour"},
    {"id":"p07","category":"paraphrase","question":"How quickly do on-call engineers respond to a critical outage?","expected_document":"policy-incident","expected_fragment":"ten minutes"},
    {"id":"p08","category":"paraphrase","question":"How long do we have to write an incident retrospective?","expected_document":"policy-incident","expected_fragment":"five business days"},
    {"id":"p09","category":"paraphrase","question":"When are diagnostic records purged?","expected_document":"policy-data","expected_fragment":"90 days"},
    {"id":"p10","category":"paraphrase","question":"Can I paste customer records into a public chatbot?","expected_document":"policy-data","expected_fragment":"must not copy customer data"},
    {"id":"u01","category":"unanswerable","question":"What is the weather forecast in Tokyo?","expected_document":None,"expected_fragment":None},
    {"id":"u02","category":"unanswerable","question":"How many paid vacation days do employees get?","expected_document":None,"expected_fragment":None},
    {"id":"u03","category":"unanswerable","question":"What is the office parking reimbursement?","expected_document":None,"expected_fragment":None},
    {"id":"u04","category":"unanswerable","question":"What is the parental leave policy?","expected_document":None,"expected_fragment":None},
    {"id":"u05","category":"unanswerable","question":"Does the company reimburse gym memberships?","expected_document":None,"expected_fragment":None},
    {"id":"u06","category":"unanswerable","question":"What is the employee stock vesting schedule?","expected_document":None,"expected_fragment":None},
    {"id":"u07","category":"unanswerable","question":"Which laptop models may employees order?","expected_document":None,"expected_fragment":None},
    {"id":"u08","category":"unanswerable","question":"Can I work remotely from another country?","expected_document":None,"expected_fragment":None},
]


def percentile(values: list[float], percent: float) -> float:
    """Linearly interpolate a percentile of recorded milliseconds."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def _rate(numerator: int | float, denominator: int) -> float:
    return round(float(numerator / denominator), 4) if denominator else 0.0


def summarize(rows: list[dict]) -> dict:
    positives = [x for x in rows if x["expected_document"] is not None]
    negatives = [x for x in rows if x["expected_document"] is None]
    durations = [x["latency_ms"] for x in rows]
    return {
        "cases": len(rows),
        "answerable": len(positives),
        "unanswerable": len(negatives),
        "hit_at_1": _rate(sum(bool(x["hit_at_1"]) for x in positives), len(positives)),
        "recall_at_3": _rate(sum(bool(x["hit_at_3"]) for x in positives), len(positives)),
        "mrr": _rate(sum(float(x["reciprocal_rank"]) for x in positives), len(positives)),
        "answer_exact_support_rate": _rate(sum(bool(x["answer_supported"]) for x in positives), len(positives)),
        "abstention_accuracy": _rate(sum(bool(x["abstained"]) for x in negatives), len(negatives)),
        "case_pass_rate": _rate(sum(bool(x["passed"]) for x in rows), len(rows)),
        "latency_ms": {
            "mean": round(mean(durations), 2) if durations else 0.0,
            "p50": round(percentile(durations, 50), 2),
            "p95": round(percentile(durations, 95), 2),
            "min": round(min(durations), 2) if durations else 0.0,
            "max": round(max(durations), 2) if durations else 0.0,
        },
    }


def build_report(index, cases=None, *, generated_at=None, verifier=True) -> dict:
    examples = LABELED_CASES if cases is None else cases
    ids = [item["id"] for item in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("Evaluation case IDs must be unique")
    rows = []
    for case in examples:
        start = perf_counter()
        raw_hits = index.search(case["question"], top_k=12 if verifier and index.provider == "fastembed" else 3)
        hits = (select_evidence(raw_hits, index.provider, case["question"], 3)
                if verifier else baseline_evidence(raw_hits, index.provider))
        elapsed_ms = (perf_counter() - start) * 1000
        predicted = hits[0] if hits else None
        expected = case["expected_document"]
        position = next((rank for rank, hit in enumerate(hits, 1)
                         if hit["doc_id"] == expected), None) if expected else None
        answer = cited_source_sentence(predicted, "[1]") if predicted else ""
        answer_supported = bool(expected and predicted and
                                predicted["doc_id"] == expected and
                                case["expected_fragment"].lower() in predicted["text"].lower())
        abstained = predicted is None
        rows.append({
            "id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "expected_document": expected,
            "expected_fragment": case["expected_fragment"],
            "predicted_document": predicted["doc_id"] if predicted else None,
            "predicted_title": predicted["title"] if predicted else None,
            "answer": answer,
            "score": predicted["score"] if predicted else None,
            "abstained": abstained,
            "hit_at_1": position == 1,
            "hit_at_3": position is not None,
            "reciprocal_rank": round(1 / position, 4) if position else 0.0,
            "answer_supported": answer_supported,
            "passed": abstained if expected is None else answer_supported,
            "latency_ms": round(elapsed_ms, 2),
        })
    now = generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {
        "schema_version": 1,
        "verification_enabled": verifier,
        "generated_at_utc": now,
        "dataset": {
            "name": "EvidenceDesk fictional policy QA regression set",
            "version": DATASET_VERSION,
            "description": "Manually authored queries and source labels; not held-out or representative of production traffic.",
        },
        "model": {
            "retriever": index.provider,
            "embedding_model": "BAAI/bge-small-en-v1.5" if index.provider == "fastembed" else index.provider,
            "answer_mode": "sentence_extract" if index.provider == "fastembed" else "extractive",
            "corpus_documents": len(index.documents),
            "retrieval_unit": "source sentence" if index.provider == "fastembed" else "word chunk",
        },
        "filter": {
            "minimum_similarity": MIN_BGE_SIMILARITY if index.provider == "fastembed" else None,
            "maximum_gap": MAX_BGE_GAP if index.provider == "fastembed" else None,
        },
        "summary": summarize(rows),
        "groups": [
            {"category": category, **summarize([row for row in rows if row["category"] == category])}
            for category in ("direct", "paraphrase", "unanswerable")
        ],
        "cases": rows,
        "limitations": [
            "Synthetic policies and manually written labels are not an external held-out benchmark.",
            "Similarity scores are cosine similarities, not correctness probabilities.",
            "Expected answer support is a verbatim substring check, not human-graded entailment.",
            "Latency is one warm single-process pass per question in the image build; cold start, network, browser, and concurrent requests are excluded.",
            "Changing source documents or retrieval thresholds requires rebuilding this static evaluation snapshot.",
        ],
    }


def write_report(report: dict, path: str | Path):
    location = Path(path)
    location.parent.mkdir(parents=True, exist_ok=True)
    location.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
