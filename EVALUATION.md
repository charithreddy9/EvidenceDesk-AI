# How I checked answer quality

The first versions of EvidenceDesk showed why retrieval accuracy and answer accuracy are not the same thing. Semantic search was good at finding a related policy, but sometimes it returned the wrong sentence. When I made filtering stricter, the app started rejecting valid paraphrases.

I made the evaluation dashboard to keep those trade-offs visible.

## The questions

The main test set has 30 manually written questions over four fictional policies: 12 direct questions, 10 paraphrases, and 8 questions that shouldn't have an answer. Each answerable question has an expected document and a short piece of text that should appear in the retrieved sentence.

There is also a 14-question development set with different wording. It uses the same four policies, and I inspected those results while adjusting retrieval. It's useful for regression testing, but it is **not** a blind or independent benchmark.

## What the numbers mean

- **Hit@1:** how often the expected document is ranked first among retained evidence.
- **Recall@3:** how often it appears in the first three retained results.
- **MRR:** average reciprocal rank of the expected document; missing results count as zero.
- **Abstention accuracy:** how often the app correctly returns no evidence for an unanswerable question.
- **Source-fragment support:** whether the top sentence comes from the expected document and contains the labeled text. This is a string check, not a human or model judgment about entailment.
- **Latency:** time spent on local retrieval and filtering in a warm single-process evaluation run. It excludes network delay, cold starts, and concurrent requests.

The dashboard includes individual cases so a higher average score can't hide a broken question.

## Reproduce it

Install the FastEmbed and development dependencies first, then run:

```bash
RETRIEVAL_BACKEND=fastembed python -m backend.evaluate --report /tmp/evidencedesk-evaluation.json
python -m backend.quality_gate /tmp/evidencedesk-evaluation.json
```

The staging Docker build produces the same type of report in `/app/reports/evaluation.json`. The frontend reads it from `GET /api/evaluation`. Opening the page does **not** trigger a fresh model evaluation.

The quality gate checks a few regression thresholds on both development sets and specifically checks the previously troublesome booking and credential-storage paraphrases. That doesn't prove the rules will work for a different collection of documents.

## What I learned, and what's still missing

More aggressive abstention can make a system look safe while silently losing valid answers. Adding TF-IDF alongside BGE helped recover useful candidates, but the remaining source check still uses practical heuristics. Similarity is not a confidence percentage, and retrieving a sentence doesn't independently prove an answer is faithful.

The strongest next test would use unseen documents, question labels collected without looking at the results, and human review of whether each answer really follows from its citation.

Staging keeps paid AI features off and uses fictional documents only. I wouldn't put confidential material into this public demo.
