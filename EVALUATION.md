# EvidenceDesk: reproducible model-quality dashboard

The staging branch has a read-only **Evaluation dashboard** in the React sidebar. Its data is generated during the staging Docker build with the real **BAAI/bge-small-en-v1.5 FastEmbed ONNX encoder**, then served from a static JSON snapshot at `GET /api/evaluation`. Visitors cannot launch model evaluations. Production `main` is unaffected.

## Labeled dataset and metrics

The dataset in `backend/evaluation_report.py` includes 12 direct questions, 10 paraphrases and 8 deliberately unanswerable questions over four fictional policies. Every answerable question has an expected document and a verbatim answer fragment. Every unanswerable question has a null expected source.

- **Hit@1**: fraction of answerable queries with the expected source ranked first **after evidence filtering**.
- **Recall@3**: fraction of answerable queries with the expected source somewhere in the returned top three **after filtering**.
- **MRR**: average reciprocal rank of the expected source, zero if none retained.
- **Abstention accuracy**: fraction of unanswerable queries for which no evidence is returned.
- **Answer-fragment support**: fraction of answerable queries where the selected top sentence comes from the expected document and contains the manually labeled text. This is not a factual-entailment metric.
- **Latency**: single warm-process elapsed time around embedding, retrieval and filtering for each query, in milliseconds. Mean, p50 and p95 are displayed.

Case explorer filters by question type and pass/fail, and exposes predicted source, extracted answer, latency and expected answer text. A failed test case remains visible; nothing is hidden to make metrics look better.

## Generate/reproduce

With FastEmbed dependencies installed and a model cache available:

```sh
RETRIEVAL_BACKEND=fastembed python -m backend.evaluate --report /tmp/evidence-evaluation.json
```

Review `/tmp/evidence-evaluation.json`. Staging Docker creates the same report at `/app/reports/evaluation.json` during `docker build`. Report generation performs **no OpenAI calls**.

## Limitations

This tiny, hand-authored dataset is neither representative of live traffic nor independent of the corpus. Metrics are illustrative, not general-purpose QA accuracy. Similarity scores are cosine similarity, not probability of correctness. Answer-fragment matching is literal substring containment, not independent entailment validation. The report is a single offline run on the build machine—not live user latency, cold starts, model drift or long-term monitoring. Retraining, corpus edits or threshold changes require a fresh image build.

**Safety:** The staging Blueprint still specifies `ENABLE_PAID_LLM=false`, `ENABLE_PAID_EMBEDDINGS=false`, `ALLOW_PUBLIC_UPLOAD=false`, and `autoDeployTrigger: "off"`. The code and report are only on `staging-fastembed` until explicitly promoted.
