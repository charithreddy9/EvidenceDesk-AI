# EvidenceDesk-AI: 3-minute technical demo

**Live staging:** https://evidencedesk-ai-staging-bge.onrender.com  
**Production (different implementation):** https://evidencedesk-ai.onrender.com

> Staging is manually deployed and may not yet be running the latest Git commit. Confirm the deployed revision and health endpoint before recording. Render's free tier may need a cold start. All policies are fictional.

## 0:00–0:35 — Problem and value
Open the staging demo. “A document Q&A system needs to return evidence, not a plausible-sounding guess. EvidenceDesk gives a concise answer with the specific source text and a transparent abstention when that evidence is missing.”

## 0:35–1:20 — Grounded sentence retrieval
Ask **“How much can I spend per day on meals?”**. The expected policy statement is **“The meal allowance is $65 per day.”** Inspect the *source evidence* card and document title; do not claim the similarity percentage is a confidence score.

Then try **“When is the post-incident report due?”** and show the cited source. Do not narrate a correct answer until it appears in the live demo.

## 1:20–1:50 — Out-of-scope safety
Ask **“How many vacation days do employees receive?”**. The application should abstain rather than answer from an unrelated reimbursement policy. This is a development regression case, not a proof against adversarial questions.

## 1:50–2:30 — Evaluation lab
Click **Evaluation dashboard**. Review Recall@3, MRR, abstention accuracy, response latency, before/after comparisons, and **failed** cases. Explain the small 30-question development set and 14-question development holdout, where the source labels are manually authored. Show the *Failed* filter.

## 2:30–3:00 — Engineering and deployment
Open [ARCHITECTURE.md](ARCHITECTURE.md), explain BGE ONNX + local TF-IDF fusion, source verification, FastAPI, React, SQLite, Docker, and GitHub Actions. Show the linked workflow results. Mention the 512 MiB memory gate, and that paid OpenAI features and anonymous writes are disabled.

## Capture checklist
- Record your own browser only after confirming the staging revision; hide access tokens and personal information.
- Suggested screenshots: (1) cited answer, (2) abstained question, (3) evaluation dashboard, (4) GitHub Actions checks.
- Place user-created captures under `docs/screenshots/` and link them from README. Do not fabricate dashboard captures.
- Describe strengths and limitations accurately; the synthetic benchmark does **not** represent general-world answer accuracy.

## Interview talking points
**Trade-off:** BGE improves semantic matching but can retrieve irrelevant policy fragments. TF-IDF, answer-type validation and evidence gating reduce that risk while introducing precision/recall trade-offs.

**Safety:** Exact source sentence with a citation; no paid generative model; public writes disabled.

**Measurement:** Both answerable and unanswerable cases; no hidden failures; regressions can fail the staging Docker build.

**Next steps:** Human-labeled, unseen documents; robust entailment evaluation; calibrated answerability; stronger authentication; concurrency and cold-start profiling.
