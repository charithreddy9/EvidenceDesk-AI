# EvidenceDesk — Full-Stack AI Document Intelligence

### [🌐 Open Live Demo](https://evidencedesk-ai.onrender.com) · [View API health](https://evidencedesk-ai.onrender.com/api/health)

**Deployment:** Render dashboard reports **Live** (free-tier single-service deployment). Public endpoint behavior should be manually smoke-tested before calling the demo fully verified.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/charithreddy9/EvidenceDesk-AI)

**[Deployment instructions](DEPLOY.md)** — one Render service builds and serves both the React frontend and FastAPI backend.

**Status: public hosted demo · developing portfolio prototype · NOT production-secure**

Document-question answering with **FastAPI + React**, SQLite-backed source records, PDF/TXT/Markdown ingestion, and transparent retrieval. Four *explicitly configured* retrieval backends:

- `tfidf`: fully offline lexical retrieval; default, works without embedding model or API keys.
- `fastembed`: **real local ONNX semantic embeddings** using BGE-small-en-v1.5. Install `backend/requirements-fastembed.txt` and set `RETRIEVAL_BACKEND=fastembed`. The public Render deployment does **not** install or enable this by default. Initial model download and memory requirements must be verified separately.
- `sentence-transformers`: **real local semantic embeddings** (default all-MiniLM-L6-v2). Requires `backend/requirements-semantic.txt` and downloading model weights during initial setup.
- `openai`: **real hosted semantic embeddings** via `text-embedding-3-small`; requires paid `OPENAI_API_KEY`.

Optional **OpenAI LLM** synthesis is **disabled by default even when an API key is present**. It requires explicit `ENABLE_PAID_LLM=true`, a server-side `OPENAI_API_KEY`, and a configured `ADMIN_TOKEN`; an in-process limit of 10 generation requests per hour applies. Enable it only after approving costs and deployment security. When enabled it is prompted to provide concise source-cited answers. Citation numbers and response size are validated; factual entailment is **not** independently verified. Uncited answers and unknown citation markers cause abstention.

## Architecture

```text
Browser / React ─► Nginx (proxy /api) ─► FastAPI ─► SQLite documents
                                              │
                                   Page-aware text chunking
                                              │
                                   Explicitly selected retriever
                        ┌─────────────────────┼──────────────────────┐
                    TF-IDF                Local embeddings       OpenAI embeddings
                        └─────────────────────┼──────────────────────┘
                                     Ranked source passages
                                              │
                              Extractive answer OR optional LLM
                                              │
                                Validated reference markers [1]
```

Sources are stored persistently. The in-process vector index is rebuilt on startup and after uploads or deletion; **embeddings are not yet persisted**. This is a single-instance demo, not a scalable vector database system.

## Quick start: Docker (recommended)

From the repository root:

```bash
cp .env.example .env
# Edit .env and replace ADMIN_TOKEN with a long random secret
docker compose up --build
```

Browse **http://localhost:8080**. Compose intentionally binds to loopback only; use a TLS-terminating reverse proxy and additional hardening for any public deployment. Paste the same `ADMIN_TOKEN` into the app's token field to access protected endpoints and upload documents. The token stays in React state (not local storage). Do **not** embed the token in the frontend bundle.

For local embeddings, change `RETRIEVAL_BACKEND=sentence-transformers` and `INSTALL_SEMANTIC=true` in `.env`, then rebuild. First model download requires internet and increases image size/startup latency.

For OpenAI embeddings, use `RETRIEVAL_BACKEND=openai` and set `OPENAI_API_KEY`. LLM synthesis uses the same server-side key and can incur charges.

## Local development without Docker

Requires Python 3.11+ and Node.js 22+.

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r backend/requirements-dev.txt
# For local semantic inference, also install: pip install -r backend/requirements-semantic.txt
# Optional: set ADMIN_TOKEN in your environment before launching
uvicorn backend.app:app --reload --port 8000
```

In a second terminal:

```bash
cd frontend
npm install
# Direct local backend from Vite: create frontend/.env.local containing:
# VITE_API_BASE_URL=http://localhost:8000
npm run dev
```

API reference in development: **http://localhost:8000/docs**.

## API

| Endpoint | Notes |
| --- | --- |
| `GET /api/health` | Unauthenticated health |
| `GET /api/config` | Retriever and feature capability flags |
| `GET /api/documents` | Document list; titles/IDs and chunk counts |
| `POST /api/upload` | Ingest PDF, TXT, or MD, max 2 MiB; write protection applies |
| `POST /api/documents` | JSON text ingestion |
| `DELETE /api/documents/{id}` | Delete user-imported docs (seed documents protected) |
| `POST /api/ask` | Top-k retrieval; optional `generate: true` LLM synthesis |

`ADMIN_TOKEN` protects read/write endpoints via `Authorization: Bearer …`; **without a token, read-only sample mode is enabled by default**. In a public deployment, always set a strong token. This is basic single-tenant protection, not production identity or RBAC.

## Quality checks

```bash
python -m pytest backend/tests -q
python -m backend.evaluate
cd frontend && npm run build
```

CI runs offline TF-IDF tests, including a deterministic fake encoder test for the semantic retrieval integration. Real embedding model loading is available as an explicit **manual GitHub Actions workflow_dispatch** smoke job (downloads BGE ONNX weights). Standard push CI runs offline semantic-vector tests without downloading weights. Hosted API calls are never performed in CI.

## Limitations and roadmap

- Four **fictional sample policies** and a five-question toy retrieval evaluation. No claims of real-world benchmark accuracy.
- PDF extraction only handles text-based PDFs, not scanned documents/OCR.
- TF-IDF or embedding cosine score is **not a calibrated confidence score**. Retrieval thresholds are development heuristics, and semantic search can return semantically related but irrelevant chunks.
- LLM citation marker validation does not guarantee factual accuracy; rigorous faithfulness evaluation and adversarial prompt-injection tests remain to be done.
- SQLite and in-process vector index are single-instance; process restarts rebuild the model index.
- Shared admin token, no per-user authorization, rate limits, virus scanning, document redaction, audit logs, or distributed storage. **Do not ingest sensitive or regulated documents in an untrusted deployment.**
- Planned next: persistent vector store (pgvector), reranking, evaluated semantic retrieval on benchmark data, OAuth/OIDC login, durable ingest jobs, audit trails, and public deployment with hardened ingress.

## Safe semantic RAG evaluation

- **Live Render:** stays on `RETRIEVAL_BACKEND=tfidf`, `ENABLE_PAID_LLM=false`, and `ALLOW_PUBLIC_UPLOAD=false`. No paid calls or anonymous document writes.
- **Offline dense semantics:** install `pip install -r backend/requirements-fastembed.txt`, set `RETRIEVAL_BACKEND=fastembed`, then run `python -m backend.evaluate`. First run downloads the local ONNX BGE model; the build and host need sufficient disk/RAM.
- **Unit CI:** fake-vector semantic ranking and retrieval smoke tests; **manual CI:** real model download/inference. Current five-question toy hit@k is a regression check, **not a statistical retrieval benchmark**.
- **Paid service safeguard:** `ENABLE_PAID_LLM=true` + `ADMIN_TOKEN` + `OPENAI_API_KEY` are all required before hosted generation; the 10/hour cap is per process, not a production-wide accounting guarantee. Never switch it on publicly without an authenticated trusted-user gateway and budget monitoring.
- **Grounding limitation:** verifying allowed citation numbers does not prove a generated factual claim is supported; human review and labeled faithfulness tests remain important.

**Repository:** https://github.com/charithreddy9/EvidenceDesk-AI. The earlier portfolio-workspace copy remains available for reference.
