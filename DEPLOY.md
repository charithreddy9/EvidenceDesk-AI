# Deploy EvidenceDesk as a single Render web service

This is the simplest recruiter-demo deployment. **One publicly accessible URL** serves both the React SPA and FastAPI, without cross-origin requests.

## One-click blueprint

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/charithreddy9/EvidenceDesk-AI)

1. Sign in to Render and authorize GitHub repository access when requested.
2. Select the blueprint configuration in `render.yaml` and review the service name, plan and any billing prompts.
3. Deploy. The Docker build compiles the React app and installs the Python API.
4. Visit your assigned HTTPS service URL. Try asking **What is the meal allowance?**
5. Check `/api/health` to confirm that the backend is responding.

This demo is intentionally **read-only**. `RETRIEVAL_BACKEND=tfidf` runs without paid API keys. There is no shared admin password and uploads are disabled by default. Only fictional policy samples are shipped.

**Free-tier limitation:** the SQLite database is on ephemeral storage; it can reset on restart or redeploy. Do not upload private/regulated data. Sleeping instances can take time to wake. Enable a persistent database and stronger access controls before allowing upload features.

## Configuration

The deployment is defined in `render.yaml` using `Dockerfile.render` and `backend/web.py`.

- `RETRIEVAL_BACKEND=tfidf`: offline baseline.
- `ALLOW_PUBLIC_UPLOAD=false`: restrict document changes.
- `DATA_DB_PATH=/tmp/evidencedesk.sqlite3`: ephemeral sample-only SQLite.
- `ENABLE_API_DOCS=false`: hide interactive API docs for the public demo.
- `PORT`: Render provides this automatically.

Local Docker Compose continues to work as previously documented. This Render blueprint is independent of `compose.yaml`.

## Advanced AI options

For semantic embeddings, select `openai` and add an `OPENAI_API_KEY` only through the hosting service's secret environment settings; this incurs API usage costs. Do not commit keys.

For local Sentence Transformers, build with the semantic dependencies and account for model download size, RAM and cold start limits. Neither hosted mode nor the semantic model is validated by default CI.

For optional LLM synthesis, a server-side `OPENAI_API_KEY` is needed. If you enable it, implement authentication and cost controls before sharing a public URL to prevent unbounded paid usage.
