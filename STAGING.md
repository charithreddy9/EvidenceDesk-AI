# Isolated local-semantic staging deployment

**Production safety:** Work is isolated in the `staging-fastembed` Git branch. The original `main` branch, `render.yaml`, existing `evidencedesk-ai` Render service, and its TF-IDF config are unchanged.

## First: RAM gate

The Render **Free** web service has **512 MB RAM and 0.1 CPU** (Render Compute Plans, October 2026). Model artifact size is **not** equal to runtime RAM consumption. Real ONNX model initialization plus Python, NumPy, scikit-learn and Uvicorn may exceed that limit. Do **not** assume a model that works on GitHub's larger hosted runner fits the free service.

The GitHub workflow [FastEmbed Staging Memory Gate](.github/workflows/staging-memory.yml) builds `Dockerfile.staging`, preloads **BAAI/bge-small-en-v1.5**, and runs FastAPI with `--memory=512m --memory-swap=512m`, then verifies retrieval with a paraphrased query. It captures `docker stats` and checks OOMKilled. CI limits CPU to 0.5—not Render's 0.1—so real Render latency and cold starts will still need measuring.

## Create a separate staging service (no production changes)

Use Render Dashboard > **New > Blueprint**, connecting `charithreddy9/EvidenceDesk-AI`.

- **Blueprint name:** `evidencedesk-ai-staging-blueprint`
- **Branch:** `staging-fastembed` (**not** `main`)
- **Blueprint Path:** `render.staging.yaml` (**not** `render.yaml`)
- Review that **ONLY** `evidencedesk-ai-staging-bge` will be created, with **Free** plan and Dockerfile `Dockerfile.staging`.
- Leave auto-deploy off; start by manually reviewing staging deployments.

Do not choose the existing production Blueprint or update its Blueprint Path. If the staging service shows a paid plan or charge, stop and review before approving it.

## Expected staging behavior

- `RETRIEVAL_BACKEND=fastembed`, BGE-small ONNX local dense embeddings
- `ENABLE_PAID_LLM=false`, `ENABLE_PAID_EMBEDDINGS=false`, no `OPENAI_API_KEY`
- `ALLOW_PUBLIC_UPLOAD=false` and anonymous writes rejected
- Four fictional policy documents, ephemeral SQLite data
- Check `/api/health` then ask: **How much can I spend per day on meals?**; the Sample Travel Policy should rank first
- Compare output with production at https://evidencedesk-ai.onrender.com

If the 512 MB gate fails or Render kills staging, keep production unchanged and switch to a separately approved higher-memory plan or smaller model/service architecture. **Do not upgrade paid services without explicit approval.**

## CI result interpretation

- Unit tests and embedding smoke tests confirm correctness and actual model inference.
- The 512 MB container test checks **one startup and one real retrieval**, not a full peak-memory, concurrency, cold-start, sustained load, or adversarial benchmark.
- Render's free service may sleep after inactivity; persistent state is not guaranteed.
- Staging is not yet a production-grade RAG system: retrieval is semantic, but paid LLM synthesis remains disabled, and citations are not independently checked for entailment.
