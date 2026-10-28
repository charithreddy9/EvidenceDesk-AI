"""Exercise one-origin staging API and real semantic retrieval, using only stdlib."""
from __future__ import annotations
import json
import sys
import time
import urllib.error
import urllib.request

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:18080"

def call(path: str, payload=None):
    body = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        BASE + path,
        data=body,
        headers={"Content-Type": "application/json"} if body else {},
        method="POST" if body else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())

def main():
    for attempt in range(60):
        try:
            code, health = call("/api/health")
            if code == 200 and health.get("retriever") == "fastembed":
                break
        except (OSError, ValueError):
            pass
        time.sleep(3)
    else:
        raise SystemExit("Staging container never became healthy within 180 seconds")
    assert health["documents"] == 4, health
    code, config = call("/api/config")
    assert code == 200 and config["semantic_enabled"] is True
    assert config["llm_available"] is False and config["uploads_enabled"] is False
    code, answer = call("/api/ask", {"question": "How much can I spend per day on meals?", "generate": False})
    assert code == 200, (code, answer)
    assert answer["retriever"] == "fastembed", answer
    assert answer["citations"] and answer["citations"][0]["doc_id"] == "policy-travel", answer
    print("BGE retrieval top match:", answer["citations"][0]["doc_id"], "score:", answer["citations"][0]["score"])
    code, _ = call("/api/ask", {"question": "What is the meal allowance?", "generate": True})
    assert code == 403, "Paid LLM must remain disabled in staging"
    code, _ = call("/api/documents", {"title": "Forbidden upload", "content": "x" * 100})
    assert code == 403, "Anonymous writes must remain disabled in staging"
    print("Staging API, real ONNX semantic retrieval, and safety gates: PASSED")

if __name__ == "__main__":
    main()
