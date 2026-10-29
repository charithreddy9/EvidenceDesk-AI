"""Evidence retrieval with explicit lexical, local-semantic, and remote-semantic modes.

Use RETRIEVAL_BACKEND=tfidf (offline lexical), fastembed (local ONNX dense vectors),
sentence-transformers (local PyTorch), or openai (paid hosted vectors).
Never silently switch backends or send document text to an external provider.
"""
from __future__ import annotations

import os
import re
from uuid import uuid4

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .evidence import terms

DEMO_DOCUMENTS = [
    {"id": "policy-travel", "title": "Sample Travel & Expenses Policy", "text": "Approved business travel expenses are reimbursable. Employees must submit an itemized receipt within 30 days of purchase. The meal allowance is $65 per day. Airfare costing more than $800 requires manager preapproval. All travel must be booked through the company travel portal. This is a fictional demonstration policy."},
    {"id": "policy-security", "title": "Sample Security Handbook", "text": "Production credentials must be stored in an approved secrets manager and must never be committed to source control. Multi-factor authentication is mandatory for company accounts. Report any suspected security incident to the security team within one hour. This is a fictional demonstration policy."},
    {"id": "policy-incident", "title": "Sample Incident Response Playbook", "text": "Priority-one incidents require on-call acknowledgment within ten minutes. A customer-impacting incident must have an assigned incident commander. After mitigation, the team publishes a post-incident report within five business days. This is a fictional demonstration policy."},
    {"id": "policy-data", "title": "Sample Data Governance Guide", "text": "Application diagnostic logs are retained for 90 days. Customer support exports must be encrypted and deleted after 14 days. Personal data access is limited by least-privilege permissions. Teams must not copy customer data into public AI tools. This is a fictional demonstration policy."},
]

def chunks(text: str, window: int = 90, overlap: int = 20) -> list[str]:
    if not 0 <= overlap < window:
        raise ValueError("overlap must be smaller than window")
    words = re.findall(r"\S+", text)
    if not words:
        return []
    step = window - overlap
    return [" ".join(words[i:i + window]) for i in range(0, len(words), step)
            if words[i:i + window] and (i == 0 or len(words) - i > overlap)]

def sentence_chunks(text: str) -> list[str]:
    """Split source text into bounded, unmodified sentences for local BGE QA.

    Sentence retrieval narrows citations to exactly the source span used
    for an extractive answer. Long/no-punctuation text falls back to
    word-bounded chunks to avoid excessive input.
    """
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text.strip())
    results = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if len(piece.split()) > 90:
            results.extend(chunks(piece, window=90, overlap=15))
        else:
            results.append(piece)
    return results


class SentenceTransformerEncoder:
    """Lazy import keeps lexical installs lightweight."""
    def __init__(self, model_name: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name or os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, batch_size=32, show_progress_bar=False), dtype=np.float32)

class FastEmbedEncoder:
    """Real local ONNX dense embedding model; no remote inference/API cost.

    The model is downloaded on first install/build. Pin its revision in production
    and review the model license before commercial reuse.
    """
    def __init__(self, model_name: str | None = None):
        from fastembed import TextEmbedding
        self.model_name = model_name or os.getenv("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")
        self.model = TextEmbedding(
            model_name=self.model_name,
            cache_dir=os.getenv("FASTEMBED_CACHE_DIR") or None,
            threads=max(1, min(int(os.getenv("FASTEMBED_THREADS", "1")), 4)),
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(list(self.model.embed(texts)), dtype=np.float32)

    def encode_query(self, texts: list[str]) -> np.ndarray:
        # BGE v1.5 was trained with a retrieval query instruction.
        prepared = ["Represent this sentence for searching relevant passages: " + t for t in texts]
        return self.encode(prepared)


class OpenAIEncoder:
    def __init__(self, model_name: str | None = None):
        from openai import OpenAI
        if os.getenv("ENABLE_PAID_EMBEDDINGS", "false").lower() != "true":
            raise ValueError("Paid embeddings disabled: explicitly set ENABLE_PAID_EMBEDDINGS=true after approving costs")
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is required for openai embeddings")
        self.client = OpenAI(timeout=30)
        self.model = model_name or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    def encode(self, texts: list[str]) -> np.ndarray:
        response = self.client.embeddings.create(model=self.model, input=texts)
        return np.asarray([item.embedding for item in sorted(response.data, key=lambda x: x.index)], dtype=np.float32)

class EvidenceIndex:
    def __init__(self, documents=None, provider="tfidf", encoder=None):
        self.documents = list(DEMO_DOCUMENTS if documents is None else documents)
        self.provider = provider
        if provider not in {"tfidf", "fastembed", "sentence-transformers", "openai"}:
            raise ValueError("RETRIEVAL_BACKEND must be tfidf, fastembed, sentence-transformers, or openai")
        self.encoder = encoder if encoder is not None else (
            SentenceTransformerEncoder() if provider == "sentence-transformers" else
            FastEmbedEncoder() if provider == "fastembed" else
            OpenAIEncoder() if provider == "openai" else None
        )
        self._rebuild()

    def _rebuild(self):
        self.passages = []
        for doc in self.documents:
            pages = doc.get("pages") or [doc["text"]]
            for page_number, page_text in enumerate(pages, start=1):
                parts = sentence_chunks(page_text) if self.provider == "fastembed" else chunks(page_text)
                for chunk_index, part in enumerate(parts):
                    self.passages.append({
                        "doc_id": doc["id"], "title": doc["title"], "chunk": chunk_index,
                        "page": page_number if doc.get("pages") else None, "text": part
                    })
        self.matrix = None
        self.vectorizer = None
        self.lexical_vectorizer = None
        self.lexical_matrix = None
        if not self.passages:
            return
        texts = [p["text"] for p in self.passages]
        if self.provider == "tfidf":
            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
            self.matrix = self.vectorizer.fit_transform(texts)
        else:
            self.matrix = self._normalized(self.encoder.encode(texts))
            if self.provider == 'fastembed':
                self.lexical_vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
                self.lexical_matrix = self.lexical_vectorizer.fit_transform([' '.join(terms(p['title'] + ' ' + p['text'])) for p in self.passages])

    @staticmethod
    def _normalized(array):
        values = np.asarray(array, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("Embedding encoder must return a 2D matrix")
        return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)

    def replace_documents(self, documents):
        self.documents = list(documents)
        self._rebuild()

    def add_document(self, title: str, content: str, pages=None):
        doc = {"id": str(uuid4()), "title": title.strip(), "text": content.strip()}
        if pages:
            doc["pages"] = pages
        self.documents.append(doc)
        self._rebuild()
        return {"id": doc["id"], "title": doc["title"], "chunks": len([p for p in self.passages if p["doc_id"] == doc["id"]])}

    def search(self, question: str, top_k: int = 3, minimum_score: float | None = None):
        if self.matrix is None:
            return []
        if minimum_score is None:
            minimum_score = 0.11 if self.provider == "tfidf" else 0.30
        if self.provider == "tfidf":
            scores = cosine_similarity(self.vectorizer.transform([question]), self.matrix).ravel()
        else:
            encode_query = getattr(self.encoder, "encode_query", self.encoder.encode)
            query = self._normalized(encode_query([question]))
            scores = (query @ self.matrix.T).ravel()
        order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), self.passages[i]["doc_id"], self.passages[i]["chunk"]))
        return [{**self.passages[i], "score": round(float(scores[i]), 4)}
                for i in order if float(scores[i]) >= minimum_score][:top_k]

    def search_hybrid(self, question: str, top_k: int = 12):
        """Combine local BGE cosine with normalized local TF-IDF; no extra model."""
        if self.provider != "fastembed":
            return self.search(question, top_k=top_k)
        if self.matrix is None:
            return []
        query = self._normalized(getattr(self.encoder, "encode_query", self.encoder.encode)([question]))
        dense = (query @ self.matrix.T).ravel()
        q = " ".join(terms(question))
        sparse = cosine_similarity(self.lexical_vectorizer.transform([q]), self.lexical_matrix).ravel() if q else np.zeros(len(self.passages))
        indices = [i for i in range(len(dense)) if dense[i] >= .40 or sparse[i] >= .08]
        indices.sort(key=lambda i: (-.73*float(dense[i])-.27*float(sparse[i]), -float(dense[i]), self.passages[i]["doc_id"], self.passages[i]["chunk"]))
        return [{**self.passages[i], "score":round(float(dense[i]),4), "lexical_score":round(float(sparse[i]),4)} for i in indices[:top_k]]
