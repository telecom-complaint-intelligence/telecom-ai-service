import json
from pathlib import Path
from typing import Any

from app.agents.solution.config import (
    EMBEDDING_MODEL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_URL,
)

DATA_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "knowledge_base.json"
)


class VectorKnowledgeRetriever:
    """
    Semantic vector search retriever using Qdrant and SentenceTransformers,
    with an automatic offline JSON knowledge base fallback.
    """

    def __init__(self):
        self.client = None
        self.collection_name = QDRANT_COLLECTION
        self.model = None

        if QDRANT_URL:
            try:
                from qdrant_client import QdrantClient
                from sentence_transformers import SentenceTransformer

                self.client = QdrantClient(
                    url=QDRANT_URL, api_key=QDRANT_API_KEY
                )
                self.model = SentenceTransformer(EMBEDDING_MODEL)
            except Exception as e:
                print(
                    f"⚠️ Qdrant client initialization skipped ({e}). Using local KB."
                )

    def _fallback_search(
        self,
        domain: str,
        component: str,
        failure_type: str,
        complaint_text: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Fallback local keyword/entity search on knowledge_base.json"""
        if not DATA_PATH.exists():
            return []

        try:
            with open(DATA_PATH, encoding="utf-8") as f:
                kb_data = json.load(f)
        except Exception:
            return []

        query_tokens = set(
            f"{domain} {component} {failure_type} {complaint_text}".lower().split()
        )
        scored = []
        for item in kb_data:
            item_text = (
                f"{item.get('title', '')} {item.get('domain', '')} {item.get('component', '')} {item.get('content', '')}".lower()
            )
            item_tokens = set(item_text.split())
            overlap = len(query_tokens.intersection(item_tokens))
            scored.append((overlap, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def search(
        self,
        domain: str,
        component: str,
        failure_type: str,
        complaint_text: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        if self.client and self.model:
            try:
                query_text = f"{domain} {component} {failure_type}. {complaint_text}"
                query_vector = self.model.encode(query_text).tolist()
                results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=top_k,
                ).points
                return [result.payload for result in results]
            except Exception as e:
                print(f"⚠️ Qdrant vector search failed, using local KB: {e}")

        return self._fallback_search(
            domain, component, failure_type, complaint_text, top_k
        )
