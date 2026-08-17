import os
from typing import Any, Dict, List

from app.agents.solution.config import (
    EMBEDDING_MODEL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_URL,
)


class HighAgentVectorRetriever:
    """
    Semantic vector search retriever for the HIGH Agent using Qdrant.
    """

    def __init__(self):
        self.enabled = False
        self.client = None
        self.model = None

        if QDRANT_URL and QDRANT_API_KEY:
            try:
                from qdrant_client import QdrantClient
                from sentence_transformers import SentenceTransformer

                self.client = QdrantClient(
                    url=QDRANT_URL, api_key=QDRANT_API_KEY
                )
                self.collection_name = QDRANT_COLLECTION
                self.model = SentenceTransformer(EMBEDDING_MODEL)
                self.enabled = True
            except Exception as e:
                print(f"⚠️ HighAgentVectorRetriever optional init skipped: {e}")

    def search(
        self,
        domain: str,
        problem_type: str,
        complaint_text: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        if not self.enabled or not self.client or not self.model:
            return []

        try:
            query_text = f"{domain} {problem_type}. {complaint_text}"
            query_vector = self.model.encode(query_text).tolist()

            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
            ).points

            documents = [result.payload for result in results]
            return documents
        except Exception as e:
            print(f"⚠️ Error querying Qdrant: {e}")
            return []


high_agent_retriever = HighAgentVectorRetriever()
