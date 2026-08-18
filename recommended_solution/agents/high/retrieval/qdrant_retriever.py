import os
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from agents.solution.config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION, EMBEDDING_MODEL

class HighAgentVectorRetriever:
    """
    Semantic vector search retriever for the HIGH Agent using Qdrant.
    """
    def __init__(self):
        self.enabled = bool(QDRANT_URL and QDRANT_API_KEY)
        if self.enabled:
            self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            self.collection_name = QDRANT_COLLECTION
            self.model = SentenceTransformer(EMBEDDING_MODEL)

    def search(self, domain: str, problem_type: str, complaint_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.enabled:
            print("⚠️ Qdrant credentials not found. HighAgentVectorRetriever is disabled.")
            return []
            
        try:
            query_text = f"{domain} {problem_type}. {complaint_text}"
            query_vector = self.model.encode(query_text).tolist()
            
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k
            ).points
            
            documents = [result.payload for result in results]
            return documents
        except Exception as e:
            print(f"⚠️ Error querying Qdrant: {e}")
            return []

high_agent_retriever = HighAgentVectorRetriever()
