import os
import sys
import json

# Setup path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from agents.solution.config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION, EMBEDDING_MODEL

def main():
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Inputs for EVAL-001
    domain = "portal_auth"
    component = "portal"
    failure_type = "authentication_error"
    complaint_text = "I keep getting an error 403 password mismatch when trying to open my bill profile page."
    
    query_text = f"{domain} {component} {failure_type}. {complaint_text}"
    query_vector = model.encode(query_text).tolist()
    
    print("==================================================")
    print("QUERY TEXT:")
    print(query_text)
    print("==================================================")
    
    # Search Qdrant and get raw points with scores
    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=5,
        with_payload=True
    ).points
    
    print("\nRETRIEVED POINTS FROM QDRANT CLOUD:")
    print("==================================================")
    for idx, point in enumerate(results):
        print(f"\nPosition {idx+1}:")
        print(f"  ID: {point.id}")
        print(f"  Score: {point.score:.4f}")
        print(f"  Payload:")
        print(json.dumps(point.payload, indent=4))

if __name__ == "__main__":
    main()
