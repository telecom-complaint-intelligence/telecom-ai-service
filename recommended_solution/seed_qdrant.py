import os
import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

def main():
    # Load .env file
    load_dotenv()
    
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection_name = os.getenv("QDRANT_COLLECTION", "telecom_knowledge_base")
    embedding_model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    if not qdrant_url:
        print("Error: QDRANT_URL is not set. Please update your .env file.")
        return
        
    print(f"Connecting to Qdrant at: {qdrant_url}")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    
    print(f"Loading embedding model: {embedding_model_name}...")
    model = SentenceTransformer(embedding_model_name)
    
    # Sentence Transformer all-MiniLM-L6-v2 vectors are 384 dimensions
    vector_size = 384
    
    print(f"Recreating collection: '{collection_name}' in Qdrant...")
    try:
        # Check and delete existing collection to refresh it
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
    except Exception:
        # Fallback for older library versions where collection_exists is not available
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
            
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    
    # Load local knowledge base entries
    kb_path = os.path.join("data", "knowledge_base.json")
    if not os.path.exists(kb_path):
        print(f"Error: Knowledge base file not found at {kb_path}")
        return
        
    with open(kb_path, "r", encoding="utf-8") as f:
        articles = json.load(f)
        
    print(f"Loaded {len(articles)} articles from {kb_path}")
    
    # Process and upsert articles
    points = []
    for idx, article in enumerate(articles):
        domain = article.get("domain", "")
        component = article.get("component", "")
        failure_type = article.get("failure_type", "")
        content = article.get("content", "")
        
        # Conjoin the attributes to match the retrieval query format
        text_to_embed = f"{domain} {component} {failure_type}. {content}"
        vector = model.encode(text_to_embed).tolist()
        
        points.append(
            PointStruct(
                id=idx + 1,
                vector=vector,
                payload=article
            )
        )
        
    print(f"Upserting vectors into '{collection_name}'...")
    client.upsert(
        collection_name=collection_name,
        points=points
    )
    print("✅ Qdrant database successfully initialized and seeded!")

if __name__ == "__main__":
    main()
