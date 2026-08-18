import os
import sys
import json
from dotenv import load_dotenv

# Setup path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

load_dotenv()

from qdrant_client import QdrantClient
from agents.solution.config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION

def main():
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    # Retrieve all points in the collection using scroll
    print(f"Fetching all KB documents from Qdrant Cloud collection '{QDRANT_COLLECTION}'...")
    try:
        results = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=200,
            with_payload=True,
            with_vectors=False
        )[0]
        
        # Sort by KB ID numerically
        kbs = []
        for r in results:
            payload = r.payload
            kbs.append(payload)
            
        kbs.sort(key=lambda x: x.get("id") or x.get("knowledge_id") or "")
        
        output_file = os.path.join(PROJECT_ROOT, "tests", "evaluation", "ragas", "cloud_knowledge_base.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(kbs, f, indent=2)
            
        print(f"Successfully retrieved {len(kbs)} KB documents.")
        print(f"Saved complete cloud KB metadata to: {output_file}")
        
        # Print list of IDs and Titles for review
        print("\nAll Cloud KB Documents:")
        print("==================================================")
        for kb in kbs:
            kb_id = kb.get("id") or kb.get("knowledge_id")
            title = kb.get("title")
            domain = kb.get("domain")
            print(f"  {kb_id} - {title} (domain: {domain})")
            
    except Exception as e:
        print(f"Error fetching scroll: {e}")

if __name__ == "__main__":
    main()
