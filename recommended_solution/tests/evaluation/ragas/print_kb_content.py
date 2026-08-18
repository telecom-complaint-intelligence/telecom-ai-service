import os
import sys
import json
from dotenv import load_dotenv

# Setup path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

load_dotenv()

from agents.solution.knowledge.retriever import VectorKnowledgeRetriever

def main():
    r = VectorKnowledgeRetriever()
    # Search for top 10 documents
    docs = r.search(
        domain="Account Access",
        component="",
        failure_type="",
        complaint_text="403 password mismatch bill profile page",
        top_k=10
    )
    
    print("\nRETRIEVED DOCUMENT CONTENTS:")
    print("==================================================")
    for idx, d in enumerate(docs):
        print(f"\n[{idx+1}] ID: {d.get('id') or d.get('knowledge_id')}")
        print(f"Title: {d.get('title')}")
        print(f"Domain: {d.get('domain')}")
        print(f"Content: {d.get('content')}")
        print("--------------------------------------------------")

if __name__ == "__main__":
    main()
