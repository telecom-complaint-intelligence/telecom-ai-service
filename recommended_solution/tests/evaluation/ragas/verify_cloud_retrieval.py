import os
import sys
import json

# Setup path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from agents.solution.knowledge.retriever import VectorKnowledgeRetriever

def main():
    input_file = os.path.join(PROJECT_ROOT, "tests", "inputs", "eval_001.json")
    with open(input_file, "r", encoding="utf-8") as f:
        complaint_data = json.load(f)
        
    complaint_text = complaint_data.get("complaint")
    
    print("\nEVAL-001")
    print("\nComplaint:")
    print(complaint_text)
    
    retriever = VectorKnowledgeRetriever()
    retriever.print_debug_info(complaint_text, top_k=3)

if __name__ == "__main__":
    main()
