import os
import sys
import json

# Setup path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from agents.solution.graph.solution_graph import solution_graph

def main():
    input_file = os.path.join(PROJECT_ROOT, "tests", "inputs", "eval_001.json")
    with open(input_file, "r", encoding="utf-8") as f:
        complaint_data = json.load(f)
        
    solution_state = {"complaint_input": complaint_data}
    print("Invoking solution_graph for EVAL-001...")
    solution_result = solution_graph.invoke(solution_state)
    
    final_solution = solution_result.get("final_solution", {})
    retrieved_kb = solution_result.get("retrieved_knowledge", [])
    
    print("\n==================================================")
    print("COMPLAINT:")
    print("==================================================")
    print(complaint_data.get("complaint"))
    
    print("\n==================================================")
    print("RETRIEVED KNOWLEDGE:")
    print("==================================================")
    for idx, doc in enumerate(retrieved_kb):
        print(f"\nDocument {idx+1}:")
        print(f"  ID: {doc.get('knowledge_id') or doc.get('id') or doc.get('doc_id')}")
        print(f"  Title: {doc.get('title')}")
        print(f"  Content: {doc.get('content')}")
        
    print("\n==================================================")
    print("FINAL SOLUTION:")
    print("==================================================")
    print(json.dumps(final_solution, indent=2))

if __name__ == "__main__":
    main()
