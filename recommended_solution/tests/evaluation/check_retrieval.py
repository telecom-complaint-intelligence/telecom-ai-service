import os
import sys

# Ensure the root recommended_solution directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agents.solution.knowledge.retriever import VectorKnowledgeRetriever

retriever = VectorKnowledgeRetriever()

cases = {
    "EVAL-002": {
        "domain": "Account Access",
        "component": "",
        "failure_type": "",
        "complaint": "My password reset link is not arriving in my email.",
    },

    "EVAL-003": {
        "domain": "Internet Performance",
        "component": "",
        "failure_type": "",
        "complaint": "There is an internet outage affecting my area.",
    },

    "EVAL-009": {
        "domain": "Internet Performance",
        "component": "",
        "failure_type": "",
        "complaint": "Internet service is down for multiple customers in the area.",
    },

    "EVAL-006": {
        "domain": "Billing",
        "component": "",
        "failure_type": "",
        "complaint": "Customer was charged twice for the same recharge.",
    },

    "EVAL-008": {
        "domain": "Internet Performance",
        "component": "",
        "failure_type": "",
        "complaint": "My internet speed is very slow.",
    },

    "EVAL-012": {
        "domain": "Internet Performance",
        "component": "router",
        "failure_type": "connectivity",
        "complaint": "My router is not providing internet connectivity.",
    },

    "EVAL-013": {
        "domain": "Internet Performance",
        "component": "router",
        "failure_type": "intermittent_connectivity",
        "complaint": "My Wi-Fi keeps disconnecting intermittently.",
    },
}


for test_id, case in cases.items():

    print("\n" + "=" * 80)
    print(test_id)
    print("=" * 80)

    print("Complaint:")
    print(case["complaint"])

    results = retriever.search(
        domain=case["domain"],
        component=case["component"],
        failure_type=case["failure_type"],
        complaint_text=case["complaint"],
        top_k=5,
    )

    print("\nRetrieved KBs:")

    for i, doc in enumerate(results, 1):
        print(
            f"{i}. {doc.get('id')} - {doc.get('title')}"
            f"\n   Domain: {doc.get('domain')}"
            f"\n   Component: {doc.get('component')}"
            f"\n   Failure Type: {doc.get('failure_type')}"
            f"\n   Score: {doc.get('score', 'N/A')}"
        )