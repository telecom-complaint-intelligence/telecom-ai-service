import json
from app.aggregator.aggregator import aggregate_complaint_features

dataset_tickets = [
    {
        "Ticket #": "250635",
        "Customer Complaint": "account is not working properly and I cannot log in",
        "Received Via": "Mobile App",
        "Expected Category": "Account",
    },
    {
        "Ticket #": "223441",
        "Customer Complaint": "the network tower near my house got bursted and area has no signal for 4 days",
        "Received Via": "Internet",
        "Expected Category": "Internet / Connectivity",
    },
    {
        "Ticket #": "242732",
        "Customer Complaint": "wifi router light is blinking red and internet is offline",
        "Received Via": "Customer Care",
        "Expected Category": "Equipment / Router",
    },
    {
        "Ticket #": "277946",
        "Customer Complaint": "I was charged twice on my monthly bill invoice for March",
        "Received Via": "Email",
        "Expected Category": "Billing / Payment",
    },
    {
        "Ticket #": "299102",
        "Customer Complaint": "Where is your nearest branch office located?",
        "Received Via": "Web Portal",
        "Expected Category": "Other",
    },
]

print("\n" + "=" * 80)
print("TELECOM AI SERVICE - END-TO-END DEMO EXECUTION")
print("=" * 80 + "\n")

for item in dataset_tickets:
    ticket_num = item["Ticket #"]
    complaint_text = item["Customer Complaint"]

    print(f"================== [Ticket #{ticket_num}] Channel: {item['Received Via']} ==================")
    print(f"Complaint: \"{complaint_text}\"")

    res = aggregate_complaint_features(complaint_text)

    print(f" -> Category: \"{res['category']}\" (Confidence: {res['category_confidence'] * 100:.2f}%)")
    print(f" -> Negativity: {res['negativity_score']:.4f} | Total Complexity Score: {res['total_complexity_score']}/100")
    print(f" -> Severity/Complexity: {res['complexity']} (Base: {res.get('base_complexity')})")
    print(f" -> Decision Reason: {res.get('decision_reason')}")
    
    if res.get("solution_a"):
        print(f" -> Solution / Instructions:\n{res['solution_a']}")
    if res.get("solution_high"):
        print(f" -> High-Agent Action Plan:\n{res['solution_high']}")
    if res.get("diagnosis"):
        print(f" -> Diagnosis: {res['diagnosis']} | Final Decision: {res['final_decision']}")
    print("-" * 80 + "\n")

