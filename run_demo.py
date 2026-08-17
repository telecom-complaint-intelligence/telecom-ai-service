import json
from app.models.categorization.category_predictor import predict_category
from app.models.sentiment.sentiment_scorer import analyze_complaint_negativity
from app.extraction.service import analyze_complaint

dataset_tickets = [
    {
        "Ticket #": "250635",
        "Customer Complaint": "Comcast Cable Internet Speeds",
        "Received Via": "Customer Care Call",
        "City": "abingdon",
        "State": "maryland",
        "Zip code": "21009",
        "Date_parsed": "2015-04-22",
        "Time_parsed": "15:53:50",
        "Status": 0
    },
    {
        "Ticket #": "223441",
        "Customer Complaint": "the network tower near my house got bursted",
        "Received Via": "Internet",
        "City": "acworth",
        
        "State": "georgia",
        "Zip code": "30102",
        "Date_parsed": "2015-08-04",
        "Time_parsed": "10:22:56",
        "Status": 0
    },
    {
        "Ticket #": "242732",
        "Customer Complaint": "Speed and Service",
        "Received Via": "Internet",
        "City": "acworth",
        "State": "georgia",
        "Zip code": "30101",
        "Date_parsed": "2015-04-18",
        "Time_parsed": "09:55:47",
        "Status": 0
    },
    {
        "Ticket #": "277946",
        "Customer Complaint": "Comcast Imposed a New Usage Cap of 300GB that punishes streaming.",
        "Received Via": "Internet",
        "City": "acworth",
        "State": "georgia",
        "Zip code": "30101",
        "Date_parsed": "2015-07-05",
        "Time_parsed": "11:59:35",
        "Status": 1
    }
]

print("\n" + "=" * 80)
print("TELECOM AI SERVICE - CATEGORIZATION, PIPELINE & SENTIMENT DEMO")
print("=" * 80 + "\n")

for item in dataset_tickets:
    ticket_num = item["Ticket #"]
    complaint_text = item["Customer Complaint"]
    location = f"{item['City']}, {item['State']} ({item['Zip code']})"

    print(f"[Ticket #{ticket_num}] Channel: {item['Received Via']} | Location: {location} | Status: {item['Status']}")
    print(f"Complaint: \"{complaint_text}\"")

    # 1. DistilBERT Category Prediction
    cat, cat_conf = predict_category(complaint_text)
    print(f" -> Category: \"{cat}\" (Confidence: {cat_conf * 100:.2f}%)")

    # 2. Sentiment & Negativity Score * 15
    sentiment = analyze_complaint_negativity(complaint_text)
    print(f" -> Negativity Score: {sentiment['negativity_score']:.4f} | Weighted Score (*15): {sentiment['weighted_negativity_score']:.4f}")

    # 3. Technical Extraction & Complexity
    pipeline_res = analyze_complaint(complaint_text)
    comp = pipeline_res["technical_complexity"]
    tech = pipeline_res["technical_information"]

    print(f" -> Technical Info: component={tech['component']}, failure_type={tech['failure_type']}, scope={tech['scope']}, impact={tech['service_impact']}")
    print(f" -> Technical Complexity: {comp['complexity']} (Score: {comp['complexity_score']}/100)")
    print("-" * 80 + "\n")
