from typing import Dict, Any
from app.models.categorization.category_predictor import predict_category
from app.models.sentiment.sentiment_scorer import measure_negativity
from app.extraction.hybrid_extractor import extract_technical_information
from app.priority.complexity import calculate_complexity, calculate_total_complexity

def aggregate_complaint_features(complaint_text: str) -> Dict[str, Any]:
    """
    Feature Aggregator (F.A.) — Merges BERT/DistilBERT Category + RoBERTa Sentiment + 
    Information Extraction + Priority Complexity engine into a single feature representation.
    """
    # 1. DistilBERT Category Prediction
    category, category_confidence = predict_category(complaint_text)

    # 2. RoBERTa Sentiment Negativity Scorer
    negativity_score = measure_negativity(complaint_text)

    # 3. Information Extraction (ML / LangGraph Agent)
    extraction_result = extract_technical_information(complaint_text)
    tech_info = extraction_result["technical_information"]
    extraction_source = extraction_result["extraction_source"]
    lowest_confidence = extraction_result["lowest_confidence"]

    # 4. Priority & Complexity Engine
    complexity_result = calculate_complexity(tech_info)

    # 5. Total Complexity Calculation (85% Complexity + 15% Sentiment Negativity)
    total_complexity = calculate_total_complexity(
        complexity_result["complexity_score"],
        negativity_score
    )

    return {
        "complaint": complaint_text,
        "category": category,
        "category_confidence": category_confidence,
        "negativity_score": negativity_score,
        "sentiment_score": total_complexity["sentiment_score"],
        "extraction_source": extraction_source,
        "lowest_confidence": lowest_confidence,
        "technical_information": tech_info,
        "complexity": complexity_result["complexity"],
        "complexity_score": complexity_result["complexity_score"],
        "base_complexity": complexity_result.get("base_complexity"),
        "modifier": complexity_result.get("modifier", 0),
        "critical_override": complexity_result.get("critical_override", False),
        "decision_reason": complexity_result.get("decision_reason"),
        "weighted_complexity_score": total_complexity["weighted_complexity_score"],
        "weighted_negativity_score": total_complexity["weighted_negativity_score"],
        "total_complexity_score": total_complexity["total_complexity_score"],
    }
