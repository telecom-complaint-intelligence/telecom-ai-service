import re
from typing import Dict, Any

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

_tokenizer = None
_model = None
_negative_index = None
_load_attempted = False

def _load_sentiment_model():
    global _tokenizer, _model, _negative_index, _load_attempted
    if _load_attempted:
        return _tokenizer, _model, _negative_index

    _load_attempted = True
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.eval()

        neg_idx = None
        for index, label in model.config.id2label.items():
            if str(label).lower() == "negative":
                neg_idx = int(index)
                break

        if neg_idx is not None:
            _tokenizer = tokenizer
            _model = model
            _negative_index = neg_idx
    except Exception as e:
        print(f"Notice: Using rule-based negativity scorer fallback ({e}).")

    return _tokenizer, _model, _negative_index

def _fallback_negativity_score(complaint: str) -> float:
    text = complaint.lower()
    neg_words = [
        "not working", "broken", "damaged", "outage", "down", "slow",
        "horrible", "terrible", "worst", "fail", "failed", "unusable",
        "cant", "cannot", "issue", "problem", "disaster", "cut", "no internet"
    ]
    matches = sum(1 for word in neg_words if word in text)
    score = min(1.0, 0.3 + (matches * 0.2))
    return round(score, 4)

def measure_negativity(complaint: str) -> float:
    if not isinstance(complaint, str) or not complaint.strip():
        raise ValueError("Please enter a valid complaint.")

    tokenizer, model, negative_index = _load_sentiment_model()

    if tokenizer is not None and model is not None and negative_index is not None:
        import torch
        inputs = tokenizer(
            complaint,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )
        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=-1)[0]
        negative_score = probabilities[negative_index].item()
        negative_score = max(0.0, min(1.0, negative_score))
        return round(float(negative_score), 4)

    return _fallback_negativity_score(complaint)

def analyze_complaint_negativity(complaint: str) -> Dict[str, Any]:
    score = measure_negativity(complaint)
    weighted_score = round(score * 15.0, 4)
    return {
        "complaint": complaint,
        "negativity_score": score,
        "weighted_negativity_score": weighted_score,
        "multiplier": 15.0
    }
