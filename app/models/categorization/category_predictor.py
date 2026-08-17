
import os
from typing import Any

import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "Model", "distilbert_category_model")

MAX_LENGTH = 128
CONFIDENCE_THRESHOLD = 0.45
MARGIN_THRESHOLD = 0.10

_tokenizer = None
_model = None
_load_attempted = False
_device = None

def _load_categorization_model():
    global _tokenizer, _model, _device, _load_attempted
    if _load_attempted:
        return _tokenizer, _model, _device

    _load_attempted = True
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if os.path.exists(MODEL_PATH):
            print(f"Loading trained DistilBERT category model from '{MODEL_PATH}'...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
            model.to(device)
            model.eval()

            _tokenizer = tokenizer
            _model = model
            _device = device
            print("DistilBERT category model loaded successfully!")
        else:
            print(f"Notice: Model path '{MODEL_PATH}' not found. Using fallback categorization.")
    except Exception as e:
        print(f"Warning: Could not load DistilBERT model ({e}). Using fallback categorization.")

    return _tokenizer, _model, _device

def _fallback_category_prediction(text: str) -> tuple[str, float]:
    t = text.lower()
    if any(k in t for k in ["account", "login", "password", "profile", "portal", "credentials", "username", "access"]):
        return "Account", 0.85
    elif any(k in t for k in ["billing", "payment", "bill", "invoice", "charged", "refund", "receipt", "due"]):
        return "Billing / Payment", 0.85
    elif any(k in t for k in ["cancel", "cancellation", "terminate", "close account", "unsubscribe"]):
        return "Cancellation", 0.85
    elif any(k in t for k in ["agent", "support", "representative", "helpdesk", "call back", "executive"]):
        return "Customer Support", 0.80
    elif any(k in t for k in ["router", "modem", "ont", "wifi", "device", "power", "hardware", "adapter"]):
        return "Equipment / Router", 0.85
    elif any(k in t for k in ["install", "installation", "setup", "technician visit", "new connection", "wiring"]):
        return "Installation", 0.85
    elif any(k in t for k in ["internet", "connectivity", "speed", "slow", "down", "outage", "fiber", "cable cut", "tower", "network", "data", "disconnect"]):
        return "Internet / Connectivity", 0.85
    elif any(k in t for k in ["plan", "upgrade", "downgrade", "subscription", "package", "tariff", "recharge", "validity"]):
        return "Service / Plan", 0.85
    return "Other", 0.50

def predict_category(text: str) -> tuple[str, float]:
    if not isinstance(text, str) or not text.strip():
        return "Other", 0.0

    tokenizer, model, device = _load_categorization_model()

    if tokenizer is None or model is None:
        return _fallback_category_prediction(text)

    try:
        inputs = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        inputs.pop("token_type_ids", None)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)

        top_values, top_indices = torch.topk(probabilities, k=min(2, probabilities.shape[1]), dim=1)

        top1_id = top_indices[0, 0].item()
        top1_confidence = float(top_values[0, 0].item())

        top1_category = model.config.id2label.get(top1_id, "Other")

        top2_confidence = float(top_values[0, 1].item()) if top_values.shape[1] > 1 else 0.0

        margin = top1_confidence - top2_confidence

        if top1_category == "Other" or top1_confidence < CONFIDENCE_THRESHOLD or margin < MARGIN_THRESHOLD:
            final_category = "Other"
        else:
            final_category = top1_category

        return final_category, round(top1_confidence, 4)
    except Exception as e:
        print(f"Error during category prediction: {e}")
        return _fallback_category_prediction(text)

def analyze_complaint_category(text: str) -> dict[str, Any]:
    category, confidence = predict_category(text)
    return {
        "category": category,
        "category_confidence": confidence
    }
