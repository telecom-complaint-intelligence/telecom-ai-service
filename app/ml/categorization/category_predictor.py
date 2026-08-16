import os
import torch
from typing import Tuple, Dict, Any

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
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

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

def _fallback_category_prediction(text: str) -> Tuple[str, float]:
    t = text.lower()
    if any(k in t for k in ["tower", "bursted", "outage", "blackout", "infrastructure", "fiber", "cable cut"]):
        return "Network & Infrastructure", 0.85
    elif any(k in t for k in ["speed", "slow", "lag", "streaming", "cap", "bandwidth"]):
        return "Internet Speeds & Performance", 0.82
    elif any(k in t for k in ["payment", "billing", "bill", "disconnect", "charged"]):
        return "Billing & Payment", 0.80
    elif any(k in t for k in ["router", "modem", "ont", "wifi", "device", "power"]):
        return "Equipment & Devices", 0.80
    return "Other", 0.50

def predict_category(text: str) -> Tuple[str, float]:
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

        if top_values.shape[1] > 1:
            top2_confidence = float(top_values[0, 1].item())
        else:
            top2_confidence = 0.0

        margin = top1_confidence - top2_confidence

        if top1_category == "Other":
            final_category = "Other"
        elif top1_confidence < CONFIDENCE_THRESHOLD:
            final_category = "Other"
        elif margin < MARGIN_THRESHOLD:
            final_category = "Other"
        else:
            final_category = top1_category

        return final_category, round(top1_confidence, 4)
    except Exception as e:
        print(f"Error during category prediction: {e}")
        return _fallback_category_prediction(text)

def analyze_complaint_category(text: str) -> Dict[str, Any]:
    category, confidence = predict_category(text)
    return {
        "category": category,
        "category_confidence": confidence
    }
