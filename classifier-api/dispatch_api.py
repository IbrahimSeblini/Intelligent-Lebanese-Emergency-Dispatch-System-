from __future__ import annotations

import os
import uuid

from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)
CORS(app)

classifier_calls = Counter(
    "classifier_calls_total", "Total calls to the classifier API"
)
classifier_calls_by_category = Counter(
    "classifier_calls_by_category_total",
    "Classifier calls broken down by category",
    ["category"],
)
classifier_errors = Counter(
    "classifier_errors_total", "Total classifier errors"
)

CLASSIFIER_MODE = os.environ.get("CLASSIFIER_MODE", "lite")
MAX_TEXT_LEN = 16000
MAX_PHONE_LEN = 64

_classifier = None


def get_classifier():
    global _classifier
    if _classifier is None:
        if CLASSIFIER_MODE == "full":
            from hybrid_emergency import get_zero_shot_classifier
            _classifier = get_zero_shot_classifier()
        else:
            import joblib
            _classifier = joblib.load(
                os.path.join(os.path.dirname(__file__), "emergency_classifier.pkl")
            )
    return _classifier


def category_to_services(category: str) -> list[str]:
    mapping = {
        "police": ["POLICE"],
        "firefighters": ["FIRE"],
        "red_cross": ["MEDICAL"],
        "uncertain": ["POLICE", "FIRE", "MEDICAL"],
    }
    return mapping.get(category, ["POLICE", "FIRE", "MEDICAL"])


def infer_priority(text: str, category: str, confidence: float) -> str:
    t = text.lower()
    urgent = (
        "not breathing",
        "unconscious",
        "shooting",
        "stabbed",
        "can't breathe",
        "cannot breathe",
        "collapsed",
        "trapped",
        "gas leak",
        "explosion",
        "حريق",
        "نار",
        "غمية",
    )
    if any(w in t for w in urgent):
        return "CRITICAL"
    if category == "uncertain" or confidence < 0.45:
        return "HIGH"
    return "MEDIUM"


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.post("/api/dispatch")
def dispatch():
    from hybrid_emergency import classify_with_details

    classifier_calls.inc()
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    if not text:
        classifier_errors.inc()
        return jsonify({"error": "text is required"}), 400
    if len(text) > MAX_TEXT_LEN:
        classifier_errors.inc()
        return jsonify({"error": "text is too long"}), 400

    conv_id = payload.get("conversation_id") or str(uuid.uuid4())
    caller_phone = payload.get("caller_phone")
    if caller_phone is not None:
        caller_phone = str(caller_phone).strip()
        if len(caller_phone) > MAX_PHONE_LEN:
            classifier_errors.inc()
            return jsonify({"error": "caller_phone is too long"}), 400
        if not caller_phone:
            caller_phone = None
    call_sid = payload.get("call_sid")
    allow_uncertain = payload.get("allow_uncertain") is True
    threshold = float(os.environ.get("DISPATCH_UNCERTAIN_THRESHOLD", "0.5"))

    try:
        result = classify_with_details(
            text,
            classifier=get_classifier(),
            threshold=threshold,
            force_label=not allow_uncertain,
            fallback_phone=caller_phone,
            mode=CLASSIFIER_MODE,
        )
    except Exception as e:
        classifier_errors.inc()
        return jsonify({"error": "classification failed", "detail": str(e)}), 500

    cat = result["category"]
    classifier_calls_by_category.labels(category=cat).inc()
    services = category_to_services(cat)
    conf = float(result.get("confidence") or 0.0)
    priority = infer_priority(text, cat, conf)
    contact = result.get("contact") or {}
    location = (contact.get("location") or "").strip() or None
    phone = contact.get("phone")
    name = contact.get("name")
    emergency_text = f"{name} — {text}" if name else text

    body = {
        "conversation_id": conv_id,
        "classification": {
            "category": cat,
            "services": services,
            "confidence": result.get("confidence"),
            "source": result.get("source"),
            "priority": priority,
        },
        "contact": contact,
        "missing_fields": result.get("missing_fields"),
    }
    return jsonify(body), 200


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5050"))
    app.run(host="0.0.0.0", port=port, debug=False)
