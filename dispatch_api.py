"""
HTTP API: classify emergency text with hybrid_emergency and insert a row into
Supabase `calls` so the realtime dashboards receive the incident.

Environment (optional overrides):
  SUPABASE_URL, SUPABASE_ANON_KEY — default to the same project as dashboard/dispatch-dashboard.html
  PORT — default 5050
  DISPATCH_UNCERTAIN_THRESHOLD — passed to classifier when allow_uncertain is used

Run: python dispatch_api.py
"""

from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime, timezone

import requests

# Shared classifier logic lives in classifier-api/
_ROOT = os.path.dirname(os.path.abspath(__file__))
_CLASSIFIER_DIR = os.path.join(_ROOT, "classifier-api")
if _CLASSIFIER_DIR not in sys.path:
    sys.path.insert(0, _CLASSIFIER_DIR)
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Local-dev defaults (same Supabase project as the HTML dashboards). Override with env vars.
_DEFAULT_SUPABASE_URL = "https://botkfihiheybseanehwa.supabase.co"
_DEFAULT_SUPABASE_ANON_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJvdGtmaWhpaGV5YnNlYW5laHdhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzY2MDgxMTMsImV4cCI6MjA5MjE4NDExM30.L1fRBbfSnl8k5ABK_9zwz9RcXhUmBtK1126CyW-jLnI"
)

_zs_classifier = None


def _supabase_url_and_key() -> tuple[str, str]:
    base = os.environ.get("SUPABASE_URL", _DEFAULT_SUPABASE_URL).rstrip("/")
    key = os.environ.get("SUPABASE_ANON_KEY", _DEFAULT_SUPABASE_ANON_KEY)
    return base, key


def get_classifier():
    global _zs_classifier
    if _zs_classifier is None:
        import joblib

        pkl = os.path.join(_CLASSIFIER_DIR, "emergency_classifier.pkl")
        if not os.path.isfile(pkl):
            raise FileNotFoundError(
                f"Missing {pkl}. Run: python classifier-api/train_model.py"
            )
        _zs_classifier = joblib.load(pkl)
    return _zs_classifier


def category_to_services(category: str) -> list[str]:
    mapping = {
        "police": ["POLICE"],
        "firefighters": ["FIRE"],
        "red_cross": ["MEDICAL"],
        # Route to all three boards for manual triage
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


def insert_supabase_row(row: dict) -> tuple[bool, str | None]:
    base, key = _supabase_url_and_key()
    if not base or not key:
        return False, "SUPABASE_URL or SUPABASE_ANON_KEY not set"

    resp = requests.post(
        f"{base}/rest/v1/calls",
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        },
        json=row,
        timeout=20,
    )
    if resp.status_code not in (200, 201):
        return False, resp.text or resp.reason
    return True, None


@app.post("/api/dispatch")
def dispatch():
    from hybrid_emergency import classify_with_details

    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400

    conv_id = payload.get("conversation_id") or str(uuid.uuid4())
    caller_phone = payload.get("caller_phone")
    call_sid = payload.get("call_sid")
    allow_uncertain = payload.get("allow_uncertain") is True
    threshold = float(os.environ.get("DISPATCH_UNCERTAIN_THRESHOLD", "0.5"))

    result = classify_with_details(
        text,
        classifier=get_classifier(),
        threshold=threshold,
        force_label=not allow_uncertain,
        fallback_phone=caller_phone,
    )
    cat = result["category"]
    services = category_to_services(cat)
    conf = float(result.get("confidence") or 0.0)
    priority = infer_priority(text, cat, conf)
    contact = result.get("contact") or {}
    location = (contact.get("location") or "").strip() or None
    phone = contact.get("phone")
    name = contact.get("name")
    emergency_text = f"{name} — {text}" if name else text

    row = {
        "conversation_id": conv_id,
        "call_sid": call_sid or phone or conv_id[:12],
        "services": services,
        "priority": priority,
        "location": location,
        "emergency": emergency_text,
        "from_number": phone,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "active",
    }

    stored, err = insert_supabase_row(row)
    body = {
        "stored": stored,
        "supabase_error": err,
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
    return jsonify(body), (201 if stored else 200)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5050"))
    app.run(host="0.0.0.0", port=port, debug=False)
