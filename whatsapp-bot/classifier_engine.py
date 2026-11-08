from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

DEFAULT_PRETRAINED_NLI_MODEL = "facebook/bart-large-mnli"
DEFAULT_MULTILINGUAL_NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
DEFAULT_HYPOTHESIS_TEMPLATE = "This message requires {}."

CANDIDATE_TO_CATEGORY = {
    "police intervention for crime or personal threat": "police",
    "medical emergency requiring ambulance or Red Cross": "red_cross",
    "fire emergency with smoke, flames, or gas leak": "firefighters",
    "unclear request for help without enough details": "uncertain",
}
FORCED_CANDIDATE_TO_CATEGORY = {
    k: v for k, v in CANDIDATE_TO_CATEGORY.items() if v != "uncertain"
}

NAME_PATTERNS = (
    r"\bmy name is\s+([a-zA-Z][a-zA-Z\s]{1,30})",
    r"\bthis is\s+([a-zA-Z][a-zA-Z\s]{1,30})",
    r"\bana\s+([a-zA-Z][a-zA-Z\s]{1,30})",
    r"\bانا\s+([\u0600-\u06FF][\u0600-\u06FF\s]{1,30})",
    r"\bاسمي\s+([\u0600-\u06FF][\u0600-\u06FF\s]{1,30})",
)
LOCATION_PATTERNS = (
    r"\b(?:in|at|near|around)\s+([a-zA-Z][a-zA-Z0-9\-\s]{2,40})",
    r"\b(?:bi|bel|b)\s+([a-zA-Z0-9\u0600-\u06FF][a-zA-Z0-9\u0600-\u06FF\-\s]{1,40})",
    r"\b(?:في|بالقرب من|جنب|حد)\s+([\u0600-\u06FF][\u0600-\u06FF\s]{1,40})",
)
PHONE_PATTERN = r"(?:\+?\d[\d\-\s]{6,}\d)"

_cached_zero_shot_classifier: Any = None


def _normalize_text(text: str) -> str:
    t = " ".join(text.lower().strip().split())
    return t.replace("2", "a").replace("5", "kh").replace("8", "gh")


def _looks_arabic_or_arabizi(text: str) -> bool:
    t = text.lower()
    if re.search(r"[\u0600-\u06FF]", t):
        return True
    arabizi_markers = (
        "3",
        "7",
        "6",
        "9",
        "fi ",
        "bel ",
        "bi ",
        "harame",
        "nar",
        "dukhen",
        "yetnafas",
    )
    return any(marker in t for marker in arabizi_markers)


def _rule_based_arabic_hint(text: str) -> Optional[str]:
    t = _normalize_text(text)
    police_terms = ("harame", "ser2a", "sar2a", "fet 3ala bayte", "سرقة", "حرامي", "اقتحم")
    fire_terms = ("nar", "7are2", "hare2", "dukhen", "gas leak", "حريق", "نار", "دخان")
    medical_terms = (
        "ma 3am yetnafas",
        "mesh 3am yetnafas",
        "wa2a3",
        "not breathing",
        "ambulance",
        "اسعاف",
        "ما عم يتنفس",
        "وقع",
    )
    if any(term in t for term in police_terms):
        return "police"
    if any(term in t for term in fire_terms):
        return "firefighters"
    if any(term in t for term in medical_terms):
        return "red_cross"
    return None


def get_zero_shot_classifier():
    global _cached_zero_shot_classifier
    if _cached_zero_shot_classifier is not None:
        return _cached_zero_shot_classifier

    from transformers import pipeline

    model = (
        os.getenv("CLASSIFIER_ZS_MODEL_MULTI", DEFAULT_MULTILINGUAL_NLI_MODEL)
        if os.getenv("CLASSIFIER_PREFER_MULTI", "0").strip().lower() in {"1", "true", "yes"}
        else os.getenv("CLASSIFIER_ZS_MODEL", DEFAULT_PRETRAINED_NLI_MODEL)
    )
    _cached_zero_shot_classifier = pipeline("zero-shot-classification", model=model)
    return _cached_zero_shot_classifier


def classify_emergency_zero_shot(
    text: str,
    *,
    classifier=None,
    uncertain_threshold: float = 0.6,
    force_label: bool = False,
) -> Dict[str, Any]:
    rule_label = _rule_based_arabic_hint(text)
    if rule_label is not None:
        return {
            "category": rule_label,
            "confidence": 0.95,
            "source": "arabic_keyword_rule",
        }

    labels = (
        list(FORCED_CANDIDATE_TO_CATEGORY.keys())
        if force_label
        else list(CANDIDATE_TO_CATEGORY.keys())
    )
    candidate_map = FORCED_CANDIDATE_TO_CATEGORY if force_label else CANDIDATE_TO_CATEGORY
    clf = classifier or get_zero_shot_classifier()

    result = clf(
        text,
        labels,
        hypothesis_template=DEFAULT_HYPOTHESIS_TEMPLATE,
        multi_label=False,
    )
    best_candidate = result["labels"][0]
    confidence = float(result["scores"][0])
    best_label = candidate_map.get(best_candidate, "uncertain")
    if (not force_label) and best_label != "uncertain" and confidence < uncertain_threshold:
        best_label = "uncertain"
    return {
        "category": best_label,
        "confidence": round(confidence, 3),
        "source": "pretrained_nli_multi" if _looks_arabic_or_arabizi(text) else "pretrained_nli",
    }


def extract_contact_fields(
    text: str,
    *,
    fallback_phone: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    normalized = " ".join(text.strip().split())

    name = None
    for pat in NAME_PATTERNS:
        match = re.search(pat, normalized, flags=re.IGNORECASE)
        if match:
            name = match.group(1).strip(" ,.-")
            break

    location = None
    for pat in LOCATION_PATTERNS:
        match = re.search(pat, normalized, flags=re.IGNORECASE)
        if match:
            location = match.group(1).strip(" ,.-")
            break

    phone_match = re.search(PHONE_PATTERN, normalized)
    phone = phone_match.group(0).strip() if phone_match else (fallback_phone or None)
    return {"name": name, "location": location, "phone": phone}


def classify_with_details(
    text: str,
    *,
    classifier=None,
    threshold: float = 0.6,
    force_label: bool = False,
    fallback_phone: Optional[str] = None,
) -> Dict[str, Any]:
    classification = classify_emergency_zero_shot(
        text,
        classifier=classifier,
        uncertain_threshold=threshold,
        force_label=force_label,
    )
    contact = extract_contact_fields(text, fallback_phone=fallback_phone)
    missing_fields = [k for k in ("name", "location") if not contact.get(k)]
    return {
        "category": classification["category"],
        "confidence": classification["confidence"],
        "source": classification["source"],
        "contact": contact,
        "missing_fields": missing_fields,
    }
