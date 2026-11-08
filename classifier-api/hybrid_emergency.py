"""
Hybrid classifier: trust the trained sklearn model when it is confident (> threshold),
otherwise fall back to a stronger pretrained multilingual NLI classifier.

Requires: pip install transformers torch

Default pretrained model:
  facebook/bart-large-mnli

Override model (optional):
  HYBRID_ZS_MODEL=typeform/distilbert-base-uncased-mnli
"""

from __future__ import annotations

import os
import re
from typing import Any

from sklearn.pipeline import Pipeline

# Strong pretrained models for translation and zero-shot classification.
DEFAULT_AR_EN_TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-ar-en"
DEFAULT_FR_EN_TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-fr-en"
DEFAULT_PRETRAINED_NLI_MODEL = "facebook/bart-large-mnli"
DEFAULT_MULTILINGUAL_NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

# Descriptive candidate labels improve NLI performance vs bare class IDs.
CANDIDATE_TO_CATEGORY = {
    "police intervention for crime or personal threat": "police",
    "medical emergency requiring ambulance or Red Cross": "red_cross",
    "fire emergency with smoke, flames, or gas leak": "firefighters",
    "unclear request for help without enough details": "uncertain",
}
FORCED_CANDIDATE_TO_CATEGORY = {
    k: v for k, v in CANDIDATE_TO_CATEGORY.items() if v != "uncertain"
}
DEFAULT_HYPOTHESIS_TEMPLATE = "This message requires {}."

_zs_pipelines: dict[str, Any] = {}
_translation_models: dict[str, tuple[Any, Any]] = {}

LOCATION_PATTERNS = (
    r"\b(?:in|at|near|around)\s+([a-zA-Z][a-zA-Z0-9\-\s]{2,40})",
    r"\b(?:bi|bel|b)\s+([a-zA-Z0-9\u0600-\u06FF][a-zA-Z0-9\u0600-\u06FF\-\s]{1,40})",
    r"\b(?:في|بالقرب من|جنب|حد)\s+([\u0600-\u06FF][\u0600-\u06FF\s]{1,40})",
)
NAME_PATTERNS = (
    r"\bmy name is\s+([a-zA-Z][a-zA-Z\s]{1,30})",
    r"\bthis is\s+([a-zA-Z][a-zA-Z\s]{1,30})",
    r"\bana\s+([a-zA-Z][a-zA-Z\s]{1,30})",
    r"\bانا\s+([\u0600-\u06FF][\u0600-\u06FF\s]{1,30})",
    r"\bاسمي\s+([\u0600-\u06FF][\u0600-\u06FF\s]{1,30})",
)
PHONE_PATTERN = r"(?:\+?\d[\d\-\s]{6,}\d)"


def _normalize_text(text: str) -> str:
    t = " ".join(text.lower().strip().split())
    # normalize common Arabizi variants
    t = t.replace("2", "a").replace("5", "kh").replace("8", "gh")
    return t


def _looks_arabic_or_arabizi(text: str) -> bool:
    t = text.lower()
    if re.search(r"[\u0600-\u06FF]", t):
        return True
    arabizi_markers = (
        "3", "7", "6", "9",
        "fi ", "bel ", "bi ", "hada", "hadan", "harame", "nar", "dukhen", "yetnafas",
    )
    return any(marker in t for marker in arabizi_markers)


def _has_arabic_script(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text))


def _looks_french_text(text: str) -> bool:
    t = text.lower()
    if re.search(r"[àâçéèêëîïôûùüÿœæ]", t):
        return True
    # lightweight heuristic to avoid extra dependencies
    french_markers = (
        " le ", " la ", " les ", " un ", " une ", " des ",
        " je ", " suis ", " est ", " pas ", " avec ",
        " au ", " aux ", " dans ", " merci ", " urgence ",
        " feu ", " police ", " ambulance ",
    )
    padded = f" {t} "
    marker_hits = sum(1 for marker in french_markers if marker in padded)
    return marker_hits >= 2


def _rule_based_arabic_hint(text: str) -> str | None:
    """
    Quick keyword hints for Arabic/Arabizi emergency phrases.
    Returns one of: police, firefighters, red_cross, or None.
    """
    t = _normalize_text(text)

    police_terms = (
        "harame",
        "7arame",
        "ser2a",
        "sar2a",
        "sari2a",
        "fet 3a bayte",
        "fet 3al bayt",
        "da5al 3al beit",
        "3am y7awel yfot",
        "fet 3ala bayte",
        "haddan fet 3al beit",
        "سرقة",
        "حرامي",
        "اقتحم",
    )
    fire_terms = (
        "nar",
        "7are2",
        "hare2",
        "7ari2",
        "7ri2",
        "dukhen",
        "tesrib gaz",
        "gas leak",
        "حريق",
        "نار",
        "دخان",
        "تسرب غاز",
    )
    medical_terms = (
        "ma 3am yetnafas",
        "mesh 3am yetnafas",
        "mish 3am yetnafas",
        "wa2a3",
        "غمية",
        "غمي",
        "ma byetnafas",
        "saket",
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


def extract_contact_fields(
    text: str,
    *,
    fallback_phone: str | None = None,
) -> dict[str, str | None]:
    """Extract name/location/phone from free text with lightweight regexes."""
    normalized = " ".join(text.strip().split())

    name = None
    for pat in NAME_PATTERNS:
        m = re.search(pat, normalized, flags=re.IGNORECASE)
        if m:
            name = m.group(1).strip(" ,.-")
            break

    location = None
    for pat in LOCATION_PATTERNS:
        m = re.search(pat, normalized, flags=re.IGNORECASE)
        if m:
            location = m.group(1).strip(" ,.-")
            break

    phone_match = re.search(PHONE_PATTERN, normalized)
    phone = phone_match.group(0).strip() if phone_match else (fallback_phone or None)

    return {
        "name": name,
        "location": location,
        "phone": phone,
    }


def classify_emergency_lite(
    text: str,
    *,
    sklearn_model=None,
    force_label: bool = True,
    threshold: float = 0.6,
) -> dict:
    """Rules + sklearn only — no PyTorch or HuggingFace required."""
    rule_label = _rule_based_arabic_hint(text)
    if rule_label is not None:
        return {
            "category": rule_label,
            "confidence": 0.95,
            "source": "arabic_keyword_rule",
            "matched_candidate": "keyword_match",
            "classification_text": text,
            "translated_text": None,
        }

    if sklearn_model is not None:
        label, conf, _ = get_ml_raw_prediction(sklearn_model, text)
        if force_label or conf >= threshold:
            return {
                "category": label,
                "confidence": round(conf, 3),
                "source": "ml",
                "matched_candidate": None,
                "classification_text": text,
                "translated_text": None,
            }

    return {
        "category": "uncertain",
        "confidence": 0.0,
        "source": "lite_fallback",
        "matched_candidate": None,
        "classification_text": text,
        "translated_text": None,
    }


def classify_with_details(
    text: str,
    *,
    classifier=None,
    threshold: float = 0.6,
    force_label: bool = True,
    fallback_phone: str | None = None,
    mode: str = "full",
) -> dict:
    """
    Classify category and extract actionable dispatch fields.
    mode='lite' uses rules + sklearn only (no HuggingFace).
    mode='full' uses the zero-shot NLI pipeline.
    """
    if mode == "lite":
        cls = classify_emergency_lite(
            text,
            sklearn_model=classifier,
            force_label=force_label,
            threshold=threshold,
        )
    else:
        cls = classify_emergency_zero_shot(
            text,
            classifier=classifier,
            uncertain_threshold=threshold,
            force_label=force_label,
        )
    fields = extract_contact_fields(text, fallback_phone=fallback_phone)
    missing_fields = [k for k in ("name", "location") if not fields.get(k)]
    return {
        "text": text,
        "classification_text": cls.get("classification_text", text),
        "category": cls["category"],
        "confidence": cls["confidence"],
        "source": cls["source"],
        "matched_candidate": cls.get("matched_candidate"),
        "translated_text": cls.get("translated_text"),
        "contact": fields,
        "missing_fields": missing_fields,
    }


def build_missing_fields_prompt(missing_fields: list[str]) -> str | None:
    """
    Build a user-facing follow-up prompt depending on missing required fields.
    """
    needed = set(missing_fields)
    if {"name", "location"}.issubset(needed):
        return "Please provide your name and your location."
    if "name" in needed:
        return "Please provide your name."
    if "location" in needed:
        return "Please provide your location."
    return None


def get_translation_model(model_name: str | None = None):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    resolved = model_name or os.environ.get(
        "HYBRID_TRANSLATION_MODEL",
        DEFAULT_AR_EN_TRANSLATION_MODEL,
    )
    if resolved not in _translation_models:
        tokenizer = AutoTokenizer.from_pretrained(resolved)
        model = AutoModelForSeq2SeqLM.from_pretrained(resolved)
        _translation_models[resolved] = (tokenizer, model)
    return _translation_models[resolved]


def _translate_text(text: str, *, model_name: str, translator=None) -> str:
    """Translate text to English with fail-safe fallback."""
    model_bundle = translator or get_translation_model(model_name)
    try:
        tokenizer, model = model_bundle
        inputs = tokenizer(
            [text],
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=4,
        )
        translated = tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
    except Exception:
        return text
    return translated or text


def translate_arabic_to_english(text: str, *, translator=None) -> str:
    """Translate Arabic-script text to English; return original on failure."""
    if not _has_arabic_script(text):
        return text
    return _translate_text(text, model_name=DEFAULT_AR_EN_TRANSLATION_MODEL, translator=translator)


def translate_french_to_english(text: str, *, translator=None) -> str:
    """Translate French-like text to English; return original on failure."""
    if not _looks_french_text(text):
        return text
    return _translate_text(text, model_name=DEFAULT_FR_EN_TRANSLATION_MODEL, translator=translator)


def get_zero_shot_classifier(
    model_name: str | None = None,
    *,
    prefer_multilingual: bool = False,
):
    """Lazy-load pipelines and pick English vs multilingual model by context."""
    from transformers import pipeline

    if model_name:
        resolved = model_name
    else:
        if prefer_multilingual:
            resolved = os.environ.get("HYBRID_ZS_MODEL_MULTI", DEFAULT_MULTILINGUAL_NLI_MODEL)
        else:
            resolved = os.environ.get("HYBRID_ZS_MODEL", DEFAULT_PRETRAINED_NLI_MODEL)

    if resolved not in _zs_pipelines:
        _zs_pipelines[resolved] = pipeline(
            "zero-shot-classification",
            model=resolved,
        )
    return _zs_pipelines[resolved]


def get_ml_raw_prediction(model: Pipeline, text: str) -> tuple[str, float, dict[str, float]]:
    """Top sklearn class, its probability, and all class scores."""
    probs = model.predict_proba([text])[0]
    classes = model.classes_
    best_idx = int(probs.argmax())
    scores = {str(c): round(float(p), 4) for c, p in zip(classes, probs)}
    return str(classes[best_idx]), float(probs[best_idx]), scores


def classify_emergency_zero_shot(
    text: str,
    *,
    classifier=None,
    translator=None,
    labels: list[str] | None = None,
    hypothesis_template: str = DEFAULT_HYPOTHESIS_TEMPLATE,
    uncertain_threshold: float = 0.6,
    force_label: bool = False,
) -> dict:
    """Standalone pretrained NLI path with emergency-specific label prompts."""
    rule_label = _rule_based_arabic_hint(text)
    if rule_label is not None:
        return {
            "category": rule_label,
            "confidence": 0.95,
            "source": "arabic_keyword_rule",
            "matched_candidate": "keyword_match",
            "classification_text": text,
            "translated_text": None,
        }

    classification_text = text
    translated_text = None
    translated_from = None
    if _has_arabic_script(text):
        translated_text = translate_arabic_to_english(text, translator=translator)
        classification_text = translated_text
        translated_from = "ar"
    elif _looks_french_text(text):
        translated_text = translate_french_to_english(text, translator=translator)
        classification_text = translated_text
        translated_from = "fr"

    if labels is None:
        labels = (
            list(FORCED_CANDIDATE_TO_CATEGORY.keys())
            if force_label
            else list(CANDIDATE_TO_CATEGORY.keys())
        )
    candidate_map = (
        FORCED_CANDIDATE_TO_CATEGORY if force_label else CANDIDATE_TO_CATEGORY
    )
    prefer_multilingual = _looks_arabic_or_arabizi(text) and translated_text is None
    clf = classifier or get_zero_shot_classifier(prefer_multilingual=prefer_multilingual)
    result = clf(
        classification_text,
        labels,
        hypothesis_template=hypothesis_template,
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
        "source": f"pretrained_nli_translated_{translated_from}" if translated_text else "pretrained_nli",
        "matched_candidate": best_candidate,
        "classification_text": classification_text,
        "translated_text": translated_text,
    }


def classify_emergency_hybrid(
    model: Pipeline,
    text: str,
    *,
    ml_confidence_threshold: float = 0.7,
    zs_uncertain_threshold: float = 0.6,
    zero_shot_classifier=None,
    hypothesis_template: str = DEFAULT_HYPOTHESIS_TEMPLATE,
) -> dict:
    """
    If sklearn top-class probability > ml_confidence_threshold, use that prediction.
    Otherwise use zero-shot classification over police / red_cross / firefighters.
    """
    ml_label, ml_conf, ml_scores = get_ml_raw_prediction(model, text)

    if ml_conf > ml_confidence_threshold:
        return {
            "text": text,
            "category": ml_label,
            "confidence": round(ml_conf, 3),
            "source": "ml",
            "ml_scores": ml_scores,
        }

    zs = classify_emergency_zero_shot(
        text,
        classifier=zero_shot_classifier,
        hypothesis_template=hypothesis_template,
        uncertain_threshold=zs_uncertain_threshold,
    )
    return {
        "text": text,
        "category": zs["category"],
        "confidence": zs["confidence"],
        "source": zs["source"],
        "ml_scores": ml_scores,
    }

if __name__ == "__main__":
    import joblib

    m = joblib.load("emergency_classifier.pkl")
    for s in [
        "my father collapsed and is not breathing",
        "there is fire in the building",
        "someone is breaking into my house",
    ]:
        print(classify_emergency_hybrid(m, s))

