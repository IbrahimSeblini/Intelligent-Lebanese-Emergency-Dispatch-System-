import os
from typing import Any, Dict, List, Optional, Tuple

import httpx


CLASSIFIER_API_URL = os.getenv("CLASSIFIER_API_URL", "http://classifier-api:5050")

EMERGENCY_NUMBERS: Dict[str, str] = {
    "fire": "175",
    "medical": "140",
    "police": "112",
}

CATEGORY_TO_INTENT = {
    "firefighters": "fire",
    "red_cross": "medical",
    "police": "police",
}

INTENT_KEYWORDS: Dict[str, List[str]] = {
    "fire": ["fire", "smoke", "burning", "flame", "حريق", "دخان", "نار"],
    "medical": [
        "heart attack", "injury", "bleeding", "faint", "not breathing",
        "ambulance", "medical", "chest pain",
        "اسعاف", "نزيف", "ما عم يتنفس", "وجع صدر", "اصابة",
    ],
    "police": [
        "robbery", "stolen", "attack", "fight", "weapon", "thief",
        "police", "break in",
        "سرقة", "هجوم", "سلاح", "شرطة", "لص",
    ],
}


async def route_emergency(
    text: str,
    general_fallback_number: str,
    caller_phone: Optional[str] = None,
) -> Dict[str, Any]:
    # Call classifier-api (same as dispatcher-api does)
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{CLASSIFIER_API_URL}/api/dispatch",
                json={"text": text, "caller_phone": caller_phone, "allow_uncertain": True},
            )
        if resp.status_code == 200:
            data = resp.json()
            cls = data.get("classification", {})
            category = cls.get("category", "uncertain")
            confidence = float(cls.get("confidence") or 0.0)
            intent = CATEGORY_TO_INTENT.get(category, "unknown")
            number = EMERGENCY_NUMBERS.get(intent, general_fallback_number)
            return {
                "intent": intent,
                "confidence": round(confidence, 3),
                "number": number,
                "source": cls.get("source", "classifier-api"),
                "source_system": "classifier-api",
                "contact": data.get("contact", {}),
                "missing_fields": data.get("missing_fields", []),
            }
    except Exception as exc:
        print({"classifier_api_error": str(exc), "falling_back_to": "keyword_router"})

    # Fallback: keyword matching if classifier-api is unreachable
    lowered = text.lower()
    scores: Dict[str, int] = {
        intent: sum(1 for kw in keywords if kw in lowered)
        for intent, keywords in INTENT_KEYWORDS.items()
    }
    top_intent, top_score, second_score = _top_two(scores)

    if top_score == 0:
        return {
            "intent": "unknown",
            "confidence": 0.0,
            "number": general_fallback_number,
            "source": "keyword_router",
            "source_system": "keyword_fallback",
        }

    confidence = _confidence(top_score, second_score)
    return {
        "intent": top_intent,
        "confidence": confidence,
        "number": EMERGENCY_NUMBERS.get(top_intent, general_fallback_number),
        "source": "keyword_router",
        "source_system": "keyword_fallback",
    }


def _top_two(scores: Dict[str, int]) -> Tuple[str, int, int]:
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_intent, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0
    return top_intent, top_score, second_score


def _confidence(top_score: int, second_score: int) -> float:
    base = min(0.95, 0.45 + (top_score * 0.15))
    gap_penalty = 0.15 if top_score == second_score else 0.0
    return round(max(0.1, base - gap_penalty), 2)
