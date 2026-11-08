import json
import os
from typing import Any, Dict, Optional

import asyncpg
import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from openai import AsyncOpenAI

from router import route_emergency

app = FastAPI(title="WhatsApp Emergency Bot")

META_ACCESS_TOKEN        = os.getenv("META_ACCESS_TOKEN", "")
META_PHONE_NUMBER_ID     = os.getenv("META_PHONE_NUMBER_ID", "")
META_VERIFY_TOKEN        = os.getenv("META_VERIFY_TOKEN", "")
META_GRAPH_VERSION       = os.getenv("META_GRAPH_VERSION", "v22.0")
OPENAI_API_KEY           = os.getenv("OPENAI_API_KEY", "")
OPENAI_TRANSCRIBE_MODEL  = os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")
GENERAL_EMERGENCY_NUMBER = os.getenv("GENERAL_EMERGENCY_NUMBER", "112")

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
_db_pool = None

SYSTEM_PROMPT = """You are an emergency dispatcher assistant for Lebanon's unified emergency services, responding via WhatsApp.

The emergency service has already been determined. Your job is to:
1. Assess the severity:
   - CRITICAL (life threatening, immediate danger)
   - HIGH (serious but stable)
   - MEDIUM (urgent but not life threatening)

2. Write a short calm WhatsApp reply to keep the caller engaged and collect any critical missing detail (location, injuries, number of people). Let the caller speak naturally — do not instruct them how to format their messages.

3. Formulate a clean dispatcher-ready emergency summary from all available information so far.

Return ONLY a JSON with: severity, response, dispatch_note, emergency.
- severity: CRITICAL or HIGH or MEDIUM
- response: short calm WhatsApp message, max 2 sentences
- dispatch_note: brief note for the dispatcher
- emergency: clean 1-2 sentence summary of the emergency for the dispatcher (location, what happened, injuries if known)

Return ONLY the JSON. No explanation, no preamble, no backticks."""

SYSTEM_PROMPT_UNCERTAIN = """You are an emergency dispatcher assistant for Lebanon's unified emergency services, responding via WhatsApp.

The emergency type is unclear. Your job is to:
1. Determine the most likely service: police, firefighters, or red_cross
2. Assess the severity:
   - CRITICAL (life threatening, immediate danger)
   - HIGH (serious but stable)
   - MEDIUM (urgent but not life threatening)
3. Write a short calm WhatsApp reply to keep the caller engaged and collect the most critical missing detail.
4. Formulate a clean dispatcher-ready emergency summary from all available information so far.

Return ONLY a JSON with: category, severity, response, dispatch_note, emergency.
- category: police, firefighters, or red_cross
- severity: CRITICAL or HIGH or MEDIUM
- response: short calm WhatsApp message, max 2 sentences
- dispatch_note: brief note for the dispatcher
- emergency: clean 1-2 sentence summary of the emergency for the dispatcher (location, what happened, injuries if known)

Return ONLY the JSON. No explanation, no preamble, no backticks."""

INTENT_TO_SERVICES = {
    "fire":        ["FIRE"],
    "firefighters": ["FIRE"],
    "medical":     ["MEDICAL"],
    "red_cross":   ["MEDICAL"],
    "police":      ["POLICE"],
}

PRIORITY_RANK = {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1}


def max_priority(a: str, b: str) -> str:
    return a if PRIORITY_RANK.get(a, 0) >= PRIORITY_RANK.get(b, 0) else b


async def get_pool() -> asyncpg.Pool:
    global _db_pool
    if _db_pool is None:
        ssl = "require" if os.getenv("NODE_ENV") == "production" else None
        _db_pool = await asyncpg.create_pool(
            host=os.getenv("DB_HOST"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            ssl=ssl,
        )
    return _db_pool


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/webhooks/whatsapp")
def verify_whatsapp_webhook(
    hub_mode: str = Query(default="", alias="hub.mode"),
    hub_challenge: str = Query(default="", alias="hub.challenge"),
    hub_verify_token: str = Query(default="", alias="hub.verify_token"),
) -> PlainTextResponse:
    if hub_mode == "subscribe" and hub_verify_token == META_VERIFY_TOKEN:
        return PlainTextResponse(content=hub_challenge)
    raise HTTPException(status_code=403, detail="Webhook verification failed.")


@app.post("/webhooks/whatsapp")
async def whatsapp_webhook(request: Request) -> Dict[str, str]:
    payload = await request.json()
    user_text, sender, extract_reason = await extract_user_text(payload)

    if sender and extract_reason == "audio_transcription_failed":
        await send_whatsapp_text(
            sender,
            "I could not understand the voice message. Please type your emergency.",
        )
        return {"status": "ok", "reason": extract_reason}

    if not user_text:
        return {"status": "ignored", "reason": extract_reason or "no_text_message"}

    # Check for an existing active conversation
    existing = await get_active_call(sender)

    if existing:
        # Follow-up: preserve original classification
        existing_services = json.loads(existing["services"])
        services     = existing_services
        final_intent = next(
            (k for k, v in INTENT_TO_SERVICES.items() if v == services), "unknown"
        )

        # Run classifier only to extract location from new message
        cls = await route_emergency(user_text, GENERAL_EMERGENCY_NUMBER, caller_phone=sender)
        new_location = (cls.get("contact") or {}).get("location")
        location = new_location or existing["location"] or "Unknown (WhatsApp)"

        history = await load_messages(sender)
        gpt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": user_text},
        ]
        completion = await openai_client.chat.completions.create(
            model="gpt-4o", messages=gpt_messages,
        )
        gpt = json.loads(completion.choices[0].message.content)

        severity      = max_priority(existing["priority"], gpt["severity"])
        reply         = gpt["response"]
        dispatch_note = gpt["dispatch_note"]
        emergency_text = gpt.get("emergency", user_text)

    else:
        # New conversation: full classify
        cls = await route_emergency(user_text, GENERAL_EMERGENCY_NUMBER, caller_phone=sender)
        is_uncertain = cls["intent"] == "unknown"

        gpt_messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT_UNCERTAIN if is_uncertain else SYSTEM_PROMPT,
            },
            {"role": "user", "content": f"[Service: {cls['intent']}] {user_text}"},
        ]
        completion = await openai_client.chat.completions.create(
            model="gpt-4o", messages=gpt_messages,
        )
        gpt = json.loads(completion.choices[0].message.content)

        final_intent  = gpt.get("category", cls["intent"]) if is_uncertain else cls["intent"]
        severity      = gpt["severity"]
        reply         = gpt["response"]
        dispatch_note = gpt["dispatch_note"]
        services      = INTENT_TO_SERVICES.get(final_intent, ["POLICE", "FIRE", "MEDICAL"])
        location      = (cls.get("contact") or {}).get("location") or "Unknown (WhatsApp)"
        emergency_text = gpt.get("emergency", user_text)

    # Persist messages for conversation context
    await save_message(sender, "user", user_text)
    await save_message(sender, "assistant", reply)

    # Save/update the call record
    await save_or_update(
        sender=sender,
        caller_id=sender,
        services=services,
        priority=severity,
        location=location,
        emergency=emergency_text,
    )

    # Send GPT reply back to user
    try:
        await send_whatsapp_text(sender, reply)
    except Exception as exc:
        print({"send_error": str(exc), "from": sender})

    print({
        "from": sender,
        "intent": final_intent,
        "severity": severity,
        "dispatch_note": dispatch_note,
        "is_followup": existing is not None,
    })

    return {"status": "ok"}


async def get_active_call(sender: str):
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchrow(
            """SELECT id, services, priority, location, emergency FROM calls
               WHERE call_sid = $1
                 AND status = 'active'
                 AND updated_at > NOW() - INTERVAL '2 hours'
               ORDER BY updated_at DESC LIMIT 1""",
            sender,
        )


async def load_messages(call_sid: str) -> list:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT role, content FROM messages WHERE call_sid = $1 ORDER BY created_at ASC",
            call_sid,
        )
        return [{"role": r["role"], "content": r["content"]} for r in rows]


async def save_message(call_sid: str, role: str, content: str) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO messages (call_sid, role, content) VALUES ($1, $2, $3)",
            call_sid, role, content,
        )


async def save_or_update(
    sender: str,
    caller_id: str,
    services: list,
    priority: str,
    location: str,
    emergency: str,
) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        # Reuse the existing row only if it's still active and within the last 2 hours.
        # A dispatched or old conversation means a new emergency — create a fresh row.
        existing = await conn.fetchrow(
            """SELECT id, conversation_id FROM calls
               WHERE call_sid = $1
                 AND status = 'active'
                 AND updated_at > NOW() - INTERVAL '2 hours'
               ORDER BY updated_at DESC LIMIT 1""",
            sender,
        )
        if existing:
            await conn.execute(
                """UPDATE calls
                   SET services = $1, priority = $2, location = $3,
                       emergency = $4, source = 'whatsapp', updated_at = NOW()
                   WHERE id = $5""",
                json.dumps(services), priority, location, emergency, existing["id"],
            )
        else:
            import uuid as _uuid
            conversation_id = f"wa-{sender}-{_uuid.uuid4().hex[:8]}"
            await conn.execute(
                """INSERT INTO calls
                   (call_sid, conversation_id, source, services, priority, location, emergency, status)
                   VALUES ($1, $2, 'whatsapp', $3, $4, $5, $6, 'active')""",
                caller_id, conversation_id,
                json.dumps(services), priority, location, emergency,
            )


async def send_whatsapp_text(to_number: str, text: str) -> None:
    if not META_ACCESS_TOKEN or not META_PHONE_NUMBER_ID:
        raise HTTPException(
            status_code=500,
            detail="Missing META_ACCESS_TOKEN or META_PHONE_NUMBER_ID.",
        )
    url = f"https://graph.facebook.com/{META_GRAPH_VERSION}/{META_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {META_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "text": {"body": text},
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(url, headers=headers, json=payload)
    if response.status_code >= 300:
        raise HTTPException(
            status_code=502,
            detail=f"Meta send failed: {response.status_code} {response.text}",
        )


async def extract_user_text(
    payload: Dict[str, Any],
) -> tuple[Optional[str], Optional[str], str]:
    try:
        entry   = payload.get("entry", [])[0]
        change  = entry.get("changes", [])[0]
        value   = change.get("value", {})
        messages = value.get("messages", [])
        if not messages:
            return None, None, "no_messages"

        msg    = messages[0]
        sender = msg.get("from")
        if not sender:
            return None, None, "missing_sender"

        msg_type = msg.get("type")
        if msg_type == "text":
            text = msg.get("text", {}).get("body", "").strip()
            return (text, sender, "text_ok") if text else (None, sender, "empty_text")

        if msg_type == "audio":
            audio_id = msg.get("audio", {}).get("id")
            if not audio_id:
                return None, sender, "missing_audio_id"
            transcript = await transcribe_whatsapp_audio(audio_id)
            if not transcript:
                return None, sender, "audio_transcription_failed"
            return transcript, sender, "audio_ok"

        return None, sender, f"unsupported_type_{msg_type}"
    except (IndexError, AttributeError):
        return None, None, "malformed_payload"


async def transcribe_whatsapp_audio(audio_media_id: str) -> Optional[str]:
    if not OPENAI_API_KEY:
        print({"voice_error": "missing_openai_api_key"})
        return None

    media_bytes, mime_type = await download_whatsapp_media(audio_media_id)
    if not media_bytes:
        print({"voice_error": "media_download_failed", "media_id": audio_media_id})
        return None

    filename = pick_audio_filename(mime_type)
    try:
        transcript = await openai_client.audio.transcriptions.create(
            model=OPENAI_TRANSCRIBE_MODEL,
            file=(filename, media_bytes, mime_type),
        )
        text = (transcript.text or "").strip()
        return text if text else None
    except Exception as exc:
        print({"voice_error": "transcription_failed", "detail": str(exc)})
        return None


async def download_whatsapp_media(media_id: str) -> tuple[Optional[bytes], str]:
    headers = {"Authorization": f"Bearer {META_ACCESS_TOKEN}"}
    media_info_url = f"https://graph.facebook.com/{META_GRAPH_VERSION}/{media_id}"
    async with httpx.AsyncClient(timeout=20.0) as client:
        meta_res = await client.get(media_info_url, headers=headers)
        if meta_res.status_code >= 300:
            return None, "audio/ogg"
        media_url  = meta_res.json().get("url")
        mime_type  = meta_res.json().get("mime_type", "audio/ogg")
        if not media_url:
            return None, mime_type
        media_res = await client.get(media_url, headers=headers)
        if media_res.status_code >= 300:
            return None, mime_type
        return media_res.content, mime_type


def pick_audio_filename(mime_type: str) -> str:
    if "mpeg" in mime_type:
        return "voice.mp3"
    if "wav" in mime_type:
        return "voice.wav"
    if "mp4" in mime_type:
        return "voice.m4a"
    return "voice.ogg"
