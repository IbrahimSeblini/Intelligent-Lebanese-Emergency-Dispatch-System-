import express from "express";
import pg from "pg";
import OpenAI from "openai";
import client from "prom-client";
import { pathToFileURL } from "node:url";

const register = new client.Registry();
client.collectDefaultMetrics({ register });

const webhookCalls = new client.Counter({
  name: "webhook_calls_total",
  help: "Total webhook calls received",
  registers: [register],
});
const webhookErrors = new client.Counter({
  name: "webhook_errors_total",
  help: "Total webhook errors",
  registers: [register],
});
const callsByService = new client.Counter({
  name: "calls_by_service_total",
  help: "Calls broken down by dispatched service",
  labelNames: ["service"],
  registers: [register],
});
const callsBySeverity = new client.Counter({
  name: "calls_by_severity_total",
  help: "Calls broken down by severity",
  labelNames: ["severity"],
  registers: [register],
});

const app = express();
app.use(express.json());

const pool = new pg.Pool({
  host: process.env.DB_HOST,
  port: parseInt(process.env.DB_PORT || "5432"),
  database: process.env.DB_NAME,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  ssl: process.env.NODE_ENV === "production" ? { rejectUnauthorized: false } : false,
});

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY || "placeholder-not-for-production",
});

const CLASSIFIER_API_URL = process.env.CLASSIFIER_API_URL || "http://classifier-api:5050";

const MAX_CONVERSATION_ID_LEN = 256;
const MAX_EMERGENCY_TEXT_LEN = 8000;
const MAX_CALLER_LOCATION_LEN = 500;
const CLASSIFIER_MAX_ATTEMPTS = 3;
const CLASSIFIER_RETRY_BASE_MS = 400;
const OPENAI_MAX_ATTEMPTS = 2;
const ALLOWED_CALL_STATUS = new Set(["active", "dispatched", "closed"]);

const delay = (ms) => new Promise((r) => setTimeout(r, ms));

async function fetchClassifierDispatch(body) {
  const url = `${CLASSIFIER_API_URL}/api/dispatch`;
  const options = {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
  let lastThrown = null;
  for (let attempt = 0; attempt < CLASSIFIER_MAX_ATTEMPTS; attempt++) {
    try {
      const classifierRes = await fetch(url, options);
      let data = null;
      try { data = await classifierRes.json(); } catch { data = null; }
      if (classifierRes.ok) return { ok: true, status: classifierRes.status, data };
      if (classifierRes.status === 400) return { ok: false, status: classifierRes.status, data, retryable: false };
      const retryable = classifierRes.status >= 500 || [502, 503, 504].includes(classifierRes.status);
      if (retryable && attempt < CLASSIFIER_MAX_ATTEMPTS - 1) {
        await delay(CLASSIFIER_RETRY_BASE_MS * (attempt + 1));
        continue;
      }
      return { ok: false, status: classifierRes.status, data, retryable };
    } catch (e) {
      lastThrown = e;
      if (attempt < CLASSIFIER_MAX_ATTEMPTS - 1) {
        await delay(CLASSIFIER_RETRY_BASE_MS * (attempt + 1));
        continue;
      }
    }
  }
  return { ok: false, status: 503, data: null, retryable: true, lastThrown };
}

const SYSTEM_PROMPT = `You are an emergency dispatcher for Lebanon's unified emergency services.

The emergency service has already been determined. Your job is to:
1. Assess the severity:
   - CRITICAL (life threatening, immediate danger)
   - HIGH (serious but stable)
   - MEDIUM (urgent but not life threatening)

2. Return ONLY a JSON object with these fields: severity, response, dispatch_note.

- severity: CRITICAL or HIGH or MEDIUM
- response: what to say back to the caller in a calm reassuring tone, max 2 sentences
- dispatch_note: brief note for the dispatcher with key details

Return ONLY the JSON. No explanation, no preamble, no backticks.`;

const SYSTEM_PROMPT_UNCERTAIN = `You are an emergency dispatcher for Lebanon's unified emergency services.

The emergency type is unclear. Your job is to:
1. Determine ALL services needed (can be more than one):
   - POLICE (crime, theft, assault, threat, break-in)
   - FIRE (fire, smoke, gas leak, explosion)
   - MEDICAL (injury, illness, unconscious, not breathing, ambulance)

2. Assess the severity:
   - CRITICAL (life threatening, immediate danger)
   - HIGH (serious but stable)
   - MEDIUM (urgent but not life threatening)

3. Return ONLY a JSON object with these fields: category, services, severity, response, dispatch_note.

- category: the PRIMARY service as police, firefighters, or red_cross
- services: array of all needed services, e.g. ["FIRE", "MEDICAL"] or ["POLICE"] or ["POLICE", "FIRE", "MEDICAL"]
- severity: CRITICAL or HIGH or MEDIUM
- response: what to say back to the caller in a calm reassuring tone, max 2 sentences
- dispatch_note: brief note for the dispatcher with key details

Return ONLY the JSON. No explanation, no preamble, no backticks.`;

function parseGptJson(raw) {
  if (raw == null || typeof raw !== "string") return null;
  let s = raw.trim();
  const fence = s.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/i);
  if (fence) s = fence[1].trim();
  try { return JSON.parse(s); } catch { return null; }
}

app.post("/webhook", async (req, res) => {
  webhookCalls.inc();
  const { caller_id, conversation_id, emergency_description, caller_location, source } = req.body ?? {};

  // Support both WhatsApp-bot (source="whatsapp") and voice pipeline (source="voice")
  const callSource = source || "voice";

  const convId = conversation_id != null ? String(conversation_id).trim() : "";
  const description = emergency_description != null ? String(emergency_description).trim() : "";
  const location = caller_location != null && caller_location !== "" ? String(caller_location).trim() : "";

  if (!convId) { webhookErrors.inc(); return res.status(400).json({ error: "conversation_id is required" }); }
  if (convId.length > MAX_CONVERSATION_ID_LEN) { webhookErrors.inc(); return res.status(400).json({ error: "conversation_id is too long" }); }
  if (!description) { webhookErrors.inc(); return res.status(400).json({ error: "emergency_description is required" }); }
  if (description.length > MAX_EMERGENCY_TEXT_LEN) { webhookErrors.inc(); return res.status(400).json({ error: "emergency_description is too long" }); }
  if (location.length > MAX_CALLER_LOCATION_LEN) { webhookErrors.inc(); return res.status(400).json({ error: "caller_location is too long" }); }

  try {
    const { rows: existing } = await pool.query(
      "SELECT id FROM calls WHERE conversation_id = $1 LIMIT 1", [convId]
    );
    const isNew = existing.length === 0;

    if (isNew) {
      await pool.query(
        "INSERT INTO calls (call_sid, conversation_id, source, status) VALUES ($1, $2, $3, 'active')",
        [caller_id, convId, callSource]
      );
    }

    const classifierOutcome = await fetchClassifierDispatch({
      text: description,
      caller_phone: caller_id,
      allow_uncertain: true,
    });

    let cls;
    let classifierDegraded = false;

    if (classifierOutcome.ok && classifierOutcome.data) {
      const classifierResult = classifierOutcome.data;
      cls = classifierResult.classification ?? classifierResult;
      if (!cls || typeof cls.category !== "string") {
        webhookErrors.inc();
        classifierDegraded = true;
        cls = { category: "uncertain", services: ["POLICE", "FIRE", "MEDICAL"] };
      }
    } else if (classifierOutcome.status === 400 && classifierOutcome.data?.error) {
      webhookErrors.inc();
      return res.status(400).json({ error: classifierOutcome.data.error });
    } else {
      webhookErrors.inc();
      classifierDegraded = true;
      cls = { category: "uncertain", services: ["POLICE", "FIRE", "MEDICAL"] };
    }

    const isUncertain = cls.category === "uncertain";
    const category = cls.category;
    const services_needed = Array.isArray(cls.services) ? cls.services : [String(cls.category).toUpperCase()];

    let gptResult = null;
    const userPreamble = classifierDegraded ? "[Classifier unavailable — treat as unclear emergency.]\n" : "";
    for (let attempt = 0; attempt < OPENAI_MAX_ATTEMPTS; attempt++) {
      try {
        const completion = await openai.chat.completions.create({
          model: "gpt-4o",
          messages: [
            { role: "system", content: isUncertain ? SYSTEM_PROMPT_UNCERTAIN : SYSTEM_PROMPT },
            {
              role: "user",
              content: userPreamble + (isUncertain
                ? `Emergency Description: ${description}\nLocation: ${location}`
                : `Service dispatched: ${category}\nEmergency Description: ${description}\nLocation: ${location}`),
            },
          ],
        });
        gptResult = parseGptJson(completion.choices[0]?.message?.content);
        break;
      } catch (e) {
        const status = e?.status ?? e?.response?.status;
        const retryable = status === 429 || (status >= 500 && status < 600);
        if (retryable && attempt < OPENAI_MAX_ATTEMPTS - 1) { await delay(1000 * (attempt + 1)); continue; }
        console.error("openai webhook:", e);
        break;
      }
    }

    if (!gptResult || typeof gptResult !== "object") {
      webhookErrors.inc();
      gptResult = isUncertain
        ? { category: "uncertain", services: ["POLICE", "FIRE", "MEDICAL"], severity: "HIGH",
            response: "We're connecting you with emergency services. Please stay on the line and share your exact location if you can.",
            dispatch_note: (classifierDegraded ? "[Classifier degraded] " : "") + "Fallback: model unavailable or non-JSON reply; triage manually." }
        : { severity: "HIGH",
            response: "Help is being arranged. Please stay where you are if it is safe and keep your phone available.",
            dispatch_note: (classifierDegraded ? "[Classifier degraded] " : "") + `Fallback for ${category}; verify details with caller.` };
    }

    const finalCategory = isUncertain ? gptResult.category ?? "uncertain" : category;
    const finalServices = isUncertain
      ? (Array.isArray(gptResult.services) ? gptResult.services : ["POLICE", "FIRE", "MEDICAL"])
      : services_needed;
    const classification = {
      category: finalCategory,
      services_needed: finalServices,
      severity: gptResult.severity ?? "HIGH",
      response: gptResult.response ?? "",
      dispatch_note: gptResult.dispatch_note ?? "",
    };
    console.log("classification:", JSON.stringify(classification));

    callsBySeverity.inc({ severity: classification.severity });
    for (const svc of classification.services_needed) callsByService.inc({ service: svc });

    await pool.query(
      `UPDATE calls SET services = $1, priority = $2, location = $3, emergency = $4 WHERE conversation_id = $5`,
      [JSON.stringify(classification.services_needed), classification.severity, location || null, description, convId]
    );

    res.send("help is on the way");
  } catch (e) {
    webhookErrors.inc();
    console.error("webhook:", e);
    if (!res.headersSent) res.status(500).json({ error: "internal_error" });
  }
});

process.on("unhandledRejection", () => webhookErrors.inc());

app.get("/calls", async (_req, res) => {
  try {
    const { rows } = await pool.query(
      `SELECT *, call_sid AS from_number, created_at AS started_at FROM calls ORDER BY created_at DESC`
    );
    res.json(rows);
  } catch (e) {
    console.error("GET /calls:", e);
    res.status(500).json({ error: "internal_error" });
  }
});

app.patch("/calls/:conversationId", async (req, res) => {
  const conversationId = req.params.conversationId != null ? String(req.params.conversationId).trim() : "";
  if (!conversationId || conversationId.length > MAX_CONVERSATION_ID_LEN) {
    return res.status(400).json({ error: "invalid conversationId" });
  }
  const { status } = req.body ?? {};
  if (typeof status !== "string" || !ALLOWED_CALL_STATUS.has(status)) {
    return res.status(400).json({ error: "status must be active or dispatched" });
  }
  try {
    await pool.query("UPDATE calls SET status = $1, updated_at = NOW() WHERE conversation_id = $2", [status, conversationId]);
    res.json({ ok: true });
  } catch (e) {
    console.error("PATCH /calls:", e);
    res.status(500).json({ error: "internal_error" });
  }
});

app.get("/health", (_req, res) => res.json({ status: "ok" }));

app.get("/metrics", async (_req, res) => {
  res.set("Content-Type", register.contentType);
  res.send(await register.metrics());
});

const isMain = Boolean(process.argv[1]) && import.meta.url === pathToFileURL(process.argv[1]).href;
if (isMain) app.listen(3000, () => console.log("Dispatcher running on port 3000"));

export default app;
