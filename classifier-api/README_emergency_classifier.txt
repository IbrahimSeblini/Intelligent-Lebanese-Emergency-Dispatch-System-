# Emergency Routing Classifier (WhatsApp-Ready)

This project classifies emergency text messages into:
- `police`
- `red_cross` (medical)
- `firefighters`
- optionally `uncertain`

It also extracts dispatch details from free text:
- caller `name`
- `location`
- `phone` (from message or fallback caller number)

The current system is designed for real-world noisy messages (English + Arabic/Arabizi), and for integration with a WhatsApp number.

## Project Goals

1. Route emergency messages to the right responder quickly.
2. Handle vague, typo-heavy, and mixed-language messages better than basic keyword matching.
3. Return structured dispatch data, not only a label.
4. Ask follow-up questions when required information is missing.

## Current Architecture

### 1) Pretrained classifier (default path)

File: `hybrid_emergency.py`

- Uses Hugging Face zero-shot NLI via `transformers.pipeline("zero-shot-classification")`
- Default pretrained model: `facebook/bart-large-mnli`
- Candidate prompts map model output to project categories
- Can run in two modes:
  - **force label**: always return one of `police/red_cross/firefighters`
  - **allow uncertain**: return `uncertain` when confidence is below threshold

Why:
- Pretrained NLI generalizes better to unseen phrasing than TF-IDF.
- No retraining required for each new sentence pattern.

### 2) Arabic/Arabizi rule layer (high-priority override)

File: `hybrid_emergency.py`

- Lightweight keyword rules run before zero-shot for critical local phrases.
- Added because some Arabizi emergency intents were misclassified by pure zero-shot.
- Example fixed case:
  - "fi harame fet 3ala bayte" -> `police`

Why:
- Safety-critical phrases should be deterministic.
- Rules improve precision for common local wording.

### 3) Contact and location extraction

File: `hybrid_emergency.py`

- Extracts `name`, `location`, and `phone` from text.
- If phone is not in message, uses fallback caller number (for WhatsApp integration).
- Returns `missing_fields` so caller can be asked for what is missing.

Why:
- Dispatch needs actionable metadata, not only classification.

### 4) Interactive operator flow

File: `interactive_test.py`

- CLI test flow for message classification.
- Prompts user for missing `name` and `location`.
- Supports:
  - `--caller-phone` (fallback phone)
  - `--allow-uncertain`
  - `--threshold`

Why:
- Simulates real conversational flow before webhook integration.

### 5) Batch smoke test

File: `test_model.py`

- Runs fixed sample messages through current classifier settings.
- Useful for regression checks after changes.

## Why We Moved Beyond the Original TF-IDF Baseline

Original baseline:
- `train_emergency_classifier.py`
- `emergency_dataset_starter.csv`
- local model (`emergency_classifier.pkl`)

Problem observed:
- TF-IDF + logistic regression is brittle on unseen wording.
- New message phrasing not in dataset often produced low-confidence outputs.

Decision:
- Use pretrained NLI as primary runtime classifier.
- Keep training pipeline as optional baseline/experimentation path.

## Trade-offs and Design Decisions

### Pretrained zero-shot (pros/cons)

Pros:
- Better semantic generalization to new phrasing.
- Fast iteration without retraining.
- Works across many language variations.

Cons:
- First run downloads model weights (large).
- Inference can be slower than local sparse ML model.
- Confidence can still be low for vague messages.

### Rule layer (pros/cons)

Pros:
- Deterministic handling for critical local Arabizi phrases.
- Improves precision where model was weak.

Cons:
- Requires maintenance as new slang/phrases appear.
- Can overfit if rules become too broad.

### Force-label mode vs uncertain mode

Force-label:
- Always returns a dispatch class.
- Better for "must route every message" systems.
- Risk: may confidently route vague messages incorrectly.

Allow-uncertain:
- Safer for ambiguous messages.
- Requires follow-up message or human triage.

## Files and Their Roles

- `hybrid_emergency.py`  
  Core runtime logic: rules + pretrained classification + field extraction.

- `interactive_test.py`  
  Manual testing flow with missing-field prompts.

- `test_model.py`  
  Batch smoke test for expected behavior.

- `requirements.txt`  
  Dependencies (`transformers`, `torch`, `scikit-learn`, etc.).

- `train_emergency_classifier.py`  
  Legacy/optional training pipeline for local supervised model.

- `emergency_dataset_starter.csv`  
  Labeled dataset for the local training path.

## How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Interactive run (recommended)

```bash
python interactive_test.py --caller-phone +96170123456
```

Optional flags:
- `--allow-uncertain`
- `--threshold 0.5`

### Batch run

```bash
python test_model.py
```

### Optional: retrain baseline local model

```bash
python train_emergency_classifier.py --csv emergency_dataset_starter.csv --model_out emergency_classifier.pkl
```

## WhatsApp Integration Notes

Expected production flow:
1. Receive incoming WhatsApp message text and sender phone.
2. Call classifier with message + sender as fallback phone.
3. If `missing_fields` has `name` or `location`, ask follow-up question.
4. When required fields exist, create dispatch payload and route by category.

This project already supports steps 2-3 in code.

## Recommended Next Improvements

1. Add persistent session state per phone number (DB/Redis) for multi-turn collection.
2. Expand Arabic/Arabizi rule coverage with validated phrase lists.
3. Add evaluation set from real chat logs (anonymized) and measure routing precision/recall.
4. Add confidence guardrails and optional human escalation for low-confidence forced labels.
5. Add API/webhook service wrapper (FastAPI/Flask) for direct WhatsApp provider integration.
