CREATE TABLE IF NOT EXISTS calls (
  id              SERIAL PRIMARY KEY,
  call_sid        TEXT,
  conversation_id TEXT UNIQUE NOT NULL,
  source          TEXT DEFAULT 'voice',
  services        JSONB,
  priority        TEXT,
  location        TEXT,
  emergency       TEXT,
  status          TEXT DEFAULT 'active',
  created_at      TIMESTAMPTZ DEFAULT NOW(),
  updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Migration for existing deployments (safe to run multiple times)
ALTER TABLE calls ADD COLUMN IF NOT EXISTS source TEXT DEFAULT 'voice';
ALTER TABLE calls ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'active';
