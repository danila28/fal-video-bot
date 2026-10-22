-- Publications: one row per (published video, selected account).
-- Blotato answers POST /posts with a postSubmissionId immediately; the public
-- URL only appears later, when GET /posts/{id} reports status 'published'.
--
-- status: submitted | scheduled | published | failed | url_missing
CREATE TABLE IF NOT EXISTS publications (
    id                 BIGSERIAL   PRIMARY KEY,
    bot_id             TEXT        NOT NULL DEFAULT 'fvb',
    chat_id            BIGINT      NOT NULL REFERENCES chats (chat_id) ON DELETE CASCADE,
    user_id            BIGINT      NOT NULL REFERENCES users (user_id) ON DELETE CASCADE,
    account_row_id     INTEGER     NOT NULL REFERENCES chat_accounts (id) ON DELETE CASCADE,
    platform           TEXT        NOT NULL,
    post_submission_id TEXT,
    status             TEXT        NOT NULL DEFAULT 'submitted',
    public_url         TEXT,
    error_text         TEXT,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS publications_chat_created
    ON publications (chat_id, created_at DESC);

-- Cost ledger: one row per paid provider call.
-- est_usd is an estimate from the local price table; price_verified says
-- whether that price was confirmed against the provider's own price list.
CREATE TABLE IF NOT EXISTS cost_ledger (
    id             BIGSERIAL     PRIMARY KEY,
    bot_id         TEXT          NOT NULL DEFAULT 'fvb',
    chat_id        BIGINT        NOT NULL REFERENCES chats (chat_id) ON DELETE CASCADE,
    user_id        BIGINT        NOT NULL REFERENCES users (user_id) ON DELETE CASCADE,
    provider       TEXT          NOT NULL,
    model          TEXT          NOT NULL,
    kind           TEXT          NOT NULL CHECK (kind IN ('video', 'text', 'image', 'tts')),
    seconds        NUMERIC(10,2),
    est_usd        NUMERIC(12,4),
    price_verified BOOLEAN       NOT NULL DEFAULT FALSE,
    input_tokens   BIGINT,
    output_tokens  BIGINT,
    created_at     TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS cost_ledger_chat_created
    ON cost_ledger (chat_id, created_at DESC);
