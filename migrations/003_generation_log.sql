-- Generation log: one row per Veo generation attempt.
-- gen_type: 'initial' | 'regen' | 'extend'
CREATE TABLE IF NOT EXISTS generation_log (
    id            BIGSERIAL PRIMARY KEY,
    user_id       BIGINT    NOT NULL,
    chat_id       BIGINT    NOT NULL,
    gen_type      TEXT      NOT NULL DEFAULT 'initial',
    video_model   TEXT      NOT NULL DEFAULT '',
    target_duration INT     NOT NULL DEFAULT 0,
    success       BOOLEAN   NOT NULL DEFAULT TRUE,
    error_text    TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS generation_log_user_chat
    ON generation_log (user_id, chat_id, created_at DESC);
