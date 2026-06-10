-- Video history: stores Telegram file_id for each generated video so the
-- bot can re-send them without keeping files on disk.
CREATE TABLE IF NOT EXISTS video_history (
    id         BIGSERIAL PRIMARY KEY,
    user_id    BIGINT    NOT NULL,
    chat_id    BIGINT    NOT NULL,
    file_id    TEXT      NOT NULL,
    title      TEXT      NOT NULL DEFAULT '',
    gen_type   TEXT      NOT NULL DEFAULT 'initial',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS video_history_user_chat
    ON video_history (user_id, chat_id, created_at DESC);
