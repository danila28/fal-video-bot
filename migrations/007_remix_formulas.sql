-- Formula library for the remix flow: confirmed/saved video formulas that can
-- be re-used later ("generate again by formula #3, but about coffee").
CREATE TABLE IF NOT EXISTS remix_formulas (
    id SERIAL PRIMARY KEY,
    chat_id BIGINT NOT NULL,
    user_id BIGINT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    formula JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_remix_formulas_chat
    ON remix_formulas (chat_id, id DESC);
