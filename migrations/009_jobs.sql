-- Atlas jobs: one row per submitted generation job (video or image).
--
-- Why: the Atlas prediction id used to live only in a local variable, so a
-- restart (every deploy runs `docker compose down`) orphaned paid jobs and left
-- users without any reply. Persisting the id lets the bot, on the next start,
-- find jobs that were in flight, tell the affected users and reset their state.
--
-- status lifecycle:
--   submitted    -> Atlas accepted the job, we are polling it
--   completed    -> Atlas finished, output url received
--   failed       -> Atlas reported failure / timeout / poll error
--   interrupted  -> bot restarted while the job was in flight (user notified)
--   expired      -> same, but too old to be worth notifying about
--
-- bot_id is part of the key because this database is shared with video-gen-bot.
CREATE TABLE IF NOT EXISTS jobs (
    bot_id     TEXT        NOT NULL,
    pred_id    TEXT        NOT NULL,
    user_id    BIGINT      NOT NULL,
    chat_id    BIGINT      NOT NULL,
    model      TEXT        NOT NULL,
    status     TEXT        NOT NULL DEFAULT 'submitted'
               CHECK (status IN ('submitted', 'completed', 'failed', 'interrupted', 'expired')),
    error_text TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (bot_id, pred_id)
);

-- Startup recovery only looks at in-flight rows — keep that lookup tiny.
CREATE INDEX IF NOT EXISTS jobs_inflight
    ON jobs (bot_id) WHERE status = 'submitted';

-- Daily retention sweep deletes by age.
CREATE INDEX IF NOT EXISTS jobs_created_at
    ON jobs (created_at);
