-- Reference tables for Telegram users and chats + foreign keys.
--
-- Until now every table carried bare chat_id / user_id numbers with nothing to
-- tie them to. That meant no referential integrity (typos and orphan rows
-- were possible) and no way to remove everything that belongs to a user or a
-- chat in one statement. Now:
--   users(user_id) and chats(chat_id) are the parents,
--   every other table references them with ON DELETE CASCADE,
--   so `DELETE FROM users WHERE user_id = X` wipes that user's settings,
--   FSM state, history, logs and jobs.
--
-- The bot keeps the parents filled: see EnsureIdentityMiddleware, which runs
-- before every handler.

CREATE TABLE IF NOT EXISTS users (
    user_id    BIGINT      PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chats (
    chat_id    BIGINT      PRIMARY KEY,
    chat_type  TEXT        NOT NULL
               CHECK (chat_type IN ('private', 'group', 'supergroup', 'channel')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Backfill from everything already in the database ────────────────────────
-- Foreign keys below fail validation if any child row has no parent, so every
-- id that exists anywhere has to be registered first.
INSERT INTO users (user_id)
SELECT user_id FROM (
    SELECT user_id FROM fsm_states
    UNION SELECT user_id FROM fsm_data
    UNION SELECT user_id FROM user_settings
    UNION SELECT user_id FROM generation_log
    UNION SELECT user_id FROM video_history
    UNION SELECT user_id FROM jobs
) AS ids
ON CONFLICT DO NOTHING;

-- Telegram id ranges: positive = private chat, -100xxxxxxxxxx = supergroup or
-- channel, other negatives = basic group. Channels cannot be told apart from
-- supergroups by id alone, so backfilled rows get 'supergroup'; the bot
-- corrects chat_type from the real update the next time that chat is seen.
INSERT INTO chats (chat_id, chat_type)
SELECT chat_id,
       CASE WHEN chat_id > 0 THEN 'private'
            WHEN chat_id < -1000000000000 THEN 'supergroup'
            ELSE 'group' END
FROM (
    SELECT chat_id FROM fsm_states
    UNION SELECT chat_id FROM fsm_data
    UNION SELECT chat_id FROM user_settings
    UNION SELECT chat_id FROM chat_accounts
    UNION SELECT chat_id FROM generation_log
    UNION SELECT chat_id FROM video_history
    UNION SELECT chat_id FROM jobs
) AS ids
ON CONFLICT DO NOTHING;

-- ── Foreign keys ────────────────────────────────────────────────────────────
-- No extra indexes on the child columns: cascade deletes are rare and these
-- tables are small. Add them if a user/chat delete ever becomes slow.
ALTER TABLE fsm_states
    ADD CONSTRAINT fsm_states_user_fk FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE,
    ADD CONSTRAINT fsm_states_chat_fk FOREIGN KEY (chat_id) REFERENCES chats (chat_id) ON DELETE CASCADE;

ALTER TABLE fsm_data
    ADD CONSTRAINT fsm_data_user_fk FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE,
    ADD CONSTRAINT fsm_data_chat_fk FOREIGN KEY (chat_id) REFERENCES chats (chat_id) ON DELETE CASCADE;

ALTER TABLE user_settings
    ADD CONSTRAINT user_settings_user_fk FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE,
    ADD CONSTRAINT user_settings_chat_fk FOREIGN KEY (chat_id) REFERENCES chats (chat_id) ON DELETE CASCADE;

ALTER TABLE chat_accounts
    ADD CONSTRAINT chat_accounts_chat_fk FOREIGN KEY (chat_id) REFERENCES chats (chat_id) ON DELETE CASCADE;

ALTER TABLE generation_log
    ADD CONSTRAINT generation_log_user_fk FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE,
    ADD CONSTRAINT generation_log_chat_fk FOREIGN KEY (chat_id) REFERENCES chats (chat_id) ON DELETE CASCADE;

ALTER TABLE video_history
    ADD CONSTRAINT video_history_user_fk FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE,
    ADD CONSTRAINT video_history_chat_fk FOREIGN KEY (chat_id) REFERENCES chats (chat_id) ON DELETE CASCADE;

ALTER TABLE jobs
    ADD CONSTRAINT jobs_user_fk FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE,
    ADD CONSTRAINT jobs_chat_fk FOREIGN KEY (chat_id) REFERENCES chats (chat_id) ON DELETE CASCADE;
