-- Add bot_id to user_settings so multiple bots can share the same database
-- without their settings colliding. Existing rows belong to video-gen-bot ('vgb').
ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS bot_id TEXT NOT NULL DEFAULT 'vgb';

-- Swap the primary key from (chat_id, user_id) to (chat_id, user_id, bot_id).
-- The DO block is idempotent: it only acts when the old 2-column PK still exists.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM   pg_constraint c
        JOIN   pg_class      t ON t.oid = c.conrelid
        WHERE  c.conname = 'user_settings_pkey'
        AND    t.relname = 'user_settings'
        AND    array_length(c.conkey, 1) = 2
    ) THEN
        ALTER TABLE user_settings DROP CONSTRAINT user_settings_pkey;
        ALTER TABLE user_settings ADD PRIMARY KEY (chat_id, user_id, bot_id);
    END IF;
END $$;
