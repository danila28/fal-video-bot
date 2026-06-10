CREATE TABLE IF NOT EXISTS chat_accounts (
    id        SERIAL PRIMARY KEY,
    chat_id   BIGINT NOT NULL,
    platform  TEXT   NOT NULL,
    account_id TEXT  NOT NULL,
    label     TEXT   NOT NULL DEFAULT '',
    UNIQUE (chat_id, platform, account_id)
);
