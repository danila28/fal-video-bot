CREATE TABLE IF NOT EXISTS fsm_states (
    chat_id BIGINT,
    user_id BIGINT,
    state TEXT,
    PRIMARY KEY (chat_id, user_id)
);

CREATE TABLE IF NOT EXISTS fsm_data (
    chat_id BIGINT,
    user_id BIGINT,
    data JSONB,
    PRIMARY KEY (chat_id, user_id)
);

CREATE TABLE IF NOT EXISTS user_settings (
    chat_id BIGINT,
    user_id BIGINT,
    data JSONB,
    PRIMARY KEY (chat_id, user_id)
);