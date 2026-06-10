-- Remove the legacy accounts_for_publication key from user_settings.
-- This field was used before the chat_accounts table was introduced
-- (migration 002). The bot no longer reads or writes this key —
-- accounts are now stored in chat_accounts. Safe to drop everywhere.
UPDATE user_settings
SET    data = data - 'accounts_for_publication'
WHERE  data ? 'accounts_for_publication';
