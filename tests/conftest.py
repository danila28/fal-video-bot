"""Test environment: utils/config.py validates required variables at import time,
so the tests provide harmless placeholder values before any project module loads.
Real values from a developer's .env are never used by the tests (setdefault keeps
an explicitly exported variable if one is set)."""

import os

_TEST_ENV = {
    "TELEGRAM_TOKEN": "123456:test-token",
    "ATLAS_API_KEY": "atlas-test",
    "CORE_DB_NAME": "test_db",
    "DB_USERNAME": "test_user",
    "DB_PASSWORD": "test_password",
    "OPENAI_API_KEY": "sk-test",
}

for _key, _value in _TEST_ENV.items():
    os.environ.setdefault(_key, _value)
