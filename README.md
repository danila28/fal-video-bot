# fal-video-bot

Telegram-бот для генерации видео через fal.ai с тремя моделями:

| Модель | Тип | Что делает |
|---|---|---|
| **Kling Avatar v2** | lip-sync | Говорящая голова с лип-синком (фото + аудио → видео) |
| **OmniHuman-1** | lip-sync | Аналог Kling, чуть дороже / реалистичнее |
| **Seedance 2.0** | scene clips | Сцены 10-сек клипами (image-to-video) с TTS поверх |

Фото генерируется через **Google Imagen 4 Fast** ("банано") с авто-fallback на **fal.ai FLUX Pro 1.1**.

## Pipeline

```
идея → сюжет (Gemini) → фото-промт → фото (Imagen / Gemini image) →
видео-промт + voiceover → видео (Kling / OmniHuman / Seedance) →
озвучка (ElevenLabs) → опц. фоновая музыка → караоке-субтитры →
публикация (Blotato — TikTok / YouTube, с расписанием по UTC)
```

## Что переиспользуется из других проектов

| Модуль | Источник | Назначение |
|---|---|---|
| `services/db.py`, `bot/storage.py` | video-gen-bot | PostgreSQL + settings + история |
| `services/blotato.py` | video-gen-bot | Публикация с scheduled_at |
| `services/elevenlabs.py` | video-gen-bot | TTS с word_timings для караоке |
| `services/gemini.py` | новый (на основе vertex.py) | Текст-генерация + ffmpeg утилиты |
| `services/imagegen.py` | ai-blogger-bot | Imagen + FLUX fallback |
| `services/kling.py`, `omnihuman.py`, `seedance.py` | ai-blogger-bot | fal.ai |
| `bot/handlers/publish.py` | video-gen-bot | Publish flow + расписание |
| `bot/handlers/settings.py` | video-gen-bot | Все настройки |
| Все utils, keyboards, states | video-gen-bot | Без изменений |

## Запуск

1. Скопируйте `.env.example` → `.env` и заполните ключи.
2. Убедитесь, что `video-gen-bot` запущен (нам нужен его контейнер postgres).
3. Запустите:

```bash
docker compose up -d --build
```

4. В Telegram: `/start` → ⚙️ Settings → выставьте:
   - 🧠 Text model
   - 🖼 Image model — `imagen-4.0-fast-generate-001` (рекомендуется)
   - 🎬 Video model — `kling` / `omnihuman` / `seedance`
   - 📝 Plot prompt, 🖼 Image prompt, 🎬 Video prompt — системные промпты
   - 🎙 Voice — ID голоса ElevenLabs (обязательно для Kling/OmniHuman)
   - 📤 Accounts — Blotato аккаунты для публикации

## Особенности pipeline

### Kling / OmniHuman (lip-sync)
1. Озвучка генерируется **до** видео — её аудио нужно модели как вход
2. Длительность видео определяется длиной озвучки (target_duration не используется)
3. Голос модели уже встроен в видео → TTS не муксится поверх; только опц. музыка
4. Караоке-субтитры работают благодаря тем же word_timings, что и встроенная озвучка

### Seedance (сцены)
1. Видео генерируется **первым** — без звука
2. Воиcoвер разбивается на N сцен (по 10 сек) и каждая отдельный clip
3. Клипы конкатенируются, TTS муксится поверх → караоке-субтитры
4. Между клипами last-frame передаётся как референс для непрерывности

## Замечания о БД

Бот использует **ту же** PostgreSQL базу, что и `video-gen-bot`. Таблицы те же
(`user_settings`, `chat_accounts`, `video_history`, `generation_log`,
`fsm_states`, `fsm_data`). Каждый чат должен принадлежать только одному боту,
иначе настройки (`video_model`, `image_model`) будут перекрываться.
