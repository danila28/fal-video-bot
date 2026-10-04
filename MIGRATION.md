# Миграция текстовой генерации: Gemini → OpenAI (luna/terra/sol)

## Что изменилось

### Новая архитектура: `services/llm.py`

Текстовая генерация теперь использует **многопровайдерный маршрутизатор** вместо жёсткой привязки к Gemini:

- **`LLMService`** — роутер, выбирающий провайдера по имени модели
- **`OpenAITextProvider`** — новый провайдер для Luna, Terra, Sol
- **`GeminiTextProvider`** — перенесён Gemini текстовой код (опциональный)

### Модели

#### OpenAI (новые)
```
luna           💰 ~$0.30/1M токенов  (быстрая, дешёвая)
terra          ⚖️  ~$0.75/1M токенов  (средняя — по умолчанию)
sol            🏆 ~$3.00/1M токенов  (мощная, точная)
```

#### Gemini (сохранены)
```
gemini-3.1-flash-lite  💰 ~$0.10/1M токенов
gemini-3.5-flash       ⚖️  ~$0.15/1M токенов
gemini-3.1-pro         🏆 ~$1.25/1M токенов
```

### Конфигурация

#### `.env` — новые переменные

```bash
# Требуется одна из них (или обе)
OPENAI_API_KEY=sk-proj-...
GEMINI_API_KEY=...
```

#### Проверка при старте
- Требуется хотя бы один из `OPENAI_API_KEY` или `GEMINI_API_KEY`
- Меню модели собирается динамически из доступных провайдеров
- Старые пользователи с `gemini-...` в БД автоматически перейдут на `terra` (по умолчанию)

### Измененные файлы

#### 1. **`services/llm.py`** (новый)
- `LLMService` — маршрутизатор
- `OpenAITextProvider` — провайдер OpenAI
- `GeminiTextProvider` — провайдер Gemini
- Обработка ошибок: `LLMEmptyResponse`, `LLMRefusal`
- Ретраи: OpenAI только на временные ошибки (429, 5xx), не на 400/401

#### 2. **`services/gemini.py`**
- Удалены: `generate_text()`, `get_text_models()`
- Осталось: ffmpeg-методы (`apply_grade`, `mux_audio`, `concat_videos` и т.д.)
- API ключ больше не требуется в конструкторе

#### 3. **`main.py`**
```python
# Было
container.instance(GeminiService(api_key=config["GEMINI_API_KEY"]))

# Стало
llm_providers = []
if config.get("OPENAI_API_KEY"):
    llm_providers.append(OpenAITextProvider(api_key=config["OPENAI_API_KEY"]))
if config.get("GEMINI_API_KEY"):
    llm_providers.append(GeminiTextProvider(api_key=config["GEMINI_API_KEY"]))

container.instance(LLMService(providers=llm_providers, default_model="terra"))
container.instance(GeminiService(api_key=config.get("GEMINI_API_KEY") or ""))
```

#### 4. **`bot/handlers/`**
- 11 вызовов `gemini.generate_text()` → `llm.generate_text()`
- Функции `_build_video_prompt`, `_build_image_prompt`, `_build_ref_image_prompts` теперь принимают `llm: LLMService`
- Обновлено меню выбора моделей в `settings_text_model()`
- Добавлена валидация в `handle_text_model_selected()` — проверка существования модели

#### 5. **`utils/config.py`**
- Добавлен `OPENAI_API_KEY`
- Убран `GEMINI_API_KEY` из `_REQUIRED`
- Проверка: хотя бы один из LLM-ключей обязателен
- `OPENAI_API_KEY` добавлен в `_ASCII_ONLY`

#### 6. **`requirements.txt`**
- Добавлен `openai>=1.50.0`

#### 7. **`.env.example` и `docker-compose.yml`**
- Добавлены переменные `OPENAI_API_KEY`

### Обработка ошибок

#### OpenAI
- **Ретраи** (макс 3 попытки): 429, 5xx, сетевые ошибки
- **Без ретраев**: 400/401 (неправильный ключ, ограничение по правам)
- **`LLMEmptyResponse`**: пустой ответ модели
- **`LLMRefusal`**: модель отказала (например, на описание реальных людей)

#### Fallback на Vision-модель
Если пользователь передал фото, но модель не поддерживает vision:
- Роутер автоматически выбирает первую vision-модель из того же провайдера
- Записывает warning в лог

### Дополнительно

#### Смоук-тест
```bash
OPENAI_API_KEY=sk-proj-... python scripts/smoke_llm.py
OPENAI_API_KEY=sk-proj-... python scripts/smoke_llm.py --photo ~/my_photo.jpg
```

Проверяет:
- Инициализацию провайдеров
- Список доступных моделей
- Текстовую генерацию
- Vision-возможность (если передано фото)
- Обработку отказов и пустых ответов

#### Тесты
```bash
pytest tests/test_llm_router.py -v
```

### Развёртывание

1. **На сервере**: добавить в `.env`
   ```bash
   OPENAI_API_KEY=sk-proj-...
   GEMINI_API_KEY=...  # опционально
   ```

2. **Docker**:
   ```bash
   docker-compose down
   docker-compose up -d
   ```

3. **Проверка**: меню выбора модели должно содержать luna, terra, sol (+ gemini, если ключ есть)

### Известные ограничения

- **GPT и реальные лица**: модель может отказать описывать фото реальных людей чаще, чем Gemini
  - Решение: использовать Gemini как fallback для запросов с фото (при необходимости)
- **Параметры моделей**: reasoning-модели (O-series) имеют ограничения на `max_tokens` и `temperature`
  - Решение: запросы содержат только `model` и `messages`, без доп. параметров

### Откат

Если нужно вернуться на Gemini:
1. Обновить `GEMINI_API_KEY` в `.env`
2. Удалить `OPENAI_API_KEY` или оставить пустым
3. Меню автоматически вернёт только Gemini-модели

---

**Дата**: 2026-10-04  
**Автор**: Claude Haiku 4.5 LLM  
**Статус**: ✅ Готово к развёртыванию
