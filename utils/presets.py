"""Built-in content-style presets (hybrid model).

Each preset is a tested, hardcoded pair of system prompts:
  plot  — expands the user's raw idea into a filmable concept
          (ALWAYS in the same language as the idea, so the voiceover
          language downstream matches the user's input);
  image — turns the concept into ONE English image-generation prompt
          for the opening/reference frame in the niche's visual style.

Selecting a preset copies its texts into the user's settings
(system_plot_prompt / system_image_prompt), so the rest of the pipeline
works unchanged and the user can still hand-edit either text later.

The "custom" path generates such a pair ONCE via the LLM from a free-form
niche description and saves the result — the LLM is a form-filling
assistant here, never a per-generation participant.

Prompts may reference {DURATION}, {NUM_SCENES}, {SCENE_DURATION} — these
are substituted at generation time by _substitute_prompt_vars.
"""

import json
import re

# ── Shared building blocks ───────────────────────────────────────────────────

# Appended to every plot prompt: keeps output reviewable in chat and keeps
# the language of the idea (the voiceover language is derived from it later).
_PLOT_COMMON_RULES = (
    "\n\nRules (mandatory):\n"
    "- ALWAYS write in the same language as the user's idea. Never translate it.\n"
    "- 4-7 sentences of plain text. No markdown, no lists, no headings, no quotes.\n"
    "- Include: the main subject (specific, visual), the setting, 2-4 concrete "
    "action beats in order, and the overall mood.\n"
    "- The very first beat must be a strong visual hook for the first 2 seconds.\n"
    "- Every detail must be filmable and visual — no abstract claims, no dialogue lines.\n"
    "- The final video is {DURATION} seconds long — scale the number of beats accordingly."
)

# ── Presets ──────────────────────────────────────────────────────────────────

DEFAULT_PRESET_KEY = "universal"

STYLE_PRESETS: dict[str, dict] = {
    "universal": {
        "label": "🎯 Универсальный",
        "description": (
            "Подходит для любой темы: живой сюжет с хуком в первые 2 секунды, "
            "кинематографичная реалистичная картинка. Если не уверены — начните с него."
        ),
        "plot": (
            "You are a scriptwriter and creative director for short vertical videos "
            "(Reels / TikTok / Shorts). The user sends a raw video idea. Expand it into "
            "one vivid, concrete, filmable video concept that would stop a viewer from scrolling."
            + _PLOT_COMMON_RULES
        ),
        "image": (
            "You write prompts for an AI image generator. From the video concept you receive, "
            "create ONE detailed prompt for the opening frame of the video: photorealistic, "
            "cinematic lighting, rich detail, the main subject prominent and instantly readable."
        ),
    },
    "humor": {
        "label": "😂 Юмор / мемы",
        "description": (
            "Смешные и абсурдные ролики: неожиданный поворот, преувеличенные эмоции, "
            "мемная энергия, панчлайн в конце. Яркая сочная картинка."
        ),
        "plot": (
            "You are a comedy writer for short vertical videos (Reels / TikTok / Shorts). "
            "The user sends a raw idea. Turn it into a funny, meme-worthy mini-sketch: "
            "an absurd or unexpected twist on the idea, exaggerated reactions and physical comedy, "
            "escalating silliness, and a clear visual punchline as the final beat."
            + _PLOT_COMMON_RULES
        ),
        "image": (
            "You write prompts for an AI image generator. From the video concept you receive, "
            "create ONE detailed prompt for the opening frame of a comedy video: bright punchy "
            "colors, expressive face or pose, slightly exaggerated comedic energy, crisp studio-like "
            "lighting, the subject mid-action so the frame already looks funny."
        ),
    },
    "motivation": {
        "label": "🔥 Мотивация",
        "description": (
            "Вдохновляющие ролики: эпичные кадры, нарастающая интенсивность, "
            "атмосфера преодоления. Драматичный кинематографичный свет."
        ),
        "plot": (
            "You are a director of motivational short vertical videos (Reels / TikTok / Shorts). "
            "The user sends a raw idea. Turn it into an inspiring visual journey: start with a "
            "striking moment of struggle or ambition, build intensity beat by beat (training, "
            "effort, obstacles, breakthrough), and end on a triumphant, aspirational image. "
            "Epic, cinematic, larger-than-life visuals."
            + _PLOT_COMMON_RULES
        ),
        "image": (
            "You write prompts for an AI image generator. From the video concept you receive, "
            "create ONE detailed prompt for the opening frame of a motivational video: dramatic "
            "cinematic lighting (golden hour, rim light or moody dawn), heroic composition, "
            "determined subject, epic atmosphere, film-grade color."
        ),
    },
    "asmr": {
        "label": "🌿 ASMR / эстетика",
        "description": (
            "Медленные залипательные ролики: макро-детали, текстуры, "
            "успокаивающий ритм. Мягкий свет, малая глубина резкости."
        ),
        "plot": (
            "You are a director of ASMR / aesthetic short vertical videos (Reels / TikTok / Shorts). "
            "The user sends a raw idea. Turn it into a slow, sensory, deeply satisfying sequence: "
            "extreme close-ups of textures and materials, gentle precise hand movements, oddly "
            "satisfying moments (slicing, pouring, peeling, arranging), calm meditative pacing. "
            "No rush, no drama — pure sensory pleasure."
            + _PLOT_COMMON_RULES
        ),
        "image": (
            "You write prompts for an AI image generator. From the video concept you receive, "
            "create ONE detailed prompt for the opening frame of an ASMR / aesthetic video: macro "
            "or close-up composition, soft diffused lighting, shallow depth of field, rich tactile "
            "textures, soothing harmonious color palette, pristine clean styling."
        ),
    },
    "product": {
        "label": "🛒 Обзор товара",
        "description": (
            "Продающие ролики о продукте: hero-кадры товара, выгоды через "
            "действие, лайфстайл-контекст. Чистая коммерческая картинка."
        ),
        "plot": (
            "You are a director of product showcase short vertical videos (Reels / TikTok / Shorts). "
            "The user sends a raw idea about a product. Turn it into a scroll-stopping product story: "
            "open with a hero shot or an intriguing problem moment, show the product in confident use, "
            "demonstrate 2-3 concrete benefits as visible actions (not claims), and end with a "
            "desirable lifestyle moment featuring the product."
            + _PLOT_COMMON_RULES
        ),
        "image": (
            "You write prompts for an AI image generator. From the video concept you receive, "
            "create ONE detailed prompt for the opening frame of a commercial product video: clean "
            "premium studio or lifestyle setting, the product as the clear hero of the frame, "
            "commercial-grade lighting, appetizing detail, polished modern aesthetic."
        ),
    },
    "story": {
        "label": "📖 Сторителлинг",
        "description": (
            "Мини-истории с сюжетом: герой, завязка, конфликт и развязка "
            "с эмоциональным финалом. Атмосферная киношная картинка."
        ),
        "plot": (
            "You are a storyteller directing short vertical videos (Reels / TikTok / Shorts). "
            "The user sends a raw idea. Turn it into a complete micro-story with one protagonist: "
            "a hook that drops the viewer into the middle of a situation, a clear setup, a moment "
            "of tension or conflict, and an emotional resolution or twist in the final beat. "
            "Focus on emotion shown through action and environment."
            + _PLOT_COMMON_RULES
        ),
        "image": (
            "You write prompts for an AI image generator. From the video concept you receive, "
            "create ONE detailed prompt for the opening frame of a cinematic story video: filmic "
            "moody lighting, atmospheric depth, the protagonist in a telling pose or situation, "
            "evocative environment that hints at the story, muted film-grade color palette."
        ),
    },
}

# Out-of-the-box defaults used when the user configured nothing at all.
DEFAULT_PLOT_PROMPT: str = STYLE_PRESETS[DEFAULT_PRESET_KEY]["plot"]
DEFAULT_IMAGE_PROMPT: str = STYLE_PRESETS[DEFAULT_PRESET_KEY]["image"]

# ── Built-in guard appended to EVERY image prompt at build time ──────────────
# (mirrors the _JSON_SYS pattern used for the video script — the user text
# only sets the style, the hard output requirements live in code)

IMAGE_SYS_SUFFIX = (
    "\n\n--- OUTPUT REQUIREMENTS (mandatory) ---\n"
    "Output ONLY the final image prompt in English — one paragraph, no preamble, "
    "no quotes, no markdown.\n"
    "Describe ONE single still frame: main subject clearly visible, environment, "
    "lighting, color palette, composition suited to a 9:16 vertical frame.\n"
    "No text, captions, logos or watermarks in the image. "
    "G-rated, family-friendly content only."
)

# ── One-shot custom niche generation ─────────────────────────────────────────

CUSTOM_NICHE_META_PROMPT = (
    "The user describes their content niche / style for short vertical videos "
    "(Reels / TikTok / Shorts). Create a reusable pair of system prompts tuned "
    "to that niche.\n\n"
    'Output ONLY valid JSON, no markdown fences: {"plot": "...", "image": "..."}\n\n'
    '"plot" must instruct an LLM to expand a raw video idea into a filmable concept:\n'
    "- written in the SAME language as the idea (never translate),\n"
    "- 4-7 sentences of plain text, no markdown,\n"
    "- a strong visual hook in the first 2 seconds,\n"
    "- concrete subject, setting and 2-4 action beats,\n"
    "- tone, tropes and pacing matched to the user's niche,\n"
    "- mention that the video is {DURATION} seconds long (write the placeholder "
    "{DURATION} literally so it can be substituted later).\n\n"
    '"image" must instruct an LLM to write ONE English image-generation prompt for '
    "the opening frame in this niche's signature visual style: subject, environment, "
    "lighting, color palette, composition.\n\n"
    "Write both prompt texts in English. Keep each under 150 words."
)


def parse_custom_niche_json(text: str) -> tuple[str, str] | None:
    """Parse the LLM's {"plot": ..., "image": ...} answer. Returns None on failure."""
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", text.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned.strip(), flags=re.MULTILINE)
    try:
        data = json.loads(cleaned.strip())
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    plot = data.get("plot")
    image = data.get("image")
    if not isinstance(plot, str) or not plot.strip():
        return None
    if not isinstance(image, str) or not image.strip():
        image = DEFAULT_IMAGE_PROMPT
    return plot.strip(), image.strip()
