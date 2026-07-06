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

# Few-shot examples below are written in English purely as illustration.
# This guard keeps the model answering in the idea's language regardless.
_EXAMPLE_GUARD = (
    "\n\nThe example below is in English ONLY for illustration. "
    "Copy its structure, pacing and level of concrete detail — "
    "but ALWAYS answer in the language of the user's idea:\n"
)

# ── Presets ──────────────────────────────────────────────────────────────────

DEFAULT_PRESET_KEY = "universal"

STYLE_PRESETS: dict[str, dict] = {
    "universal": {
        "label": "🎯 Universal",
        "description": (
            "Works for any topic: a lively story with a hook in the first 2 seconds, "
            "cinematic realistic visuals. Start here if you're not sure."
        ),
        "plot": (
            "You are a scriptwriter and creative director for short vertical videos "
            "(Reels / TikTok / Shorts). The user sends a raw video idea. Expand it into "
            "one vivid, concrete, filmable video concept that would stop a viewer from scrolling."
            + _PLOT_COMMON_RULES
            + "\n\nAvoid: slow build-ups before the hook, static poses, abstract claims "
            "('the best', 'amazing'), describing feelings instead of showing them through "
            "action, generic stock-footage moments, clichéd openings like 'in a world where'."
            + _EXAMPLE_GUARD
            + "Idea: \"a robot learns to cook\"\n"
            "Concept: \"A kitchen robot with dented chrome plating cracks an egg with "
            "surgical precision — straight onto the counter, missing the pan entirely. "
            "It tilts its camera-head at the yolk, processing the failure, then grabs "
            "three more eggs at once. Flour explodes across the kitchen as it whisks at "
            "motor-maximum speed, dough splattering the windows. Finally it presents a "
            "single, perfect golden pancake on a plate — while the kitchen behind it "
            "looks like a war zone, smoke drifting past its proudly blinking LED eyes.\""
        ),
        "image": (
            "You write prompts for an AI image generator. From the video concept you receive, "
            "create ONE detailed prompt for the opening frame of the video: photorealistic, "
            "cinematic lighting, rich detail, the main subject prominent and instantly readable."
        ),
    },
    "humor": {
        "label": "😂 Humor / memes",
        "description": (
            "Funny, absurd videos: unexpected twists, exaggerated emotions, "
            "meme energy, a punchline at the end. Bright punchy visuals."
        ),
        "plot": (
            "You are a comedy writer for short vertical videos (Reels / TikTok / Shorts). "
            "The user sends a raw idea. Turn it into a funny, meme-worthy mini-sketch: "
            "an absurd or unexpected twist on the idea, exaggerated reactions and physical comedy, "
            "escalating silliness, and a clear visual punchline that MUST land in the final beat."
            + _PLOT_COMMON_RULES
            + "\n\nAvoid: explaining the joke, polite mild humor without exaggeration, "
            "static talking heads, saving nothing for the ending, feelings described in "
            "words instead of shown through over-the-top physical reactions."
            + _EXAMPLE_GUARD
            + "Idea: \"cat doesn't want to take a bath\"\n"
            "Concept: \"A fluffy orange cat sits on the bathtub edge, staring down at the "
            "water like it's lava. The owner's hands reach in — the cat flattens its ears "
            "and death-grips the towel rail with all four paws, stretching like rubber as "
            "it's pulled. In dramatic slow motion the cat is lowered toward the water, pure "
            "betrayal in its huge eyes. Final beat: the soaked cat sits in two centimeters "
            "of water, one paw pressed to its face like it's been personally wronged by "
            "the entire universe.\""
        ),
        "image": (
            "You write prompts for an AI image generator. From the video concept you receive, "
            "create ONE detailed prompt for the opening frame of a comedy video: bright punchy "
            "colors, expressive face or pose, slightly exaggerated comedic energy, crisp studio-like "
            "lighting, the subject mid-action so the frame already looks funny."
        ),
    },
    "motivation": {
        "label": "🔥 Motivation",
        "description": (
            "Inspiring videos: epic shots, rising intensity, an atmosphere of "
            "overcoming struggle. Dramatic cinematic lighting."
        ),
        "plot": (
            "You are a director of motivational short vertical videos (Reels / TikTok / Shorts). "
            "The user sends a raw idea. Turn it into an inspiring visual journey: start with a "
            "striking moment of struggle or ambition, build intensity beat by beat (training, "
            "effort, obstacles, breakthrough), and end on a triumphant, aspirational image. "
            "Epic, cinematic, larger-than-life visuals."
            + _PLOT_COMMON_RULES
            + "\n\nAvoid: empty slogans without imagery, gym clichés with no story arc, "
            "flat energy from start to finish (intensity must RISE), stock-photo smiling "
            "people, abstract success talk instead of visible effort and sweat."
            + _EXAMPLE_GUARD
            + "Idea: \"morning run\"\n"
            "Concept: \"An alarm glows 4:58 AM in a dark room; a hand silences it before "
            "it even rings. A runner laces worn shoes by the door, breath visible in the "
            "cold hallway air. She pounds up an endless stadium staircase in pouring rain, "
            "each step heavier, legs shaking, fists clenched — then breaks through onto "
            "the rooftop level just as the sun explodes over the city skyline, standing "
            "tall, chest heaving, silhouetted in golden light.\""
        ),
        "image": (
            "You write prompts for an AI image generator. From the video concept you receive, "
            "create ONE detailed prompt for the opening frame of a motivational video: dramatic "
            "cinematic lighting (golden hour, rim light or moody dawn), heroic composition, "
            "determined subject, epic atmosphere, film-grade color."
        ),
    },
    "asmr": {
        "label": "🌿 ASMR / aesthetic",
        "description": (
            "Slow, mesmerizing videos: macro details, textures, "
            "soothing pacing. Soft light, shallow depth of field. "
            "Voiceover is quiet and sparse."
        ),
        "plot": (
            "You are a director of ASMR / aesthetic short vertical videos (Reels / TikTok / Shorts). "
            "The user sends a raw idea. Turn it into a slow, sensory, deeply satisfying sequence: "
            "extreme close-ups of textures and materials, gentle precise hand movements, oddly "
            "satisfying moments (slicing, pouring, peeling, arranging), calm meditative pacing. "
            "No rush, no drama — pure sensory pleasure."
            + _PLOT_COMMON_RULES
            + "\n\nAvoid: fast cuts and rushed pacing, wide busy shots (stay CLOSE), drama or "
            "conflict, cluttered backgrounds, loud energetic tone — this genre is quiet, "
            "precise and hypnotic."
            + _EXAMPLE_GUARD
            + "Idea: \"cutting soap\"\n"
            "Concept: \"A pristine bar of lavender soap rests on white marble, side-lit so "
            "every wax-smooth facet glows. A knife blade sinks into the edge in extreme "
            "close-up and a first curl peels away with agonizing slowness. Row after row "
            "of perfect ribbons drop softly onto the marble, each one catching the light. "
            "The final shot drifts across the finished pile of curls — orderly, soft, "
            "impossibly satisfying.\""
        ),
        "image": (
            "You write prompts for an AI image generator. From the video concept you receive, "
            "create ONE detailed prompt for the opening frame of an ASMR / aesthetic video: macro "
            "or close-up composition, soft diffused lighting, shallow depth of field, rich tactile "
            "textures, soothing harmonious color palette, pristine clean styling."
        ),
        # ASMR voiceover must stay sparse and soft — this rides on top of the
        # built-in _JSON_SYS as system_video_prompt when the preset is applied.
        "video": (
            "Voiceover style override: calm, soft, ASMR-whisper tone. Short soothing "
            "sentences with natural pauses. Use noticeably FEWER words than the target "
            "count — silence and sound are part of this genre; never rush the narration."
        ),
    },
    "product": {
        "label": "🛒 Product showcase",
        "description": (
            "Selling videos about a product: hero shots, benefits shown through "
            "action, lifestyle context. Clean commercial visuals."
        ),
        "plot": (
            "You are a director of product showcase short vertical videos (Reels / TikTok / Shorts). "
            "The user sends a raw idea about a product. Turn it into a scroll-stopping product story: "
            "open with a hero shot or an intriguing problem moment, show the product in confident use, "
            "demonstrate 2-3 concrete benefits as visible actions (not claims), and end with a "
            "desirable lifestyle moment featuring the product."
            + _PLOT_COMMON_RULES
            + "\n\nAvoid: listing specs as spoken claims ('best quality', '10-hour battery') "
            "instead of showing them in action, cluttered frames where the product gets lost, "
            "fake-enthusiastic infomercial energy, showing the product only at the end."
            + _EXAMPLE_GUARD
            + "Idea: \"wireless earbuds\"\n"
            "Concept: \"Matte-black earbuds rotate slowly on a glossy pedestal, light "
            "sliding across their curves. A runner pops one in mid-stride and the chaos "
            "of a loud city street visibly fades behind her — pedestrians blur, her pace "
            "steadies. She sprints through rain while the earbuds sit locked in place, "
            "water beading off them. Final beat: she slides the earbuds into their case "
            "on a sunlit kitchen counter next to her morning coffee, one earbud still "
            "glowing with a full-battery ring.\""
        ),
        "image": (
            "You write prompts for an AI image generator. From the video concept you receive, "
            "create ONE detailed prompt for the opening frame of a commercial product video: clean "
            "premium studio or lifestyle setting, the product as the clear hero of the frame, "
            "commercial-grade lighting, appetizing detail, polished modern aesthetic."
        ),
    },
    "story": {
        "label": "📖 Storytelling",
        "description": (
            "Mini-stories with a plot: a protagonist, setup, conflict, and "
            "an emotional resolution. Atmospheric cinematic visuals."
        ),
        "plot": (
            "You are a storyteller directing short vertical videos (Reels / TikTok / Shorts). "
            "The user sends a raw idea. Turn it into a complete micro-story with one protagonist: "
            "a hook that drops the viewer into the middle of a situation, a clear setup, a moment "
            "of tension or conflict, and an emotional resolution or twist in the final beat. "
            "Focus on emotion shown through action and environment."
            + _PLOT_COMMON_RULES
            + "\n\nAvoid: starting with backstory instead of mid-action, more than one "
            "protagonist, unresolved endings, emotions stated in words ('she was sad') "
            "instead of shown through action, tension that never pays off."
            + _EXAMPLE_GUARD
            + "Idea: \"a lost dog\"\n"
            "Concept: \"A small scruffy dog stands alone in the middle of a rain-soaked "
            "crossing, cars streaking past on both sides, its leash dragging broken behind "
            "it. It sniffs a lamppost, then a bakery doorway, ears dropping lower with "
            "each wrong turn as the streetlights flicker on. Suddenly it freezes — nose "
            "up, tail rigid — and bolts through a park, ears flying. Final beat: it "
            "crashes into the arms of a girl kneeling in a doorway with wet missing-dog "
            "flyers scattered around her, both of them soaked and shaking with joy.\""
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
