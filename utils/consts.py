allowed_users = {684025534, 6946129587, 875131196, 5992207498}

# ── Generic (used for pre-caching when platform is unknown) ──────────────────
generate_hashtags_prompt = """
Analyze the content below and determine:
- the main theme
- key ideas
- target audience

Then generate 3 to 5 highly relevant hashtags.

Strict output rules:
- Output ONLY HASHTAGS (no explanations, no extra text)
- Each hashtag must start with #
- 3–5 hashtags total (not more, not less)
- Use the same language as the input content
- Keep hashtags short, specific, and social-media friendly
- Avoid generic or overly obvious terms (e.g., #content, #post)
- Avoid repeating similar meanings
- No spaces or punctuation inside hashtags

Output format:
#tag1 #tag2 #tag3
"""

# ── YouTube — SEO-first, searchable, descriptive ─────────────────────────────
generate_hashtags_prompt_youtube = """
Analyze the content below and generate YouTube-optimised hashtags.

YouTube hashtag rules:
- 3 to 5 hashtags
- SEO-focused: use phrases people actually search for on YouTube
- Mix: 1 broad niche tag + 2–3 mid-tail specific tags + 1 trending/viral tag if relevant
- Avoid spaces inside hashtags
- Use the same language as the input content
- Output ONLY the hashtags, nothing else

Output format:
#tag1 #tag2 #tag3
"""

# ── TikTok — trending, punchy, discovery-focused ─────────────────────────────
generate_hashtags_prompt_tiktok = """
Analyze the content below and generate TikTok-optimised hashtags.

TikTok hashtag rules:
- 4 to 6 hashtags
- Mix: 1–2 mega-tags (millions of views, e.g. #fyp #foryou) + 2–3 niche/topic tags
- Short, punchy, and viral-friendly
- Lowercase preferred
- Use the same language as the input content for niche tags; mega-tags stay in English
- Output ONLY the hashtags, nothing else

Output format:
#tag1 #tag2 #tag3 #tag4
"""


def get_hashtags_prompt_for_platforms(platforms: set[str]) -> str:
    """Return the best hashtag prompt for the given set of publish platforms.

    Single-platform publishes use the tailored prompt; mixed or unknown
    fall back to the generic one.
    """
    if platforms == {"youtube"}:
        return generate_hashtags_prompt_youtube
    if platforms == {"tiktok"}:
        return generate_hashtags_prompt_tiktok
    return generate_hashtags_prompt