"""Gemini text generation + ffmpeg post-processing.

This service replaces video-gen-bot's VertexService for atlas-video-bot.
Video generation lives in dedicated Atlas Cloud services (Kling / Seedance);
this module keeps only:
  • Gemini text generation (idea / plot / image / video prompts, hashtags)
  • ffmpeg helpers shared across pipelines: mux audio, karaoke ASS,
    apply grade, apply speed, concat, append outro, extract last frame.
"""

import asyncio
import logging
import os
import re
import uuid
import ffmpeg
from google import genai
from google.genai import types
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

logger = logging.getLogger(__name__)


_retry_cheap = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)


class GeminiService:
    """Gemini text + ffmpeg post-processing utilities."""

    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.static_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static"
        )

    # ── Model catalogues ───────────────────────────────────────────────────

    def get_text_models(self):
        return [
            {"name": "gemini-3.1-flash-lite",  "price": "💰 ~$0.10/1M tok"},
            {"name": "gemini-3.5-flash",        "price": "⚖️  ~$0.15/1M tok"},
            {"name": "gemini-3.1-pro",          "price": "🏆 ~$1.25/1M tok"},
        ]

    def get_image_models(self):
        return [
            {"separator": True, "label": "── Atlas Cloud ──"},
            {"name": "google/nano-banana-2/text-to-image",   "price": "⚡ Nano Banana 2 · 4K native"},
            {"name": "google/nano-banana-pro/text-to-image", "price": "🏆 Nano Banana Pro · 1K/2K/4K"},
            {"separator": True, "label": "── Gemini Developer API ──"},
            {"name": "gemini-3.1-flash-image", "price": "🏆 Nano Banana Flash · best quality"},
            {"name": "gemini-3-pro-image",     "price": "⚡ Nano Banana Pro · fast"},
        ]

    def get_video_models(self):
        # Atlas Cloud scene generation models. Pricing is approximate; verify at atlascloud.ai/pricing.
        # Entries with separator=True render as non-clickable section headers in the keyboard.
        # "price" is the full button label shown to the user — keep it short and
        # self-contained (model name + type + price) so it reads fine on a phone
        # without depending on the section header above it.
        return [
            {"separator": True, "label": "── Seedance ──"},
            {"name": "seedance",           "price": "🖼 Seedance I2V · $0.08/s"},
            {"name": "seedance_fast",      "price": "🖼⚡ Seedance Fast I2V · $0.05/s"},
            {"name": "seedance_mini",      "price": "🖼💰 Seedance Mini I2V · $0.045/s"},
            {"name": "seedance_t2v",       "price": "📝 Seedance T2V · $0.08/s"},
            {"name": "seedance_mini_t2v",  "price": "📝💰 Seedance Mini T2V · $0.045/s"},
            {"name": "seedance_ref",       "price": "🎭 Seedance Ref · $0.08/s"},
            {"name": "seedance_fast_ref",  "price": "🎭⚡ Seedance Fast Ref · $0.05/s"},
            {"separator": True, "label": "── Kling ──"},
            {"name": "kling",              "price": "🖼 Kling v3 Pro I2V · $0.10/s"},
            {"name": "kling_v3_std",       "price": "🖼 Kling v3 Std I2V · $0.07/s"},
            {"name": "kling_turbo",        "price": "🖼⚡ Kling v3 Turbo I2V · $0.05/s"},
            {"name": "kling_o3_pro",       "price": "🏆 Kling O3 Pro I2V · $0.14/s"},
            {"name": "kling_o3_std",       "price": "🖼 Kling O3 Std I2V · $0.10/s"},
            {"name": "kling_t2v",          "price": "📝 Kling v3 Pro T2V · $0.10/s"},
            {"name": "kling_turbo_t2v",    "price": "📝⚡ Kling v3 Turbo T2V · $0.05/s"},
            {"name": "kling_o3_pro_ref",   "price": "🎭 Kling O3 Pro Ref · $0.14/s"},
            {"name": "kling_o3_std_ref",   "price": "🎭 Kling O3 Std Ref · $0.10/s"},
        ]

    # ── Text generation ────────────────────────────────────────────────────

    @_retry_cheap
    async def generate_text(
        self, prompt: str, system_prompt: str = "", model: str = "gemini-3.1-flash-lite"
    ) -> str:
        # Callers pass settings.get("text_model") or "" — treat empty as default
        # so the bot works out of the box without any configuration.
        model = model or "gemini-3.1-flash-lite"
        try:
            contents = [
                types.Content(role="user", parts=[types.Part(text=prompt)]),
            ]
            config = types.GenerateContentConfig(system_instruction=system_prompt)
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=model,
                contents=contents,
                config=config,
            )
            text = getattr(response, "text", None)
            if not text or not text.strip():
                candidates = getattr(response, "candidates", None) or []
                finish_reasons = [getattr(c, "finish_reason", None) for c in candidates]
                feedback = getattr(response, "prompt_feedback", None)
                raise Exception(
                    f"Empty text response from LLM (model={model}, "
                    f"finish_reasons={finish_reasons}, prompt_feedback={feedback})"
                )
            return text
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise Exception(f"Failed to generate text: {str(e)}")

    # ── Voiceover parsing ──────────────────────────────────────────────────

    _VOICEOVER_OPEN  = "\"“„«‘‚‹`"
    _VOICEOVER_CLOSE = "\"”“»’‛›`"

    def extract_voiceover(self, video_prompt: str) -> str:
        opens = re.escape(self._VOICEOVER_OPEN)
        closes = re.escape(self._VOICEOVER_CLOSE)
        labelled = re.compile(
            rf"[Vv]oice[-\s]?over[:\s]+[{opens}](.+?)[{closes}]",
            re.DOTALL,
        )
        match = labelled.search(video_prompt)
        if match:
            return match.group(1).strip()
        fallback = re.compile(rf"[{opens}](.+?)[{closes}]", re.DOTALL)
        match = fallback.search(video_prompt)
        if match:
            return match.group(1).strip()
        return ""

    @staticmethod
    def parse_script_json(text: str) -> dict | None:
        """Parse a Gemini JSON script response. Strips markdown fences, validates structure."""
        import json as _json
        cleaned = re.sub(r'^```(?:json)?\s*\n?', '', text.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r'\n?```\s*$', '', cleaned.strip(), flags=re.MULTILINE)
        cleaned = cleaned.strip()
        try:
            data = _json.loads(cleaned)
        except (_json.JSONDecodeError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        shots = data.get("shots")
        if not isinstance(shots, list) or not shots:
            return None
        # Normalise per-shot fields and set defaults
        for i, shot in enumerate(shots):
            if not isinstance(shot, dict):
                return None
            # Accept kling_prompt / seedance_prompt / prompt as scene_prompt
            for alt in ("kling_prompt", "seedance_prompt", "prompt"):
                if alt in shot and "scene_prompt" not in shot:
                    shot["scene_prompt"] = shot[alt]
                    break
            if "scene_prompt" not in shot:
                return None
            shot.setdefault("index", i)
            shot.setdefault("duration_seconds", 5)
            shot.setdefault("transition", "cut")
        return data

    def extract_caption(self, video_prompt: str) -> str:
        """Extract the Caption: field generated by Gemini alongside the video prompt."""
        opens = re.escape(self._VOICEOVER_OPEN)
        closes = re.escape(self._VOICEOVER_CLOSE)
        quoted = re.search(
            rf'Caption:\s*[{opens}](.+?)[{closes}]',
            video_prompt, re.DOTALL
        )
        if quoted:
            return quoted.group(1).strip()
        bare = re.search(r'Caption:\s*(.+)', video_prompt)
        if bare:
            return bare.group(1).strip().strip('"\'')
        return ""

    # ── ASS subtitle builders ──────────────────────────────────────────────

    @staticmethod
    def _ass_escape(text: str) -> str:
        return (
            text.replace("\\", "\\\\")
            .replace("{", "\\{")
            .replace("}", "\\}")
            .replace("\r", " ")
            .replace("\n", " ")
        )

    @staticmethod
    def _format_ass_time(seconds: float) -> str:
        h = int(seconds // 3600)
        seconds -= h * 3600
        m = int(seconds // 60)
        seconds -= m * 60
        return f"{h}:{m:02d}:{seconds:05.2f}"

    def _build_ass(self, text: str, video_duration: float = 8.0) -> str:
        fade_ms = 300
        start = 0.30
        end = max(start + 1.0, video_duration - 0.30)
        style_line = (
            "Style: Default,DejaVu Sans,68,"
            "&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
            "1,0,0,0,100,100,0,0,1,3,2,2,40,40,140,1"
        )
        safe = self._ass_escape(text)
        dialogue_text = f"{{\\fad({fade_ms},{fade_ms})}}{safe}"
        return (
            "[Script Info]\n"
            "ScriptType: v4.00+\n"
            "PlayResX: 720\n"
            "PlayResY: 1280\n"
            "WrapStyle: 0\n"
            "ScaledBorderAndShadow: yes\n"
            "YCbCr Matrix: TV.709\n"
            "\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding\n"
            f"{style_line}\n"
            "\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            f"Dialogue: 0,{self._format_ass_time(start)},{self._format_ass_time(end)},"
            f"Default,,0,0,0,,{dialogue_text}\n"
        )

    _KARAOKE_GAP_BREAK = 0.5

    @staticmethod
    def _max_words_for_duration(video_duration: float) -> int:
        if video_duration <= 16:
            return 4
        if video_duration <= 32:
            return 5
        if video_duration <= 48:
            return 6
        return 7

    def _build_karaoke_ass(
        self, word_timings, video_duration: float = 8.0, max_words: int = 4
    ) -> str:
        style_line = (
            "Style: Default,DejaVu Sans,68,"
            "&H00FFFFFF,&H0000F0FF,&H00000000,&H80000000,"
            "1,0,0,0,100,100,0,0,1,3,2,2,40,40,140,1"
        )

        chunks: list[list] = []
        current: list = []
        for w, s, e in word_timings:
            gap_break = current and (s - current[-1][2] > self._KARAOKE_GAP_BREAK)
            size_break = len(current) >= max_words
            if gap_break or size_break:
                if current:
                    chunks.append(current)
                current = []
            current.append((w, s, e))
        if current:
            chunks.append(current)

        events = []
        for chunk in chunks:
            if not chunk:
                continue
            line_start = chunk[0][1]
            line_end = min(chunk[-1][2] + 0.10, video_duration)
            parts = []
            for i, (w, s, e) in enumerate(chunk):
                cs = max(1, int(round((e - s) * 100)))
                safe = self._ass_escape(w)
                if i == 0:
                    parts.append(f"{{\\fad(120,80)\\k{cs}}}{safe}")
                else:
                    parts.append(f"{{\\k{cs}}}{safe}")
            text = " ".join(parts)
            events.append(
                f"Dialogue: 0,{self._format_ass_time(line_start)},"
                f"{self._format_ass_time(line_end)},Default,,0,0,0,,{text}"
            )

        events_block = "\n".join(events) if events else ""

        return (
            "[Script Info]\n"
            "ScriptType: v4.00+\n"
            "PlayResX: 720\n"
            "PlayResY: 1280\n"
            "WrapStyle: 0\n"
            "ScaledBorderAndShadow: yes\n"
            "YCbCr Matrix: TV.709\n"
            "\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding\n"
            f"{style_line}\n"
            "\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            f"{events_block}\n"
        )

    async def burn_karaoke_subtitles(
        self, video_path: str, word_timings
    ) -> str:
        if not word_timings:
            return video_path
        try:
            os.makedirs(self.static_dir, exist_ok=True)
            duration = await asyncio.to_thread(self._probe_duration, video_path)
            max_words = self._max_words_for_duration(duration)

            ass_path = os.path.join(self.static_dir, f"{uuid.uuid4()}.ass")
            with open(ass_path, "w", encoding="utf-8") as f:
                f.write(self._build_karaoke_ass(word_timings, duration, max_words))

            output_path = os.path.join(self.static_dir, f"{uuid.uuid4()}.mp4")
            ass_for_filter = ass_path.replace("\\", "/")

            await asyncio.to_thread(
                lambda: (
                    ffmpeg
                    .input(video_path)
                    .output(output_path, vf=f"subtitles={ass_for_filter}",
                            vcodec="libx264", crf=18, preset="fast", acodec="copy")
                    .overwrite_output()
                    .run(quiet=True)
                )
            )
            logger.info(f"Karaoke subtitles burned into {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error burning karaoke subtitles: {e}")
            return video_path

    async def burn_subtitles(self, video_path: str, subtitle_text: str) -> str:
        if not subtitle_text or not subtitle_text.strip():
            return video_path
        try:
            os.makedirs(self.static_dir, exist_ok=True)
            duration = await asyncio.to_thread(self._probe_duration, video_path)

            ass_path = os.path.join(self.static_dir, f"{uuid.uuid4()}.ass")
            with open(ass_path, "w", encoding="utf-8") as f:
                f.write(self._build_ass(subtitle_text, duration))

            output_path = os.path.join(self.static_dir, f"{uuid.uuid4()}.mp4")
            ass_for_filter = ass_path.replace("\\", "/")

            await asyncio.to_thread(
                lambda: (
                    ffmpeg
                    .input(video_path)
                    .output(
                        output_path, vf=f"subtitles={ass_for_filter}",
                        vcodec="libx264", crf=18, preset="fast", acodec="copy",
                    )
                    .overwrite_output()
                    .run(quiet=True)
                )
            )
            logger.info(f"Subtitles burned into {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error burning subtitles: {e}")
            return video_path

    # ── Audio mux ──────────────────────────────────────────────────────────

    async def mux_audio(
        self,
        video_path: str,
        audio_path: str,
        sfx_path: str = "",
        sfx_volume: float = 0.45,
        music_path: str = "",
        music_volume: float = 0.18,
        replace_existing_audio: bool = True,
    ) -> str:
        """Mix TTS + optional music + optional SFX into the video.

        When `replace_existing_audio=False`, the original audio of the input
        video is preserved and mixed *with* the new tracks.
        """
        try:
            os.makedirs(self.static_dir, exist_ok=True)
            output_path = os.path.join(self.static_dir, f"{uuid.uuid4()}.mp4")
            video_dur = await asyncio.to_thread(self._probe_duration, video_path)
            has_sfx   = bool(sfx_path   and os.path.exists(sfx_path))
            has_music = bool(music_path and os.path.exists(music_path))
            has_tts   = bool(audio_path and os.path.exists(audio_path))
            keep_orig = (not replace_existing_audio) and await asyncio.to_thread(
                self._probe_has_audio, video_path
            )

            def _run():
                in_v = ffmpeg.input(video_path)
                audio_inputs = []

                if keep_orig:
                    audio_inputs.append(in_v["a"])

                if has_music:
                    music_a = (
                        ffmpeg.input(music_path, stream_loop=-1)["a"]
                        .filter("volume", music_volume)
                        .filter("atrim", duration=video_dur)
                        .filter("afade", type="out", start_time=max(0, video_dur - 1.5), duration=1.5)
                    )
                    audio_inputs.append(music_a)

                if has_sfx:
                    sfx_a = (
                        ffmpeg.input(sfx_path, stream_loop=-1)["a"]
                        .filter("volume", sfx_volume)
                        .filter("atrim", duration=video_dur)
                    )
                    audio_inputs.append(sfx_a)

                if has_tts:
                    tts_a = ffmpeg.input(audio_path)["a"].filter("apad")
                    audio_inputs.append(tts_a)

                if not audio_inputs:
                    # Nothing to mix — just copy through.
                    (
                        ffmpeg.output(in_v["v"], output_path, vcodec="copy", t=video_dur)
                        .overwrite_output()
                        .run(quiet=True)
                    )
                    return

                n = len(audio_inputs)
                mixed = (
                    audio_inputs[0] if n == 1
                    else ffmpeg.filter(audio_inputs, "amix", inputs=n, duration="first", normalize=0)
                )

                (
                    ffmpeg
                    .output(in_v["v"], mixed, output_path,
                            vcodec="copy", acodec="aac", t=video_dur)
                    .overwrite_output()
                    .run(quiet=True)
                )

            await asyncio.to_thread(_run)
            mode = "+".join(filter(None, [
                "Orig" if keep_orig else "",
                "Music" if has_music else "",
                "SFX"   if has_sfx   else "",
                "TTS"   if has_tts   else "",
            ]))
            logger.info(f"Muxed audio ({mode}) → {output_path} (dur={video_dur:.2f}s)")
            return output_path
        except Exception as e:
            logger.error(f"Error muxing audio: {e}")
            return video_path

    # ── ffprobe utilities ──────────────────────────────────────────────────

    @staticmethod
    def _probe_duration(video_path: str) -> float:
        try:
            info = ffmpeg.probe(video_path)
            for stream in info.get("streams", []):
                if stream.get("codec_type") == "video" and stream.get("duration"):
                    return float(stream["duration"])
            if info.get("format", {}).get("duration"):
                return float(info["format"]["duration"])
        except Exception as e:
            logger.warning(f"ffprobe failed, defaulting duration: {e}")
        return 8.0

    @staticmethod
    def _probe_has_audio(video_path: str) -> bool:
        try:
            info = ffmpeg.probe(video_path)
            return any(
                s.get("codec_type") == "audio"
                for s in info.get("streams", [])
            )
        except Exception:
            return False

    async def extract_last_frame(self, video_path: str) -> str:
        try:
            os.makedirs(self.static_dir, exist_ok=True)
            duration = await asyncio.to_thread(self._probe_duration, video_path)
            seek = max(0.0, duration - 0.1)
            out_path = os.path.join(self.static_dir, f"{uuid.uuid4()}.png")
            await asyncio.to_thread(
                lambda: (
                    ffmpeg
                    .input(video_path, ss=seek)
                    .output(out_path, vframes=1)
                    .overwrite_output()
                    .run(quiet=True)
                )
            )
            logger.info(f"Last frame extracted → {out_path}")
            return out_path
        except Exception as e:
            logger.warning(f"extract_last_frame failed (non-fatal): {e}")
            return ""

    @staticmethod
    def _has_audio(video_path: str) -> bool:
        try:
            info = ffmpeg.probe(video_path)
            return any(s.get("codec_type") == "audio" for s in info.get("streams", []))
        except Exception:
            return False

    _TRANSITION_DURATIONS: dict[str, float] = {"cut": 0.0, "dissolve": 0.5, "fade": 0.8}

    async def _merge_two_clips(
        self, path_a: str, path_b: str, transition: str, has_audio: bool
    ) -> str:
        """Merge exactly two clips with the given transition type."""
        out = os.path.join(self.static_dir, f"{uuid.uuid4()}.mp4")
        t_dur = self._TRANSITION_DURATIONS.get(transition, 0.0)

        def _run():
            in_a = ffmpeg.input(path_a)
            in_b = ffmpeg.input(path_b)
            if transition == "cut" or t_dur <= 0:
                if has_audio:
                    j = ffmpeg.concat(in_a["v"], in_a["a"], in_b["v"], in_b["a"], v=1, a=1).node
                    (ffmpeg.output(j[0], j[1], out,
                                   vcodec="libx264", crf=18, preset="fast",
                                   pix_fmt="yuv420p", acodec="aac")
                     .overwrite_output().run(quiet=True))
                else:
                    j = ffmpeg.concat(in_a["v"], in_b["v"], v=1, a=0).node
                    (ffmpeg.output(j[0], out,
                                   vcodec="libx264", crf=18, preset="fast", pix_fmt="yuv420p")
                     .overwrite_output().run(quiet=True))
            else:
                dur_a = self._probe_duration(path_a)
                offset = max(0.0, dur_a - t_dur)
                v = ffmpeg.filter([in_a.video, in_b.video], "xfade",
                                  transition=transition, duration=t_dur, offset=offset)
                if has_audio:
                    a = ffmpeg.filter([in_a.audio, in_b.audio], "acrossfade", duration=t_dur)
                    (ffmpeg.output(v, a, out,
                                   vcodec="libx264", crf=18, preset="fast",
                                   pix_fmt="yuv420p", acodec="aac")
                     .overwrite_output().run(quiet=True))
                else:
                    (ffmpeg.output(v, out,
                                   vcodec="libx264", crf=18, preset="fast", pix_fmt="yuv420p")
                     .overwrite_output().run(quiet=True))

        try:
            await asyncio.to_thread(_run)
        except ffmpeg.Error as e:
            logger.error(
                f"_merge_two_clips({transition}) failed:\n"
                f"{e.stderr.decode(errors='replace') if e.stderr else e}"
            )
            raise
        return out

    async def concat_videos(
        self,
        segment_paths: list[str],
        crossfade: float = 0.5,
        transitions: list[str] | None = None,
    ) -> str:
        if len(segment_paths) == 1:
            return segment_paths[0]

        os.makedirs(self.static_dir, exist_ok=True)
        output_path = os.path.join(self.static_dir, f"{uuid.uuid4()}.mp4")
        has_audio = await asyncio.to_thread(self._has_audio, segment_paths[0])

        # ── Per-shot transitions path ─────────────────────────────────────────
        if transitions is not None:
            gaps = len(segment_paths) - 1
            t_list = (list(transitions) + ["cut"] * gaps)[:gaps]

            # Fast path: all cuts → simple concat demuxer
            if all(t == "cut" for t in t_list):
                return await self.concat_videos(segment_paths, crossfade=0)

            original_paths = set(segment_paths)
            temp_files: list[str] = []
            current = segment_paths[0]

            for next_clip, t in zip(segment_paths[1:], t_list):
                merged = await self._merge_two_clips(current, next_clip, t, has_audio)
                if current not in original_paths:
                    temp_files.append(current)
                current = merged

            for f in temp_files:
                try:
                    os.remove(f)
                except OSError:
                    pass

            logger.info(
                f"Concatenated {len(segment_paths)} segments with per-shot transitions → {current}"
            )
            return current

        if crossfade <= 0 or len(segment_paths) < 2:
            concat_list = os.path.join(self.static_dir, f"{uuid.uuid4()}_list.txt")
            with open(concat_list, "w", encoding="utf-8") as fh:
                for p in segment_paths:
                    fh.write(f"file '{p.replace(chr(92), '/')}'\n")

            def _run_simple():
                out_kwargs = {"vcodec": "copy"}
                if has_audio:
                    out_kwargs["acodec"] = "aac"
                try:
                    (
                        ffmpeg
                        .input(concat_list, format="concat", safe=0)
                        .output(output_path, **out_kwargs)
                        .overwrite_output()
                        .run(quiet=True)
                    )
                except ffmpeg.Error as e:
                    logger.error(f"concat_videos simple failed:\n{e.stderr.decode(errors='replace') if e.stderr else e}")
                    raise
            await asyncio.to_thread(_run_simple)
            logger.info(f"Concatenated {len(segment_paths)} segments → {output_path}")
            return output_path

        # Crossfade via xfade filter — requires re-encoding
        durations = await asyncio.gather(
            *[asyncio.to_thread(self._probe_duration, p) for p in segment_paths]
        )

        def _run_xfade():
            inputs = [ffmpeg.input(p) for p in segment_paths]
            v_streams = [inp.video for inp in inputs]

            offset = durations[0] - crossfade
            v = ffmpeg.filter([v_streams[0], v_streams[1]], "xfade",
                              transition="fade", duration=crossfade, offset=offset)
            if has_audio:
                a_streams = [inp.audio for inp in inputs]
                a = ffmpeg.filter([a_streams[0], a_streams[1]], "acrossfade",
                                  duration=crossfade)

            for i in range(2, len(segment_paths)):
                offset += durations[i - 1] - crossfade
                v = ffmpeg.filter([v, v_streams[i]], "xfade",
                                  transition="fade", duration=crossfade, offset=offset)
                if has_audio:
                    a = ffmpeg.filter([a, a_streams[i]], "acrossfade",
                                      duration=crossfade)

            try:
                if has_audio:
                    (
                        ffmpeg
                        .output(v, a, output_path,
                                vcodec="libx264", acodec="aac",
                                pix_fmt="yuv420p", crf=18)
                        .overwrite_output()
                        .run(quiet=True)
                    )
                else:
                    (
                        ffmpeg
                        .output(v, output_path,
                                vcodec="libx264", pix_fmt="yuv420p", crf=18)
                        .overwrite_output()
                        .run(quiet=True)
                    )
            except ffmpeg.Error as e:
                logger.error(f"concat_videos xfade failed:\n{e.stderr.decode(errors='replace') if e.stderr else e}")
                raise

        await asyncio.to_thread(_run_xfade)
        logger.info(f"Concatenated {len(segment_paths)} segments with {crossfade}s crossfade → {output_path}")
        return output_path

    async def append_outro(self, video_path: str, outro_path: str) -> str:
        if not outro_path or not os.path.exists(outro_path):
            return video_path
        try:
            os.makedirs(self.static_dir, exist_ok=True)
            output_path = os.path.join(self.static_dir, f"{uuid.uuid4()}.mp4")
            outro_has_audio = await asyncio.to_thread(self._probe_has_audio, outro_path)
            outro_dur = await asyncio.to_thread(self._probe_duration, outro_path)

            def _run():
                main_in  = ffmpeg.input(video_path)
                outro_in = ffmpeg.input(outro_path)

                v0 = main_in["v"].filter("scale", 1080, 1920)
                v1 = outro_in["v"].filter("scale", 1080, 1920)
                a0 = main_in["a"]
                if outro_has_audio:
                    a1 = outro_in["a"]
                else:
                    a1 = ffmpeg.input(
                        "anullsrc=r=44100:cl=stereo",
                        format="lavfi",
                        t=outro_dur,
                    )["a"]

                joined = ffmpeg.concat(v0, a0, v1, a1, v=1, a=1).node
                (
                    ffmpeg
                    .output(
                        joined[0], joined[1], output_path,
                        vcodec="libx264", crf=23, preset="fast", acodec="aac",
                    )
                    .overwrite_output()
                    .run(quiet=True)
                )

            await asyncio.to_thread(_run)
            logger.info(f"Outro appended ({outro_dur:.1f}s) → {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error appending outro: {e}")
            return video_path

    # ── Colour grade + speed ───────────────────────────────────────────────

    GRADE_DEFAULTS = {"contrast": 1.12, "saturation": 1.25, "sharpen": 0.8}

    @classmethod
    def parse_grade_params(cls, raw: str | None) -> dict:
        params = dict(cls.GRADE_DEFAULTS)
        if not raw:
            return params
        try:
            parts = [p.strip() for p in raw.split(",")]
            keys = ("contrast", "saturation", "sharpen")
            for key, val in zip(keys, parts):
                if val:
                    params[key] = float(val)
        except (ValueError, TypeError) as e:
            logger.warning(f"Bad grade params {raw!r}, using defaults: {e}")
            return dict(cls.GRADE_DEFAULTS)
        params["contrast"]   = max(0.5, min(2.0, params["contrast"]))
        params["saturation"] = max(0.0, min(3.0, params["saturation"]))
        params["sharpen"]    = max(0.0, min(2.0, params["sharpen"]))
        return params

    async def apply_grade(self, video_path: str, params: dict | None = None) -> str:
        p = params or dict(self.GRADE_DEFAULTS)
        try:
            os.makedirs(self.static_dir, exist_ok=True)
            output_path = os.path.join(self.static_dir, f"{uuid.uuid4()}.mp4")
            vf = (
                f"eq=contrast={p['contrast']}:saturation={p['saturation']},"
                f"unsharp=5:5:{p['sharpen']}:5:5:0.0"
            )
            await asyncio.to_thread(
                lambda: (
                    ffmpeg
                    .input(video_path)
                    .output(output_path, vf=vf, vcodec="libx264", crf=18, preset="fast", acodec="copy")
                    .overwrite_output()
                    .run(quiet=True)
                )
            )
            logger.info(f"Colour grade applied: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error applying colour grade: {e}")
            return video_path

    async def apply_speed(self, video_path: str, speed: float) -> str:
        if abs(speed - 1.0) < 0.01:
            return video_path
        try:
            os.makedirs(self.static_dir, exist_ok=True)
            output_path = os.path.join(self.static_dir, f"{uuid.uuid4()}.mp4")
            has_audio = await asyncio.to_thread(self._probe_has_audio, video_path)
            pts_factor = 1.0 / speed

            def _run():
                in_v = ffmpeg.input(video_path)
                vstream = in_v["v"].filter("setpts", f"{pts_factor:.4f}*PTS")
                if has_audio:
                    astream = in_v["a"]
                    remaining = speed
                    while remaining > 2.0:
                        astream = astream.filter("atempo", 2.0)
                        remaining /= 2.0
                    while remaining < 0.5:
                        astream = astream.filter("atempo", 0.5)
                        remaining /= 0.5
                    astream = astream.filter("atempo", remaining)
                    (
                        ffmpeg.output(vstream, astream, output_path,
                                      vcodec="libx264", crf=18, preset="fast", acodec="aac")
                        .overwrite_output()
                        .run(quiet=True)
                    )
                else:
                    (
                        ffmpeg.output(vstream, output_path,
                                      vcodec="libx264", crf=18, preset="fast", an=None)
                        .overwrite_output()
                        .run(quiet=True)
                    )

            await asyncio.to_thread(_run)
            logger.info(f"Speed {speed:.2f}x applied → {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error applying speed: {e}")
            return video_path

    # ── Reference video analysis ───────────────────────────────────────────

    # Gemini generateContent caps the whole request at ~20 MB; anything close
    # to that must go through the Files API instead of inline bytes.
    _INLINE_VIDEO_LIMIT = 15 * 1024 * 1024
    _MAX_REFERENCE_DURATION = 300  # 5 min — longer refs are slow & expensive to analyze
    _VOICEOVER_WORDS_PER_SECOND = 2.2

    async def _upload_video_for_analysis(self, video_path: str):
        """Upload a video via the Gemini Files API and wait until it's ACTIVE."""
        uploaded = await asyncio.to_thread(self.client.files.upload, file=video_path)
        # Video files are processed asynchronously; poll until ready.
        waited = 0
        while getattr(uploaded.state, "name", str(uploaded.state)) == "PROCESSING":
            if waited >= 120:
                raise Exception("Gemini Files API: video processing timeout (120s)")
            await asyncio.sleep(3)
            waited += 3
            uploaded = await asyncio.to_thread(self.client.files.get, name=uploaded.name)
        state = getattr(uploaded.state, "name", str(uploaded.state))
        if state != "ACTIVE":
            raise Exception(f"Gemini Files API: video upload failed (state={state})")
        return uploaded

    @_retry_cheap
    async def analyze_reference_video(
        self,
        video_path: str,
        target_duration: int = 30,
        num_scenes: int = 0,
        min_shot_seconds: int = 4,
        max_shot_seconds: int = 15,
    ) -> dict:
        """Analyze a reference video and extract its structure as a script.

        The reference and the output may have very different durations: the
        prompt asks Gemini to preserve the *proportional* pacing (hook /
        build-up / payoff) rather than absolute timestamps, and the voiceover
        is rewritten to a word budget that fits `target_duration`. Final shot
        durations are additionally renormalized in code by
        `_normalize_shot_durations` before generation, so the output always
        matches the target regardless of what the LLM returns.

        Returns a dict in the same shots format `parse_script_json` validates:
        {
            "title": ..., "voiceover": ...,
            "metadata": {"detected_language", "detected_tone", "detected_tempo",
                         "reference_duration_seconds"},
            "shots": [{"scene_prompt", "duration_seconds", "transition"}, ...]
        }
        """
        try:
            duration_secs = await asyncio.to_thread(self._probe_duration, video_path)
            if duration_secs > self._MAX_REFERENCE_DURATION:
                raise Exception(
                    f"Reference video is too long ({duration_secs:.0f}s). "
                    f"Maximum is {self._MAX_REFERENCE_DURATION}s — send a shorter clip."
                )

            if num_scenes <= 0:
                num_scenes = max(1, round(target_duration / 10))
            vo_word_budget = int(target_duration * self._VOICEOVER_WORDS_PER_SECOND)

            system_prompt = (
                "You are a video content analyzer for a vertical short-video (9:16) production pipeline.\n"
                "Analyze the provided reference video and design a NEW video with the same vibe, "
                "pacing and narrative formula — not a copy.\n"
                "Return ONLY a valid JSON object with this exact structure:\n"
                "{\n"
                '  "title": "catchy hook or title — MUST be in the same language as the reference speech/captions",\n'
                '  "voiceover": "rewritten narration that fits the target duration, empty string if the reference has no speech",\n'
                '  "image_prompt": "one rich ENGLISH description (60-100 words) of the main character AND the visual style '
                '(art style, palette, lighting, mood) — used to generate reference photos that anchor every scene",\n'
                '  "sfx_description": "short ENGLISH description (10-25 words) of the ambient sound design / atmosphere '
                'of the reference (e.g. rain, crowd murmur, tense drone) — NEVER name specific songs or artists",\n'
                '  "metadata": {\n'
                '    "detected_language": "en",\n'
                '    "detected_tone": "upbeat|serious|funny|educational|mixed",\n'
                '    "detected_tempo": "fast|medium|slow"\n'
                '  },\n'
                '  "shots": [\n'
                '    {"scene_prompt": "detailed visual description, 40-80 words", '
                '"duration_seconds": 5, "transition": "cut|dissolve|fade"},\n'
                '    ...\n'
                '  ]\n'
                "}\n\n"
                "TIMING RULES:\n"
                f"- Reference duration: {duration_secs:.1f}s. Target output duration: {target_duration}s. "
                "They may differ a lot — preserve the PROPORTIONAL pacing (how much of the total time the hook, "
                "build-up and payoff take), never absolute timestamps.\n"
                f"- Produce exactly {num_scenes} shots; each duration_seconds between "
                f"{min_shot_seconds} and {max_shot_seconds}, summing to ~{target_duration}s.\n"
                "- If the reference cuts faster than the shot limits allow, merge several reference "
                "moments into one shot description that carries the same energy.\n"
                f"- The voiceover must fit the target duration when spoken: MAXIMUM {vo_word_budget} words. "
                "Compress or rewrite the reference narration, keep its language and tone.\n\n"
                "VISUAL RULES:\n"
                "- Every scene_prompt must describe a VERTICAL 9:16 composition, even if the reference is horizontal.\n"
                "- NEVER ask for on-screen text, captions or subtitles inside scene_prompt — AI generators "
                "render text badly. Put textual content into the voiceover or title instead.\n"
                "- Do NOT identify real people or brands: describe generic appearance "
                "(e.g. 'a young woman with long dark hair'), never names or logos.\n"
                "- No URLs, copyright notices or watermarks in descriptions.\n"
                "- Focus on: camera angle & motion, lighting, color palette, subject action, composition.\n"
                "- Each scene_prompt must be RICH (40-80 words): what happens, how the camera moves, "
                "what the light and colors do, what emotion the moment carries. Thin one-line prompts "
                "produce flat, generic clips.\n"
                "- If one recurring character appears in several shots, describe them IDENTICALLY "
                "in every shot where they appear (same wording), so generated scenes stay consistent."
            )

            user_prompt = (
                f"Analyze this {duration_secs:.1f}s reference video and design a {target_duration}s "
                f"video with the same formula: same hook style, same pacing feel, same emotional arc."
            )

            # Small files go inline (SDK base64-encodes raw bytes itself);
            # bigger ones must use the Files API to stay under the request cap.
            file_size = os.path.getsize(video_path)
            if file_size <= self._INLINE_VIDEO_LIMIT:
                with open(video_path, "rb") as f:
                    video_bytes = f.read()
                video_part = types.Part(
                    inline_data=types.Blob(mime_type="video/mp4", data=video_bytes)
                )
            else:
                uploaded = await self._upload_video_for_analysis(video_path)
                video_part = types.Part(
                    file_data=types.FileData(
                        file_uri=uploaded.uri,
                        mime_type=uploaded.mime_type or "video/mp4",
                    )
                )

            contents = [
                types.Content(role="user", parts=[types.Part(text=user_prompt), video_part]),
            ]
            config = types.GenerateContentConfig(system_instruction=system_prompt)

            # Verified against ListModels for this key: gemini-3.5-flash exists,
            # "gemini-3.1-pro" does NOT (only -preview). 2.5 family kept as a
            # proven multimodal fallback.
            models_to_try = ["gemini-3.5-flash", "gemini-2.5-pro", "gemini-2.5-flash"]
            response = None
            last_error = None
            for model_name in models_to_try:
                try:
                    response = await asyncio.to_thread(
                        self.client.models.generate_content,
                        model=model_name,
                        contents=contents,
                        config=config,
                    )
                    logger.info(f"Video analysis successful with model {model_name}")
                    break
                except Exception as model_error:
                    last_error = model_error
                    logger.warning(f"Model {model_name} failed, trying next: {model_error}")
                    continue

            if response is None:
                raise last_error or Exception("All Gemini models failed for video analysis")

            text = getattr(response, "text", None)
            if not text or not text.strip():
                raise Exception("Empty response from video analysis")

            result = self.parse_script_json(text)
            if result is None:
                raise Exception(f"Failed to parse video analysis response: {text[:500]}")

            result.setdefault("metadata", {})
            if isinstance(result["metadata"], dict):
                result["metadata"]["reference_duration_seconds"] = round(duration_secs, 1)

            logger.info(
                f"Reference video analyzed: {len(result.get('shots', []))} shots, "
                f"ref={duration_secs:.1f}s → target={target_duration}s"
            )
            return result

        except Exception as e:
            # Full traceback into the log — encoding/SDK errors are impossible
            # to localize from the one-line message alone.
            logger.exception("Error analyzing reference video")
            raise Exception(f"Failed to analyze video [{type(e).__name__}]: {str(e)}")

    @_retry_cheap
    async def refine_reference_formula(
        self,
        analysis: dict,
        instruction: str,
        target_duration: int = 30,
        num_scenes: int = 0,
        min_shot_seconds: int = 4,
        max_shot_seconds: int = 15,
        model: str = "",
    ) -> dict:
        """Rework an existing formula with a text instruction — no video needed.

        Used for: user edits ("make the hero a woman, more serious tone"),
        re-timing to a new target duration, and adapting a saved library
        formula to a new topic. Returns the same JSON structure as
        `analyze_reference_video`; raises if the model returns invalid JSON.
        """
        import json as _json

        if num_scenes <= 0:
            num_scenes = max(1, round(target_duration / 10))
        vo_word_budget = int(target_duration * self._VOICEOVER_WORDS_PER_SECOND)

        system_prompt = (
            "You are editing a video formula (a JSON script for a vertical 9:16 AI-generated video).\n"
            "Apply the user's instruction to the given formula and return ONLY the updated JSON with the "
            "SAME structure and ALL the same keys (title, voiceover, image_prompt, sfx_description, "
            "metadata, shots[]). Keep everything the instruction doesn't touch.\n\n"
            "HARD RULES (always enforce, even after edits):\n"
            f"- Exactly {num_scenes} shots; each duration_seconds between {min_shot_seconds} and "
            f"{max_shot_seconds}, summing to ~{target_duration}s.\n"
            f"- voiceover fits the duration when spoken: MAXIMUM {vo_word_budget} words, keep its language "
            "unless the instruction says otherwise.\n"
            "- title in the same language as the voiceover; image_prompt and sfx_description in English.\n"
            "- scene_prompt: rich 40-80 word visual descriptions, vertical 9:16 composition, no on-screen "
            "text, no real people's names or brand logos.\n"
            "- If a recurring character appears in several shots, describe them identically in each."
        )
        user_prompt = (
            "Current formula:\n```json\n"
            + _json.dumps(analysis, ensure_ascii=False, indent=2)
            + "\n```\n\nInstruction: "
            + instruction.strip()
        )

        text = await self.generate_text(user_prompt, system_prompt, model)
        result = self.parse_script_json(text)
        if result is None:
            raise Exception(f"Formula refine returned invalid JSON: {text[:300]}")

        # Preserve fields the model may have dropped.
        for key in ("image_prompt", "sfx_description"):
            result.setdefault(key, analysis.get(key, ""))
        result.setdefault("metadata", {})
        if isinstance(result["metadata"], dict) and isinstance(analysis.get("metadata"), dict):
            for mk, mv in analysis["metadata"].items():
                result["metadata"].setdefault(mk, mv)

        logger.info(
            f"Formula refined: {len(result.get('shots', []))} shots, instruction={instruction[:80]!r}"
        )
        return result
