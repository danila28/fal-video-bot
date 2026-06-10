"""Gemini text generation + ffmpeg post-processing.

This service replaces video-gen-bot's VertexService for fal-video-bot.
Video generation lives in dedicated fal.ai services (Kling / OmniHuman / Seedance);
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
            {"name": "gemini-2.5-flash-lite",  "price": "💰 ~$0.10/1M tok"},
            {"name": "gemini-2.5-flash",       "price": "⚖️  ~$0.15/1M tok"},
            {"name": "gemini-2.5-pro",         "price": "🏆 ~$1.25/1M tok"},
        ]

    def get_image_models(self):
        # Imagen 4 Fast = "банано" — primary photo generator.
        # fal-ai/flux-pro/v1.1 is the automatic fallback inside ImageGenService.
        return [
            {"name": "imagen-4.0-fast-generate-001", "price": "💰 ~$0.02/img"},
            {"name": "gemini-2.5-flash-image",       "price": "💰 ~$0.04/img"},
            {"name": "fal-ai/flux-pro/v1.1",         "price": "⚖️  ~$0.04/img"},
        ]

    def get_video_models(self):
        # fal.ai models. Pricing — approximate, verify at fal.ai/pricing.
        return [
            # Lip-sync talking head — produces video with embedded audio from photo + audio.
            {"name": "kling",     "price": "🎙 Lip-sync · ~$0.28/sec"},
            {"name": "omnihuman", "price": "🎙 Lip-sync · ~$0.40/sec"},
            # Scene-by-scene visual storytelling — no embedded audio (TTS added later).
            {"name": "seedance",  "price": "🎬 Scene clips · ~$0.05/sec"},
        ]

    # ── Text generation ────────────────────────────────────────────────────

    @_retry_cheap
    async def generate_text(
        self, prompt: str, system_prompt: str = "", model: str = "gemini-2.5-flash-lite"
    ) -> str:
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
        video is preserved and mixed *with* the new tracks — used for Kling /
        OmniHuman lip-sync videos that already contain the voice.
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

    async def concat_videos(self, segment_paths: list[str]) -> str:
        if len(segment_paths) == 1:
            return segment_paths[0]

        os.makedirs(self.static_dir, exist_ok=True)
        concat_list = os.path.join(self.static_dir, f"{uuid.uuid4()}_list.txt")
        output_path = os.path.join(self.static_dir, f"{uuid.uuid4()}.mp4")

        with open(concat_list, "w", encoding="utf-8") as fh:
            for p in segment_paths:
                fh.write(f"file '{p.replace(chr(92), '/')}'\n")

        def _run():
            (
                ffmpeg
                .input(concat_list, format="concat", safe=0)
                .output(output_path, vcodec="copy", acodec="aac")
                .overwrite_output()
                .run(quiet=True)
            )

        await asyncio.to_thread(_run)
        logger.info(f"Concatenated {len(segment_paths)} segments → {output_path}")
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
