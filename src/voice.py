"""
Voice input for the chat: transcribe recorded microphone audio to text with
Gemini (multimodal), so a spoken question becomes a normal agent turn.

Kept deliberately small and provider-honest: Claude can't ingest audio, so we
always use Gemini for speech-to-text — independent of which LLM runs the chat
agent. If no Gemini backend is configured, `transcribe()` returns None and the
UI simply hides the mic (no fake capability).

Anti-fabrication (important): a silent / empty recording must NEVER become a
made-up question. Two guards enforce that:
  1. We measure the audio locally (duration + loudness) and DON'T even call the
     model when the clip is too short or effectively silent.
  2. The transcription prompt is neutral — it carries no domain words to parrot —
     and instructs the model to emit NO_SPEECH (which we map to None) when it
     can't hear intelligible speech.
"""
from __future__ import annotations

import io
import wave

from . import llm

try:  # stdlib in 3.11/3.12; used only for a loudness (RMS) reading
    import audioop
except Exception:  # pragma: no cover
    audioop = None

# Neutral on purpose: NO airport/congestion/expansion hints — content words in
# the prompt get parroted back on silent audio (that was the fabrication bug).
_STT_PROMPT = (
    "You are a strict speech-to-text transcriber. Transcribe the words actually "
    "spoken in this audio, verbatim. Output ONLY those words — no quotes, no "
    "notes, no translation. If the audio has no intelligible speech (silence, "
    "noise, or empty), output exactly NO_SPEECH and nothing else. Never guess or "
    "invent words that are not clearly audible."
)

# Reject clips shorter than this or quieter than this before spending a model call.
_MIN_DURATION_S = 0.45
_MIN_RMS = 90  # 16-bit PCM: true silence ~<30, room noise ~<80, speech in the 100s+


def available() -> bool:
    return llm.has_gemini()


def _audio_stats(b: bytes):
    """(duration_seconds, rms) for a WAV clip, or (None, None) if unparseable."""
    try:
        with wave.open(io.BytesIO(b), "rb") as w:
            nframes = w.getnframes()
            rate = w.getframerate() or 1
            width = w.getsampwidth()
            frames = w.readframes(nframes)
        duration = nframes / float(rate)
        rms = audioop.rms(frames, width) if (audioop and frames) else None
        return duration, rms
    except Exception:
        return None, None


def is_silent(audio_bytes: bytes) -> bool:
    """True when the clip is too short or too quiet to contain a real question."""
    dur, rms = _audio_stats(audio_bytes)
    if dur is not None and dur < _MIN_DURATION_S:
        return True
    if rms is not None and rms < _MIN_RMS:
        return True
    return False


def transcribe(audio_bytes: bytes, mime_type: str = "audio/wav") -> str | None:
    """Transcribed text, or None if empty/silent/failed (caller degrades gracefully)."""
    if not audio_bytes or not available():
        return None
    # Guard 1: don't transcribe an empty/silent recording — that is what let the
    # model fabricate a question from nothing.
    if is_silent(audio_bytes):
        return None
    try:
        from google.genai import types

        client = llm.get_gemini_client()
        resp = client.models.generate_content(
            model=llm.GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                _STT_PROMPT,
            ],
        )
        text = (getattr(resp, "text", "") or "").strip()
        if len(text) >= 2 and text[0] in "\"'" and text[-1] == text[0]:
            text = text[1:-1].strip()
        # Guard 2: the model's own "no speech" signal.
        if not text or text.strip().upper().strip(".!") == "NO_SPEECH":
            return None
        return text
    except Exception:
        return None
