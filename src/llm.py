"""
LLM-provider seam. Supports two backends behind one small interface:

  - "anthropic": Claude native tool-use  (src/agent.py:AirportAgent)
  - "gemini":    Gemini function-calling (src/agent_gemini.py:GeminiAirportAgent)

Provider selection (see `provider()`):
  1. explicit LLM_PROVIDER=anthropic|gemini, else
  2. auto: a Gemini key present with no Anthropic key -> gemini, else anthropic.

The deterministic core, tools, system prompt, and UI are provider-agnostic —
only the per-turn model call + message/tool-result shapes differ between the two
agent classes. `src.agent.make_agent()` returns the right one.
"""
from __future__ import annotations

import os

ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
MAX_TOKENS = int(os.environ.get("ANTHROPIC_MAX_TOKENS", "4096"))
MAX_OUTPUT_TOKENS = int(os.environ.get("GEMINI_MAX_TOKENS", "4096"))
# gemini-2.5-flash does function calling fine with thinking off (0). Bump this
# (e.g. 1024) if you point GEMINI_MODEL at a 3.x model that needs thinking.
GEMINI_THINKING_BUDGET = int(os.environ.get("GEMINI_THINKING_BUDGET", "0"))


def _secret(name: str) -> str | None:
    """Read a config value: environment variable first, then Streamlit Cloud
    secrets (`st.secrets`). Lets the SAME code use a local .env when run locally
    and a hosted secret when deployed to Streamlit Community Cloud — no code
    change between the two."""
    v = os.environ.get(name)
    if v:
        return v
    try:
        import streamlit as st
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return None


def _gemini_key() -> str | None:
    return _secret("GEMINI_API_KEY") or _secret("GOOGLE_API_KEY")


def _use_vertex() -> bool:
    return ((_secret("GOOGLE_GENAI_USE_VERTEXAI") or "").lower() in ("1", "true")
            or (_secret("GEMINI_VERTEX") or "").lower() in ("1", "true"))


def _anthropic_available() -> bool:
    if _secret("ANTHROPIC_API_KEY"):
        return True
    return os.path.isdir(os.path.expanduser("~/.config/anthropic"))


def provider() -> str:
    p = (_secret("LLM_PROVIDER") or "").strip().lower()
    if p in ("anthropic", "gemini"):
        return p
    if (_gemini_key() or _use_vertex()) and not _secret("ANTHROPIC_API_KEY"):
        return "gemini"
    return "anthropic"


def active_model() -> str:
    return GEMINI_MODEL if provider() == "gemini" else ANTHROPIC_MODEL


def have_credentials() -> bool:
    if provider() == "gemini":
        return bool(_gemini_key()) or _use_vertex()
    return _anthropic_available()


def has_gemini() -> bool:
    """Whether a Gemini backend is reachable — used by voice (Gemini transcribes
    the mic audio regardless of which provider runs the chat agent)."""
    return bool(_gemini_key()) or _use_vertex()


# --- clients ----------------------------------------------------------------
def get_anthropic_client():
    import anthropic
    key = _secret("ANTHROPIC_API_KEY")
    return anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()


def get_gemini_client():
    from google import genai
    if _use_vertex():
        return genai.Client(
            vertexai=True,
            project=_secret("GOOGLE_CLOUD_PROJECT") or _secret("VERTEX_PROJECT"),
            location=_secret("GOOGLE_CLOUD_LOCATION")
            or _secret("VERTEX_LOCATION") or "global",
        )
    return genai.Client(api_key=_gemini_key())


# Backward-compat alias (older code referenced llm.MODEL / llm.get_client)
MODEL = active_model()
get_client = get_anthropic_client
