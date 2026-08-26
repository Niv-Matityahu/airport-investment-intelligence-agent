"""
Thin LLM-provider seam.

The agent is built on Anthropic's native tool-use loop (see agent.py). Isolating
the client + model here keeps that swap small: to move to another provider you
replace `get_client()` / `MODEL` and adapt the message/So tool-result shapes in
agent.py. We deliberately do NOT build a heavy multi-provider abstraction — the
exam values clarity over speculative generality.

Model default: claude-opus-4-8 (strongest reasoning, best for clear
explanations). Override with ANTHROPIC_MODEL, e.g. claude-sonnet-5 for a
cheaper/faster demo ($3/$15 per 1M tok vs Opus $5/$25).
"""
from __future__ import annotations

import os

import anthropic

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8")
MAX_TOKENS = int(os.environ.get("ANTHROPIC_MAX_TOKENS", "4096"))


def get_client() -> anthropic.Anthropic:
    """Anthropic client. Reads ANTHROPIC_API_KEY (or an `ant auth login` profile)."""
    return anthropic.Anthropic()


def have_credentials() -> bool:
    """Best-effort check so the UI can show a friendly message instead of crashing."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True
    # an `ant auth login` profile also works; assume present if the config dir exists
    cfg = os.path.expanduser("~/.config/anthropic")
    return os.path.isdir(cfg)
