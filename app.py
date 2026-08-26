"""
Streamlit chat UI for the Airport Investment Intelligence Agent.

Run:  streamlit run app.py
(Set ANTHROPIC_API_KEY first — see .env.example. The sidebar dashboard works
without a key; the chat needs one.)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


# --- load .env (tiny, dependency-free) --------------------------------------
def _load_env():
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_env()

from src import llm, scoring  # noqa: E402
from src.data_layer import load_meta  # noqa: E402

st.set_page_config(page_title="Airport Investment Intelligence Agent",
                   page_icon="🛫", layout="wide")


@st.cache_data(show_spinner=False)
def _scored_df() -> pd.DataFrame:
    return scoring.build_scored()


@st.cache_data(show_spinner=False)
def _meta() -> dict:
    return load_meta()


# ---------------------------------------------------------------------------
# Sidebar: scope, assumptions, and a national opportunity dashboard
# ---------------------------------------------------------------------------
def render_sidebar():
    meta = _meta()
    df = _scored_df()
    with st.sidebar:
        st.header("Scope & assumptions")
        st.caption(
            f"**{meta.get('n_airports', len(df))}** US commercial airports · "
            f"FAA CY2024 enplanements · BTS On-Time months "
            f"{', '.join(meta.get('ontime_months', []))}."
        )
        a = meta.get("assumptions", {})
        with st.expander("Methodology & caveats", expanded=False):
            st.markdown(
                "**Expansion Opportunity Score (EOS)** = weighted national "
                "percentiles:\n"
                "- Congestion 35% · Growth 30% · Scale 20% · Utilization 15%\n\n"
                f"- Long-haul = stage length ≥ {a.get('long_haul_miles', 2000)} mi. "
                f"{a.get('long_haul_scope', '')}\n"
                f"- Congestion basis: {a.get('congestion_basis', '')}\n"
                "- *Unmet demand* is a transparent proxy, not a forecast."
            )

        st.subheader("Top expansion candidates")
        top = df.sort_values("eos", ascending=False).head(12)
        st.dataframe(
            top[["iata", "name", "state", "eos", "congestion_level"]]
            .rename(columns={"iata": "IATA", "name": "Airport", "state": "ST",
                             "eos": "EOS", "congestion_level": "Congestion"}),
            hide_index=True, use_container_width=True,
        )

        st.subheader("National opportunity map")
        st.caption("Top-40 airports by EOS (bubble = score).")
        m = df.sort_values("eos", ascending=False).head(40)[["lat", "lon", "eos"]].dropna()
        m = m.rename(columns={"lat": "latitude", "lon": "longitude"})
        m["size"] = (m["eos"] ** 2) / 6
        try:
            st.map(m, size="size", color="#e4572e")
        except TypeError:  # older streamlit without size/color kwargs
            st.map(m[["latitude", "longitude"]])


# ---------------------------------------------------------------------------
# Main: chat
# ---------------------------------------------------------------------------
def get_agent():
    if "agent" not in st.session_state:
        from src.agent import AirportAgent
        st.session_state.agent = AirportAgent()
    return st.session_state.agent


EXAMPLES = [
    "Which airports in New England are strong candidates for terminal expansion?",
    "Compare LA and Santa Ana airport congestion levels.",
    "What percentage of flights out of Anchorage are long-haul?",
    "What is the unmet flight demand at SFO, and why?",
]


def render_chat():
    st.title("🛫 Airport Investment Intelligence Agent")
    st.caption("Ask about US airport expansion opportunities. Answers are grounded "
               "in a deterministic scoring engine — expand *How I got this* to see the tools.")

    if "history" not in st.session_state:
        st.session_state.history = []

    has_key = llm.have_credentials()
    if not has_key:
        st.warning("No `ANTHROPIC_API_KEY` found — the chat is disabled, but the "
                   "sidebar dashboard is fully live. Set a key (see `.env.example`) "
                   "and rerun to enable the agent.")

    # example chips
    cols = st.columns(len(EXAMPLES))
    for c, ex in zip(cols, EXAMPLES):
        if c.button(ex, use_container_width=True, disabled=not has_key):
            st.session_state.pending = ex

    # replay history
    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("trace"):
                with st.expander("How I got this (tool calls)"):
                    for t in turn["trace"]:
                        st.markdown(f"**`{t['tool']}`** · input `{t['input']}`")
                        st.code(t["result_preview"], language="json")

    prompt = st.chat_input("Ask about airport expansion opportunities…", disabled=not has_key)
    prompt = prompt or st.session_state.pop("pending", None)

    if prompt:
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing…"):
                result = get_agent().chat(prompt)
            if result.get("error"):
                st.error(result["error"])
                reply = f"*(error: {result['error']})*"
            else:
                reply = result["reply"]
                st.markdown(reply)
                if result.get("trace"):
                    with st.expander("How I got this (tool calls)"):
                        for t in result["trace"]:
                            st.markdown(f"**`{t['tool']}`** · input `{t['input']}`")
                            st.code(t["result_preview"], language="json")
        st.session_state.history.append(
            {"role": "assistant", "content": reply, "trace": result.get("trace")})


def main():
    try:
        _scored_df()
    except FileNotFoundError:
        st.error("Data snapshot not found. Run `python data/build_dataset.py` first.")
        return
    render_sidebar()
    render_chat()


main()
