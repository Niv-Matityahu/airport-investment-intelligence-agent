"""
Airport Investment Intelligence — dark enterprise dashboard.

Layout:  top bar  |  [ conversational agent | map + rankings | charts + signals ]
The rankings/map/charts are deterministic (real scoring data); the left panel is
the grounded chat agent. Run:  streamlit run app.py
"""
from __future__ import annotations

import html as _html
import os
import sys
import time
from pathlib import Path

import markdown as _md
import pandas as pd
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _load_env():
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_env()

from src import llm, scoring, voice  # noqa: E402
from src.data_layer import airports_in_region, list_known_regions, load_meta  # noqa: E402

st.set_page_config(page_title="Airport Investment Intelligence", page_icon="🛫",
                   layout="wide", initial_sidebar_state="collapsed")

# ---- palette ---------------------------------------------------------------
RAMP = ["#22d3ee", "#38bdf8", "#3b82f6", "#818cf8", "#c084fc"]  # EOS low->high
GOOD, WARN, BAD = "#22c55e", "#f59e0b", "#ef4444"

# theme palettes (dark = default / primary; light = alternate)
THEMES = {
    "dark": dict(BG="#0b1524", PANEL="#0f1e33", PANEL2="#122239", TOPBAR="#0e1c30",
                 BORDER="#1c3350", BORDER2="#23405f", TEXT="#e6edf5", TEXT2="#cbd8e8",
                 MUTED="#7f96b3", BUBBLEBOT="#15263d", CHIP="#16304c", INPUTBG="#0d1c30",
                 INPUTTEXT="#e6edf5", BTNBG="#122a44", HOVER="#132743", TILE="#12233b",
                 ACCENTTEXT="#5fe0f5", LOGO="#22d3ee", CODEBG="#0b1524",
                 MAPSTYLE="dark", MAPTEXT=[230, 240, 250], MAPSTROKE=[120, 220, 245]),
    "light": dict(BG="#eef2f8", PANEL="#ffffff", PANEL2="#f4f8ff", TOPBAR="#ffffff",
                  BORDER="#dbe4f0", BORDER2="#cdd9ea", TEXT="#0c1a33", TEXT2="#33415c",
                  MUTED="#5f6f88", BUBBLEBOT="#f1f5fb", CHIP="#e2eefc", INPUTBG="#ffffff",
                  INPUTTEXT="#0c1a33", BTNBG="#eef4fc", HOVER="#eef4fc", TILE="#f4f8ff",
                  ACCENTTEXT="#0e7490", LOGO="#0891b2", CODEBG="#eef3fb",
                  MAPSTYLE="light", MAPTEXT=[30, 45, 70], MAPSTROKE=[30, 90, 140]),
}


def _pal():
    return THEMES[st.session_state.get("theme", "dark")]


def _rgb(h): return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def ramp_color(t):
    t = max(0.0, min(1.0, t)); n = len(RAMP) - 1
    i = min(int(t * n), n - 1); f = t * n - i
    a, b = _rgb(RAMP[i]), _rgb(RAMP[i + 1])
    return "#%02x%02x%02x" % tuple(round(a[j] + (b[j] - a[j]) * f) for j in range(3))


def eos_color(v, lo, hi): return ramp_color(0.5 if hi <= lo else (v - lo) / (hi - lo))


@st.cache_data(show_spinner=False)
def _scored(): return scoring.build_scored()


@st.cache_data(show_spinner=False)
def _meta(): return load_meta()


# ---------------------------------------------------------------------------
_CSS = """
<style>
  #MainMenu, footer, header[data-testid="stHeader"] {visibility:hidden; height:0;}
  .stApp {background:@BG@; color:@TEXT@;}
  .block-container {padding:0.6rem 1.2rem 2rem; max-width:1520px;}
  [data-testid="stCaptionContainer"], .stCaption, .stMarkdown p {color:@MUTED@;}
  /* native widgets follow the theme */
  [data-baseweb="select"] > div {background:@INPUTBG@ !important; border-color:@BORDER@ !important;}
  [data-baseweb="select"] div, [data-baseweb="select"] span {color:@TEXT@ !important;}
  [data-baseweb="popover"] li, ul[role="listbox"] {background:@PANEL@ !important; color:@TEXT@ !important;}
  .stTextInput input, .stChatInput textarea {background:@INPUTBG@ !important; color:@INPUTTEXT@ !important;}
  /* top bar */
  .topbar {display:flex; align-items:center; gap:14px; background:@TOPBAR@;
    border:1px solid @BORDER@; border-radius:12px; padding:11px 18px; margin-bottom:12px;}
  .topbar .logo {font-weight:800; font-size:1.06rem; color:@TEXT@; letter-spacing:.01em;}
  .topbar .logo b {color:@LOGO@;}
  .topbar .sub {color:@MUTED@; font-size:.9rem; border-left:1px solid @BORDER2@; padding-left:12px;}
  .topbar .who {margin-left:auto; color:@TEXT2@; font-size:.82rem; display:flex;
    align-items:center; gap:9px;}
  .topbar .who .av {width:30px; height:30px; border-radius:50%;
    background:linear-gradient(135deg,#22d3ee,#3b82f6); display:flex; align-items:center;
    justify-content:center; font-size:15px;}
  /* panel */
  .panel {background:@PANEL@; border:1px solid @BORDER@; border-radius:12px;
    padding:12px 14px; margin-bottom:12px;}
  /* map box (Streamlit container) + rankings panel (own scroll for sticky header) */
  .st-key-mapbox [data-testid="stVerticalBlockBorderWrapper"] {
    background:@PANEL@ !important; border:1px solid @BORDER@ !important; border-radius:12px !important;}
  .rank-panel {height:480px; overflow:hidden; display:flex; flex-direction:column; margin-bottom:0;}
  .rank-scroll {overflow-y:auto; flex:1; margin-top:2px;}
  .ph {font-size:.72rem; text-transform:uppercase; letter-spacing:.07em; color:@MUTED@;
    font-weight:800; margin-bottom:9px; display:flex; align-items:center; gap:7px;}
  .ph .dotc {width:7px; height:7px; border-radius:50%; background:#22d3ee;}
  /* chat bubbles */
  .chatlog {display:flex; flex-direction:column; gap:9px;}
  .msgrow {display:flex; align-items:flex-end; gap:7px;}
  .msgrow.user {justify-content:flex-end;} .msgrow.bot {justify-content:flex-start;}
  .av {width:24px; height:24px; border-radius:50%; display:flex; align-items:center;
    justify-content:center; font-size:13px; flex:0 0 24px; background:@CHIP@;}
  .av.bot {background:linear-gradient(135deg,#22d3ee,#3b82f6);
    box-shadow:0 2px 8px rgba(34,211,238,.35);}
  .bub {max-width:84%; padding:8px 11px; font-size:.82rem; line-height:1.42;}
  .bub.u {background:linear-gradient(120deg,#0ea5c4,#2563eb); color:#fff; border-radius:13px 13px 3px 13px;}
  .bub.b {background:linear-gradient(135deg,#123a4d,#0f2740); color:#fff;
    border:1px solid #1c4a63; border-left:3px solid #22d3ee; border-radius:13px 13px 13px 3px;}
  .bub.b p, .bub.b li, .bub.b span {color:#fff !important;}
  .bub.b table {border-collapse:collapse; margin:.35rem 0; font-size:.76rem; width:100%;}
  .bub.b th,.bub.b td {border:1px solid @BORDER2@; padding:2px 6px;} .bub.b th{color:@ACCENTTEXT@;}
  .bub.b code {background:@CODEBG@; padding:1px 4px; border-radius:4px;}
  .dots {display:inline-flex; gap:4px; padding:3px 2px;}
  .dots span {width:7px; height:7px; border-radius:50%; background:@MUTED@;
    display:inline-block; animation:dotpulse 1.3s infinite ease-in-out;}
  .dots span:nth-child(2){animation-delay:.18s;} .dots span:nth-child(3){animation-delay:.36s;}
  @keyframes dotpulse {0%,80%,100%{opacity:.3; transform:translateY(0);}
    40%{opacity:1; transform:translateY(-3px);}}
  details.trace summary {cursor:pointer; color:#22d3ee; font-weight:700; font-size:.74rem;}
  details.trace pre {background:#08111e; color:#9fb4cf; padding:7px; border-radius:6px;
    font-size:.68rem; overflow:auto; border:1px solid #1c3350;}
  details.trace .tr {color:@MUTED@; font-size:.72rem; margin-top:5px;}
  /* rankings table */
  table.rank {width:100%; border-collapse:separate; border-spacing:0; font-size:.8rem;}
  table.rank thead th {color:@MUTED@; font-weight:700; text-align:left; padding:8px;
    font-size:.66rem; text-transform:uppercase; letter-spacing:.04em;
    border-bottom:1px solid @BORDER@; position:sticky; top:0; z-index:5;
    background:@PANEL@; box-shadow:inset 0 -1px 0 @BORDER@;}
  table.rank td {padding:7px 8px; border-bottom:1px solid @BORDER@; color:@TEXT2@;}
  table.rank tr:hover td {background:@HOVER@;}
  table.rank thead th[title] {cursor:help; text-decoration:underline dotted @MUTED@;
    text-underline-offset:3px; text-decoration-thickness:1px;}
  table.rank tr.dim td {opacity:.5;}
  table.rank tr.divider td {background:@PANEL2@; color:@MUTED@; font-size:.68rem;
    font-weight:700; text-transform:uppercase; letter-spacing:.04em; padding:7px 8px;
    position:sticky; top:34px; z-index:4;}
  .rk {font-weight:800; color:@ACCENTTEXT@;}
  .code {font-weight:800; color:@TEXT@;}
  .sub2 {color:@MUTED@; font-size:.72rem;}
  .badge {display:inline-block; border-radius:6px; padding:2px 8px; font-weight:800;
    font-size:.76rem; color:#04121f;}
  .pill {display:inline-block; border-radius:999px; padding:1px 9px; font-weight:700; font-size:.72rem;}
  .lock {color:#f59e0b; font-weight:800;}
  /* signal tiles */
  .tiles {display:grid; grid-template-columns:1fr 1fr; gap:9px;}
  .tile {background:@TILE@; border:1px solid @BORDER2@; border-left:3px solid var(--ac,#22d3ee);
    border-radius:10px; padding:9px 11px;}
  .tile .l {font-size:.62rem; text-transform:uppercase; letter-spacing:.05em; color:@MUTED@; font-weight:700;}
  .tile .v {font-size:1.28rem; font-weight:800; color:@TEXT@; font-variant-numeric:tabular-nums;}
  .cap {color:@MUTED@; font-size:.7rem; margin-top:4px;}
  /* filter row: pin dropdown + label to an identical 44px box so text lines up */
  div[data-testid="stSelectbox"], div[data-testid="stSelectbox"] > div {margin:0 !important;}
  div[data-baseweb="select"] > div:first-child {min-height:44px !important;}
  .flab {color:@TEXT2@; font-size:.98rem; font-weight:600; white-space:nowrap;
    text-align:right; padding-right:2px; margin:0; height:44px; line-height:44px;}
  /* color legend under the map */
  .maplegend {display:flex; align-items:center; gap:8px; margin-top:7px;
    font-size:.72rem; color:@MUTED@;}
  .maplegend .grad {flex:1; height:9px; border-radius:5px; min-width:70px;
    border:1px solid @BORDER@;}
  .maplegend .lm {margin-left:6px; color:@TEXT2@; font-weight:600; white-space:nowrap;}
  .disc {color:@MUTED@; font-size:.75rem; text-align:center; padding:9px 10px;
    border-top:1px solid @BORDER@; margin-top:10px;}
  .disc b {color:@TEXT2@;}
  /* buttons */
  .stButton>button {background:@BTNBG@; border:1px solid @BORDER2@; color:@TEXT2@;}
  .stButton>button:hover {border-color:#22d3ee; color:@ACCENTTEXT@;}
  /* ---- floating chat (pinned via key-classes, no component) ---- */
  .st-key-chatfab {position:fixed !important; right:1.8rem !important; bottom:1.8rem !important;
    width:auto !important; z-index:9999 !important;}
  .st-key-chatpanel {position:fixed !important; right:1.8rem !important; bottom:1.8rem !important;
    width:430px !important; max-height:86vh; background:@PANEL@; border:1px solid @BORDER2@;
    border-radius:15px; padding:8px 13px 8px; box-shadow:0 24px 60px rgba(2,8,20,.5);
    z-index:9999 !important;}
  .st-key-chatpanel div[data-testid="stVerticalBlockBorderWrapper"] {border-color:@BORDER@ !important;
    background:transparent !important;}
  .st-key-open_chat button {
    background:linear-gradient(135deg,#22d3ee 0%,#3b82f6 100%) !important;
    color:#04121f !important; font-weight:800 !important; font-size:.98rem !important;
    border:none !important; border-radius:999px !important; padding:14px 24px !important;
    box-shadow:0 12px 34px rgba(34,211,238,.5), 0 0 0 4px rgba(34,211,238,.12) !important;
    animation:avpulse 2.6s infinite;}
  .st-key-open_chat button:hover {transform:translateY(-2px); filter:brightness(1.06);}
  @keyframes avpulse {0%,100%{box-shadow:0 12px 34px rgba(34,211,238,.5),0 0 0 0 rgba(34,211,238,.4)}
    50%{box-shadow:0 12px 40px rgba(34,211,238,.65),0 0 0 12px rgba(34,211,238,0)}}
  .st-key-new_chat, .st-key-close_chat {display:flex; justify-content:center;}
  .st-key-close_chat button, .st-key-new_chat button {background:transparent !important;
    border:none !important; color:@MUTED@ !important; padding:2px 3px !important;
    min-width:0 !important; font-size:1.05rem !important;}
  .st-key-close_chat button:hover {color:#ef4444 !important;}
  .st-key-new_chat button:hover {color:#22d3ee !important;}
  /* send button (cyan) */
  .st-key-send_btn button {
    background:linear-gradient(135deg,#22d3ee,#3b82f6) !important; color:#04121f !important;
    border:none !important; font-weight:800 !important; border-radius:9px !important;}
  .st-key-send_btn button:hover {filter:brightness(1.08);}
  /* native voice recorder: blend with the dark panel (shows live secs + waveform) */
  .st-key-voice_in [data-testid="stAudioInput"] {background:@INPUTBG@ !important;
    border:1px solid @BORDER@ !important; border-radius:10px !important; margin-top:6px;}
  .st-key-voice_in [data-testid="stAudioInput"] * {color:@TEXT2@ !important;}
  /* ⛶ full-screen icon docked in the rankings panel's top-right corner */
  .st-key-rankwrap {position:relative;}
  .st-key-rank_full {position:absolute; top:8px; right:10px; width:auto !important; z-index:20;}
  .st-key-rank_full button {background:transparent !important; border:none !important;
    color:@MUTED@ !important; font-size:1.15rem !important; padding:2px 6px !important;
    min-height:0 !important; line-height:1 !important;}
  .st-key-rank_full button:hover {color:#22d3ee !important;}
</style>
"""


def inject_css():
    css = _CSS
    for k, v in _pal().items():
        if isinstance(v, str):
            css = css.replace(f"@{k}@", v)
    st.markdown(css, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
def render_topbar(meta):
    st.markdown("""
    <div class="topbar">
      <span class="logo">🛫 AVIA <b>INTELLIGENCE</b></span>
      <span class="sub">Airport Investment Intelligence Agent · US airport-expansion opportunities</span>
    </div>
    """, unsafe_allow_html=True)


# ---------- chat panel (left) ----------
def get_agent():
    if "agent" not in st.session_state:
        from src.agent import make_agent
        st.session_state.agent = make_agent()
    return st.session_state.agent


EXAMPLES = [
    "Which airports in New England are strong candidates for terminal expansion?",
    "Compare LA and Santa Ana congestion.",
    "% long-haul flights out of Anchorage?",
    "Unmet flight demand at SFO, and why?",
]


def _trace_html(trace):
    if not trace:
        return ""
    rows = "".join(f'<div class="tr"><code>{_html.escape(t["tool"])}</code> · '
                   f'{_html.escape(str(t["input"]))}</div><pre>{_html.escape(t["result_preview"])}</pre>'
                   for t in trace)
    return f'<details class="trace"><summary>tool calls</summary>{rows}</details>'


# crisp inline-SVG robot avatar (no external asset; scales cleanly, matches theme)
BOT_AV = (
    '<div class="av bot">'
    '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="#04121f" '
    'stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">'
    '<rect x="4.5" y="8.2" width="15" height="10.3" rx="3"/>'
    '<path d="M12 8.2V5.2"/><circle cx="12" cy="4" r="1.3" fill="#04121f" stroke="none"/>'
    '<circle cx="9.4" cy="13" r="1.25" fill="#04121f" stroke="none"/>'
    '<circle cx="14.6" cy="13" r="1.25" fill="#04121f" stroke="none"/>'
    '<path d="M10 16.2h4"/><path d="M4.5 12.2H3M19.5 12.2H21"/></svg></div>'
)


def _bubble(turn):
    if turn["role"] == "user":
        return (f'<div class="msgrow user"><div class="bub u">'
                f'{_html.escape(turn["content"]).replace(chr(10), "<br>")}</div>'
                f'<div class="av">🧑‍💼</div></div>')
    body = _md.markdown(turn["content"] or "", extensions=["tables", "fenced_code", "nl2br"])
    return (f'<div class="msgrow bot">{BOT_AV}'
            f'<div class="bub b">{body}</div></div>')


def _bot_stream(text, cursor=True):  # a bot bubble with growing text (no chatlog wrapper)
    safe = _html.escape(text).replace("\n", "<br>")
    cur = '<span style="opacity:.55">▌</span>' if cursor else ""
    return (f'<div class="msgrow bot">{BOT_AV}'
            f'<div class="bub b">{safe}{cur}</div></div>')


def _typing_dots():  # WhatsApp-style animated typing indicator
    return (f'<div class="msgrow bot">{BOT_AV}'
            '<div class="bub b"><span class="dots"><span></span><span></span>'
            '<span></span></span></div></div>')


def _scroll_to_bottom(nonce):
    """Pin the chat box to its newest message. Finds the actual scrollable element
    inside .st-key-chatbox and retries a few times to catch async re-layout."""
    st.components.v1.html(
        "<script>/*" + str(nonce) + "*/var d=window.parent.document;"
        "function s(){var b=d.querySelector('.st-key-chatbox');if(!b)return;"
        "[b].concat([].slice.call(b.querySelectorAll('*'))).forEach(function(e){"
        "if(e.scrollHeight>e.clientHeight+4){e.scrollTop=e.scrollHeight;}});}"
        "s();setTimeout(s,60);setTimeout(s,180);setTimeout(s,400);setTimeout(s,800);"
        "</script>", height=0)


def render_floating_chat():
    ss = st.session_state
    ss.setdefault("chat_open", False)
    ss.setdefault("history", [])
    has_key = llm.have_credentials()

    # ---- collapsed: a bold launcher pinned bottom-right (CSS via key-class) ----
    if not ss.chat_open:
        with st.container(key="chatfab"):
            n = sum(1 for t in ss.history if t["role"] == "user")
            if st.button("💬  Ask Agent" + (f"  ·  {n}" if n else ""), key="open_chat"):
                ss.chat_open = True
                st.rerun()
        return

    # ---- expanded: a docked chat panel pinned bottom-right ----
    prompt = ss.pop("pending", None)  # from an example chip or the input form

    with st.container(key="chatpanel"):
        h1, hb = st.columns([7, 2])
        h1.markdown('<div class="ph" style="margin:6px 0 2px"><span class="dotc"></span>'
                    'AVIA AI</div>', unsafe_allow_html=True)
        b1, b2 = hb.columns(2, gap="small")
        if b1.button("🔄", key="new_chat", help="New conversation"):
            ss.history = []
            if "agent" in ss:
                ss.agent.reset()
            st.rerun()
        if b2.button("✕", key="close_chat", help="Close"):
            ss.chat_open = False
            st.rerun()

        box = st.container(height=360, key="chatbox")
        # ONE placeholder holds the whole message area — writing to it REPLACES the
        # example chips instantly, so they never linger behind a streaming reply.
        slot = box.empty()

        # --- input row created FIRST so the textbox renders EMPTY immediately (the
        # send-clears-input value is set in the callback), even while the reply is
        # still streaming into `slot` above it. Enter and ➤ both submit. ---
        def _submit_text():
            v = (ss.get("msgbox") or "").strip()
            if v:
                ss.pending = v
                ss.msgbox = ""  # clear the box on send

        cols = st.columns([6, 1])
        cols[0].text_input("m", key="msgbox", label_visibility="collapsed",
                           placeholder="Ask the analyst…", disabled=not has_key,
                           on_change=_submit_text)
        cols[1].button("➤", key="send_btn", on_click=_submit_text,
                       disabled=not has_key, use_container_width=True)

        # voice (bonus): native recorder — live waveform + running-seconds timer.
        # Tap the mic to start, tap stop → Gemini transcribes and auto-sends.
        if has_key and voice.available():
            audio = st.audio_input("voice", key="voice_in", label_visibility="collapsed")
            if audio is not None:
                data = audio.getvalue()
                sig = (len(data), hash(data))
                if data and sig != ss.get("voice_sig"):
                    ss.voice_sig = sig
                    with st.spinner("Transcribing…"):
                        heard = voice.transcribe(data)
                    if heard:
                        ss.pending = heard
                        st.rerun()
                    else:
                        st.caption("🔇 Didn't catch any speech — tap the mic and try again.")

        # --- now fill the message area (into `slot`, which sits ABOVE the input) ---
        if not has_key:
            with slot.container():
                st.info("Set GEMINI_API_KEY / ANTHROPIC_API_KEY to chat.")
        elif prompt:
            ss.history.append({"role": "user", "content": prompt})
            base = "".join(_bubble(t) for t in ss.history)
            slot.markdown(f'<div class="chatlog">{base}{_typing_dots()}</div>',
                          unsafe_allow_html=True)
            _scroll_to_bottom(f"send-{len(ss.history)}")  # show the sent msg + typing
            result = get_agent().chat(prompt)
            reply = (f"*(error: {result['error']})*" if result.get("error") else result["reply"])
            trace = result.get("trace")
            acc = ""
            step = max(1, len(reply) // 300)
            for i, ch in enumerate(reply):
                acc += ch
                if i % step == 0:
                    slot.markdown(f'<div class="chatlog">{base}{_bot_stream(acc)}</div>',
                                  unsafe_allow_html=True)
                    time.sleep(0.008)
            ss.history.append({"role": "assistant", "content": reply, "trace": trace})
            slot.markdown('<div class="chatlog">'
                          + "".join(_bubble(t) for t in ss.history) + "</div>",
                          unsafe_allow_html=True)
        elif ss.history:
            slot.markdown('<div class="chatlog">'
                          + "".join(_bubble(t) for t in ss.history) + "</div>",
                          unsafe_allow_html=True)
        else:
            with slot.container():
                st.caption("Ask about US airport expansion:")
                for i, ex in enumerate(EXAMPLES):
                    if st.button(ex, use_container_width=True, key=f"ex{i}"):
                        ss.pending = ex
                        st.rerun()

        # keep the newest message in view (auto-scroll the box to the bottom)
        _scroll_to_bottom(f"done-{len(ss.history)}")


# ---------- center: filters + map + rankings ----------
SCALE_FLOOR = 500_000  # investment-scale floor: exclude sub-500k-passenger airports


def _region_rows(df, region):
    """All airports in a region (no scale floor)."""
    if region == "United States":
        return df.copy()
    return df[df["state"].isin(airports_in_region(region)["states"])].copy()


def _region_subset(df, region):
    """Investment-scale candidates in a region (>= 500k annual passengers)."""
    rows = _region_rows(df, region)
    return rows[rows["enplanements_2024"].fillna(0) >= SCALE_FLOOR]


def render_center(df):
    lc1, sc1, lc2, sc2, _sp = st.columns([1.1, 2.4, 0.75, 2.4, 2.35],
                                         gap="small", vertical_alignment="center")
    lc1.markdown('<div class="flab">Filter by region</div>', unsafe_allow_html=True)
    region = sc1.selectbox("region", ["United States"]
                           + [r.title() for r in list_known_regions()],
                           label_visibility="collapsed")
    lc2.markdown('<div class="flab">↕ Order by</div>', unsafe_allow_html=True)
    metric = sc2.selectbox("order", ["Expansion Opportunity Score", "Capacity choke",
                                     "Demand growth", "Scale (size)"],
                           label_visibility="collapsed")
    mcol = {"Expansion Opportunity Score": "eos", "Capacity choke": "choke_pct",
            "Demand growth": "pax_growth_pct", "Scale (size)": "enplanements_2024"}[metric]
    mlabel = {"eos": "EOS", "choke_pct": "Choke %ile", "pax_growth_pct": "Growth %",
              "enplanements_2024": "Passengers/yr"}[mcol]
    region_rows = _region_rows(df, region)
    at_scale = region_rows["enplanements_2024"].fillna(0) >= SCALE_FLOOR
    cand = region_rows[at_scale].sort_values(mcol, ascending=False, na_position="last")
    below = region_rows[~at_scale].sort_values(mcol, ascending=False, na_position="last")
    sub = cand.head(10)
    n_region = len(region_rows)  # pre-floor total, for the "X of Y" note

    # rankings (left) and map/analysis (right) side by side, in equal-height boxes
    H = 480
    tablecol, mapcol = st.columns([1.15, 1], gap="medium")
    with tablecol:
        _render_rankings(sub, cand, below, mlabel, n_region)
    with mapcol:
        with st.container(height=H, border=True, key="mapbox"):
            st.markdown(f'<div class="ph"><span class="dotc"></span>'
                        f'Analysis · Expansion Opportunity · by {mlabel}</div>',
                        unsafe_allow_html=True)
            _render_map(sub, metric_col=mcol, metric_label=mlabel, height=H - 120)
            st.markdown(_map_legend(mlabel), unsafe_allow_html=True)
    return region, sub


def _map_legend(mlabel):
    """Gradient key for the map: dot color (and size) rises low → high."""
    stops = ", ".join(RAMP)
    return (f'<div class="maplegend"><span>Low</span>'
            f'<span class="grad" style="background:linear-gradient(90deg,{stops})"></span>'
            f'<span>High</span><span class="lm">{mlabel}</span></div>')


def _render_map(sub, metric_col="eos", metric_label="EOS", height=270):
    m = sub[["lat", "lon", metric_col, "iata", "name"]].dropna(
        subset=["lat", "lon", metric_col]).copy()
    if m.empty:
        st.caption("No mappable airports.")
        return
    vals = m[metric_col]
    vmin, vmax = vals.min(), vals.max()
    m["rgb"] = vals.map(lambda v: list(_rgb(eos_color(v, vmin, vmax))) + [220])
    m["radius"] = 16000 + (vals - vmin) / max(vmax - vmin, 1) * 42000
    m["mval"] = vals.map(lambda v: f"{v/1e6:.1f}M" if abs(v) >= 10000 else f"{v:.1f}")
    lat0, lon0 = m["lat"].mean(), m["lon"].mean()
    span = max(m["lat"].max() - m["lat"].min(), m["lon"].max() - m["lon"].min(), 2)
    zoom = 6.6 if span < 3 else 5.4 if span < 8 else 3.6
    P = _pal()
    layers = [
        pdk.Layer("ScatterplotLayer", m, get_position="[lon, lat]", get_fill_color="rgb",
                  get_radius="radius", pickable=True, opacity=0.85,
                  stroked=True, get_line_color=P["MAPSTROKE"], line_width_min_pixels=1),
        pdk.Layer("TextLayer", m, get_position="[lon, lat]", get_text="iata",
                  get_size=13, get_color=P["MAPTEXT"], get_alignment_baseline="'bottom'"),
    ]
    deck = pdk.Deck(
        layers=layers, map_style=P["MAPSTYLE"],
        initial_view_state=pdk.ViewState(latitude=float(lat0), longitude=float(lon0),
                                         zoom=zoom, pitch=0),
        tooltip={"text": "{iata} — {name}\n" + metric_label + ": {mval}"})
    st.pydeck_chart(deck, use_container_width=True, height=height)


def _unmet_band(iata):
    ud = scoring.unmet_demand(iata)
    pct = ud.get("unmet_demand_pct") if ud and ud.get("available") else None
    if pct is None:
        return "—", "#334155"
    if pct >= 22:
        return "High", BAD
    if pct >= 12:
        return "Medium", WARN
    return "Low", GOOD


def _fmt_scale(en):
    if pd.isna(en):
        return "—"
    if en >= 1e6:
        return f"{en/1e6:.1f}M"
    if en >= 1e3:
        return f"{en/1e3:.0f}K"
    return f"{int(en)}"


def _rankings_rows_html(sub, start=1, dim=False, vrange=None):
    vmin, vmax = vrange if vrange else (sub["eos"].min(), sub["eos"].max())
    tr_cls = ' class="dim"' if dim else ""
    rows = ""
    for i, (_, r) in enumerate(sub.iterrows(), start):
        # NOTE: only the EOS badge (score) and the ● dot (confidence) are colored.
        # Congestion/Choke/Scale/Unmet/Growth are neutral so their color can't be
        # mistaken for the green/amber/orange confidence signal.
        sc_col = eos_color(r["eos"], vmin, vmax)
        cong = (r["congestion_pct"] / 10) if pd.notna(r["congestion_pct"]) else None
        choke = f'{r["choke_pct"]:.0f}%' if pd.notna(r["choke_pct"]) else "—"
        scale = _fmt_scale(r.get("enplanements_2024"))
        band, _ = _unmet_band(r["iata"])
        lock = '<span class="lock">🔒</span> ' if r.get("slot_controlled") else ""
        g = r["pax_growth_pct"]
        growth = "—" if pd.isna(g) else f"{g:+.1f}%"
        conf = r.get("data_confidence", "High")
        cdot = {"High": GOOD, "Medium": WARN, "Low": "#f97316"}.get(conf, "#64748b")
        dot = (f'<span title="{conf} data confidence" '
               f'style="color:{cdot};font-size:.55rem;vertical-align:middle;margin-left:5px">●</span>')
        rows += (
            f'<tr{tr_cls}><td class="rk">{i}</td>'
            f'<td><span class="code">{r["iata"]}</span>{dot}<div class="sub2">{r["name"][:26]}</div></td>'
            f'<td><span class="badge" style="background:{sc_col}">{r["eos"]:.0f}/100</span></td>'
            f'<td>{lock}{choke}</td>'
            f'<td class="sub2" style="white-space:nowrap;font-weight:700">{scale}</td>'
            f'<td>{band}</td>'
            f'<td style="font-weight:700">{growth}</td>'
            f'<td style="font-weight:700">{("%.1f"%cong) if cong is not None else "—"}</td></tr>')
    return rows


_RANK_HEAD = (
    '<table class="rank"><thead><tr>'
    '<th title="Rank position, by the Order-by metric you selected.">#</th>'
    '<th title="Airport code and name. The colored dot shows overall data confidence '
    'for the airport: green = High, amber = Medium, orange = Low.">Airport</th>'
    '<th title="Expansion Opportunity Score, 0-100 — the overall investment score. '
    'Higher = a better place to build capacity. Combines choke (45%), demand (30%) and '
    'scale (25%), then dampened if the airport is not congested.">EOS</th>'
    '<th title="How capacity-constrained the airport is, as a national percentile '
    '(higher = more choked). The lock means the FAA legally caps the number of flights.'
    '">Choke (slots)</th>'
    '<th title="Size of the airport: total passengers boarding per year.">Scale</th>'
    '<th title="Estimated flights the airport is turning away because it is full: '
    'High / Medium / Low. A directional proxy, not a forecast.">Unmet demand</th>'
    '<th title="Passenger growth vs the prior year.">Growth</th>'
    '<th title="How bad the delays are, on a 0-10 scale. Built from the share of flights '
    'delayed 15+ minutes, the average departure delay, and cancellations.">Congestion</th>'
    '</tr></thead>')
_RANK_CAP = '<div class="cap">🔒 = FAA slot-controlled (flights capped).</div>'


def _rankings_panel_html(sub, title="Airport Expansion Rankings", panel_style=""):
    return (f'<div class="panel rank-panel" style="{panel_style}">'
            f'<div class="ph"><span class="dotc"></span>{title}</div>'
            f'<div class="rank-scroll">{_RANK_HEAD}'
            f'<tbody>{_rankings_rows_html(sub)}</tbody></table></div>'
            f'{_RANK_CAP}</div>')


def _rankings_full_html(cand, below, mlabel):
    """Full list: ranked investment-scale candidates, then below-floor airports
    in a clearly-labelled, dimmed section (shown for completeness, NOT ranked)."""
    both = pd.concat([cand, below]) if len(below) else cand
    vrange = (both["eos"].min(), both["eos"].max())
    body = _rankings_rows_html(cand, start=1, vrange=vrange)
    if len(below):
        body += ('<tr class="divider"><td colspan="8">▾ Below investment-scale floor '
                 '(under 500k passengers/yr) — shown for completeness, NOT ranked as '
                 'candidates (growth is volatile on a tiny base)</td></tr>')
        body += _rankings_rows_html(below, start=len(cand) + 1, dim=True, vrange=vrange)
    title = f"Ranked by {mlabel} · {len(cand)} candidates + {len(below)} below floor"
    return (f'<div class="panel rank-panel" style="height:68vh">'
            f'<div class="ph"><span class="dotc"></span>{title}</div>'
            f'<div class="rank-scroll">{_RANK_HEAD}<tbody>{body}</tbody></table></div>'
            f'{_RANK_CAP}</div>')


@st.dialog("Airport Expansion Rankings", width="large")
def _rankings_dialog(cand, below, mlabel, n_region):
    st.caption(
        f"All **{n_region}** airports in view. Top = **{len(cand)}** investment-scale "
        f"candidates (≥ 500,000 passengers/yr), ranked by {mlabel}. Below the divider = "
        f"the remaining **{len(below)}** smaller airports, shown for completeness but "
        f"excluded from ranking (their growth swings wildly on a tiny base). Scroll for all."
    )
    st.markdown(_rankings_full_html(cand, below, mlabel), unsafe_allow_html=True)


def _render_rankings(sub, cand, below, mlabel, n_region):
    with st.container(key="rankwrap"):
        # ⛶ icon docked in the panel's top-right corner (like the map's native one)
        if st.button("⛶", key="rank_full", help="Full screen — all airports"):
            _rankings_dialog(cand, below, mlabel, n_region)
        st.markdown(_rankings_panel_html(sub), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
def render_footer(meta):
    """Always-visible assumptions / scope / uncertainty + a decision-support disclaimer."""
    a = meta.get("assumptions", {})
    months = ", ".join(meta.get("ontime_months", [])) or "sampled 2024 months"
    n = meta.get("n_airports", "—")
    n_ont = meta.get("n_with_ontime", "—")
    with st.expander("ℹ️  How this works — in plain English", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                "**How the score works**\n\n"
                "Every airport gets one score from **0 to 100** — the Expansion Opportunity "
                "Score (EOS). Higher = a better place to spend money building terminals or "
                "runways.\n\n"
                "`EOS = (Choke 45% + Demand 30% + Scale 25%) × Choke-gate`\n\n"
                "It mixes three things: how **jammed** the airport is (choke), how fast it's "
                "**growing** (demand), and how **big** it is (scale). Then the **choke-gate** "
                "shrinks the score if the airport *isn't* jammed — so a huge, growing airport "
                "that still has room can't score high, because building there wouldn't pay "
                "off. We only want airports that are full and turning flights away.\n\n"
                "**Jammed (choke)** = delays + cancellations + how packed the runways are + a "
                "bonus when the FAA legally caps flights (JFK, LGA, DCA, EWR)."
            )
            st.markdown(
                "**What's included**\n\n"
                f"- **US airports only** — {n} of them.\n"
                f"- Delay & long-flight data comes from a US-**domestic** dataset (sample "
                f"months {months}; {n_ont} airports covered).\n"
                "- **International flights aren't counted**, so global hubs like SFO and JFK "
                "look a little smaller here than they really are."
            )
        with c2:
            st.markdown(
                "**The rules we chose**\n\n"
                f"- A **long-haul** flight = {a.get('long_haul_miles', 2000):,}+ miles. A "
                f"flight counts as **late** if it leaves more than "
                f"{a.get('delay_threshold_min', 15)} minutes behind.\n"
                "- **Growth** = 2024 passengers vs 2023 (official FAA numbers).\n"
                "- We assume an airport's FAA code and its 3-letter code match "
                "(true for the big ones).\n"
                "- The flight-capped list is hand-checked from FAA's 2024 rules."
            )
            st.markdown(
                "**What we're not sure about**\n\n"
                "- Each airport has a **confidence dot ●** — small airports with shaky "
                "numbers get a lower rating.\n"
                "- Delays can come from **weather or air-traffic control**, not just a full "
                "airport — so we never judge on delays alone; we also use runway use and "
                "flight caps.\n"
                "- **\"Unmet demand\"** is a rough estimate of flights being turned away — "
                "not a real forecast.\n"
                "- There's **no public data** for gate/terminal counts or how full planes "
                "are — we say so instead of guessing (details in `docs/DESIGN.md`)."
            )
    st.markdown(
        '<div class="disc"><b>Decision-support only — not investment advice.</b> '
        'Every figure is code-computed from public FAA · BTS · OurAirports data; '
        'the AI plans, narrates and cites — it never fabricates numbers.</div>',
        unsafe_allow_html=True)


# ---------------------------------------------------------------------------
def main():
    st.session_state["theme"] = "dark"  # dark only (light mode removed)
    inject_css()
    try:
        df = _scored()
    except FileNotFoundError:
        st.error("Snapshot missing. Run: python data/build_dataset.py")
        return
    meta = _meta()

    render_topbar(meta)

    # full-width dashboard: map + rankings side by side
    render_center(df)

    # always-visible methodology / assumptions / scope / uncertainty + disclaimer
    render_footer(meta)

    # the agent, docked as a floating bubble bottom-right
    render_floating_chat()


main()
