# 🛫 Airport Investment Intelligence Agent

A conversational AI agent that helps investment analysts identify **US airports
where terminal/runway expansion will be most profitable** — ranking airports
where demand is large and growing but capacity is already strained, so new
capacity converts into more served flights and passengers.

Built for the Forward Deployed Engineer exercise. **Design write-up:**
[the deck](docs/Airport%20Investment%20Intelligence%20Agent.html) (13 slides — arrow keys, press F) ·
[`docs/design.html`](docs/design.html) (visual one-pager) ·
[`docs/DESIGN.md`](docs/DESIGN.md) (plain text). All open in a browser.

It answers questions like:
- *Which airports in New England are strong candidates for terminal expansion?*
- *Compare LA and Santa Ana airport congestion levels.*
- *What percentage of flights out of Anchorage are long-haul?*
- *What is the unmet flight demand at SFO, and why?*

## What it does

- **Deterministic scoring, not just an LLM.** A **choke-gated Expansion
  Opportunity Score (EOS)** = (Capacity-choke 45% + Demand 30% + Scale 25%) × a
  **choke gate**, so a big, growing, but un-choked airport *can't* rank high —
  the score requires a bottleneck by construction (the investment thesis). Choke
  includes FAA slot-control (JFK/LGA/DCA/EWR). The LLM plans and explains; all
  numbers come from code. See [`docs/DESIGN.md`](docs/DESIGN.md) §3.
  EOS ranks *where capacity is most productive to add* — a **precondition for
  profit**, not an ROI estimate (no capex/revenue modeling). It's the shortlist
  for diligence, not the diligence.
- **Chat interface** — a **floating chat widget** (streaming replies) plus a
  terminal CLI; **voice input** (bonus) lets you *speak* a question. The core
  deliverable is the scoring + grounded agent; the dark dashboard (national
  opportunity map + live rankings) is optional presentation polish on the same
  tools.
- **Hybrid public data** — cached snapshot from FAA + BTS + OurAirports, plus
  live OpenSky (real-time traffic) and BTS T-100 REST-API tools.
- **Honest about itself** — a persistent *Methodology · assumptions · scope ·
  uncertainty* panel + a "decision-support, not investment advice" disclaimer,
  per-airport **data-confidence** dots, and proxies (e.g. "unmet demand")
  labelled as such — in both the UI and the agent's answers.
- **Trustworthy numbers** — every figure is code-computed by `scoring.py` /
  the live APIs; the system prompt forbids stating any statistic from memory, so
  the LLM never hallucinates a figure.

## Quick start

Requires Python 3.11+ and **one** LLM key — Gemini *or* Anthropic. Voice input
needs a Gemini key (Gemini does the speech-to-text).

```bash
pip install -r requirements.txt

# 1) Build the data snapshot (downloads public data → data/airport_snapshot.parquet)
#    Skip if the parquet is already committed; ~2 min otherwise.
python data/build_dataset.py            # or --quick for a single month

# 2) Add your key
cp .env.example .env        # then edit .env
# or: export GEMINI_API_KEY=...         # (or ANTHROPIC_API_KEY=sk-ant-...)

# 3a) Chat in the browser
streamlit run app.py

# 3b) …or in the terminal
python chat_cli.py
python chat_cli.py "Compare LAX and SNA congestion"     # one-shot
```

No key yet? The **dashboard, scoring, map, and rankings still work** — only the
chat + voice need the LLM. Get a Gemini key at
<https://aistudio.google.com/apikey> or an Anthropic key at
<https://console.anthropic.com/>.

## Tests

```bash
pytest tests/ -v      # deterministic-engine unit tests (no key, no network)
```

## How it's built

```
data/build_dataset.py   offline build: download → join → derive → parquet snapshot
src/scoring.py          EOS + percentiles, unmet-demand proxy, long-haul, ranking  ← the graded core
src/data_layer.py       snapshot load, airport-name resolver, region lookup
src/live_api.py         OpenSky real-time traffic (best-effort)
src/tools.py            agent tool schemas + deterministic dispatch
src/agent.py            Claude tool-use loop + system prompt
src/agent_gemini.py     Gemini function-calling loop (same tools/prompt)
src/llm.py              provider seam (model / client)
src/voice.py            Gemini speech-to-text for voice input (bonus)
app.py                  Streamlit dashboard + floating chat + voice
chat_cli.py             terminal chat
```

Data snapshot: **478 US commercial airports**, FAA CY2024 passenger data + BTS
On-Time delay/route data (sampled months) + OurAirports geo/runways.

## Configuration

Everything tunable lives in [`src/config.py`](src/config.py) — EOS weights,
long-haul threshold, region/state maps, airport aliases — and env vars:

| Var | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | auto | `gemini` or `anthropic` (auto-detects from whichever key is set) |
| `GEMINI_API_KEY` | — | Gemini chat + **voice** transcription |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model id |
| `ANTHROPIC_API_KEY` | — | Anthropic chat (alternative to Gemini) |
| `ANTHROPIC_MODEL` | `claude-opus-4-8` | e.g. `claude-sonnet-5` for cheaper/faster |

## Deploy (free, permanent URL)

The app runs as-is on **[Streamlit Community Cloud](https://share.streamlit.io)**
(free, HTTPS — so browser voice input works):

1. Push this repo to GitHub.
2. share.streamlit.io → **New app** → pick the repo and `app.py`.
3. In the app's **Settings → Secrets**, add your key (TOML):
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
   `src/llm.py` reads env vars **or** `st.secrets`, so no code change is needed.
4. Deploy → you get a permanent `https://<app>.streamlit.app` URL.

Notes: the committed `data/airport_snapshot.parquet` means there's no build step;
the dashboard/map/rankings work with **no key** (only chat + voice need one); and
the key is shared with anyone who opens the URL, so set a spend cap for public use.
Hugging Face Spaces (Streamlit SDK) works the same way.

## Scope & caveats (short version)

US commercial airports only. Congestion & long-haul come from the **domestic**
BTS On-Time dataset (sampled months) — international long-haul is not included.
"Unmet demand" is a transparent directional proxy, not a forecast. Small
airports post volatile growth %, so expansion rankings apply a passenger-volume
floor by default. Full discussion in [`docs/DESIGN.md`](docs/DESIGN.md) §5.
