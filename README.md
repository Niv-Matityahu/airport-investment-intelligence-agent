# 🛫 Airport Investment Intelligence Agent

A conversational AI agent that helps investment analysts identify **US airports
where terminal/runway expansion will be most profitable** — ranking airports
where demand is large and growing but capacity is already strained, so new
capacity converts into more served flights and passengers.

Built for the Forward Deployed Engineer exercise. **Design write-up:
[`docs/DESIGN.md`](docs/DESIGN.md).**

It answers questions like:
- *Which airports in New England are strong candidates for terminal expansion?*
- *Compare LA and Santa Ana airport congestion levels.*
- *What percentage of flights out of Anchorage are long-haul?*
- *What is the unmet flight demand at SFO, and why?*

## What it does

- **Deterministic scoring, not just an LLM.** An **Expansion Opportunity Score
  (EOS)** ranks airports on four transparent, weighted, national-percentile
  pillars — congestion, growth, scale, utilization. The LLM plans and explains;
  all numbers come from code. See [`docs/DESIGN.md`](docs/DESIGN.md) §3.
- **Chat interface** (Streamlit) with a live sidebar dashboard + national
  opportunity map, or a terminal CLI.
- **Hybrid public data** — cached snapshot from FAA + BTS + OurAirports, plus a
  live OpenSky real-time traffic tool.
- **Honest about itself** — surfaces assumptions, scope (US domestic), data
  confidence, and labels proxies (e.g. "unmet demand") as such.
- **Auditable** — every answer shows the tool calls that produced it.

## Quick start

Requires Python 3.11+ and an Anthropic API key.

```bash
pip install -r requirements.txt

# 1) Build the data snapshot (downloads public data → data/airport_snapshot.parquet)
#    Skip if the parquet is already committed; ~2 min otherwise.
python data/build_dataset.py            # or --quick for a single month

# 2) Add your key
cp .env.example .env        # then edit .env
# or: export ANTHROPIC_API_KEY=sk-ant-...

# 3a) Chat in the browser
streamlit run app.py

# 3b) …or in the terminal
python chat_cli.py
python chat_cli.py "Compare LAX and SNA congestion"     # one-shot
```

No key yet? The **sidebar dashboard, scoring, and map still work** — only the
chat needs the LLM. Get a key at <https://console.anthropic.com/>.

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
src/llm.py              provider seam (model / client)
app.py                  Streamlit chat UI + dashboard
chat_cli.py             terminal chat
```

Data snapshot: **478 US commercial airports**, FAA CY2024 passenger data + BTS
On-Time delay/route data (sampled months) + OurAirports geo/runways.

## Configuration

Everything tunable lives in [`src/config.py`](src/config.py) — EOS weights,
long-haul threshold, region/state maps, airport aliases — and env vars:

| Var | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | required for chat |
| `ANTHROPIC_MODEL` | `claude-opus-4-8` | e.g. `claude-sonnet-5` for cheaper/faster |

## Scope & caveats (short version)

US commercial airports only. Congestion & long-haul come from the **domestic**
BTS On-Time dataset (sampled months) — international long-haul is not included.
"Unmet demand" is a transparent directional proxy, not a forecast. Small
airports post volatile growth %, so expansion rankings apply a passenger-volume
floor by default. Full discussion in [`docs/DESIGN.md`](docs/DESIGN.md) §5.
