# Design & Architecture — Airport Investment Intelligence Agent

A conversational agent that helps an investment analyst find **US airports where
terminal/runway expansion capital will be most productive** — i.e. where demand
is large and growing but the airport is already capacity-constrained, so added
capacity converts directly into more served flights and passengers.

---

## 1. Architecture at a glance

```
                    ┌──────────────────────────────────────────────┐
   user (chat) ───► │  Agent loop (Claude, tool-use)                │
                    │  - plans, calls tools, narrates, follows up   │
                    └───────────────┬──────────────────────────────┘
                                    │ tool calls (typed args)
                    ┌───────────────▼──────────────────────────────┐
                    │  Tools (src/tools.py)  — deterministic         │
                    │  resolve_airport · airport_report · rank ·     │
                    │  compare · long_haul · unmet_demand ·          │
                    │  live_traffic · universe_summary               │
                    └───────┬─────────────────────────┬─────────────┘
                            │                          │
              ┌─────────────▼───────────┐   ┌──────────▼───────────┐
              │ Scoring engine          │   │ Live API             │
              │ (src/scoring.py)        │   │ (src/live_api.py)    │
              │ EOS + percentiles,      │   │ OpenSky real-time    │
              │ unmet demand, long-haul │   │ traffic (best-effort)│
              └─────────────┬───────────┘   └──────────────────────┘
                            │ reads
              ┌─────────────▼───────────────────────────────────────┐
              │ Cached snapshot  data/airport_snapshot.parquet       │
              │ built offline from public data (data/build_dataset)  │
              └──────────────────────────────────────────────────────┘
```

**Hybrid data strategy.** A one-time offline build (`data/build_dataset.py`)
pulls public government/open datasets, joins them per airport, derives the
investment metrics, and caches one parquet the app reads instantly. A live
OpenSky call supplies real-time traffic on demand. This gives deep, reproducible
scoring *and* a genuine live-API integration, while keeping the demo fast and
robust (the snapshot ships in the repo, so nothing breaks if a source is down).

---

## 2. Data sources (all public, no key)

| Source | Gives us | Used for |
|---|---|---|
| **OurAirports** (`airports.csv`, `runways.csv`) | Airport metadata: IATA/ICAO, geo, state/region, runway count | Region filters, geo/map, runway utilization |
| **FAA ACAIS** CY2024 enplanements xlsx | Passengers 2024 + 2023, % change, hub size | Demand **scale** & **growth** pillars |
| **BTS On-Time Performance** (sampled months Feb/Jul/Oct 2024) | Per-flight departure delay, cancellation, route distance | **Congestion** pillar, **long-haul %**, unmet demand |
| **OpenSky Network** (live) | Aircraft currently airborne/on-ground near an airport | Real-time activity gauge |

Coverage in the shipped snapshot: **478 US commercial airports**, 332 with
On-Time (delay/long-haul) data.

---

## 3. Scoring methodology — the Expansion Opportunity Score (EOS)

The core deterministic artifact. Each airport gets a **0–100 EOS** = weighted sum
of four **national-percentile** pillars. Percentile-ranking (not raw values) makes
pillars unit-free, outlier-robust, and interpretable ("92nd percentile for
congestion nationally").

| Pillar | Weight | Signal | Why it belongs in an expansion thesis |
|---|---|---|---|
| **Congestion** | 35% | delay-rate (>15 min), cancellation rate, mean dep delay | Delays/cancellations are the market's signal the airport can't clear current demand. This is the strongest "expansion is needed *now*" signal → highest weight. |
| **Growth** | 30% | YoY enplanement growth (FAA CY24 vs CY23) | Forward demand — a growing airport will outrun its footprint. |
| **Scale** | 20% | log(annual enplanements) | Absolute payoff: 1% more capacity at a 40M-pax hub is worth far more passengers than at a regional field. Log-scaled because volume is heavy-tailed. |
| **Utilization** | 15% | departures per (non-closed) runway | Physical capacity pressure — are the existing runways worked hard? |

Weights live in `src/config.py:EOS_WEIGHTS` (asserted to sum to 1.0). Every score
is fully decomposable — `airport_report` returns each pillar's percentile,
weight, and contribution, so the agent can *explain* a ranking rather than assert
it.

**Derived analytics** built on the same data:
- **Congestion level** — Low/Moderate/Elevated/High/Severe band from the
  congestion percentile (answers "compare X and Y congestion").
- **Long-haul %** — share of departures with stage length ≥ 2000 statute miles
  (answers "% long-haul out of Anchorage"). Definition + scope caveat returned
  with the number.
- **Unmet demand** — a transparent proxy (see §5), not a forecast.

---

## 4. Where and how AI is used

**The LLM plans and narrates; the code decides.** A strict split:

- **AI (Claude, native tool-use loop in `src/agent.py`):** interprets the
  question, resolves ambiguous references (calls `resolve_airport`), picks the
  right tool(s) and arguments, sequences multi-step work, explains the reasoning
  and methodology in plain language, and handles conversational follow-ups
  (message history persists on the agent instance).
- **Deterministic code (`scoring.py`):** every airport number — passenger counts,
  delay rates, growth, long-haul %, scores, rankings. The system prompt forbids
  the model from stating any figure from memory; if a tool lacks it, the agent
  says so.

This satisfies the "deterministic scoring, not only LLM output" requirement and
prevents hallucinated statistics — the failure mode that would make an
investment tool untrustworthy. The tool trace is surfaced in the UI ("How I got
this") so every answer is auditable back to a code-computed result.

Model default `claude-opus-4-8` (strongest reasoning; override via
`ANTHROPIC_MODEL`, e.g. `claude-sonnet-5` for a cheaper/faster demo). The
provider is isolated in `src/llm.py` for swappability.

---

## 5. Key tradeoffs & assumptions (explicitly communicated)

The agent is built to **surface** these, not hide them.

1. **Unmet demand is a proxy, not a forecast.** Model (monthly):
   `served_ontime = departures × (1 − delayed% − cancelled%)`,
   `latent_demand = departures × (1 + max(growth,0))`,
   `unmet = max(0, latent − served)`. Rationale: delays/cancellations reveal
   demand the airport can't clear on schedule; growth adds forward pressure. It
   is directional — some delay is weather/ATC, not capacity — and the caveats
   ride along with every result.

2. **Long-haul % is domestic-only.** BTS On-Time covers US carriers with ≥0.5%
   of domestic scheduled revenue; international segments aren't included, so
   long-haul % **understates** intercontinental activity at gateways (SFO, JFK).
   Flagged whenever long-haul is discussed.

3. **Small-airport growth is volatile.** A regional field going 25k→45k
   passengers posts +80% growth and can top an unfiltered ranking. The engine
   assigns it **Low** `data_confidence`, and the agent applies a ~500k-passenger
   floor by default for "expansion candidate" rankings (and says so). This keeps
   recommendations credible while remaining transparent and overridable.

4. **Seasonality sampling.** Congestion uses 3 months (Feb/Jul/Oct 2024) to
   balance seasonal spread against build time — not the full year. Configurable
   in `build_dataset.py`.

5. **Weights are a defensible prior, not ground truth.** They encode the thesis
   "constrained-but-growing demand". All in one config block for easy
   sensitivity analysis. A firm with a different thesis (e.g. pure growth plays)
   would retune them.

6. **Scope is US commercial airports.** Non-US, cargo-only, and GA fields are out
   of scope. FAA↔IATA code join assumes they match (true for major commercial
   airports).

7. **Data confidence is first-class.** Each airport carries High/Medium/Low
   confidence from data completeness + departure sample size; imputed pillars
   (median fill) are tracked and flagged.

---

## 6. What I'd add with more time

- International segments (BTS T-100) for true intercontinental long-haul.
- Runway/gate capacity from FAA facility data for a real capacity-utilization
  denominator (current utilization uses runway *count* as a proxy).
- Weather-adjusted delays to sharpen the unmet-demand signal.
- A sensitivity view: how rankings shift as EOS weights change.
- Voice I/O (browser Web Speech / `st.audio_input` + transcription) — the bonus.

---

## 7. File map

```
data/build_dataset.py   offline build: download → join → derive → parquet
src/config.py           weights, thresholds, regions, aliases (all tunables)
src/data_layer.py       snapshot load, airport-name resolver, region lookup
src/scoring.py          EOS, percentiles, unmet demand, long-haul, ranking
src/live_api.py         OpenSky real-time traffic (best-effort)
src/tools.py            tool schemas + deterministic dispatch
src/agent.py            Claude tool-use loop + system prompt
src/llm.py              provider seam (model id / client)
app.py                  Streamlit chat UI + dashboard + map
chat_cli.py             terminal chat (no UI needed)
tests/test_scoring.py   deterministic-engine unit tests
```
