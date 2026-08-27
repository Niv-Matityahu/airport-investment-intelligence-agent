# Design & Architecture — Airport Investment Intelligence Agent

A conversational agent that helps an investment analyst find **US airports where
terminal/runway expansion capital will be most productive**.

## 0. The investment thesis (what "profitable" means here)

The fund makes money by **building physical capacity** (terminals, gates,
runways) and earning airline fees + passenger/retail revenue on the new traffic.
That pays off only at a **choked bottleneck** — an airport that is maxed out and
turning flights away. So a good target needs **three conditions at once**:

1. **Capacity choke** — near-100% utilization, chronic delays, or an FAA cap on
   flights. *This is the differentiator: high demand alone isn't enough — if the
   airport has room, it just absorbs the demand and no build is needed.*
2. **Demand momentum** — traffic is growing.
3. **Scale** — a base large enough to justify hundreds of millions in capex.

Everything below is built to find the **intersection** of the three.

### What the score is — and is NOT (the one misread to avoid)
EOS ranks **where new capacity would be most productive** — a *precondition* for
profit, not profit itself. It deliberately does **not** model construction cost
(capex), revenue-per-passenger, airline fee structures, or ROI — those are
deal-specific and can't be screened from public data. Read a high EOS as *"this
airport most needs capacity and would convert a build into served flights and
passengers"* — the **shortlist for financial diligence**, not the diligence.
Treating "Expansion Opportunity Score" as "expected profit" is the single
dangerous misread, so the agent is explicitly instructed to draw this line when
asked (see `src/agent.py`, and it is *not investment advice*).

### Scope for one day (what got the effort, and why)
The graded core is the **deterministic KPI (`scoring.py`) + the grounded agent**;
that is where the reasoning lives and where time was spent. The dashboard, map,
full-screen ranking, and voice are **optional presentation polish** on top of the
same tools — nice to have, but not the substance. If you read one file, read
`src/scoring.py`.

---

## 1. Architecture

```
   user (chat) ─► Agent loop (Claude / Gemini, tool-use) ─► narrates, follows up
                        │ typed tool calls
                        ▼
              deterministic scoring  (src/scoring.py)
              EOS + choke gate, unmet demand, ranking
                        │
                        ▼
        cached snapshot  data/airport_snapshot.parquet
        (built offline from public data — data/build_dataset.py)
```

**Snapshot strategy.** All analytics read a cached snapshot built once from public
datasets — fast, reproducible, and demo-safe (the same answer every run). The
"use public APIs to gather data" requirement is met by the build pipeline
(`data/build_dataset.py`), which pulls from public FAA / BTS / OurAirports sources.
*(Two live public REST APIs — OpenSky + BTS T-100 — are implemented in
`src/live_api.py` but left out of the MVP tool set to keep it simple; re-enabling
them is a one-line change.)*

**Provider-agnostic.** The LLM sits behind `src/llm.py`; `AirportAgent` (Claude)
and `GeminiAirportAgent` share the same tools, prompt, and scoring.

---

## 2. Data sources — what we use, and the "do we need more APIs?" answer

| Source | Type | Gives us | Role |
|---|---|---|---|
| **OurAirports** | HTTP CSV | geo, region/state, runways | metadata, map, utilization denominator |
| **FAA ACAIS CY2024** | public file | passengers 2024/2023, hub size | **scale** + **demand** pillars |
| **BTS On-Time** (sampled 2024) | public file | per-flight delays, cancels, route distance | **choke** signal + long-haul % |
| **FAA capacity designations** | curated (14 CFR 93; FAA Core 30) | slot-controlled + Core-30 flags | **choke** — the authoritative bottleneck marker |

*(Live extensions, coded but not in the MVP tool set: **OpenSky** real-time aircraft
and **BTS T-100** official domestic segments — see `src/live_api.py`.)*

**Sources we deliberately did *not* add (and why):**

- **Load factor / seats (raw BTS T-100 segment).** The best *additional* "planes
  are full" signal, but the seats field only exists in a bulk file whose current
  URL we couldn't resolve in this environment (the ArcGIS mirror drops seats).
  Documented as the first extension. We already capture choke via delays +
  utilization + slot-control, so this is additive, not blocking.
- **Gate / terminal counts.** The thesis is literally about gates, but there is
  **no clean public API** for per-airport gate capacity. We proxy physical
  capacity with runway count + slot-control. Honest limitation.
- **FAA TAF (forward forecasts) / ASPM (hourly capacity vs demand).** Richer
  forward-demand and true capacity ratios, but portal/form-gated, not clean
  APIs. Noted as production extensions.

**Verdict:** for the four exam questions we have everything; for the deeper
thesis, the FAA capacity designations were the highest-value addition available,
and we added them.

---

## 3. Scoring methodology — the choke-gated Expansion Opportunity Score

`EOS = (Choke 0.45 + Demand 0.30 + Scale 0.25) × ChokeGate`, each pillar a
0–100 **national percentile** (unit-free, outlier-robust, interpretable).

| Pillar | Weight | Built from |
|---|---|---|
| **Capacity choke** | 45% | delay rate, cancellations, mean delay, departures-per-runway, **+ a bonus for FAA slot-controlled airports** |
| **Demand** | 30% | YoY passenger growth (FAA CY24 vs CY23) |
| **Scale** | 20%→25% | log(annual enplanements) |

**The choke gate is the key design choice.**
`gate = 0.60 + 0.40 × (choke_pct / 100)`. It multiplies the score *down* when
choke is low, so a big, growing, but **un-choked** airport can never rank high.
This makes the score match "profitable bottleneck" **by construction**, not by
luck — a weighted sum alone would let scale+growth inflate an airport that
doesn't actually need a build.

**Validation — face-validity + sensitivity (a ranking like this can't be
"proven" against ground truth, so we check it behaves):**
- **Face validity:** top candidates — CLT, DFW, PHX, SAN, DEN, PHL, **LGA**, MIA,
  **DCA**, ORD — are all high-choke big hubs (choke 89th–100th percentile). The
  FAA **slot-controlled** airports (LGA, DCA, JFK, EWR) rise on the slot bonus —
  exactly the "flights are capped, no room" bottlenecks.
- **The gate does its job:** it **demotes** big-but-unchoked airports — OGG/Maui
  (scale 89th, choke 28th) falls from base 37 → EOS 27 via a 0.71 gate. A
  pure weighted sum would rank it far higher.
- **Guarded by tests:** `tests/test_scoring.py` asserts the invariants that make
  the thesis hold — a high EOS *requires* real choke, the gate demotes the
  un-choked, slot-control raises choke — so a future weight tweak can't silently
  break the logic.
- **Sensitivity is tunable, not hidden:** all weights + the 0.60 gate floor live
  in `src/config.py`; the ordering of the top hubs is stable across a reasonable
  floor band (≈0.5–0.7) because they lead on *both* choke and scale.

Every score is fully decomposable — `airport_report` returns each pillar's
percentile, weight, contribution, the base, and the gate, so the agent can
*explain* a ranking, not assert it.

**Derived analytics** on the same data: `congestion_level` (delay-based band, for
"compare X vs Y congestion"), `long_haul_pct` (≥2000 mi share), and a transparent
`unmet_demand` proxy (latent demand vs on-time-served capacity — directional, not
a forecast).

---

## 4. Where and how AI is used

**AI plans and narrates; code decides every number.** Claude/Gemini interprets
the question, resolves ambiguous names (`resolve_airport`), picks tools + args,
sequences multi-step work, and explains reasoning + methodology. The system
prompt forbids stating any figure from memory — if a tool lacks it, the agent
says so. All airport facts come from `scoring.py`. This satisfies
"deterministic scoring, not only LLM output" and prevents hallucinated
statistics — the failure mode that would make an investment tool untrustworthy.

**Voice (bonus).** A recorded microphone clip is transcribed by Gemini
(`src/voice.py`) and fed through the *identical* agent path as typed text — no
separate voice branch, so it inherits the same grounding and honesty guarantees.
Speech-to-text always uses Gemini (Claude can't ingest audio); the mic only
appears when a Gemini backend is configured, so the UI never advertises a
capability it can't deliver.

---

## 5. Key tradeoffs, assumptions & uncertainty (surfaced, not hidden)

1. **Not a profit/ROI model** (see §0): EOS ranks capacity *need/productivity*, a
   precondition for profit — no capex, revenue, or fee modeling. Biggest scoping
   line; the agent states it when asked.
2. **The demand pillar is a SINGLE YEAR of growth** (FAA CY24 vs CY23) — this is
   the **weakest pillar**. One year is noisy and mean-reverting: a single new
   route, a post-COVID rebound, or a base effect can spike or sink it, and it is
   backward-looking (not a forecast of future demand). Two things limit the
   damage: (a) the **choke gate** means growth alone can't inflate an un-choked
   airport, and (b) rankings apply a **~500k-passenger floor** so a tiny airport's
   +300% can't top the list. A multi-year CAGR or an FAA TAF forward forecast
   would be sturdier — the first analytics upgrade I'd make.
3. **Choke gate weighting** encodes the thesis (choke is necessary). Floor 0.60
   is a judgment call — all weights/floor live in `src/config.py` for sensitivity.
4. **31% of airports (146/478) have no delay data** (BTS covers only carriers with
   ≥0.5% of domestic revenue). Their choke pillar is median-imputed for the score
   and flagged **Medium/Low** `data_confidence`; verified they do **not** reach the
   top ranks. A more conservative choice would impute *low* (a no-data airport is
   probably not choked) or exclude them from ranking — noted.
5. **Slot-control list is curated** (JFK, LGA, DCA, EWR — FAA Level 3 as of 2024)
   and changes rarely; cited in `src/faa_designations.py`.
6. **Delays mix weather/ATC with physical choke.** Mitigated by combining them
   with utilization + slot-control rather than relying on delays alone.
7. **Long-haul % is domestic-only** (BTS On-Time scope) — understates
   intercontinental activity at gateways (SFO, JFK). Flagged in-answer.
8. **Unmet demand is a proxy, not a forecast.** Caveats returned with every result.
9. **Seasonality:** choke uses 3 sampled months (Feb/Jul/Oct 2024) — Jul skews busy.
10. **Scope:** US commercial airports; FAA↔IATA codes assumed to match (true for
    major commercial airports). **Known data gaps** (§2): seat load-factor and
    gate/terminal capacity — documented, not faked.

---

## 6. File map

```
data/build_dataset.py     offline build: download → join → derive → parquet
src/faa_designations.py   FAA slot-control + Core-30 lists (curated, cited)
src/scoring.py            choke-gated EOS, pillars, unmet demand, ranking   ← core IP
src/data_layer.py         snapshot load, airport-name resolver, region lookup
src/live_api.py           OpenSky + BTS T-100 live APIs (dormant — not in MVP tools)
src/tools.py              tool schemas + deterministic dispatch
src/agent.py              Claude tool-use loop + thesis system prompt
src/agent_gemini.py       Gemini function-calling loop (same tools/prompt)
src/llm.py                provider seam (Claude / Gemini)
src/voice.py              Gemini speech-to-text — voice input (bonus)
app.py                    dark dashboard (map + rankings) + floating chat + voice
chat_cli.py               terminal chat
tests/test_scoring.py     deterministic-engine tests (incl. gate behavior)
```
