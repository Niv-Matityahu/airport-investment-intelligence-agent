"""
Central configuration: file paths, scoring weights, thresholds, region and
alias lookups. Everything a reviewer might want to challenge or tune lives
here, in the open, rather than being buried in logic.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SNAPSHOT_PARQUET = DATA_DIR / "airport_snapshot.parquet"
SNAPSHOT_META = DATA_DIR / "snapshot_meta.json"

# --- Expansion Opportunity Score (EOS) --------------------------------------
# Investment thesis (three conditions must CO-OCCUR): expansion pays off where
#   1. CAPACITY CHOKE  — the airport is physically/regulatorily maxed out, so it
#      spills flights it cannot serve (delays, cancellations, runways worked hard,
#      FAA slot control). This is THE differentiator: no choke → no bottleneck to
#      relieve → no fast payoff, however big the airport.
#   2. DEMAND momentum — traffic is growing (more flights/passengers to capture).
#   3. SCALE           — a base large enough to justify major capex.
#
# EOS = (weighted percentile blend) * CHOKE GATE, so a big, growing, but
# UN-choked airport is dampened — the score REQUIRES choke, not just rewards it.
# Weights must sum to 1.0 (asserted at load time).
EOS_WEIGHTS = {
    "choke": 0.45,    # capacity choke — the bottleneck (the differentiator)
    "demand": 0.30,   # YoY passenger growth
    "scale": 0.25,    # revenue base (enplanements, log)
}

# Capacity-choke composite (before percentile-ranking): delays + cancellations
# + runway utilisation. Slot-control adds a bonus (see SLOT_CHOKE_BONUS).
CHOKE_WEIGHTS = {
    "pct_delayed_15": 0.35,
    "mean_dep_delay_min": 0.20,
    "pct_cancelled": 0.15,
    "dep_per_runway_month": 0.30,
}
# FAA slot-controlled airports get this added to their normalised choke (0-1)
# before ranking — a hard "the FAA caps flights here" signal.
SLOT_CHOKE_BONUS = 0.20
# Choke gate: final EOS = base * (FLOOR + (1-FLOOR) * choke_pct/100).
# FLOOR=0.6 means a zero-choke airport keeps only 60% of its base score.
CHOKE_GATE_FLOOR = 0.60

# Pure delay-based congestion (for "compare X vs Y congestion" questions — kept
# separate from the broader choke pillar).
CONGESTION_WEIGHTS = {
    "pct_delayed_15": 0.5,
    "mean_dep_delay_min": 0.3,
    "pct_cancelled": 0.2,
}

# Congestion percentile -> human label (upper bound, inclusive).
CONGESTION_BANDS = [
    (25, "Low"),
    (50, "Moderate"),
    (75, "Elevated"),
    (90, "High"),
    (100, "Severe"),
]

# Long-haul tiering by stage length (statute miles). Mirrors build_dataset.py.
LONG_HAUL_MILES = 2000
MEDIUM_HAUL_MILES = 700

# --- named US regions -> USPS state codes -----------------------------------
REGIONS = {
    "new england": ["CT", "ME", "MA", "NH", "RI", "VT"],
    "northeast": ["CT", "ME", "MA", "NH", "RI", "VT", "NY", "NJ", "PA"],
    "mid-atlantic": ["NY", "NJ", "PA", "DE", "MD", "DC", "VA"],
    "southeast": ["VA", "NC", "SC", "GA", "FL", "AL", "MS", "TN", "KY", "WV"],
    "midwest": ["OH", "MI", "IN", "IL", "WI", "MN", "IA", "MO", "ND", "SD", "NE", "KS"],
    "south": ["TX", "OK", "AR", "LA", "MS", "AL", "TN", "KY", "GA", "FL", "SC", "NC"],
    "southwest": ["AZ", "NM", "NV", "TX", "OK"],
    "mountain west": ["MT", "ID", "WY", "CO", "UT", "NV"],
    "west coast": ["CA", "OR", "WA"],
    "pacific northwest": ["WA", "OR"],
    "west": ["CA", "OR", "WA", "NV", "AZ", "ID", "UT", "CO", "MT", "WY", "NM"],
    "california": ["CA"],
    "new york metro": ["NY", "NJ"],
}

# --- curated aliases for ambiguous / common colloquial names ----------------
# Only the cases where a substring match on city/name would be wrong or
# ambiguous. Everything else is resolved by fuzzy city/name matching.
AIRPORT_ALIASES = {
    "la": "LAX",
    "los angeles": "LAX",
    "nyc": "JFK",
    "new york": "JFK",
    "new york city": "JFK",
    "santa ana": "SNA",
    "orange county": "SNA",
    "john wayne": "SNA",
    "anchorage": "ANC",
    "sf": "SFO",
    "san francisco": "SFO",
    "bay area": "SFO",
    "chicago": "ORD",
    "washington": "DCA",
    "washington dc": "DCA",
    "dc": "DCA",
    "dallas": "DFW",
    "houston": "IAH",
    "boston": "BOS",
    "miami": "MIA",
    "vegas": "LAS",
    "las vegas": "LAS",
    "denver": "DEN",
    "seattle": "SEA",
    "atlanta": "ATL",
    "phoenix": "PHX",
}

# Metros with several commercial airports — surfaced when a user names the city
# so the agent can offer the alternatives.
METRO_GROUPS = {
    "los angeles": ["LAX", "BUR", "LGB", "SNA", "ONT"],
    "new york": ["JFK", "LGA", "EWR"],
    "washington": ["DCA", "IAD", "BWI"],
    "chicago": ["ORD", "MDW"],
    "san francisco": ["SFO", "OAK", "SJC"],
    "houston": ["IAH", "HOU"],
    "dallas": ["DFW", "DAL"],
}
