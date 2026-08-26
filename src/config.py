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
# The investment thesis: renovation/expansion pays off where demand is large
# and growing but the airport is already capacity-constrained (congested), so
# added capacity converts directly into more served flights & passengers.
#
# Four pillars, each normalised to a 0-100 national percentile, then weighted.
# Weights must sum to 1.0 (asserted at load time).
EOS_WEIGHTS = {
    "congestion": 0.35,   # is it constrained NOW? (delays, cancellations)
    "growth": 0.30,       # is demand still rising? (YoY enplanement growth)
    "scale": 0.20,        # how big is the revenue base? (enplanements, log)
    "utilization": 0.15,  # how hard are existing runways worked?
}

# Weights for the composite congestion metric (before percentile-ranking).
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
