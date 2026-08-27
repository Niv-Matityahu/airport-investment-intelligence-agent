"""
FAA capacity designations — the authoritative "is this airport constrained?"
markers, encoded as small curated lists (they change rarely and aren't in a
clean public API).

Why this matters for the thesis: the fund profits by *building capacity where
it is choked*. The FAA itself designates the choked airports:

  * Slot-controlled / Level 3 — the FAA legally CAPS the number of scheduled
    operations. This is the purest "no room for new flights" signal there is.
    Sources: 14 CFR Part 93 (JFK, LGA High Density / slot rules; DCA), and the
    FAA's 2024 reinstatement of EWR as IATA Level 3 (schedule-coordinated).

  * FAA Core 30 — the 30 airports the FAA tracks for system congestion/capacity
    (the OEP-35 successor). Membership marks a major, congestion-relevant hub.
    Source: FAA Operations Network (OPSNET) / ASPM "Core 30".

These are used to (a) boost the capacity-choke pillar and (b) surface an
explicit, explainable "capacity flag" in the agent's answers.
"""
from __future__ import annotations

# FAA Level 3 / slot-controlled (schedule-coordinated) US airports.
SLOT_CONTROLLED = {"JFK", "LGA", "DCA", "EWR"}

# FAA Core 30 (30 busiest / congestion-tracked airports).
FAA_CORE_30 = {
    "ATL", "BOS", "BWI", "CLT", "DCA", "DEN", "DFW", "DTW", "EWR", "FLL",
    "HNL", "IAD", "IAH", "JFK", "LAS", "LAX", "LGA", "MCO", "MDW", "MEM",
    "MIA", "MSP", "ORD", "PHL", "PHX", "SAN", "SEA", "SFO", "SLC", "TPA",
}


def capacity_flags(iata: str) -> dict:
    code = (iata or "").upper()
    return {
        "slot_controlled": code in SLOT_CONTROLLED,
        "faa_core30": code in FAA_CORE_30,
    }
