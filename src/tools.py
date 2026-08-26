"""
Agent tool surface: JSON-Schema declarations given to the LLM, plus a
deterministic dispatcher that maps each tool call to the scoring/data layer.

Design note: the LLM never computes airport metrics. It picks a tool and
arguments; the tool returns code-computed facts; the LLM narrates them. This is
the "deterministic scoring, LLM describes" split the exam asks for.
"""
from __future__ import annotations

from . import data_layer, live_api, scoring

# ---------------------------------------------------------------------------
# Tool declarations (Anthropic tool-use schema)
# ---------------------------------------------------------------------------
TOOL_SPECS = [
    {
        "name": "resolve_airport",
        "description": (
            "Resolve a free-text airport reference (IATA code, city, or airport "
            "name) to one or more IATA codes. ALWAYS call this first when the user "
            "names an airport by anything other than a clear 3-letter IATA code, or "
            "when a city may have several airports (e.g. 'LA', 'New York'). Returns "
            "candidate matches and any sibling airports in the same metro."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "e.g. 'Santa Ana', 'LA', 'SFO', 'Boston Logan'"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "airport_report",
        "description": (
            "Full metric + Expansion Opportunity Score (EOS) breakdown for ONE "
            "airport by IATA code. Returns passenger volume, YoY growth, congestion "
            "(delay/cancel rates + national percentile), long-haul %, runways, and "
            "the four weighted EOS pillars with their contributions. Use for "
            "'tell me about X' or as the basis for an expansion recommendation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"iata": {"type": "string", "description": "3-letter IATA code, e.g. SFO"}},
            "required": ["iata"],
        },
    },
    {
        "name": "rank_airports",
        "description": (
            "Rank airports by a metric (default the Expansion Opportunity Score). "
            "Scope with a named region (e.g. 'New England', 'West Coast') OR a list "
            "of state codes, or omit both to rank the whole US universe. Use for "
            "'which airports in <region> are strong expansion candidates'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {"type": "string", "description": "named region, e.g. 'New England'"},
                "states": {"type": "array", "items": {"type": "string"}, "description": "USPS state codes, e.g. ['MA','CT']"},
                "by": {
                    "type": "string",
                    "description": "metric to rank by",
                    "enum": ["eos", "congestion_pct", "pax_growth_pct", "enplanements_2024",
                             "long_haul_pct", "unmet_demand", "departures_per_month"],
                },
                "top_n": {"type": "integer", "description": "how many to return (default 10)"},
                "min_enplanements": {"type": "number", "description": "optional floor on annual passengers to exclude tiny airports"},
            },
        },
    },
    {
        "name": "compare_airports",
        "description": (
            "Side-by-side comparison of 2+ airports on congestion, growth, volume, "
            "long-haul %, runways and EOS. Use for 'compare X and Y', especially "
            "congestion questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "iatas": {"type": "array", "items": {"type": "string"}, "description": "IATA codes, e.g. ['LAX','SNA']"}
            },
            "required": ["iatas"],
        },
    },
    {
        "name": "long_haul_profile",
        "description": (
            "Long-haul / medium / short flight mix for ONE airport, with the exact "
            "distance definition and the domestic-only scope caveat. Use for "
            "'what percentage of long-haul flights out of X'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"iata": {"type": "string"}},
            "required": ["iata"],
        },
    },
    {
        "name": "unmet_demand",
        "description": (
            "Estimate unmet flight demand for ONE airport using a transparent proxy "
            "(latent demand vs on-time-served capacity) and return the driver "
            "breakdown + caveats. Use for 'what is the unmet demand at X and why'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"iata": {"type": "string"}},
            "required": ["iata"],
        },
    },
    {
        "name": "live_traffic",
        "description": (
            "Real-time aircraft near an airport from the OpenSky live API (a coarse "
            "current-activity gauge, not scheduled capacity). Best-effort; may be "
            "unavailable. Use only when the user asks about live/current traffic."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"iata": {"type": "string"}},
            "required": ["iata"],
        },
    },
    {
        "name": "universe_summary",
        "description": (
            "What the agent can see: airport count, data coverage, the sampled "
            "months, the scoring assumptions, and the current top-5 by EOS. Use to "
            "explain scope/methodology or when the user asks what data you have."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
def dispatch(name: str, tool_input: dict) -> dict:
    """Run one tool call. Returns a JSON-serialisable dict (never raises to caller)."""
    try:
        return _dispatch(name, tool_input or {})
    except Exception as e:  # noqa: BLE001 - surface as tool error, keep loop alive
        return {"error": f"{type(e).__name__}: {e}"}


def _dispatch(name: str, i: dict) -> dict:
    if name == "resolve_airport":
        return data_layer.resolve_airport(i["query"])

    if name == "airport_report":
        r = scoring.airport_report(i["iata"])
        return r or {"error": f"no airport found for IATA '{i['iata']}'"}

    if name == "rank_airports":
        states = i.get("states")
        if i.get("region"):
            reg = data_layer.airports_in_region(i["region"])
            if not reg["states"]:
                return {"error": f"unknown region '{i['region']}'. "
                        f"Known: {', '.join(data_layer.list_known_regions())}, or pass state codes."}
            states = reg["states"]
        # special metric: unmet_demand isn't a column — rank by it explicitly
        by = i.get("by", "eos")
        if by == "unmet_demand":
            rows = scoring.rank_airports(states=states, by="congestion_pct",
                                         top_n=i.get("top_n", 10),
                                         min_enplanements=i.get("min_enplanements"))
            for row in rows:
                ud = scoring.unmet_demand(row["iata"])
                row["unmet_flights_per_month"] = ud.get("unmet_flights_per_month") if ud else None
            return {"ranked_by": "unmet_demand (approx via congestion)", "results": rows}
        return {"ranked_by": by,
                "results": scoring.rank_airports(states=states, by=by,
                                                 top_n=i.get("top_n", 10),
                                                 min_enplanements=i.get("min_enplanements"))}

    if name == "compare_airports":
        return scoring.compare_airports(i["iatas"])

    if name == "long_haul_profile":
        r = scoring.long_haul_profile(i["iata"])
        return r or {"error": f"no airport found for IATA '{i['iata']}'"}

    if name == "unmet_demand":
        r = scoring.unmet_demand(i["iata"])
        return r or {"error": f"no airport found for IATA '{i['iata']}'"}

    if name == "live_traffic":
        return live_api.nearby_traffic(i["iata"])

    if name == "universe_summary":
        return scoring.universe_summary()

    return {"error": f"unknown tool '{name}'"}
