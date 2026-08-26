"""
Live aviation data via the OpenSky Network public API (no key required).

This is the *real-time* half of the hybrid data strategy. Where the cached
snapshot answers "how congested is this airport structurally?", OpenSky answers
"what does the sky around it look like right now?". It is a supplementary,
best-effort signal: rate-limited and occasionally down, so every call fails
soft and says so rather than blocking the agent.
"""
from __future__ import annotations

import time

import requests

from .data_layer import get_airport

OPENSKY_STATES = "https://opensky-network.org/api/states/all"
_CACHE: dict[str, tuple[float, dict]] = {}
_TTL_SECONDS = 120  # OpenSky updates slowly; don't hammer it


def nearby_traffic(iata: str, radius_deg: float = 0.6) -> dict:
    """
    Snapshot of aircraft currently within ~radius_deg of the airport.

    ~0.6 deg latitude is roughly 65 km. Returns counts of airborne vs
    on-ground aircraft as a coarse real-time activity indicator, plus a few
    sample callsigns. Always returns a dict with an "available" flag.
    """
    ap = get_airport(iata)
    if not ap:
        return {"available": False, "reason": f"unknown airport {iata}"}
    lat, lon = ap.get("lat"), ap.get("lon")
    if lat is None or lon is None:
        return {"available": False, "reason": "no coordinates for airport"}

    cache_key = f"{iata}:{radius_deg}"
    now = time.time()
    if cache_key in _CACHE and now - _CACHE[cache_key][0] < _TTL_SECONDS:
        return _CACHE[cache_key][1]

    params = {
        "lamin": lat - radius_deg, "lamax": lat + radius_deg,
        "lomin": lon - radius_deg, "lomax": lon + radius_deg,
    }
    try:
        r = requests.get(OPENSKY_STATES, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
    except Exception as e:  # noqa: BLE001 - live API, degrade gracefully
        return {"available": False, "reason": f"OpenSky unavailable: {e}",
                "airport": iata}

    states = data.get("states") or []
    airborne, on_ground, samples = 0, 0, []
    for s in states:
        # OpenSky state vector: idx 1 callsign, 8 on_ground, 7 baro_altitude
        on_gnd = bool(s[8]) if len(s) > 8 else False
        if on_gnd:
            on_ground += 1
        else:
            airborne += 1
        if len(samples) < 8 and s[1]:
            samples.append(s[1].strip())

    result = {
        "available": True,
        "airport": iata,
        "airport_name": ap.get("name"),
        "observed_at_unix": data.get("time"),
        "radius_km_approx": round(radius_deg * 111),
        "aircraft_total": len(states),
        "airborne": airborne,
        "on_ground": on_ground,
        "sample_callsigns": samples,
        "note": "Real-time OpenSky snapshot; a coarse activity gauge, not a "
                "capacity or scheduled-traffic measure.",
    }
    _CACHE[cache_key] = (now, result)
    return result
