"""
Data-access layer over the cached snapshot.

Responsibilities:
  * load the parquet snapshot + provenance metadata (once, cached)
  * resolve free-text airport references ("LA", "Santa Ana", "Boston Logan",
    "SFO") to IATA codes, flagging ambiguity and multi-airport metros
  * resolve named regions ("New England") and bare state codes to airport sets

No scoring here - this layer only knows facts, not judgments.
"""
from __future__ import annotations

import json
from functools import lru_cache

import pandas as pd

from . import config


@lru_cache(maxsize=1)
def load_snapshot() -> pd.DataFrame:
    if not config.SNAPSHOT_PARQUET.exists():
        raise FileNotFoundError(
            f"Snapshot not found at {config.SNAPSHOT_PARQUET}. "
            "Run: python data/build_dataset.py"
        )
    df = pd.read_parquet(config.SNAPSHOT_PARQUET)
    df["iata"] = df["iata"].str.upper()
    return df


@lru_cache(maxsize=1)
def load_meta() -> dict:
    if config.SNAPSHOT_META.exists():
        return json.loads(config.SNAPSHOT_META.read_text())
    return {}


def get_airport(iata: str) -> dict | None:
    df = load_snapshot()
    row = df[df["iata"] == iata.upper()]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def _norm(s: str) -> str:
    return " ".join(str(s).lower().strip().split())


def resolve_airport(query: str) -> dict:
    """
    Resolve a free-text airport reference.

    Returns a dict:
      { "query": str,
        "matches": [ {iata, name, city, state}, ... ],   # best-first
        "ambiguous": bool,          # >1 plausible airport
        "metro_alternatives": [iata, ...]  # other airports in the same metro
      }
    An empty "matches" list means no match was found.
    """
    df = load_snapshot()
    q = _norm(query)
    result = {"query": query, "matches": [], "ambiguous": False,
              "metro_alternatives": []}

    def row_to_match(iata: str) -> dict | None:
        r = df[df["iata"] == iata]
        if r.empty:
            return None
        r = r.iloc[0]
        return {"iata": r["iata"], "name": r["name"],
                "city": r.get("city"), "state": r.get("state")}

    # 1) exact IATA code
    if len(q) == 3 and q.upper() in set(df["iata"]):
        m = row_to_match(q.upper())
        if m:
            result["matches"] = [m]
            _attach_metro(q, result)
            return result

    # 2) curated alias
    if q in config.AIRPORT_ALIASES:
        m = row_to_match(config.AIRPORT_ALIASES[q])
        if m:
            result["matches"] = [m]
            _attach_metro(q, result)
            return result

    # 3) fuzzy match on city / name (substring, both directions)
    city = df["city"].fillna("").map(_norm)
    name = df["name"].fillna("").map(_norm)
    mask = (
        city.eq(q)
        | name.eq(q)
        | city.str.contains(rf"\b{q}\b", regex=True, na=False)
        | name.str.contains(rf"\b{q}\b", regex=True, na=False)
    )
    hits = df[mask].copy()
    if hits.empty:
        # looser containment fallback
        mask2 = city.str.contains(q, na=False) | name.str.contains(q, na=False)
        hits = df[mask2].copy()

    if not hits.empty:
        # prefer larger airports (by enplanements) when several match
        hits = hits.sort_values("enplanements_2024", ascending=False,
                                 na_position="last")
        result["matches"] = [
            {"iata": r["iata"], "name": r["name"], "city": r.get("city"),
             "state": r.get("state")}
            for _, r in hits.head(6).iterrows()
        ]
        result["ambiguous"] = len(hits) > 1
        _attach_metro(q, result)
    return result


def _attach_metro(query_norm: str, result: dict) -> None:
    """If a resolved airport belongs to a multi-airport metro, list its siblings.

    Keyed off the *resolved IATA's* membership in a metro group — not a substring
    match on the query (which wrongly matched e.g. 'la' inside 'dallas').
    """
    chosen = {m["iata"] for m in result["matches"]}
    df_codes = set(load_snapshot()["iata"])
    for codes in config.METRO_GROUPS.values():
        if chosen & set(codes):  # a resolved airport is in this metro
            result["metro_alternatives"] = [
                c for c in codes if c in df_codes and c not in chosen
            ]
            return


def airports_in_region(name: str) -> dict:
    """
    Resolve a named region or state code to its airports (as a snapshot slice).
    Returns {"label", "states", "df"}.
    """
    df = load_snapshot()
    key = _norm(name)
    if key in config.REGIONS:
        states = config.REGIONS[key]
        label = name.title()
    elif len(key) == 2 and key.upper() in set(df["state"]):
        states = [key.upper()]
        label = key.upper()
    else:
        # try a state name -> code isn't tracked; fall back to empty
        states = []
        label = name
    sub = df[df["state"].isin(states)].copy() if states else df.iloc[0:0].copy()
    return {"label": label, "states": states, "df": sub}


def list_known_regions() -> list[str]:
    return sorted(config.REGIONS.keys())
