"""
Deterministic scoring & analytics engine.

This module is intentionally LLM-free. Every number the agent reports about an
airport's investment attractiveness, congestion, long-haul mix, or unmet demand
is computed here with transparent, reproducible arithmetic. The LLM's job is to
*route* questions to these functions and *narrate* their output - never to
invent the figures.

Core artefact: the Expansion Opportunity Score (EOS), a 0-100 weighted blend of
four national-percentile pillars. See config.EOS_WEIGHTS for the thesis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .data_layer import load_snapshot


# ---------------------------------------------------------------------------
# scored universe (computed once, cached)
# ---------------------------------------------------------------------------
_SCORED: pd.DataFrame | None = None


def _pct_rank(s: pd.Series) -> pd.Series:
    """National percentile (0-100), NaNs left as NaN."""
    return s.rank(pct=True) * 100


def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or hi == lo:
        return pd.Series(np.nan, index=s.index)
    return (s - lo) / (hi - lo)


def build_scored() -> pd.DataFrame:
    """Compute pillar percentiles + EOS for every airport. Cached."""
    global _SCORED
    if _SCORED is not None:
        return _SCORED

    assert abs(sum(config.EOS_WEIGHTS.values()) - 1.0) < 1e-9, "EOS weights must sum to 1"
    df = load_snapshot().copy()

    # --- congestion composite -> percentile ---------------------------------
    cw = config.CONGESTION_WEIGHTS
    comp = pd.Series(0.0, index=df.index)
    wsum = pd.Series(0.0, index=df.index)
    for col, w in cw.items():
        if col in df:
            norm = _minmax(df[col])
            comp = comp.add(norm * w, fill_value=0)
            wsum = wsum.add(norm.notna() * w, fill_value=0)
    df["congestion_raw"] = np.where(wsum > 0, comp / wsum.replace(0, np.nan), np.nan)
    df["congestion_pct"] = _pct_rank(df["congestion_raw"])

    # --- growth --------------------------------------------------------------
    df["growth_pct_rank"] = _pct_rank(df.get("pax_growth_pct"))

    # --- scale (log enplanements) -------------------------------------------
    df["scale_raw"] = np.log10(df["enplanements_2024"].clip(lower=1))
    df["scale_pct"] = _pct_rank(df["scale_raw"])

    # --- utilisation (departures per runway) --------------------------------
    df["utilization_pct"] = _pct_rank(df.get("dep_per_runway_month"))

    # --- Expansion Opportunity Score ----------------------------------------
    w = config.EOS_WEIGHTS
    pillars = {
        "congestion": "congestion_pct",
        "growth": "growth_pct_rank",
        "scale": "scale_pct",
        "utilization": "utilization_pct",
    }
    # median-impute missing pillars so partial-data airports still rank, but
    # track how much was imputed to express confidence.
    eos = pd.Series(0.0, index=df.index)
    imputed = pd.Series(0, index=df.index)
    for name, col in pillars.items():
        vals = df[col]
        filled = vals.fillna(vals.median())
        imputed += vals.isna().astype(int)
        eos += filled * w[name]
    df["eos"] = eos.round(1)
    df["pillars_imputed"] = imputed
    df["data_confidence"] = _confidence(df)

    # descriptive labels
    df["congestion_level"] = df["congestion_pct"].map(_congestion_band)

    _SCORED = df
    return df


def _confidence(df: pd.DataFrame) -> pd.Series:
    """Coarse High/Medium/Low confidence from data completeness + sample size."""
    dep = df.get("departures_per_month")
    imp = df["pillars_imputed"]
    out = []
    for i in df.index:
        n = dep.iloc[df.index.get_loc(i)] if dep is not None else np.nan
        k = imp.loc[i]
        if k == 0 and pd.notna(n) and n >= 500:
            out.append("High")
        elif k <= 1 and (pd.isna(n) or n >= 100):
            out.append("Medium")
        else:
            out.append("Low")
    return pd.Series(out, index=df.index)


def _congestion_band(pct: float) -> str:
    if pd.isna(pct):
        return "Unknown"
    for upper, label in config.CONGESTION_BANDS:
        if pct <= upper:
            return label
    return "Severe"


# ---------------------------------------------------------------------------
# public API used by the agent tools
# ---------------------------------------------------------------------------
_REPORT_FIELDS = [
    "iata", "name", "city", "state", "hub_size",
    "enplanements_2024", "pax_growth_pct",
    "pct_delayed_15", "mean_dep_delay_min", "pct_cancelled",
    "long_haul_pct", "mean_stage_length_mi", "n_destinations",
    "departures_per_month", "num_runways", "dep_per_runway_month",
    "congestion_pct", "congestion_level", "growth_pct_rank",
    "scale_pct", "utilization_pct", "eos", "data_confidence",
    "pillars_imputed",
]


def _clean(rec: dict) -> dict:
    out = {}
    for k, v in rec.items():
        if isinstance(v, (np.floating,)):
            v = float(v)
        elif isinstance(v, (np.integer,)):
            v = int(v)
        if isinstance(v, float) and (np.isnan(v)):
            v = None
        elif isinstance(v, float):
            v = round(v, 2)
        out[k] = v
    return out


def airport_report(iata: str) -> dict | None:
    df = build_scored()
    row = df[df["iata"] == iata.upper()]
    if row.empty:
        return None
    rec = {k: row.iloc[0].get(k) for k in _REPORT_FIELDS if k in df.columns}
    rec = _clean(rec)
    rec["eos_breakdown"] = _eos_breakdown(row.iloc[0])
    return rec


def _eos_breakdown(row: pd.Series) -> list[dict]:
    """Per-pillar contribution to the EOS, for transparent explanations."""
    w = config.EOS_WEIGHTS
    pillars = {
        "congestion": "congestion_pct",
        "growth": "growth_pct_rank",
        "scale": "scale_pct",
        "utilization": "utilization_pct",
    }
    out = []
    for name, col in pillars.items():
        pctl = row.get(col)
        imputed = pd.isna(pctl)
        val = float(pctl) if not imputed else None
        used = val if val is not None else 50.0  # median impute
        out.append({
            "pillar": name,
            "weight": w[name],
            "percentile": None if val is None else round(val, 1),
            "imputed": bool(imputed),
            "contribution": round(used * w[name], 1),
        })
    return out


def rank_airports(
    iatas: list[str] | None = None,
    states: list[str] | None = None,
    by: str = "eos",
    top_n: int = 10,
    min_enplanements: float | None = None,
) -> list[dict]:
    """
    Rank a subset of airports by a metric (default EOS).
    Provide `iatas` OR `states` to scope; omit both to rank the whole universe.
    """
    df = build_scored()
    if iatas:
        df = df[df["iata"].isin([i.upper() for i in iatas])]
    if states:
        df = df[df["state"].isin([s.upper() for s in states])]
    if min_enplanements is not None:
        df = df[df["enplanements_2024"].fillna(0) >= min_enplanements]
    if by not in df.columns:
        by = "eos"
    df = df.sort_values(by, ascending=False, na_position="last").head(top_n)
    return [_clean({k: r.get(k) for k in _REPORT_FIELDS if k in df.columns})
            for _, r in df.iterrows()]


def compare_airports(iatas: list[str], metrics: list[str] | None = None) -> dict:
    """Side-by-side comparison of named airports on selected metrics."""
    df = build_scored()
    codes = [i.upper() for i in iatas]
    sub = df[df["iata"].isin(codes)]
    default_metrics = [
        "eos", "congestion_pct", "congestion_level", "pct_delayed_15",
        "mean_dep_delay_min", "pct_cancelled", "pax_growth_pct",
        "enplanements_2024", "long_haul_pct", "departures_per_month",
        "num_runways", "dep_per_runway_month", "data_confidence",
    ]
    metrics = metrics or default_metrics
    rows = {}
    for _, r in sub.iterrows():
        rows[r["iata"]] = _clean(
            {"name": r["name"], "state": r.get("state"),
             **{m: r.get(m) for m in metrics if m in df.columns}}
        )
    # note any requested airports we couldn't find
    missing = [c for c in codes if c not in rows]
    return {"airports": rows, "metrics": metrics, "missing": missing}


def long_haul_profile(iata: str) -> dict | None:
    """
    Long-haul / medium / short flight mix for one airport, with the exact
    definition and scope caveats attached so the agent can be honest.
    """
    df = build_scored()
    row = df[df["iata"] == iata.upper()]
    if row.empty:
        return None
    r = row.iloc[0]
    return _clean({
        "iata": r["iata"],
        "name": r["name"],
        "long_haul_pct": r.get("long_haul_pct"),
        "mean_stage_length_mi": r.get("mean_stage_length_mi"),
        "departures_per_month": r.get("departures_per_month"),
        "n_destinations": r.get("n_destinations"),
        "definition": f"long-haul = scheduled departures with stage length "
                      f">= {config.LONG_HAUL_MILES} statute miles",
        "scope_caveat": _long_haul_scope(),
    })


def _long_haul_scope() -> str:
    from .data_layer import load_meta
    return (load_meta().get("assumptions", {}) or {}).get(
        "long_haul_scope", "domestic mainline flights only")


def unmet_demand(iata: str) -> dict | None:
    """
    Transparent proxy for unmet flight demand.

    Model (all monthly, deterministic):
      served_ontime = departures * (1 - pct_delayed/100 - pct_cancelled/100)
      latent_demand = departures * (1 + max(growth,0)/100)
      unmet = max(0, latent_demand - served_ontime)

    Interpretation: delays & cancellations reveal demand the airport cannot
    currently clear on schedule; positive passenger growth adds forward demand.
    This is a *directional indicator*, not a queueing-theoretic forecast - the
    caveats are returned alongside the numbers.
    """
    df = build_scored()
    row = df[df["iata"] == iata.upper()]
    if row.empty:
        return None
    r = row.iloc[0]
    dep = r.get("departures_per_month")
    delayed = r.get("pct_delayed_15")
    cancelled = r.get("pct_cancelled")
    growth = r.get("pax_growth_pct")
    if pd.isna(dep) or pd.isna(delayed):
        return _clean({
            "iata": r["iata"], "name": r["name"],
            "available": False,
            "reason": "insufficient on-time / flight data for this airport",
        })
    cancelled = 0.0 if pd.isna(cancelled) else cancelled
    growth = 0.0 if pd.isna(growth) else growth
    served = dep * (1 - delayed / 100 - cancelled / 100)
    latent = dep * (1 + max(growth, 0) / 100)
    unmet = max(0.0, latent - served)
    return _clean({
        "iata": r["iata"], "name": r["name"], "available": True,
        "departures_per_month": dep,
        "served_ontime_per_month": served,
        "latent_demand_per_month": latent,
        "unmet_flights_per_month": unmet,
        "unmet_demand_pct": (unmet / latent * 100) if latent else None,
        "congestion_level": r.get("congestion_level"),
        "drivers": {
            "pct_delayed_15": _num(delayed),
            "pct_cancelled": _num(cancelled),
            "pax_growth_pct": _num(growth),
            "runway_utilization_pctile": _num(r.get("utilization_pct")),
        },
        "caveats": [
            "Directional proxy, not a demand-forecast model.",
            "Delays are used as a spilled-demand signal; some delay is weather/"
            "ATC, not capacity.",
            "Domestic mainline flights only (BTS On-Time scope).",
        ],
    })


def _num(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), 2)


def universe_summary() -> dict:
    """High-level description of what the agent can and cannot see."""
    df = build_scored()
    from .data_layer import load_meta
    meta = load_meta()
    return {
        "n_airports": int(len(df)),
        "n_with_congestion_data": int(df["congestion_pct"].notna().sum()),
        "top5_by_eos": rank_airports(top_n=5),
        "ontime_months": meta.get("ontime_months"),
        "assumptions": meta.get("assumptions"),
        "built_at_utc": meta.get("built_at_utc"),
    }
