"""
Build the airport intelligence snapshot from public data sources.

This is the *offline* half of the hybrid data strategy: we pull public
government / open datasets once, join them on airport (IATA) code, derive a
handful of investment-relevant metrics, and cache the result as a single
parquet file that the agent reads at query time. The live half (OpenSky
real-time traffic) is handled separately in src/live_api.py.

Sources (all public, no API key):
  1. OurAirports        - airport metadata: geo, region/state, type
                          https://ourairports.com/data/
  2. OurAirports runways- runway counts (capacity proxy)
  3. FAA ACAIS          - CY2024 + CY2023 passenger enplanements, hub size,
                          %change (volume + growth signal)
  4. BTS On-Time Perf   - per-flight departure delays, cancellations, and
                          route distance -> congestion + long-haul %

Run:
    python data/build_dataset.py                 # default months
    python data/build_dataset.py --months 2024_7 2024_10
    python data/build_dataset.py --quick         # single month (fast dev)

Output:
    data/airport_snapshot.parquet   (shipped in repo; app reads this)
    data/snapshot_meta.json         (provenance + row counts + assumptions)
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(exist_ok=True)

OUT_PARQUET = DATA_DIR / "airport_snapshot.parquet"
OUT_META = DATA_DIR / "snapshot_meta.json"

# --- source URLs -----------------------------------------------------------
URL_AIRPORTS = "https://davidmegginson.github.io/ourairports-data/airports.csv"
URL_RUNWAYS = "https://davidmegginson.github.io/ourairports-data/runways.csv"
URL_FAA_ENPLANE = (
    "https://www.faa.gov/airports/planning_capacity/passenger_allcargo_stats/"
    "passenger/ARP-cy2024-all-enplanements.xlsx"
)
BTS_PREZIP = "https://transtats.bts.gov/PREZIP/"
BTS_ONTIME_TMPL = (
    BTS_PREZIP + "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{month}.zip"
)

# --- tunable assumptions (surfaced in snapshot_meta.json) ------------------
# "Long-haul" has no single FAA definition. In the *domestic* On-Time dataset
# the longest stage lengths are ~transcontinental (~2,700 mi) plus Alaska/
# Hawaii. We tier by great-circle stage length in statute miles:
LONG_HAUL_MILES = 2000     # >= this counts as long-haul
MEDIUM_HAUL_MILES = 700    # [700, 2000) medium; < 700 short
DELAY_THRESHOLD_MIN = 15   # BTS-standard "delayed" definition (DepDel15)

DEFAULT_MONTHS = ["2024_2", "2024_7", "2024_10"]  # winter / summer / fall spread

HEADERS = {"User-Agent": "airport-intel-agent/1.0 (take-home exam; contact analyst)"}


def log(msg: str) -> None:
    print(f"[build] {msg}", flush=True)


def download(url: str, dest: Path, min_bytes: int = 1000) -> Path:
    """Stream a URL to disk, skipping if a plausible cached copy exists."""
    if dest.exists() and dest.stat().st_size >= min_bytes:
        log(f"cache hit: {dest.name} ({dest.stat().st_size:,} B)")
        return dest
    log(f"downloading {url}")
    with requests.get(url, headers=HEADERS, stream=True, timeout=300) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        tmp.rename(dest)
    log(f"saved {dest.name} ({dest.stat().st_size:,} B)")
    return dest


# ---------------------------------------------------------------------------
# 1 + 2. OurAirports metadata & runway counts
# ---------------------------------------------------------------------------
def build_metadata() -> pd.DataFrame:
    ap_path = download(URL_AIRPORTS, RAW_DIR / "airports.csv")
    rw_path = download(URL_RUNWAYS, RAW_DIR / "runways.csv")

    ap = pd.read_csv(ap_path, low_memory=False)
    # US commercial-service airports with an IATA code.
    ap = ap[
        (ap["iso_country"] == "US")
        & (ap["type"].isin(["large_airport", "medium_airport"]))
        & (ap["scheduled_service"] == "yes")
        & (ap["iata_code"].notna())
        & (ap["iata_code"].str.len() == 3)
    ].copy()
    ap["iata"] = ap["iata_code"].str.upper()
    ap["state"] = ap["iso_region"].str.replace("US-", "", regex=False)
    meta = ap[
        ["iata", "ident", "name", "municipality", "iso_region", "state",
         "latitude_deg", "longitude_deg", "type"]
    ].rename(
        columns={"ident": "icao", "municipality": "city",
                 "latitude_deg": "lat", "longitude_deg": "lon"}
    )

    # runway counts by ICAO ident (exclude closed runways)
    rw = pd.read_csv(rw_path, low_memory=False)
    if "closed" in rw.columns:
        rw = rw[pd.to_numeric(rw["closed"], errors="coerce").fillna(0) == 0]
    rw_counts = (
        rw[rw["airport_ident"].isin(meta["icao"])]
        .groupby("airport_ident")
        .size()
        .rename("num_runways")
    )
    meta = meta.merge(rw_counts, left_on="icao", right_index=True, how="left")
    meta["num_runways"] = meta["num_runways"].fillna(1).astype(int)

    # de-dupe on IATA (a few shared codes); keep the large_airport row
    meta["_rank"] = (meta["type"] == "large_airport").astype(int)
    meta = (
        meta.sort_values("_rank", ascending=False)
        .drop_duplicates("iata")
        .drop(columns="_rank")
    )
    log(f"metadata: {len(meta)} US commercial airports")
    return meta


# ---------------------------------------------------------------------------
# 3. FAA passenger enplanements (volume + YoY growth + hub size)
# ---------------------------------------------------------------------------
def build_faa() -> pd.DataFrame:
    path = download(URL_FAA_ENPLANE, RAW_DIR / "faa_enplanements_cy2024.xlsx")
    # FAA sheets carry a title/blank row before the header; detect it.
    raw = pd.read_excel(path, header=None, dtype=str)
    header_row = None
    for i in range(min(12, len(raw))):
        row = raw.iloc[i].astype(str).str.strip().str.lower().tolist()
        if any(c == "locid" for c in row):
            header_row = i
            break
    if header_row is None:
        raise RuntimeError("Could not locate 'Locid' header row in FAA xlsx")
    df = pd.read_excel(path, header=header_row)
    df.columns = [str(c).strip() for c in df.columns]

    def find_col(*needles: str) -> str | None:
        for c in df.columns:
            cl = c.lower()
            if all(n in cl for n in needles):
                return c
        return None

    c_loc = find_col("locid")
    c_hub = find_col("hub")
    c_cy24 = find_col("cy 24") or find_col("cy24") or find_col("2024")
    c_cy23 = find_col("cy 23") or find_col("cy23") or find_col("2023")
    c_pct = find_col("%") or find_col("change")
    if not (c_loc and c_cy24):
        raise RuntimeError(f"FAA columns not found. Got: {list(df.columns)}")

    out = pd.DataFrame({
        "iata": df[c_loc].astype(str).str.strip().str.upper(),
        "hub_size": df[c_hub].astype(str).str.strip() if c_hub else None,
        "enplanements_2024": pd.to_numeric(
            df[c_cy24].astype(str).str.replace(",", "", regex=False), errors="coerce"),
    })
    if c_cy23:
        out["enplanements_2023"] = pd.to_numeric(
            df[c_cy23].astype(str).str.replace(",", "", regex=False), errors="coerce")
    if c_pct:
        out["pax_growth_pct"] = pd.to_numeric(
            df[c_pct].astype(str).str.replace("%", "", regex=False), errors="coerce")

    out = out[out["iata"].str.len() == 3].dropna(subset=["enplanements_2024"])
    # compute growth if the % column was absent
    if "pax_growth_pct" not in out and "enplanements_2023" in out:
        out["pax_growth_pct"] = (
            (out["enplanements_2024"] - out["enplanements_2023"])
            / out["enplanements_2023"] * 100
        )
    # FAA stores "% Change" as a fraction (0.0306 == 3.06%); normalise to percent
    if "pax_growth_pct" in out:
        med = out["pax_growth_pct"].abs().median()
        if pd.notna(med) and med < 1.5:
            out["pax_growth_pct"] = out["pax_growth_pct"] * 100
    out = out.drop_duplicates("iata")
    log(f"FAA enplanements: {len(out)} airports")
    return out


# ---------------------------------------------------------------------------
# 4. BTS On-Time Performance (congestion + long-haul share)
# ---------------------------------------------------------------------------
ONTIME_COLS = ["Origin", "Dest", "DepDelayMinutes", "DepDel15",
               "Cancelled", "Distance", "Reporting_Airline"]


def _read_ontime_month(month: str) -> pd.DataFrame:
    url = BTS_ONTIME_TMPL.format(month=month)
    dest = RAW_DIR / f"ontime_{month}.zip"
    download(url, dest, min_bytes=100_000)
    with zipfile.ZipFile(dest) as z:
        csv_name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        with z.open(csv_name) as fh:
            # header first, so we only load the columns we actually need
            head = pd.read_csv(io.BytesIO(fh.read(1 << 16)), nrows=0)
        cols = [c for c in ONTIME_COLS if c in head.columns]
        with z.open(csv_name) as fh:
            df = pd.read_csv(fh, usecols=cols, low_memory=False)
    df["_month"] = month
    return df


def build_ontime(months: list[str]) -> pd.DataFrame:
    frames = [_read_ontime_month(m) for m in months]
    ot = pd.concat(frames, ignore_index=True)
    log(f"On-Time raw flights: {len(ot):,} across {len(months)} month(s)")

    ot["is_delayed"] = pd.to_numeric(ot.get("DepDel15"), errors="coerce").fillna(0)
    ot["is_cancelled"] = pd.to_numeric(ot.get("Cancelled"), errors="coerce").fillna(0)
    ot["dep_delay_min"] = pd.to_numeric(ot.get("DepDelayMinutes"), errors="coerce")
    ot["distance"] = pd.to_numeric(ot.get("Distance"), errors="coerce")
    ot["is_long_haul"] = (ot["distance"] >= LONG_HAUL_MILES).astype(float)

    g = ot.groupby("Origin")
    agg = pd.DataFrame({
        "departures_sampled": g.size(),
        "pct_delayed_15": g["is_delayed"].mean() * 100,
        "mean_dep_delay_min": g["dep_delay_min"].mean(),
        "pct_cancelled": g["is_cancelled"].mean() * 100,
        "long_haul_pct": g["is_long_haul"].mean() * 100,
        "mean_stage_length_mi": g["distance"].mean(),
        "n_destinations": g["Dest"].nunique(),
    })
    # normalise to average monthly departures for interpretability
    agg["departures_per_month"] = (agg["departures_sampled"] / len(months)).round(0)
    agg = agg.reset_index().rename(columns={"Origin": "iata"})
    agg["iata"] = agg["iata"].str.upper()
    log(f"On-Time aggregated: {len(agg)} origin airports")
    return agg


# ---------------------------------------------------------------------------
# assemble
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="*", default=None,
                    help="BTS On-Time months as YYYY_M (e.g. 2024_7)")
    ap.add_argument("--quick", action="store_true",
                    help="single-month sample for fast dev iteration")
    args = ap.parse_args()

    if args.quick:
        months = ["2024_7"]
    elif args.months:
        months = args.months
    else:
        months = DEFAULT_MONTHS

    meta = build_metadata()
    faa = build_faa()
    ontime = build_ontime(months)

    df = meta.merge(faa, on="iata", how="left").merge(ontime, on="iata", how="left")

    # derived capacity/utilisation proxy: departures relative to runway count
    df["dep_per_runway_month"] = (
        df["departures_per_month"] / df["num_runways"].clip(lower=1)
    ).round(0)

    # Keep airports that have *either* meaningful passenger volume or flight
    # activity (drops GA/tiny fields that slipped through the type filter).
    has_signal = (df["enplanements_2024"].fillna(0) > 0) | (
        df["departures_per_month"].fillna(0) > 0)
    df = df[has_signal].copy()

    df = df.sort_values("enplanements_2024", ascending=False, na_position="last")
    df.to_parquet(OUT_PARQUET, index=False)

    meta_out = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_airports": int(len(df)),
        "n_with_enplanements": int(df["enplanements_2024"].notna().sum()),
        "n_with_ontime": int(df["departures_per_month"].notna().sum()),
        "ontime_months": months,
        "sources": {
            "airports": URL_AIRPORTS,
            "runways": URL_RUNWAYS,
            "faa_enplanements": URL_FAA_ENPLANE,
            "bts_ontime_template": BTS_ONTIME_TMPL,
        },
        "assumptions": {
            "long_haul_miles": LONG_HAUL_MILES,
            "medium_haul_miles": MEDIUM_HAUL_MILES,
            "delay_threshold_min": DELAY_THRESHOLD_MIN,
            "long_haul_scope": (
                "Domestic mainline flights only. The BTS On-Time dataset covers "
                "US carriers with >=0.5% of domestic scheduled passenger revenue; "
                "international segments are NOT included, so long-haul % understates "
                "intercontinental activity at gateway airports."
            ),
            "enplanements_year": "FAA ACAIS CY2024 vs CY2023",
            "congestion_basis": (
                f"share of departures delayed >{DELAY_THRESHOLD_MIN} min + "
                "cancellation rate + mean departure delay, from sampled months"
            ),
        },
        "columns": list(df.columns),
    }
    OUT_META.write_text(json.dumps(meta_out, indent=2))
    log(f"WROTE {OUT_PARQUET.name}: {len(df)} airports, {len(df.columns)} cols")
    log(f"WROTE {OUT_META.name}")

    # quick sanity print
    cols = ["iata", "name", "state", "enplanements_2024", "pax_growth_pct",
            "pct_delayed_15", "long_haul_pct", "num_runways"]
    cols = [c for c in cols if c in df.columns]
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(df[cols].head(12).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
