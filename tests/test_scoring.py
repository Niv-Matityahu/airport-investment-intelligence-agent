"""
Unit tests for the deterministic engine — no LLM, no network, runs against the
committed snapshot. These lock in the scoring contract the agent depends on.

    pytest tests/ -v
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, data_layer, scoring  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return scoring.build_scored()


def test_eos_weights_sum_to_one():
    assert abs(sum(config.EOS_WEIGHTS.values()) - 1.0) < 1e-9


def test_snapshot_has_core_columns(df):
    for col in ["iata", "name", "state", "enplanements_2024", "eos",
                "congestion_pct", "long_haul_pct", "num_runways"]:
        assert col in df.columns
    assert len(df) > 100  # a real US universe


def test_percentiles_bounded(df):
    for col in ["congestion_pct", "growth_pct_rank", "scale_pct", "utilization_pct", "eos"]:
        s = df[col].dropna()
        assert s.min() >= 0 and s.max() <= 100


# --- resolver ---------------------------------------------------------------
def test_resolve_exact_iata():
    r = data_layer.resolve_airport("SFO")
    assert r["matches"][0]["iata"] == "SFO"


def test_resolve_city_alias():
    assert data_layer.resolve_airport("Santa Ana")["matches"][0]["iata"] == "SNA"
    assert data_layer.resolve_airport("Anchorage")["matches"][0]["iata"] == "ANC"


def test_resolve_la_metro_no_false_substring():
    r = data_layer.resolve_airport("LA")
    assert r["matches"][0]["iata"] == "LAX"
    # the 'la' in 'dallas' bug: DFW/DAL must NOT appear as LA metro siblings
    assert "DFW" not in r["metro_alternatives"]
    assert "DAL" not in r["metro_alternatives"]
    assert "SNA" in r["metro_alternatives"]


# --- regions ----------------------------------------------------------------
def test_new_england_region():
    reg = data_layer.airports_in_region("New England")
    assert set(reg["states"]) == {"CT", "ME", "MA", "NH", "RI", "VT"}
    assert (reg["df"]["state"].isin(reg["states"])).all()


# --- scoring API ------------------------------------------------------------
def test_airport_report_breakdown_sums_to_eos():
    rep = scoring.airport_report("SFO")
    total = sum(p["contribution"] for p in rep["eos_breakdown"])
    assert math.isclose(total, rep["eos"], abs_tol=0.15)


def test_compare_returns_all_requested():
    cmp = scoring.compare_airports(["LAX", "SNA"])
    assert set(cmp["airports"]) == {"LAX", "SNA"}
    assert not cmp["missing"]


def test_long_haul_profile_bounds():
    lh = scoring.long_haul_profile("ANC")
    assert 0 <= lh["long_haul_pct"] <= 100
    assert str(config.LONG_HAUL_MILES) in lh["definition"]


def test_unmet_demand_non_negative():
    ud = scoring.unmet_demand("SFO")
    assert ud["available"] is True
    assert ud["unmet_flights_per_month"] >= 0
    assert set(ud["drivers"]) >= {"pct_delayed_15", "pct_cancelled", "pax_growth_pct"}


def test_rank_respects_volume_floor():
    rows = scoring.rank_airports(states=["MA", "CT", "ME", "NH", "RI", "VT"],
                                 by="eos", top_n=10, min_enplanements=500_000)
    assert all(r["enplanements_2024"] >= 500_000 for r in rows)
    assert rows == sorted(rows, key=lambda r: r["eos"], reverse=True)


def test_rank_whole_universe_default():
    rows = scoring.rank_airports(top_n=5)
    assert len(rows) == 5
    assert rows[0]["eos"] >= rows[-1]["eos"]


def test_unknown_airport_returns_none():
    assert scoring.airport_report("ZZZ") is None
    assert scoring.long_haul_profile("ZZZ") is None
