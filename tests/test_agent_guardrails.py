"""
Regression tests for two agent guardrails found by live stress-testing:

  1. Tool results must be JSON-serialisable. `airport_report` returned numpy
     bool_ (slot_controlled / faa_core30), which crashed the whole Gemini turn
     with "Object of type bool_ is not JSON serializable". `tools.dispatch` now
     sanitizes every result (numpy -> native).

  2. Grounding detector. The agent once answered "How is the LA airport doing?"
     from memory (only resolve_airport called), fabricating EOS 88 and +10%
     growth when LAX actually shrank -7.8%. `tools.needs_grounding` flags any
     answer that states airport figures so the loop can force a tool call.
"""
import json

import numpy as np

from src import tools


def _assert_json(obj):
    json.dumps(obj)  # raises TypeError if any numpy/pandas type leaked through


def test_dispatch_results_are_json_serialisable():
    for name, args in [
        ("airport_report", {"iata": "SFO"}),
        ("airport_report", {"iata": "JFK"}),   # slot-controlled -> numpy bool_
        ("rank_airports", {"by": "eos", "top_n": 5, "min_enplanements": 500000}),
        ("compare_airports", {"iatas": ["BOS", "BDL"]}),
        ("unmet_demand", {"iata": "SFO"}),
        ("long_haul_profile", {"iata": "ANC"}),
    ]:
        _assert_json(tools.dispatch(name, args))


def test_slot_controlled_is_native_bool():
    r = tools.dispatch("airport_report", {"iata": "JFK"})
    assert isinstance(r["slot_controlled"], bool)
    assert not isinstance(r["slot_controlled"], np.generic)


def test_sanitize_coerces_numpy_and_nan():
    dirty = {"b": np.bool_(True), "i": np.int64(7), "f": np.float64(1.5),
             "nan": np.float64("nan"), "arr": np.array([1, 2]),
             "nested": {"x": np.bool_(False)}}
    clean = tools._sanitize(dirty)
    _assert_json(clean)
    assert clean["b"] is True and isinstance(clean["b"], bool)
    assert clean["i"] == 7 and isinstance(clean["i"], int)
    assert clean["nan"] is None
    assert clean["arr"] == [1, 2]
    assert clean["nested"]["x"] is False


def test_needs_grounding_flags_fabricated_figures():
    flag = [
        "LAX has an Expansion Opportunity Score (EOS) of 85.1.",
        "Choke is in the 99th percentile nationally.",
        "It handled 21,090,721 enplanements in 2024.",
        "Passenger growth of 10.1% year-over-year.",
        "The choke gate is 0.94.",
    ]
    for t in flag:
        assert tools.needs_grounding(t), f"should flag: {t}"


def test_needs_grounding_passes_legitimate_toolless_answers():
    ok = [
        "Not necessarily — the core of the thesis is capacity choke, not size.",
        "I cannot provide average load factors; my tools do not track how full planes are.",
        "I cannot forecast future demand.",
        "I can only compare US commercial airports. Heathrow is not in my database.",
        "I cannot fulfill that request. I am the Airport Investment Intelligence Agent.",
    ]
    for t in ok:
        assert not tools.needs_grounding(t), f"should pass: {t}"
