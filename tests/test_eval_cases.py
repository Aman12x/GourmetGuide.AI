"""Offline checks on the eval set itself and the eval scoring code."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from run_eval import field_ok, gate, load_cases, summarize  # noqa: E402

from retrieval import Filters, matches


def test_case_ids_unique_and_relevant_ids_exist(dishes):
    cases = load_cases()
    assert len(cases) >= 30
    assert len({c["id"] for c in cases}) == len(cases)
    ids = {d["id"] for d in dishes}
    for case in cases:
        assert set(case["relevant"]) <= ids, case["id"]


def test_every_labelled_dish_satisfies_its_cases_constraints(dishes):
    by_id = {d["id"]: d for d in dishes}
    for case in load_cases():
        filters = Filters.from_plan(case["expect"])
        for dish_id in case["relevant"]:
            assert matches(by_id[dish_id], filters), (case["id"], dish_id)


def test_field_ok_rules():
    assert field_ok("cuisines", ["Indian", "South Indian"], ["Indian"])
    assert not field_ok("cuisines", ["Chinese"], ["Indian"])
    assert field_ok("exclude_allergens", ["peanut", "soy"], ["peanuts"])
    assert not field_ok("exclude_allergens", [], ["dairy"])
    assert field_ok("max_price", 12.0, 12) and not field_ok("max_price", None, 12)
    assert field_ok("vegetarian", True, True) and not field_ok("vegetarian", None, True)


def test_summary_and_gate():
    results = [
        {"intent_ok": True, "filters_ok": {"max_price": True}, "hit": True, "model_calls": 2, "seconds": 1.0,
         "picks": [{"grounded": True, "violates": False}]},
        {"intent_ok": True, "filters_ok": {}, "model_calls": 1, "seconds": 0.5},
    ]
    s = summarize(results)
    assert s["hit_rate"] == 1.0 and s["groundedness"] == 1.0 and s["mean_model_calls"] == 1.5
    assert gate(s) == []
    results[0]["picks"][0]["violates"] = True
    assert gate(summarize(results)) == ["constraint_violations"]
