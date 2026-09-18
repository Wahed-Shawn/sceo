"""Offline validation harness for the GridWise pipeline.

Loads the public sample case pack, POSTs every case through the real FastAPI
endpoint, then checks, per case:
  1. directive_interpretation matches the public reference semantics,
  2. the returned hourly_plan is VALID against the reference ground-truth
     directives and every GridWise constraint (energy balance, effective solar,
     battery bounds/rates, transition, end-of-day neutrality, directives),
  3. recalculated totals match the reported totals,
  4. total cost matches the reference optimal cost (equivalent optimal
     schedules are accepted; only the optimal-cost value is compared).

Run with GRIDWISE_LLM_MODE=rules for a deterministic, key-free check.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

TOL = 0.011

SAMPLE_FILE = Path(__file__).with_name("BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json")


def balance_check(row: Dict[str, Any], demand: float) -> Tuple[float, float]:
    charge = row["battery_kwh"] if row["battery_action"] == "charge" else 0.0
    discharge = row["battery_kwh"] if row["battery_action"] == "discharge" else 0.0
    lhs = row["grid_kwh"] + row["solar_used_kwh"] + discharge
    rhs = demand + charge
    return lhs, rhs


def effective_solar(solar: List[float], ground_truth: List[Dict[str, Any]]) -> List[float]:
    factors: List[float] = [1.0] * 24
    floors: List[float] = [0.0] * 24
    caps: List[float | None] = [None] * 24
    nc: List[bool] = [False] * 24
    nd: List[bool] = [False] * 24
    for d in ground_truth:
        if not d["applies"]:
            continue
        adj = d.get("structured_adjustment") or {}
        t = d["directive_type"]
        if t == "solar_reduction":
            for h in adj.get("hours", []):
                factors[h] *= adj["factor"]
        elif t == "minimum_battery_reserve":
            for h in adj.get("hours", []):
                floors[h] = max(floors[h], adj["minimum_energy_kwh"])
        elif t == "max_grid_window":
            for h in adj.get("hours", []):
                caps[h] = adj["max_grid_kwh"] if caps[h] is None else min(caps[h], adj["max_grid_kwh"])
        elif t == "no_charge_window":
            for h in adj.get("hours", []):
                nc[h] = True
        elif t == "no_discharge_window":
            for h in adj.get("hours", []):
                nd[h] = True
    return [solar[h] * factors[h] for h in range(24)], floors, caps, nc, nd


def validate_plan(
    case: Dict[str, Any], plan: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]
) -> List[str]:
    errors: List[str] = []
    inp = case["input"]
    battery = inp["battery"]
    hours = sorted(inp["hours"], key=lambda h: h["hour"])
    demand = [h["demand_kwh"] for h in hours]
    solar = [h["solar_kwh"] for h in hours]

    eff, floors, caps, nc, nd = effective_solar(solar, ground_truth)
    if len(plan) != 24 or [r["hour"] for r in plan] != list(range(24)):
        return ["hourly_plan must have exactly 24 entries for hours 0..23"]

    prev = None
    for h, row in enumerate(plan):
        if row["grid_kwh"] < -TOL:
            errors.append(f"h{h} grid negative")
        if row["solar_used_kwh"] < -TOL:
            errors.append(f"h{h} solar_used negative")
        lhs, rhs = balance_check(row, demand[h])
        if abs(lhs - rhs) > TOL:
            errors.append(f"h{h} energy balance off by {lhs - rhs:.4f}")
        if row["solar_used_kwh"] > eff[h] + TOL:
            errors.append(f"h{h} solar_used exceeds effective solar {eff[h]:.2f}")

        if row["battery_action"] == "charge":
            if row["battery_kwh"] > battery["max_charge_kwh_per_hour"] + TOL:
                errors.append(f"h{h} charge rate exceeded")
            net = row["battery_kwh"]
        elif row["battery_action"] == "discharge":
            if row["battery_kwh"] > battery["max_discharge_kwh_per_hour"] + TOL:
                errors.append(f"h{h} discharge rate exceeded")
            net = -row["battery_kwh"]
        else:
            if (
                abs(row["battery_kwh"]) > TOL
                or abs(row["grid_kwh"] + row["solar_used_kwh"] - demand[h]) > TOL
            ):
                errors.append(f"h{h} idle hour has nonzero battery flow or unmet demand")
            net = 0.0

        if row["battery_energy_after_kwh"] < -TOL:
            errors.append(f"h{h} energy negative")
        if row["battery_energy_after_kwh"] > battery["capacity_kwh"] + TOL:
            errors.append(f"h{h} energy above capacity")
        if row["battery_energy_after_kwh"] < floors[h] - TOL:
            errors.append(f"h{h} energy below reserve floor {floors[h]:.2f}")
        if caps[h] is not None and row["grid_kwh"] > caps[h] + TOL:
            errors.append(f"h{h} grid above cap {caps[h]:.2f}")
        if nc[h] and row["battery_action"] == "charge":
            errors.append(f"h{h} charged inside no-charge window")
        if nd[h] and row["battery_action"] == "discharge":
            errors.append(f"h{h} discharged inside no-discharge window")

        if prev is not None:
            expected = prev + net
            if abs(expected - row["battery_energy_after_kwh"]) > TOL:
                errors.append(f"h{h} battery transition off by {expected - row['battery_energy_after_kwh']:.4f}")
        else:
            expected = battery["initial_energy_kwh"] + net
            if abs(expected - row["battery_energy_after_kwh"]) > TOL:
                errors.append(f"h{h} first-hour transition off by {expected - row['battery_energy_after_kwh']:.4f}")
        prev = row["battery_energy_after_kwh"]

    if abs(plan[-1]["battery_energy_after_kwh"] - battery["initial_energy_kwh"]) > TOL:
        errors.append("end-of-day neutrality violated")
    return errors


def interpretations_match(got: List[Dict[str, Any]], expected: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    if [d["note_index"] for d in got] != [d["note_index"] for d in expected]:
        errors.append("note_index order mismatch")
    if len(got) != len(expected):
        return ["interpretation entry count mismatch"]
    for g, e in zip(got, expected):
        idx = g["note_index"]
        if g["directive_type"] != e["directive_type"]:
            errors.append(f"note {idx}: directive_type {g['directive_type']} != {e['directive_type']}")
        if g["applies"] != e["applies"]:
            errors.append(f"note {idx}: applies mismatch")
        ga, ea = g.get("structured_adjustment"), e.get("structured_adjustment")
        if (ga is None) != (ea is None):
            errors.append(f"note {idx}: adjustment presence mismatch")
            continue
        if ga is None:
            continue
        if sorted(ga.get("hours", [])) != sorted(ea.get("hours", [])):
            errors.append(f"note {idx}: hours {ga.get('hours')} != {ea.get('hours')}")
        for key in ("factor", "minimum_energy_kwh", "max_grid_kwh"):
            if key in ea and abs(float(ga.get(key, 0.0)) - float(ea[key])) > 0.011:
                errors.append(f"note {idx}: {key} {ga.get(key)} != {ea[key]}")
    return errors


def main() -> int:
    mode = os.environ.get("GRIDWISE_LLM_MODE", "openai")
    print(f"Interpretation mode: {mode}")

    from fastapi.testclient import TestClient

    import main  # noqa: F401

    client = TestClient(main.app)

    data = json.loads(SAMPLE_FILE.read_text(encoding="utf-8"))
    cases = data["cases"]
    tol_ok = 0
    failures = []

    for case in cases:
        cid = case["id"]
        resp = client.post("/optimize-energy", json=case["input"])
        body = resp.json()
        if resp.status_code != 200:
            failures.append(f"{cid}: HTTP {resp.status_code}")
            continue
        expected = case["expected_output"]

        interr = interpretations_match(body["directive_interpretation"], expected["directive_interpretation"])
        planerr = validate_plan(case, body["hourly_plan"], expected["directive_interpretation"])

        tot_grid = sum(r["grid_kwh"] for r in body["hourly_plan"])
        peak = max(r["grid_kwh"] for r in body["hourly_plan"])
        hours = sorted(case["input"]["hours"], key=lambda h: h["hour"])
        tariff = {h["hour"]: h["tariff_bdt_per_kwh"] for h in hours}
        cost = sum(r["grid_kwh"] * tariff[r["hour"]] for r in body["hourly_plan"])

        toterr = []
        if abs(tot_grid - body["total_grid_kwh"]) > TOL:
            toterr.append("total_grid_kwh recalc mismatch")
        if abs(peak - body["peak_grid_kwh"]) > TOL:
            toterr.append("peak_grid_kwh recalc mismatch")
        if abs(cost - body["total_cost_bdt"]) > TOL:
            toterr.append("total_cost_bdt recalc mismatch")
        if abs(body["total_cost_bdt"] - expected["total_cost_bdt"]) > TOL:
            toterr.append(f"cost {body['total_cost_bdt']} vs reference optimal {expected['total_cost_bdt']}")

        status = "PASS" if not (interr or planerr or toterr) else "FAIL"
        if status == "PASS":
            tol_ok += 1
        print(f"{cid}: {status}  cost={body['total_cost_bdt']} (ref {expected['total_cost_bdt']})")
        for e in interr:
            print(f"    [interp] {e}")
        for e in planerr:
            print(f"    [plan]   {e}")
        for e in toterr:
            print(f"    [total]  {e}")
        if status == "FAIL":
            failures.append(cid)

    print(f"\n{len(cases)} cases, {tol_ok} passed.")
    if failures:
        print("FAILED:", ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())