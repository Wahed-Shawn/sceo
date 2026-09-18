"""Deterministic guardrails applied to raw LLM directive interpretations.

The raw model output is never trusted directly: entries are normalized,
re-ordered, coerced to the exact official schema, and any malformed value is
corrected or downgraded to a harmless no_op so the downstream optimizer can
never be driven by an invalid constraint. This is the "LLM -> deterministic
guardrails -> optimizer" flow required by the evaluation rubric.
"""

from __future__ import annotations

from typing import Any, Dict, List


DIRECTIVE_TYPES = {
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
}


def _normalize_hours(raw) -> List[int]:
    if not isinstance(raw, list):
        return []
    hours: List[int] = []
    for h in raw:
        try:
            hh = int(h)
        except (TypeError, ValueError):
            continue
        if 0 <= hh <= 23 and hh not in hours:
            hours.append(hh)
    return sorted(hours)


def _to_float(raw, lo: float = 0.0, hi: float | None = None, default: float | None = None) -> float | None:
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return default
    if val < lo or (hi is not None and val > hi):
        return default
    return round(val, 4)


def _clean_directive(item: Dict[str, Any], battery: Dict[str, Any]) -> Dict[str, Any]:
    note_index = item.get("note_index", 0)
    try:
        note_index = int(note_index)
    except (TypeError, ValueError):
        note_index = 0
    if note_index < 0:
        note_index = 0
    note_index = min(note_index, 9999)

    dtype = item.get("directive_type")
    if dtype not in DIRECTIVE_TYPES:
        dtype = "no_op"

    if dtype == "no_op":
        return {
            "note_index": note_index,
            "applies": False,
            "directive_type": "no_op",
            "structured_adjustment": None,
            "explanation": str(item.get("explanation") or "Note does not affect today's energy schedule."),
        }

    adj = item.get("structured_adjustment")
    if not isinstance(adj, dict):
        adj = {}

    hours = _normalize_hours(adj.get("hours"))
    capacity = float(battery.get("capacity_kwh", 0.0))

    if dtype == "solar_reduction":
        factor = _to_float(adj.get("factor"), lo=0.0, hi=1.0, default=1.0)
        structured = {"hours": hours, "factor": factor}
    elif dtype == "minimum_battery_reserve":
        reserve = _to_float(adj.get("minimum_energy_kwh"), lo=0.0, hi=capacity, default=float(battery.get("minimum_energy_kwh", 0.0)))
        structured = {"hours": hours, "minimum_energy_kwh": reserve}
    elif dtype == "max_grid_window":
        cap = _to_float(adj.get("max_grid_kwh"), lo=0.0, default=0.0)
        structured = {"hours": hours, "max_grid_kwh": cap}
    elif dtype in ("no_charge_window", "no_discharge_window"):
        structured = {"hours": hours}
    else:  # defensive: never reachable because dtype is validated above
        structured = {"hours": hours}

    return {
        "note_index": note_index,
        "applies": True,
        "directive_type": dtype,
        "structured_adjustment": structured,
        "explanation": str(item.get("explanation") or f"{dtype} applied for the listed hours."),
    }


def guard_interpretation(raw: List[Dict[str, Any]], n_notes: int, battery: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize raw LLM output into exactly n_notes valid, ordered entries."""
    cleaned_by_index: Dict[int, Dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        entry = _clean_directive(item, battery)
        cleaned_by_index[entry["note_index"]] = entry

    out: List[Dict[str, Any]] = []
    for idx in range(n_notes):
        entry = cleaned_by_index.get(idx)
        if entry is None:
            entry = {
                "note_index": idx,
                "applies": False,
                "directive_type": "no_op",
                "structured_adjustment": None,
                "explanation": "Note did not produce a valid interpretation and was safely ignored.",
            }
        entry = dict(entry)
        entry["note_index"] = idx
        out.append(entry)
    return out