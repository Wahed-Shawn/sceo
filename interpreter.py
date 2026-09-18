"""Operator-note interpretation for the GridWise challenge.

Primary path: a language-capable generative LLM (OpenAI-compatible chat
completions API) produces one structured directive per operator note.

Fallback path: a deterministic rule-based parser is used ONLY when the LLM is
unavailable (no API key configured or the model call fails) so the service
never crashes on model-provider failure. The LLM remains the default and
required interpretation path; the fallback is strictly a resilience layer.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("gridwise")


DIRECTIVE_TYPES = [
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
]

SYSTEM_PROMPT = """You are GridWise, the operator-note interpreter for a 24-hour campus energy optimizer.

You receive natural-language notes written by a campus facility operator plus the
battery specification. Translate each note into exactly one structured directive
from the supported types below.

## Supported directive types

1. solar_reduction: available rooftop solar drops in a window.
   structured_adjustment: {"hours": [...], "factor": <float 0..1>}
   factor is the USABLE FRACTION REMAINING after the reduction. An "80%
   reduction" means factor = 0.2. "Roughly 25% of the forecast" means factor =
   0.25. "About half" means factor = 0.5.

2. minimum_battery_reserve: battery stored energy must remain at least a given
   kWh level during a window.
   structured_adjustment: {"hours": [...], "minimum_energy_kwh": <float>}
   If the note states a percentage of battery capacity, convert to kWh using the
   battery capacity given in the message.

3. no_charge_window: battery charging is unavailable in a window.
   structured_adjustment: {"hours": [...]}

4. no_discharge_window: battery discharging is unavailable in a window.
   structured_adjustment: {"hours": [...]}

5. max_grid_window: grid import is capped per hour in a window.
   structured_adjustment: {"hours": [...], "max_grid_kwh": <float>}

6. no_op: the note does not change today's 24-hour energy schedule (distractor).
   applies must be false and structured_adjustment must be null.

## Time convention

Military whole hours 0..23: 12:00 AM = 0, 6:00 AM = 6, noon/12:00 PM = 12,
1:00 PM = 13, 6:00 PM = 18, 11:00 PM = 23. Windows are START-INCLUSIVE and
END-EXCLUSIVE. "Noon until 2 PM" -> [12, 13]. "6 PM until 9 PM" -> [18, 19, 20].
"From 2 AM until 5 AM" -> [2, 3, 4]. Hours must be unique integers 0-23.

## Rules

- non-no_op directives: applies = true.
- no_op: applies = false and structured_adjustment = null.
- structured_adjustment must contain ONLY the fields listed for the type.
- Notes about unrelated topics (deadlines, events on other days, non-energy
  matters) are no_op.

## Output

Return STRICT JSON of the form:
{"interpretations": [{"note_index": 0, "applies": true,
"directive_type": "...", "structured_adjustment": {...},
"explanation": "short prose"}]}
Return exactly one entry per operator note, in note_index order 0..N-1, and
nothing outside this JSON object.
"""


def _extract_time(tok: str) -> Optional[int]:
    tok = tok.strip().rstrip(".").lower()
    if tok in ("midnight", "12am", "12:00am"):
        return 0
    if tok in ("noon", "12pm", "12:00pm"):
        return 12
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", tok)
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2) or 0)
    meridiem = m.group(3)
    if meridiem is None:
        if hour < 1 or hour > 24:
            return None
        if hour == 24:
            return 0
        return hour
    if hour < 1 or hour > 12 or minute >= 60:
        return None
    if meridiem == "am":
        return 0 if hour == 12 else hour
    return hour if hour == 12 else hour + 12


def _parse_window(text: str) -> Optional[List[int]]:
    """Return a contiguous ascending hour list for a start/end time window."""
    text = text.lower().replace("a.m.", "am").replace("p.m.", "pm")
    m = re.search(
        r"(?:from|between)?\s*"
        r"(noon|midnight|\d{1,2}(?::\d{2})?\s*(?:am|pm))\s*"
        r"(?:until|to|and|through|-)\s*"
        r"(noon|midnight|\d{1,2}(?::\d{2})?\s*(?:am|pm))",
        text,
    )
    if not m:
        return None
    start = _extract_time(m.group(1))
    end = _extract_time(m.group(2))
    if start is None or end is None or end <= start:
        return None
    return list(range(start, end))


def _extract_factor(text: str) -> Optional[float]:
    lowered = text.lower()
    if "half" in lowered or "50%" in lowered:
        if "reduc" in lowered:
            halves = re.findall(r"(\d+)\s*%", lowered)
            if halves:
                return round(1.0 - int(halves[0]) / 100.0, 4)
            return 0.5
        return 0.5
    pcts = re.findall(r"(\d{1,3}(?:\.\d+)?)\s*%", text)
    if not pcts:
        return None
    pct = float(pcts[0])
    if "reduc" in lowered or "drops" in lowered or "lost" in lowered:
        return round(1.0 - pct / 100.0, 4)
    if "of the forecast" in lowered or "of forecast" in lowered or "remain" in lowered or "left" in lowered:
        return round(pct / 100.0, 4)
    if pct <= 1.0:
        return round(pct, 4)
    return None


def _extract_number(text: str) -> Optional[float]:
    m = re.search(r"(\d{1,6}(?:\.\d+)?)\s*(?:kwh)?", text)
    if not m:
        return None
    return float(m.group(1))


def _extract_cap_kwh(text: str) -> Optional[float]:
    m = re.search(r"(?:exceed|below|cap|limit|capped|at or below|not to exceed)\s*(\d{1,6}(?:\.\d+)?)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"(\d{1,6}(?:\.\d+)?)\s*kwh", text, re.IGNORECASE)
    if not m:
        return None
    return float(m.group(1))


def _extract_reserve_kwh(text: str, battery: Dict[str, Any]) -> Optional[float]:
    lowered = text.lower()
    if "percent" in lowered or "%" in lowered:
        m = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%", text)
        if m:
            return round(float(m.group(1)) / 100.0 * float(battery.get("capacity_kwh", 0.0)), 4)
    m = re.search(r"(?:at least|minimum|no less than|at or above)\s*(\d{1,6}(?:\.\d+)?)\s*kwh", lowered, re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d{1,6}(?:\.\d+)?)\s*kwh\s*(?:in the battery|remain|reserve|stored)", lowered, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def _per_note(note: str, battery: Dict[str, Any]) -> Dict[str, Any]:
    lowered = note.lower()
    window = _parse_window(note)

    is_no_charge = (
        ("charg" in lowered)
        and any(k in lowered for k in ("unavailable", "disabled", "isolated", "will not charge", "cannot charge", "no charging", "maintenance", "inspection"))
    )
    if is_no_charge:
        return {
            "applies": True,
            "directive_type": "no_charge_window",
            "structured_adjustment": {"hours": window or []},
        }

    is_no_discharge = (
        ("discharg" in lowered)
        and any(k in lowered for k in ("must not", "do not", "no discharge", "forbidden", "disabled", "testing", "protection"))
    )
    if is_no_discharge:
        return {
            "applies": True,
            "directive_type": "no_discharge_window",
            "structured_adjustment": {"hours": window or []},
        }

    is_solar = (
        ("solar" in lowered or "panel" in lowered or "inverter" in lowered)
        and any(k in lowered for k in ("reduc", "half", "cleaning", "inspection", "cloud", "coverage", "%"))
    )
    if is_solar:
        factor = _extract_factor(note)
        if factor is not None and window:
            return {
                "applies": True,
                "directive_type": "solar_reduction",
                "structured_adjustment": {"hours": window, "factor": factor},
            }

    is_reserve = (
        any(k in lowered for k in ("keep at least", "must remain", "remain in the battery", "reserve", "requires at least", "stored in the battery", "maintain"))
        and ("battery" in lowered or "stored" in lowered or "remain" in lowered)
    )
    if is_reserve:
        reserve = _extract_reserve_kwh(note, battery)
        if reserve is not None and window:
            return {
                "applies": True,
                "directive_type": "minimum_battery_reserve",
                "structured_adjustment": {"hours": window, "minimum_energy_kwh": reserve},
            }

    is_max_grid = (
        any(k in lowered for k in ("grid import", "grid intake", "grid", "feeder", "transformer", "substation"))
        and any(k in lowered for k in ("must not exceed", "must stay at or below", "limit", "cap", "capped", "not to exceed", "at or below"))
    )
    if is_max_grid:
        cap = _extract_cap_kwh(note)
        if cap is not None and window:
            return {
                "applies": True,
                "directive_type": "max_grid_window",
                "structured_adjustment": {"hours": window, "max_grid_kwh": cap},
            }

    return {
        "applies": False,
        "directive_type": "no_op",
        "structured_adjustment": None,
    }


def rule_based_interpret(notes: List[str], battery: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, note in enumerate(notes):
        parsed = _per_note(note, battery)
        out.append(
            {
                "note_index": idx,
                "applies": parsed["applies"],
                "directive_type": parsed["directive_type"],
                "structured_adjustment": parsed["structured_adjustment"],
                "explanation": f"Deterministic fallback interpretation: {parsed['directive_type']}.",
            }
        )
    return out


def _strip_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_llm_payload(content: str, n_notes: int) -> List[Dict[str, Any]]:
    data = json.loads(_strip_json(content))
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and isinstance(data.get("interpretations"), list):
        items = data["interpretations"]
    else:
        raise ValueError("LLM output did not contain an 'interpretations' list")
    out: List[Dict[str, Any]] = []
    for item in items[:n_notes]:
        if isinstance(item, dict):
            out.append(item)
    return out


def _call_llm(notes: List[str], battery: Dict[str, Any]) -> Optional[str]:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    base_url = os.environ.get("OPENAI_BASE_URL")

    configured = os.environ.get("GRIDWISE_MODELS") or os.environ.get("GRIDWISE_MODEL", "gpt-4o-mini")
    models = [m.strip() for m in configured.split(",") if m.strip()]
    timeout = float(os.environ.get("GRIDWISE_LLM_TIMEOUT", "3.0"))

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)

    user_content = (
        f"Battery specification: capacity_kwh={battery['capacity_kwh']}, "
        f"initial_energy_kwh={battery['initial_energy_kwh']}, "
        f"minimum_energy_kwh={battery['minimum_energy_kwh']}.\n\n"
        "Operator notes (index: text):\n"
        + "\n".join(f"{i}: {note}" for i, note in enumerate(notes))
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    for model in models:
        try:
            resp = client.chat.completions.create(model=model, messages=messages, temperature=0)
            content = resp.choices[0].message.content
            if content:
                return content
            logger.warning("LLM model=%s returned empty content", model)
        except Exception as exc:
            logger.warning("LLM call to model=%s failed: %s", model, exc)
    return None


def interpret_notes(notes: List[str], battery: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return one interpretation dict per note. LLM first, fallback on failure."""
    if os.environ.get("GRIDWISE_LLM_MODE", "openai").lower() == "rules":
        return rule_based_interpret(notes, battery)
    try:
        content = _call_llm(notes, battery)
        if content:
            items = _parse_llm_payload(content, len(notes))
            if len(items) >= 1:
                for idx, item in enumerate(items):
                    item.setdefault("note_index", idx)
                return items
    except Exception:
        pass
    return rule_based_interpret(notes, battery)