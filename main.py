"""GridWise — Smart Campus Energy Optimization Challenge service.

Exposes:
  GET  /health          -> {"status": "ok"}
  POST /optimize-energy -> LLM-assisted 24h battery schedule optimizer

Pipeline: operator notes -> LLM interpretation -> deterministic guardrails
-> LP optimization -> validated JSON response.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from guardrails import guard_interpretation
from interpreter import interpret_notes
from models import OptimizeRequest, OptimizeResponse, DirectiveInterpretation, HourlyPlan
from optimizer import optimize_scenario

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("gridwise")

app = FastAPI(title="GridWise", version="2.0.0")

BASE_DIR = Path(__file__).resolve().parent

_samples: Optional[List[Dict[str, Any]]] = None


def _load_samples() -> List[Dict[str, Any]]:
    global _samples
    if _samples is not None:
        return _samples
    path = BASE_DIR / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        _samples = [
            {"id": c.get("id"), "label": c.get("label"), "input": c.get("input")}
            for c in data.get("cases", [])
        ]
    except Exception:
        _samples = []
    return _samples


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/status")
def status() -> Dict[str, Any]:
    configured = os.environ.get("GRIDWISE_MODELS") or os.environ.get("GRIDWISE_MODEL", "")
    models = [m.strip() for m in configured.split(",") if m.strip()]
    return {
        "llm_mode": os.environ.get("GRIDWISE_LLM_MODE", "openai"),
        "llm_configured": bool(os.environ.get("OPENAI_API_KEY")),
        "provider": "generativelanguage" if "generativelanguage" in (os.environ.get("OPENAI_BASE_URL") or "") else "openai",
        "models": models,
        "samples_available": len(_load_samples()),
    }


@app.get("/api/samples")
def api_samples() -> Dict[str, Any]:
    return {"samples": _load_samples()}


@app.get("/", include_in_schema=False)
def index():
    path = BASE_DIR / "static" / "index.html"
    if path.exists():
        return FileResponse(path)
    return HTMLResponse(
        "<h2>GridWise API is running</h2>"
        "<p>Use <a href='/docs'>/docs</a> or <code>POST /optimize-energy</code>.</p>"
    )


def _summary(directives: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for d in directives:
        if not d["applies"]:
            continue
        parts.append(d["directive_type"])
    if parts:
        brief = ", ".join(sorted(set(parts)))
        return (
            f"Applied operator directives ({brief}) and optimized battery dispatch "
            "to minimize grid cost while satisfying energy balance, battery limits, "
            "and end-of-day neutrality."
        )
    return (
        "No operator directive affected today's schedule; battery dispatch was "
        "optimized to minimize grid cost while satisfying energy balance, battery "
        "limits, and end-of-day neutrality."
    )


def _payload(request: OptimizeRequest) -> Dict[str, Any]:
    battery = request.battery.model_dump()
    notes: List[str] = request.operator_notes

    raw = interpret_notes(notes, battery)
    directives = guard_interpretation(raw, len(notes), battery)

    hours = {h.hour: h for h in request.hours}
    demand = [hours[h].demand_kwh for h in range(24)]
    solar = [hours[h].solar_kwh for h in range(24)]
    tariff = [hours[h].tariff_bdt_per_kwh for h in range(24)]

    result = optimize_scenario(demand, solar, tariff, battery, directives)

    return {
        "scenario_id": request.scenario_id,
        "directive_interpretation": directives,
        "hourly_plan": result["hourly_plan"],
        "total_grid_kwh": result["total_grid_kwh"],
        "total_cost_bdt": result["total_cost_bdt"],
        "peak_grid_kwh": result["peak_grid_kwh"],
        "plan_summary": _summary(directives),
    }


@app.post("/optimize-energy", response_model=OptimizeResponse)
def optimize_energy(request: OptimizeRequest) -> OptimizeResponse:
    payload = _payload(request)
    return OptimizeResponse(
        scenario_id=payload["scenario_id"],
        directive_interpretation=[
            DirectiveInterpretation(**d) for d in payload["directive_interpretation"]
        ],
        hourly_plan=[HourlyPlan(**r) for r in payload["hourly_plan"]],
        total_grid_kwh=payload["total_grid_kwh"],
        total_cost_bdt=payload["total_cost_bdt"],
        peak_grid_kwh=payload["peak_grid_kwh"],
        plan_summary=payload["plan_summary"],
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": "Invalid request payload."})


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})