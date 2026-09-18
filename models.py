from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class HourInput(BaseModel):
    hour: int = Field(ge=0, le=23)
    demand_kwh: float = Field(ge=0.0)
    solar_kwh: float = Field(ge=0.0)
    tariff_bdt_per_kwh: float = Field(ge=0.0)


class BatteryInput(BaseModel):
    capacity_kwh: float = Field(gt=0.0)
    initial_energy_kwh: float = Field(ge=0.0)
    minimum_energy_kwh: float = Field(ge=0.0)
    max_charge_kwh_per_hour: float = Field(ge=0.0)
    max_discharge_kwh_per_hour: float = Field(ge=0.0)


class OptimizeRequest(BaseModel):
    scenario_id: str
    operator_notes: List[str] = Field(min_length=1, max_length=3)
    hours: List[HourInput]
    battery: BatteryInput

    @field_validator("operator_notes")
    @classmethod
    def notes_non_empty(cls, v: List[str]) -> List[str]:
        if any(not isinstance(s, str) or not s.strip() for s in v):
            raise ValueError("operator_notes must contain non-empty strings")
        return [s.strip() for s in v]

    @field_validator("hours")
    @classmethod
    def hours_are_24_unique(cls, v: List[HourInput]) -> List[HourInput]:
        hours = [h.hour for h in v]
        if len(hours) != 24:
            raise ValueError("hours must contain exactly 24 entries")
        if sorted(hours) != list(range(24)):
            raise ValueError("hours must contain each hour 0..23 exactly once")
        return sorted(v, key=lambda h: h.hour)

    @model_validator(mode="after")
    def battery_sane(self) -> "OptimizeRequest":
        b = self.battery
        if b.minimum_energy_kwh > b.capacity_kwh + 1e-9:
            raise ValueError("minimum_energy_kwh cannot exceed capacity_kwh")
        if b.initial_energy_kwh > b.capacity_kwh + 1e-9:
            raise ValueError("initial_energy_kwh cannot exceed capacity_kwh")
        return self


DirectiveType = Literal[
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
]

BatteryAction = Literal["charge", "discharge", "idle"]


class DirectiveInterpretation(BaseModel):
    note_index: int = Field(ge=0)
    applies: bool
    directive_type: DirectiveType
    structured_adjustment: Optional[Dict[str, Any]] = None
    explanation: str


class HourlyPlan(BaseModel):
    hour: int = Field(ge=0, le=23)
    grid_kwh: float = Field(ge=0.0)
    solar_used_kwh: float = Field(ge=0.0)
    battery_action: BatteryAction
    battery_kwh: float = Field(ge=0.0)
    battery_energy_after_kwh: float = Field(ge=0.0)


class OptimizeResponse(BaseModel):
    scenario_id: str
    directive_interpretation: List[DirectiveInterpretation]
    hourly_plan: List[HourlyPlan]
    total_grid_kwh: float = Field(ge=0.0)
    total_cost_bdt: float = Field(ge=0.0)
    peak_grid_kwh: float = Field(ge=0.0)
    plan_summary: str