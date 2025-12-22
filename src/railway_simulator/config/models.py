from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


class ConfigBase(BaseModel):
    model_config = {"extra": "forbid"}


class LinearLaw(ConfigBase):
    type: Literal["linear"]
    k_N_per_m: float
    gap_m: float = 0.0

    @field_validator("k_N_per_m")
    @classmethod
    def _k_positive(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError("k_N_per_m must be > 0")
        return value

    @field_validator("gap_m")
    @classmethod
    def _gap_nonneg(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError("gap_m must be >= 0")
        return value


class PiecewiseLinearLaw(ConfigBase):
    type: Literal["piecewise_linear"]
    points: List[Tuple[float, float]]
    gap_m: float = 0.0

    @field_validator("points")
    @classmethod
    def _validate_points(cls, value: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if len(value) < 2:
            raise ValueError("points must contain at least two [disp_m, force_N] pairs")
        last_disp = None
        for disp, force in value:
            if disp < 0.0:
                raise ValueError("displacement values must be >= 0")
            if last_disp is not None and disp <= last_disp:
                raise ValueError("displacement values must be strictly increasing")
            last_disp = disp
            if force < 0.0:
                raise ValueError("force values must be >= 0")
        return value

    @field_validator("gap_m")
    @classmethod
    def _gap_nonneg(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError("gap_m must be >= 0")
        return value


class PlateauLaw(ConfigBase):
    type: Literal["plateau"]
    ramp_to_force_N: float
    ramp_disp_m: float
    plateau_force_N: float
    plateau_disp_m: float
    final_force_N: float
    final_disp_m: float
    gap_m: float = 0.0

    @model_validator(mode="after")
    def _validate_segments(self) -> "PlateauLaw":
        if self.ramp_to_force_N <= 0.0:
            raise ValueError("ramp_to_force_N must be > 0")
        if self.ramp_disp_m <= 0.0:
            raise ValueError("ramp_disp_m must be > 0")
        if self.plateau_force_N <= 0.0:
            raise ValueError("plateau_force_N must be > 0")
        if self.plateau_disp_m <= self.ramp_disp_m:
            raise ValueError("plateau_disp_m must be greater than ramp_disp_m")
        if self.final_force_N <= 0.0:
            raise ValueError("final_force_N must be > 0")
        if self.final_disp_m <= self.plateau_disp_m:
            raise ValueError("final_disp_m must be greater than plateau_disp_m")
        if self.gap_m < 0.0:
            raise ValueError("gap_m must be >= 0")
        return self


class TabulatedCurveLaw(ConfigBase):
    type: Literal["tabulated_curve"]
    csv_path: str
    gap_m: float = 0.0

    @field_validator("csv_path")
    @classmethod
    def _path_required(cls, value: str) -> str:
        if not value:
            raise ValueError("csv_path is required")
        return value

    @field_validator("gap_m")
    @classmethod
    def _gap_nonneg(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError("gap_m must be >= 0")
        return value


InterfaceLawSpec = Union[LinearLaw, PiecewiseLinearLaw, PlateauLaw, TabulatedCurveLaw]


class InterfacePresetRef(ConfigBase):
    type: Literal["en15227_preset"]
    preset_id: str

    @field_validator("preset_id")
    @classmethod
    def _preset_required(cls, value: str) -> str:
        if not value:
            raise ValueError("preset_id is required")
        return value


class RigidWallPartner(ConfigBase):
    type: Literal["rigid_wall"] = "rigid_wall"


class ReferenceMassPartner(ConfigBase):
    type: Literal["reference_mass_1d"]
    mass_kg: float
    dofs: List[str] = Field(default_factory=lambda: ["x"])

    @field_validator("mass_kg")
    @classmethod
    def _mass_positive(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError("mass_kg must be > 0")
        return value

    @field_validator("dofs")
    @classmethod
    def _dofs_allowed(cls, value: List[str]) -> List[str]:
        allowed = {"x"}
        if not value:
            raise ValueError("dofs must not be empty")
        invalid = [d for d in value if d not in allowed]
        if invalid:
            raise ValueError(f"dofs contains unsupported entries: {invalid}")
        return value


class EnPresetPartner(ConfigBase):
    type: Literal["en15227_preset"]
    preset_id: str

    @field_validator("preset_id")
    @classmethod
    def _preset_required(cls, value: str) -> str:
        if not value:
            raise ValueError("preset_id is required")
        return value


PartnerSpec = Union[RigidWallPartner, ReferenceMassPartner, EnPresetPartner]


class CollisionSpec(ConfigBase):
    partner: PartnerSpec = Field(default_factory=RigidWallPartner)
    interface: Optional[Union[InterfaceLawSpec, InterfacePresetRef]] = None
    scenario: Optional[str] = None


class SimulationConfig(BaseModel):
    model_config = {"extra": "allow"}
    units: str = "SI"
    collision: Optional[CollisionSpec] = None
    legacy: Optional[Dict[str, Any]] = None

    @field_validator("units")
    @classmethod
    def _units_si(cls, value: str) -> str:
        if value != "SI":
            raise ValueError("Only SI units are supported currently")
        return value


def format_validation_error(exc: ValidationError, *, filename: str) -> str:
    parts = []
    for error in exc.errors():
        loc = ".".join(str(item) for item in error.get("loc", []))
        msg = error.get("msg", "invalid value")
        parts.append(f"{loc}: {msg}")
    details = "; ".join(parts) if parts else str(exc)
    return f"{filename}: invalid configuration: {details}"
