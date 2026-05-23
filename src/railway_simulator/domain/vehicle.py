"""Vehicle and consist domain helpers.

The objects here are intentionally lightweight: the validated engine still owns
numerical time integration, while this module owns reusable vehicle/consist
construction for studies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MassPoint:
    mass_kg: float
    x_m: float
    y_m: float = 0.0


@dataclass(frozen=True)
class CrushLink:
    i: int
    j: int
    fy_N: float
    uy_m: float
    kind: str = "internal"

    @property
    def k_N_per_m(self) -> float:
        if self.uy_m == 0.0:
            return 0.0
        return self.fy_N / self.uy_m


@dataclass(frozen=True)
class Coupler(CrushLink):
    kind: str = "coupler"
    gap_m: float = 0.0


@dataclass(frozen=True)
class VehicleConsist:
    name: str
    mass_points: list[MassPoint]
    links: list[CrushLink]
    couplers: list[Coupler] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def total_mass_kg(self) -> float:
        return float(sum(mp.mass_kg for mp in self.mass_points))

    def to_engine_overrides(self) -> dict[str, Any]:
        """Return engine config overrides for n_masses/masses/x/y/fy/uy/k_train."""
        all_links: list[CrushLink] = [*self.links, *self.couplers]
        fy = [float(link.fy_N) for link in all_links]
        uy = [float(link.uy_m) for link in all_links]
        return {
            "n_masses": len(self.mass_points),
            "masses": [float(mp.mass_kg) for mp in self.mass_points],
            "x_init": [float(mp.x_m) for mp in self.mass_points],
            "y_init": [float(mp.y_m) for mp in self.mass_points],
            "fy": fy,
            "uy": uy,
            "k_train": [float(f) / float(u) if float(u) else 0.0 for f, u in zip(fy, uy)],
        }


def consist_from_engine_config(name: str, cfg: dict[str, Any], *, meta: dict[str, Any] | None = None) -> VehicleConsist:
    """Build a VehicleConsist from an existing engine config dict."""
    masses = [float(v) for v in cfg.get("masses", [])]
    x = [float(v) for v in cfg.get("x_init", [0.0] * len(masses))]
    y = [float(v) for v in cfg.get("y_init", [0.0] * len(masses))]
    fy = [float(v) for v in cfg.get("fy", [])]
    uy = [float(v) for v in cfg.get("uy", [])]
    mass_points = [MassPoint(m, xi, yi) for m, xi, yi in zip(masses, x, y)]
    links = [CrushLink(i=i, j=i + 1, fy_N=f, uy_m=u) for i, (f, u) in enumerate(zip(fy, uy))]
    return VehicleConsist(name=name, mass_points=mass_points, links=links, meta=meta or {})


def build_traxx_full_freight_proxy(base_cfg: dict[str, Any], *, wagon_count: int = 4) -> VehicleConsist:
    """Build the TRAXX + freight wagons proxy used by consist comparison studies.

    Assumptions are deliberately encoded in metadata so downstream reports can
    flag this as a reduced parametric model, not calibrated crash test data.
    """
    solo = consist_from_engine_config(
        "traxx_br187_lok_solo",
        base_cfg,
        meta={"source": "configs/traxx_freight.yml"},
    )
    mass_points = list(solo.mass_points)
    links: list[CrushLink] = list(solo.links)
    couplers: list[Coupler] = []

    gap_between_units_m = 0.5
    wagon_length_m = 18.0
    wagon_internal_fy_N = 20.0e6
    wagon_internal_uy_m = 0.05
    coupling_fy_N = 2.5e6
    coupling_uy_m = 0.05

    last_x = mass_points[-1].x_m if mass_points else 0.0
    last_idx = len(mass_points) - 1
    for wagon_no in range(wagon_count):
        start = last_x + gap_between_units_m
        new_points = [
            MassPoint(20_000.0, start, 0.0),
            MassPoint(40_000.0, start + 0.5 * wagon_length_m, 0.0),
            MassPoint(20_000.0, start + wagon_length_m, 0.0),
        ]
        first_new_idx = len(mass_points)
        mass_points.extend(new_points)
        couplers.append(
            Coupler(
                i=last_idx,
                j=first_new_idx,
                fy_N=coupling_fy_N,
                uy_m=coupling_uy_m,
                gap_m=gap_between_units_m,
            )
        )
        links.extend(
            [
                CrushLink(first_new_idx, first_new_idx + 1, wagon_internal_fy_N, wagon_internal_uy_m),
                CrushLink(first_new_idx + 1, first_new_idx + 2, wagon_internal_fy_N, wagon_internal_uy_m),
            ]
        )
        last_idx = first_new_idx + 2
        last_x = start + wagon_length_m

    return VehicleConsist(
        name="traxx_br187_full_gueterzug_proxy",
        mass_points=mass_points,
        links=links,
        couplers=couplers,
        meta={
            "scope": "reduced multibody proxy / parametric full consist model",
            "wagon_count": int(wagon_count),
            "wagon_mass_t": 80.0,
            "wagon_length_assumed_m": wagon_length_m,
            "gap_between_units_m": gap_between_units_m,
            "coupling_fy_N": coupling_fy_N,
            "coupling_uy_m": coupling_uy_m,
            "not_calibrated_to_crash_tests": True,
        },
    )
