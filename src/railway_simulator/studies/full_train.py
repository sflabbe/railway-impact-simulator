"""Reusable full-train parametric study runner.

This is the package-level version of the train-consist comparison workflow.  It
builds scenarios, delegates dynamics to ``SimulationService`` and optionally
delegates SRS curves to ``SpectrumService``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from railway_simulator.domain.common import utc_now_iso
from railway_simulator.domain.scenario import Scenario
from railway_simulator.domain.spectrum import SRSSettings
from railway_simulator.domain.study import StudyDefinition
from railway_simulator.domain.vehicle import build_traxx_full_freight_proxy, consist_from_engine_config
from railway_simulator.persistence.repositories import StudyRepository
from railway_simulator.services.simulation_service import SimulationService
from railway_simulator.spectrum.service import SpectrumService


@dataclass(frozen=True)
class FullTrainStudySpec:
    """Specification for a locomotive-solo vs full-consist speed/contact sweep."""

    project_id: str
    base_config_id: str
    name: str = "train_consist_comparison"
    vehicles: tuple[str, ...] = ("traxx_br187",)
    modes: tuple[str, ...] = ("lok_solo", "zug_full")
    speeds_kmh: tuple[float, ...] = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0)
    contact_models: tuple[str, ...] = ("anagnostopoulos", "flores", "lankarani-nikravesh")
    mu_values: tuple[float, ...] = (0.30,)
    zeta_srs: tuple[float, ...] = (0.05,)
    Tn_grid_ms: tuple[float, ...] | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_study_definition(self) -> StudyDefinition:
        return StudyDefinition(
            project_id=self.project_id,
            name=self.name,
            study_type="full_train",
            base_config_id=self.base_config_id,
            parameters={
                "vehicles": list(self.vehicles),
                "modes": list(self.modes),
                "speeds_kmh": list(self.speeds_kmh),
                "contact_models": list(self.contact_models),
                "mu_values": list(self.mu_values),
                "zeta_srs": list(self.zeta_srs),
                "Tn_grid_ms": list(self.Tn_grid_ms) if self.Tn_grid_ms is not None else None,
                **self.parameters,
            },
        )


class FullTrainStudyRunner:
    """Build and run full-train scenarios against the existing engine."""

    def __init__(
        self,
        *,
        simulation_service: SimulationService,
        study_repository: StudyRepository | None = None,
        spectrum_service: SpectrumService | None = None,
    ):
        self.simulation_service = simulation_service
        self.study_repository = study_repository
        self.spectrum_service = spectrum_service

    def build_scenarios(
        self,
        *,
        study: StudyDefinition,
        base_config: dict[str, Any],
        spec: FullTrainStudySpec,
    ) -> list[Scenario]:
        scenarios: list[Scenario] = []
        base = dict(base_config)
        base.setdefault("solver", "picard")
        base.setdefault("angle_rad", 0.0)

        for vehicle in spec.vehicles:
            if vehicle != "traxx_br187":
                raise ValueError(f"Unsupported vehicle in package runner: {vehicle}")
            for mode in spec.modes:
                if mode == "lok_solo":
                    consist = consist_from_engine_config("traxx_br187_lok_solo", base)
                elif mode == "zug_full":
                    consist = build_traxx_full_freight_proxy(base)
                else:
                    raise ValueError(f"Unsupported mode: {mode}")

                for contact_model in spec.contact_models:
                    for mu in spec.mu_values:
                        for speed_kmh in spec.speeds_kmh:
                            params = dict(base)
                            params.update(consist.to_engine_overrides())
                            params.update(
                                {
                                    "case_name": f"{vehicle}_{mode}_{contact_model}_{speed_kmh:g}kmh",
                                    "v0_init": -float(speed_kmh) / 3.6,
                                    "contact_model": contact_model,
                                    "mu_s": float(mu),
                                    "mu_k": float(mu),
                                }
                            )
                            scenarios.append(
                                Scenario(
                                    study_id=study.id,
                                    name=f"{vehicle}/{mode}/{contact_model}/v{speed_kmh:g}/mu{mu:g}",
                                    params=params,
                                    meta={
                                        "vehicle": vehicle,
                                        "mode": mode,
                                        "contact_model": contact_model,
                                        "speed_kmh": float(speed_kmh),
                                        "mu": float(mu),
                                        "contact_patch_version": "contact_state_per_mass_v1",
                                        "consist": consist.meta,
                                    },
                                )
                            )
        return scenarios

    def run(
        self,
        *,
        spec: FullTrainStudySpec,
        base_config: dict[str, Any],
        persist_scenarios: bool = True,
        compute_srs: bool = True,
        progress_callback: Callable[[int, int, Scenario, Any], None] | None = None,
    ) -> dict[str, Any]:
        study = spec.to_study_definition()
        if self.study_repository is not None:
            self.study_repository.create_study(study)

        scenarios = self.build_scenarios(study=study, base_config=base_config, spec=spec)
        if persist_scenarios and self.study_repository is not None:
            for scenario in scenarios:
                self.study_repository.add_scenario(scenario)

        runs = []
        curves = []
        total = len(scenarios)
        for index, scenario in enumerate(scenarios, start=1):
            run = self.simulation_service.run_scenario(scenario)
            runs.append(run)
            if progress_callback is not None:
                progress_callback(index, total, scenario, run)
            if compute_srs and run.status == "ok" and self.spectrum_service is not None:
                for zeta in spec.zeta_srs:
                    curves.append(
                        self.spectrum_service.compute_and_store(
                            run,
                            SRSSettings(zeta=float(zeta), Tn_grid_ms=spec.Tn_grid_ms),
                        )
                    )

        if self.study_repository is not None:
            failed = sum(1 for run in runs if run.status != "ok")
            status = "completed_with_failures" if failed else "completed"
            self.study_repository.update_study_status(study.id, status, finished_at=utc_now_iso())

        return {"study": study, "scenarios": scenarios, "runs": runs, "srs_curves": curves}
