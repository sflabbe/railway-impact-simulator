"""Repository layer for project/study/run/SRS persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from railway_simulator.domain.common import new_id, json_dumps_stable, stable_hash, utc_now_iso
from railway_simulator.domain.project import Project
from railway_simulator.domain.result import RunMetric, SimulationRun
from railway_simulator.domain.scenario import Scenario
from railway_simulator.domain.spectrum import SRSCurve
from railway_simulator.domain.study import StudyDefinition
from railway_simulator.persistence.database import ProjectDatabase


@dataclass(frozen=True)
class ConfigSnapshot:
    id: str
    project_id: str
    config_hash: str
    config: dict[str, Any]
    source_path: Path | None
    created_at: str


class ProjectRepository:
    def __init__(self, db: ProjectDatabase):
        self.db = db

    def create(self, project: Project) -> None:
        with self.db.connect() as con:
            con.execute(
                """
                INSERT INTO projects(id, name, description, root_dir, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    project.id,
                    project.name,
                    project.description,
                    str(project.normalized_root()),
                    project.created_at,
                ),
            )
            con.commit()

    def get(self, project_id: str) -> Project:
        with self.db.connect() as con:
            row = con.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        if row is None:
            raise KeyError(f"project not found: {project_id}")
        return Project(
            id=row["id"],
            name=row["name"],
            description=row["description"] or "",
            root_dir=Path(row["root_dir"]),
            created_at=row["created_at"],
        )

    def list(self) -> list[Project]:
        with self.db.connect() as con:
            rows = con.execute("SELECT * FROM projects ORDER BY created_at, name").fetchall()
        return [
            Project(
                id=row["id"],
                name=row["name"],
                description=row["description"] or "",
                root_dir=Path(row["root_dir"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]


class ConfigSnapshotRepository:
    def __init__(self, db: ProjectDatabase):
        self.db = db

    def create(self, project_id: str, config: dict[str, Any], source_path: str | Path | None = None) -> ConfigSnapshot:
        config_hash = stable_hash(config)
        snapshot = ConfigSnapshot(
            id=new_id("cfg"),
            project_id=project_id,
            config_hash=config_hash,
            config=dict(config),
            source_path=Path(source_path) if source_path is not None else None,
            created_at=utc_now_iso(),
        )
        with self.db.connect() as con:
            con.execute(
                """
                INSERT INTO config_snapshots(id, project_id, config_hash, config_json, source_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.id,
                    snapshot.project_id,
                    snapshot.config_hash,
                    json_dumps_stable(snapshot.config),
                    str(snapshot.source_path) if snapshot.source_path is not None else None,
                    snapshot.created_at,
                ),
            )
            con.commit()
        return snapshot

    def get(self, config_id: str) -> ConfigSnapshot:
        with self.db.connect() as con:
            row = con.execute("SELECT * FROM config_snapshots WHERE id = ?", (config_id,)).fetchone()
        if row is None:
            raise KeyError(f"config snapshot not found: {config_id}")
        return ConfigSnapshot(
            id=row["id"],
            project_id=row["project_id"],
            config_hash=row["config_hash"],
            config=json.loads(row["config_json"]),
            source_path=Path(row["source_path"]) if row["source_path"] else None,
            created_at=row["created_at"],
        )


class StudyRepository:
    def __init__(self, db: ProjectDatabase):
        self.db = db

    def create_study(self, study: StudyDefinition) -> None:
        with self.db.connect() as con:
            con.execute(
                """
                INSERT INTO studies(id, project_id, name, study_type, base_config_id,
                                    parameters_json, status, created_at, finished_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    study.id,
                    study.project_id,
                    study.name,
                    study.study_type,
                    study.base_config_id,
                    json_dumps_stable(study.parameters),
                    study.status,
                    study.created_at,
                    study.finished_at,
                ),
            )
            con.commit()

    def update_study_status(self, study_id: str, status: str, *, finished_at: str | None = None) -> None:
        with self.db.connect() as con:
            con.execute(
                "UPDATE studies SET status = ?, finished_at = COALESCE(?, finished_at) WHERE id = ?",
                (status, finished_at, study_id),
            )
            con.commit()

    def list_studies(self, project_id: str) -> list[StudyDefinition]:
        with self.db.connect() as con:
            rows = con.execute(
                "SELECT * FROM studies WHERE project_id = ? ORDER BY created_at",
                (project_id,),
            ).fetchall()
        return [
            StudyDefinition(
                id=row["id"],
                project_id=row["project_id"],
                name=row["name"],
                study_type=row["study_type"],
                base_config_id=row["base_config_id"],
                parameters=json.loads(row["parameters_json"]),
                status=row["status"],
                created_at=row["created_at"],
                finished_at=row["finished_at"],
            )
            for row in rows
        ]



    def get_study(self, study_id: str) -> StudyDefinition:
        """Return one study definition by id."""
        with self.db.connect() as con:
            row = con.execute("SELECT * FROM studies WHERE id = ?", (study_id,)).fetchone()
        if row is None:
            raise KeyError(f"study not found: {study_id}")
        return StudyDefinition(
            id=row["id"],
            project_id=row["project_id"],
            name=row["name"],
            study_type=row["study_type"],
            base_config_id=row["base_config_id"],
            parameters=json.loads(row["parameters_json"]),
            status=row["status"],
            created_at=row["created_at"],
            finished_at=row["finished_at"],
        )

    def add_scenario(self, scenario: Scenario) -> None:
        with self.db.connect() as con:
            con.execute(
                """
                INSERT INTO scenarios(id, study_id, name, weight, params_json, meta_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    scenario.id,
                    scenario.study_id,
                    scenario.name,
                    float(scenario.weight),
                    json_dumps_stable(scenario.params),
                    json_dumps_stable(scenario.meta),
                ),
            )
            con.commit()

    def list_scenarios(self, study_id: str) -> list[Scenario]:
        with self.db.connect() as con:
            rows = con.execute(
                "SELECT * FROM scenarios WHERE study_id = ? ORDER BY name",
                (study_id,),
            ).fetchall()
        return [
            Scenario(
                id=row["id"],
                study_id=row["study_id"],
                name=row["name"],
                weight=float(row["weight"]),
                params=json.loads(row["params_json"]),
                meta=json.loads(row["meta_json"]),
            )
            for row in rows
        ]

    def add_run(self, run: SimulationRun) -> None:
        with self.db.connect() as con:
            con.execute(
                """
                INSERT INTO runs(id, scenario_id, status, config_hash, result_csv_path,
                                 elapsed_s, error_message, started_at, finished_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.id,
                    run.scenario_id,
                    run.status,
                    run.config_hash,
                    str(run.result_csv_path) if run.result_csv_path is not None else None,
                    run.elapsed_s,
                    run.error_message,
                    run.started_at,
                    run.finished_at,
                ),
            )
            con.commit()

    def list_runs_for_study(self, study_id: str) -> list[SimulationRun]:
        with self.db.connect() as con:
            rows = con.execute(
                """
                SELECT r.* FROM runs r
                JOIN scenarios s ON s.id = r.scenario_id
                WHERE s.study_id = ?
                ORDER BY r.started_at
                """,
                (study_id,),
            ).fetchall()
        return [
            SimulationRun(
                id=row["id"],
                scenario_id=row["scenario_id"],
                status=row["status"],
                config_hash=row["config_hash"],
                result_csv_path=Path(row["result_csv_path"]) if row["result_csv_path"] else None,
                elapsed_s=row["elapsed_s"],
                error_message=row["error_message"],
                started_at=row["started_at"],
                finished_at=row["finished_at"],
            )
            for row in rows
        ]



    def list_run_records_for_study(self, study_id: str) -> list[dict[str, Any]]:
        """Return run rows joined with scenario names and metadata.

        This is intentionally a dict-based read model for UI/reporting code.  The
        typed domain methods above remain the canonical write path.
        """
        with self.db.connect() as con:
            rows = con.execute(
                """
                SELECT
                    r.id AS run_id,
                    r.scenario_id AS scenario_id,
                    r.status AS run_status,
                    r.config_hash AS config_hash,
                    r.result_csv_path AS result_csv_path,
                    r.elapsed_s AS elapsed_s,
                    r.error_message AS error_message,
                    r.started_at AS started_at,
                    r.finished_at AS finished_at,
                    s.name AS scenario_name,
                    s.weight AS scenario_weight,
                    s.meta_json AS scenario_meta_json
                FROM runs r
                JOIN scenarios s ON s.id = r.scenario_id
                WHERE s.study_id = ?
                ORDER BY r.started_at, s.name
                """,
                (study_id,),
            ).fetchall()
        records: list[dict[str, Any]] = []
        for row in rows:
            meta = json.loads(row["scenario_meta_json"] or "{}")
            records.append(
                {
                    "run_id": row["run_id"],
                    "scenario_id": row["scenario_id"],
                    "scenario_name": row["scenario_name"],
                    "scenario_weight": float(row["scenario_weight"]),
                    "run_status": row["run_status"],
                    "config_hash": row["config_hash"],
                    "result_csv_path": row["result_csv_path"],
                    "elapsed_s": row["elapsed_s"],
                    "error_message": row["error_message"],
                    "started_at": row["started_at"],
                    "finished_at": row["finished_at"],
                    **{f"meta_{k}": v for k, v in meta.items() if not isinstance(v, (dict, list))},
                }
            )
        return records

    def add_metric(self, metric: RunMetric) -> None:
        with self.db.connect() as con:
            con.execute(
                "INSERT INTO run_metrics(id, run_id, name, value, unit) VALUES (?, ?, ?, ?, ?)",
                (metric.id, metric.run_id, metric.name, float(metric.value), metric.unit),
            )
            con.commit()

    def list_metrics(self, run_id: str) -> list[RunMetric]:
        with self.db.connect() as con:
            rows = con.execute(
                "SELECT * FROM run_metrics WHERE run_id = ? ORDER BY name",
                (run_id,),
            ).fetchall()
        return [
            RunMetric(
                id=row["id"],
                run_id=row["run_id"],
                name=row["name"],
                value=float(row["value"]),
                unit=row["unit"] or "",
            )
            for row in rows
        ]


class SpectrumRepository:
    def __init__(self, db: ProjectDatabase):
        self.db = db



    def list_curve_records_for_study(self, study_id: str) -> list[dict[str, Any]]:
        """Return SRS rows joined with run/scenario metadata for UI comparison."""
        with self.db.connect() as con:
            rows = con.execute(
                """
                SELECT
                    c.id AS curve_id,
                    c.run_id AS run_id,
                    c.zeta AS zeta,
                    c.oscillator_mass AS oscillator_mass,
                    c.force_column AS force_column,
                    c.curve_csv_path AS curve_csv_path,
                    c.created_at AS curve_created_at,
                    s.id AS scenario_id,
                    s.name AS scenario_name,
                    s.meta_json AS scenario_meta_json,
                    r.status AS run_status
                FROM srs_curves c
                JOIN runs r ON r.id = c.run_id
                JOIN scenarios s ON s.id = r.scenario_id
                WHERE s.study_id = ?
                ORDER BY c.zeta, s.name
                """,
                (study_id,),
            ).fetchall()
        records: list[dict[str, Any]] = []
        for row in rows:
            meta = json.loads(row["scenario_meta_json"] or "{}")
            records.append(
                {
                    "curve_id": row["curve_id"],
                    "run_id": row["run_id"],
                    "zeta": float(row["zeta"]),
                    "oscillator_mass": float(row["oscillator_mass"]),
                    "force_column": row["force_column"],
                    "curve_csv_path": row["curve_csv_path"],
                    "curve_created_at": row["curve_created_at"],
                    "scenario_id": row["scenario_id"],
                    "scenario_name": row["scenario_name"],
                    "run_status": row["run_status"],
                    **{f"meta_{k}": v for k, v in meta.items() if not isinstance(v, (dict, list))},
                }
            )
        return records

    def add_curve(self, curve: SRSCurve) -> None:
        with self.db.connect() as con:
            con.execute(
                """
                INSERT INTO srs_curves(id, run_id, zeta, oscillator_mass, force_column,
                                       curve_csv_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    curve.id,
                    curve.run_id,
                    float(curve.zeta),
                    float(curve.oscillator_mass),
                    curve.force_column,
                    str(curve.curve_csv_path),
                    curve.created_at,
                ),
            )
            con.commit()

    def list_curves_for_study(self, study_id: str) -> list[SRSCurve]:
        with self.db.connect() as con:
            rows = con.execute(
                """
                SELECT c.* FROM srs_curves c
                JOIN runs r ON r.id = c.run_id
                JOIN scenarios s ON s.id = r.scenario_id
                WHERE s.study_id = ?
                ORDER BY c.created_at
                """,
                (study_id,),
            ).fetchall()
        return [
            SRSCurve(
                id=row["id"],
                run_id=row["run_id"],
                zeta=float(row["zeta"]),
                oscillator_mass=float(row["oscillator_mass"]),
                force_column=row["force_column"],
                curve_csv_path=Path(row["curve_csv_path"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]
