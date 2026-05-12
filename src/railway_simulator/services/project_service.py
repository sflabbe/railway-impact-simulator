"""Project creation/opening service."""

from __future__ import annotations

from pathlib import Path

from railway_simulator.domain.project import Project
from railway_simulator.persistence.database import ProjectDatabase, initialize_project_database
from railway_simulator.persistence.repositories import ProjectRepository


class ProjectService:
    """Create and open reproducible project folders."""

    def create_project(self, *, name: str, root_dir: str | Path, description: str = "") -> tuple[Project, ProjectDatabase]:
        root = Path(root_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        for child in ["configs", "runs", "studies", "spectra", "artifacts"]:
            (root / child).mkdir(exist_ok=True)
        db = initialize_project_database(root / "project.sqlite")
        project = Project(name=name, root_dir=root, description=description)
        ProjectRepository(db).create(project)
        return project, db

    def open_database(self, path: str | Path) -> ProjectDatabase:
        return initialize_project_database(path)
