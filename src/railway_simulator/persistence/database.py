"""SQLite connection helpers for railway simulator projects."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from railway_simulator.domain.common import utc_now_iso


_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


@dataclass(frozen=True)
class ProjectDatabase:
    """Small SQLite wrapper with explicit schema initialization."""

    path: Path

    def connect(self) -> sqlite3.Connection:
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(path)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        return con

    def initialize(self) -> None:
        schema = _SCHEMA_PATH.read_text(encoding="utf-8")
        with self.connect() as con:
            con.executescript(schema)
            con.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                (1, utc_now_iso()),
            )
            con.commit()


def initialize_project_database(path: str | Path) -> ProjectDatabase:
    """Create or open a project database and apply the current schema."""
    db = ProjectDatabase(Path(path))
    db.initialize()
    return db
