"""Study definition domain object."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from railway_simulator.domain.common import new_id, stable_hash, utc_now_iso


@dataclass(frozen=True)
class StudyDefinition:
    """Parametric or deterministic study definition."""

    project_id: str
    name: str
    study_type: str
    base_config_id: str
    parameters: dict[str, Any] = field(default_factory=dict)
    status: str = "created"
    id: str = field(default_factory=lambda: new_id("std"))
    created_at: str = field(default_factory=utc_now_iso)
    finished_at: str | None = None

    @property
    def parameters_hash(self) -> str:
        return stable_hash(self.parameters)
