"""Scenario domain object."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from railway_simulator.domain.common import new_id, stable_hash


@dataclass(frozen=True)
class Scenario:
    """One executable configuration inside a study."""

    study_id: str
    name: str
    params: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    id: str = field(default_factory=lambda: new_id("scn"))

    @property
    def config_hash(self) -> str:
        return stable_hash(self.params)
