"""Project domain object."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from railway_simulator.domain.common import new_id, utc_now_iso


@dataclass(frozen=True)
class Project:
    """A reproducible workspace containing configs, studies and artifacts."""

    name: str
    root_dir: Path
    description: str = ""
    id: str = field(default_factory=lambda: new_id("prj"))
    created_at: str = field(default_factory=utc_now_iso)

    def normalized_root(self) -> Path:
        return Path(self.root_dir).expanduser().resolve()
