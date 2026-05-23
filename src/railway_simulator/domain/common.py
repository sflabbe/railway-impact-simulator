"""Small shared helpers for domain objects."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    """Return a timezone aware UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    """Generate a compact readable identifier."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def json_dumps_stable(value: Any) -> str:
    """Serialize JSON with stable ordering and pathlib support."""

    def default(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if is_dataclass(obj):
            return asdict(obj)
        if hasattr(obj, "tolist"):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=default)


def stable_hash(value: Any) -> str:
    """SHA256 hash of a stable JSON representation."""
    return hashlib.sha256(json_dumps_stable(value).encode("utf-8")).hexdigest()
