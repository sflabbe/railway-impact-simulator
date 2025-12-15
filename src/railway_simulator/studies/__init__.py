"""
Studies framework: reproducible convergence and sensitivity analyses.

These helpers are intentionally lightweight and avoid coupling to internal
solver details. All simulations are executed via `railway_simulator.core.engine.run_simulation`.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple
import copy
import json
import subprocess
import re
import math

# ----------------------------
# Path helpers (dot + [idx])
# ----------------------------

_TOKEN_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)(\[(\d+)\])?$")

def _parse_path_tokens(path: str) -> List[Tuple[str, int | None]]:
    """
    Parse parameter paths:
      - 'k_wall' -> [('k_wall', None)]
      - 'fy[0]' -> [('fy', 0)]
      - 'train.fy[2]' -> [('train', None), ('fy', 2)]
    """
    tokens: List[Tuple[str, int | None]] = []
    for part in path.split("."):
        m = _TOKEN_RE.fullmatch(part.strip())
        if not m:
            raise ValueError(
                f"Invalid param path token: {part!r} (full path: {path!r}). "
                "Use dot notation and optional [index], e.g. 'fy[0]' or 'building.k'."
            )
        key = m.group(1)
        idx = int(m.group(3)) if m.group(3) is not None else None
        tokens.append((key, idx))
    return tokens


def get_by_path(cfg: Dict[str, Any], path: str) -> Any:
    """Get cfg value using dot + [idx] path."""
    d: Any = cfg
    for key, idx in _parse_path_tokens(path):
        if not isinstance(d, dict) or key not in d:
            raise KeyError(f"Path '{path}' not found at key '{key}'")
        d = d[key]
        if idx is not None:
            if not isinstance(d, (list, tuple)):
                raise TypeError(f"Path '{path}' expects list at '{key}', got {type(d)}")
            try:
                d = d[idx]
            except IndexError as e:
                raise IndexError(f"Path '{path}' index {idx} out of range for '{key}'") from e
    return d


def set_by_path(cfg: Dict[str, Any], path: str, value: Any) -> Dict[str, Any]:
    """
    Return a deep-copied cfg where the nested path is set to `value`.
    Creates intermediate dicts as needed; list indices must exist.
    """
    new_cfg = copy.deepcopy(cfg)
    d: Any = new_cfg
    tokens = _parse_path_tokens(path)
    for (key, idx) in tokens[:-1]:
        if key not in d or not isinstance(d[key], dict):
            d[key] = {}
        d = d[key]
        if idx is not None:
            if not isinstance(d, (list, tuple)):
                raise TypeError(f"Path '{path}' expects list at '{key}', got {type(d)}")
            d = d[idx]

    last_key, last_idx = tokens[-1]
    if last_key not in d:
        # allow setting new key
        d[last_key] = [] if last_idx is not None else None
    if last_idx is None:
        d[last_key] = value
    else:
        if not isinstance(d[last_key], list):
            raise TypeError(f"Path '{path}' expects list at '{last_key}', got {type(d[last_key])}")
        if last_idx >= len(d[last_key]):
            raise IndexError(f"Path '{path}' index {last_idx} out of range for '{last_key}'")
        d[last_key][last_idx] = value
    return new_cfg


# ----------------------------
# Reproducibility utilities
# ----------------------------

def get_git_hash() -> str:
    """Return current git hash (or 'unknown')."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return proc.stdout.strip()
    except Exception:
        return "unknown"


def _json_default(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return str(obj)


def save_study_metadata(output_dir: Path, *, metadata: Dict[str, Any]) -> None:
    """Write metadata JSON file to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "git_hash": get_git_hash(),
        **metadata,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )


# ----------------------------
# Simulation config helpers
# ----------------------------

def merge_with_engine_defaults(cfg_overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge user overrides with engine defaults, returning a full config dict.

    This mirrors what `run_simulation()` does internally, but allows studies
    to inspect defaults (e.g., base k_wall) without running a simulation.
    """
    from railway_simulator.core.engine import get_default_simulation_params

    base = get_default_simulation_params()
    base.update(copy.deepcopy(cfg_overrides))
    return base


def harmonize_time_grid(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure time integration parameters are self-consistent:
      - 'T_int' matches (0, T_max)
      - 'step' matches ceil(T_max / h_init)

    Returns a deep-copied cfg.
    """
    new = copy.deepcopy(cfg)
    T_max = float(new.get("T_max", 0.4))
    h = float(new.get("h_init", 1e-4))
    if h <= 0:
        raise ValueError("h_init must be > 0")
    new["T_int"] = (0.0, T_max)
    new["step"] = int(math.ceil(T_max / h))
    return new


def parse_floats_csv(s: str) -> List[float]:
    """Parse '1,2,3' or '1 2 3' into list of floats."""
    parts = re.split(r"[,\s]+", s.strip())
    return [float(p) for p in parts if p]
