from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from railway_simulator.config.loader import migrate_config_dict


def _iter_inputs(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            yield from path.glob("*.yml")
            yield from path.glob("*.yaml")
        else:
            yield path


def migrate_file(path: Path, output_dir: Path | None) -> Path:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    migrated = migrate_config_dict(data)
    target = path if output_dir is None else output_dir / path.name
    target.write_text(yaml.safe_dump(migrated, sort_keys=False), encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate railway-sim YAML configs.")
    parser.add_argument("paths", nargs="+", type=Path, help="Config files or directories.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory (defaults to overwrite in-place).",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for path in _iter_inputs(args.paths):
        if not path.exists():
            raise FileNotFoundError(path)
        target = migrate_file(path, output_dir)
        print(f"Migrated {path} -> {target}")


if __name__ == "__main__":
    main()
