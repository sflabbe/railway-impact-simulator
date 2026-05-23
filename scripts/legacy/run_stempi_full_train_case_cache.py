#!/usr/bin/env python3
"""Deprecated legacy chunked/cached runner for the historical consist table.

TODO(legacy): keep this script frozen for reproducibility. New production
workflows should use the neutral Project Workbench / study services instead.

Useful on CI or hosted notebooks with short command timeouts.  It writes one CSV
per scenario under results/stempi_full_train/case_rows and can finalize the full
report once all 36 rows exist.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.legacy.generate_stempi_full_train_table import (  # noqa: E402
    ICE4_GEOM,
    TRAXX_GEOM,
    add_ratios_and_status,
    build_ice4_full_train,
    build_ice4_lok_solo,
    build_traxx_full_train,
    build_traxx_lok_solo,
    evaluate_case_payload,
    hooke_single_mass_sanity,
    known_lok_solo_check,
    make_plots,
    make_response_grid,
    monotonicity_check,
    write_generated_configs,
    write_markdown_outputs,
)


def build_payloads() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    builders = {
        ("TRAXX BR187", "Lok_solo"): (TRAXX_GEOM, build_traxx_lok_solo),
        ("TRAXX BR187", "Zug_full"): (TRAXX_GEOM, build_traxx_full_train),
        ("ICE4 Endwagen proxy", "Lok_solo"): (ICE4_GEOM, build_ice4_lok_solo),
        ("ICE4 Endwagen proxy", "Zug_full"): (ICE4_GEOM, build_ice4_full_train),
    }
    speeds = [80.0, 120.0, 160.0]
    distances = [3.0, 5.0, 10.0]
    mu = 0.30

    configs: dict[tuple[str, str], dict[str, Any]] = {}
    couplings: dict[tuple[str, str], list[int]] = {}
    config_meta: dict[str, Any] = {}
    generated_configs: dict[str, dict[str, Any]] = {}
    for key, (_geom, builder) in builders.items():
        cfg, cidx, meta = builder()
        configs[key] = cfg
        couplings[key] = cidx
        config_meta[f"{key[0]} / {key[1]}"] = meta
        generated_configs[f"{key[0]}_{key[1]}"] = cfg

    cases: list[dict[str, Any]] = []
    total_cases = len(builders) * len(distances) * len(speeds)
    case_no = 0
    for (vehicle, mode), (geom, _builder) in builders.items():
        for a in distances:
            for v0 in speeds:
                case_no += 1
                cases.append(
                    {
                        "case_no": case_no,
                        "total_cases": total_cases,
                        "vehicle": vehicle,
                        "mode": mode,
                        "geom": geom,
                        "cfg": configs[(vehicle, mode)],
                        "coupling_indices": couplings[(vehicle, mode)],
                        "a_m": a,
                        "v0_kmh": v0,
                        "mu": mu,
                    }
                )
    metadata = {
        "scope": "reduced multibody proxy / parametric full consist model",
        "not_calibrated_to_crash_tests": True,
        "mu_runout": mu,
        "speeds_kmh": speeds,
        "distances_m": distances,
        "engine_settings": {
            "solver": "picard",
            "h_init": 5.0e-5,
            "T_max": 0.15,
            "picard_max_iters": 30,
            "picard_tol": 1.0e-5,
            "contact_model": "lankarani-nikravesh",
            "emit_peak_diagnostics": False,
        },
        "geometry": {
            "TRAXX BR187": TRAXX_GEOM.__dict__ | {"l_eff_m": TRAXX_GEOM.l_eff_m},
            "ICE4 Endwagen proxy": ICE4_GEOM.__dict__ | {"l_eff_m": ICE4_GEOM.l_eff_m},
            "zug_cap_applied_to_both_modes_for_side_by_side_comparability": True,
        },
        "configurations": config_meta,
    }
    return cases, generated_configs, metadata


def parse_case_spec(spec: str, n_cases: int) -> list[int]:
    if spec.strip().lower() in {"all", "*"}:
        return list(range(1, n_cases + 1))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [x for x in out if 1 <= x <= n_cases]


def finalize(outdir: Path, metadata: dict[str, Any], generated_configs: dict[str, dict[str, Any]]) -> None:
    case_dir = outdir / "case_rows"
    files = sorted(case_dir.glob("case_*.csv"))
    if len(files) != 36:
        missing = [i for i in range(1, 37) if not (case_dir / f"case_{i:03d}.csv").exists()]
        raise SystemExit(f"Need 36 case rows, found {len(files)}. Missing: {missing}")
    full_df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    full_df = add_ratios_and_status(full_df)
    full_df = full_df.sort_values(["Fahrzeug", "a_m", "v0_kmh", "mode"]).reset_index(drop=True)
    grid_df = make_response_grid(full_df)

    outdir.mkdir(parents=True, exist_ok=True)
    write_generated_configs(outdir, generated_configs)
    full_df.to_csv(outdir / "stempi_full_train_comparison_full.csv", index=False)
    grid_df.to_csv(outdir / "response_grid_lok_vs_zug.csv", index=False)

    known_check = known_lok_solo_check(full_df)
    mono_violations = monotonicity_check(full_df)
    hooke_check = hooke_single_mass_sanity(outdir)
    metadata = dict(metadata)
    metadata.update(
        {
            "known_lok_solo_check": known_check,
            "monotonicity_violations": mono_violations,
            "hooke_single_mass_sanity": hooke_check,
        }
    )
    (outdir / "stempi_full_train_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    make_plots(outdir, grid_df)
    write_markdown_outputs(
        outdir=outdir,
        full_df=full_df,
        grid_df=grid_df,
        metadata=metadata,
        known_check=known_check,
        mono_violations=mono_violations,
        hooke_check=hooke_check,
    )
    print(f"Finalized {outdir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=ROOT / "results" / "stempi_full_train")
    parser.add_argument("--cases", default="all", help="Case spec, e.g. 1-6,7,12 or all")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--run", action="store_true", help="Run selected cases. If neither --run nor --finalize is set, both are done.")
    args = parser.parse_args()

    cases, generated_configs, metadata = build_payloads()
    run_cases = args.run or not args.finalize
    do_finalize = args.finalize or not args.run
    outdir: Path = args.outdir
    case_dir = outdir / "case_rows"
    case_dir.mkdir(parents=True, exist_ok=True)

    if run_cases:
        selected = parse_case_spec(args.cases, len(cases))
        for case_no in selected:
            path = case_dir / f"case_{case_no:03d}.csv"
            if args.skip_existing and path.exists():
                print(f"case {case_no:03d}: skip existing")
                continue
            row, log = evaluate_case_payload(cases[case_no - 1])
            pd.DataFrame([row]).to_csv(path, index=False)
            print(log, flush=True)

    if do_finalize:
        finalize(outdir, metadata, generated_configs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
