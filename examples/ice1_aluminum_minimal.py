from pathlib import Path
import matplotlib.pyplot as plt
import yaml

from railway_simulator.core.engine import (
    run_simulation,
    _coerce_scalar_types_for_simulation,
)


def main():
    project_root = Path(__file__).resolve().parents[1]
    cfg_path = project_root / "configs" / "ice1_aluminum.yml"

    with cfg_path.open("r") as f:
        raw = yaml.safe_load(f)

    params = _coerce_scalar_types_for_simulation(raw)
    df = run_simulation(params)

    plt.figure()
    plt.plot(df["Time_ms"], df["Impact_Force_MN"])
    plt.xlabel("t [ms]")
    plt.ylabel("F_impact [MN]")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    E0 = float(df["E_total_initial_J"].iloc[0])
    rel_err = df["E_balance_error_J"].abs().max() / (E0 + 1e-16)
    print(f"Max relative energy error: {rel_err:.3e}")


if __name__ == "__main__":
    main()
