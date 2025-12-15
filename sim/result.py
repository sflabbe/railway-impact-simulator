from dataclasses import dataclass
import numpy as np
from typing import Dict, Any, Optional

@dataclass
class SimulationResult:
    time: np.ndarray                 # shape (N,)
    contact_force: np.ndarray        # shape (N,)
    penetration: np.ndarray          # shape (N,)  (or max_penetration)
    data: Dict[str, Any]             # any extra series
    metadata: Dict[str, Any]         # config snapshot, git hash, etc.
    energy: Optional[Dict[str, np.ndarray]] = None   # kinetic, internal, etc.

