import pickle
import numpy as np
from pathlib import Path

POOLS_PKL = Path(__file__).resolve().parent / "data" / "derived" / "pools_by_quarter.pkl"

class ReplayEnv:
    def __init__(self, rounds_per_quarter=200, max_quarters=20, pools_path=POOLS_PKL):
        with open(pools_path, "rb") as f:
            payload = pickle.load(f)

        self.pools    = payload["pools"]
        self.arms     = payload["meta"]["arms"]
        self.K        = payload["meta"]["K"]
        self.quarters = payload["meta"]["quarters"][:max_quarters]
        self.Tb       = rounds_per_quarter
        self.T        = len(self.quarters) * self.Tb
        self._rng     = None
        self._t       = 0

    def reset(self, seed=0):
        self._rng = np.random.default_rng(seed)
        self._t   = 0
        return self

    def step(self, arm):
        q_idx        = min(self._t // self.Tb, len(self.quarters) - 1)
        quarter      = self.quarters[q_idx]
        key          = (quarter, arm)
        pool         = self.pools.get(key) or self._nearest(arm, quarter)
        r_vec        = np.array(pool[self._rng.integers(len(pool))], dtype=float)
        self._t     += 1
        return r_vec[:2]  # <--- Sliced to 2D: [effectiveness, resistance]

    def _nearest(self, arm, quarter):
        for off in range(1, len(self.quarters) + 1):
            for q in [quarter - off, quarter + off]:
                if self.pools.get((q, arm)):
                    return self.pools[(q, arm)]
        raise RuntimeError(f"No pool data for arm {arm}")