"""
"What should this AC read right now if it were healthy?"

For each refrigerant/air reading we learn the unit's healthy behaviour from its
own fault-free history, as a function of weather (OAT), room air (RAT, RAH) and
compressor stage. A fault then shows up as a residual: actual minus expected,
divided by how much that reading normally wobbles (a z-score).

z-scores make different units comparable (F vs C, Pa vs bar, 5-ton vs 10-ton),
which is what lets a classifier trained on simulated faults run on a real unit.

Model: TabPFN regressor if installed (small healthy history, no training loop),
otherwise scikit-learn gradient boosting.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from rtu_data import INPUTS, TARGETS

Z_CLIP = 15.0


def _regressor(use_tabpfn: bool):
    if use_tabpfn:
        try:
            from tabpfn import TabPFNRegressor
            return TabPFNRegressor(random_state=0)
        except Exception:  # noqa: BLE001
            pass
    return HistGradientBoostingRegressor(max_iter=200, random_state=0)


class NormalModel:
    def __init__(self, use_tabpfn: bool = False, max_rows: int = 3000):
        self.use_tabpfn, self.max_rows = use_tabpfn, max_rows
        self.models, self.scale = {}, {}

    def fit(self, healthy_fit: pd.DataFrame, healthy_cal: pd.DataFrame):
        """healthy_fit trains the expectation; healthy_cal (a later, unseen healthy
        period) measures normal wobble so z-scores are honest."""
        fit = healthy_fit.sample(min(len(healthy_fit), self.max_rows), random_state=0)
        for t in TARGETS:
            m = _regressor(self.use_tabpfn).fit(fit[INPUTS], fit[t])
            self.models[t] = m
            resid = healthy_cal[t] - m.predict(healthy_cal[INPUTS])
            floor = 0.005 * max(healthy_cal[t].abs().mean(), 1e-6)   # constant signals (e.g. sim fan flow)
            self.scale[t] = (float(resid.median()), max(float(resid.std()), floor))
        return self

    def zscores(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for t in TARGETS:
            bias, s = self.scale[t]
            r = df[t] - self.models[t].predict(df[INPUTS]) - bias
            out[f"z_{t}"] = (r / s).clip(-Z_CLIP, Z_CLIP)
        out["STAGE"] = df["STAGE"].to_numpy()
        return out


Z_FEATURES = [f"z_{t}" for t in TARGETS] + ["STAGE"]

READABLE = {
    "z_SUCT_P": "suction (low-side) pressure", "z_DISC_P": "discharge (high-side) pressure",
    "z_SUCT_T": "suction line temperature", "z_LIQ_T": "liquid line temperature",
    "z_DISC_T": "compressor discharge temperature", "z_SAT": "supply air temperature",
    "z_SA_FLOW": "indoor airflow",
}


def evidence(z: pd.Series, top: int = 3) -> list[str]:
    """Plain facts computed by code: which readings are furthest from healthy."""
    items = sorted(((k, z[k]) for k in READABLE if k in z and pd.notna(z[k])),
                   key=lambda kv: -abs(kv[1]))[:top]
    return [f"{READABLE[k]} is {'higher' if v > 0 else 'lower'} than this unit's healthy level "
            f"({abs(v):.1f}x its normal variation)" for k, v in items if abs(v) >= 2]


def anomaly_score(z: pd.DataFrame) -> pd.Series:
    """Fault detection without naming the fault: root-mean-square of the z-scores."""
    cols = [c for c in z.columns if c.startswith("z_")]
    return np.sqrt((z[cols] ** 2).mean(axis=1))
