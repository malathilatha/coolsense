"""
Load the real LBNL rooftop-unit (RTU) data and turn it into 10-minute windows
while the compressor is running.

Two sources, mapped to the same canonical names:
  * lbnl/Simulated RTU/  NREL Modelica model of a real 5-ton R410A unit,
                         6 refrigerant/airflow faults at several severities.
  * lbnl/Field RTU/      Real measurements from a 10-ton unit in Colchester, CT:
                         healthy months + one month with circuit B 40% undercharged.

Units differ (simulated: F and Pa; field: C and bar). That is fine: every unit
is compared only against its own healthy behaviour (see baseline.py).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent / "lbnl"
WINDOW = "10min"

INPUTS = ["OAT", "RAT", "RAH", "STAGE"]                 # operating conditions
TARGETS = ["SUCT_P", "DISC_P", "SUCT_T", "LIQ_T", "DISC_T", "SAT", "SA_FLOW"]

SIM_MAP = {
    "RTU_OA_TEMP": "OAT", "RTU_RA_TEMP": "RAT", "RTU_RA_HUM": "RAH",
    "RTU_STG_STA": "STAGE", "RTU_REFG_SUCT_PRES": "SUCT_P", "RTU_REFG_DISC_PRES": "DISC_P",
    "RTU_REFG_SUCT_TEMP": "SUCT_T", "RTU_REFG_COND_TEMP": "LIQ_T",
    "RTU_REFG_DISC_TEMP": "DISC_T", "RTU_SA_TEMP": "SAT", "RTU_SA_FLOW": "SA_FLOW",
}

# Simulated file stem -> (fault class shown to people, severity text)
SIM_FAULTS = {
    "baseline": ("Normal", "-"),
    **{f"condfouling{s}": ("Dirty outdoor (condenser) coil", f"{s}% less airflow") for s in (10, 20, 30, 40, 50)},
    **{f"evapfouling{s}": ("Low indoor airflow (dirty filter / indoor coil)", f"{s}% less airflow") for s in (10, 20, 30, 40, 50)},
    **{f"liquidpipe{s}bar": ("Blocked liquid line / filter-drier", f"{int(s)} bar drop") for s in ("01", "04", "08", "10")},
    **{f"suctionpipe{s}bar": ("Restricted suction line", f"{int(s)} bar drop") for s in ("01", "03", "06", "09")},
    **{f"overcharge{s}": ("Too much refrigerant (overcharge)", f"+{s}%") for s in (10, 15, 20)},
    **{f"undercharge{s}": ("Low refrigerant (gas leak)", f"-{s}%") for s in (10, 15, 20)},
}


def _windows(df: pd.DataFrame, running: pd.Series) -> pd.DataFrame:
    """10-minute means, keeping only windows where the compressor ran the whole
    time AND was already running in the previous window (skip start-up)."""
    df = df.copy()
    df["_run"] = running.astype(float)
    w = df.resample(WINDOW).mean()
    full = w["_run"] >= 0.999
    w = w[full & full.shift(1, fill_value=False)]
    return w.drop(columns="_run").dropna(subset=TARGETS + INPUTS)


def load_sim(stem: str) -> pd.DataFrame:
    path = ROOT / "Simulated RTU" / f"RTU_sim_{stem}.csv"
    raw = pd.read_csv(path, usecols=["Datetime", *SIM_MAP], parse_dates=["Datetime"], index_col="Datetime")
    raw = raw.rename(columns=SIM_MAP)
    # Stage: 0.1 = off, 0.67 = low, 1.0 = high. Keep windows with a constant stage.
    running = raw["STAGE"] >= 0.6
    w = _windows(raw, running)
    w = w[(w["STAGE"].round(2).isin([0.67, 1.0]))]
    label, severity = SIM_FAULTS[stem]
    w["label"], w["severity"], w["source"] = label, severity, stem
    return w


def load_field(name: str, circuit: int) -> pd.DataFrame:
    """One refrigerant circuit of the real Colchester unit treated as its own 'unit'."""
    path = ROOT / "Field RTU" / f"{name}.csv"
    raw = pd.read_csv(path, parse_dates=["Datetime"], index_col="Datetime")
    k = circuit
    run = {c: (raw[f"RTU_REFG_DISC_PRES_{c}"] - raw[f"RTU_REFG_SUCT_PRES_{c}"]) > 3 for c in (1, 2)}
    out = pd.DataFrame({
        "OAT": raw["RTU_OA_TEMP"], "RAT": raw["RTU_RA_TEMP"], "RAH": raw["RTU_RA_HUM"],
        # Both circuits on = high stage, one = low stage (same meaning as the simulation).
        "STAGE": np.where(run[1] & run[2], 1.0, 0.67),
        "SUCT_P": raw[f"RTU_REFG_SUCT_PRES_{k}"], "DISC_P": raw[f"RTU_REFG_DISC_PRES_{k}"],
        "SUCT_T": raw[f"RTU_REFG_SUCT_TEMP_{k}"], "LIQ_T": raw[f"RTU_REFG_COND_TEMP_{k}"],
        "DISC_T": raw[f"RTU_REFG_DISC_TEMP_{k}"], "SAT": raw["RTU_SA_TEMP"],
        "SA_FLOW": raw["RTU_SA_FLOW"],
    }, index=raw.index)
    w = _windows(out, run[k])
    w["label"] = "Low refrigerant (gas leak)" if "Undercharged" in name and circuit == 2 else "Normal"
    w["severity"] = "-40%" if w["label"].iat[0] != "Normal" else "-"
    w["source"] = f"{name}_c{circuit}"
    return w


def time_split(df: pd.DataFrame, frac: float):
    """First `frac` of the timeline vs the rest (no shuffling across time)."""
    cut = df.index.min() + (df.index.max() - df.index.min()) * frac
    return df[df.index <= cut], df[df.index > cut]
