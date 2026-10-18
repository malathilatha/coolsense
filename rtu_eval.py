"""
CoolSense on real LBNL rooftop-unit data.

  1. Learn each unit's healthy behaviour (baseline.py) -> z-score residuals.
  2. Diagnose the fault from the z-scores with a classifier trained on the
     simulated faults: fixed rules vs gradient boosting vs TabPFN.
  3. Run the same thing on a REAL unit (Colchester, CT) that had circuit B
     undercharged by 40%, without retraining.
  4. Few-shot: give the classifier a couple of days of labelled real data and
     see if diagnosis improves (TabPFN just takes them as context).

Writes results/rtu_metrics.md.

    python rtu_eval.py              # uses TabPFN if installed
    python rtu_eval.py --no-tabpfn
"""
import argparse
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

import rtu_data as R
from baseline import NormalModel, Z_FEATURES, anomaly_score

warnings.filterwarnings("ignore")
LOW = "Low refrigerant (gas leak)"

# Hold these severities out completely to test "a fault level it never saw".
UNSEEN = {"condfouling30", "evapfouling30", "liquidpipe04bar", "suctionpipe03bar",
          "overcharge15", "undercharge15"}


# ----------------------------------------------------------------- data
def load_windows():
    os.makedirs("cache", exist_ok=True)
    if os.path.exists("cache/sim_windows.pkl"):
        sims = pd.read_pickle("cache/sim_windows.pkl")
    else:
        sims = {s: R.load_sim(s) for s in R.SIM_FAULTS}
        pd.to_pickle(sims, "cache/sim_windows.pkl")
    if os.path.exists("cache/field_windows.pkl"):
        field = pd.read_pickle("cache/field_windows.pkl")
    else:
        field = {(n, c): R.load_field(n, c) for n in ["Site2_Unfaulted", "Site2_Undercharged40"] for c in (1, 2)}
        pd.to_pickle(field, "cache/field_windows.pkl")
    return sims, field


def sim_sets(sims, nm, b_cal, b_test):
    tr, te = [], []
    for s, w in sims.items():
        if s == "baseline":
            parts = [(b_cal, tr), (b_test, te)]
        else:
            a, b = R.time_split(w, 0.7)
            parts = [(a, tr), (b, te)]
        for part, bucket in parts:
            z = nm.zscores(part)
            z["label"], z["src"] = part["label"].iat[0], s
            bucket.append(z)
    return pd.concat(tr), pd.concat(te)


def balanced_context(df, per_class=450, seed=0):
    """TabPFN context: equal rows per fault so 'Normal' is not drowned out."""
    return pd.concat([g.sample(min(len(g), per_class), random_state=seed)
                      for _, g in df.groupby("label")])


# ----------------------------------------------------------------- models
def rules(z: pd.DataFrame) -> np.ndarray:
    """A technician's rules of thumb, expressed on the z-scores."""
    out = []
    for _, r in z.iterrows():
        if r.z_SA_FLOW < -3:
            out.append("Low indoor airflow (dirty filter / indoor coil)")
        elif r.z_DISC_P > 3 and r.z_LIQ_T > 2:
            out.append("Dirty outdoor (condenser) coil")
        elif r.z_SUCT_P < -3 and r.z_DISC_T > 3:
            out.append(LOW)
        elif r.z_DISC_P < -3:
            out.append(LOW)
        else:
            out.append("Normal")
    return np.array(out)


def make_models(use_tabpfn):
    models = {"Gradient boosting": lambda: HistGradientBoostingClassifier(
        max_iter=300, class_weight="balanced", random_state=0)}
    if use_tabpfn:
        try:
            from tabpfn import TabPFNClassifier
            models["TabPFN"] = lambda: TabPFNClassifier(random_state=0)
        except Exception as e:  # noqa: BLE001
            print(f"TabPFN not available ({e}); continuing without it.")
    return models


def gate(z: pd.DataFrame, diag: np.ndarray, tau: float) -> np.ndarray:
    """Two-stage: call it Normal unless the unit is clearly off its healthy
    behaviour (anomaly > tau); only then name the fault."""
    return np.where(anomaly_score(z).to_numpy() > tau, diag, "Normal")


def diagnose(model, z: pd.DataFrame, tau: float) -> np.ndarray:
    """Detect first, then name the fault only for windows over the alarm level.
    Same result as gate(z, model.predict(z), tau) but far faster on CPU,
    because the classifier only sees flagged windows."""
    out = np.full(len(z), "Normal", dtype=object)
    flag = anomaly_score(z).to_numpy() > tau
    if flag.any():
        out[flag] = model.predict(z.loc[flag, Z_FEATURES])
    return out


def scores(y, p):
    y, p = np.asarray(y), np.asarray(p)
    n = y == "Normal"
    return {"accuracy": accuracy_score(y, p), "macro_F1": f1_score(y, p, average="macro"),
            "faults_detected": float(np.mean(p[~n] != "Normal")),
            "false_alarms": float(np.mean(p[n] != "Normal"))}


# ----------------------------------------------------------------- main
TAU_Q = 0.99   # simulated unit: alarm = 99th percentile of its own healthy anomaly score
# Real unit: its healthy baseline is only 2 months long (Aug-Sep 2020), and a year
# later a healthy circuit drifts by 2-4x its normal variation (weather range,
# sensor drift). So the field alarm level is set higher, at 5x. Section 4 shows
# the full threshold sweep so readers can judge this choice.
FIELD_TAU = 5.0


def main(use_tabpfn=True, per_source=None):
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    sims, field = load_windows()
    report = []

    # ---- 1. Simulated unit: healthy model fit / calibrate / test in time order.
    base = sims["baseline"]
    b_fit, rest = R.time_split(base, 0.5)
    b_cal, b_test = R.time_split(rest, 0.5)
    nm_sim = NormalModel(use_tabpfn=False).fit(b_fit, b_cal)  # GB here: fast on 100k+ windows; TabPFN is used for diagnosis
    tau_sim = float(anomaly_score(nm_sim.zscores(b_cal)).quantile(TAU_Q))
    tr, te = sim_sets(sims, nm_sim, b_cal, b_test)
    if per_source:   # keeps CPU time reasonable; same sample for every model
        te = pd.concat([g.sample(min(len(g), per_source), random_state=0) for _, g in te.groupby("src")])
    ctx = balanced_context(tr[tr.label != "Normal"])          # diagnoser sees faults only
    ctx.to_csv("models/rtu_context.csv", index=False)
    joblib.dump({"normal": nm_sim, "tau": tau_sim}, "models/rtu_detector_sim.joblib")
    print(f"sim train={len(tr):,} test={len(te):,} context={len(ctx)} tau={tau_sim:.2f}")

    models = make_models(use_tabpfn)
    fitted, rows = {}, []
    by_sev = te[["src", "label"]].copy()
    p = rules(te[Z_FEATURES]); by_sev["Rules"] = p
    rows.append({"model": "Rules of thumb", **scores(te.label, p), "seconds": 0})
    for name, mk in models.items():
        t = time.time()
        m = mk().fit(ctx[Z_FEATURES], ctx["label"])
        p = diagnose(m, te, tau_sim)
        rows.append({"model": f"{name} ({len(ctx)} rows)", **scores(te.label, p), "seconds": round(time.time() - t, 1)})
        by_sev[name], fitted[name] = p, m
    best = "TabPFN" if "TabPFN" in fitted else "Gradient boosting"
    joblib.dump(fitted["Gradient boosting"], "models/rtu_gboost.joblib")
    report.append("## 1. Simulated unit, later weeks it never saw (detect, then diagnose)\n\n"
                  + pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f"))

    sev = by_sev.assign(severity=by_sev.src.map(lambda s: R.SIM_FAULTS[s][1]))
    for c in ["Rules", *fitted]:
        sev[c] = (sev[c] == sev["label"]).astype(float)
    tab = sev.groupby(["label", "severity"])[["Rules", *fitted]].mean().reset_index()
    report.append("## 2. Share of windows correctly diagnosed, by severity\n\n" + tab.to_markdown(index=False, floatfmt=".2f"))

    ctx_u = balanced_context(tr[(tr.label != "Normal") & ~tr.src.isin(UNSEEN)])
    te_u = te[te.src.isin(UNSEEN)]
    rows_u = [{"model": n, "correct_diagnosis": float((diagnose(mk().fit(ctx_u[Z_FEATURES], ctx_u["label"]), te_u, tau_sim) == te_u.label).mean())}
              for n, mk in models.items()]
    report.append("## 3. Fault levels never shown in training (" + ", ".join(sorted(UNSEEN)) + ")\n\n"
                  + pd.DataFrame(rows_u).to_markdown(index=False, floatfmt=".3f"))

    # ---- 4. Real unit. Healthy reference = Aug-Sep 2020 (after the documented fault).
    zf, det = {}, {}
    for c in (1, 2):
        healthy = field[("Site2_Unfaulted", c)]
        h20, h21 = healthy[healthy.index.year == 2020], healthy[healthy.index.year == 2021]
        f_fit, f_cal = R.time_split(h20, 0.6)
        nm = NormalModel(use_tabpfn=False).fit(f_fit, f_cal)  # GB here: fast on 100k+ windows; TabPFN is used for diagnosis
        z_cal = nm.zscores(f_cal)
        det[c] = {"normal": nm, "tau": FIELD_TAU, "z_cal": z_cal}
        joblib.dump({"normal": nm, "tau": det[c]["tau"]}, f"models/rtu_detector_field_c{c}.joblib")
        zf[(c, "fault")] = nm.zscores(field[("Site2_Undercharged40", c)])
        zf[(c, "2021")] = nm.zscores(h21)

    PERIODS = {"fault": "Jun 2020 (documented 40% undercharge on circuit B)", "2021": "Jun-Sep 2021 (dataset says unfaulted)"}

    def field_table(diagnoser, tag):
        out = []
        for (c, per), z in zf.items():
            tau = det[c]["tau"]
            a = anomaly_score(z)
            day = a.groupby(a.index.date).median()
            p = pd.Series(diagnose(diagnoser, z, tau))
            flagged = p[p != "Normal"]
            out.append({"circuit": "A" if c == 1 else "B", "period": PERIODS[per],
                        "days alarmed (daily median > 5)": f"{(day > tau).sum()}/{len(day)}",
                        "windows flagged": round(float((p != "Normal").mean()), 2),
                        f"most common diagnosis ({tag})": flagged.value_counts().index[0] if len(flagged) else "-",
                        "flagged windows called low refrigerant": round(float((flagged == LOW).mean()), 2) if len(flagged) else 0})
        return pd.DataFrame(out)

    zmeans = pd.DataFrame([{"circuit": "A" if c == 1 else "B", "period": PERIODS[per],
                            **z[[k for k in Z_FEATURES if k.startswith("z_")]].mean().round(1).to_dict()}
                           for (c, per), z in zf.items()])
    sweep = []
    for (c, per), z in zf.items():
        day = anomaly_score(z).groupby(z.index.date).median()
        sweep.append({"circuit": "A" if c == 1 else "B", "period": PERIODS[per], "days": len(day),
                      **{f"alarm days at >{t}": int((day > t).sum()) for t in (2, 3, 5, 8)}})
    report.append("## 4. Real rooftop unit (Colchester, CT), diagnoser trained only on simulation\n\n"
                  "Daily alarms at different alarm levels (anomaly = RMS of z-scores; healthy calibration ~1):\n\n"
                  + pd.DataFrame(sweep).to_markdown(index=False) + "\n\n"
                  + field_table(fitted[best], "simulation only").to_markdown(index=False)
                  + "\n\nAverage z-scores (distance from each circuit's own healthy level):\n\n"
                  + zmeans.to_markdown(index=False))

    # ---- 5. Few-shot: technician confirms the first 2 days of the real fault.
    zB = zf[(2, "fault")]
    cut = zB.index.min() + pd.Timedelta(days=2)
    shot = zB[zB.index < cut].assign(label=LOW)
    rest_B = zB[zB.index >= cut]
    few = []
    for name, mk in models.items():
        for k, c_ in [("simulation only", ctx), ("+2 confirmed days of the real fault", pd.concat([ctx, shot]))]:
            m = mk().fit(c_[Z_FEATURES], c_["label"])
            pB = pd.Series(diagnose(m, rest_B, det[2]["tau"]))
            p21 = pd.Series(diagnose(m, zf[(2, "2021")], det[2]["tau"]))
            pA = pd.Series(diagnose(m, zf[(1, "2021")], det[1]["tau"]))
            few.append({"model": name, "context": k,
                        "B fault days 3-26: called low refrigerant": round(float((pB == LOW).mean()), 2),
                        "B 2021: called low refrigerant": round(float((p21 == LOW).mean()), 2),
                        "A 2021: called normal": round(float((pA == "Normal").mean()), 2)})
            if name == best and k.startswith("+2"):
                joblib.dump(m, "models/rtu_diagnoser_fewshot.joblib")
                c_.to_csv("models/rtu_context_fewshot.csv", index=False)
    report.append("## 5. Few-shot: add 2 technician-confirmed days of the real fault to the context\n\n"
                  + pd.DataFrame(few).to_markdown(index=False))

    text = "\n\n".join(report) + "\n"
    with open("results/rtu_metrics.md", "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-tabpfn", action="store_true")
    ap.add_argument("--per-source", type=int, default=None,
                    help="test windows per simulated scenario (default: 300 with TabPFN, all without)")
    a = ap.parse_args()
    main(use_tabpfn=not a.no_tabpfn,
         per_source=a.per_source if a.per_source else (None if a.no_tabpfn else 300))
