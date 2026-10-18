"""
CoolSense live demo on real LBNL rooftop-unit data.
Your laptop plays the small box beside the AC.

    python rtu_eval.py      # once: builds models/
    streamlit run app.py
"""
import time

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import HistGradientBoostingClassifier

import rtu_data as R
from baseline import Z_FEATURES, anomaly_score, evidence
from explain import KB, explain

st.set_page_config(page_title="CoolSense", page_icon="❄️", layout="wide")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stMetric {font-family: Inter, sans-serif;}
.block-container {padding-top: 1.6rem; max-width: 1400px;}
.hero {background: radial-gradient(900px 400px at 85% 0%, #1d4a72 0%, #0f2640 60%); border: 1px solid #23415e;
       border-radius: 22px; padding: 28px 34px; margin-bottom: 14px;}
.hero .tag {color: #5cc8ff; letter-spacing: 3px; font-size: 1.25rem; font-weight: 600; text-transform: uppercase;}
.hero h1 {font-size: 2.5rem; font-weight: 800; margin: 6px 0 8px 0; color: #f4f8fb; line-height: 1.1;}
.hero h1 span {color: #5cc8ff;}
.hero p {color: #b9cadb; font-size: 1.2rem; margin: 0;}
.pills {margin-top: 14px;}
.pill {display: inline-block; background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.15);
       border-radius: 999px; padding: 6px 16px; margin: 4px 8px 0 0; font-size: 1.05rem; color: #f4f8fb;}
.pill b {color: #5cc8ff;}
[data-testid="stMetric"] {background: #11283d; border: 1px solid #23415e; border-radius: 16px; padding: 14px 18px;}
[data-testid="stMetricValue"] {font-weight: 800;}
h3, h4 {font-weight: 800 !important;}
[data-testid="stCaptionContainer"] {font-size: 1rem !important;}
[data-testid="stSidebar"] {border-right: 1px solid #23415e;}
</style>""", unsafe_allow_html=True)


def hero(tag, title_html, sub, pills=()):
    ps = "".join(f'<span class="pill">{x}</span>' for x in pills)
    st.markdown(f'<div class="hero"><div class="tag">{tag}</div><h1>{title_html}</h1><p>{sub}</p>'
                f'<div class="pills">{ps}</div></div>', unsafe_allow_html=True)

COLORS = {"idle": ("#11283d", "#9fb3c6"), "ok": ("#0f2f2a", "#4fd18b"),
          "alert": ("#33230f", "#ffb547"), "done": ("#0f2a40", "#5cc8ff")}


def step_card(ph, title, body, state="idle"):
    bg, fg = COLORS[state]
    ph.markdown(f"""<div style="background:{bg};border-left:6px solid {fg};border-radius:14px;
        padding:14px 18px;min-height:104px"><div style="color:{fg};font-weight:700;font-size:0.85rem;letter-spacing:1px">{title}</div>
        <div style="font-size:1.1rem;margin-top:6px;color:#f4f8fb">{body}</div></div>""", unsafe_allow_html=True)


def chat_bubble(text, language):
    rtl = language in ("Arabic", "Urdu")
    return f"""<div style="background:#dcf8c6;border-radius:14px;padding:12px 16px;max-width:95%;
        direction:{'rtl' if rtl else 'ltr'};font-size:1.05rem;color:#111;box-shadow:0 1px 2px rgba(0,0,0,.15)">
        {text}<div style="text-align:right;font-size:0.75rem;color:#667;margin-top:6px">CoolSense ✓✓</div></div>"""

STREAMS = {
    "real_B_fault": "🔴 Real AC with a gas leak (June 2020)",
    "real_A_2021": "🟢 Same AC, healthy side (summer 2021)",
    "real_B_2021": "🟠 Same AC a year later (labelled healthy, but is it?)",
    "sim_condenser": "🧪 Simulated AC: outdoor coil gets dirty",
}


@st.cache_resource
def diagnoser(context_csv: str):
    ctx = pd.read_csv(context_csv)
    try:
        from tabpfn import TabPFNClassifier
        return TabPFNClassifier(random_state=0).fit(ctx[Z_FEATURES], ctx["label"]), "TabPFN"
    except Exception:  # noqa: BLE001
        m = HistGradientBoostingClassifier(max_iter=300, class_weight="balanced", random_state=0)
        return m.fit(ctx[Z_FEATURES], ctx["label"]), "Gradient boosting (TabPFN not installed)"


@st.cache_data
def stream(key: str):
    """Returns (windows, detector file, short description of the healthy reference)."""
    if key.startswith("real"):
        field = pd.read_pickle("cache/field_windows.pkl")
        if key == "real_B_fault":
            return field[("Site2_Undercharged40", 2)], "models/rtu_detector_field_c2.joblib"
        c = 1 if key == "real_A_2021" else 2
        h = field[("Site2_Unfaulted", c)]
        return h[h.index.year == 2021], f"models/rtu_detector_field_c{c}.joblib"
    sims = pd.read_pickle("cache/sim_windows.pkl")
    base, fault = sims["baseline"], sims["condfouling40"]
    start = base.index.min()
    # Late part of the timeline (not used to fit the healthy model), healthy then faulty.
    t0 = start + pd.Timedelta(days=75)
    w = pd.concat([base[(base.index >= t0) & (base.index < t0 + pd.Timedelta(days=5))],
                   fault[(fault.index >= t0 + pd.Timedelta(days=5)) & (fault.index < t0 + pd.Timedelta(days=12))]])
    return w, "models/rtu_detector_sim.joblib"


with st.sidebar:
    st.header("❄️ CoolSense")
    view = st.radio("View", ["📱 Live AC (owner + technician)", "🏢 Maintenance company (B2B)"])
    key = st.selectbox("Data stream", list(STREAMS), format_func=STREAMS.get)
    few = st.toggle("Diagnoser includes 2 technician-confirmed real days", value=True,
                    help="Off = trained on simulated faults only. On = the same context plus the first "
                         "2 days of the real Colchester fault (TabPFN just takes them as extra rows).")
    language = st.selectbox("Message language", ["English", "Arabic", "Hindi", "Tamil", "Malayalam", "Urdu", "Tagalog"])
    llm = st.selectbox("Gemma model", ["gemma3:4b", "gemma3:1b"],
                       help="4b writes Tamil/Malayalam/Arabic much better; 1b is faster.")
    speed = st.slider("Windows per second (1 window = 10 min)", 2, 60, 20)
    go = st.button("▶ Start replay", type="primary")



# ------------------------------------------------------------------ B2B view
@st.cache_data(show_spinner=False)
def fleet_rows(few_shot: bool):
    """Status of every unit in the demo fleet: days outside normal + most likely fault."""
    ctx_p = "models/rtu_context_fewshot.csv" if few_shot else "models/rtu_context.csv"
    # The fleet scan checks thousands of readings at once, so it uses the fast gradient-boosting
    # model on the same labelled examples (TabPFN on a laptop CPU is too slow for a whole fleet).
    ctx = pd.read_csv(ctx_p)
    mdl = HistGradientBoostingClassifier(max_iter=300, class_weight="balanced", random_state=0).fit(
        ctx[Z_FEATURES], ctx["label"])
    rows = []
    for k, unit in [("real_B_fault", "Rooftop AC · circuit B · Jun 2020"),
                    ("real_B_2021", "Rooftop AC · circuit B · summer 2021"),
                    ("real_A_2021", "Rooftop AC · circuit A · summer 2021"),
                    ("sim_condenser", "Simulated AC · test unit")]:
        w, dp = stream(k)
        if k == "real_B_fault" and few_shot:
            w = w[w.index >= w.index.min() + pd.Timedelta(days=2)]
        d = joblib.load(dp)
        z = d["normal"].zscores(w)
        s = anomaly_score(z)
        days = s.groupby(z.index.date).median()
        bad = int((days > d["tau"]).sum())
        flagged = z[s > d["tau"]]
        if len(flagged):
            top = pd.Series(mdl.predict(flagged[Z_FEATURES])).value_counts(normalize=True)
            fault, share = top.index[0], top.iloc[0]
        else:
            fault, share = "-", 0.0
        share_bad = bad / max(len(days), 1)
        # A day or two of blips in a whole summer is noise, not a fault.
        if share_bad > 0.5:
            prio = "🔴 Visit this week"
        elif share_bad > 0.05:
            prio = "🟠 Check at next service"
        else:
            prio, fault = "🟢 Healthy", "-"
        rows.append({"Unit": unit, "Days monitored": len(days), "Days outside normal": bad,
                     "Likely fault": fault, "Priority": prio,
                     "Bring": ", ".join(KB[fault]["parts"][:3]) if fault in KB else "-"})
    return pd.DataFrame(rows)


if view.startswith("🏢"):
    hero("B2B · Maintenance company view", "Visit the right AC, <span>with the right parts.</span>",
         "Who pays: AC maintenance companies and facility managers who look after hundreds of units. "
         "Instead of visiting every AC on a schedule, they see which units need a visit, what is probably wrong, "
         "and what to bring, so the first visit can fix it.",
         ["🔴 Priority list", "🧰 What to bring", "💰 Return on your numbers"])
    with st.spinner("Checking every unit in the fleet…"):
        fleet = fleet_rows(few)
    c1, c2, c3 = st.columns(3)
    c1.metric("Units monitored", len(fleet))
    c2.metric("Need a visit now", int(fleet["Priority"].str.startswith("🔴").sum()))
    c3.metric("Healthy", int(fleet["Priority"].str.startswith("🟢").sum()))
    st.dataframe(fleet, hide_index=True, use_container_width=True)
    st.caption("Demo fleet built from the LBNL recordings (one real rooftop AC, two circuits, two summers) plus one "
               "simulated unit. A real deployment lists every AC under contract. The fleet scan uses the fast "
               "gradient-boosting model on the same labelled examples; the live view uses TabPFN.")

    st.markdown("#### 💰 Is it worth it? (your numbers)")
    st.caption("A simple calculator. Every number below is an input you control, not a measured result.")
    a1, a2, a3, a4 = st.columns(4)
    units = a1.number_input("ACs under contract", 10, 100000, 200, step=10)
    saved = a2.number_input("Wasted visits avoided per AC per year", 0.0, 10.0, 1.0, step=0.5)
    visit = a3.number_input("Cost of one visit (AED)", 0, 5000, 250, step=50)
    price = a4.number_input("CoolSense price per AC per month (AED)", 0, 1000, 10, step=5)
    value, cost = units * saved * visit, units * price * 12
    b1, b2, b3 = st.columns(3)
    b1.metric("Visit cost avoided per year", f"AED {value:,.0f}")
    b2.metric("CoolSense cost per year", f"AED {cost:,.0f}")
    b3.metric("Net for the company", f"AED {value - cost:,.0f}", f"{(value / cost if cost else 0):.1f}x return")
    st.markdown("**Also for the business:** fewer second visits in peak summer, leaks fixed instead of topped up, "
                "and a written record per unit for the building owner.  \n"
                "**For the resident (B2C):** a clear message in their own language before the AC fails.")
    st.stop()

hero("Live AC · replay of real sensor data", "CoolSense: the AC that tells you <span>what's wrong.</span>",
     "Real readings from a rooftop AC with a documented gas leak, replayed as if arriving live. Everything runs on this laptop, no cloud.",
     ["① Detect · <b>code</b>", "② Diagnose · <b>TabPFN</b>", "③ Explain · <b>Gemma</b>", "🌍 7 languages"])
with st.expander("ℹ️ What am I looking at? (30-second explanation)", expanded=False):
    st.markdown("""
- **The AC:** a real 10-ton rooftop AC on a warehouse in Connecticut, USA. Researchers (LBNL / University of Connecticut)
  recorded its pressures and temperatures every minute. In **June 2020 one of its two refrigerant circuits was 40% low on gas** (documented).
- **The demo:** your screen replays that recording quickly, as if the readings were arriving live from a small box beside the AC.
- **① Detect:** the box learned this AC's normal readings from its healthy months. It raises a flag when readings drift far from normal.
- **② Diagnose (TabPFN):** names the fault from a few hundred labelled examples, with no training step.
- **③ Explain (Gemma, local):** writes a plain message for the owner in their language, plus a job card for the technician.
- **In a real product** the same code reads live sensors (or the building's management system) instead of this recording.
""")


windows, det_path = stream(key)
det = joblib.load(det_path)
ctx_path = "models/rtu_context_fewshot.csv" if few else "models/rtu_context.csv"
model, model_name = diagnoser(ctx_path)
_ctx = pd.read_csv(ctx_path)
n_ctx, n_real = len(_ctx), (len(_ctx) - len(pd.read_csv("models/rtu_context.csv")) if few else 0)
st.caption(f"{STREAMS[key]} · {len(windows):,} running windows · alarm level {det['tau']:.1f} · "
           f"diagnoser: {model_name} · all on this laptop")
if key == "real_B_fault" and few:
    # The first 2 days are the technician-confirmed examples in the diagnoser's
    # context, so the replay starts after them (no testing on what it was shown).
    windows = windows[windows.index >= windows.index.min() + pd.Timedelta(days=2)]
    st.info("👷 A technician confirmed the leak on days 1-2; those readings were given to TabPFN as examples, so the replay starts on day 3.")

s1, s2, s3 = (c.empty() for c in st.columns(3))
step_card(s1, "① DETECT · is the AC healthy?", "Waiting for readings…")
step_card(s2, "② DIAGNOSE · TabPFN", "Only runs when ① raises a flag")
step_card(s3, "③ EXPLAIN · Gemma", "Writes the message once a fault is confirmed")
st.write("")
m1, m2, m3, m4 = (c.empty() for c in st.columns(4))
msg_ph = st.empty()
left, right = st.columns([3, 2])
left.markdown("#### 📈 How far each reading is from this AC's normal")
chart_ph = left.empty()
right.markdown(f"#### 🧠 {model_name.split(' (')[0]} diagnosis")
right.caption(f"No training step: TabPFN reads {n_ctx:,} labelled examples as context "
              f"({n_ctx - n_real:,} simulated faults" + (f" + {n_real} technician-confirmed real windows" if n_real else "")
              + ") and scores each flagged 10-minute window.")
tab_ph = right.empty()
status_ph = st.empty()
if not go:
    status_ph.info("Press ▶ Start replay.")
    st.stop()

z_all = det["normal"].zscores(windows)
# Expected ("normal") value of each reading for the same weather and load, in real units.
nm = det["normal"]
expected = {t: nm.models[t].predict(windows[R.INPUTS]) + nm.scale[t][0] for t in ("SUCT_P", "DISC_T", "SAT")}
real = key.startswith("real")
U = {"P": "bar" if real else "(sim units)", "T": "°C" if real else "°F"}
if not real:   # simulated unit reports pressure in Pa
    windows = windows.assign(SUCT_P=windows["SUCT_P"] / 1e6)
    expected["SUCT_P"] = expected["SUCT_P"] / 1e6
score = anomaly_score(z_all)
# Detect first; ask the diagnoser only about windows over the alarm level (fast on CPU).
flag = (score > det["tau"]).to_numpy()
diag = np.full(len(z_all), "Normal", dtype=object)
proba_df = pd.DataFrame()
recent_i = []
conf = np.ones(len(z_all))
if flag.any():
    with st.spinner(f"{model_name} is diagnosing {flag.sum()} flagged windows…"):
        proba = model.predict_proba(z_all.loc[flag, Z_FEATURES])
        proba_df = pd.DataFrame(proba, index=np.flatnonzero(flag), columns=model.classes_)
        diag[flag] = model.classes_[proba.argmax(axis=1)]
        conf[flag] = proba.max(axis=1)

hist, streak, alerted, recent = [], 0, None, []
for i, ts in enumerate(z_all.index):
    flagged = score.iat[i] > det["tau"]
    hist.append({"time": ts, "Health score (higher = worse)": min(score.iat[i], 20), "Alarm line": det["tau"],
                 "Low-side pressure": z_all["z_SUCT_P"].iat[i], "Compressor temperature": z_all["z_DISC_T"].iat[i],
                 "High-side pressure": z_all["z_DISC_P"].iat[i]})
    if i % 3 == 0:
        w = windows.iloc[i]
        m1.metric("Outdoor temperature", f"{w['OAT']:.0f} {U['T']}")
        m2.metric("Low-side pressure", f"{w['SUCT_P']:.1f} {U['P']}",
                  f"{w['SUCT_P'] - expected['SUCT_P'][i]:+.1f} vs normal", delta_color="off")
        m3.metric("Compressor temperature", f"{w['DISC_T']:.0f} {U['T']}",
                  f"{w['DISC_T'] - expected['DISC_T'][i]:+.0f} vs normal", delta_color="inverse")
        m4.metric("Air leaving the AC", f"{w['SAT']:.1f} {U['T']}",
                  f"{w['SAT'] - expected['SAT'][i]:+.1f} vs normal", delta_color="inverse")
        chart_ph.line_chart(pd.DataFrame(hist).set_index("time").tail(300))
        if flagged and recent_i:
            # Average TabPFN probability per fault over the last 6 flagged windows.
            p = proba_df.loc[recent_i].mean().sort_values(ascending=True)
            p.index = [s.split(" (")[0] for s in p.index]
            tab_ph.bar_chart(p.rename("probability"), horizontal=True, height=260)
        elif not flagged:
            tab_ph.success("Unit within its normal range: TabPFN not needed for this window.")
    streak = streak + 1 if flagged else 0
    if flagged:
        recent.append(diag[i])
        del recent[:-6]
        recent_i.append(i)
        del recent_i[:-6]
    # Majority of the last 6 flagged windows: one noisy window can't flip the diagnosis.
    fault = max(set(recent), key=recent.count) if recent else diag[i]
    if i % 3 == 0 or streak in (1, 3):
        if flagged:
            step_card(s1, f"① DETECT · {ts:%d %b %H:%M}", f"⚠️ Readings far from normal (score {score.iat[i]:.1f} > {det['tau']:.0f})", "alert")
            p_avg = proba_df.loc[recent_i].mean().sort_values(ascending=False) if recent_i else pd.Series(dtype=float)
            alt = p_avg.index[1] if len(p_avg) > 1 and p_avg.iloc[1] >= 0.25 else None
            if alt == fault:
                alt = p_avg.index[0]
            extra = (f"<br><small>also possible: {alt.split(' (')[0].lower()}, technician checks subcooling</small>"
                     if alt else f"<br><small>{recent.count(fault)} of last {len(recent)} flagged windows</small>")
            step_card(s2, "② DIAGNOSE · TabPFN", f"<b>{fault}</b>{extra}", "alert")
        else:
            step_card(s1, f"① DETECT · {ts:%d %b %H:%M}", f"✅ Normal (score {score.iat[i]:.1f})", "ok")
            if alerted is None:
                step_card(s2, "② DIAGNOSE · TabPFN", "Not needed: AC is normal", "ok")
    if not flagged:
        status_ph.success(f"{ts:%a %d %b %Y %H:%M} · ✅ Normal (anomaly {score.iat[i]:.1f})")
    else:
        status_ph.warning(f"{ts:%a %d %b %Y %H:%M} · ⚠️ {fault} ({recent.count(fault)}/{len(recent)} recent windows) · anomaly {score.iat[i]:.1f} · {streak} windows")
    # 3 windows in a row (30 min of running) before alerting.
    if streak >= 3 and alerted != fault:
        alerted = fault
        facts = evidence(z_all.iloc[i])
        step_card(s3, "③ EXPLAIN · Gemma", f"✍️ Writing the {language} message locally…", "alert")
        with st.spinner(f"Gemma ({llm}) is writing the message…"):
            out = explain(fault, recent.count(fault) / len(recent), facts, language, llm)
        step_card(s3, "③ EXPLAIN · Gemma", f"📱 Owner messaged in {language}<br>🛠️ Job card sent to technician", "done")
        with msg_ph.container():
            a, b = st.columns(2)
            a.subheader(f"📱 Message to the owner ({language})")
            a.markdown(chat_bubble(out["resident_message"], language), unsafe_allow_html=True)
            b.subheader("🛠️ Technician job card")
            card = out["job_card"]
            label = "most likely" if card["confidence"] < 0.6 else "diagnosis"
            b.markdown(f"**{card['fault']}** · {label} ({card['confidence']:.0%}) · urgency **{card['urgency']}**")
            b.markdown("**Evidence:** " + "; ".join(card["evidence"] or ["readings far from healthy level"]))
            b.markdown("**Bring:** " + ", ".join(card["bring"]))
            b.markdown("**Check:** " + " · ".join(card["checks"]))
            st.caption(f"Written by {out['llm']} in {out['seconds']} s, locally. Fault and numbers come from "
                       f"the detector and {model_name}, not from the LLM.")
    time.sleep(1 / speed)

chart_ph.line_chart(pd.DataFrame(hist).set_index("time"))
flagged_diag = pd.Series(diag[flag])
days = pd.Series(score.to_numpy(), index=z_all.index).groupby(z_all.index.date).median()
summary = (f"**Replay finished.** Over {len(days)} days the AC was outside its normal range on "
           f"**{(days > det['tau']).sum()} days**.")
if len(flagged_diag):
    top = flagged_diag.value_counts(normalize=True)
    summary += f" Most common diagnosis: **{top.index[0]}** ({top.iloc[0]:.0%} of flagged readings)."
status_ph.info(summary)
# Final state of card 1 = the verdict for the whole replay, not just the last 10 minutes.
bad_days = int((days > det["tau"]).sum())
if bad_days:
    step_card(s1, "① DETECT · replay summary", f"⚠️ Outside normal on <b>{bad_days} of {len(days)} days</b>", "alert")
else:
    step_card(s1, "① DETECT · replay summary", f"✅ Normal on all {len(days)} days", "ok")
