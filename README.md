# CoolSense: an AC that notices its own fault and explains it in your language

A small box beside an air-conditioning unit learns what *healthy* looks like for
that unit, notices when it drifts, names the likely fault, and tells the
resident or shop owner what is happening in their language. A technician gets a
job card with what to bring. Everything runs locally.

- **Detect:** a healthy-behaviour model per unit turns every reading into a z-score
  ("how far from this unit's normal"). Units, sizes and sensors no longer matter.
- **Diagnose:** TabPFN classifies the fault from the z-scores (gradient boosting as a baseline).
- **Explain:** Gemma, running locally, writes the message. It never chooses the fault or the numbers.

Data: **real** LBNL rooftop-unit fault data (CC BY 4.0, DOI 10.25984/1881324):
a simulated R410A unit with 6 fault types at several severities, and a **real
field unit in Colchester, CT** with a documented 40% refrigerant undercharge.

## Files

| File | What it does |
|---|---|
| `rtu_data.py` | Loads LBNL simulated + field data into 10-minute windows while the compressor runs |
| `baseline.py` | Healthy-behaviour model and z-scores; anomaly score; plain-language evidence |
| `rtu_eval.py` | All experiments; writes `results/rtu_metrics.md` |
| `explain.py` | Gemma message + knowledge-base job card |
| `app.py` | Live Streamlit replay of 4 streams |

## Run it (Windows, PowerShell)

```powershell
cd coolsense
# put the LBNL RTU zip contents here, so these folders exist:
#   lbnl\Simulated RTU\RTU_sim_baseline.csv ...
#   lbnl\Field RTU\Site2_Unfaulted.csv ...
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt     # tabpfn pulls in PyTorch (large)
ollama pull gemma3:1b

python rtu_eval.py                  # ~1 min without TabPFN, longer with it on CPU
streamlit run app.py
```

`python rtu_eval.py --no-tabpfn` runs without TabPFN if it fails to install.

## Results so far (gradient boosting; TabPFN row comes from your laptop run)

**Real unit, no retraining on it.** Alarm when the daily median anomaly > 5:

| Circuit | Period | Days alarmed |
|---|---|---|
| B | June 2020, documented 40% undercharge | 12 / 16 (16/16 at level 3) |
| A | Summer 2021, healthy | 2 / 109 |
| B | Summer 2021, dataset says unfaulted | **53 / 53** |

**Diagnosis on the real fault:**
- Trained on simulation only, it calls the real undercharge a *blocked liquid
  line*. Both starve the indoor coil, so the readings look alike. The job card
  tells the technician the one measurement that separates them (subcooling).
- Add 2 technician-confirmed days of the real fault as extra context, and 95% of
  circuit B's 2021 alarm windows are called *low refrigerant*.

**Simulated unit, later weeks:** 2.6% false alarms. Low indoor airflow and
suction-line restriction (3 bar+) are diagnosed almost every time; dirty
condenser at 30–50% in 78–98% of windows; liquid-line blockage at 8–10 bar in
69–93%. Mild faults, and the simulated over/undercharge (which barely move the
readings in this model), mostly go unnoticed at this alarm level.

## Findings to state carefully

- **Circuit B in 2021.** The dataset labels it unfaulted, but its suction pressure
  is 4.2 bar vs 5.4 bar after the 2020 fix, and discharge temperature is about
  93 °C vs 76 °C: the same direction as the documented undercharge, about half as
  far. It *looks like* the leak came back after a top-up. That is a hypothesis;
  there are no service records to confirm it.
- **Simulation vs reality.** In the simulated unit, undercharge barely moves suction
  pressure; in the real unit it drops sharply. Training only on simulation is not
  enough, which is why the few-shot step matters.
- **Field healthy baseline is short** (Aug–Sep 2020). A year later a healthy circuit
  drifts 2–4x its normal variation, so the field alarm level is 5. The threshold
  sweep is in `results/rtu_metrics.md`.
- **Hardware:** the demo replays recorded data. A real product needs pressure and
  temperature sensors on the unit (common on commercial units, add-on for homes).
- **Licensing:** check TabPFN's licence before selling. Gemma's terms allow commercial use.
