## 1. Simulated unit, later weeks it never saw (detect, then diagnose)

| model                         |   accuracy |   macro_F1 |   faults_detected |   false_alarms |   seconds |
|:------------------------------|-----------:|-----------:|------------------:|---------------:|----------:|
| Rules of thumb                |      0.410 |      0.305 |             0.630 |          0.060 |     0.000 |
| Gradient boosting (2700 rows) |      0.564 |      0.533 |             0.561 |          0.020 |     1.900 |
| TabPFN (2700 rows)            |      0.573 |      0.550 |             0.561 |          0.020 |   237.100 |

## 2. Share of windows correctly diagnosed, by severity

| label                                           | severity         |   Rules |   Gradient boosting |   TabPFN |
|:------------------------------------------------|:-----------------|--------:|--------------------:|---------:|
| Blocked liquid line / filter-drier              | 1 bar drop       |    0.00 |                0.00 |     0.00 |
| Blocked liquid line / filter-drier              | 10 bar drop      |    0.00 |                0.94 |     0.95 |
| Blocked liquid line / filter-drier              | 4 bar drop       |    0.00 |                0.03 |     0.05 |
| Blocked liquid line / filter-drier              | 8 bar drop       |    0.00 |                0.64 |     0.65 |
| Dirty outdoor (condenser) coil                  | 10% less airflow |    0.00 |                0.00 |     0.00 |
| Dirty outdoor (condenser) coil                  | 20% less airflow |    0.79 |                0.01 |     0.01 |
| Dirty outdoor (condenser) coil                  | 30% less airflow |    0.91 |                0.70 |     0.70 |
| Dirty outdoor (condenser) coil                  | 40% less airflow |    0.99 |                0.99 |     0.99 |
| Dirty outdoor (condenser) coil                  | 50% less airflow |    1.00 |                0.97 |     0.97 |
| Low indoor airflow (dirty filter / indoor coil) | 10% less airflow |    1.00 |                1.00 |     1.00 |
| Low indoor airflow (dirty filter / indoor coil) | 20% less airflow |    1.00 |                1.00 |     1.00 |
| Low indoor airflow (dirty filter / indoor coil) | 30% less airflow |    1.00 |                1.00 |     1.00 |
| Low indoor airflow (dirty filter / indoor coil) | 40% less airflow |    1.00 |                1.00 |     1.00 |
| Low indoor airflow (dirty filter / indoor coil) | 50% less airflow |    1.00 |                1.00 |     1.00 |
| Low refrigerant (gas leak)                      | -10%             |    0.14 |                0.11 |     0.12 |
| Low refrigerant (gas leak)                      | -15%             |    0.12 |                0.07 |     0.07 |
| Low refrigerant (gas leak)                      | -20%             |    0.37 |                0.45 |     0.45 |
| Normal                                          | -                |    0.94 |                0.98 |     0.98 |
| Restricted suction line                         | 1 bar drop       |    0.00 |                0.01 |     0.04 |
| Restricted suction line                         | 3 bar drop       |    0.00 |                1.00 |     1.00 |
| Restricted suction line                         | 6 bar drop       |    0.00 |                1.00 |     1.00 |
| Restricted suction line                         | 9 bar drop       |    0.00 |                1.00 |     1.00 |
| Too much refrigerant (overcharge)               | +10%             |    0.00 |                0.04 |     0.12 |
| Too much refrigerant (overcharge)               | +15%             |    0.00 |                0.02 |     0.01 |
| Too much refrigerant (overcharge)               | +20%             |    0.00 |                0.14 |     0.22 |

## 3. Fault levels never shown in training (condfouling30, evapfouling30, liquidpipe04bar, overcharge15, suctionpipe03bar, undercharge15)

| model             |   correct_diagnosis |
|:------------------|--------------------:|
| Gradient boosting |               0.470 |
| TabPFN            |               0.472 |

## 4. Real rooftop unit (Colchester, CT), diagnoser trained only on simulation

Daily alarms at different alarm levels (anomaly = RMS of z-scores; healthy calibration ~1):

| circuit   | period                                             |   days |   alarm days at >2 |   alarm days at >3 |   alarm days at >5 |   alarm days at >8 |
|:----------|:---------------------------------------------------|-------:|-------------------:|-------------------:|-------------------:|-------------------:|
| A         | Jun 2020 (documented 40% undercharge on circuit B) |     20 |                 11 |                  4 |                  0 |                  0 |
| A         | Jun-Sep 2021 (dataset says unfaulted)              |    109 |                 44 |                 10 |                  2 |                  0 |
| B         | Jun 2020 (documented 40% undercharge on circuit B) |     16 |                 16 |                 16 |                 12 |                  9 |
| B         | Jun-Sep 2021 (dataset says unfaulted)              |     53 |                 53 |                 53 |                 53 |                  2 |

| circuit   | period                                             | days alarmed (daily median > 5)   |   windows flagged | most common diagnosis (simulation only)   |   flagged windows called low refrigerant |
|:----------|:---------------------------------------------------|:----------------------------------|------------------:|:------------------------------------------|-----------------------------------------:|
| A         | Jun 2020 (documented 40% undercharge on circuit B) | 0/20                              |              0.04 | Too much refrigerant (overcharge)         |                                     0    |
| A         | Jun-Sep 2021 (dataset says unfaulted)              | 2/109                             |              0.02 | Low refrigerant (gas leak)                |                                     0.54 |
| B         | Jun 2020 (documented 40% undercharge on circuit B) | 12/16                             |              0.7  | Blocked liquid line / filter-drier        |                                     0    |
| B         | Jun-Sep 2021 (dataset says unfaulted)              | 53/53                             |              1    | Blocked liquid line / filter-drier        |                                     0    |

Average z-scores (distance from each circuit's own healthy level):

| circuit   | period                                             |   z_SUCT_P |   z_DISC_P |   z_SUCT_T |   z_LIQ_T |   z_DISC_T |   z_SAT |   z_SA_FLOW |
|:----------|:---------------------------------------------------|-----------:|-----------:|-----------:|----------:|-----------:|--------:|------------:|
| A         | Jun 2020 (documented 40% undercharge on circuit B) |        0.9 |       -0.3 |       -0.2 |      -2.9 |       -1   |     1.8 |         1.3 |
| A         | Jun-Sep 2021 (dataset says unfaulted)              |       -0.6 |       -0.9 |       -0.1 |       0.6 |       -0.5 |    -0.3 |        -1.4 |
| B         | Jun 2020 (documented 40% undercharge on circuit B) |      -11.6 |       -6.7 |        6.6 |      -1.3 |       11.9 |     2.1 |         3   |
| B         | Jun-Sep 2021 (dataset says unfaulted)              |      -14.8 |       -3.5 |        5.2 |      -0.2 |       10.3 |     0.6 |        -1.8 |

## 5. Few-shot: add 2 technician-confirmed days of the real fault to the context

| model             | context                             |   B fault days 3-26: called low refrigerant |   B 2021: called low refrigerant |   A 2021: called normal |
|:------------------|:------------------------------------|--------------------------------------------:|---------------------------------:|------------------------:|
| Gradient boosting | simulation only                     |                                        0    |                             0    |                    0.98 |
| Gradient boosting | +2 confirmed days of the real fault |                                        0.53 |                             0.95 |                    0.98 |
| TabPFN            | simulation only                     |                                        0    |                             0    |                    0.98 |
| TabPFN            | +2 confirmed days of the real fault |                                        0.69 |                             1    |                    0.98 |
