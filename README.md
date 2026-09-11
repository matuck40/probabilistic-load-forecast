# Probabilistic load forecasting: does a boosted tree beat the ruler?

Day-ahead hourly forecasts of Brazilian electricity load, with prediction
intervals, evaluated the only way that means anything for a time series:
walk-forward, on data the model had never seen at the moment of the forecast.

Five models compete on the same folds, the same features and the same metrics:
two seasonal naive baselines, LightGBM in two loss functions, and Chronos, a
time-series foundation model, zero-shot.

**Data.** ONS *Curva de Carga Horária*, Southeast/Centre-West subsystem,
1 Jan 2021 to 8 Sep 2026. 49,848 hourly observations. Public, CC-BY.
The SHA-256 of every input file is recorded in `results/run.json`.

---

## Results

Mean across six quarterly walk-forward folds. WAPE is Σ|y−ŷ|/Σ|y|.
Coverage is the share of observations falling inside the P10–P90 interval;
**nominal coverage is 80%**.

| Model | WAPE | MAE (MWmed) | Pinball | Coverage | Fit time |
|---|---|---|---|---|---|
| Seasonal naive, 24 h | 5.77% | 2557 | 906 | **79.4%** | 0 s |
| Seasonal naive, 168 h | 5.31% | 2366 | 774 | 73.3% | 0 s |
| **LightGBM, L1** | **2.11%** | **940** | **327** | 62.4% | 732 s |
| LightGBM, L2 | 2.15% | 961 | 331 | 62.3% | 722 s |
| Chronos, zero-shot | 2.39% | 1057 | 360 | 68.3% | 164 s |

### Three things worth saying out loud

**1. The gain over the ruler is real and it is large.** LightGBM with an L1
objective cuts WAPE from 5.31% to 2.11%, a 60% reduction against the stronger
of the two naive baselines. That gap holds in all six folds, not on average
only.

**2. A foundation model with no training at all beats both baselines.**
Chronos never sees the training set. It reaches 2.39% WAPE, within 13% of the
tuned LightGBM, in a fifth of the wall-clock time. If the task were to ship
something useful this week, the zero-shot model would be a defensible choice.

**3. The sharpest point forecast has the worst-calibrated intervals, and this
is the finding that matters.** The P10–P90 band is supposed to contain 80% of
observations. LightGBM delivers 62%. Chronos delivers 68%. The seasonal naive
of 24 hours, the crudest model here, delivers 79.4%, almost exactly nominal.

Ranking models by point error alone would pick the model whose uncertainty
estimate is least trustworthy. Any decision downstream that consumes the
interval rather than the point, reserve sizing, risk limits, procurement
cover, would be badly served by that choice. Quantile regression minimises
pinball loss; it is not thereby calibrated, and nothing in the training
objective makes it so.

Recalibration is not implemented here. It is named as the next step rather
than quietly performed, so the uncalibrated number stays on the record.

---

## Method

**Evaluation.** Rolling-origin walk-forward. The training window expands, the
test window advances one quarter at a time, six folds from 2025 Q1 to 2026 Q2.
2026 Q3 is excluded because ONS revises recently published data, and scoring a
model against numbers that later change is scoring noise.

**Horizons.** The forecast is issued at 00:00 of day D. The last observation
available is 23:00 of day D−1, so horizon *h* targets 00:00 + (h−1) hours.
Twenty-four horizons, predicted directly, one model per lag signature. Not
recursive: recursive forecasting feeds a model its own errors.

**Features.** Hour-by-hour lags for the short horizons, the same hour on
previous days, the mean of the same hour over the previous seven days, plus
calendar position and Brazilian holiday flags including bridge days. Calendar
position is encoded as raw integers, not sine and cosine: trees split on
thresholds, and a circular transform only blurs where the cut should fall.

**Leakage.** Every lag list passes through one filter that keeps only lags
observable at that horizon, and is then re-verified by an independent check
that raises on violation. Two guards for one rule, because the failure is
silent and flatters the result. A separate test asserts that `predict()`
cannot see past the fold's test boundary.

**Tuning.** Five hyperparameters were tuned on a temporal holdout carved out of
the training window. The gain did not survive scrutiny and the tuned
parameters were **not** adopted; the reasoning is recorded in the commit
history rather than deleted.

---

## Reproduce

Python 3.12. The run on record used 3.12.12; library versions and platform are
captured in `results/run.json`.

```bash
git clone https://github.com/matuck40/probabilistic-load-forecast
cd probabilistic-load-forecast

make setup      # virtualenv and pinned dependencies
make all        # data, lint, tests, five models, figures
```

`make all` runs the whole chain in order: download and audit the ONS CSVs,
lint, the test suite, then the five models and the figures. Expect roughly
half an hour of compute after the download; the two LightGBM runs account for
almost all of it.

To run one piece at a time:

```bash
make data                            # download.py, then audit.py
make lint
make test                            # fast tests, then the Chronos test alone
make clean                           # caches only; keeps .venv and data/raw
.venv/bin/python -m src.run --model lightgbm_l1
.venv/bin/python -m src.run --figures
```

`--model` accepts `naive24`, `naive168`, `lightgbm_l1`, `lightgbm_l2`,
`chronos` or `all`.

Raw data is downloaded, never committed: `data/manifest.json` holds the source
URL, licence and expected checksum of each file, and `data/audit.py` verifies
them. The seed is fixed at 42.

The test suite splits in two on purpose. torch and LightGBM each load their own
OpenMP runtime and crash when they share a process, and pytest imports every
test module during collection, so the Chronos test runs in a separate pytest
invocation.

## Limitations

- One subsystem. The Southeast/Centre-West only; no claim is made about the
  other three.
- No exogenous variables. Temperature is the obvious omission and would very
  likely move the error, particularly in the hours where it is worst.
- Intervals are uncalibrated, as stated above.
- Chronos is evaluated zero-shot. Fine-tuning was not attempted.

---

## Layout

```
src/       dataset, features, splits, models, metrics, evaluation, report
tests/     13 test modules, including the leakage and boundary guards
results/   metrics.json, error_analysis.json, tuning.json, run.json, figures/
data/      download script and manifest; raw files are gitignored
```

MIT licensed.
