# SurrogateFactory

A Python package for building **surrogate models** (metamodels) that replace
expensive engineering simulations. It predicts full multi-output **time-series**
QoIs (Quantities of Interest) from a handful of design parameters, with Kriging,
RBF and Neural-Network backends, a fair benchmarking harness, uncertainty
quantification, and a REST API for serving.

## The key idea — predict the *compressed* curve, not the raw series

A reservoir simulation emits a long time-series (e.g. **500 time-steps**) per QoI.
Trying to predict those 500 outputs **directly** is the trap: the model fights a
500-dimensional, heavily-correlated output space with only a few hundred training
runs, and nothing stops neighbouring time-steps from disagreeing with each
other.

The insight that makes it work is **PCA on the output time-series**. The clever
bit is *where* PCA is applied: **not on the inputs** (the usual textbook use of
PCA), but on the **output temporal points** — we compress *what we predict*, not
*what we predict from*. Each QoI curve of ~500 time-steps is reduced to a few
latent modes:

1. Those 500 time-points are massively redundant — **1 to 4 principal components
   explain ≥ 99.9 %** of the variance across all simulated curves
   (`pca_cumulative_variance.png`).
2. So we fit a PCA per QoI **over its time-series**, and have the models regress
   only the **K latent scalars**, then reconstruct the full curve with the
   inverse PCA.
3. Because every PCA eigen-curve comes from real (smooth) simulations, *any*
   linear combination is smooth and physical — no high-frequency noise.

The 3–4 input parameters are left untouched; it is the **high-dimensional output**
that PCA tames.

This turns a `4 → 500` regression into a `4 → 4` one. On the UNISIM-IV benchmark
all three model types reach a held-out **global R² ≈ 1.0** (table below).

```
físico → MinMax → PCA → StandardScaler → [model predicts K latent scalars]
                                        → inverse → physical curve
```

### Why it's also *fast*

Kriging trains **one Gaussian Process per latent component**, so training time
scales with the number of components. PCA collapses 500 → ~4 components, so the
same near-perfect accuracy is reached in **seconds instead of hours**
(`pca_speed_impact.csv`, `pca_speed_vs_components.png`): on UNISIM-IV the Kriging
train time grows linearly with components (2→26 s … 32→257 s) while R² stays
pinned at 0.9999. Extrapolated to the raw 500 outputs that would be ~1 h/QoI —
PCA buys orders of magnitude at **no accuracy cost**.

## Results (UNISIM-IV / PFC Case 3, held-out test set)

| Model | best global R² | train | inference (full test set) |
|-------|---------------:|------:|--------------------------:|
| **Kriging** | **0.999996** | 670 s | 0.069 s |
| Ensemble (mean) | 0.999958 | — | — |
| Neural Network | 0.999845 | 35 s | 0.047 s |
| RBF | 0.999759 | 54 s | 0.059 s |

Measured 2026-06-08 with `evaluation.reporting.produce_full_results` on UNISIM-IV
case 3: 76 QoIs, 32 test runs, QoIs trained one after another. The dataset is not
distributed with this repository.

`produce_full_results` writes per run: `benchmark_metrics.csv` (global R², RMSE,
MAE, NRMSE, per-QoI; train/inference timing), grouped bar charts,
`timing_by_model.png`, `pca_cumulative_variance.png`, parity plots, multi-model
overlays, Kriging ±2σ uncertainty bands, `cv_metrics.csv` (K-fold), and
`WHY_IT_WORKED.md`.

> **Note on metrics.** The headline number is the **pooled/global R²** (variance
> explained across all test points). The per-run R² (one curve at a time) is
> deliberately also reported but is brutally harsh on near-flat curves — it is a
> diagnostic, not the headline.
>
> **Honest limits.** Predictability is QoI/dataset dependent. H2S *flow rate* on
> the UNISIM / 3-phase-LGR campaigns reaches ≈ 1.0; the harder *H2S water
> concentration* on the 10-input CaseModel campaign tops out at ≈ 0.6 (0.37 CV) —
> an intrinsic data limit, not a method gap.

## Installation

```bash
pip install -e .            # core: train, evaluate, predict, persist
pip install -e ".[esss]"    # + h5py, for the ESSS result.sourres parser
pip install -e ".[api]"     # + FastAPI server
pip install -e ".[dev]"     # + pytest, ruff, mypy, build (includes api)
```

Python ≥ 3.10. `requirements.lock` pins the core + API stack for the Docker image;
it was compiled with Python 3.13 and its pins need Python ≥ 3.11.

## Quick Start

```python
import pandas as pd
from surrogate_factory import SurrogateFactory, ModelType

factory = SurrogateFactory(ModelType.KRIGING)   # or RBF, NN; n_jobs=-1 trains QoIs in parallel
factory.load_data("features.csv", "targets.json", "metadata.json")
factory.train(test_size=0.2, val_size=0.1)

metrics = factory.evaluate_and_plot(plot_prefix="plots/cmp")   # {qoi: {metric: value}}
new_runs = pd.read_csv("features.csv")
predictions = factory.predict(new_runs)                         # {qoi: (n, n_steps)} physical

qoi = factory.qoi_names[0]
mean, std = factory.predict_with_uncertainty(new_runs)[qoi]    # Kriging only
cv_metrics = factory.cross_validate(k=5)                        # K-fold mean/std; refits

factory.save_model("model.pkl")                                 # slim (inference-only)
factory.save_model("model_eval.pkl", include_eval_data=True)    # keep test data too
factory = SurrogateFactory.load_model("model.pkl")
```

Already have the data in memory? `factory.set_data(features_df, {run_id: {qoi: vector}})`
replaces `load_data`.

## Data Formats

**features.csv** — a `run_id` (or `run_number`) column plus numeric/categorical inputs.

**targets.json** — `{"1": {"qoi_a": [t1, t2, ...], "qoi_b": [...]}}`; each QoI is
trained as its own model. A flat `{"1": [t1, ...]}` is read as one QoI named
`outputs`.

**metadata.json** (optional) — `{"CategoryColumn": ["ValueA", "ValueB"]}` for
ordinal encoding of categorical inputs.

### ESSS studies

The ESSS parsers read a sensitivity-analysis export directly: the
`<project>.runs-specs.json` design table plus a results directory holding one
`<project>_R<NNNNN>/` folder per run. Both return `(features_df, {run_number: {qoi: vector}})`, which goes straight
into `set_data`:

```python
from pathlib import Path
from surrogate_factory.parsers.esss_json_parser import load_specs_and_qois_for_runs
from surrogate_factory.parsers.esss_hdf5_parser import load_specs_and_log_qois_for_runs  # needs [esss]

specs = Path("study/project.runs-specs.json")
results = Path("study/project")      # project_R00001/, project_R00002/, ...
runs = list(range(1, 101))

# QoIs exported to each run's sa_summary.json (e.g. H2S flow rate)
features, qois = load_specs_and_qois_for_runs(specs, results, runs)

# or any logged transient curve from each run's result.sourres (HDF5)
features, qois = load_specs_and_log_qois_for_runs(
    specs, results, runs, quantity_names=["Producer H2S Water Concentration"]
)

factory = SurrogateFactory(ModelType.RBF)
factory.set_data(features, qois)
```

Categorical metadata comes from the study's `global-sa-input.json` via
`esss_json_parser._parse_variable_metadata`; assign it to `factory.metadata`.

## Performance

Training dominates the cost; prediction is milliseconds. On UNISIM-IV case 3
(76 QoIs, see Results) training took 670.7 s (Kriging), 53.7 s (RBF) and 35.1 s
(NN), measured 2026-06-08 with QoIs trained one after another.

| Knob | Where | Effect |
|------|-------|--------|
| `n_jobs` | `SurrogateFactory(model_type, n_jobs=-1)`, `benchmark_models`, `produce_full_results` | Trains QoIs in parallel processes; predictions are identical to `n_jobs=1` (BLAS runs on one thread per QoI either way). The models' own warnings stay in the workers |
| `n_pca_components` | factory attribute, default `0.999` | Variance kept per QoI; Kriging fits one GP per kept component |
| `KRIGING_N_START` | `models/kriging_model.py`, default 5 | Optimiser restarts per GP |
| `KRIGING_POLYNOMIALS` × `KRIGING_CORRELATIONS` | `models/kriging_model.py` | 3 × 3 candidate fits per component |
| `_DEFAULT_NUM_TRIES` | `models/rbf_model.py`, default 100 | `d0` grid size; × 3 polynomial degrees |

## Project Structure

```
src/surrogate_factory/
├── factory.py              # Core: load, train, cross_validate, predict(+uncertainty), persist
├── api/app.py              # FastAPI REST endpoint for inference
├── models/
│   ├── kriging_model.py    # Kriging (GP) via SMT — one model per latent component
│   ├── rbf_model.py        # Radial Basis Function via SMT (native multi-output)
│   └── neural_network.py   # MLP via sklearn, manual early-stopping on the real val set
├── parsers/
│   ├── general_parser.py   # CSV + JSON loader (load_data)
│   ├── esss_json_parser.py # ESSS sa_summary.json loader
│   └── esss_hdf5_parser.py # ESSS result.sourres (HDF5) transient-log loader
├── preprocessing/processor.py   # feature encoding, scaling, PCA + latent-standardisation
└── evaluation/
    ├── metrics.py          # RMSE, R², MAE, NRMSE, max-error
    ├── plotting.py         # comparison / parity / PCA-variance / timing plots
    ├── benchmark.py        # train all models on one split → metrics table (+ timing)
    └── reporting.py        # full deliverable: benchmark + ensemble + plots + CV + PCA-speed
tests/  k8s/  Dockerfile  docker-compose.yaml  Makefile  .github/workflows/ci.yaml
```

## Model Types

| Type | Backend | Notes |
|------|---------|-------|
| `KRIGING` | SMT `KRG` | Most accurate on smooth, densely-sampled spaces; slowest to train (one GP per component) |
| `RBF` | SMT `RBF` | Fastest to train; native multi-output |
| `NN` | sklearn `MLPRegressor` | Manual early stopping on the supplied validation set |

All three are trained on the **same split** so the comparison is fair; an
equal-weight **ensemble** is also reported and is often the most robust.

## Development

```bash
make install lint test build    # dev deps · ruff + mypy · pytest · sdist/wheel
```

CI (`.github/workflows/ci.yaml`) runs lint, mypy and pytest with a coverage gate on
Python 3.10–3.12 for pushes and PRs to `master`, then builds the wheel and the
Docker image.

## Serving

The model can also be served over HTTP; this is not needed for training or
prediction from Python.

- REST: `SURROGATE_MODEL_PATH=model.pkl uvicorn surrogate_factory.api.app:app`
  (`GET /healthz`, `POST /predict`, `GET /model/info`; needs `[api]`).
- Docker: `make docker-run` (image built from `requirements.lock`).
- Kubernetes: manifests in `k8s/`.

## Dependencies

- **Core**: numpy, pandas, scikit-learn, joblib, threadpoolctl, smt, matplotlib, seaborn
- **esss**: h5py · **api**: fastapi, pydantic, uvicorn · **dev**: pytest, pytest-cov,
  httpx, ruff, mypy, build
