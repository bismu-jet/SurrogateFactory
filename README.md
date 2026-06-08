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
runs. The result is a negative R², noisy curves, and physically impossible
oscillations — a metamodel useless for engineering.

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

This turns an impossible `4 → 500` regression into a trivial `4 → 4` one. On the
UNISIM-IV benchmark this lifts the held-out **global R² from negative to ≈ 1.0**
for all three model types.

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
| RBF | 0.999759 | 54 s | 0.054 s |

Reproduce with:

```bash
python Scripts/run_caso3_unisim.py        # full deliverable under Results/Caso3 UNISIM
python Scripts/run_caso3_unisim.py --skip-cv
```

Artefacts written per run: `benchmark_metrics.csv` (global R², RMSE, MAE, NRMSE,
per-QoI; train/inference timing), grouped bar charts, `timing_by_model.png`,
`pca_cumulative_variance.png`, parity plots, multi-model overlays, Kriging ±2σ
uncertainty bands, `cv_metrics.csv` (K-fold), and `WHY_IT_WORKED.md`.

> **Note on metrics.** The headline number is the **pooled/global R²** (variance
> explained across all test points). The per-run R² (one curve at a time) is
> deliberately also reported but is brutally harsh on near-flat curves — it is a
> diagnostic, not the headline.
>
> **Honest limits.** Predictability is QoI/dataset dependent. H2S *flow rate* on
> the UNISIM / 3-phase-LGR campaigns reaches ≈ 1.0; the harder *H2S water
> concentration* on the 10-input CaseModel campaign tops out at ≈ 0.6 (0.37 CV) —
> an intrinsic data limit, not a method gap.

## Quick Start

```python
from surrogate_factory import SurrogateFactory, ModelType

factory = SurrogateFactory(ModelType.KRIGING)  # or RBF, NN
factory.load_data("features.csv", "targets.json", "metadata.json")
factory.train(test_size=0.2, val_size=0.1)

predictions = factory.predict(new_features_df)            # {qoi: (n, n_steps)} physical
mean, std   = factory.predict_with_uncertainty(df)["q"]   # Kriging only
cv_metrics  = factory.cross_validate(k=5)                 # K-fold mean±std

factory.save_model("model.pkl")                # slim (inference-only)
factory.save_model("model.pkl", include_eval_data=True)   # keep test data too
```

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
│   ├── general_parser.py   # CSV + JSON loader
│   ├── esss_json_parser.py # ESSS sa_summary.json loader
│   └── esss_hdf5_parser.py # ESSS result.sourres (HDF5) transient-log loader
├── preprocessing/processor.py   # feature encoding, scaling, PCA + latent-standardisation
└── evaluation/
    ├── metrics.py          # RMSE, R², MAE, NRMSE, max-error
    ├── plotting.py         # comparison / parity / PCA-variance / timing plots
    ├── benchmark.py        # train all models on one split → metrics table (+ timing)
    └── reporting.py        # full deliverable: benchmark + ensemble + plots + CV + PCA-speed
Scripts/                    # run_caso3_unisim.py, run_pfc_concentration.py, benchmark_models.py
tests/  k8s/  Dockerfile  docker-compose.yaml  Makefile  .github/workflows/ci.yaml
```

## Data Formats

**features.csv** — a `run_id`/`run_number` column plus numeric/categorical inputs.

**targets.json** — `{ "1": {"qoi": [t1, t2, ...]} }` (or flat `{ "1": [t1, ...] }`).

**metadata.json** (optional) — `{ "CategoryColumn": ["ValueA", "ValueB"] }` for
ordinal encoding of categorical inputs.

ESSS data is also read natively: per-run `sa_summary.json` (flow-rate QoIs) via
`esss_json_parser`, and any logged scalar time-series (e.g. *H2S Water
Concentration*) from the raw `result.sourres` HDF5 via `esss_hdf5_parser`.

## Model Types

| Type | Backend | Notes |
|------|---------|-------|
| `KRIGING` | SMT `KRG` | Most accurate on smooth, densely-sampled spaces; slowest to train (one GP per component) |
| `RBF` | SMT `RBF` | Fastest to train; native multi-output |
| `NN` | sklearn `MLPRegressor` | Manual early stopping on the supplied validation set |

All three are trained on the **same split** so the comparison is fair; an
equal-weight **ensemble** is also reported and is often the most robust.

## REST API

```bash
SURROGATE_MODEL_PATH=model.pkl uvicorn surrogate_factory.api.app:app
# GET /healthz   POST /predict   GET /model/info
```

## Docker, Kubernetes & Dev

```bash
make docker-run                 # build + run via docker-compose
kubectl apply -f k8s/           # Deployment + Service + ConfigMap + PVC
make install lint test          # dev deps · ruff+mypy · pytest
```

Reproducible builds use a hashed `requirements.lock` (pip-compile); CI runs
lint + mypy + pytest (coverage gate) across Python 3.10–3.12 and builds the image.

## Dependencies

- **Core**: numpy, pandas, scikit-learn, smt, h5py, matplotlib, seaborn
- **API**: fastapi, uvicorn · **Dev**: pytest, pytest-cov, httpx, ruff, mypy, pip-tools
