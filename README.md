# SurrogateFactory

A Python package for creating surrogate models (metamodels) that approximate expensive computational simulations. Implements multiple surrogate modelling approaches with support for multi-output time-series prediction and a REST API for model serving.

## Quick Start

```python
from surrogate_factory import SurrogateFactory, ModelType

factory = SurrogateFactory(ModelType.KRIGING)  # or RBF, NN
factory.load_data("features.csv", "targets.json", "metadata.json")
factory.train(test_size=0.2, val_size=0.1)
predictions = factory.predict(new_features_df)

factory.save_model("model.pkl")
loaded = SurrogateFactory.load_model("model.pkl")
```

## Project Structure

```
src/surrogate_factory/
├── __init__.py             # Public API: SurrogateFactory, ModelType
├── factory.py              # Core factory (load, train, predict, persist)
├── api/
│   └── app.py              # FastAPI REST endpoint for inference
├── models/
│   ├── kriging_model.py    # Kriging (Gaussian Process) via SMT
│   ├── rbf_model.py        # Radial Basis Function via SMT
│   └── neural_network.py   # MLP regressor via scikit-learn
├── parsers/
│   ├── general_parser.py   # CSV + JSON loader
│   └── esss_json_parser.py # ESSS simulation format loader
├── preprocessing/
│   └── processor.py        # Scaling, PCA compression, feature encoding
└── evaluation/
    ├── metrics.py          # RMSE, R²
    └── plotting.py         # Comparison plots
tests/                      # pytest suite
k8s/                        # Kubernetes manifests
Dockerfile                  # Multi-stage production image
docker-compose.yaml         # Local dev stack
Makefile                    # Common dev commands
.github/workflows/ci.yaml  # CI pipeline (lint, test, build, docker)
```

## Data Format

### features.csv
| Column | Description |
|--------|-------------|
| `run_id` / `run_number` | Unique simulation run identifier |
| *feature columns* | Numerical or categorical input parameters |

### targets.json
```json
{ "1": {"qoi_name": [t1, t2, ...]}, "2": {"qoi_name": [t1, t2, ...]} }
```
Or flat: `{ "1": [t1, t2, ...] }`

### metadata.json (optional)
```json
{ "CategoryColumn": ["ValueA", "ValueB", "ValueC"] }
```

## Model Types

| Type | Backend | Best For |
|------|---------|----------|
| `KRIGING` | SMT `KRG` | Small datasets, smooth response surfaces |
| `RBF` | SMT `RBF` | Balanced speed / accuracy |
| `NN` | sklearn `MLPRegressor` | Larger datasets, complex non-linearities |

## REST API

The package ships with a FastAPI application for serving predictions.

```bash
# Run locally
SURROGATE_MODEL_PATH=model.pkl uvicorn surrogate_factory.api.app:app

# Endpoints
GET  /healthz        # Kubernetes liveness / readiness probe
POST /predict        # Run inference (JSON body)
GET  /model/info     # Model metadata
```

## Docker & Kubernetes

```bash
# Build and run via Docker Compose
make docker-run

# Or deploy to a Kubernetes cluster
kubectl apply -f k8s/
```

## Common Commands

```bash
make install       # Install with dev deps
make lint          # Ruff + mypy
make format        # Auto-format
make test          # Run pytest
make build         # Build sdist + wheel
make docker        # Build Docker image
make clean         # Remove artefacts
```

## Dependencies

- **Core**: numpy, pandas, scikit-learn, smt, matplotlib, seaborn
- **API**: fastapi, uvicorn
- **Dev**: pytest, pytest-cov, httpx, ruff, mypy
