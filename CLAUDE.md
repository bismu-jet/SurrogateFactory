# SurrogateFactory

A Python package for creating surrogate models (metamodels) that approximate expensive computational simulations. Implements multiple surrogate modeling approaches with support for multi-output time-series prediction.

## Quick Start

```python
from surrogate_factory.factory import SurrogateFactory, ModelType

# Initialize factory with model type
factory = SurrogateFactory(ModelType.KRIGING)  # or ModelType.RBF, ModelType.NN

# Load data
factory.load_data("features.csv", "targets.json", "metadata.json")

# Train model (70% train, 10% validation, 20% test)
factory.train(test_size=0.2, val_size=0.1)

# Make predictions
predictions = factory.predict(new_features_df)

# Save/load model
factory.save_model("model.pkl")
loaded_factory = SurrogateFactory.load_model("model.pkl")
```

## Project Structure

```
src/surrogate_factory/
├── factory.py              # Main SurrogateFactory class
├── models/
│   ├── kriging_model.py    # Kriging (Gaussian Process) implementation
│   ├── rbf_model.py        # Radial Basis Function implementation
│   └── neural_network.py   # Neural Network implementation
├── parsers/
│   ├── general_parser.py   # CSV/JSON data parser
│   └── esss_json_parser.py # ESSS simulation format parser
├── preprocessing/
│   └── processor.py        # Feature scaling, PCA compression, data structuring
└── evaluation/
    ├── metrics.py          # RMSE, R² calculations
    └── plotting.py         # Visualization utilities
```

## Data Format

### features.csv
CSV file with columns:
- `run_id` or `run_number`: Unique identifier for each simulation run
- Feature columns: Numerical or categorical input parameters

### targets.json
JSON dictionary mapping run IDs to output vectors:
```json
{
  "1": {"qoi_name": [t1, t2, t3, ...]},
  "2": {"qoi_name": [t1, t2, t3, ...]}
}
```
Or flat format:
```json
{
  "1": [t1, t2, t3, ...],
  "2": [t1, t2, t3, ...]
}
```

### metadata.json (optional)
For categorical features encoding:
```json
{
  "CategoryColumn": ["ValueA", "ValueB", "ValueC"]
}
```

## Model Types

- **KRIGING**: Gaussian Process regression via SMT library. Best for smaller datasets with smooth response surfaces.
- **RBF**: Radial Basis Function interpolation. Good balance of speed and accuracy.
- **NN**: Neural Network with scikit-learn MLPRegressor. Handles larger datasets, uses scaled inputs.

## Key Features

- **PCA Compression**: Automatically compresses high-dimensional time-series outputs using PCA (default: 99.9% variance retention)
- **Train/Val/Test Split**: Configurable data splitting with reproducible random states
- **Evaluation Metrics**: RMSE and R² calculation with optional reference model comparison
- **Model Persistence**: Save/load trained models via pickle

## Running Tests

```bash
pytest tests/ -v
```

## Dependencies

- pandas, numpy
- scikit-learn (preprocessing, PCA, neural networks)
- smt (Kriging, RBF models)
- matplotlib, seaborn (plotting)

## Common Commands

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_factory.py -v

# Install in development mode
pip install -e .
```
