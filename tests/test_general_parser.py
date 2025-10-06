import json
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from surrogate_factory.parsers.general_parser import load_standard_format
from surrogate_factory.parsers.esss_json_parser import load_data as load_esss_data

def test_load_standard_format(tmp_path):
    """
    Testa o parser generalista (base_parser) que lê o formato CSV + JSON.
    """
    features_csv_content = """run_id,FeatureA,FeatureB
1,0.5,Red
2,0.8,Blue
"""
    targets_json_content = """{
  "1": [10.1, 10.2, 10.3],
  "2": [20.4, 20.5, 20.6]
}"""
    
    features_file = tmp_path / "features.csv"
    targets_file = tmp_path / "targets.json"
    features_file.write_text(features_csv_content)
    targets_file.write_text(targets_json_content)

    df = load_standard_format(features_path=features_file, targets_path=targets_file)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert 'outputs' in df.columns
    assert df['FeatureA'].iloc[0] == 0.5
    assert df['outputs'].iloc[1] == [20.4, 20.5, 20.6]


def test_load_esss_data(tmp_path):
    """
    Testa o parser específico para os dados da ESSS.
    """
    run_specs_content = json.dumps([
        {"run_number": 1, "metrics": [{"DiscreteMetric": {"caption": "VarA", "value": "Low"}}]},
        {"run_number": 2, "metrics": [{"DiscreteMetric": {"caption": "VarA", "value": "High"}}]}
    ])
    metadata_content = json.dumps({
        "input_metrics": [{"InternalOptimizerDiscreteValues": {"caption": "VarA", "valid_values": ["Low", "High"]}}]
    })

    sa_summary_1_content = json.dumps({"results": [{"caption": "Curve1", "image": [1, 2, 3]}]})
    sa_summary_2_content = json.dumps({"results": [{"caption": "Curve1", "image": [4, 5, 6]}]})

    run_specs_file = tmp_path / "project.runs-specs.json"
    metadata_file = tmp_path / "project.global-sa-input.json"
    run_specs_file.write_text(run_specs_content)
    metadata_file.write_text(metadata_content)

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "project_R00001").mkdir()
    (results_dir / "project_R00002").mkdir()
    (results_dir / "project_R00001" / "sa_summary.json").write_text(sa_summary_1_content)
    (results_dir / "project_R00002" / "sa_summary.json").write_text(sa_summary_2_content)

    df, metadata = load_esss_data(
        run_specs_path=run_specs_file,
        results_base_dir=results_dir,
        metadata_path=metadata_file
    )

    assert len(df) == 2
    assert "VarA" in df.columns
    assert "outputs" in df.columns
    assert df['outputs'].iloc[0] == [1, 2, 3]
    assert "VarA" in metadata