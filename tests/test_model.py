import os
import pandas as pd
import json


def test_data_schema_basic():
    data_path = "data/prepared/train.csv"
    assert os.path.exists(data_path), f"Data not found: {data_path}"
    df = pd.read_csv(data_path)
    required_cols = {"Churn"}
    missing = required_cols - set(df.columns)

    assert not missing, f"Missing columns: {missing}"
    assert df["Churn"].notna().all(), "Churn contains NaN"
    assert df.shape[0] >= 100, "Dataset too small"


def test_artifacts_exist():
    assert os.path.exists("data/models/model.pkl"), "model.pkl not found"
    assert os.path.exists("metrics.json"), "metrics.json not found"
    assert os.path.exists("data/models/confusion_matrix.png"), "confusion_matrix.png not found"


def test_quality_gate_f1():
    threshold = float(os.getenv("F1_THRESHOLD", "0.50"))

    with open("metrics.json", "r") as f:
        metrics = json.load(f)

    f1 = float(metrics["test_f1"])

    assert f1 >= threshold, f"Quality Gate not passed: f1={f1} < {threshold}"