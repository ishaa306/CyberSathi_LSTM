"""
Automated Test Suite for CyberSaathi LSTM Evaluation Engine & Endpoints.
Verifies:
1. Default unavailable state when no test dataset exists (honest reporting).
2. Reproducible evaluation pipeline on held-out test data (all 14 classes, confusion matrix, zero-division).
3. API endpoints: /api/ml/evaluation and /api/ml/comparison.
4. Error guards for invalid/empty datasets and missing columns.
"""

import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.ml.lstm_classifier import get_lstm_classifier
from app.ml.evaluate_lstm import (
    evaluate_dataset,
    get_evaluation_results,
    detect_columns,
)


@pytest.fixture
def client():
    return TestClient(app)


# ─── PART 1: Default Unavailable State Tests ─────────────────────────────────

def test_evaluation_unavailable_when_no_heldout_dataset(tmp_path, monkeypatch):
    """
    Verify that when no evaluation cache or held-out test dataset exists,
    the system reports 'unavailable' honestly and explains what is required.
    """
    # Point cache paths to empty temp directory
    dummy_cache = tmp_path / "non_existent_eval.json"
    monkeypatch.setattr("app.ml.evaluate_lstm.DEFAULT_CACHE_PATHS", [dummy_cache])

    result = get_evaluation_results()

    assert result["status"] == "unavailable"
    assert "model_info" in result
    assert result["model_info"]["model_name"] == "CyberSaathi LSTM"
    assert result["model_info"]["max_len"] == 200
    assert result["model_info"]["num_classes"] == 14
    assert len(result["model_info"]["classes"]) == 14
    assert result["model_info"]["is_loaded"] is True
    assert result["model_info"]["is_active_in_workflow"] is True

    eval_data = result["evaluation"]
    assert eval_data["status"] == "unavailable"
    assert "held-out test dataset required" in eval_data["reason"].lower()
    assert eval_data["metrics"] is None
    assert eval_data["dataset_size"] is None
    assert eval_data["confusion_matrix"] is None
    assert len(eval_data["per_class"]) == 0
    assert "python -m app.ml.evaluate_lstm" in eval_data["instructions"]


def test_api_evaluation_endpoint_unavailable_state(client, tmp_path, monkeypatch):
    """
    Verify GET /api/ml/evaluation returns 200 with the unavailable schema
    when no test dataset has been evaluated.
    """
    dummy_cache = tmp_path / "non_existent_eval.json"
    monkeypatch.setattr("app.ml.evaluate_lstm.DEFAULT_CACHE_PATHS", [dummy_cache])

    res = client.get("/api/ml/evaluation")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "unavailable"
    assert data["model_info"]["model_name"] == "CyberSaathi LSTM"
    assert data["evaluation"]["status"] == "unavailable"


# ─── PART 2: Evaluation Pipeline on Test Data ────────────────────────────────

def test_column_detection_helpers():
    """Verify robust detection of text and category columns across common naming conventions."""
    df1 = pd.DataFrame({"complaint_text": ["abc"], "crime_category": ["def"]})
    assert detect_columns(df1) == ("complaint_text", "crime_category")

    df2 = pd.DataFrame({"text": ["abc"], "category": ["def"]})
    assert detect_columns(df2) == ("text", "category")

    df3 = pd.DataFrame({"description": ["abc"], "label": ["def"]})
    assert detect_columns(df3) == ("description", "label")

    with pytest.raises(ValueError, match="Could not find a text column"):
        detect_columns(pd.DataFrame({"foo": [1], "category": ["x"]}))

    with pytest.raises(ValueError, match="Could not find a category/label column"):
        detect_columns(pd.DataFrame({"text": ["foo"], "bar": [1]}))


def test_evaluate_dataset_computation(tmp_path):
    """
    Verify end-to-end evaluation using a held-out test dataframe.
    Ensures:
    - Scikit-learn metrics calculated (Accuracy, Macro/Weighted Precision/Recall/F1)
    - All 14 classes appear in per-class report
    - Confusion matrix is 14x14
    - Classes with 0 samples have support=0 and precision/recall/f1=0.0 (safe zero-division)
    - Low support flag (<5 samples) works
    """
    clf = get_lstm_classifier()
    assert clf.is_loaded is True

    # Construct a test dataset covering selected NCRP categories
    test_samples = [
        {"text": "Unauthorized transaction of Rs 50000 debited via fake UPI link", "category": "Online Financial Fraud"},
        {"text": "Scammer called pretending to be SBI manager and took my OTP", "category": "Online Financial Fraud"},
        {"text": "Stole Rs 25000 using unauthorized net banking transfer", "category": "Online Financial Fraud"},
        {"text": "Fake profile made on Instagram using my photos and harassing friends", "category": "Online and Social Media Related Crime"},
        {"text": "Someone hacked my social media account and is demanding money", "category": "Online and Social Media Related Crime"},
        {"text": "Transferred 1.5 BTC to a fraudulent crypto doubling scheme", "category": "Cryptocurrency Crime"},
        {"text": "Ransomware locked all our database files demanding bitcoin", "category": "Ransomware"},
        {"text": "Denial of service attack brought down our company web server", "category": "Cyber Attack/ Dependent Crimes"},
    ]
    df = pd.DataFrame(test_samples)

    cache_file = tmp_path / "test_eval_output.json"
    result = evaluate_dataset(df, save_cache=True, cache_path=cache_file)

    # 1. Status & Overall metrics
    assert result["status"] == "available"
    assert result["dataset_size"] == 8
    metrics = result["metrics"]
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_precision"] <= 1.0
    assert 0.0 <= metrics["macro_recall"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert 0.0 <= metrics["weighted_f1"] <= 1.0

    # 2. All 14 classes MUST be present
    per_class = result["per_class"]
    assert len(per_class) == 14
    class_names = [p["category"] for p in per_class]
    assert class_names == clf.classes

    # 3. Check specific class supports
    fin_fraud = [p for p in per_class if p["category"] == "Online Financial Fraud"][0]
    assert fin_fraud["support"] == 3
    assert fin_fraud["is_low_support"] is True  # 3 < 5

    # Check a zero-support class (e.g. Cyber Terrorism)
    cyber_terr = [p for p in per_class if p["category"] == "Cyber Terrorism"][0]
    assert cyber_terr["support"] == 0
    assert cyber_terr["precision"] == 0.0
    assert cyber_terr["recall"] == 0.0
    assert cyber_terr["f1_score"] == 0.0
    assert cyber_terr["is_low_support"] is True

    # 4. Confusion matrix validation
    cm_obj = result["confusion_matrix"]
    assert len(cm_obj["labels"]) == 14
    assert len(cm_obj["matrix"]) == 14
    for row in cm_obj["matrix"]:
        assert len(row) == 14
        assert all(isinstance(val, int) for val in row)
    # Sum of confusion matrix entries must equal total sample size (8)
    assert sum(sum(row) for row in cm_obj["matrix"]) == 8

    # 5. Cache persistence validation
    assert cache_file.exists()


def test_api_evaluation_endpoint_available_state(client, tmp_path, monkeypatch):
    """
    Verify GET /api/ml/evaluation returns the full evaluation metrics
    when a valid evaluation cache exists.
    """
    cache_file = tmp_path / "evaluation_results.json"
    dummy_data = {
        "status": "available",
        "dataset_name": "test_split.csv",
        "dataset_size": 100,
        "evaluated_at": "2026-10-18T12:00:00Z",
        "metrics": {
            "accuracy": 0.88,
            "macro_precision": 0.86,
            "macro_recall": 0.85,
            "macro_f1": 0.855,
            "weighted_f1": 0.875,
        },
        "per_class": [
            {
                "class_index": i,
                "category": f"Cat_{i}",
                "precision": 0.85,
                "recall": 0.85,
                "f1_score": 0.85,
                "support": 10,
                "is_low_support": False,
            }
            for i in range(14)
        ],
        "confusion_matrix": {
            "labels": [f"Cat_{i}" for i in range(14)],
            "matrix": [[0]*14 for _ in range(14)],
        },
    }
    import json
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(dummy_data, f)

    monkeypatch.setattr("app.ml.evaluate_lstm.DEFAULT_CACHE_PATHS", [cache_file])

    res = client.get("/api/ml/evaluation")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "available"
    assert data["model_info"]["model_name"] == "CyberSaathi LSTM"
    assert data["evaluation"]["metrics"]["accuracy"] == 0.88
    assert len(data["evaluation"]["per_class"]) == 14


# ─── PART 3: Backward Compatibility & Guard Tests ────────────────────────────

def test_comparison_endpoint_includes_evaluation(client):
    """Verify /api/ml/comparison still contains backwards-compatible keys and new evaluation fields."""
    res = client.get("/api/ml/comparison")
    assert res.status_code == 200
    data = res.json()
    # Legacy keys preserved
    assert "production_model" in data
    assert "metrics" in data
    assert "best_model" in data
    # New evaluation & model info keys attached
    assert "model_info" in data
    assert "evaluation" in data
    assert data["model_info"]["model_name"] == "CyberSaathi LSTM"
    assert data["model_info"]["num_classes"] == 14


def test_evaluate_dataset_empty_data_raises_error():
    """Verify that passing an empty dataset raises an informative ValueError."""
    empty_df = pd.DataFrame({"text": ["", "  "], "category": ["Online Financial Fraud", "Ransomware"]})
    with pytest.raises(ValueError, match="0 valid samples"):
        evaluate_dataset(empty_df, save_cache=False)
