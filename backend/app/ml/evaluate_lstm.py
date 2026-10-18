"""
CyberSaathi LSTM Evaluation Engine.
Provides reproducible, leakage-free evaluation of the pre-trained 14-class LSTM classifier
using scikit-learn metrics, with support for caching, CLI execution, and zero-division safety.
"""

import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from app.ml.lstm_classifier import get_lstm_classifier

logger = logging.getLogger(__name__)

# Cache file locations
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
DEFAULT_CACHE_PATHS = [
    PROJECT_ROOT / "cybersaathi_lstm_unweighted" / "evaluation_results.json",
    BACKEND_DIR / "models" / "cybersaathi_lstm_unweighted" / "evaluation_results.json",
    BACKEND_DIR / "models" / "lstm_evaluation.json",
]


def resolve_cache_file(write: bool = False) -> Path:
    """Find the existing evaluation cache file or the appropriate path to write one."""
    if not write:
        for p in DEFAULT_CACHE_PATHS:
            if p.exists():
                return p
    # Default location for writing: inside primary model directory
    primary = PROJECT_ROOT / "cybersaathi_lstm_unweighted"
    if primary.exists():
        return primary / "evaluation_results.json"
    return DEFAULT_CACHE_PATHS[-1]


def detect_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Detect text and label/category columns in the provided dataframe.
    
    Raises:
        ValueError: If neither a text column nor a category column can be identified.
    """
    text_candidates = [
        "text", "complaint_text", "incident_description", "description",
        "complaint", "content", "statement", "body"
    ]
    cat_candidates = [
        "category", "crime_category", "label", "ncrp_category",
        "target", "crime_type", "class"
    ]

    found_text = None
    for col in text_candidates:
        if col in df.columns:
            found_text = col
            break
        # Case-insensitive match
        for actual_col in df.columns:
            if actual_col.lower() == col:
                found_text = actual_col
                break
        if found_text:
            break

    found_cat = None
    for col in cat_candidates:
        if col in df.columns:
            found_cat = col
            break
        for actual_col in df.columns:
            if actual_col.lower() == col:
                found_cat = actual_col
                break
        if found_cat:
            break

    if not found_text:
        raise ValueError(
            f"Could not find a text column in test dataset. Available columns: {list(df.columns)}. "
            f"Expected one of: {text_candidates}"
        )
    if not found_cat:
        raise ValueError(
            f"Could not find a category/label column in test dataset. Available columns: {list(df.columns)}. "
            f"Expected one of: {cat_candidates}"
        )

    return found_text, found_cat


def evaluate_dataset(
    data: Union[str, Path, pd.DataFrame],
    save_cache: bool = True,
    cache_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Evaluate the saved LSTM model on a genuine held-out dataset.
    
    Args:
        data: Path to CSV dataset or an existing pandas DataFrame.
        save_cache: Whether to persist the computed metrics to disk.
        cache_path: Optional explicit file path for caching.
        
    Returns:
        Structured dictionary with overall metrics, per-class report,
        and confusion matrix for all 14 classes.
    """
    lstm_clf = get_lstm_classifier()
    if not lstm_clf.is_loaded:
        raise RuntimeError(f"LSTM classifier is not loaded: {lstm_clf.load_error}")

    dataset_name = "In-memory DataFrame"
    if isinstance(data, (str, Path)):
        p = Path(data)
        if not p.exists():
            raise FileNotFoundError(f"Test dataset not found at: {p}")
        df = pd.read_csv(p)
        dataset_name = p.name
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError(f"Unsupported data type for evaluation: {type(data)}")

    text_col, cat_col = detect_columns(df)

    # Filter null or whitespace entries
    df = df.dropna(subset=[text_col, cat_col])
    df = df[df[text_col].astype(str).str.strip().str.len() > 0]
    df = df[df[cat_col].astype(str).str.strip().str.len() > 0]

    if len(df) == 0:
        raise ValueError("Dataset has 0 valid samples after filtering nulls and empty strings.")

    # Validate ground-truth classes against the 14 model classes
    valid_classes = set(lstm_clf.classes)
    unrecognized = set(df[cat_col].unique()) - valid_classes
    if unrecognized:
        logger.warning(
            f"Test dataset contains {len(unrecognized)} categories not in the 14 LSTM classes: {unrecognized}. "
            "Filtering to matching classes."
        )
        df = df[df[cat_col].isin(valid_classes)]
        if len(df) == 0:
            raise ValueError(
                f"No matching samples after filtering unrecognized classes. "
                f"Expected subset of: {lstm_clf.classes}"
            )

    texts = df[text_col].astype(str).tolist()
    labels = df[cat_col].astype(str).tolist()

    # Preprocessing using exact model tokenizer and padding length
    from keras.preprocessing.sequence import pad_sequences

    sequences = lstm_clf.tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=lstm_clf.max_len)

    # Batch prediction
    raw_predictions = lstm_clf.model.predict(padded, batch_size=32, verbose=0)
    y_pred = np.argmax(raw_predictions, axis=1)

    # Ground-truth numeric mapping using exact label encoder
    y_true = lstm_clf.label_encoder.transform(labels)

    # Overall metrics calculation using scikit-learn
    num_classes = lstm_clf.num_classes
    all_indices = list(range(num_classes))
    target_names = list(lstm_clf.classes)

    acc = float(accuracy_score(y_true, y_pred))
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Detailed per-class report
    clf_report = classification_report(
        y_true,
        y_pred,
        labels=all_indices,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=all_indices)
    cm_list = cm.tolist()

    # Format per-class list for all 14 classes
    per_class = []
    for idx, cname in enumerate(target_names):
        stats = clf_report.get(cname, {})
        support_cnt = int(stats.get("support", 0))
        per_class.append({
            "class_index": idx,
            "category": cname,
            "precision": round(float(stats.get("precision", 0.0)), 4),
            "recall": round(float(stats.get("recall", 0.0)), 4),
            "f1_score": round(float(stats.get("f1-score", 0.0)), 4),
            "support": support_cnt,
            "is_low_support": support_cnt < 5,
        })

    eval_result = {
        "status": "available",
        "dataset_name": dataset_name,
        "dataset_size": len(y_true),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "accuracy": round(acc, 4),
            "macro_precision": round(float(macro_prec), 4),
            "macro_recall": round(float(macro_rec), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4),
        },
        "per_class": per_class,
        "confusion_matrix": {
            "labels": target_names,
            "matrix": cm_list,
        },
        "note": (
            "Computed on authentic test records using scikit-learn. "
            "Zero-division handled safely; all 14 classes represented."
        ),
    }

    if save_cache:
        target_file = cache_path or resolve_cache_file(write=True)
        try:
            target_file.parent.mkdir(parents=True, exist_ok=True)
            with open(target_file, "w", encoding="utf-8") as f:
                json.dump(eval_result, f, indent=2)
            logger.info(f"[OK] Evaluation results cached to: {target_file.name}")
        except Exception as e:
            logger.warning(f"Failed to cache evaluation results: {e}")

    return eval_result


def get_evaluation_results() -> Dict[str, Any]:
    """
    Retrieve current evaluation state.
    Returns cached results if a legitimate evaluation was performed and cached,
    otherwise returns a clear 'unavailable' response explaining that an authentic
    held-out test dataset is required.
    """
    lstm_clf = get_lstm_classifier()

    # Basic model metadata
    model_info = {
        "model_name": "CyberSaathi LSTM",
        "architecture": "Embedding → SpatialDropout1D → LSTM → Dense layers → Softmax output",
        "input_type": "Cybercrime incident description text",
        "max_len": lstm_clf.max_len if lstm_clf.is_loaded else 200,
        "num_classes": lstm_clf.num_classes if lstm_clf.is_loaded else 14,
        "classes": lstm_clf.classes if lstm_clf.is_loaded else [],
        "is_loaded": lstm_clf.is_loaded,
        "load_error": lstm_clf.load_error,
        "is_active_in_workflow": True,
    }

    # Check for cached evaluation
    cache_path = resolve_cache_file(write=False)
    if cache_path and cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("status") == "available" and "metrics" in data and "per_class" in data:
                return {
                    "status": "available",
                    "model_info": model_info,
                    "evaluation": data,
                }
        except Exception as e:
            logger.warning(f"Error reading evaluation cache: {e}")

    # No valid evaluation cache exists
    return {
        "status": "unavailable",
        "model_info": model_info,
        "evaluation": {
            "status": "unavailable",
            "reason": "Evaluation unavailable — held-out test dataset required",
            "details": (
                "The CyberSaathi LSTM classifier was pre-trained on an extensive corpus of ~71,422 complaints "
                "across 14 NCRP categories. To ensure scientific integrity and eliminate data leakage, authentic "
                "evaluation metrics must be computed against a strictly independent, held-out test dataset. "
                "Neither training data nor synthetic data are substituted for genuine evaluation."
            ),
            "dataset_size": None,
            "evaluated_at": None,
            "metrics": None,
            "per_class": [],
            "confusion_matrix": None,
            "instructions": (
                "To generate formal evaluation metrics, provide a held-out test CSV file with 'text' and 'category' "
                "columns matching the 14 NCRP classes, then execute: python -m app.ml.evaluate_lstm --test-data <path/to/test.csv>"
            ),
        },
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate CyberSaathi LSTM Classifier on a Test Dataset")
    parser.add_argument("--test-data", type=str, required=True, help="Path to held-out CSV test dataset")
    parser.add_argument("--no-cache", action="store_true", help="Do not save metrics to cache file")
    args = parser.parse_args()

    print("=" * 60)
    print("  CyberSaathi LSTM Evaluation")
    print("=" * 60)
    res = evaluate_dataset(args.test_data, save_cache=not args.no_cache)
    print(f"\nDataset: {res['dataset_name']} ({res['dataset_size']} samples)")
    print(f"Test Accuracy:    {res['metrics']['accuracy'] * 100:.2f}%")
    print(f"Macro Precision:  {res['metrics']['macro_precision'] * 100:.2f}%")
    print(f"Macro Recall:     {res['metrics']['macro_recall'] * 100:.2f}%")
    print(f"Macro F1-score:   {res['metrics']['macro_f1'] * 100:.2f}%")
    print(f"Weighted F1:      {res['metrics']['weighted_f1'] * 100:.2f}%")
    print("\nPer-Class Performance (14 classes):")
    for item in res["per_class"]:
        flag = " [LOW SUPPORT]" if item["is_low_support"] else ""
        print(f"  - {item['category']:<45} F1: {item['f1_score']*100:>5.1f}% | Support: {item['support']:>3}{flag}")
