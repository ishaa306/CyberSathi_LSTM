"""
LSTM Cybercrime Classifier — Isolated Deep Learning Inference Service.
Loads the pre-trained Keras LSTM model and companion artifacts from cybersaathi_lstm_unweighted/.
Provides thread-safe singleton inference with text sequence padding (MAX_LEN=200).
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np

# Suppress verbose TensorFlow / oneDNN info messages
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

logger = logging.getLogger(__name__)

# Determine repository root and artifact directories reliably
CURRENT_FILE = Path(__file__).resolve()
BACKEND_DIR = CURRENT_FILE.parent.parent.parent  # .../backend
PROJECT_ROOT = BACKEND_DIR.parent                # .../CyberSathi

DEFAULT_ARTIFACT_DIRS = [
    PROJECT_ROOT / "cybersaathi_lstm_unweighted",
    BACKEND_DIR / "models" / "cybersaathi_lstm_unweighted",
    BACKEND_DIR / "models",
]


class LSTMClassifier:
    """
    Dedicated LSTM inference service for CyberSaathi.
    Encapsulates Keras model, tokenizer, and label encoder.
    Maintains loaded weights in memory for fast multi-query reuse.
    """

    def __init__(self, artifact_dir: Optional[Path] = None):
        self.artifact_dir = self._resolve_artifact_dir(artifact_dir)
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.max_len = 200
        self.num_classes = 0
        self.classes: List[str] = []
        self.is_loaded = False
        self.load_error: Optional[str] = None

        self._load()

    def _resolve_artifact_dir(self, explicit_dir: Optional[Path] = None) -> Path:
        """Find the directory containing all 4 required LSTM artifacts."""
        if explicit_dir and explicit_dir.exists():
            return explicit_dir

        for candidate in DEFAULT_ARTIFACT_DIRS:
            if candidate.exists() and (candidate / "cybersaathi_lstm_unweighted.keras").exists():
                return candidate

        # Return primary expected root location as default
        return PROJECT_ROOT / "cybersaathi_lstm_unweighted"

    def _load(self):
        """Load config, label encoder, tokenizer, and Keras model from disk."""
        try:
            model_path = self.artifact_dir / "cybersaathi_lstm_unweighted.keras"
            tok_path = self.artifact_dir / "tokenizer.pkl"
            le_path = self.artifact_dir / "label_encoder.pkl"
            cfg_path = self.artifact_dir / "config.json"

            # Check that all artifacts exist
            for path_name, path_obj in [
                ("Keras Model", model_path),
                ("Tokenizer", tok_path),
                ("Label Encoder", le_path),
                ("Config", cfg_path),
            ]:
                if not path_obj.exists():
                    raise FileNotFoundError(f"{path_name} artifact not found at: {path_obj}")

            # 1. Load config
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                self.max_len = int(cfg.get("MAX_LEN", 200))

            # 2. Load Label Encoder
            with open(le_path, "rb") as f:
                self.label_encoder = pickle.load(f)
                self.classes = [str(c) for c in self.label_encoder.classes_]
                self.num_classes = len(self.classes)

            # 3. Load Tokenizer
            with open(tok_path, "rb") as f:
                self.tokenizer = pickle.load(f)

            # 4. Load Keras Model (lazy import to keep module import fast)
            import keras
            self.model = keras.models.load_model(str(model_path))

            # Verify output shape matches label encoder classes
            output_dim = self.model.output_shape[-1]
            if output_dim != self.num_classes:
                raise ValueError(
                    f"Model output dimension ({output_dim}) does not match "
                    f"label encoder classes ({self.num_classes})"
                )

            self.is_loaded = True
            self.load_error = None
            logger.info(
                f"[OK] LSTM Classifier loaded successfully from {self.artifact_dir.name} "
                f"({self.num_classes} classes, MAX_LEN={self.max_len})"
            )

        except Exception as e:
            self.is_loaded = False
            self.load_error = str(e)
            logger.error(f"Failed to initialize LSTM classifier: {e}")

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Run classification on the provided text.
        
        Args:
            text: Complaint text to classify.
            
        Returns:
            Dictionary containing predicted category, confidence, probabilities,
            and status metadata.
        """
        # Guard 1: Verify model readiness
        if not self.is_loaded or self.model is None or self.tokenizer is None or self.label_encoder is None:
            return {
                "status": "error",
                "error": f"LSTM model is not ready: {self.load_error or 'Unknown initialization error'}",
                "category": None,
                "confidence": 0.0,
                "alternatives": [],
            }

        # Guard 2: Handle empty or whitespace input safely
        if not text or not text.strip():
            return {
                "status": "error",
                "error": "Empty or whitespace-only text provided for classification.",
                "category": None,
                "confidence": 0.0,
                "alternatives": [],
            }

        try:
            from keras.preprocessing.sequence import pad_sequences

            # Preprocessing: convert text to sequence of token IDs and pad
            sequences = self.tokenizer.texts_to_sequences([text])
            padded = pad_sequences(sequences, maxlen=self.max_len)

            # Predict probabilities
            raw_predictions = self.model.predict(padded, verbose=0)
            probabilities = raw_predictions[0]

            # Validate numerical properties
            if not np.all(np.isfinite(probabilities)):
                raise ValueError("Model produced non-finite probability values (NaN/Inf).")

            top_idx = int(np.argmax(probabilities))
            top_confidence = float(probabilities[top_idx])
            top_category = self.classes[top_idx]

            # Sort top 3 alternatives
            sorted_indices = np.argsort(probabilities)[::-1]
            alternatives = [
                {
                    "category": self.classes[idx],
                    "confidence": round(float(probabilities[idx]), 4),
                    "class_index": int(idx),
                }
                for idx in sorted_indices[:3]
            ]

            # Class probability mapping for all classes
            prob_dict = {
                self.classes[idx]: round(float(probabilities[idx]), 5)
                for idx in range(self.num_classes)
            }

            return {
                "status": "success",
                "category": top_category,
                "confidence": round(top_confidence, 4),
                "class_index": top_idx,
                "all_classes": self.classes,
                "alternatives": alternatives,
                "probabilities": prob_dict,
                "method": "LSTM (cybersaathi_lstm_unweighted)",
            }

        except Exception as e:
            logger.error(f"LSTM prediction error: {e}")
            return {
                "status": "error",
                "error": f"Inference failed: {str(e)}",
                "category": None,
                "confidence": 0.0,
                "alternatives": [],
            }


# ─── Global Singleton ──────────────────────────────────────────────────────────

_lstm_classifier_instance: Optional[LSTMClassifier] = None


def get_lstm_classifier() -> LSTMClassifier:
    """Return the global singleton instance of LSTMClassifier."""
    global _lstm_classifier_instance
    if _lstm_classifier_instance is None:
        _lstm_classifier_instance = LSTMClassifier()
    return _lstm_classifier_instance
