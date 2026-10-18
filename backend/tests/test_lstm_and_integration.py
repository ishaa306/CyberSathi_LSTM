"""
Comprehensive Automated Test Suite for CyberSaathi LSTM Integration.
Covers:
- Part A: LSTM Classifier unit tests (sample texts, guards, probabilities, 14 classes)
- Part B: Conversational greeting & low-confidence thresholding
- Part C: Category adapter, database enum safety, and risk recalculation regression tests
- Part D: FastAPI endpoint tests (Health, ML comparison, Classify, Chat, Complaint CRUD, PDF Export)
"""

import pytest
import numpy as np
from datetime import datetime
from fastapi.testclient import TestClient

from app.main import app
from app.ml.lstm_classifier import get_lstm_classifier
from app.ml.category_adapter import (
    adapt_lstm_prediction,
    is_conversational_message,
    LSTM_NCRP_CATEGORIES,
    MIN_CONFIDENCE_THRESHOLD,
)
from app.models.complaint import CrimeCategory, Complaint, ComplaintStatus
from app.ml.risk_scorer import calculate_risk_score, CRIME_CATEGORY_BASE_SCORES


# ─── PART A: LSTM Classifier Unit Tests ────────────────────────────────────────

def test_lstm_model_artifacts_loaded():
    """Verify that LSTM weights, tokenizer, and label encoder are loaded correctly."""
    clf = get_lstm_classifier()
    assert clf.is_loaded is True
    assert clf.load_error is None
    assert clf.num_classes == 14
    assert clf.max_len == 200
    assert len(clf.classes) == 14


def test_lstm_financial_fraud_sample():
    """Verify prediction on a detailed financial fraud report."""
    clf = get_lstm_classifier()
    text = "Fraudulent UPI transfer of Rs 45000 was deducted from my bank account after a scammer called posing as bank manager"
    res = clf.predict(text)
    assert res["status"] == "success"
    assert res["category"] in clf.classes
    assert res["category"] == "Online Financial Fraud"
    assert res["confidence"] > 0.50
    assert np.all(np.isfinite(list(res["probabilities"].values())))


def test_lstm_social_media_sample():
    """Verify prediction on a social media / impersonation incident."""
    clf = get_lstm_classifier()
    text = "Someone made a fake Instagram profile using my personal photos and is sending abusive messages to my contacts"
    res = clf.predict(text)
    assert res["status"] == "success"
    assert res["category"] in clf.classes
    assert np.all(np.isfinite(list(res["probabilities"].values())))


def test_lstm_ransomware_sample():
    """Verify prediction on ransomware text returns a valid 14-class prediction."""
    clf = get_lstm_classifier()
    text = "All company files on our Windows server were encrypted with extension .locked and attacker demanded 1 BTC ransom"
    res = clf.predict(text)
    assert res["status"] == "success"
    assert res["category"] in clf.classes
    assert 0.0 <= res["confidence"] <= 1.0


def test_lstm_empty_and_whitespace_guards():
    """Verify that empty or whitespace text is handled safely without crashing."""
    clf = get_lstm_classifier()
    res_empty = clf.predict("")
    assert res_empty["status"] == "error"
    assert res_empty["category"] is None

    res_ws = clf.predict("     ")
    assert res_ws["status"] == "error"
    assert res_ws["category"] is None


# ─── PART B: Conversational & Confidence Thresholding ──────────────────────────

def test_conversational_greeting_detection():
    """Verify that greetings do not falsely trigger crime classifications."""
    clf = get_lstm_classifier()
    greetings = ["hello", "hi", "namaste", "good morning", "hey", "help me please", "i need help"]
    for g in greetings:
        assert is_conversational_message(g) is True
        raw = clf.predict(g)
        adapted = adapt_lstm_prediction(raw, input_text=g)
        assert adapted.status == "conversational"
        assert adapted.authoritative_category == "Pending Incident Details"
        assert adapted.needs_clarification is True


def test_incident_text_not_flagged_conversational():
    """Verify that actual cybercrime descriptions are not flagged as greetings."""
    incidents = [
        "Someone stole 25000 rs from my account via fake UPI QR code",
        "My WhatsApp account was hacked and scammer asking money from friends",
        "Got blackmail email threatening to release private webcam video",
    ]
    for inc in incidents:
        assert is_conversational_message(inc) is False


def test_later_greeting_preserves_existing_category():
    """Verify that saying 'ok thanks' or 'hello' later does NOT erase an existing valid classification."""
    clf = get_lstm_classifier()
    raw = clf.predict("hello")
    # Simulate existing classification from earlier message
    adapted = adapt_lstm_prediction(
        raw,
        input_text="ok thank you",
        existing_category="Online Financial Fraud",
        existing_confidence=0.95,
    )
    assert adapted.authoritative_category == "Online Financial Fraud"
    assert adapted.confidence == 0.95
    assert adapted.is_confident is True


def test_low_confidence_uncertainty_flag():
    """Verify that confidence below threshold (0.40) marks status as uncertain."""
    dummy_low = {
        "status": "success",
        "category": "Cyber Attack/ Dependent Crimes",
        "confidence": 0.25,
        "class_index": 3,
        "alternatives": [],
        "probabilities": {},
    }
    adapted = adapt_lstm_prediction(dummy_low, input_text="weird phone issue")
    assert adapted.status == "uncertain"
    assert adapted.is_confident is False
    assert adapted.needs_clarification is True
    assert adapted.confidence == 0.25
    assert adapted.authoritative_category == "Cyber Attack/ Dependent Crimes"


# ─── PART C: Category Adapter, Enum Safety & Risk Recalculation ───────────────

def test_all_14_classes_map_to_valid_legacy_enums():
    """Verify that all 14 LSTM categories map safely to valid SQLAlchemy CrimeCategory enum values."""
    valid_enum_values = set(e.value for e in CrimeCategory)
    for cat in LSTM_NCRP_CATEGORIES:
        dummy = {"status": "success", "category": cat, "confidence": 0.90, "class_index": 0}
        adapted = adapt_lstm_prediction(dummy)
        assert adapted.legacy_category in valid_enum_values


def test_risk_scorer_all_14_categories_have_weights():
    """Verify that all 14 official NCRP categories have calibrated risk weights in risk_scorer."""
    for cat in LSTM_NCRP_CATEGORIES:
        risk = calculate_risk_score(crime_category=cat, financial_loss=5000, text="test incident")
        assert risk["score"] > 0
        assert risk["level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")


def test_risk_recalculation_uses_authoritative_category():
    """
    REGRESSION TEST: Verify that recalculating risk on an updated complaint
    uses the authoritative category ('Online Financial Fraud') rather than legacy ('Other / Unknown').
    """
    # 1. With authoritative category (Online Financial Fraud, base score 25)
    risk_authoritative = calculate_risk_score(
        crime_category="Online Financial Fraud",
        financial_loss=10000,
        text="scammer transferred money",
    )
    # 2. With legacy fallback (Other / Unknown, base score 0)
    risk_legacy = calculate_risk_score(
        crime_category="Other / Unknown",
        financial_loss=10000,
        text="scammer transferred money",
    )
    # The authoritative category score MUST be higher by exactly the base score difference (25 points)
    assert risk_authoritative["score"] > risk_legacy["score"]
    assert risk_authoritative["score"] - risk_legacy["score"] == 25


# ─── PART D: FastAPI Endpoints (Health, ML, CRUD, PDF) ────────────────────────

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    """Verify /api/health returns 200 with ML status."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "ml" in data
    assert data["ml"]["lstm_loaded"] is True
    assert data["ml"]["classes_count"] == 14


def test_ml_comparison_endpoint(client):
    """Verify /api/ml/comparison returns model comparison and production model metadata."""
    res = client.get("/api/ml/comparison")
    assert res.status_code == 200
    data = res.json()
    assert "production_model" in data
    assert data["production_model"]["model_name"] == "LSTM (cybersaathi_lstm_unweighted)"
    assert data["production_model"]["taxonomy"] == "14 NCRP Categories"


def test_ml_classify_endpoint_financial_fraud(client):
    """Verify /api/ml/classify runs primary LSTM classification."""
    res = client.post("/api/ml/classify", json={"text": "Lost 50000 rupees in UPI lottery scam call"})
    assert res.status_code == 200
    data = res.json()
    assert data["primary_model"] == "LSTM (cybersaathi_lstm_unweighted)"
    assert data["category"] == "Online Financial Fraud"
    assert data["confidence"] > 0.50
    assert "disclaimer" in data
    assert "comparison_traditional_models" in data


def test_ml_classify_endpoint_greeting_handling(client):
    """Verify /api/ml/classify handles greetings cleanly."""
    res = client.post("/api/ml/classify", json={"text": "hello"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "conversational"
    assert data["category"] == "Pending Incident Details"
    assert data["needs_clarification"] is True


def test_auth_and_complaint_workflow(client):
    """Verify registration, login, complaint start, message, update, and list."""
    # 1. Register test user with unique email
    test_email = f"test_lstm_{datetime.now().strftime('%Y%m%d%H%M%S%f')}@example.com"
    reg_res = client.post("/api/auth/register", json={
        "email": test_email,
        "password": "Password123!",
        "full_name": "LSTM Test User",
        "phone": "9876543210"
    })
    assert reg_res.status_code in (200, 201), reg_res.text
    token = reg_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Start complaint
    start_res = client.post("/api/chat/start", headers=headers)
    assert start_res.status_code == 201
    complaint_id = start_res.json()["complaint_id"]
    assert complaint_id > 0

    # 3. Send greeting first — should not assign crime
    msg1_res = client.post("/api/chat/message", headers=headers, json={
        "complaint_id": complaint_id,
        "message": "hello"
    })
    assert msg1_res.status_code == 200
    data1 = msg1_res.json()
    assert data1["classification"]["category"] == "Pending Incident Details"

    # 4. Send detailed cybercrime incident
    msg2_res = client.post("/api/chat/message", headers=headers, json={
        "complaint_id": complaint_id,
        "message": "I lost 45000 rupees when someone sent a fraudulent Google Pay link pretending to be electricity board"
    })
    assert msg2_res.status_code == 200
    data2 = msg2_res.json()
    assert data2["classification"]["category"] == "Online Financial Fraud"
    assert data2["classification"]["confidence"] > 0.50

    # 5. Send follow-up polite message — must NOT erase classification!
    msg3_res = client.post("/api/chat/message", headers=headers, json={
        "complaint_id": complaint_id,
        "message": "ok thank you"
    })
    assert msg3_res.status_code == 200
    data3 = msg3_res.json()
    assert data3["classification"]["category"] == "Online Financial Fraud"

    # 6. List complaints — verify confidence and authoritative category are present
    list_res = client.get("/api/complaint/list", headers=headers)
    assert list_res.status_code == 200
    comps = list_res.json()["complaints"]
    matched = [c for c in comps if c["id"] == complaint_id]
    assert len(matched) == 1
    assert matched[0]["authoritative_category"] == "Online Financial Fraud"
    assert matched[0]["confidence"] is not None
    assert matched[0]["confidence"] > 0.50

    # 7. Update complaint — verify risk recalculation maintains authoritative category
    patch_res = client.patch(f"/api/complaint/{complaint_id}", headers=headers, json={
        "financial_loss": 50000.0,
        "incident_date": "2026-10-15"
    })
    assert patch_res.status_code == 200
    patch_data = patch_res.json()
    assert patch_data["authoritative_category"] == "Online Financial Fraud"
    assert patch_data["confidence"] > 0.50
    # Financial loss increased to 50k -> risk score reflects higher threshold + 25 base score
    assert patch_data["risk_score"] >= 50

    # 8. Download PDF export
    pdf_res = client.get(f"/api/complaint/{complaint_id}/pdf", headers=headers)
    assert pdf_res.status_code == 200
    assert pdf_res.headers["content-type"] == "application/pdf"
    assert len(pdf_res.content) > 1000  # PDF generated with content
