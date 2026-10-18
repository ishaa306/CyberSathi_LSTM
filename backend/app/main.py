"""
AI-Powered Cyber Crime Complaint & Assistance System
FastAPI Application Entrypoint
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import logging

from app.core.config import settings
from app.core.database import create_tables
import app.models  # noqa: F401
from app.api import auth, chat, evidence, complaint, models_comparison

# Configure logging
logging.basicConfig(level=logging.INFO if settings.DEBUG else logging.WARNING)
logger = logging.getLogger(__name__)

# ─── Application ──────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "AI-powered cybercrime complaint analysis system using NLP, ML classification, "
        "OCR evidence processing, and structured complaint generation."
    ),
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# ─── CORS ─────────────────────────────────────────────────────────────────────

origins = settings.allowed_origins_list
has_wildcard = any("*" in o for o in origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[] if has_wildcard else origins,
    allow_origin_regex=r"^https?://.*" if has_wildcard else r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ──────────────────────────────────────────────────────────────────

app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(evidence.router)
app.include_router(complaint.router)
app.include_router(models_comparison.router)

# ─── Startup Events ───────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")

    # Create database tables
    try:
        await create_tables()
        logger.info("[OK] Database tables created/verified")
    except Exception as e:
        logger.warning(f"Database connection warning on startup: {e}")

    # Ensure upload directory exists
    Path(settings.UPLOAD_DIR).mkdir(exist_ok=True)
    logger.info(f"[OK] Upload directory ready: {settings.UPLOAD_DIR}")

    # Pre-load ML models
    try:
        from app.ml.lstm_classifier import get_lstm_classifier
        lstm_clf = get_lstm_classifier()
        logger.info(f"[OK] LSTM classifier ready (loaded={lstm_clf.is_loaded}, classes={lstm_clf.num_classes})")
    except Exception as e:
        logger.warning(f"LSTM pre-load warning: {e}")

    try:
        from app.ml.classifier import get_classifier
        classifier = get_classifier()
        logger.info("[OK] Crime classifier ready")
    except Exception as e:
        logger.warning(f"Traditional classifier pre-load warning: {e}")


@app.on_event("shutdown")
async def shutdown():
    logger.info("Shutting down application...")


# ─── Health Check ─────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    from app.ml.lstm_classifier import get_lstm_classifier
    lstm = get_lstm_classifier()
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "ml": {
            "lstm_loaded": lstm.is_loaded,
            "classes_count": lstm.num_classes,
            "primary_model": "LSTM (cybersaathi_lstm_unweighted)",
        }
    }


@app.get("/")
async def root():
    return {
        "message": "AI-Powered Cyber Crime Complaint & Assistance System API",
        "docs": "/api/docs",
        "version": settings.APP_VERSION,
    }
