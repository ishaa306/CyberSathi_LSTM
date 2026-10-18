"""
Category Compatibility Layer — Reconciles LSTM 14-class NCRP taxonomy with legacy application requirements.
Ensures:
1. The LSTM prediction is preserved as the authoritative, primary classification.
2. Legacy database enum (CrimeCategory) constraints are never violated.
3. No false granular inference (e.g. claiming broad Online Financial Fraud is specifically UPI or OTP fraud).
4. Unmapped categories fall back safely to 'Other / Unknown' in the legacy schema without data loss.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

# The 14 official categories output by the trained LSTM model
LSTM_NCRP_CATEGORIES = [
    "Any Other Cyber Crime",
    "Child Pornography CPChild Sexual Abuse Material CSAM",
    "Cryptocurrency Crime",
    "Cyber Attack/ Dependent Crimes",
    "Cyber Terrorism",
    "Hacking  Damage to computercomputer system etc",
    "Online Cyber Trafficking",
    "Online Financial Fraud",
    "Online Gambling  Betting",
    "Online and Social Media Related Crime",
    "Ransomware",
    "RapeGang Rape RGRSexually Abusive Content",
    "Sexually Explicit Act",
    "Sexually Obscene material",
]

# Legacy SQLAlchemy Enum values from app.models.complaint.CrimeCategory:
# [
#   "UPI / Payment Fraud", "Banking Fraud", "OTP / Social Engineering", "Phishing",
#   "Job / Employment Fraud", "Investment Fraud", "E-commerce Fraud", "Social Media Fraud",
#   "Account Compromise", "Identity Theft", "Impersonation", "Cyber Extortion",
#   "Malware / Ransomware", "Cryptocurrency Fraud", "Other / Unknown"
# ]

# Explicit, documented mapping rules for legacy schema compatibility only.
# When a direct 1-to-1 semantic equivalence exists, we map it.
# When a category is broader (e.g. Online Financial Fraud) or new (e.g. CSAM, Cyber Terrorism),
# we fall back to "Other / Unknown" in the legacy enum to prevent false specificity and DB errors.
LEGACY_ENUM_MAPPING = {
    # Direct 1-to-1 or near-identical semantic equivalences:
    "Cryptocurrency Crime": ("Cryptocurrency Fraud", True, "Direct semantic match to Cryptocurrency Fraud"),
    "Ransomware": ("Malware / Ransomware", True, "Mapped to Malware / Ransomware"),
    "Any Other Cyber Crime": ("Other / Unknown", True, "Direct semantic match to Other / Unknown"),
    "Online and Social Media Related Crime": ("Social Media Fraud", True, "Mapped to Social Media Fraud"),
    "Hacking  Damage to computercomputer system etc": ("Account Compromise", True, "Mapped to Account Compromise"),

    # Broad financial category:
    # Do NOT force into UPI, Banking, OTP, or Phishing.
    "Online Financial Fraud": (
        "Other / Unknown",
        False,
        "Broad NCRP category 'Online Financial Fraud'; mapped to 'Other / Unknown' in legacy DB enum to avoid false specific inference."
    ),

    # New / High-Severity categories with NO equivalent in legacy 15-class enum:
    "Child Pornography CPChild Sexual Abuse Material CSAM": (
        "Other / Unknown",
        False,
        "High-severity category with no legacy enum representation; preserved as authoritative in metadata."
    ),
    "Cyber Terrorism": (
        "Other / Unknown",
        False,
        "National security category with no legacy enum representation; preserved as authoritative in metadata."
    ),
    "Online Cyber Trafficking": (
        "Other / Unknown",
        False,
        "Trafficking category with no legacy enum representation; preserved as authoritative in metadata."
    ),
    "Online Gambling  Betting": (
        "Other / Unknown",
        False,
        "Gambling category with no legacy enum representation; preserved as authoritative in metadata."
    ),
    "RapeGang Rape RGRSexually Abusive Content": (
        "Other / Unknown",
        False,
        "Sensitive sexual abuse category with no legacy enum representation; preserved as authoritative in metadata."
    ),
    "Sexually Explicit Act": (
        "Other / Unknown",
        False,
        "Sensitive content category with no legacy enum representation; preserved as authoritative in metadata."
    ),
    "Sexually Obscene material": (
        "Other / Unknown",
        False,
        "Obscene material category with no legacy enum representation; preserved as authoritative in metadata."
    ),
    "Cyber Attack/ Dependent Crimes": (
        "Other / Unknown",
        False,
        "Infrastructure attack category with no legacy enum representation; preserved as authoritative in metadata."
    ),
}


# Minimum confidence required for a classification to be treated as definite rather than uncertain
MIN_CONFIDENCE_THRESHOLD = 0.40

# Heuristic list of common greetings and pleasantries (distinct from cybercrime incident descriptions)
GREETING_WORDS = {
    "hello", "hi", "hey", "namaste", "namaskar", "good morning", "good evening",
    "good afternoon", "greetings", "howdy", "sup", "yo", "hai"
}

GENERIC_HELP_PATTERNS = {
    "help", "help me", "help me please", "can you help", "can you help me",
    "i need help", "please help", "please help me", "i need assistance",
    "are you there", "who are you", "what can you do", "test", "testing",
    "ok", "okay", "thanks", "thank you", "bye", "goodbye"
}


def is_conversational_message(text: str) -> bool:
    """
    Detect whether a message is purely a greeting, pleasantry, or generic plea
    without any substantive cybercrime incident details.
    
    This is a heuristic conversational filter, NOT an additional AI model.
    """
    if not text:
        return True

    cleaned = text.strip().lower()
    if not cleaned:
        return True

    # Check exact match with generic help phrases or single-word greetings
    if cleaned in GENERIC_HELP_PATTERNS or cleaned in GREETING_WORDS:
        return True

    import re
    words = re.findall(r"\b\w+\b", cleaned)
    if not words:
        return True

    # Any numbers/digits indicate dates, amounts, phone numbers, or account details — not just a greeting
    if any(char.isdigit() for char in cleaned):
        return False

    # Key cybercrime incident indicators
    crime_keywords = {
        "scam", "fraud", "hacked", "hack", "stolen", "deducted", "transfer",
        "transferred", "lost", "threat", "blackmail", "extortion", "compromised",
        "otp", "pin", "upi", "bank", "card", "account", "money", "rupees",
        "rs", "inr", "crypto", "bitcoin", "ransom", "ransomware", "phishing",
        "link", "click", "apk", "whatsapp", "telegram", "instagram", "facebook",
        "caller", "police", "arrest", "lottery", "job", "investment", "trafficking",
        "nude", "leaked", "video", "call", "cheated", "victim", "attacker"
    }
    if any(kw in words for kw in crime_keywords):
        return False

    # Short message (<= 5 words) composed exclusively of greeting and generic help words
    greeting_and_help = GREETING_WORDS.union({
        "help", "please", "sir", "madam", "me", "i", "need", "can", "you",
        "there", "anyone", "assistant", "saathi", "cybersaathi", "am", "a", "some"
    })
    if len(words) <= 5 and all(w in greeting_and_help for w in words):
        return True

    return False


@dataclass
class CategoryAdaptationResult:
    authoritative_category: Optional[str]
    confidence: float
    legacy_category: str
    is_direct_mapping: bool
    mapping_note: str
    class_index: int = 0
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    probabilities: Dict[str, float] = field(default_factory=dict)
    model: str = "LSTM (cybersaathi_lstm_unweighted)"
    is_ai_generated: bool = True
    status: str = "confident"  # "confident" | "uncertain" | "conversational" | "error"
    is_confident: bool = True
    needs_clarification: bool = False
    raw_category: Optional[str] = None
    raw_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.authoritative_category,
            "authoritative_category": self.authoritative_category,
            "confidence": self.confidence,
            "legacy_category": self.legacy_category,
            "is_direct_mapping": self.is_direct_mapping,
            "mapping_note": self.mapping_note,
            "class_index": self.class_index,
            "alternatives": self.alternatives,
            "probabilities": self.probabilities,
            "model": self.model,
            "method": "AI Deep Learning (LSTM)",
            "status": self.status,
            "is_confident": self.is_confident,
            "needs_clarification": self.needs_clarification,
            "raw_category": self.raw_category,
            "raw_confidence": self.raw_confidence,
            "threshold": MIN_CONFIDENCE_THRESHOLD,
            "disclaimer": "This is an AI-generated incident classification for organizational assistance, not an official legal determination.",
        }


def adapt_lstm_prediction(
    lstm_result: Dict[str, Any],
    input_text: str = "",
    existing_category: Optional[str] = None,
    existing_confidence: Optional[float] = None,
) -> CategoryAdaptationResult:
    """
    Transform raw LSTM output into an adapted result that protects legacy modules
    while honoring the authoritative deep learning prediction.
    
    Handles:
    - Conversational messages (greetings/pleas) that lack incident facts
    - Low-confidence predictions (< MIN_CONFIDENCE_THRESHOLD)
    - Preservation of existing valid category when new message is non-informative
    """
    if lstm_result.get("status") != "success" or not lstm_result.get("category"):
        # Fallback for errors or empty input
        return CategoryAdaptationResult(
            authoritative_category=existing_category or "Pending Incident Details",
            confidence=existing_confidence or 0.0,
            legacy_category="Other / Unknown",
            is_direct_mapping=True,
            mapping_note="Default fallback due to missing or empty inference output.",
            status="error",
            is_confident=False,
            needs_clarification=True,
            raw_category=None,
            raw_confidence=0.0,
        )

    raw_cat = lstm_result["category"]
    raw_conf = float(lstm_result.get("confidence", 0.0))

    if raw_cat in LEGACY_ENUM_MAPPING:
        legacy_cat, is_direct, note = LEGACY_ENUM_MAPPING[raw_cat]
    else:
        legacy_cat = "Other / Unknown"
        is_direct = False
        note = f"Unrecognized category '{raw_cat}' mapped to 'Other / Unknown'."

    # 1. Check if input text is purely conversational / greeting without incident details
    is_conv = is_conversational_message(input_text)
    if is_conv:
        # If an existing incident classification already exists, preserve it!
        if existing_category and existing_category not in ("Pending Incident Details", "Other / Unknown"):
            return CategoryAdaptationResult(
                authoritative_category=existing_category,
                confidence=existing_confidence if existing_confidence is not None else raw_conf,
                legacy_category=legacy_cat if existing_category == raw_cat else "Other / Unknown",
                is_direct_mapping=is_direct,
                mapping_note=f"Preserved existing category '{existing_category}' as current message was greeting/conversational.",
                class_index=int(lstm_result.get("class_index", 0)),
                alternatives=lstm_result.get("alternatives", []),
                probabilities=lstm_result.get("probabilities", {}),
                status="conversational",
                is_confident=True,
                needs_clarification=False,
                raw_category=raw_cat,
                raw_confidence=raw_conf,
            )
        else:
            # First greeting or inquiry without prior incident facts
            return CategoryAdaptationResult(
                authoritative_category="Pending Incident Details",
                confidence=raw_conf,
                legacy_category="Other / Unknown",
                is_direct_mapping=False,
                mapping_note="Conversational greeting/plea detected; awaiting incident details before classification.",
                class_index=int(lstm_result.get("class_index", 0)),
                alternatives=lstm_result.get("alternatives", []),
                probabilities=lstm_result.get("probabilities", {}),
                status="conversational",
                is_confident=False,
                needs_clarification=True,
                raw_category=raw_cat,
                raw_confidence=raw_conf,
            )

    # 2. Check confidence threshold
    if raw_conf < MIN_CONFIDENCE_THRESHOLD:
        # Model is uncertain (< 0.40)
        # If an existing confident classification exists, do not overwrite it with uncertain noise
        if existing_category and (existing_confidence or 0.0) >= MIN_CONFIDENCE_THRESHOLD:
            return CategoryAdaptationResult(
                authoritative_category=existing_category,
                confidence=existing_confidence,
                legacy_category=legacy_cat,
                is_direct_mapping=is_direct,
                mapping_note=f"Preserved existing confident classification '{existing_category}' over low-confidence prediction ({raw_conf:.1%}).",
                class_index=int(lstm_result.get("class_index", 0)),
                alternatives=lstm_result.get("alternatives", []),
                probabilities=lstm_result.get("probabilities", {}),
                status="confident",
                is_confident=True,
                needs_clarification=False,
                raw_category=raw_cat,
                raw_confidence=raw_conf,
            )
        else:
            return CategoryAdaptationResult(
                authoritative_category=raw_cat,
                confidence=raw_conf,
                legacy_category=legacy_cat,
                is_direct_mapping=is_direct,
                mapping_note=f"Low model confidence ({raw_conf:.1%}) below threshold ({MIN_CONFIDENCE_THRESHOLD:.0%}); more details needed.",
                class_index=int(lstm_result.get("class_index", 0)),
                alternatives=lstm_result.get("alternatives", []),
                probabilities=lstm_result.get("probabilities", {}),
                status="uncertain",
                is_confident=False,
                needs_clarification=True,
                raw_category=raw_cat,
                raw_confidence=raw_conf,
            )

    # 3. Confident classification (>= 0.40)
    return CategoryAdaptationResult(
        authoritative_category=raw_cat,
        confidence=raw_conf,
        legacy_category=legacy_cat,
        is_direct_mapping=is_direct,
        mapping_note=note,
        class_index=int(lstm_result.get("class_index", 0)),
        alternatives=lstm_result.get("alternatives", []),
        probabilities=lstm_result.get("probabilities", {}),
        model=lstm_result.get("method", "LSTM (cybersaathi_lstm_unweighted)"),
        status="confident",
        is_confident=True,
        needs_clarification=False,
        raw_category=raw_cat,
        raw_confidence=raw_conf,
    )
