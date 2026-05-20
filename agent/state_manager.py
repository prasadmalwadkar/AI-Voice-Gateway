"""
State Manager — Central Application State + Loopback Tool
==========================================================

This module owns the single source of truth for the HELOC application
during a phone call: an instance of HelocApplicationState.

The @loopback_tool `update_application_state` lets the LLM push extracted
fields into the state after each conversational turn. Crucially, the tool
returns a summary of what's still missing — this keeps the agent laser-
focused on collecting the remaining fields instead of re-asking for
information the caller already provided.

Design decisions:
- One global state instance per call (reset in get_agent factory).
- Every parameter uses Annotated[Optional[...], "description"] so the
  LLM knows exactly what each field expects — it only sends the fields
  it actually extracted from the user's speech.
- The "missing fields" feedback loop is the key UX trick: it turns a
  dumb chatbot into a structured intake agent that never forgets where
  it left off.
- confidence_score enables the LLM to self-assess extraction quality.
  Fields below 0.85 confidence trigger re-confirmation before moving on.
- Legal consent fields (privacy bundle, electronic communications) require explicit
  high-confidence "I agree" — vague responses are not accepted.
- UNCERTAINTY_MARKERS provide a deterministic safety net: even if the LLM
  reports high confidence, hedging language in the user's speech ("maybe",
  "or", "I think") blocks the save and forces re-confirmation.
"""

import asyncio
import contextvars
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Optional

from pydantic import ValidationError
from line.llm_agent import loopback_tool

from agent.schema import (
    HelocApplicationState,
    HeuristicResult,
    OccupancyType,
    OwnershipType,
    IncomeSource,
    STRICT_EMAIL_REGEX,
)
# New deterministic parsers module for decoupling voice from logic.
from agent.deterministic_parsers import (
    normalize_answer_scaffold,
    infer_ownership_from_transcript,
    infer_occupancy_from_transcript,
    infer_occupancy_with_optional_recovery,
    infer_for_sale_from_transcript,
    infer_income_source_from_transcript,
    infer_annual_income_meta,
    _INCOME_AMBIGUOUS_STT_CONFIDENCE,
    infer_annual_income_from_transcript,
    infer_other_income_from_transcript,
    infer_dob_from_transcript,
    infer_partial_dob_month_day,
    infer_phone_from_transcript,
    _spoken_digit_sequence_to_digits,
    infer_email_from_transcript,
    infer_property_address_from_transcript,
    infer_property_city_state_from_transcript,
    build_address_autocomplete_type_query,
    infer_name_fields_from_transcript,
    infer_yes_no_from_transcript,
    detect_uncertainty,
    looks_like_answer_for_current_field,
    transcript_is_status_check,
    transcript_is_dnc,
    transcript_is_intake_decline,
    transcript_seeks_question_repeat,
    transcript_requests_intake_default,
    UNCERTAINTY_MARKERS,
    PARTIAL_DOB_REPROMPT,
    PARTIAL_MOBILE_PHONE_REPROMPT,
    _stt_confidence_triggers_reprompt,
    _name_slot_focus_from_missing,
    _use_deterministic_address_parsing,
    _transcript_supports_saved_dob,
    _spoken_letters_to_word,
    CONFIDENCE_THRESHOLD,
    ADDRESS_CONFIDENCE_THRESHOLD,
    ADDRESS_CONFIDENCE_FIELDS,
    HIGH_CONFIDENCE_FIELDS,
    DEFAULT_INTAKE_OWNERSHIP,
    DEFAULT_INTAKE_OCCUPANCY,
    DEFAULT_INTAKE_FOR_SALE,
    DEFAULT_INTAKE_OTHER_INCOME,
    DEFAULT_INTAKE_ANNUAL_PRE_TAX,
    DEFAULT_INTAKE_PRIMARY_SOURCE,
    VOICE_SCRIPT_OWNERSHIP_ASK,
    VOICE_SCRIPT_OCCUPANCY_ASK,
    VOICE_SCRIPT_FOR_SALE_ASK,
    OCCUPANCY_LOW_CONF_REPROMPT,
    OWNERSHIP_LOW_CONF_REPROMPT,
    FOR_SALE_LOW_CONF_REPROMPT,
    _GREETING_SOFT_DECLINE_PATTERNS,
)

# Import queue publisher for completed intake payloads.
from agent.queue_publisher import publish_completed_intake

logger = logging.getLogger("voip.state")

# Maps conversation stages (logic-based) to the internal identifiers used
# throughout the file for stage gating and heuristics. Browser is no longer
# running in the voice path.
_COMPLETED_CLOSING_LINE = (
    "Great, that's everything we need. Your application has been submitted and "
    "someone from West Capital Lending will be in touch soon. Thanks for your time, goodbye!"
)

# Seconds after a preline success during which subsequent preline calls return _NOT_HANDLED.
# Prevents trailing STT finals from the same utterance from confirming the *next* field.
_PRELINE_COOLDOWN_SECS = 1.5

_CONV_STAGE_TO_INTAKE_ID: dict[str, str] = {
    "PROPERTY": "property",
    "PRIMARY_HOME": "primary-home",
    "FOR_SALE": "for-sale",
    "INCOME": "income",
    "PERSONAL": "name",
    "BIRTHDAY": "birthday",
    "CONTACT": "contact",
    "CONSENT": "consent",
    "GREETING": "greeting",
    "COMPLETED": "completed",
}


# =============================================================================
# Per-call state lives in a ContextVar so overlapping calls in one server process
# cannot reset or mutate each other's HELOC application.
# =============================================================================
@dataclass
class _CallState:
    application_state: HelocApplicationState
    pending_low_conf_confirmation: Optional[dict] = None
    customer_prefill: Optional[dict[str, Any]] = None
    prefill_confirm_pending: Optional[dict[str, Any]] = None
    prefill_rejected_fields: Optional[set[str]] = None
    consent_decline_round: Optional[dict[str, int]] = None
    voice_field_user_sources: Optional[dict[str, str]] = None
    last_prefill_yes_mono: float = 0.0
    last_prefill_yes_fields: frozenset[str] = frozenset()
    # Track whether empathy has already been shown this call so the LLM
    # context can explicitly tell it not to repeat the acknowledgement.
    empathy_shown: bool = False
    # Call metadata for queue publishing
    call_sid: str = ""
    caller_from: str = ""
    caller_to: str = ""
    intake_published: bool = False  # guards against double-publish
    greeting_completed: bool = False  # set to True when caller confirms OK to proceed
    _preline_last_handled_at: float = 0.0  # monotonic timestamp of last preline success

    def __post_init__(self) -> None:
        self.customer_prefill = {}
        self.prefill_rejected_fields = set()
        self.consent_decline_round = {}
        self.voice_field_user_sources = {}


_call_state_registry: dict[str, _CallState] = {}
_call_state_ctx: contextvars.ContextVar[_CallState] = contextvars.ContextVar(
    "heloc_call_state",
    default=_CallState(HelocApplicationState()),
)


def _new_call_state() -> _CallState:
    return _CallState(HelocApplicationState())


def bind_call_state(call_id: str | None, *, fresh: bool = False) -> None:
    """Bind this async context to the state bucket for one call."""
    key = (call_id or "").strip()
    if not key:
        if fresh:
            _call_state_ctx.set(_new_call_state())
        return
    state = _call_state_registry.get(key)
    if state is None:
        state = _new_call_state()
        _call_state_registry[key] = state
    elif fresh:
        state.application_state = HelocApplicationState()
    _call_state_ctx.set(state)


def release_call_state(call_id: str | None) -> None:
    key = (call_id or "").strip()
    if key:
        _call_state_registry.pop(key, None)


def store_call_metadata(
    call_sid: str,
    caller_from: Optional[str] = None,
    caller_to: Optional[str] = None,
) -> None:
    """Persist Twilio call identifiers so publish_completed_intake has non-empty IDs."""
    cs = _current_call_state()
    cs.call_sid = call_sid or ""
    cs.caller_from = caller_from or ""
    cs.caller_to = caller_to or ""


def call_state_registered(call_id: str | None) -> bool:
    """True when Path2 / Twilio already created the registry bucket for this call."""
    key = (call_id or "").strip()
    return bool(key and key in _call_state_registry)


def _current_call_state() -> _CallState:
    return _call_state_ctx.get()


class _ApplicationStateProxy:
    def __getattr__(self, name: str) -> Any:
        return getattr(_current_call_state().application_state, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(_current_call_state().application_state, name, value)

    def __repr__(self) -> str:
        return repr(_current_call_state().application_state)


application_state = _ApplicationStateProxy()

_SENSITIVE_FIELDS = frozenset({"password", "confirm_password"})

_PREFILL_FIELD_ALIASES = {
    "property_address": "address_line_1",
    "address": "address_line_1",
    "property_type": "occupancy_type",
    "phone": "mobile_phone_number",
    "phone_number": "mobile_phone_number",
    "email": "email_address",
    "for_sale": "is_currently_for_sale",
    "currently_for_sale": "is_currently_for_sale",
    "primary_home_address": "primary_home_address_line_1",
    "primary_residence_address": "primary_home_address_line_1",
    "primary_home_street": "primary_home_address_line_1",
}
_PREFILL_ALLOWED_FIELDS = frozenset(
    {
        "address_line_1",
        "apt_suite",
        "city",
        "state",
        "zipcode",
        "ownership_type",
        "occupancy_type",
        "primary_home_address_line_1",
        "primary_home_apt_suite",
        "primary_home_city",
        "primary_home_state",
        "primary_home_zipcode",
        "is_currently_for_sale",
        "annual_pre_tax_income",
        "other_income",
        "primary_source_of_income",
        "first_name",
        "middle_name",
        "last_name",
        "suffix",
        "date_of_birth",
        "mobile_phone_number",
        "email_address",
    }
)
def _occupancy_requires_primary_home_intake(occ: Optional[str]) -> bool:
    return occ in ("Secondary property", "Investment property")


def _use_deterministic_address_parsing() -> bool:
    return os.getenv("PATH2_DETERMINISTIC_ADDRESS", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def _redact_sensitive_map(data: dict) -> dict:
    out = dict(data)
    for key in _SENSITIVE_FIELDS:
        if key in out and out[key] is not None:
            out[key] = "***REDACTED***"
    return out


def _redact_if_sensitive(field_name: str, value: object) -> object:
    if field_name in _SENSITIVE_FIELDS and value is not None:
        return "***REDACTED***"
    return value


def _normalize_prefill_key(key: str) -> str:
    normalized = (key or "").strip().lower().replace("-", "_").replace(" ", "_")
    return _PREFILL_FIELD_ALIASES.get(normalized, normalized)


def _normalize_prefill_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "listed", "for sale"}:
        return True
    if text in {"0", "false", "no", "n", "not listed", "not for sale"}:
        return False
    return None


def _normalize_prefill_value(field_name: str, value: Any) -> Any:
    if value is None or value == "":
        return None
    if field_name == "ownership_type":
        text = str(value).strip().lower()
        if "joint" in text:
            return "Joint owner"
        if "sole" in text or "single" in text:
            return "Sole owner"
    if field_name == "occupancy_type":
        text = str(value).strip().lower()
        if "secondary" in text or "second" in text:
            return "Secondary property"
        if "investment" in text or "rental" in text:
            return "Investment property"
        if "primary" in text or "main" in text:
            return "Primary property"
    if field_name == "is_currently_for_sale":
        return _normalize_prefill_bool(value)
    if field_name in {"annual_pre_tax_income", "other_income"}:
        if isinstance(value, int):
            return value
        digits = re.sub(r"[^\d]", "", str(value))
        return int(digits) if digits else None
    if field_name == "primary_source_of_income":
        text = str(value).strip()
        normalized = re.sub(r"\s+", " ", text).lower()
        allowed = {
            "alimony": "Alimony",
            "child support": "Child support",
            "employed full-time": "Employed full-time",
            "employed full time": "Employed full-time",
            "employed part-time": "Employed part-time",
            "employed part time": "Employed part-time",
            "public assistance program": "Public assistance program",
            "retired": "Retired",
            "self-employed": "Self-employed",
            "self employed": "Self-employed",
            "separate maintenance payments": "Separate maintenance payments",
            "unemployed / furloughed": "Unemployed / furloughed",
            "unemployed": "Unemployed / furloughed",
            "furloughed": "Unemployed / furloughed",
        }
        return allowed.get(normalized, text)
    if isinstance(value, str):
        return value.strip()
    return value


def set_customer_prefill(
    payload: Optional[dict[str, Any]],
    *,
    caller_from: Optional[str] = None,
    call_sid: Optional[str] = None,
) -> dict[str, Any]:
    """
    Store external customer data as unconfirmed per-call prefill candidates.

    Values are validated through ``HelocApplicationState`` assignment but are not
    written into ``application_state`` until the caller explicitly confirms.
    """
    call_state = _current_call_state()
    customer_prefill = call_state.customer_prefill or {}
    customer_prefill.clear()
    call_state.customer_prefill = customer_prefill
    call_state.prefill_confirm_pending = None
    call_state.prefill_rejected_fields = set()
    call_state.last_prefill_yes_mono = 0.0
    call_state.last_prefill_yes_fields = frozenset()
    _clear_pending_low_confidence_confirmation()
    if not payload:
        return {}

    validator_state = HelocApplicationState()
    for raw_key, raw_value in payload.items():
        field_name = _normalize_prefill_key(str(raw_key))
        if field_name not in _PREFILL_ALLOWED_FIELDS:
            continue
        value = _normalize_prefill_value(field_name, raw_value)
        if value is None:
            continue
        try:
            setattr(validator_state, field_name, value)
        except (ValidationError, ValueError) as e:
            logger.warning(
                "Ignoring invalid prefill field %s: %s",
                field_name,
                str(e).split("\n")[-1],
            )
            continue
        customer_prefill[field_name] = getattr(validator_state, field_name)

    if customer_prefill:
        logger.info(
            "Customer prefill loaded | fields=%s caller_last4=%s call_sid=%s",
            sorted(customer_prefill.keys()),
            re.sub(r"\D", "", caller_from or "")[-4:] or "unknown",
            call_sid or "unknown",
        )
    return dict(customer_prefill)


def get_customer_prefill_fields() -> list[str]:
    return sorted((_current_call_state().customer_prefill or {}).keys())


def _format_prefill_value(field_name: str, value: Any) -> str:
    if field_name == "is_currently_for_sale":
        return "currently for sale" if value else "not currently for sale"
    if field_name in {"annual_pre_tax_income", "other_income"}:
        try:
            return f"${int(value):,}"
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _prefill_prompt_for_values(values: dict[str, Any]) -> str:
    if "first_name" in values and "last_name" in values:
        return (
            "I have your legal name as "
            f"{values['first_name']} {values['last_name']}. Is that correct?"
        )
    field_name, value = next(iter(values.items()))
    label = _field_display_name(field_name)
    if field_name == "address_line_1":
        parts = [str(value)]
        city = values.get("city")
        state = values.get("state")
        zipcode = values.get("zipcode")
        if city:
            parts.append(str(city))
        if state:
            parts.append(str(state))
        if zipcode:
            parts.append(str(zipcode))
        return (
            "I have the property address on file as "
            f"{', '.join(parts)}. Is that correct?"
        )
    if field_name == "ownership_type":
        prefix = (
            "Got it, I've saved your address. "
            if application_state.address_line_1 not in (None, "")
            else ""
        )
        if value == "Sole owner":
            return (
                f"{prefix}Regarding ownership, I have you down as the sole owner. "
                "Is that still correct?"
            )
        if value == "Joint owner":
            return (
                f"{prefix}Regarding ownership, I have you down as a joint owner. "
                "Is that still correct?"
            )
    if field_name == "occupancy_type":
        return (
            "I have this listed as "
            f"{_format_prefill_value(field_name, value).lower()}. Is that still correct?"
        )
    if field_name == "mobile_phone_number":
        return f"I have your mobile number as {value}. Is that correct?"
    if field_name == "email_address":
        return f"I have your email as {value}. Is that correct?"
    if field_name == "is_currently_for_sale":
        return f"I have the property as {_format_prefill_value(field_name, value)}. Is that still correct?"
    if field_name == "primary_home_address_line_1":
        parts = [str(value)]
        for ek in ("primary_home_city", "primary_home_state", "primary_home_zipcode"):
            ev = values.get(ek)
            if ev:
                parts.append(str(ev))
        return (
            "I have your primary home address on file as "
            f"{', '.join(parts)}. Is that correct?"
        )
    return (
        f"I have your {label} as "
        f"{_format_prefill_value(field_name, value)}. Is that still correct?"
    )


def _prefill_values_for_field(field_name: str, stage_num: int) -> dict[str, Any]:
    call_state = _current_call_state()
    customer_prefill = call_state.customer_prefill or {}
    rejected_fields = call_state.prefill_rejected_fields or set()
    if field_name in rejected_fields or field_name not in customer_prefill:
        return {}
    if getattr(application_state, field_name, None) not in (None, ""):
        return {}
    values = {field_name: customer_prefill[field_name]}
    if field_name == "address_line_1":
        for extra_field in ("apt_suite", "city", "state", "zipcode"):
            if (
                extra_field in customer_prefill
                and extra_field not in rejected_fields
                and getattr(application_state, extra_field, None) in (None, "")
            ):
                values[extra_field] = customer_prefill[extra_field]
    if field_name == "primary_home_address_line_1":
        for extra_field in (
            "primary_home_apt_suite",
            "primary_home_city",
            "primary_home_state",
            "primary_home_zipcode",
        ):
            if (
                extra_field in customer_prefill
                and extra_field not in rejected_fields
                and getattr(application_state, extra_field, None) in (None, "")
            ):
                values[extra_field] = customer_prefill[extra_field]
    return values


def _set_prefill_confirm_pending(values: dict[str, Any]) -> str:
    prompt = _prefill_prompt_for_values(values)
    _current_call_state().prefill_confirm_pending = {"values": dict(values), "ask": prompt}
    return prompt


def has_pending_prefill_confirmation() -> bool:
    return bool(_current_call_state().prefill_confirm_pending)


_AGENT_SELF_INTRO = (
    "This is Alex, a digital AI assistant with West Capital Lending — and this call is recorded. "
)


def get_initial_intake_ask() -> Optional[str]:
    """Return the current deterministic Ask next for dynamic call introductions."""
    first_name = str((_current_call_state().customer_prefill or {}).get("first_name") or "").strip()
    hi = f"Hi {first_name}!" if first_name else "Hi there!"
    full_intro = f"{hi} {_AGENT_SELF_INTRO}"

    # In production this is always called at call-start (GREETING stage).
    # When GREETING, return the full self-intro + good-time-to-talk ask.
    if get_current_conversation_stage() == "GREETING":
        return full_intro + "Is now a good time to go through a few quick questions about your hee-lok application?"

    # Past GREETING (test scaffolding / edge case): return a concise field ask.
    ask = _ask_from_missing_line(get_missing_fields() or "")
    if ask:
        ask = ask.replace("HELOC", "hee-lok")

    greeting = f"Hi {first_name}, " if first_name else ""

    _PROPERTY_ASKS = {
        "What is the property address?",
        "Can you tell me the property address?",
        "What's your property address for financing?",
    }
    if not ask or ask in _PROPERTY_ASKS:
        if first_name:
            return f"Hi {first_name} — what's your property address for financing?"
        return None
    if first_name and ask.startswith("I have "):
        return greeting + ask
    if first_name and not ask.strip().lower().startswith("hi "):
        return greeting + ask
    return ask


def _current_prefill_confirmation_values() -> dict[str, Any]:
    """
    Recover the active on-file confirmation values if a stale TTS/LLM path lost
    ``_prefill_confirm_pending`` before the caller answered yes/no.
    """
    pending = _current_call_state().prefill_confirm_pending
    if pending:
        return dict(pending.get("values") or {})
    missing_line = get_missing_fields() or ""
    if not pending:
        return {}
    ask = pending.get("ask") or ""
    if ask and ask in missing_line:
        return dict(pending.get("values") or {})
    return {}


def _clear_prefill_confirm_pending() -> None:
    _current_call_state().prefill_confirm_pending = None


def _extract_inline_answer_after_no(
    text: str, rejected_fields: dict[str, Any]
) -> dict[str, Any]:
    tail = re.sub(
        r"^(?:uh[.,]?\s*|um[.,]?\s*)*"
        r"(?:no|nope|nah)[.,!?\s]+"
        r"(?:it'?s|it\s+is|the\s+correct\s+one\s+is|actually)?\s*",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    ).strip()
    if not tail or len(tail) < 3:
        return {}
    result: dict[str, Any] = {}
    if "mobile_phone_number" in rejected_fields:
        ph = infer_phone_from_transcript(tail)
        if ph:
            result["mobile_phone_number"] = ph
    if "email_address" in rejected_fields:
        em = infer_email_from_transcript(tail)
        if em:
            result["email_address"] = em
    if "first_name" in rejected_fields or "last_name" in rejected_fields:
        parsed = infer_name_fields_from_transcript(
            tail,
            need_first="first_name" in rejected_fields,
            need_last="last_name" in rejected_fields,
            slot_focus="first" if "first_name" in rejected_fields else "last",
        )
        if parsed:
            result.update(parsed)
    if "ownership_type" in rejected_fields:
        ot = infer_ownership_from_transcript(tail)
        if ot:
            result["ownership_type"] = ot
    return result


def has_pending_low_confidence_confirmation() -> bool:
    """True when a low-confidence candidate is waiting for repeat/confirm handling."""
    return bool(_current_call_state().pending_low_conf_confirmation)


def _ask_from_missing_line(missing_line: str) -> str:
    m = re.search(r'Ask next:\s*"([^"]+)"', missing_line or "")
    if not m:
        return ""
    return (m.group(1) or "").strip()


def looks_like_answer_for_current_field(text: str, missing_line: str) -> bool:
    """True when a transcript plausibly answers the current `Ask next` field."""
    if not text or not str(text).strip():
        return False
    ask = _ask_from_missing_line(missing_line).lower()
    low = text.lower()
    
    # Ownership ask?
    if "sole or joint" in ask:
        return bool(infer_ownership_from_transcript(low))
    # Occupancy ask?
    if "is this your primary" in ask:
        return bool(infer_occupancy_from_transcript(low))
    # For-sale ask?
    if "currently listed for sale" in ask or "currently for sale" in ask:
        return infer_yes_no_from_transcript(low) is not None
    # Income ask?
    if "annual pre-tax income" in ask:
        return bool(infer_annual_income_from_transcript(low))
    # Birthday ask?
    if "date of birth" in ask:
        return bool(infer_dob_from_transcript(low))
    # Personal info (name/phone/email)?
    if "your first name" in ask or "your last name" in ask:
        return len(low.split()) >= 1
    if "phone number" in ask:
        return bool(infer_phone_from_transcript(low))
    if "email address" in ask:
        return bool(infer_email_from_transcript(low))
    # Property address? Require both a street number and a street name token.
    if "property address" in ask or "street address" in ask:
        return bool(infer_property_address_from_transcript(low))
    # Default: assume anything else might be an answer
    return True


def _field_display_name(field_name: str) -> str:
    names = {
        "address_line_1": "property address (street)",
        "primary_home_address_line_1": "primary home address (street)",
        "ownership_type": "ownership type",
        "occupancy_type": "property type",
        "is_currently_for_sale": "for-sale answer",
        "annual_pre_tax_income": "annual income",
        "other_income": "other income",
        "primary_source_of_income": "income source",
        "first_name": "first name",
        "last_name": "last name",
        "date_of_birth": "date of birth",
        "mobile_phone_number": "phone number",
        "email_address": "email address",
    }
    return names.get(field_name, field_name.replace("_", " "))


def _pending_low_conf_ask() -> str:
    pending = _current_call_state().pending_low_conf_confirmation or {}
    values = pending.get("values") or {}
    attempts = int(pending.get("attempts") or 1)
    if "annual_pre_tax_income" in values:
        amt = int(values["annual_pre_tax_income"])
        return (
            f"I heard annual pre-tax income of about ${amt:,}. "
            "Is that correct?"
        )
    if "other_income" in values:
        amt = int(values["other_income"])
        return (
            f"I heard other income of about ${amt:,}. "
            "Is that correct?"
        )
    if "first_name" in values:
        return f"I heard first name {values['first_name']}. Is that correct?"
    if "last_name" in values:
        return f"I heard last name {values['last_name']}. Is that correct?"
    if "address_line_1" in values:
        if attempts >= 2:
            return "Please spell the street address slowly; there may be background noise."
        return "I didn't catch that correctly. Please repeat the street address."
    if "primary_home_address_line_1" in values:
        if attempts >= 2:
            return "Please spell your primary home street slowly; there may be background noise."
        return "I didn't catch that correctly. Please repeat your primary home street address."
    if attempts >= 2:
        return "Please say it slowly again, or spell it if needed."
    return "I didn't catch that correctly. Can you repeat?"


def _set_pending_low_confidence_confirmation(
    values: dict[str, object], required_confidence: float
) -> None:
    call_state = _current_call_state()
    previous = call_state.pending_low_conf_confirmation or {}
    previous_values = previous.get("values") or {}
    attempts = int(previous.get("attempts") or 0)
    if set(previous_values.keys()) != set(values.keys()):
        attempts = 0
    call_state.pending_low_conf_confirmation = {
        "values": dict(values),
        "required_confidence": required_confidence,
        "attempts": attempts + 1,
    }


def _clear_pending_low_confidence_confirmation() -> None:
    _current_call_state().pending_low_conf_confirmation = None


# End-of-call snapshot: all captured applicant-facing fields (passwords never exported).
_HELOC_UI_FULL_EXPORT_KEYS: tuple[str, ...] = (
    "address_line_1",
    "apt_suite",
    "city",
    "state",
    "zipcode",
    "ownership_type",
    "occupancy_type",
    "primary_home_address_line_1",
    "primary_home_apt_suite",
    "primary_home_city",
    "primary_home_state",
    "primary_home_zipcode",
    "is_currently_for_sale",
    "annual_pre_tax_income",
    "other_income",
    "primary_source_of_income",
    "first_name",
    "middle_name",
    "last_name",
    "suffix",
    "date_of_birth",
    "mobile_phone_number",
    "email_address",
    "agreed_to_privacy_policy",
    "agreed_to_commerce_communication",
    "agreed_to_credit_pull",
)


def _heloc_ui_full_transcript_snapshot() -> dict[str, str]:
    """End-of-call NDJSON: every export key (empty string if unset). Passwords omitted."""
    st = application_state
    out: dict[str, str] = {}
    for key in _HELOC_UI_FULL_EXPORT_KEYS:
        if key in _SENSITIVE_FIELDS:
            continue
        v = getattr(st, key, None)
        if v is None:
            out[key] = ""
        elif isinstance(v, bool):
            out[key] = "True" if v else "False"
        elif isinstance(v, int):
            out[key] = str(v)
        elif isinstance(v, str):
            out[key] = v.strip()
        else:
            out[key] = str(v).strip() if v is not None else ""
    return out


def _maybe_record_voice_field_user_source(
    field_name: str, raw_user_input: Optional[str]
) -> None:
    """Remember verbatim STT line last accepted into ``field_name`` (not UI-normalized)."""
    if field_name in _SENSITIVE_FIELDS:
        return
    if not raw_user_input or not str(raw_user_input).strip():
        return
    sources = _current_call_state().voice_field_user_sources or {}
    sources[field_name] = str(raw_user_input).strip()
    _current_call_state().voice_field_user_sources = sources


def export_voice_field_user_sources_for_call() -> dict[str, str]:
    return dict(sorted((_current_call_state().voice_field_user_sources or {}).items()))


def append_heloc_ui_transcript_final_snapshot(call_sid: str | None = None) -> None:
    """Append full captured state at end of call (before reset). Passwords omitted.

    Writes one ``scope: full`` line with **all** ``_HELOC_UI_FULL_EXPORT_KEYS`` present
    (empty string when a field was never collected) so NDJSON always has a complete row.
    Also writes ``voice_fields/<call_sid>.json`` (verbatim user utterances per saved field).
    """
    try:
        from app.conversation_transcript import (
            append_transcript_ui_fields,
            write_voice_sourced_fields_json,
        )
    except ImportError:
        return
    snap = _heloc_ui_full_transcript_snapshot()
    if not snap:
        return
    try:
        append_transcript_ui_fields(call_sid, snap, scope="full")
    except Exception:
        logger.debug("append_transcript_ui_fields final failed", exc_info=True)
    try:
        write_voice_sourced_fields_json(
            call_sid, export_voice_field_user_sources_for_call()
        )
    except Exception:
        logger.debug("write_voice_sourced_fields_json failed", exc_info=True)


_reset_state_lock = asyncio.Lock()


async def reset_state() -> None:
    """Reset the application state for a new call."""
    async with _reset_state_lock:
        await _reset_state_impl()


async def _reset_state_impl() -> None:
    call_state = _current_call_state()
    call_state.application_state = HelocApplicationState()
    call_state.consent_decline_round = {}
    call_state.voice_field_user_sources = {}
    call_state.intake_published = False
    _clear_pending_low_confidence_confirmation()
    call_state.customer_prefill = {}
    call_state.prefill_confirm_pending = None
    call_state.prefill_rejected_fields = set()
    call_state.last_prefill_yes_mono = 0.0
    call_state.last_prefill_yes_fields = frozenset()
    logger.info("Application state reset for new call.")


def get_current_conversation_stage() -> str:
    """Determine the current stage of the conversation based on captured data."""
    cs = _current_call_state()
    if not cs.greeting_completed:
        return "GREETING"

    state = application_state

    # 1. PROPERTY: core collateral info
    if (not state.address_line_1 or
        not state.ownership_type or 
        not state.occupancy_type):
        return "PROPERTY"
    
    # 2. PRIMARY_HOME: only for second home/investment
    if _occupancy_requires_primary_home_intake(state.occupancy_type):
        if not state.primary_home_address_line_1:
            return "PRIMARY_HOME"
            
    # 3. FOR_SALE
    if state.is_currently_for_sale is None:
        return "FOR_SALE"
        
    # 4. INCOME
    if (not state.annual_pre_tax_income or 
        not state.primary_source_of_income):
        return "INCOME"
        
    # 5. PERSONAL (name collection)
    if not state.first_name or not state.last_name:
        return "PERSONAL"

    # 5b. BIRTHDAY (date of birth — separate sub-stage once name is captured)
    if not state.date_of_birth:
        return "BIRTHDAY"
        
    # 6. CONTACT
    if (not state.mobile_phone_number or 
        not state.email_address):
        return "CONTACT"
        
    # 7. CONSENT
    if (not state.agreed_to_privacy_policy or 
        not state.agreed_to_commerce_communication):
        return "CONSENT"
        
    return "COMPLETED"

def get_contact_buffer_mode() -> Optional[str]:
    """Return 'email' or 'phone' if we're in CONTACT stage and actively asking
    for that field, or None if not in the CONTACT stage."""
    if get_current_conversation_stage() != "CONTACT":
        return None
    state = application_state
    if not state.mobile_phone_number:
        return "phone"
    if not state.email_address:
        return "email"
    return None

def get_missing_fields() -> str:
    """
    Return a structured line for the LLM: current conversation stage + missing fields.
    """
    missing = []
    state = application_state
    call_state = _current_call_state()
    pending_low_conf = call_state.pending_low_conf_confirmation
    prefill_pending = call_state.prefill_confirm_pending
    rejected_fields = call_state.prefill_rejected_fields or set()
    
    current_conv_stage = get_current_conversation_stage()
    stage_header = f"[CONVERSATION STAGE: {current_conv_stage}]"

    if current_conv_stage == "GREETING":
        return (
            f"{stage_header} Ask next: "
            '"Is now a good time to go through a few quick questions about your hee-lok application?"'
        )

    if pending_low_conf:
        pending_values = pending_low_conf.get("values") or {}
        pending_labels = [
            _field_display_name(field_name) for field_name in pending_values.keys()
        ]
        label = ", ".join(pending_labels) or "current answer"
        return (
            f"{stage_header} Still needed: {label}. "
            f'Ask next: "{_pending_low_conf_ask()}"'
        )

    if prefill_pending:
        pending_values = prefill_pending.get("values") or {}
        active_values = {
            field_name: value
            for field_name, value in pending_values.items()
            if field_name not in rejected_fields
            and getattr(state, field_name, None) in (None, "")
        }
        if active_values:
            label = ", ".join(_field_display_name(f) for f in active_values.keys())
            return (
                f"{stage_header} Still needed: {label}. "
                f'Ask next: "{prefill_pending.get("ask")}"'
            )
        _clear_prefill_confirm_pending()

    # Stage to field mapping for detailed intake
    STAGE_FIELDS = {
        "PROPERTY": ["address_line_1", "ownership_type", "occupancy_type"],
        "PRIMARY_HOME": ["primary_home_address_line_1"],
        "FOR_SALE": ["is_currently_for_sale"],
        "INCOME": ["annual_pre_tax_income", "other_income", "primary_source_of_income"],
        "PERSONAL": ["first_name", "last_name", "middle_name", "suffix"],
        "BIRTHDAY": ["date_of_birth"],
        "CONTACT": ["mobile_phone_number", "email_address"],
        "CONSENT": ["agreed_to_privacy_policy", "agreed_to_commerce_communication"],
    }

    # Short scripted asks per field. Phrased to feel like a real conversation —
    # warm openers for first-touch questions, casual follow-ups thereafter.
    field_to_question = {
        "address_line_1": "What's your property address for financing?",
        "ownership_type": VOICE_SCRIPT_OWNERSHIP_ASK,
        "occupancy_type": VOICE_SCRIPT_OCCUPANCY_ASK,
        "primary_home_address_line_1": "And where do you currently live? Just the street address is fine.",
        "is_currently_for_sale": VOICE_SCRIPT_FOR_SALE_ASK,
        "annual_pre_tax_income": "Roughly, what's your annual income before taxes?",
        "other_income": "Any other income to factor in — rental, investment, retirement, anything like that?",
        "primary_source_of_income": "And how do you earn most of that — employment, self-employed, retirement, or something else?",
        "first_name": "What's your first name?",
        "middle_name": "Do you have a middle name?",
        "last_name": "And your last name?",
        "suffix": "Any suffix like Jr or Sr?",
        "date_of_birth": "And what's your date of birth?",
        "mobile_phone_number": "What's the best phone number to reach you on?",
        "email_address": "And the best email for documents?",
        "agreed_to_privacy_policy": "To move forward, do you agree to our privacy policy and a soft credit pull?",
        "agreed_to_commerce_communication": "And do you agree to receive electronic communications from us?",
    }

    # Get fields for current stage and check which are missing
    stage_fields = STAGE_FIELDS.get(current_conv_stage, [])
    missing_in_stage = []
    optional_fields = {"middle_name", "suffix"}

    for field in stage_fields:
        if getattr(state, field, None) in (None, ""):
            if field not in optional_fields:
                missing_in_stage.append(field)

    # If all required fields in the current stage are filled, ask optional ones —
    # BUT skip other_income if income source is already captured (user moved past it)
    # and skip middle_name/suffix only when they haven't been asked at all.
    if not missing_in_stage and current_conv_stage != "COMPLETED":
        for field in stage_fields:
            if getattr(state, field, None) in (None, ""):
                # Guard: if income source is already saved, don't backtrack to other_income
                if field == "other_income" and state.primary_source_of_income is not None:
                    continue
                missing_in_stage.append(field)

    if not missing_in_stage and current_conv_stage != "COMPLETED":
        # Current stage is fully done, move to next
        return f"{stage_header} Still needed: None. [STAGE COMPLETE]"

    if current_conv_stage == "COMPLETED":
        return f"{stage_header} All fields captured. Ready to terminate."

    # Format the "Still needed" labels
    labels = [_field_display_name(f) for f in missing_in_stage]
    still_needed_str = ", ".join(labels)
    
    # Get the "Ask next" question for the first missing field in this stage
    first_missing = missing_in_stage[0]
    # If the first missing field has a prefill candidate, ask to confirm it
    # and register the pending confirmation so the heuristic can save state
    # when the user says yes/no.
    if first_missing in (call_state.customer_prefill or {}) and first_missing not in rejected_fields:
        val = call_state.customer_prefill[first_missing]
        disp_val = _format_prefill_value(first_missing, val)
        
        # Build a natural confirmation ask
        if first_missing == "ownership_type":
            confirm_ask = f"I see you're the {disp_val.lower()} of the property, is that correct?"
        elif first_missing == "occupancy_type":
            confirm_ask = f"And is this your {disp_val.lower()}, or a different type of property?"
        elif first_missing == "is_currently_for_sale":
            confirm_ask = f"I have that the property is {disp_val.lower()}, is that still right?"
        elif first_missing == "address_line_1":
            parts = [str(val)]
            prefill = call_state.customer_prefill or {}
            for ek in ("city", "state", "zipcode"):
                ev = prefill.get(ek)
                if ev:
                    parts.append(str(ev))
            confirm_ask = f"I have the property address on file as {', '.join(parts)}. Is that correct?"
        else:
            confirm_ask = f"I have your {_field_display_name(first_missing)} as {disp_val}. Is that correct?"
        
        # CRITICAL: Register this as a pending prefill confirmation so the
        # heuristic layer can intercept the user's yes/no response and write
        # the value to state. Without this, the ask repeats endlessly.
        if not call_state.prefill_confirm_pending:
            call_state.prefill_confirm_pending = {
                "values": {first_missing: val},
                "ask": confirm_ask,
            }
            
        return (
            f"{stage_header} Still needed: {_field_display_name(first_missing)}. "
            f'Ask next: "{confirm_ask}"'
        )

    ask_next = field_to_question.get(first_missing, f"What is your {first_missing}?")

    return f"{stage_header} Still needed: {still_needed_str}. Ask next: \"{ask_next}\""



def build_intake_llm_context_string(*, max_total_chars: int = 6000) -> str:
    """
    Per-user-turn snapshot for LlmAgent ``process(..., context=...)``.

    Rendered as a transient system CustomHistoryEntry (not the user's message),
    so ``raw_user_input`` stays the true transcript.
    """
    call_state = _current_call_state()
    stage = _CONV_STAGE_TO_INTAKE_ID.get(get_current_conversation_stage(), "unknown")
    missing = get_missing_fields() or ""
    data = application_state.model_dump(
        mode="json",
        exclude_none=True,
        exclude=_SENSITIVE_FIELDS,
    )
    captured = json.dumps(data, default=str)

    # Empathy flag — LLM must not repeat acknowledgement if already shown.
    empathy_line = (
        "empathy_already_shown=true — do NOT repeat empathy this turn, move forward naturally."
        if call_state.empathy_shown
        else "empathy_already_shown=false"
    )

    # Surface optional-field status so the LLM never backtracks to re-ask them.
    optional_notes: list[str] = []
    st = application_state
    if st.annual_pre_tax_income is not None and st.other_income is None:
        optional_notes.append(
            "other_income=NOT_YET_ASKED (will auto-default to 0 if caller skips it; "
            "do NOT ask for other income if income source is the current Ask next)"
        )
    if st.other_income is not None:
        optional_notes.append(f"other_income=CAPTURED ({st.other_income})")
    if optional_notes:
        optional_line = "optional_fields: " + "; ".join(optional_notes)
    else:
        optional_line = ""

    body_parts = [
        "[CURRENT_INTAKE_STATE — authoritative for this turn; trust over chat memory]",
        f"intake_stage={stage}",
        f"captured_fields_json={captured}",
        empathy_line,
    ]
    if optional_line:
        body_parts.append(optional_line)
    body_parts += [
        "still_needed_line:",
        missing,
        "[/CURRENT_INTAKE_STATE]",
    ]
    body = "\n".join(body_parts)

    if len(body) > max_total_chars:
        body = (
            body[: max_total_chars - 60].rstrip()
            + "\n... [truncated]\n[/CURRENT_INTAKE_STATE]"
        )
    return body


# =============================================================================
# Confidence threshold for field extraction. If the LLM's self-assessed
# confidence is below this, the tool instructs re-confirmation before
# moving to the next question.
# =============================================================================
# Constants and parsers moved to agent/deterministic_parsers.py.











def _spoken_email_evidence(raw_user_input: Optional[str]) -> bool:
    """True if the raw transcript contains a plausible email utterance (@ or spoken at…dot)."""
    if not raw_user_input:
        return False
    low = raw_user_input.lower()
    if "@" in low:
        return True
    return bool(re.search(r"\bat\b.*\bdot\b", low))


def _email_save_matches_transcript(raw_user_input: Optional[str], email: str) -> bool:
    """True if the email the LLM wants to save matches what the deterministic parser
    extracts from the raw transcript."""
    parsed = infer_email_from_transcript(raw_user_input or "")
    if parsed is None:
        return False
    return parsed.lower() == email.lower()


# =============================================================================
# Loopback Tool — The LLM calls this after extracting data from user speech.
#
# How it works:
# 1. User says: "I live at 742 Evergreen Terrace in Springfield, Illinois"
# 2. LLM extracts: address_line_1="742 Evergreen Terrace", city="Springfield",
#    state="IL", confidence_score=0.95
# 3. LLM calls this tool with those 3 fields (other params stay None)
# 4. Tool updates state, returns: "Updated: street address, city, state.
#    Still needed: ownership type, ..." (zip comes from site autocomplete when possible)
# 5. LLM reads the return value and asks the caller for the next missing item.
#
# Confidence scoring:
# - If confidence < 0.85, field is saved but flagged for re-confirmation.
# - Legal consent fields always require high confidence.
# =============================================================================
@loopback_tool
async def update_application_state(
    ctx,
    address_line_1: Annotated[Optional[str], "Street address of the property"] = None,
    apt_suite: Annotated[Optional[str], "Apartment or suite number"] = None,
    city: Annotated[Optional[str], "City of the property"] = None,
    state: Annotated[Optional[str], "State of the property (2-letter abbreviation)"] = None,
    zipcode: Annotated[Optional[str], "5-digit zipcode"] = None,
    ownership_type: Annotated[Optional[OwnershipType], "Sole owner or Joint owner"] = None,
    occupancy_type: Annotated[Optional[OccupancyType], "Primary, Secondary, or Investment property"] = None,
    primary_home_address_line_1: Annotated[
        Optional[str],
        "Street address of borrower's primary residence where they live most of the year; "
        "only when the HELOC property is a second home or investment property.",
    ] = None,
    primary_home_apt_suite: Annotated[
        Optional[str], "Apartment or suite for primary residence, if applicable"
    ] = None,
    primary_home_city: Annotated[Optional[str], "City of primary residence"] = None,
    primary_home_state: Annotated[
        Optional[str], "State of primary residence (2-letter abbreviation)"
    ] = None,
    primary_home_zipcode: Annotated[Optional[str], "5-digit zipcode of primary residence"] = None,
    is_currently_for_sale: Annotated[Optional[bool], "Whether the property is currently listed for sale"] = None,
    annual_pre_tax_income: Annotated[Optional[int], "Total annual pre-tax income in whole dollars (must be greater than 0)"] = None,
    other_income: Annotated[Optional[int], "Additional income (rental, investment, etc.)"] = None,
    primary_source_of_income: Annotated[Optional[IncomeSource], "Primary source of annual income"] = None,
    first_name: Annotated[Optional[str], "User's legal first name"] = None,
    middle_name: Annotated[Optional[str], "User's legal middle name or initial"] = None,
    last_name: Annotated[Optional[str], "User's legal last name"] = None,
    suffix: Annotated[Optional[str], "Name suffix (Jr, Sr, III, etc.)"] = None,
    date_of_birth: Annotated[Optional[str], "Date of birth as MM/DD/YYYY"] = None,
    mobile_phone_number: Annotated[Optional[str], "10-digit mobile phone number"] = None,
    email_address: Annotated[Optional[str], "Email address"] = None,
    agreed_to_privacy_policy: Annotated[Optional[bool], "User says 'I agree' to privacy, terms, and soft credit (first checkbox)"] = None,
    agreed_to_commerce_communication: Annotated[Optional[bool], "User says 'I agree' to electronic communications (second checkbox)"] = None,
    agreed_to_credit_pull: Annotated[Optional[bool], "Internal: mirrored when first checkbox is set; do not set alone"] = None,
    confidence_score: Annotated[Optional[float], "Your confidence (0.0-1.0) in the accuracy of the extracted data. Set lower if the caller was unclear, mumbled, or you had to guess."] = None,
    raw_user_input: Annotated[Optional[str], "The exact transcript of what the user just said. Always pass this so the system can detect uncertainty."] = None,
    deterministic_call: Annotated[Optional[bool], "Internal use only: deterministic parser call path."] = None,
) -> str:
    """
    Save extracted application data from the user's speech. Call this tool
    whenever the user provides any information relevant to their HELOC
    application. You can update one or many fields at a time — only pass
    the fields you have new information for.
    
    IMPORTANT:
    - Always include a confidence_score (0.0 to 1.0) reflecting how confident
      you are in the extracted values.
    - Always pass raw_user_input with the exact user transcript so the system
      can detect uncertainty markers like 'maybe', 'I think', 'or', etc.
    """
    global application_state
    
    # Collect which fields were actually provided in this call
    updates = {}
    validation_errors = []
    local_vars = {
        "address_line_1": address_line_1,
        "apt_suite": apt_suite,
        "city": city,
        "state": state,
        "zipcode": zipcode,
        "ownership_type": ownership_type,
        "occupancy_type": occupancy_type,
        "primary_home_address_line_1": primary_home_address_line_1,
        "primary_home_apt_suite": primary_home_apt_suite,
        "primary_home_city": primary_home_city,
        "primary_home_state": primary_home_state,
        "primary_home_zipcode": primary_home_zipcode,
        "is_currently_for_sale": is_currently_for_sale,
        "annual_pre_tax_income": annual_pre_tax_income,
        "other_income": other_income,
        "primary_source_of_income": primary_source_of_income,
        "first_name": first_name,
        "middle_name": middle_name,
        "last_name": last_name,
        "suffix": suffix,
        "date_of_birth": date_of_birth,
        "mobile_phone_number": mobile_phone_number,
        "email_address": email_address,
        "agreed_to_privacy_policy": agreed_to_privacy_policy,
        "agreed_to_commerce_communication": agreed_to_commerce_communication,
        "agreed_to_credit_pull": agreed_to_credit_pull,
    }

    # Stage-gate accepted fields so off-page hallucinated saves do not derail flow.
    stage_to_allowed_fields = {
        "property": {
            "address_line_1",
            "apt_suite",
            "city",
            "state",
            "zipcode",
            "ownership_type",
            "occupancy_type",
        },
        "primary-home": {
            "primary_home_address_line_1",
            "primary_home_apt_suite",
            "primary_home_city",
            "primary_home_state",
            "primary_home_zipcode",
        },
        "for-sale": {"is_currently_for_sale"},
        "income": {"annual_pre_tax_income", "other_income", "primary_source_of_income"},
        "name": {"first_name", "middle_name", "last_name", "suffix"},
        "birthday": {"date_of_birth"},
        "contact": {"mobile_phone_number", "email_address"},
        "consent": {"agreed_to_privacy_policy", "agreed_to_commerce_communication"},
    }
    current_stage = _CONV_STAGE_TO_INTAKE_ID.get(get_current_conversation_stage(), "unknown")
    allowed_now = stage_to_allowed_fields.get(current_stage)
    # Caller may correct an earlier field after we have advanced ("Wait, the address
    # is wrong"). Accept fields owned by any *prior* stage; still block fields from
    # *future* stages (those would be LLM hallucinating ahead of the script).
    _stage_order = ("property", "primary-home", "for-sale", "income", "name", "birthday", "contact", "consent")
    if allowed_now is not None and current_stage in _stage_order:
        cur_idx = _stage_order.index(current_stage)
        prior_field_owners: set[str] = set()
        for s in _stage_order[: cur_idx + 1]:
            prior_field_owners.update(stage_to_allowed_fields.get(s, set()))
        allowed_now = prior_field_owners

    pending_low_conf = _current_call_state().pending_low_conf_confirmation
    if pending_low_conf and any(
        value is not None for value in local_vars.values()
    ):
        _clear_pending_low_confidence_confirmation()
        pending_low_conf = None

    if raw_user_input and pending_low_conf:
        pending_yes_no = infer_yes_no_from_transcript(raw_user_input)
        if pending_yes_no is True:
            pending_values = dict((pending_low_conf.get("values") or {}))
            _clear_pending_low_confidence_confirmation()
            if pending_values:
                return await update_application_state.func(
                    None,
                    **pending_values,
                    confidence_score=1.0,
                    raw_user_input=raw_user_input,
                    deterministic_call=deterministic_call,
                )
        elif pending_yes_no is False:
            _clear_pending_low_confidence_confirmation()
            missing = get_missing_fields()
            return (
                "Okay, let's try again. "
                + (missing if missing and missing.strip() else "")
            )

    _eff_conv_stage = get_current_conversation_stage()
    if raw_user_input and _eff_conv_stage == "PROPERTY":
        # Snapshot PRE-TURN state — local_vars already contains whatever the
        # caller (or LLM tool call) is trying to save this turn, so it can't be
        # used to decide "which question were we asking".
        _pre_address_saved = application_state.address_line_1 is not None
        _pre_ownership_saved = application_state.ownership_type is not None
        _pre_occupancy_saved = application_state.occupancy_type is not None

        # Address parser runs only when address is still missing pre-turn.
        if _use_deterministic_address_parsing() and not _pre_address_saved:
            inferred_addr = infer_property_address_from_transcript(raw_user_input)
            if inferred_addr:
                local_vars["address_line_1"] = inferred_addr
        # NEVER infer ownership/occupancy from the same utterance that gave the
        # address — the ownership parser is loose (matches \bone\b, \b1\b) and
        # addresses are full of those tokens (e.g. "It's one two three West
        # Adams" → "Sole owner" is wrong). Only attempt these inferences when
        # address was ALREADY saved before this turn started.
        if _pre_address_saved and not _pre_ownership_saved:
            inferred_o = infer_ownership_from_transcript(raw_user_input)
            if inferred_o:
                local_vars["ownership_type"] = inferred_o
        if _pre_address_saved and not _pre_occupancy_saved:
            if local_vars.get("ownership_type") is None:
                inferred_occ = infer_occupancy_from_transcript(raw_user_input)
                if inferred_occ:
                    local_vars["occupancy_type"] = inferred_occ
    if raw_user_input and _eff_conv_stage == "PRIMARY_HOME":
        if (
            _use_deterministic_address_parsing()
            and local_vars.get("primary_home_address_line_1") is None
            and application_state.primary_home_address_line_1 is None
        ):
            inferred_ph = infer_property_address_from_transcript(raw_user_input)
            if inferred_ph:
                local_vars["primary_home_address_line_1"] = inferred_ph
    if raw_user_input and _eff_conv_stage == "FOR_SALE":
        if local_vars.get("is_currently_for_sale") is None:
            inferred_fs = infer_for_sale_from_transcript(raw_user_input)
            if inferred_fs is not None:
                local_vars["is_currently_for_sale"] = inferred_fs

    score = confidence_score if confidence_score is not None else 1.0
    is_low_confidence = score < CONFIDENCE_THRESHOLD
    
    matched_markers = detect_uncertainty(raw_user_input or "")
    if deterministic_call:
        _money_only = frozenset({"annual_pre_tax_income", "other_income"})
        pending_money = frozenset(
            k for k, v in local_vars.items() if v is not None and k in _money_only
        )
        if pending_money and pending_money <= _money_only:
            scaffold_scan = normalize_answer_scaffold(raw_user_input or "") or ""
            matched_markers = detect_uncertainty(scaffold_scan)

    if matched_markers:
        pending_update = {k: v for k, v in local_vars.items() if v is not None}
        if pending_update:
            markers_list = ", ".join(f"'{m}'" for m in matched_markers)
            logger.warning(
                "Uncertainty detected — markers: %s | fields: %s | raw: '%s'",
                markers_list, list(pending_update.keys()), raw_user_input,
            )
            _set_pending_low_confidence_confirmation(pending_update, CONFIDENCE_THRESHOLD)
            conf_ask = _pending_low_conf_ask()
            return (
                f"Update PENDING — uncertainty detected ({markers_list}). "
                f'Ask next: "{conf_ask}"'
            )
    
    low_conf_pending = []
    low_conf_values: dict[str, object] = {}
    pending_conf_requirements: dict[str, float] = {}
    for field_name, value in local_vars.items():
        if value is not None:
            if allowed_now is not None and field_name not in allowed_now:
                logger.info(
                    "Ignoring off-stage field '%s' while stage=%s",
                    field_name,
                    current_stage,
                )
                continue
            # -- Special handling: legal consent fields require explicit agreement --
            if field_name in HIGH_CONFIDENCE_FIELDS:
                if is_low_confidence or not value:
                    if field_name == "agreed_to_commerce_communication":
                        consent_label = "electronic communications policy"
                    else:
                        consent_label = (
                            "Privacy Policy, Terms of Service, and soft credit pull"
                        )
                    validation_errors.append(
                        f"Consent not clear for {field_name}. "
                        f"Please ask the user to explicitly say 'I agree' to the {consent_label}."
                    )
                    logger.warning(
                        "Consent field %s rejected — confidence=%.2f, value=%s",
                        field_name, score, value,
                    )
                    continue  # Do NOT save vague consent
            if field_name == "email_address" and value is not None:
                if not _spoken_email_evidence(raw_user_input):
                    validation_errors.append(
                        "email_address rejected: caller speech did not contain an email "
                        "(need '@' or spoken 'at … dot …'). Do not guess from name or context; "
                        "ask them to give the address slowly."
                    )
                    logger.warning(
                        "email_address rejected — no spoken evidence | raw=%r",
                        (raw_user_input or "")[:200],
                    )
                    continue
                if not _email_save_matches_transcript(raw_user_input, str(value)):
                    validation_errors.append(
                        "email_address rejected: transcript does not parse to the same "
                        "address you tried to save — only save what they literally said "
                        "(e.g. full local part then at domain dot com)."
                    )
                    logger.warning(
                        "email_address rejected — transcript mismatch | raw=%r saved=%r",
                        (raw_user_input or "")[:200],
                        _redact_if_sensitive("email_address", value),
                    )
                    continue
            if field_name == "date_of_birth" and value is not None:
                if not _transcript_supports_saved_dob(raw_user_input, str(value)):
                    validation_errors.append(
                        "date_of_birth rejected: caller speech did not clearly state that "
                        "full date (month, day, and year). Ask them for a complete date."
                    )
                    logger.warning(
                        "date_of_birth rejected — transcript lacks evidence | raw=%r dob=%r",
                        (raw_user_input or "")[:200],
                        value,
                    )
                    continue
            field_threshold = (
                ADDRESS_CONFIDENCE_THRESHOLD
                if field_name in ADDRESS_CONFIDENCE_FIELDS
                else CONFIDENCE_THRESHOLD
            )
            field_low_confidence = score < field_threshold
            if field_low_confidence:
                low_conf_pending.append(f"{field_name}='{value}'")
                low_conf_values[field_name] = value
                pending_conf_requirements[field_name] = field_threshold
                continue  # Save only after explicit reconfirmation
            
            # -- Special handling: income must be positive --
            if field_name == "annual_pre_tax_income" and value is not None and value <= 0:
                validation_errors.append(
                    f"Invalid data for annual_pre_tax_income: ${value}. "
                    "Annual income must be greater than $0. "
                    "Please explain this to the user and ask again."
                )
                continue
            
            try:
                setattr(application_state, field_name, value)
                updates[field_name] = value
                _maybe_record_voice_field_user_source(field_name, raw_user_input)

            except (ValidationError, ValueError) as e:
                # Validator rejected this value — tell the LLM what went wrong
                # so it can ask the caller to correct it.
                error_msg = str(e).split("\n")[-1].strip() if "\n" in str(e) else str(e)
                validation_errors.append(
                    f"Invalid data for {field_name}. {error_msg} "
                    "Please explain the requirement to the user and ask again."
                )
                logger.warning(
                    "Validation failed for %s='%s': %s",
                    field_name,
                    _redact_if_sensitive(field_name, value),
                    error_msg,
                )

    if (
        "agreed_to_privacy_policy" in updates
        and "agreed_to_credit_pull" not in updates
    ):
        pv = updates["agreed_to_privacy_policy"]
        if pv is True:
            updates["agreed_to_credit_pull"] = True
            setattr(application_state, "agreed_to_credit_pull", True)
            _maybe_record_voice_field_user_source(
                "agreed_to_credit_pull", raw_user_input
            )
        elif pv is False:
            updates["agreed_to_credit_pull"] = False
            setattr(application_state, "agreed_to_credit_pull", False)
            _maybe_record_voice_field_user_source(
                "agreed_to_credit_pull", raw_user_input
            )
    
    if low_conf_pending:
        pending = "; ".join(low_conf_pending)
        required_conf = (
            max(pending_conf_requirements.values())
            if pending_conf_requirements
            else CONFIDENCE_THRESHOLD
        )
        _set_pending_low_confidence_confirmation(low_conf_values, required_conf)
        retry_ask = _pending_low_conf_ask()
        missing = get_missing_fields()
        return (
            f"Update PENDING (low confidence {score:.0%}). "
            f"Did not save: {pending}. "
            "Ask user again for same field(s), then call update_application_state with "
            f"confidence_score >= {required_conf:.2f}. Ask next: \"{retry_ask}\"."
            + (f" {missing}" if missing and missing.strip() else "")
        )

    if updates:
        call_state = _current_call_state()
        conv_stage = get_current_conversation_stage()
        if conv_stage == "COMPLETED" and not call_state.intake_published:
            call_state.intake_published = True
            try:
                voice_sources = dict(call_state.voice_field_user_sources or {})
                message_id = await publish_completed_intake(
                    _current_call_state().application_state,
                    call_sid=call_state.call_sid,
                    caller_from=call_state.caller_from,
                    caller_to=call_state.caller_to,
                    voice_field_sources=voice_sources,
                )
                logger.info(
                    "Intake published to queue | message_id=%s call_sid=%s",
                    message_id,
                    call_state.call_sid,
                )
            except Exception as e:
                call_state.intake_published = False  # allow retry on next tool call
                logger.error("Failed to publish intake to queue: %s", e, exc_info=True)

    if not updates and not validation_errors:
        missing = get_missing_fields()
        return (
            "No fields were updated. Do not call this tool when the user gave no extractable "
            "data (e.g. only 'you there?'). You must still speak to the caller — acknowledge "
            "briefly, then ask the next question from Still needed. "
            + (missing if missing.strip() else "")
        )
    
    # Build response parts
    parts = []
    
    # Report validation errors first so the LLM knows to re-ask
    if validation_errors:
        parts.append(f"VALIDATION ERRORS: {'; '.join(validation_errors)}")
    
    if updates:
        updated_names = ", ".join(updates.keys())
        logger.info("State updated — fields: %s (confidence: %.2f)", updated_names, score)
        logger.debug(
            "Current state: %s",
            _redact_sensitive_map(application_state.model_dump(exclude_none=True)),
        )
        parts.append(f"Updated: {updated_names}")
    
    # Build the feedback message for the LLM
    missing = get_missing_fields()
    
    if not missing:
        parts.append(
            "All required fields collected."
        )
    else:
        # 'missing' is now a pre-formatted string from get_missing_fields()
        parts.append(missing)
    
    return ". ".join(parts) + "."


async def apply_voice_intake_heuristics_before_line(
    transcript: str,
    *,
    stt_confidence: Optional[float] = None,
    stt_recovery_text: Optional[str] = None,
) -> HeuristicResult:
    """
    Pre-Line heuristic hook (deterministic-only safety net).

    Returns HeuristicResult with handled=True when parser confidently
    mapped input to a field update, or handled=False to delegate to LLM.

    Scope intentionally deterministic:
      - property ownership (sole/joint)
      - property occupancy (primary/secondary/investment)
      - for-sale yes/no
      - annual/other income numeric inference
      - primary income source enum mapping
      - explicit don't-know / decline phrases only → conservative defaults (not when
        infer simply fails; see transcript_requests_intake_default)
      - DOB date parsing
      - phone digit parsing
      - email normalization parser
      - consent yes/no

    Free-text address fields remain LLM-first to avoid premature autocomplete.
    """
    _NOT_HANDLED = HeuristicResult(handled=False)

    if os.getenv("PATH2_ENABLE_PRELINE_HEURISTICS", "1").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return _NOT_HANDLED

    if not transcript or not str(transcript).strip():
        return _NOT_HANDLED

    def _extract_ask_next(tool_result: str) -> Optional[str]:
        return _ask_from_missing_line(tool_result) or None

    def _default_assumption_ask(field: str, ask: Optional[str]) -> Optional[str]:
        if not ask:
            return None
        if field == "ownership_type":
            return (
                "I'm assuming sole owner for now. "
                "Is this your primary home, a second home, or an investment property?"
            )
        if field == "occupancy_type":
            return (
                "I'm assuming primary home for now. "
                + VOICE_SCRIPT_FOR_SALE_ASK
            )
        if field == "is_currently_for_sale":
            return (
                "I'm assuming it is not listed for sale for now. "
                "Roughly, what's your annual income before taxes?"
            )
        if field == "annual_pre_tax_income":
            return (
                "I'm assuming one hundred thousand dollars annual income for now. "
                "Any other income?"
            )
        if field == "other_income":
            return (
                "I'm assuming no other income for now. "
                "What is your primary income source?"
            )
        if field == "primary_source_of_income":
            return (
                "I'm assuming employed full-time as income source for now. "
                "What's your first name?"
            )
        return ask

    def _with_save_ack(field: str, ask: Optional[str]) -> Optional[str]:
        if not ask:
            return None
        low = ask.strip().lower()
        if low.startswith(
            (
                "i didn't",
                "i couldn't",
                "please",
                "to continue",
                "do you agree",
                "i'm assuming",
            )
        ):
            return ask
        if field == "address_line_1":
            # Next question often already opens with "Got it, I've saved your address." — don't double-prefix.
            if low.startswith("got it,") or "i've saved your address" in low:
                return ask
            return f"Got it, I'll pull that up. {ask}"
        if field == "primary_home_address_line_1":
            if "primary home" in low or "saved" in low:
                return ask
            low_ask = ask.strip().lower()
            # "I'll pull that up" sounds like a yes/no confirmation; callers often say "yes"
            # before the for-sale question lands. Use neutral bridge before binary for-sale.
            if "for sale" in low_ask:
                return f"Thanks for that. {ask}"
            return f"Got it, I'll pull that up. {ask}"
        return ask

    def _ok(
        ask: Optional[str],
        field: str,
        stage: str,
        *,
        terminate_after_tts: bool = False,
    ) -> HeuristicResult:
        logger.info(
            "deterministic_handled field=%s stage=%s",
            field,
            stage,
            extra={"event": "state.deterministic_handled", "field": field, "stage": stage},
        )
        # Stamp cooldown so trailing STT finals from the same utterance don't
        # bleed into confirming the next field (answer-bleeding guard).
        call_state._preline_last_handled_at = time.monotonic()
        return HeuristicResult(
            handled=True,
            ask_next=_with_save_ack(field, ask),
            handled_field=field,
            stage=stage,
            terminate_after_tts=terminate_after_tts,
        )

    def _fail(stage: str) -> HeuristicResult:
        return HeuristicResult(handled=False, stage=stage)

    text = (transcript or "").strip()
    if not text:
        return _NOT_HANDLED
    if transcript_is_status_check(text):
        return _NOT_HANDLED

    call_state = _current_call_state()

    # Answer-bleeding guard: if the preline just fired (e.g. confirmed first name),
    # ignore STT finals that arrive within the cooldown window — they are trailing
    # fragments of the same utterance, not deliberate answers to the next question.
    _now = time.monotonic()
    if _now - call_state._preline_last_handled_at < _PRELINE_COOLDOWN_SECS:
        logger.debug(
            "preline cooldown active — skipping | age_ms=%.0f",
            (_now - call_state._preline_last_handled_at) * 1000,
        )
        return _NOT_HANDLED

    _conv_stage = get_current_conversation_stage()
    if _conv_stage == "PERSONAL":
        _pst = application_state
        stage = "birthday" if (_pst.first_name and _pst.last_name) else "name"
    else:
        stage = _CONV_STAGE_TO_INTAKE_ID.get(_conv_stage, "unknown")

    # DNC takes top priority at any stage — caller is asking us not to call
    # again. Firm goodbye, no callback offer, terminate immediately. This must
    # run before the emotional/identity bypass below so a DNC isn't routed to
    # the LLM where it might be mishandled.
    if transcript_is_dnc(text):
        return _ok(
            "Understood. You won't receive any more calls from us. Take care.",
            "dnc_request",
            stage,
            terminate_after_tts=True,
        )

    # Mid-intake decline ("I changed my mind", "let me think about it",
    # "I want to stop", "not available", "call back later", etc.) past the
    # GREETING stage — graceful goodbye + terminate. The GREETING branch below
    # handles greeting-stage soft declines separately so we don't double-fire.
    if stage not in ("greeting", "completed", "unknown") and transcript_is_intake_decline(text):
        return _ok(
            "Of course, no problem. We can pick this up another time. Take care!",
            "mid_intake_decline",
            stage,
            terminate_after_tts=True,
        )

    # EMOTIONAL & IDENTITY BYPASS — must run FIRST.
    # We bypass to the LLM if the user is emotional, asking for help, or
    # asking about the agent's identity (AI/robot).
    lower_text = transcript.lower()

    # Identity queries — let the LLM handle these directly.
    is_identity_query = bool(re.search(
        r"\bare\s+you\s+(?:an\s+)?(?:ai|robot|bot|person|human|real)\b|\br\s+u\s+n\s*[ai]\b|\bid\s+and\s+ai\b|\bare\s+you\s+there\b",
        lower_text,
    ))

    # Detect "keep going" / continuation signals — user is resuming after
    # emotional acknowledgement. These should fall through to normal flow.
    is_continuation = bool(re.search(
        r"\bkeep\s+going\b|\blet'?s\s+(?:keep|continue|go)\b|\bkeep\s+moving\b|\bi'?m\s+(?:fine|okay|ok|ready)\b",
        lower_text,
    ))

    if not is_continuation and (is_identity_query or re.search(
        r"\bsad\b|\baccident\b|\bconfused\b|\bscared\b|\bworried\b|\bstressed\b|\bupset\b|\bhurt\b|\bdepressed\b",
        lower_text,
    )):
        logger.info("Heuristic bypass: emotional or identity query detected in '%s'", transcript)
        # Only set empathy_shown if it was actually a distress keyword, not just "Are you there?"
        if not is_identity_query:
            call_state.empathy_shown = True
        return _NOT_HANDLED

    prefill_pending = call_state.prefill_confirm_pending
    yn_for_prefill = infer_yes_no_from_transcript(text)
    # Only enter prefill-confirmation path if there is EXPLICITLY a pending
    # confirmation registered. Do NOT open this path just because yn is not None
    # — that would catch ordinary off-topic yes/no utterances.
    prefill_confirmation_values = (
        _current_prefill_confirmation_values() if prefill_pending else {}
    )
    if prefill_confirmation_values:
        yn = yn_for_prefill
        if yn is None:
            ask = (
                (call_state.prefill_confirm_pending or {}).get("ask")
                or _prefill_prompt_for_values(prefill_confirmation_values)
            )
            return _ok(ask, "prefill_confirmation", stage)

        pending_values = dict(prefill_confirmation_values)

        _clear_prefill_confirm_pending()
        if yn is False:
            rejected_fields = call_state.prefill_rejected_fields or set()
            rejected_fields.update(pending_values.keys())
            call_state.prefill_rejected_fields = rejected_fields
            for fname in pending_values:
                if fname not in HelocApplicationState.model_fields:
                    continue
                try:
                    setattr(application_state, fname, None)
                except Exception:
                    logger.debug(
                        "prefill reject: could not clear %s",
                        fname,
                        exc_info=True,
                    )
            # Check if the rejection utterance also contains an inline answer.
            # e.g. "No, it's 413-275-6066" → reject prefill AND parse the phone.
            inline = _extract_inline_answer_after_no(text, pending_values)
            if inline:
                try:
                    tool_result = await update_application_state.func(
                        None,
                        **inline,
                        confidence_score=0.92,
                        raw_user_input=text,
                        deterministic_call=True,
                    )
                    return _ok(
                        _extract_ask_next(tool_result),
                        "prefill_confirmation_rejected",
                        stage,
                    )
                except Exception:
                    logger.debug("inline answer after prefill reject failed", exc_info=True)
            ask = _ask_from_missing_line(get_missing_fields() or "")
            return _ok(ask, "prefill_confirmation_rejected", stage)

        for field_name in pending_values:
            (call_state.customer_prefill or {}).pop(field_name, None)
        _clear_pending_low_confidence_confirmation()
        try:
            tool_result = await update_application_state.func(
                None,
                **pending_values,
                confidence_score=1.0,
                raw_user_input=text,
                deterministic_call=True,
            )
            call_state.last_prefill_yes_mono = time.monotonic()
            call_state.last_prefill_yes_fields = frozenset(pending_values.keys())
            return _ok(
                _extract_ask_next(tool_result),
                "prefill_confirmation",
                stage,
            )
        except Exception as e:
            logger.warning(
                "apply_voice_intake_heuristics_before_line prefill confirmation failed: %s",
                e,
                exc_info=True,
            )
            return _fail(stage)

    if has_pending_low_confidence_confirmation():
        yn = infer_yes_no_from_transcript(text)
        if yn is not None:
            try:
                tool_result = await update_application_state.func(
                    None,
                    confidence_score=1.0,
                    raw_user_input=text,
                    deterministic_call=True,
                )
                return _ok(
                    _extract_ask_next(tool_result), "pending_confirmation", stage
                )
            except Exception as e:
                logger.warning(
                    "apply_voice_intake_heuristics_before_line pending confirmation failed: %s",
                    e,
                    exc_info=True,
                )
                return _fail(stage)

    if stage == "greeting":
        _lower_text = text.lower()

        # Soft-decline patterns: "not now", "call me later", "I'm busy", etc.
        # Matched before yes/no inference so they always terminate gracefully.
        if any(p in _lower_text for p in _GREETING_SOFT_DECLINE_PATTERNS):
            return _ok(
                "No problem at all! Feel free to call us back whenever works better. Have a great day, goodbye!",
                "greeting_declined",
                stage,
                terminate_after_tts=True,
            )

        yn = infer_yes_no_from_transcript(text)
        if yn is True:
            _current_call_state().greeting_completed = True
            blob = get_missing_fields() or ""
            ask = _ask_from_missing_line(blob) or "Great, let's get started — what's your property address for financing?"
            return _ok(ask, "greeting_completed", stage)
        if yn is False:
            return _ok(
                "No problem at all! Feel free to call us back whenever it's a better time. Have a great day, goodbye!",
                "greeting_declined",
                stage,
                terminate_after_tts=True,
            )
        return _fail(stage)

    if stage == "property":
        if _use_deterministic_address_parsing() and application_state.address_line_1 is None:
            addr = infer_property_address_from_transcript(text)
            if addr is not None:
                try:
                    tool_result = await update_application_state.func(
                        None,
                        address_line_1=addr,
                        confidence_score=0.92,
                        raw_user_input=text,
                        deterministic_call=True,
                    )
                    return _ok(_extract_ask_next(tool_result), "address_line_1", stage)
                except Exception as e:
                    logger.warning(
                        "apply_voice_intake_heuristics_before_line property address failed: %s",
                        e,
                        exc_info=True,
                    )
                return _fail(stage)
        if application_state.ownership_type is None:
            miss = get_missing_fields() or ""
            ask_m = (_ask_from_missing_line(miss) or "").lower()
            own_q = "ownership type" in ask_m or "sole or joint" in ask_m
            if own_q and _stt_confidence_triggers_reprompt(stt_confidence):
                return _ok(OWNERSHIP_LOW_CONF_REPROMPT, "ownership_type", stage)
            if transcript_requests_intake_default(text):
                logger.info(
                    "intake default: ownership_type=%s (caller declined to specify)",
                    DEFAULT_INTAKE_OWNERSHIP,
                    extra={"event": "state.intake_default", "field": "ownership_type"},
                )
                try:
                    tool_result = await update_application_state.func(
                        None,
                        ownership_type=DEFAULT_INTAKE_OWNERSHIP,
                        confidence_score=0.95,
                        raw_user_input=None,
                        deterministic_call=True,
                    )
                    return _ok(
                        _default_assumption_ask(
                            "ownership_type", _extract_ask_next(tool_result)
                        ),
                        "ownership_type",
                        stage,
                    )
                except Exception as e:
                    logger.warning(
                        "apply_voice_intake_heuristics_before_line ownership default failed: %s",
                        e,
                        exc_info=True,
                    )
                return _fail(stage)
            own = infer_ownership_from_transcript(text)
            if own is not None:
                try:
                    tool_result = await update_application_state.func(
                        None,
                        ownership_type=own,
                        confidence_score=0.95,
                        raw_user_input=text,
                        deterministic_call=True,
                    )
                    return _ok(_extract_ask_next(tool_result), "ownership_type", stage)
                except Exception as e:
                    logger.warning(
                        "apply_voice_intake_heuristics_before_line property ownership failed: %s",
                        e,
                        exc_info=True,
                    )
                return _fail(stage)
            return _fail(stage)

        if application_state.occupancy_type is None:
            miss = get_missing_fields() or ""
            ask_m = (_ask_from_missing_line(miss) or "").lower()
            occ_q = "property type" in ask_m or "primary" in ask_m or "secondary" in ask_m or "investment" in ask_m
            if occ_q and _stt_confidence_triggers_reprompt(stt_confidence):
                return _ok(OCCUPANCY_LOW_CONF_REPROMPT, "occupancy_type", stage)
            if transcript_requests_intake_default(text):
                logger.info(
                    "intake default: occupancy_type=%s (caller declined to specify)",
                    DEFAULT_INTAKE_OCCUPANCY,
                    extra={"event": "state.intake_default", "field": "occupancy_type"},
                )
                try:
                    tool_result = await update_application_state.func(
                        None,
                        occupancy_type=DEFAULT_INTAKE_OCCUPANCY,
                        confidence_score=0.95,
                        raw_user_input=None,
                        deterministic_call=True,
                    )
                    return _ok(
                        _default_assumption_ask(
                            "occupancy_type", _extract_ask_next(tool_result)
                        ),
                        "occupancy_type",
                        stage,
                    )
                except Exception as e:
                    logger.warning(
                        "apply_voice_intake_heuristics_before_line occupancy default failed: %s",
                        e,
                        exc_info=True,
                    )
                return _fail(stage)
            eff, occ = infer_occupancy_with_optional_recovery(
                text, stt_recovery_text
            )
            if occ is not None:
                try:
                    tool_result = await update_application_state.func(
                        None,
                        occupancy_type=occ,
                        confidence_score=0.95,
                        raw_user_input=eff,
                        deterministic_call=True,
                    )
                    return _ok(_extract_ask_next(tool_result), "occupancy_type", stage)
                except Exception as e:
                    logger.warning(
                        "apply_voice_intake_heuristics_before_line property occupancy failed: %s",
                        e,
                        exc_info=True,
                    )
                return _fail(stage)
            return _fail(stage)

        _eff_echo, occ_infer = infer_occupancy_with_optional_recovery(
            text, stt_recovery_text
        )
        if (
            occ_infer is not None
            and occ_infer == application_state.occupancy_type
        ):
            blob = get_missing_fields() or ""
            ask_next = _ask_from_missing_line(blob)
            if ask_next:
                return _ok(ask_next, "occupancy_type", stage)
            if "waiting for navigation" in blob.lower():
                return _ok("Got it. One moment.", "occupancy_type", stage)
            return _fail(stage)

        return _fail(stage)

    if stage == "primary-home":
        if (
            _use_deterministic_address_parsing()
            and application_state.primary_home_address_line_1 is None
        ):
            addr = infer_property_address_from_transcript(text)
            if addr is not None:
                try:
                    tool_result = await update_application_state.func(
                        None,
                        primary_home_address_line_1=addr,
                        confidence_score=0.92,
                        raw_user_input=text,
                        deterministic_call=True,
                    )
                    return _ok(
                        _extract_ask_next(tool_result),
                        "primary_home_address_line_1",
                        stage,
                    )
                except Exception as e:
                    logger.warning(
                        "apply_voice_intake_heuristics_before_line primary home address failed: %s",
                        e,
                        exc_info=True,
                    )
                return _fail(stage)
        return _fail(stage)

    if stage == "consent":
        yn = infer_yes_no_from_transcript(text)
        if yn is None:
            return _fail(stage)
        consent_decline_round = call_state.consent_decline_round or {}
        call_state.consent_decline_round = consent_decline_round

        def _privacy_retry_ask() -> str:
            missing = get_missing_fields() or ""
            line = (_ask_from_missing_line(missing) or "").strip()
            return line or (
                "Do you agree to our Privacy Policy, Terms of Service, and soft credit pull? "
                'Please say \"I agree\" clearly if you do.'
            )

        def _commerce_retry_ask() -> str:
            missing = get_missing_fields() or ""
            line = (_ask_from_missing_line(missing) or "").strip()
            return line or (
                "Do you agree to receive electronic communications about this application? "
                'Please say \"I agree\" if you do.'
            )

        if application_state.agreed_to_privacy_policy is None:
            if yn is True:
                consent_decline_round.pop("privacy", None)
                try:
                    tool_result = await update_application_state.func(
                        None,
                        agreed_to_privacy_policy=True,
                        confidence_score=0.95,
                        raw_user_input=text,
                        deterministic_call=True,
                    )
                    if "Ready to terminate" in (tool_result or ""):
                        return _ok(
                            _COMPLETED_CLOSING_LINE,
                            "agreed_to_privacy_policy",
                            stage,
                            terminate_after_tts=True,
                        )
                    return _ok(
                        _extract_ask_next(tool_result),
                        "agreed_to_privacy_policy",
                        stage,
                    )
                except Exception as e:
                    logger.warning(
                        "apply_voice_intake_heuristics_before_line consent privacy yes failed: %s",
                        e,
                        exc_info=True,
                    )
                return _fail(stage)
            # Declined privacy/terms — do not call tool with False (would loop validation).
            consent_decline_round["privacy"] = consent_decline_round.get("privacy", 0) + 1
            rnd = consent_decline_round["privacy"]
            if rnd == 1:
                retry = _privacy_retry_ask()
                msg = (
                    "For regulatory reasons, agreeing to our Privacy Policy, Terms of Service, "
                    "and soft credit check is required to continue—we can't move forward without "
                    "that consent. "
                    + retry
                )
                return _ok(msg, "consent_decline_explained_privacy", stage)
            consent_decline_round.pop("privacy", None)
            return _ok(
                "Okay—without that agreement we can't continue your application today. "
                "You're welcome to call back when you're ready. Goodbye for now.",
                "consent_declined_terminal_privacy",
                stage,
                terminate_after_tts=True,
            )

        if application_state.agreed_to_commerce_communication is None:
            if yn is True:
                consent_decline_round.pop("commerce", None)
                try:
                    tool_result = await update_application_state.func(
                        None,
                        agreed_to_commerce_communication=True,
                        confidence_score=0.95,
                        raw_user_input=text,
                        deterministic_call=True,
                    )
                    if "Ready to terminate" in (tool_result or ""):
                        return _ok(
                            _COMPLETED_CLOSING_LINE,
                            "agreed_to_commerce_communication",
                            stage,
                            terminate_after_tts=True,
                        )
                    return _ok(
                        _extract_ask_next(tool_result),
                        "agreed_to_commerce_communication",
                        stage,
                    )
                except Exception as e:
                    logger.warning(
                        "apply_voice_intake_heuristics_before_line consent commerce yes failed: %s",
                        e,
                        exc_info=True,
                    )
                return _fail(stage)
            consent_decline_round["commerce"] = consent_decline_round.get("commerce", 0) + 1
            rnd = consent_decline_round["commerce"]
            if rnd == 1:
                retry = _commerce_retry_ask()
                msg = (
                    "We need your agreement on electronic communications to proceed—that "
                    "step is mandatory for this application. "
                    + retry
                )
                return _ok(msg, "consent_decline_explained_commerce", stage)
            consent_decline_round.pop("commerce", None)
            return _ok(
                "Understood—without that consent we can't finish your application. "
                "Thanks for your time; goodbye for now.",
                "consent_declined_terminal_commerce",
                stage,
                terminate_after_tts=True,
            )

        return _fail(stage)

    if stage == "name":
        need_first = application_state.first_name is None
        need_last = application_state.last_name is None
        if not need_first and not need_last:
            return _fail(stage)
        slot_focus = _name_slot_focus_from_missing(get_missing_fields() or "")
        parsed = infer_name_fields_from_transcript(
            text,
            need_first=need_first,
            need_last=need_last,
            slot_focus=slot_focus,
        )
        if not parsed:
            return _fail(stage)
        handled_field = "first_name" if parsed.get("first_name") else "last_name"
        confidence = 0.80 if _spoken_letters_to_word(text) else 0.95
        try:
            tool_result = await update_application_state.func(
                None,
                first_name=parsed.get("first_name"),
                last_name=parsed.get("last_name"),
                confidence_score=confidence,
                raw_user_input=text,
                deterministic_call=True,
            )
            return _ok(_extract_ask_next(tool_result), handled_field, stage)
        except Exception as e:
            logger.warning(
                "apply_voice_intake_heuristics_before_line name failed: %s",
                e,
                exc_info=True,
            )
        return _fail(stage)

    if stage == "income":
        if application_state.annual_pre_tax_income is None:
            if transcript_requests_intake_default(text):
                logger.info(
                    "intake default: annual_pre_tax_income=%s (caller declined to specify)",
                    DEFAULT_INTAKE_ANNUAL_PRE_TAX,
                    extra={"event": "state.intake_default", "field": "annual_pre_tax_income"},
                )
                try:
                    tool_result = await update_application_state.func(
                        None,
                        annual_pre_tax_income=DEFAULT_INTAKE_ANNUAL_PRE_TAX,
                        confidence_score=0.95,
                        raw_user_input=None,
                        deterministic_call=True,
                    )
                    return _ok(
                        _default_assumption_ask(
                            "annual_pre_tax_income", _extract_ask_next(tool_result)
                        ),
                        "annual_pre_tax_income",
                        stage,
                    )
                except Exception as e:
                    logger.warning(
                        "apply_voice_intake_heuristics_before_line annual income default failed: %s",
                        e,
                        exc_info=True,
                    )
                return _fail(stage)
            inc, fragile_income = infer_annual_income_meta(text)
            if inc is not None:
                try:
                    income_conf = (
                        _INCOME_AMBIGUOUS_STT_CONFIDENCE if fragile_income else 0.95
                    )
                    tool_result = await update_application_state.func(
                        None,
                        annual_pre_tax_income=inc,
                        confidence_score=income_conf,
                        raw_user_input=text,
                        deterministic_call=True,
                    )
                    return _ok(_extract_ask_next(tool_result), "annual_pre_tax_income", stage)
                except Exception as e:
                    logger.warning(
                        "apply_voice_intake_heuristics_before_line income amount failed: %s",
                        e,
                        exc_info=True,
                    )
                return _fail(stage)
            return _fail(stage)

        if application_state.other_income is None:
            # ── Priority check: if the transcript answers income SOURCE (not other income),
            # the caller has moved past the other-income question. Auto-default other_income=0
            # and save the income source together to prevent the repeated-question loop.
            if application_state.annual_pre_tax_income is not None:
                src_early = infer_income_source_from_transcript(text)
                if src_early is not None:
                    logger.info(
                        "income heuristic: transcript answers income source while other_income=None "
                        "— auto-defaulting other_income=0 and saving both",
                        extra={"event": "state.income_source_auto_skip_other"},
                    )
                    try:
                        tool_result = await update_application_state.func(
                            None,
                            other_income=0,
                            primary_source_of_income=src_early,
                            confidence_score=0.95,
                            raw_user_input=text,
                            deterministic_call=True,
                        )
                        return _ok(_extract_ask_next(tool_result), "primary_source_of_income", stage)
                    except Exception as e:
                        logger.warning(
                            "apply_voice_intake_heuristics_before_line income source auto-skip failed: %s",
                            e,
                            exc_info=True,
                        )
                    return _fail(stage)

            if transcript_requests_intake_default(text):
                logger.info(
                    "intake default: other_income=%s (caller declined to specify)",
                    DEFAULT_INTAKE_OTHER_INCOME,
                    extra={"event": "state.intake_default", "field": "other_income"},
                )
                try:
                    tool_result = await update_application_state.func(
                        None,
                        other_income=DEFAULT_INTAKE_OTHER_INCOME,
                        confidence_score=0.95,
                        raw_user_input=None,
                        deterministic_call=True,
                    )
                    return _ok(
                        _default_assumption_ask(
                            "other_income", _extract_ask_next(tool_result)
                        ),
                        "other_income",
                        stage,
                    )
                except Exception as e:
                    logger.warning(
                        "apply_voice_intake_heuristics_before_line other income default failed: %s",
                        e,
                        exc_info=True,
                    )
                return _fail(stage)
            oi = infer_other_income_from_transcript(text)
            if oi is not None:
                try:
                    tool_result = await update_application_state.func(
                        None,
                        other_income=oi,
                        confidence_score=0.95,
                        raw_user_input=text,
                        deterministic_call=True,
                    )
                    return _ok(_extract_ask_next(tool_result), "other_income", stage)
                except Exception as e:
                    logger.warning(
                        "apply_voice_intake_heuristics_before_line other income failed: %s",
                        e,
                        exc_info=True,
                    )
                return _fail(stage)
            # "Yes" / affirmative without dollar amount → ask how much
            yn = infer_yes_no_from_transcript(text)
            if yn is True:
                return _ok(
                    "How much additional income per year, in dollars?",
                    "other_income",
                    stage,
                )
            return _fail(stage)

        if application_state.primary_source_of_income is None:
            if transcript_requests_intake_default(text):
                logger.info(
                    "intake default: primary_source_of_income=%s (caller declined to specify)",
                    DEFAULT_INTAKE_PRIMARY_SOURCE,
                    extra={"event": "state.intake_default", "field": "primary_source_of_income"},
                )
                try:
                    tool_result = await update_application_state.func(
                        None,
                        primary_source_of_income=DEFAULT_INTAKE_PRIMARY_SOURCE,
                        confidence_score=0.95,
                        raw_user_input=None,
                        deterministic_call=True,
                    )
                    return _ok(
                        _default_assumption_ask(
                            "primary_source_of_income",
                            _extract_ask_next(tool_result),
                        ),
                        "primary_source_of_income",
                        stage,
                    )
                except Exception as e:
                    logger.warning(
                        "apply_voice_intake_heuristics_before_line primary source default failed: %s",
                        e,
                        exc_info=True,
                    )
                return _fail(stage)
            src = infer_income_source_from_transcript(text)
            if src is not None:
                try:
                    tool_result = await update_application_state.func(
                        None,
                        primary_source_of_income=src,
                        confidence_score=0.95,
                        raw_user_input=text,
                        deterministic_call=True,
                    )
                    return _ok(_extract_ask_next(tool_result), "primary_source_of_income", stage)
                except Exception as e:
                    logger.warning(
                        "apply_voice_intake_heuristics_before_line income source failed: %s",
                        e,
                        exc_info=True,
                    )
                return _fail(stage)
            return _fail(stage)
        return _fail(stage)

    if stage == "birthday":
        if application_state.date_of_birth is not None:
            return _fail(stage)
        dob = infer_dob_from_transcript(text)
        if dob is None:
            if infer_partial_dob_month_day(text):
                return _ok(PARTIAL_DOB_REPROMPT, "date_of_birth", stage)
            return _fail(stage)
        try:
            tool_result = await update_application_state.func(
                None,
                date_of_birth=dob,
                confidence_score=0.95,
                raw_user_input=text,
                deterministic_call=True,
            )
            return _ok(_extract_ask_next(tool_result), "date_of_birth", stage)
        except Exception as e:
            logger.warning(
                "apply_voice_intake_heuristics_before_line dob failed: %s",
                e,
                exc_info=True,
            )
        return _fail(stage)

    if stage == "contact":
        if application_state.mobile_phone_number is None:
            clean_text = normalize_answer_scaffold(text) or text
            digit_run = re.sub(r"\D", "", clean_text)
            spoken_digits = _spoken_digit_sequence_to_digits(clean_text)
            if spoken_digits:
                sd = spoken_digits
                if len(sd) == 11 and sd.startswith("1"):
                    sd = sd[1:]
                if 5 <= len(sd) <= 9:
                    return _ok(
                        PARTIAL_MOBILE_PHONE_REPROMPT,
                        "mobile_phone_number",
                        stage,
                    )
            if 5 <= len(digit_run) <= 9:
                return _ok(
                    PARTIAL_MOBILE_PHONE_REPROMPT,
                    "mobile_phone_number",
                    stage,
                )
            ph = infer_phone_from_transcript(text)
            if ph is not None:
                try:
                    tool_result = await update_application_state.func(
                        None,
                        mobile_phone_number=ph,
                        confidence_score=0.95,
                        raw_user_input=text,
                        deterministic_call=True,
                    )
                    return _ok(_extract_ask_next(tool_result), "mobile_phone_number", stage)
                except Exception as e:
                    logger.warning(
                        "apply_voice_intake_heuristics_before_line phone failed: %s",
                        e,
                        exc_info=True,
                    )
                return _fail(stage)
        if application_state.email_address is None:
            em = infer_email_from_transcript(text)
            if em is not None:
                try:
                    tool_result = await update_application_state.func(
                        None,
                        email_address=em,
                        confidence_score=0.95,
                        raw_user_input=text,
                        deterministic_call=True,
                    )
                    return _ok(_extract_ask_next(tool_result), "email_address", stage)
                except Exception as e:
                    logger.warning(
                        "apply_voice_intake_heuristics_before_line email failed: %s",
                        e,
                        exc_info=True,
                    )
                return _fail(stage)
            if transcript_requests_intake_default(text):
                return _fail(stage)
        return _fail(stage)

    # Stage check for for-sale: use conversation stage if browser is still catching up.
    active_stage = stage
    if active_stage == "property" and get_current_conversation_stage() == "FOR_SALE":
        active_stage = "for-sale"

    if active_stage != "for-sale":
        return _NOT_HANDLED
    if application_state.is_currently_for_sale is not None:
        fs_echo = infer_for_sale_from_transcript(text)
        if (
            fs_echo is not None
            and fs_echo == application_state.is_currently_for_sale
        ):
            blob = get_missing_fields() or ""
            ask_next = _ask_from_missing_line(blob)
            if ask_next:
                return _ok(ask_next, "is_currently_for_sale", stage)
            if "waiting for navigation" in blob.lower():
                return _ok("Got it. One moment.", "is_currently_for_sale", stage)
            return _fail(stage)
        return _fail(stage)
    miss_fs = get_missing_fields() or ""
    ask_fs = (_ask_from_missing_line(miss_fs) or "").lower()
    fs_q = "for sale" in ask_fs or "listed for sale" in ask_fs
    if fs_q and _stt_confidence_triggers_reprompt(stt_confidence):
        return _ok(FOR_SALE_LOW_CONF_REPROMPT, "is_currently_for_sale", stage)
    if transcript_requests_intake_default(text):
        fs = DEFAULT_INTAKE_FOR_SALE
        used_for_sale_default = True
        logger.info(
            "intake default: is_currently_for_sale=%s (caller declined to specify)",
            fs,
            extra={"event": "state.intake_default", "field": "is_currently_for_sale"},
        )
    else:
        fs = infer_for_sale_from_transcript(text)
        used_for_sale_default = False
    if fs is None:
        return _fail(stage)
    try:
        tool_result = await update_application_state.func(
            None,
            is_currently_for_sale=fs,
            confidence_score=0.95,
            raw_user_input=None if used_for_sale_default else text,
            deterministic_call=True,
        )
        ask_next = _extract_ask_next(tool_result)
        if used_for_sale_default:
            ask_next = _default_assumption_ask("is_currently_for_sale", ask_next)
        return _ok(ask_next, "is_currently_for_sale", stage)
    except Exception as e:
        logger.warning("apply_voice_intake_heuristics_before_line failed: %s", e, exc_info=True)
    return _fail(stage)
