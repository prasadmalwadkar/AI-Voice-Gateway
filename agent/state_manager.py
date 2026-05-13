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
# BrowserManager removed from hot path — used only by form_filler_consumer.
# Import queue publisher for completed intake payloads.
from agent.queue_publisher import publish_completed_intake

logger = logging.getLogger("voip.state")

# Maps conversation stages (logic-based) to the browser-stage strings used
# throughout the file for stage gating and heuristics. Browser is no longer
# running in the voice path — this replaces browser_manager.get_current_stage().
_COMPLETED_CLOSING_LINE = (
    "Great, that's everything we need. Your application has been submitted and "
    "someone from West Capital Lending will be in touch soon. Thanks for your time, goodbye!"
)

# Seconds after a preline success during which subsequent preline calls return _NOT_HANDLED.
# Prevents trailing STT finals from the same utterance from confirming the *next* field.
_PRELINE_COOLDOWN_SECS = 1.5

# GREETING soft-decline keyword patterns — matched before yes/no inference.
_GREETING_SOFT_DECLINE_PATTERNS = (
    "not now", "not a good time", "bad time", "not the right time",
    "call back", "call later", "call me later", "call me back",
    "not free", "busy right now", "i'm busy", "catch me later",
    "another time", "can i call", "can you call", "reach me later",
    "not ready", "not interested",
)

_CONV_STAGE_TO_BROWSER: dict[str, str] = {
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

# Canonical scripted voice lines (field_to_question + deterministic pre-Line).
VOICE_SCRIPT_OWNERSHIP_ASK = "Are you the sole owner, or is it jointly owned?"
VOICE_SCRIPT_OCCUPANCY_ASK = (
    "Is this your primary home, a second home, or an investment property?"
)
VOICE_SCRIPT_FOR_SALE_ASK = "Is the property currently listed for sale?"
OCCUPANCY_LOW_CONF_REPROMPT = (
    "Sorry, I didn't catch that. Say primary home, second home, or investment property."
)
OWNERSHIP_LOW_CONF_REPROMPT = (
    "Sorry, I didn't catch that. Are you the sole owner, or is it jointly owned?"
)
FOR_SALE_LOW_CONF_REPROMPT = (
    "Sorry, I didn't catch that. Is the property listed for sale — yes or no?"
)

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

    if not ask or ask in {"What is the property address?", "Can you tell me the property address?"}:
        return f"{greeting}can you tell me your property address?" if first_name else None
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


# Fields shown on one browser step — transcript ``ui`` lines use only these per ``current_stage``.
_HELOC_UI_PAGE_FIELDS_BY_STAGE: dict[str, frozenset[str]] = {
    "greeting": frozenset(),
    "unknown": frozenset(
        {
            "address_line_1",
            "city",
            "state",
            "zipcode",
            "ownership_type",
            "occupancy_type",
        }
    ),
    "property": frozenset(
        {
            "address_line_1",
            "city",
            "state",
            "zipcode",
            "ownership_type",
            "occupancy_type",
        }
    ),
    "primary-home": frozenset(
        {
            "primary_home_address_line_1",
            "primary_home_city",
            "primary_home_state",
            "primary_home_zipcode",
        }
    ),
    "for-sale": frozenset({"is_currently_for_sale"}),
    "income": frozenset(
        {
            "annual_pre_tax_income",
            "other_income",
            "primary_source_of_income",
        }
    ),
    "name": frozenset({"first_name", "middle_name", "last_name", "suffix"}),
    "birthday": frozenset({"date_of_birth"}),
    "contact": frozenset({"mobile_phone_number", "email_address"}),
    "consent": frozenset(
        {
            "agreed_to_privacy_policy",
            "agreed_to_commerce_communication",
            "agreed_to_credit_pull",
        }
    ),
}

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


def _heloc_ui_snap_string_values(keys: frozenset[str]) -> dict[str, str]:
    """Build string map for transcript; omit None / empty strings; booleans as True/False."""
    st = application_state
    snap: dict[str, str] = {}
    for key in keys:
        if key in _SENSITIVE_FIELDS:
            continue
        v = getattr(st, key, None)
        if v is None:
            continue
        if isinstance(v, bool):
            snap[key] = "True" if v else "False"
        elif isinstance(v, str):
            t = v.strip()
            if t:
                snap[key] = t
        elif isinstance(v, int):
            snap[key] = str(v)
        else:
            t = str(v).strip()
            if t:
                snap[key] = t
    return snap


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


def _append_heloc_ui_transcript_snapshot() -> None:
    """Append one NDJSON line: fields visible on the current browser stage only."""
    try:
        from app.conversation_transcript import append_transcript_ui_fields
    except ImportError:
        return
    stage = _CONV_STAGE_TO_BROWSER.get(get_current_conversation_stage(), "unknown")
    keys = _HELOC_UI_PAGE_FIELDS_BY_STAGE.get(stage) or _HELOC_UI_PAGE_FIELDS_BY_STAGE[
        "unknown"
    ]
    snap = _heloc_ui_snap_string_values(keys)
    if not snap:
        return
    try:
        append_transcript_ui_fields(
            None, snap, scope="page", browser_stage=stage
        )
    except Exception:
        logger.debug("append_transcript_ui_fields failed", exc_info=True)


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

    # Short scripted asks per field.
    field_to_question = {
        "address_line_1": "Can you tell me the property address?",
        "ownership_type": VOICE_SCRIPT_OWNERSHIP_ASK,
        "occupancy_type": VOICE_SCRIPT_OCCUPANCY_ASK,
        "primary_home_address_line_1": "What is the street address of your primary residence?",
        "is_currently_for_sale": VOICE_SCRIPT_FOR_SALE_ASK,
        "annual_pre_tax_income": "What's your annual pre-tax income?",
        "other_income": "Do you have any other income you'd like to include, like rental or investment income?",
        "primary_source_of_income": "And what's your primary source of income — employment, self-employed, retirement, or something else?",
        "first_name": "What's your first name?",
        "middle_name": "What's your middle name?",
        "last_name": "What's your last name?",
        "suffix": "Any suffix like Jr or Sr?",
        "date_of_birth": "And what's your date of birth?",
        "mobile_phone_number": "What's the best mobile number to reach you?",
        "email_address": "And what's your email address?",
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
    stage = _CONV_STAGE_TO_BROWSER.get(get_current_conversation_stage(), "unknown")
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
        f"browser_stage={stage}",
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
CONFIDENCE_THRESHOLD = 0.85
# Address is recovered via autocomplete + DOM sync; allow slightly lower score
# so callers don't get stuck in repeat loops on noisy 8 kHz STT.
ADDRESS_CONFIDENCE_THRESHOLD = 0.75
ADDRESS_CONFIDENCE_FIELDS = {
    "address_line_1",
    "city",
    "state",
    "zipcode",
    "primary_home_address_line_1",
    "primary_home_city",
    "primary_home_state",
    "primary_home_zipcode",
}

# Fields that require explicit high-confidence consent (legal compliance).
# Vague responses like "sure" or "I guess" are not sufficient.
HIGH_CONFIDENCE_FIELDS = {
    "agreed_to_privacy_policy",
    "agreed_to_commerce_communication",
}

# =============================================================================
# Uncertainty Markers — deterministic safety net for speech ambiguity.
#
# Even if the LLM self-reports high confidence, these words in the user's
# raw speech indicate the caller was unsure. When detected, the tool
# blocks the save and returns PENDING status, forcing re-confirmation.
#
# Example: "My address is 325 West Adams Blvd, or maybe it was East Adams"
#          → "maybe" and "or" detected → PENDING, not saved.
# =============================================================================
UNCERTAINTY_MARKERS = [
    "maybe",
    "or maybe",
    "i think",
    "not sure",
    "forgot",
    "i forget",
    "i guess",
    "actually",
    "not certain",
    "probably",
    "might be",
    "could be",
    "i believe",
    "wait",
    "hold on",
    "let me check",
]

# Path2 pre-line: only when transcript_requests_intake_default() matches — not when
# infer merely fails. raw_user_input omitted on default saves so uncertainty scan
# does not block.
DEFAULT_INTAKE_OWNERSHIP: OwnershipType = "Sole owner"
DEFAULT_INTAKE_OCCUPANCY: OccupancyType = "Primary property"
DEFAULT_INTAKE_FOR_SALE = False
DEFAULT_INTAKE_OTHER_INCOME = 0
DEFAULT_INTAKE_ANNUAL_PRE_TAX = 100_000
DEFAULT_INTAKE_PRIMARY_SOURCE: IncomeSource = "Employed full-time"

_DECLINE_TO_SPECIFY_PATTERNS = (
    r"\bi\s+(?:don'?t|do\s+not)\s+know\b",
    r"\bi\s+(?:don'?t|do\s+not)\s+remember\b",
    r"\bi\s+can'?t\s+(?:remember|think|recall)\b",
    r"\bi\s+don'?t\s+have\s+(?:any\s+)?idea\b",
    r"\bno\s+idea\b",
    r"\bpass\s+on\s+that\b",
    r"\bi'?d\s+rather\s+not\s+say\b",
    # "I'm not sure" / "I'm unsure" / "not certain" — same intent as don't know
    r"\bi'?m\s+not\s+sure\b",
    r"\bnot\s+sure\b",
    r"\bi'?m\s+unsure\b",
    r"\bnot\s+certain\b",
    r"\bi'?m\s+not\s+certain\b",
    r"\bdon'?t\s+have\s+(?:that\s+)?information\b",
    r"\bi\s+don'?t\s+know\s+(?:the\s+)?answer\b",
)


def transcript_requests_intake_default(text: str) -> bool:
    """True only for explicit decline / don't-know phrasing (then apply default)."""
    if not text or not str(text).strip():
        return False
    low = text.lower()
    # Hedging ("not sure about that address") — not the same as declining the question.
    if "not sure about" in low:
        return False
    return any(re.search(p, low) for p in _DECLINE_TO_SPECIFY_PATTERNS)


_SPEECH_FILLER_RE = re.compile(
    r"^[\s.,!?]*(yeah|yep|yea|yah|okay|ok|sure|right|"
    r"uh+|u+m+|ah+|er+|well|hmm+|oh|hey|hi|like)[\s.,!?]+",
    re.IGNORECASE,
)


def normalize_answer_scaffold(text: str) -> str:
    """Remove natural answer lead-ins + speech fillers before deterministic field inference."""
    if not text or not str(text).strip():
        return ""
    s = re.sub(r"\s+", " ", str(text).strip())
    # Strip leading speech fillers repeatedly (handles "Yeah. Um, ...")
    # Don't strip "yes"/"no" — they are valid answers for y/n questions.
    for _ in range(4):
        m = _SPEECH_FILLER_RE.match(s)
        if not m:
            break
        remainder = s[m.end():].strip()
        if not remainder:
            break
        s = remainder
    s = re.sub(
        r"^\s*(?:"
        r"it'?s|it\s+is|this\s+is|"
        r"i\s+am|i'?m|"
        r"my\s+answer\s+is|the\s+answer\s+is|"
        r"that\s+would\s+be|it\s+would\s+be|"
        r"i\s+would\s+say|probably|i\s+think"
        r")\s+",
        "",
        s,
        flags=re.IGNORECASE,
    )
    return s.strip()


def transcript_is_status_check(text: str) -> bool:
    """Presence / status pings should never be parsed as applicant data."""
    if not text or not str(text).strip():
        return False
    if transcript_seeks_question_repeat(text):
        return True
    t = str(text).strip().lower()
    t = re.sub(r"[^\w\s?']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    patterns = (
        r"\b(you there|are you there|still there|hello)\b",
        r"\b(you good|are you good|are we good|still with me)\b",
        r"\b(can you hear me|do you hear me|did you hear me)\b",
        r"\b(what'?s next|next question|go ahead|continue)\b",
        r"\b(what'?s the question|what is the question|which question)\b",
        r"\b(repeat|say that again|say again|come again)\b",
        r"\b(i didn'?t hear|didn'?t catch|what did you say)\b",
        r"\b(i'?m lost|i'?m confused|what do you need|what do i do)\b",
        r"\b(pardon|excuse me|sorry what)\b",
        r"\b(thank you|thanks)\b",
    )
    return any(re.search(p, t) for p in patterns)


def transcript_seeks_question_repeat(text: str) -> bool:
    """True when the caller wants the current question repeated (vs a pure presence ping)."""
    if not text or not str(text).strip():
        return False
    t = normalize_answer_scaffold(text).lower()
    t = re.sub(r"[^\w\s']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return False
    if re.fullmatch(r"(are you there|still there|you there)(\s+there)?(\?)?", t):
        return False
    if re.fullmatch(r"(hello|hey|hi)(\s+there)?(\?)?", t):
        return False
    if re.fullmatch(r"(thank you|thanks)(\s+you)?(\?)?", t):
        return False
    patterns = (
        r"\bwhat'?s\s+the\s+question\b",
        r"\bwhat\s+is\s+the\s+question\b",
        r"\bwhich\s+question\b",
        r"\bwhat\s+question\b",
        r"\bwhat\s+(?:exactly\s+)?(?:do\s+you\s+need|are\s+you\s+asking)\b",
        r"\bwhat\s+info(?:rmation)?\s+(?:do\s+you\s+need|should\s+i\s+give)\b",
        r"\bwhat\s+details\s+do\s+you\s+need\b",
        r"\b(remind\s+me|tell\s+me\s+again)\b",
        r"^\s*repeat\s*\??\s*$",
        r"\brepeat\s+(please|that)\b",
        r"\b(repeat|say)\s+(?:that|the\s+question|it)\s+again\b",
        r"\bsay\s+again\b",
        r"\bcome\s+again\b",
        r"\bpardon\b",
        r"\bexcuse\s+me\b",
        r"\bsorry\s*,?\s*what\b",
        r"\bwhat\s+did\s+you\s+say\b",
        r"\bi\s+didn'?t\s+(?:catch|hear|get|understand)\s+(?:that|it)\b",
        r"\bi\s+m\s+(?:lost|confused)\b",
        r"\bwhat\s+do\s+i\s+(?:need\s+to\s+)?(?:say|answer)\b",
        r"\bwhat\s+should\s+i\s+say\b",
        r"\bwhat'?s\s+next\b",
        r"\bnext\s+question\b",
        r"\bgo\s+ahead\b",
        r"\bcontinue\b",
    )
    return any(re.search(p, t) for p in patterns)


def _is_obvious_non_answer(text: str) -> bool:
    if not text or not str(text).strip():
        return True
    t = normalize_answer_scaffold(text).lower()
    t = re.sub(r"[^\w\s']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return True
    patterns = (
        r"^(sorry|sorry about that|my bad)$",
        r"^(that'?s all|that is all|all set|never mind|nevermind)$",
        r"^(good day|bye|goodbye)$",
        r"^(thank you|thanks)$",
        r"^(love in the list|lost in the list|waste items delivered)$",
    )
    return any(re.fullmatch(p, t) for p in patterns)


def _ask_from_missing_line(missing_line: str) -> str:
    m = re.search(r'Ask next:\s*"([^"]+)"', missing_line or "")
    if not m:
        return ""
    return (m.group(1) or "").strip()


def looks_like_answer_for_current_field(text: str, missing_line: str) -> bool:
    """True when a transcript plausibly answers the current `Ask next` field."""
    if not text or not str(text).strip():
        return False
    if transcript_is_status_check(text) or _is_obvious_non_answer(text):
        return False

    ask = _ask_from_missing_line(missing_line).lower()
    if not ask:
        return True

    normalized = normalize_answer_scaffold(text)
    low = normalized.lower()
    bare = re.sub(r"[^\w\s']", " ", low)
    bare = re.sub(r"\s+", " ", bare).strip()

    if (
        "on file" in ask
        or "is that correct" in ask
        or "is that still correct" in ask
    ):
        return infer_yes_no_from_transcript(normalized) is not None

    if "property address" in ask:
        if infer_property_address_from_transcript(normalized):
            return True
        if any(ch.isdigit() for ch in bare):
            return True
        address_patterns = (
            r"\blos\s+angeles\b",
            r"\bcalifornia\b",
            r"\bstreet\b",
            r"\bst\b",
            r"\bavenue\b",
            r"\bave\b",
            r"\bboulevard\b",
            r"\bblvd\b",
            r"\broad\b",
            r"\brd\b",
            r"\bdrive\b",
            r"\bdr\b",
            r"\blane\b",
            r"\bln\b",
            r"\bcourt\b",
            r"\bct\b",
            r"\bparkway\b",
            r"\bpkwy\b",
            r"\bplace\b",
            r"\bpl\b",
        )
        return any(re.search(p, bare) for p in address_patterns)

    if "ownership type" in ask or "sole or joint" in ask:
        if infer_ownership_from_transcript(normalized):
            return True
        return bool(
            re.search(
                r"\b(sole|soul|sol|joint|owner|one|two|won|too|loner|lonas|alone)\b",
                bare,
            )
        )

    if (
        "property type" in ask
        or "primary, secondary, or investment" in ask
        or ("primary home" in ask and "second home" in ask)
    ):
        return infer_occupancy_from_transcript(normalized) is not None

    if "currently for sale" in ask or "for sale" in ask:
        return infer_for_sale_from_transcript(normalized) is not None

    if "annual pre-tax income" in ask or "annual income" in ask:
        return infer_annual_income_from_transcript(normalized) is not None

    if "other income" in ask:
        return infer_other_income_from_transcript(normalized) is not None

    if "primary income source" in ask or "income source" in ask:
        return infer_income_source_from_transcript(normalized) is not None

    if "first name" in ask:
        return bool(
            infer_name_fields_from_transcript(
                normalized, need_first=True, need_last=False, slot_focus="first"
            )
        )

    if "last name" in ask:
        return bool(
            infer_name_fields_from_transcript(
                normalized, need_first=False, need_last=True, slot_focus="last"
            )
        )

    if "date of birth" in ask:
        return infer_dob_from_transcript(normalized) is not None

    if "mobile phone" in ask or "phone" in ask:
        return infer_phone_from_transcript(normalized) is not None

    if "email" in ask:
        return infer_email_from_transcript(normalized) is not None

    if "privacy policy" in ask or "electronic communications" in ask or "agree" in ask:
        return infer_yes_no_from_transcript(normalized) is not None

    return True


def infer_ownership_from_transcript(text: str) -> Optional[OwnershipType]:
    """Map common STT garbles of 'sole/joint owner' to enum when LLM omits ownership_type."""
    if not text or not str(text).strip():
        return None
    text = normalize_answer_scaffold(text)
    if not text:
        return None
    # ink-whisper / phone: "Sole" → "Stole" as a short final; check each clause
    for part in re.split(r"[.!?\n]+", text):
        seg = part.strip().lower()
        if not seg:
            continue
        seg_n = seg.rstrip(".,!?")
        # Numeric shortcut mode for noisy telephony turns.
        if re.fullmatch(r"(1|one|won|when)", seg_n):
            return "Sole owner"
        if re.fullmatch(r"(2|two|too|to)", seg_n):
            return "Joint owner"
        if re.search(r"\bjoint\b", seg):
            return "Joint owner"
        # Joint STT garbles (prefer owner-phrased to avoid spurious "join" matches).
        if re.search(
            r"\bjoin(?:ed|t)?\s+owners?\b"
            r"|\bjoist\s+owners?\b"
            r"|\bjoynt\s+owners?\b"
            r"|\bjointed\s+owners?\b"
            r"|\bjointly\s+owned\b"
            r"|\bjoint\s+owners?\b"
            r"|\bco[\s-]?owners?\b"
            r"|\bsecondary\s+owners?\b"
            r"|\bsecond\s+owners?\b",
            seg,
        ):
            return "Joint owner"
        if re.search(r"\bfull\s+owners?\b", seg):
            return "Sole owner"
        if re.fullmatch(r"stole\.?", seg):
            return "Sole owner"
        if re.fullmatch(r"soul\.?", seg):
            return "Sole owner"
        if re.fullmatch(r"sool.*", seg):
            return "Sole owner"
        if re.fullmatch(r"so,?\s*loner\.?", seg):
            return "Sole owner"
        if re.fullmatch(r"so\s+alone\.?", seg):
            return "Sole owner"
        if re.fullmatch(r"soul\s+or\s+not\.?", seg):
            return "Sole owner"
        # 8 kHz + ink-whisper: "sole" often transcribes as "phone" / "foam"
        if re.fullmatch(r"(phone|foam)\.?", seg):
            return "Sole owner"
        if re.fullmatch(r"toll\.?", seg):
            return "Sole owner"
        if re.fullmatch(r"(soon|sown|sewn|zone|sol)\.?", seg):
            return "Sole owner"
        if re.fullmatch(r"so\s+on\.?", seg):
            return "Sole owner"
        # ink-whisper: "sole owner" → "Soul bonus." / similar
        if re.search(r"\bsoul\s+bonus\b", seg):
            return "Sole owner"
    low = text.lower()
    if re.fullmatch(r"\s*(1|one|won|when)\s*[.,!?]?\s*", low):
        return "Sole owner"
    if re.fullmatch(r"\s*(2|two|too|to)\s*[.,!?]?\s*", low):
        return "Joint owner"
    if re.search(r"\bjoint\b", low):
        return "Joint owner"
    if re.search(
        r"\b(sole|soul|soil|solo)\s*owners?\b"
        r"|\bfull\s+owners?\b"
        r"|so\s+alone"
        r"|so,?\s*loner"
        r"|stole\s*,?\s*stole"
        r"|\bstole\s+owners?\b"
        r"|\bso\s+owners?\b"
        r"|\btool\s*owners?\b"
        r"|sotool"
        r"|\btoll\s+owners?\b"
        r"|\bfoam\s+owners?\b"
        r"|\bsold\s+owners?\b"
        r"|\bsool\b"
        r"|\bsoul\s+or\s+not\b"
        r"|\bstore\s+owners?\b"
        r"|\bstored\s+owner\b"
        r"|\bstore\s+owner\b",
        low,
    ):
        return "Sole owner"
    if re.search(
        r"\bjoin(?:ed|t)?\s+owners?\b"
        r"|\bjoist\s+owners?\b"
        r"|\bjoynt\s+owners?\b"
        r"|\bjointed\s+owners?\b"
        r"|\bjointly\s+owned\b"
        r"|\bjoint\s+owners?\b"
        r"|\bco[\s-]?owners?\b"        # "co-owner", "co owner"
        r"|\bsecondary\s+owners?\b"    # "secondary owner" = co-owner in conversational use
        r"|\bsecond\s+owners?\b",      # "second owner"
        low,
    ):
        return "Joint owner"
    return None


def infer_occupancy_from_transcript(text: str) -> Optional[OccupancyType]:
    """Map common STT phrasing to occupancy when LLM omits occupancy_type."""
    if not text or not str(text).strip():
        return None
    text = normalize_answer_scaffold(text)
    if not text:
        return None
    for part in re.split(r"[.!?\n]+", text):
        seg = part.strip().lower()
        if not seg:
            continue
        seg_n = seg.rstrip(".,!?")
        if re.fullmatch(r"(1|one|won|when)", seg_n):
            return "Primary property"
        if re.fullmatch(r"(2|two|too|to)", seg_n):
            return "Secondary property"
        if re.fullmatch(r"(3|three|tree|free)", seg_n):
            return "Investment property"
    low = text.lower()
    if re.fullmatch(r"\s*(1|one|won|when)\s*[.,!?]?\s*", low):
        return "Primary property"
    if re.fullmatch(r"\s*(2|two|too|to)\s*[.,!?]?\s*", low):
        return "Secondary property"
    if re.fullmatch(r"\s*(3|three|tree|free)\s*[.,!?]?\s*", low):
        return "Investment property"
    if re.search(
        r"\b(investment|investor|rental\s+property|rental\s+home|income\s+property|"
        r"non[- ]?owner\s+occupied|landlord|flip\s+(?:house|property))\b",
        low,
    ):
        return "Investment property"
    if re.search(
        r"\bsecondary\b"
        r"|\bsec+ondary\b"
        r"|\bsecond\s+home\b"
        r"|\bsecond\s+property\b"
        r"|\bvacation\s+home\b"
        r"|\bvacation\s+house\b"
        r"|\bweekend\s+home\b"
        r"|\bski\s+(?:house|home)\b",
        low,
    ):
        return "Secondary property"
    if re.search(
        r"\bprimary\s+property\b"
        r"|\bprimary\b"
        r"|\bprimer\b"
        r"|\bprime\s+home\b"
        r"|\bfirst\s+home\b"
        r"|\bmain\s+home\b"
        r"|\bmain\s+residence\b"
        r"|\bowner\s*[- ]?\s*occupied\b",
        low,
    ):
        return "Primary property"
    return None


def _stt_confidence_triggers_reprompt(conf: Optional[float]) -> bool:
    """True when provider confidence is in [0,1] and below env threshold."""
    if conf is None:
        return False
    try:
        f = float(conf)
    except (TypeError, ValueError):
        return False
    if f < 0.0 or f > 1.0:
        return False
    try:
        thresh = float(os.getenv("PATH2_STT_LOW_CONFIDENCE_REPROMPT_MAX", "0.52"))
    except ValueError:
        thresh = 0.52
    return f < thresh


def infer_occupancy_with_optional_recovery(
    text: str,
    recovery: Optional[str],
) -> tuple[str, Optional[OccupancyType]]:
    """
    Parse occupancy from final transcript, then recovery partial, then concatenations.
    Returns (effective_text_for_raw_user_input, occupancy or None).
    """
    t = (text or "").strip()
    r = (recovery or "").strip()
    seen: list[str] = []
    for cand in (t, r, f"{r} {t}".strip(), f"{t} {r}".strip()):
        if not cand or cand in seen:
            continue
        seen.append(cand)
        occ = infer_occupancy_from_transcript(cand)
        if occ is not None:
            return cand, occ
    return t, None


def infer_for_sale_from_transcript(text: str) -> Optional[bool]:
    """Map short yes/no STT to is_currently_for_sale when the model omits the tool field."""
    if not text or not str(text).strip():
        return None
    text = normalize_answer_scaffold(text)
    if not text:
        return None
    for part in re.split(r"[.!?\n]+", text):
        seg = part.strip().lower()
        if not seg:
            continue
        seg_n = seg.rstrip(".,!?")
        if re.fullmatch(r"(1|one|won|when)", seg_n):
            return True
        if re.fullmatch(r"(2|two|too|to)", seg_n):
            return False
    t = text.strip().lower().rstrip(".,!?")
    if not t:
        return None
    if re.fullmatch(r"(1|one|won|when)", t):
        return True
    if re.fullmatch(r"(2|two|too|to)", t):
        return False
    if re.fullmatch(
        r"(yes|yeah|yep|yup|correct|true|indeed|absolutely|certainly|definitely|sure|of course|it is|uh\s*-?\s*huh)",
        t,
    ):
        return True
    if re.fullmatch(r"(no|nope|nah|negative|false|naw|not|it'?s not|not really)", t):
        return False
    if len(t) <= 28 and re.match(r"^(yes|yeah|yep|yup|it is)\b", t):
        return True
    if len(t) <= 32 and re.match(r"^(no|nope|nah|it'?s not)\b", t):
        return False
    return None


def infer_income_source_from_transcript(text: str) -> Optional[IncomeSource]:
    """Map common spoken variants (without punctuation/dashes) to IncomeSource enum."""
    if not text or not str(text).strip():
        return None
    text = normalize_answer_scaffold(text)
    if not text:
        return None
    for part in re.split(r"[.!?\n]+", text):
        seg = part.strip().lower()
        if not seg:
            continue
        seg_n = seg.rstrip(".,!?")
        if re.fullmatch(r"(1|one|won|when)", seg_n):
            return "Employed full-time"
        if re.fullmatch(r"(2|two|too|to)", seg_n):
            return "Employed part-time"
        if re.fullmatch(r"(3|three|tree|free)", seg_n):
            return "Self-employed"
        if re.fullmatch(r"(4|four|for)", seg_n):
            return "Retired"
        if re.fullmatch(r"(5|five)", seg_n):
            return "Unemployed / furloughed"
    low = text.lower()
    if re.fullmatch(r"\s*(1|one|won|when)\s*[.,!?]?\s*", low):
        return "Employed full-time"
    if re.fullmatch(r"\s*(2|two|too|to)\s*[.,!?]?\s*", low):
        return "Employed part-time"
    if re.fullmatch(r"\s*(3|three|tree|free)\s*[.,!?]?\s*", low):
        return "Self-employed"
    if re.fullmatch(r"\s*(4|four|for)\s*[.,!?]?\s*", low):
        return "Retired"
    if re.fullmatch(r"\s*(5|five)\s*[.,!?]?\s*", low):
        return "Unemployed / furloughed"

    if "full time" in low or "full-time" in low or "fulltime" in low or "employee" in low or "employment" in low:
        return "Employed full-time"
    if "part time" in low or "part-time" in low or "parttime" in low:
        return "Employed part-time"
    if "self employed" in low or "self-employed" in low or "business owner" in low or "contractor" in low:
        return "Self-employed"
    if "retired" in low or "pension" in low or "social security" in low:
        return "Retired"
    if "unemployed" in low or "furlough" in low or "not working" in low:
        return "Unemployed / furloughed"
    if "alimony" in low:
        return "Alimony"
    if "child support" in low:
        return "Child support"
    if "welfare" in low or "public assistance" in low:
        return "Public assistance program"
    if "child support" in low:
        return "Child support"
    if "public assistance" in low:
        return "Public assistance program"
    if "separate maintenance" in low:
        return "Separate maintenance payments"
    # Bare "employed" without qualifier — default to full-time (most common).
    if re.search(r"\bemployed\b", low):
        return "Employed full-time"
    return None


_WORD_NUMS: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
    "a": 1,
}


def _spoken_number_to_int(text: str) -> Optional[int]:
    """Convert English spoken number phrase to int.

    Handles: "two hundred fifty thousand", "eighty thousand",
    "one point five million", "a hundred thousand", etc.
    Returns None on failure.
    """
    tokens = re.split(r"[\s,]+", text.strip().lower())
    # Strip punctuation per-token so inputs like "million." still parse.
    tokens = [re.sub(r"[^a-z0-9]", "", t) for t in tokens if t]
    tokens = [t for t in tokens if t]
    if not tokens:
        return None

    # Handle "point" for decimals before multiplier: "one point five million"
    point_idx = None
    for i, t in enumerate(tokens):
        if t == "point":
            point_idx = i
            break

    if point_idx is not None and point_idx > 0:
        whole_part = _spoken_number_to_int(" ".join(tokens[:point_idx]))
        frac_tokens = tokens[point_idx + 1:]
        if not frac_tokens or whole_part is None:
            return None
        # fraction digit(s) before multiplier
        mult_idx = None
        for i, t in enumerate(frac_tokens):
            if t in ("thousand", "million", "billion"):
                mult_idx = i
                break
        if mult_idx is not None:
            frac_word_part = frac_tokens[:mult_idx]
            mult_word = frac_tokens[mult_idx]
            multiplier = {"thousand": 1_000, "million": 1_000_000}.get(mult_word, 1)
            frac_val = _spoken_number_to_int(" ".join(frac_word_part))
            if frac_val is None:
                return None
            frac_digits = len(str(frac_val))
            result = (whole_part + frac_val / (10 ** frac_digits)) * multiplier
            return int(result)
        return None

    total = 0
    current = 0
    has_value = False
    _TENS = frozenset({20, 30, 40, 50, 60, 70, 80, 90})

    for t in tokens:
        if t in _WORD_NUMS:
            v = _WORD_NUMS[t]
            # Shorthand: "one fifty" = 150, "two thirty" = 230.
            # When current is 1-9 and next word is a tens (20-90), treat as X*100+Y.
            if v in _TENS and 1 <= current <= 9:
                current = current * 100 + v
            else:
                current += v
            has_value = True
        elif t == "hundred":
            current = max(current, 1) * 100
            has_value = True
        elif t == "thousand":
            current = max(current, 1) * 1_000
            total += current
            current = 0
            has_value = True
        elif t == "k":
            current = max(current, 1) * 1_000
            total += current
            current = 0
            has_value = True
        elif t == "million":
            current = max(current, 1) * 1_000_000
            total += current
            current = 0
            has_value = True
        elif t in ("and", "dollars", "dollar", "bucks", "per", "year", "annually"):
            continue
        else:
            return None  # unknown token — bail

    if not has_value:
        return None
    total += current
    return total if total > 0 else None


# Spoken-income heuristics can exceed 100M (e.g. "two two five million" → 225M).
_MAX_INFERRED_ANNUAL_INCOME = 999_999_999
# Below CONFIDENCE_THRESHOLD so update_application_state requires explicit confirmation.
_INCOME_AMBIGUOUS_STT_CONFIDENCE = 0.82


def infer_annual_income_meta(text: str) -> tuple[Optional[int], bool]:
    """Return (annual income in USD or None, fragile_stt_repair).

    ``fragile_stt_repair`` is True when the value depended on a brittle
    spoken-money rule (``one one five``→1.5M or digit-style ``two two
    five``+million→225M). Callers should pass ``confidence_score`` of
    ``_INCOME_AMBIGUOUS_STT_CONFIDENCE`` (below ``CONFIDENCE_THRESHOLD``) so the
    user confirms before save.
    """
    if not text or not str(text).strip():
        return (None, False)
    text = normalize_answer_scaffold(text)
    if not text:
        return (None, False)
    low = text.lower().strip()
    # Keep decimal dot in numeric forms (1.5m), but remove sentence punctuation.
    low = low.replace(",", " ")
    low = re.sub(r"(?<!\d)\.(?!\d)", " ", low)
    low = re.sub(r"[?!]", " ", low)
    low = re.sub(r"\s+", " ", low).strip()
    # Common STT garble on phone audio: "one million" -> "one medium".
    low = re.sub(r"\b(medium|millian|millon)\b", "million", low)
    # Common STT garble: "my annual" -> "manual".
    low = re.sub(r"\bmanual\b", "annual", low)
    stripped = low
    stripped = re.sub(
        r"\b(my|the|our|this|that|it)\b",
        " ",
        stripped,
    )
    stripped = re.sub(
        r"\b(annual|yearly|year)\s+(pre\s*-?\s*tax|pretax)?\s*income\b",
        " ",
        stripped,
    )
    stripped = re.sub(
        r"\b(pre\s*-?\s*tax|pretax|income|salary|is|it'?s|about|around|"
        r"could\s+be|would\s+be|might\s+be|maybe|probably|per\s+annum|"
        r"per\s+year|a\s+year|annually)\b",
        " ",
        stripped,
    )
    stripped = re.sub(r"\s+", " ", stripped).strip()

    _millionish = r"(?=\s*(?:million|\bm\b|\bk\b))"

    # "two two five million" / "… m" / "… k" = three digit words then multiplier
    # (225M / 225K). Do NOT treat as dropped "point" (that would be 2.5M — wrong).
    if re.search(rf"\btwo\s+two\s+five\b{_millionish}", low):
        if re.search(r"\btwo\s+two\s+five\s+million\b", low) or re.search(
            r"\btwo\s+two\s+five\s+m\b", low
        ):
            val = 225 * 1_000_000
        elif re.search(r"\btwo\s+two\s+five\s+k\b", low):
            val = 225 * 1_000
        else:
            val = None
        if val is not None and 0 < val <= _MAX_INFERRED_ANNUAL_INCOME:
            return (val, True)

    # STT drops "point": "one one five million" -> sums 1+1+5 then × million (wrong).
    one_one_five_point_sub = False
    if re.search(rf"\bone\s+one\s+five\b{_millionish}", low):
        one_one_five_point_sub = True
        low = re.sub(rf"\bone\s+one\s+five\b{_millionish}", "one point five", low)

    # --- suffix shorthand: "1M", "1.5m", "80K", "80 k" ---
    m = re.search(r"(\d+(?:\.\d+)?)\s*m\b", low)
    if m:
        val = int(float(m.group(1)) * 1_000_000)
        if 0 < val <= _MAX_INFERRED_ANNUAL_INCOME:
            return (val, one_one_five_point_sub)
    m = re.search(r"(\d+(?:\.\d+)?)\s*k\b", low)
    if m:
        val = int(float(m.group(1)) * 1_000)
        if 0 < val <= _MAX_INFERRED_ANNUAL_INCOME:
            return (val, one_one_five_point_sub)
    m = re.fullmatch(r"([a-z\s-]+)\s+k\b", low)
    if m:
        spoken_k = _spoken_number_to_int(m.group(1).replace("-", " "))
        if spoken_k is not None:
            val = spoken_k * 1_000
            if 0 < val <= _MAX_INFERRED_ANNUAL_INCOME:
                return (val, one_one_five_point_sub)

    # --- raw digit string: "1-0-0-0-0-0" / "1000000" / "100000" ---
    digits_only = re.sub(r"\D", "", low)
    if digits_only:
        try:
            val = int(digits_only)
            if 0 < val <= _MAX_INFERRED_ANNUAL_INCOME:
                return (val, one_one_five_point_sub)
        except ValueError:
            pass

    # --- spoken digit-sequence: "one zero zero zero zero zero" ---
    digit_word_map = {
        "zero": "0",
        "oh": "0",
        "o": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
    }
    wtokens = [re.sub(r"[^a-z0-9]", "", t) for t in stripped.split()]
    if wtokens and all(t in digit_word_map for t in wtokens):
        seq = "".join(digit_word_map[t] for t in wtokens).lstrip("0") or "0"
        try:
            val = int(seq)
            if 0 < val <= _MAX_INFERRED_ANNUAL_INCOME:
                return (val, one_one_five_point_sub)
        except ValueError:
            pass

    # --- spoken words: "eighty thousand", "one point five million" ---
    spoken = _spoken_number_to_int(low) or _spoken_number_to_int(stripped)
    if spoken is not None and spoken >= 1_000 and spoken <= _MAX_INFERRED_ANNUAL_INCOME:
        return (spoken, one_one_five_point_sub)

    # --- contextual bare number: "eighty" -> 80_000 in HELOC income context ---
    # Small bare numbers (1-999) likely mean thousands.
    if spoken is not None and 1 <= spoken <= 999:
        scaled = spoken * 1_000
        if 0 < scaled <= _MAX_INFERRED_ANNUAL_INCOME:
            return (scaled, one_one_five_point_sub)

    return (None, False)


def infer_annual_income_from_transcript(text: str) -> Optional[int]:
    """Map spoken annual income to integer dollars.

    Handles: raw digits ("100000"), suffix shorthand ("80K", "1.5M"),
    spoken words ("eighty thousand", "two hundred fifty thousand"),
    and contextual bare numbers ("eighty" -> 80000 in HELOC context).
    """
    value, _meta = infer_annual_income_meta(text)
    return value


def infer_other_income_from_transcript(text: str) -> Optional[int]:
    """Infer optional other income; map clear negatives to 0."""
    if not text or not str(text).strip():
        return None
    text = normalize_answer_scaffold(text)
    if not text:
        return None
    low = text.lower().strip()
    normalized = re.sub(r"[^\w\s]", " ", low)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    # Repeated negatives from 8kHz STT like "no, no" should still map to zero.
    if re.fullmatch(r"(no|nope|none|nothing|zero|0)(\s+(no|nope|none|nothing|zero|0))*", normalized):
        return 0
    if normalized in {
        "no other income",
        "no any other income",
        "none",
        "nope",
        "no",
        "zero",
        "0",
        "nothing",
    }:
        return 0
    # Spoken negatives that omit bare "no" (STT / natural phrasing).
    _other_income_negative_phrases = (
        "not at this moment",
        "not at the moment",
        "not right now",
        "not currently",
        "nothing right now",
        "nothing else right now",
        "no additional income",
        "dont have any other",
        "don't have any other",
        "do not have any other",
        "dont have other",
        "don't have other",
        "no other",
        "not any other",
    )
    if any(p in normalized for p in _other_income_negative_phrases):
        return 0
    # Fold apostrophes so "don't" / "I'm" style STT matches phrase list entries.
    apostrophe_fold = re.sub(r"[''`]", "", normalized)
    if any(p in apostrophe_fold for p in _other_income_negative_phrases):
        return 0
    if "dont have any other" in apostrophe_fold or "do not have any other" in apostrophe_fold:
        return 0
    # Punctuation strip splits "don't" → "don t" before apostrophe removal.
    contraction_fold = re.sub(r"\bdon\s+t\b", "dont", normalized, flags=re.IGNORECASE)
    contraction_fold = re.sub(r"\bdo\s+not\b", "donot", contraction_fold, flags=re.IGNORECASE)
    if "dont have any other" in contraction_fold or "donot have any other" in contraction_fold:
        return 0
    return infer_annual_income_from_transcript(text)


_MONTH_NAMES: dict[str, int] = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6,
    "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
_MONTH_PAT = "|".join(_MONTH_NAMES.keys())

# Regex fragments for infer_partial_dob_month_day (must match previous inlined patterns exactly).
_DOB_PARTIAL_SPOKEN_ORDINALS_ALT_RE = (
    "first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    "eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|"
    "seventeenth|eighteenth|nineteenth|twentieth|"
    "twenty\\s*-?\\s*first|twenty\\s*-?\\s*second|twenty\\s*-?\\s*third|"
    "twenty\\s*-?\\s*fourth|twenty\\s*-?\\s*fifth|twenty\\s*-?\\s*sixth|"
    "twenty\\s*-?\\s*seventh|twenty\\s*-?\\s*eighth|twenty\\s*-?\\s*ninth|"
    "thirtieth|thirty\\s*-?\\s*first"
)
_DOB_PARTIAL_CARDINAL_DAY_WORDS_RE = (
    r"one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
    r"eighteen|nineteen|twenty|twenty\s*-?\s*(?:one|two|three|four|five|six|seven|eight|nine)|"
    r"thirty|thirty\s*-?\s*one"
)
_DOB_SPOKEN_YEAR_PREFIX_RE = "eighteen|nineteen|twenty"

_ORDINAL_WORDS: dict[str, str] = {
    "first": "one",
    "second": "two",
    "third": "three",
    "fourth": "four",
    "fifth": "five",
    "sixth": "six",
    "seventh": "seven",
    "eighth": "eight",
    "ninth": "nine",
    "tenth": "ten",
    "eleventh": "eleven",
    "twelfth": "twelve",
    "thirteenth": "thirteen",
    "fourteenth": "fourteen",
    "fifteenth": "fifteen",
    "sixteenth": "sixteen",
    "seventeenth": "seventeen",
    "eighteenth": "eighteen",
    "nineteenth": "nineteen",
    "twentieth": "twenty",
    "thirtieth": "thirty",
}


def _parse_spoken_day(value: str) -> Optional[int]:
    cleaned = re.sub(r"[^a-z0-9\s-]", " ", value.lower())
    cleaned = re.sub(r"\b(\d{1,2})(?:st|nd|rd|th)\b", r"\1", cleaned)
    cleaned = cleaned.replace("-", " ")
    tokens = [
        _ORDINAL_WORDS.get(token, token)
        for token in re.split(r"\s+", cleaned.strip())
        if token
    ]
    if not tokens:
        return None
    if len(tokens) == 1 and tokens[0].isdigit():
        day = int(tokens[0])
    else:
        day = _spoken_number_to_int(" ".join(tokens))
    if day is not None and 1 <= day <= 31:
        return day
    return None


def _parse_spoken_year(value: str) -> Optional[int]:
    cleaned = re.sub(r"[^a-z0-9\s-]", " ", value.lower()).replace("-", " ")
    tokens = [t for t in re.split(r"\s+", cleaned.strip()) if t]
    if not tokens:
        return None
    digits = "".join(t for t in tokens if t.isdigit())
    if len(digits) == 4 and all(t.isdigit() for t in tokens):
        year = int(digits)
        return year if 1900 <= year <= 2010 else None
    if len(tokens) == 1 and tokens[0].isdigit():
        year = int(tokens[0])
        return year if 1900 <= year <= 2010 else None

    if len(tokens) >= 2:
        prefix = _spoken_number_to_int(tokens[0])
        suffix = _spoken_number_to_int(" ".join(tokens[1:]))
        if prefix in {19, 20} and suffix is not None and 0 <= suffix <= 99:
            year = prefix * 100 + suffix
            if 1900 <= year <= 2010:
                return year

    year = _spoken_number_to_int(" ".join(tokens))
    if year is not None and 1900 <= year <= 2010:
        return year
    return None


def infer_dob_from_transcript(text: str) -> Optional[str]:
    """Parse DOB from numeric (MM/DD/YYYY) or spoken ('January 5th 1990') forms."""
    if not text or not str(text).strip():
        return None
    text = normalize_answer_scaffold(text)
    if not text:
        return None

    # Numeric: 01/05/1990 or 01-05-1990
    m = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", text)
    if m:
        mm, dd, yyyy = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mm <= 12 and 1 <= dd <= 31 and 1900 <= yyyy <= 2010:
            return f"{mm:02d}/{dd:02d}/{yyyy:04d}"

    low = text.lower().strip()
    # Deepgram/phone garble: "first Jan ..." can arrive as "first ten ...".
    low = re.sub(
        rf"\b(first|1st)\s+ten\s+(?={_DOB_SPOKEN_YEAR_PREFIX_RE}|\d{{4}}\b)",
        r"\1 jan ",
        low,
    )

    # Spoken: "January 5th 1990", "march 15 1985"
    m = re.search(
        rf"\b({_MONTH_PAT})\s+(\d{{1,2}})\s*(?:st|nd|rd|th)?\s*,?\s*(\d{{4}})\b",
        low,
    )
    if m:
        mm = _MONTH_NAMES[m.group(1)]
        dd, yyyy = int(m.group(2)), int(m.group(3))
        if 1 <= dd <= 31 and 1900 <= yyyy <= 2010:
            return f"{mm:02d}/{dd:02d}/{yyyy:04d}"

    # Reversed: "5th January 1990"
    m = re.search(
        rf"\b(\d{{1,2}})\s*(?:st|nd|rd|th)?\s+({_MONTH_PAT})\s+(\d{{4}})\b",
        low,
    )
    if m:
        dd = int(m.group(1))
        mm = _MONTH_NAMES[m.group(2)]
        yyyy = int(m.group(3))
        if 1 <= dd <= 31 and 1900 <= yyyy <= 2010:
            return f"{mm:02d}/{dd:02d}/{yyyy:04d}"

    # Fully spoken: "January first nineteen ninety two"
    m = re.search(
        rf"\b({_MONTH_PAT})\s+([a-z0-9\s-]+?)\s+((?:{_DOB_SPOKEN_YEAR_PREFIX_RE}|\d{{4}})\b[a-z0-9\s-]*)",
        low,
    )
    if m:
        mm = _MONTH_NAMES[m.group(1)]
        dd = _parse_spoken_day(m.group(2))
        yyyy = _parse_spoken_year(m.group(3))
        if dd is not None and yyyy is not None:
            return f"{mm:02d}/{dd:02d}/{yyyy:04d}"

    # Fully spoken reversed: "first January nineteen ninety two"
    m = re.search(
        rf"\b([a-z0-9\s-]+?)\s+({_MONTH_PAT})\s+((?:{_DOB_SPOKEN_YEAR_PREFIX_RE}|\d{{4}})\b[a-z0-9\s-]*)",
        low,
    )
    if m:
        dd = _parse_spoken_day(m.group(1))
        mm = _MONTH_NAMES[m.group(2)]
        yyyy = _parse_spoken_year(m.group(3))
        if dd is not None and yyyy is not None:
            return f"{mm:02d}/{dd:02d}/{yyyy:04d}"

    return None


PARTIAL_DOB_REPROMPT = (
    "Could you also include the year? For example, month, day, and four-digit year."
)


def infer_partial_dob_month_day(text: str) -> bool:
    if not text or not str(text).strip():
        return False
    text = normalize_answer_scaffold(text)
    if not text:
        return False
    low = text.lower().strip()

    if re.search(rf"\b({_MONTH_PAT})\s+\d{{1,2}}\s*(?:st|nd|rd|th)?\s*\.?$", low):
        return True
    if re.search(rf"\b\d{{1,2}}\s*(?:st|nd|rd|th)?\s+({_MONTH_PAT})\s*\.?$", low):
        return True
    if re.search(
        rf"\b({_MONTH_PAT})\s+(?:{_DOB_PARTIAL_SPOKEN_ORDINALS_ALT_RE})\s*\.?$", low
    ):
        return True
    if re.search(
        rf"\b(?:{_DOB_PARTIAL_SPOKEN_ORDINALS_ALT_RE})\s+(?:of\s+)?({_MONTH_PAT})\s*\.?$", low
    ):
        return True

    if re.fullmatch(rf"in\s+({_MONTH_PAT})\s*\.?", low):
        return True
    if re.fullmatch(rf"({_MONTH_PAT})\s*\.?", low):
        return True

    dm = re.search(
        rf"\b(?:{_DOB_PARTIAL_CARDINAL_DAY_WORDS_RE})\s+(?:{_MONTH_PAT})\b", low
    )
    if dm:
        tail = low[dm.end() :].strip().strip(".!?")
        if not tail:
            return True
        if re.match(r"^(?:19|20)\d{2}\b", tail):
            return False
        if re.match(rf"^(?:{_DOB_SPOKEN_YEAR_PREFIX_RE})\b", tail):
            return False
        return True

    return False


PARTIAL_MOBILE_PHONE_REPROMPT = (
    "Please say your full 10-digit mobile phone number."
)


def infer_phone_from_transcript(text: str) -> Optional[str]:
    """Parse US phone from spoken digits."""
    if not text or not str(text).strip():
        return None
    text = normalize_answer_scaffold(text)
    if not text:
        return None
    d = re.sub(r"\D", "", text)
    if len(d) == 11 and d.startswith("1"):
        d = d[1:]
    if len(d) == 10:
        return d
    spoken = _spoken_digit_sequence_to_digits(text)
    if len(spoken) == 11 and spoken.startswith("1"):
        spoken = spoken[1:]
    if len(spoken) == 10:
        return spoken
    return None


def _spoken_digit_sequence_to_digits(text: str) -> str:
    digit_word_map = {
        "zero": "0",
        "oh": "0",
        "o": "0",
        "one": "1",
        "won": "1",
        "two": "2",
        "too": "2",
        "to": "2",
        "three": "3",
        "four": "4",
        "for": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "ate": "8",
        "nine": "9",
    }
    cleaned = normalize_answer_scaffold(text)
    cleaned = re.sub(r"\b(area|code|phone|number|is|my|mobile|cell)\b", " ", cleaned.lower())
    tokens = re.findall(r"[a-z]+|\d", cleaned)
    digits: list[str] = []
    for token in tokens:
        if token.isdigit():
            digits.append(token)
        elif token in digit_word_map:
            digits.append(digit_word_map[token])
        else:
            return ""
    return "".join(digits)


def get_contact_buffer_mode() -> Optional[Literal["phone", "email"]]:
    """Contact stage: buffer STT finals while caller spells phone or email."""
    if get_current_conversation_stage() != "CONTACT":
        return None
    if application_state.mobile_phone_number is None:
        return "phone"
    if application_state.email_address is None:
        return "email"
    return None


_SPOKEN_EMAIL_NUM_AUX = frozenset({"hundred", "thousand", "million", "and"})
# Tokens handled by ``_spoken_digit_sequence_to_digits`` (single-digit spell-out only).
_SPOKEN_EMAIL_SPELL_DIGIT_WORDS = frozenset(
    {
        "zero",
        "oh",
        "o",
        "one",
        "won",
        "two",
        "too",
        "to",
        "three",
        "four",
        "for",
        "five",
        "six",
        "seven",
        "eight",
        "ate",
        "nine",
    }
)


def _chunk_is_digit_spell_only(tokens: list[str]) -> bool:
    """True when every token is a single-digit spoken word (not eighty, twelve, …)."""
    if not tokens:
        return False
    for t in tokens:
        tok = re.sub(r"[^a-z0-9]", "", (t or "").lower())
        if not tok:
            return False
        if tok.isdigit():
            if len(tok) != 1:
                return False
            continue
        if tok not in _SPOKEN_EMAIL_SPELL_DIGIT_WORDS:
            return False
    return True


def _token_is_spoken_email_number_piece(t: str) -> bool:
    tok = re.sub(r"[^a-z0-9]", "", (t or "").lower())
    if not tok:
        return False
    if tok.isdigit():
        return True
    return tok in _WORD_NUMS or tok in _SPOKEN_EMAIL_NUM_AUX


def _expand_spoken_numbers_in_email_segment(segment: str) -> str:
    """
    Turn spoken numbers in an email local-part into digits before '@' / space stripping.

    Examples: ``eighty four`` → ``84``; digit-by-digit ``eight four`` → ``84``.
    """
    segment = (segment or "").strip().replace("-", " ")
    if not segment:
        return segment
    tokens = re.findall(r"\S+", segment)
    out: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        if not _token_is_spoken_email_number_piece(tokens[i]):
            out.append(tokens[i])
            i += 1
            continue
        max_span = min(14, n - i)
        replaced = False
        for length in range(max_span, 0, -1):
            chunk = tokens[i : i + length]
            if not all(_token_is_spoken_email_number_piece(t) for t in chunk):
                continue
            phrase = " ".join(chunk)
            num_val: Optional[int] = None
            digit_run = ""

            if _chunk_is_digit_spell_only(chunk):
                digit_run = _spoken_digit_sequence_to_digits(phrase)
                if digit_run:
                    out.append(digit_run)
                    i += length
                    replaced = True
                    break
                num_val = _spoken_number_to_int(phrase)
            else:
                num_val = _spoken_number_to_int(phrase)
                if num_val is None:
                    digit_run = _spoken_digit_sequence_to_digits(phrase)

            if num_val is not None:
                out.append(str(num_val))
                i += length
                replaced = True
                break
            if digit_run:
                out.append(digit_run)
                i += length
                replaced = True
                break
        if not replaced:
            out.append(tokens[i])
            i += 1
    return " ".join(out)


def _normalize_spoken_email_domain(rhs: str) -> str:
    """Collapse ``g mail dot``, ``yahoo dot co dot uk``, ``gmail period com``, etc."""
    s = (rhs or "").strip().lower()
    if not s:
        return s
    changed = True
    while changed:
        changed = False
        ns = re.sub(r"\s+(?:dot|period)\s+", ".", s)
        if ns != s:
            s = ns
            changed = True
    s = re.sub(r"\s+(?:dot|period)\s*$", "", s).strip()
    s = re.sub(r"\s+", "", s)
    if s and "." not in s:
        s = f"{s}.com"
    return s


def _normalize_spoken_email_local(lhs: str) -> str:
    expanded = _expand_spoken_numbers_in_email_segment((lhs or "").strip())
    raw_l = expanded.strip().lower()
    raw_l = (
        raw_l.replace(" dot ", ".")
        .replace(" period ", ".")
        .replace(" underscore ", "_")
        .replace(" dash ", "-")
        .replace(" hyphen ", "-")
    )
    return raw_l.replace(" ", "")


def _normalize_spoken_email_separator_phrases(raw: str) -> str:
    """Map spoken @-synonyms so split/join sees a single `` at `` delimiter."""
    s = re.sub(r"\s+", " ", (raw or "").strip())
    low = s.lower()
    low = re.sub(r"\bat\s+the\s+rate\b", " at ", low)
    low = re.sub(r"\bat\s+sign\b", " at ", low)
    return low


def infer_email_from_transcript(text: str) -> Optional[str]:
    """Parse email from typed form or spoken 'at'/'dot' variants."""
    if not text or not str(text).strip():
        return None
    text = normalize_answer_scaffold(text)
    if not text:
        return None
    raw = _normalize_spoken_email_separator_phrases(text.strip().lower())
    if " at " in raw:
        left, right = raw.split(" at ", 1)
        local = _normalize_spoken_email_local(left).rstrip(".")
        normalized = f"{local}@{_normalize_spoken_email_domain(right)}"
    elif "@" in raw:
        lp, rp = raw.split("@", 1)
        local = _normalize_spoken_email_local(lp).rstrip(".")
        normalized = f"{local}@{_normalize_spoken_email_domain(rp)}"
    else:
        return None

    normalized = normalized.rstrip(".,;:!?")
    if STRICT_EMAIL_REGEX.match(normalized):
        return normalized
    return None


def _normalize_email_for_tool_compare(addr: str) -> str:
    return (addr or "").strip().lower().rstrip(".")


def _email_save_matches_transcript(raw_user_input: Optional[str], saved: str) -> bool:
    """Require parsed spoken email to match saved value — stops LLM invented addresses."""
    if not raw_user_input or not saved:
        return False
    parsed = infer_email_from_transcript(raw_user_input)
    if not parsed:
        return False
    return _normalize_email_for_tool_compare(parsed) == _normalize_email_for_tool_compare(
        str(saved)
    )


def _spoken_email_evidence(raw_user_input: Optional[str]) -> bool:
    """True when caller speech plausibly contained an email — blocks LLM-invented addresses."""
    if not raw_user_input or not str(raw_user_input).strip():
        return False
    low = str(raw_user_input).strip().lower()
    if "@" in low:
        return True
    if " at " in low:
        return True
    if re.search(r"\bat\s+the\s+rate\b", low):
        return True
    if re.search(r"\bat\s+sign\b", low):
        return True
    if re.search(r"\bat\s+[\w.+-]+\s+dot\s+", low):
        return True
    return False


def _spoken_dob_year_in_transcript(raw_user_input: str, yyyy: int) -> bool:
    """True when transcript plausibly mentions ``yyyy`` (digits or spoken)."""
    if yyyy < 1900 or yyyy > 2010:
        return True
    raw = str(raw_user_input).strip()
    if str(yyyy) in raw:
        return True
    low = normalize_answer_scaffold(raw).lower()
    low = re.sub(r"[^\w\s]", " ", low)
    low = re.sub(r"\s+", " ", low).strip()
    words = low.split()
    for i in range(len(words)):
        for j in range(i + 1, min(i + 8, len(words) + 1)):
            frag = " ".join(words[i:j])
            if _parse_spoken_year(frag) == yyyy:
                return True
    yy = yyyy % 100
    yy_pat = rf"\b{yy:02d}\b|\b{yy}\b"
    if re.search(yy_pat, low):
        century = yyyy // 100
        if century == 19 and re.search(r"\bnineteen\b", low):
            return True
        if century == 20 and re.search(r"\btwenty\b", low):
            return True
    return False


def _month_tokens_present_for_dob(mm: int, low_sp: str) -> bool:
    keys = [k for k, v in _MONTH_NAMES.items() if v == mm]
    return any(re.search(rf"\b{re.escape(k)}\b", low_sp) for k in keys)


def _day_token_evidence_for_dob(dd: int, low_for_day: str) -> bool:
    tokens = [t for t in re.split(r"\s+", low_for_day) if t]
    for tok in tokens:
        m = re.match(r"^(\d{1,2})(?:st|nd|rd|th)?$", tok, re.I)
        if m and int(m.group(1)) == dd:
            return True
        if _parse_spoken_day(tok) == dd:
            return True
    return False


def _transcript_supports_saved_dob(raw_user_input: Optional[str], dob: str) -> bool:
    """Block LLM/tool hallucinated DOB when speech lacks month, day, and year cues."""
    if not raw_user_input or not str(raw_user_input).strip() or not dob:
        return False
    parts = dob.strip().split("/")
    if len(parts) != 3:
        return True
    try:
        mm, dd, yyyy = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return True

    raw = str(raw_user_input).strip()
    scaffolded = normalize_answer_scaffold(raw) or raw
    low_sp = scaffolded.lower()
    low_sp = re.sub(r"\s+", " ", re.sub(r"[^\w\s/\.-]", " ", low_sp)).strip()

    for sep in ("/", "-", "."):
        esc = re.escape(sep)
        if re.search(
            rf"\b{mm}\s*{esc}\s*{dd}\s*{esc}\s*{yyyy}\b",
            low_sp,
        ) or re.search(
            rf"\b{mm:02d}\s*{esc}\s*{dd:02d}\s*{esc}\s*{yyyy}\b",
            low_sp,
        ):
            return True

    if not _month_tokens_present_for_dob(mm, low_sp):
        return False
    if not _spoken_dob_year_in_transcript(raw, yyyy):
        return False

    low_for_day = re.sub(r"\b(?:19|20)\d{2}\b", " ", low_sp)
    low_for_day = re.sub(r"\s+", " ", low_for_day).strip()
    return _day_token_evidence_for_dob(dd, low_for_day)


def infer_property_address_from_transcript(text: str) -> Optional[str]:
    """
    Parse likely street line from noisy STT.
    Requires leading street number to avoid accidental captures.
    """
    if not text or not str(text).strip():
        return None
    text = normalize_answer_scaffold(text)
    if not text:
        return None
    low = text.lower().strip()
    # Normalize punctuation/noise around the leading street number.
    low = re.sub(r"[^\w\s,.-]", " ", low)
    low = re.sub(r"\s+", " ", low).strip()
    had_comma = "," in low
    low_number_scan = re.sub(r"[,.-]+", " ", low)
    low_number_scan = re.sub(r"\s+", " ", low_number_scan).strip()
    words_to_digits = re.match(
        r"^((?:(?:zero|oh|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
        r"nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
        r"hundred)\s+){1,8})([a-z].*)$",
        low_number_scan,
    )
    if words_to_digits:
        g1 = words_to_digits.group(1).strip()
        g2 = (words_to_digits.group(2) or "").strip()
        merged_phrase = g1
        tail = g2
        if g2:
            parts2 = g2.split()
            if parts2:
                first_w = re.sub(r"[^a-z0-9]", "", parts2[0]).lower()
                if first_w in _WORD_NUMS:
                    combined = _spoken_number_to_int(f"{g1} {first_w}".strip())
                    head_only = _spoken_number_to_int(g1)
                    if combined is not None and (
                        head_only is None or combined != head_only
                    ):
                        merged_phrase = f"{g1} {first_w}".strip()
                        tail = " ".join(parts2[1:]).strip()
        digit_seq = _spoken_digit_sequence_to_digits(
            merged_phrase.replace("-", " ")
        )
        if digit_seq:
            number = int(digit_seq)
        else:
            number = _spoken_number_to_int(merged_phrase.replace("-", " "))
        if number is not None:
            low = f"{number} {tail}".strip()
    low = re.sub(r"\s+", " ", low).strip()
    low = re.sub(r"\b(uh|um|ah|er|like)\b", " ", low)
    low = re.sub(r"\s+", " ", low).strip()
    if re.search(r"\b(lost|list|students|items|below|where)\b", low):
        return None
    m = re.match(r"^\s*(\d{1,6})[\s,.-]+([a-z0-9\s.-]{3,})", low)
    if not m:
        return None
    num = m.group(1)
    rest = m.group(2)
    rest = re.split(r",|\bin\s+[a-z]+\b|\bcalifornia\b|\bca\b", rest, maxsplit=1)[0].strip()
    if not rest:
        return None
    rest = re.sub(r"\bblvd\b", "boulevard", rest)
    rest = re.sub(r"\bave\b", "avenue", rest)
    rest = re.sub(r"\brd\b", "road", rest)
    rest = re.sub(r"\bdr\b", "drive", rest)
    rest = re.sub(r"\bln\b", "lane", rest)
    rest = re.sub(r"\bct\b", "court", rest)
    rest = re.sub(r"\bpkwy\b", "parkway", rest)
    rest = re.sub(r"\bst\b", "street", rest)
    ordinal_word_to_num = {
        "first": "1st",
        "second": "2nd",
        "third": "3rd",
        "fourth": "4th",
        "fifth": "5th",
        "sixth": "6th",
        "seventh": "7th",
        "eighth": "8th",
        "ninth": "9th",
        "tenth": "10th",
        "eleventh": "11th",
        "twelfth": "12th",
        "thirteenth": "13th",
        "fourteenth": "14th",
        "fifteenth": "15th",
        "sixteenth": "16th",
        "seventeenth": "17th",
        "eighteenth": "18th",
        "nineteenth": "19th",
        "twentieth": "20th",
    }
    for w, n in ordinal_word_to_num.items():
        rest = re.sub(rf"\b{w}\b", n, rest)
    rest = re.sub(r"\s+", " ", rest).strip(" .,-")
    if had_comma:
        street_line = re.match(
            r"^(.+?\b(?:street|avenue|boulevard|road|drive|lane|court|parkway|place)\b)",
            rest,
        )
        if street_line:
            rest = street_line.group(1).strip()
    if re.fullmatch(r"(los angeles|california|ca)", rest):
        return None
    if len(rest) < 3:
        return None
    rest_l = rest.lower()
    if re.fullmatch(
        r"(zero|oh|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
        r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)",
        rest_l,
    ):
        return None
    formatted_tokens: list[str] = []
    for token in rest.split():
        if re.search(r"\d", token):
            formatted_tokens.append(token.lower())
        else:
            formatted_tokens.append(token.capitalize())
    return f"{num} {' '.join(formatted_tokens)}"


_STATE_NAME_TO_ABBR = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM",
    "new york": "NY", "north carolina": "NC", "north dakota": "ND", "ohio": "OH",
    "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI",
    "south carolina": "SC", "south dakota": "SD", "tennessee": "TN", "texas": "TX",
    "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
}


def _state_abbr_from_text(text: str) -> Optional[str]:
    normalized = re.sub(r"[^a-zA-Z\s]", " ", text or "")
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    for name, abbr in _STATE_NAME_TO_ABBR.items():
        if re.search(rf"\b{re.escape(name)}\b", normalized):
            return abbr
    m = re.search(r"\b([A-Z]{2})\b\s*$", (text or "").strip().upper())
    if m and m.group(1) in set(_STATE_NAME_TO_ABBR.values()):
        return m.group(1)
    return None


def infer_property_city_state_from_transcript(text: str) -> dict[str, str]:
    if not text or not str(text).strip():
        return {}
    text = normalize_answer_scaffold(text)
    if not text:
        return {}
    out: dict[str, str] = {}

    parts = [p.strip(" .") for p in re.split(r",", str(text)) if p.strip(" .")]
    if len(parts) >= 2:
        city_part = parts[1]
        state_from_city_part = _state_abbr_from_text(city_part)
        city = city_part
        if state_from_city_part:
            out["state"] = state_from_city_part
            city = re.sub(r"\b[A-Z]{2}\b", " ", city, flags=re.IGNORECASE)
            for state_name in _STATE_NAME_TO_ABBR:
                city = re.sub(rf"\b{re.escape(state_name)}\b", " ", city, flags=re.IGNORECASE)
        city = re.sub(r"\b\d{5}\b", " ", city)
        city = re.sub(r"[^a-zA-Z\s.-]", " ", city)
        city = re.sub(r"\s+", " ", city).strip(" .-")
        if city and not _state_abbr_from_text(city):
            out["city"] = city.title()
        if len(parts) >= 3:
            state = _state_abbr_from_text(parts[2])
            if state:
                out["state"] = state

    if "state" not in out:
        state = _state_abbr_from_text(str(text))
        if state:
            out["state"] = state
    return out


def build_address_autocomplete_type_query(
    street_line: Optional[str],
    spoken_transcript_hint: Optional[str],
) -> Optional[str]:
    street = (street_line or "").strip()
    if not street:
        return None
    sh = (spoken_transcript_hint or "").strip()
    if not sh:
        return street
    hints = infer_property_city_state_from_transcript(sh)
    tail = " ".join(
        x.strip()
        for x in (hints.get("city"), hints.get("state"))
        if x and str(x).strip()
    )
    if tail:
        return f"{street} {tail}".strip()
    return street


def _name_slot_focus_from_missing(missing_blob: str) -> Optional[str]:
    """If get_missing_fields() includes Ask next, map to first- or last-name-only intake."""
    ask = _ask_from_missing_line(missing_blob)
    if not ask:
        return None
    q = ask.lower()
    if "first name" in q:
        return "first"
    if "last name" in q:
        return "last"
    return None


def infer_name_fields_from_transcript(
    text: str,
    *,
    need_first: bool,
    need_last: bool,
    slot_focus: Optional[str] = None,
) -> dict[str, str]:
    """
    Parse simple spoken legal name replies for NAME stage safety net.

    Conservative by design:
    - Accept one clean token for first or last name.
    - Accept two clean tokens as first+last when both missing (unless slot_focus or dup rule).
    - Reject obvious non-name utterances ("but it's sad", "yes", "primary", etc.).

    slot_focus:
    - ``"first"``: only fill first name (e.g. Ask next is \"First name?\" even if user says
      two words — second word is ignored for this save so we still ask last name).
    - ``"last"``: only fill last name (e.g. \"Last name?\"); two tokens → use second token.
    - ``None``: use need_first / need_last only.

    Duplicate tokens (e.g. STT \"Prasad? Prasad?\"): never set first and last to the same
    string in one shot — return first name only when both slots are still needed.
    """
    if not text or not str(text).strip():
        return {}

    if transcript_is_status_check(text):
        return {}
    raw_s = text.strip().lower()
    explicit_name = bool(
        re.match(
            r"^\W*(my\s+(?:full\s+)?name\s+is|name\s+is|i\s+am|i'?m|this\s+is)\b",
            raw_s,
        )
    )
    s = normalize_answer_scaffold(text).lower()
    if not s:
        return {}
    spelled = _spoken_letters_to_word(s)
    if spelled:
        if slot_focus == "first" and need_first:
            return {"first_name": spelled}
        if slot_focus == "last" and need_last:
            return {"last_name": spelled}
        if need_first and not need_last:
            return {"first_name": spelled}
        if need_last and not need_first:
            return {"last_name": spelled}
    # Hard-reject common off-topic/garbled phrases seen on PSTN STT for name turns.
    # This avoids writing junk like "Whatever" / "Brother" into legal name fields.
    if re.search(
        r"\b("
        r"whatever|brother|sister|mother|father|"
        r"thank(s| you)?|god|word|"
        r"good|hear|there|hello|hi|"
        r"not\s+sure|sure|repeat"
        r")\b",
        s,
    ):
        return {}
    if (
        re.match(r"^\W*my\b", s)
        and not re.match(r"^\W*(my name is|name is|i am|im)\b", s)
    ):
        return {}
    s = re.sub(r"^(my name is|name is|i am|im|it's|it is|this is)\s+", "", s)
    tokens = re.findall(r"[a-zA-Z][a-zA-Z'-]{0,30}", s)
    if not tokens:
        return {}
    if len(tokens) > 2:
        return {}
    if re.search(r"\d", s):
        return {}

    stop = {
        "yes",
        "no",
        "primary",
        "secondary",
        "investment",
        "owner",
        "sole",
        "joint",
        "million",
        "thousand",
        "income",
        "address",
        "hello",
        "hi",
        "sad",
        "but",
        "whatever",
        "brother",
        "sister",
        "mother",
        "father",
        "god",
        "word",
        "thanks",
        "thank",
    }
    if any(t.lower() in stop for t in tokens):
        return {}

    # STT echo / repetition: do not treat as first+last with identical strings.
    if len(tokens) == 2 and tokens[0].lower() == tokens[1].lower():
        t = tokens[0].title()
        if need_first and need_last:
            return {"first_name": t}
        if need_first:
            return {"first_name": t}
        if need_last:
            return {"last_name": t}
        return {}

    # Align with scripted Ask next so one question does not capture both names.
    if slot_focus == "first" and need_first:
        if len(tokens) == 1:
            return {"first_name": tokens[0].title()}
        if len(tokens) == 2:
            return {"first_name": tokens[0].title()}
        return {}

    if slot_focus == "last" and need_last:
        if len(tokens) == 1:
            return {"last_name": tokens[0].title()}
        if len(tokens) == 2:
            return {"last_name": tokens[1].title()}
        return {}

    out: dict[str, str] = {}
    if len(tokens) == 1:
        t = tokens[0].title()
        if need_first:
            out["first_name"] = t
        elif need_last:
            out["last_name"] = t
        return out

    # len == 2
    t1, t2 = tokens[0].title(), tokens[1].title()
    if need_first and need_last:
        if not explicit_name:
            out["first_name"] = t1
            return out
        out["first_name"] = t1
        out["last_name"] = t2
    elif need_first:
        out["first_name"] = t1
    elif need_last:
        out["last_name"] = t2
    return out


def _spoken_letters_to_word(text: str) -> Optional[str]:
    """Join slow spelled names: 'P R A S A D' -> 'Prasad'."""
    if not text or not str(text).strip():
        return None
    letter_words = {
        "a": "a",
        "ay": "a",
        "b": "b",
        "be": "b",
        "bee": "b",
        "c": "c",
        "see": "c",
        "sea": "c",
        "d": "d",
        "dee": "d",
        "e": "e",
        "ee": "e",
        "f": "f",
        "eff": "f",
        "g": "g",
        "gee": "g",
        "h": "h",
        "aitch": "h",
        "i": "i",
        "eye": "i",
        "j": "j",
        "jay": "j",
        "k": "k",
        "kay": "k",
        "l": "l",
        "el": "l",
        "m": "m",
        "em": "m",
        "n": "n",
        "en": "n",
        "o": "o",
        "oh": "o",
        "p": "p",
        "pee": "p",
        "q": "q",
        "cue": "q",
        "queue": "q",
        "r": "r",
        "are": "r",
        "s": "s",
        "ess": "s",
        "t": "t",
        "tea": "t",
        "tee": "t",
        "u": "u",
        "you": "u",
        "v": "v",
        "vee": "v",
        "w": "w",
        "doubleyou": "w",
        "x": "x",
        "ex": "x",
        "y": "y",
        "why": "y",
        "z": "z",
        "zee": "z",
        "zed": "z",
    }
    cleaned = re.sub(r"\b(letter|letters|spelled|spell|is)\b", " ", text.lower())
    tokens = re.findall(r"[a-z]+", cleaned)
    letters = [letter_words.get(token) for token in tokens]
    if not letters or any(letter is None for letter in letters):
        return None
    if len(letters) < 2 or len(letters) > 30:
        return None
    return "".join(letters).title()


def infer_yes_no_from_transcript(text: str) -> Optional[bool]:
    """Map short affirmation/negation utterances to bool."""
    if not text or not str(text).strip():
        return None
    text = normalize_answer_scaffold(text)
    if not text:
        return None
    t = text.strip().lower().rstrip(".,!?")
    compact = re.sub(r"[^\w\s']", " ", t)
    compact = re.sub(r"\s+", " ", compact).strip()
    repeated_tokens = compact.split()
    if repeated_tokens and len(set(repeated_tokens)) == 1:
        t = repeated_tokens[0]
    if re.search(
        r"\b(yes|yeah|yep|yup|correct|right|agree|agreed|absolutely|definitely|sure)\b",
        compact,
    ) and not re.search(
        r"\b(no|nope|nah|don't|do not|not really|negative|disagree)\b",
        compact,
    ):
        return True
    if re.search(r"\b(no|nope|nah|negative|disagree)\b", compact):
        return False
    if t in {
        "yes", "yeah", "yep", "yup", "correct", "right",
        "i agree", "agree", "sure", "of course", "absolutely",
        "definitely", "certainly",
    }:
        return True
    if t in {"no", "nope", "nah", "not really", "i don't think so", "negative"}:
        return False
    return None


def detect_uncertainty(raw_input: str) -> list[str]:
    """
    Scan raw user speech for hedging / uncertainty language.
    
    Returns a list of matched markers found in the input.
    Empty list means the user sounded confident.
    """
    if not raw_input:
        return []
    
    lowered = raw_input.lower()
    return [marker for marker in UNCERTAINTY_MARKERS if marker in lowered]


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
    current_stage = _CONV_STAGE_TO_BROWSER.get(get_current_conversation_stage(), "unknown")
    allowed_now = stage_to_allowed_fields.get(current_stage)

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
        if (
            _use_deterministic_address_parsing()
            and local_vars.get("address_line_1") is None
            and application_state.address_line_1 is None
        ):
            inferred_addr = infer_property_address_from_transcript(raw_user_input)
            if inferred_addr:
                local_vars["address_line_1"] = inferred_addr
        if local_vars.get("ownership_type") is None and local_vars.get(
            "occupancy_type"
        ) is None:
            inferred_o = infer_ownership_from_transcript(raw_user_input)
            if inferred_o:
                local_vars["ownership_type"] = inferred_o
        if local_vars.get("occupancy_type") is None:
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
                f"Ask the caller to confirm: {conf_ask}"
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
                "Is the property currently listed for sale?"
            )
        if field == "is_currently_for_sale":
            return (
                "I'm assuming it is not listed for sale for now. "
                "What's your annual pre-tax income?"
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
        stage = _CONV_STAGE_TO_BROWSER.get(_conv_stage, "unknown")

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
            ask = _ask_from_missing_line(blob) or "Great! Can you tell me the property address you'd like to apply for?"
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
            occ_q = ("primary home" in ask_m and "second home" in ask_m) or (
                "investment property" in ask_m and "second home" in ask_m
            )
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
