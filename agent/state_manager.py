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
- Legal consent fields (credit pull, privacy policy) require explicit
  high-confidence "I agree" — vague responses are not accepted.
- UNCERTAINTY_MARKERS provide a deterministic safety net: even if the LLM
  reports high confidence, hedging language in the user's speech ("maybe",
  "or", "I think") blocks the save and forces re-confirmation.
"""

import logging
from typing import Annotated, Optional

from pydantic import ValidationError
from line.llm_agent import loopback_tool

from agent.schema import HelocApplicationState, OccupancyType, OwnershipType, IncomeSource

logger = logging.getLogger("voip.state")

# =============================================================================
# Global application state — one instance per call.
# Reset via reset_state() at the start of each new call in get_agent().
# =============================================================================
application_state = HelocApplicationState()


def reset_state() -> None:
    """Reset the application state for a new call. Called from get_agent()."""
    global application_state
    application_state = HelocApplicationState()
    logger.info("Application state reset for new call.")


def get_missing_fields() -> list[str]:
    """
    Return a human-readable list of fields that are still None.
    
    Groups fields by section so the agent can ask about related fields
    together (e.g., "Can I get your city and state?" rather than asking
    each individually).
    """
    missing = []
    state = application_state

    # -- Property Details --
    if not state.address_line_1:
        missing.append("street address")
    if not state.city:
        missing.append("city")
    if not state.state:
        missing.append("state")
    if not state.zipcode:
        missing.append("zipcode")
    if not state.ownership_type:
        missing.append("ownership type (sole or joint owner)")
    if not state.occupancy_type:
        missing.append("occupancy type (primary, secondary, or investment)")
    
    # -- For Sale Status --
    if state.is_currently_for_sale is None:
        missing.append("whether property is currently for sale")
    
    # -- Income --
    if state.annual_pre_tax_income is None:
        missing.append("annual pre-tax income")
    if not state.primary_source_of_income:
        missing.append("primary source of income")
    
    # -- Legal Name --
    if not state.first_name:
        missing.append("first name")
    if not state.last_name:
        missing.append("last name")
    
    # -- Birthday --
    if not state.date_of_birth:
        missing.append("date of birth")
    
    # -- Contact --
    if not state.mobile_phone_number:
        missing.append("mobile phone number")
    if not state.email_address:
        missing.append("email address")
    
    # -- Consents --
    if not state.agreed_to_privacy_policy:
        missing.append("privacy policy agreement")
    if not state.agreed_to_credit_pull:
        missing.append("credit pull consent")

    return missing


# =============================================================================
# Confidence threshold for field extraction. If the LLM's self-assessed
# confidence is below this, the tool instructs re-confirmation before
# moving to the next question.
# =============================================================================
CONFIDENCE_THRESHOLD = 0.85

# Fields that require explicit high-confidence consent (legal compliance).
# Vague responses like "sure" or "I guess" are not sufficient.
HIGH_CONFIDENCE_FIELDS = {"agreed_to_credit_pull", "agreed_to_privacy_policy"}

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
#    Still needed: zipcode, ownership type, ..."
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
    agreed_to_privacy_policy: Annotated[Optional[bool], "User explicitly says 'I agree' to Privacy Policy and Terms of Service"] = None,
    agreed_to_credit_pull: Annotated[Optional[bool], "User explicitly says 'I agree' to Electronic Communications Policy and soft credit pull"] = None,
    confidence_score: Annotated[Optional[float], "Your confidence (0.0-1.0) in the accuracy of the extracted data. Set lower if the caller was unclear, mumbled, or you had to guess."] = None,
    raw_user_input: Annotated[Optional[str], "The exact transcript of what the user just said. Always pass this so the system can detect uncertainty."] = None,
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
    needs_confirmation = []  # Fields saved but below confidence threshold
    local_vars = {
        "address_line_1": address_line_1,
        "apt_suite": apt_suite,
        "city": city,
        "state": state,
        "zipcode": zipcode,
        "ownership_type": ownership_type,
        "occupancy_type": occupancy_type,
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
        "agreed_to_credit_pull": agreed_to_credit_pull,
    }
    
    # Determine effective confidence (default to 1.0 if not provided)
    score = confidence_score if confidence_score is not None else 1.0
    is_low_confidence = score < CONFIDENCE_THRESHOLD
    
    # =========================================================================
    # SAFETY NET: Scan raw user speech for uncertainty markers.
    # This overrides the LLM's self-reported confidence — if the user said
    # "maybe" or "I think", we block the save regardless of what the LLM says.
    # =========================================================================
    matched_markers = detect_uncertainty(raw_user_input or "")
    
    if matched_markers:
        # Identify which fields the LLM was trying to update
        pending_fields = [k for k, v in local_vars.items() if v is not None]
        
        if pending_fields:
            fields_list = ", ".join(pending_fields)
            markers_list = ", ".join(f"'{m}'" for m in matched_markers)
            logger.warning(
                "Uncertainty detected in user speech — markers: %s | pending fields: %s | raw: '%s'",
                markers_list, fields_list, raw_user_input,
            )
            
            missing = get_missing_fields()
            missing_str = f" Still needed: {', '.join(missing)}." if missing else ""
            
            return (
                f"Update PENDING. Uncertainty detected in user speech: "
                f"'{raw_user_input}' (markers: {markers_list}). "
                f"You MUST ask the user to explicitly confirm the values for "
                f"{fields_list} before I can save them to the state. "
                f"Ask: 'I want to make sure I have this right — is it [value]?'"
                f"{missing_str}"
            )
    
    for field_name, value in local_vars.items():
        if value is not None:
            # -- Special handling: legal consent fields require explicit agreement --
            if field_name in HIGH_CONFIDENCE_FIELDS:
                if is_low_confidence or not value:
                    consent_label = (
                        "credit pull disclosure" if field_name == "agreed_to_credit_pull"
                        else "Privacy Policy and Terms of Service"
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
                
                # -- Low confidence: saved but needs verbal re-confirmation --
                if is_low_confidence and field_name not in HIGH_CONFIDENCE_FIELDS:
                    needs_confirmation.append(
                        f"'{field_name}' was captured as '{value}'"
                    )
                    
            except (ValidationError, ValueError) as e:
                # Validator rejected this value — tell the LLM what went wrong
                # so it can ask the caller to correct it.
                error_msg = str(e).split("\n")[-1].strip() if "\n" in str(e) else str(e)
                validation_errors.append(
                    f"Invalid data for {field_name}. {error_msg} "
                    "Please explain the requirement to the user and ask again."
                )
                logger.warning("Validation failed for %s='%s': %s", field_name, value, error_msg)
    
    if not updates and not validation_errors:
        return "No fields were updated. Please extract information from the user's speech before calling this tool."
    
    # Build response parts
    parts = []
    
    # Report validation errors first so the LLM knows to re-ask
    if validation_errors:
        parts.append(f"VALIDATION ERRORS: {'; '.join(validation_errors)}")
    
    if updates:
        updated_names = ", ".join(updates.keys())
        logger.info("State updated — fields: %s (confidence: %.2f)", updated_names, score)
        logger.debug("Current state: %s", application_state.model_dump(exclude_none=True))
        parts.append(f"Updated: {updated_names}")
    
    # Flag low-confidence fields for verbal re-confirmation
    if needs_confirmation:
        confirm_list = "; ".join(needs_confirmation)
        parts.append(
            f"NEEDS VERBAL RE-CONFIRMATION (confidence {score:.0%}): {confirm_list}. "
            "Please ask: 'I heard [value], is that correct?' before moving on."
        )
    
    # Build the feedback message for the LLM
    missing = get_missing_fields()
    
    if not missing:
        parts.append(
            "ALL required fields are now collected! "
            "Please confirm the information with the caller and proceed to submit."
        )
    else:
        parts.append(f"Still needed from the caller: {', '.join(missing)}")
    
    return ". ".join(parts) + "."
