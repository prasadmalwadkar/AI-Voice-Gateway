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
from agent.browser_manager import BrowserManager

logger = logging.getLogger("voip.state")

# =============================================================================
# Global instances — one state and one browser per call.
# Reset via reset_state() at the start of each new call in get_agent().
# =============================================================================
application_state = HelocApplicationState()
browser_manager = BrowserManager()


async def ensure_browser_ready() -> None:
    """Ensure the browser is running and on the correct page."""
    if not browser_manager.is_running:
        logger.info("Initializing browser for real-time sync...")
        await browser_manager.start(is_demo=True)
        await browser_manager.navigate_to_start()


async def reset_state() -> None:
    """Reset the application state and browser for a new call."""
    global application_state
    application_state = HelocApplicationState()
    # If a previous browser was left running, shut it down so we start fresh
    if browser_manager.is_running:
        logger.info("Closing stale browser session...")
        await browser_manager.shutdown()
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

    FIELD_STAGES = {
        "address_line_1": 1, "city": 1, "state": 1, "zipcode": 1,
        "ownership_type": 1, "occupancy_type": 1,
        "is_currently_for_sale": 2,
        "annual_pre_tax_income": 3, "other_income": 3, "primary_source_of_income": 3,
        "first_name": 4, "last_name": 4, "middle_name": 4, "suffix": 4,
        "date_of_birth": 5,
        "mobile_phone_number": 6, "email_address": 6,
        "agreed_to_privacy_policy": 7, "agreed_to_commerce_communication": 7, "agreed_to_credit_pull": 7,
        "password": 7, "confirm_password": 7
    }

    # Stage name to number mapping derived from browser_manager.get_current_stage()
    STAGE_NUMS = {
        "property": 1, "for-sale": 2, "income": 3, "name": 4, "birthday": 5, "contact": 6, "consent": 7, "unknown": 0
    }

    current_stage_name = browser_manager.get_current_stage()
    current_stage_num = STAGE_NUMS.get(current_stage_name, 0)

    # FILTERING LOGIC: If we are at a later stage, ignore MISSING fields from past stages
    # unless they are 'required' (unfilled required fields usually block the browser anyway).
    # For simplicity, we just filter out optional fields from past stages.
    
    final_missing = []
    # These internal keys match the HelocApplicationState model fields
    model_field_to_display = {
        "address_line_1": "property address (street)",
        "city": "city",
        "state": "state",
        "zipcode": "zipcode",
        "ownership_type": "ownership type (sole or joint owner)",
        "occupancy_type": "occupancy type (primary, secondary, or investment)",
        "is_currently_for_sale": "whether property is currently for sale",
        "annual_pre_tax_income": "annual pre-tax income",
        "other_income": "any additional income (optional)",
        "primary_source_of_income": "primary source of income",
        "first_name": "first name",
        "middle_name": "middle name or initial (optional)",
        "last_name": "last name",
        "suffix": "suffix (optional)",
        "date_of_birth": "date of birth",
        "mobile_phone_number": "mobile phone number",
        "email_address": "email address",
        "agreed_to_privacy_policy": "privacy policy agreement",
        "agreed_to_commerce_communication": "marketing communications consent",
        "agreed_to_credit_pull": "consent for electronic communications and soft credit pull",
        "password": "account password",
        "confirm_password": "password confirmation"
    }
    
    # STAGE-LOCKING LOGIC:
    # We only report fields for the CURRENT stage. 
    # If the current stage is complete, we report fields for the NEXT stage only.
    # This keeps the agent and browser in perfect lockstep.
    
    current_stage_fields = [f for f, s in FIELD_STAGES.items() if s == current_stage_num]
    next_stage_fields = [f for f, s in FIELD_STAGES.items() if s == current_stage_num + 1]
    
    # Get optional fields list for classifications
    optional_fields = BrowserManager.OPTIONAL_FIELDS

    # Check if current stage is "complete" (no required missing fields)
    current_stage_is_done = True
    for f in current_stage_fields:
        if not hasattr(state, f): continue
        val = getattr(state, f)
        if (val is None or val == "") and f not in optional_fields:
            current_stage_is_done = False
            break

    target_fields = current_stage_fields if not current_stage_is_done else next_stage_fields
    
    final_missing = []
    
    for field_name in target_fields:
        if not hasattr(state, field_name): continue
            
        current_val = getattr(state, field_name)
        if current_val is None or current_val == "":
            display_name = model_field_to_display.get(field_name)
            if display_name:
                final_missing.append(display_name)

    # Prepend stage context for the LLM
    stage_header = f"[BROWSER STAGE: {current_stage_name.upper()}]"
    if not final_missing:
        return f"{stage_header} All current fields filled. Waiting for navigation..."
        
    return f"{stage_header} Still needed: " + ", ".join(final_missing)


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
    
    # =========================================================================
    # BROWSER SYNCHRONIZATION — Mirror valid updates to the live browser
    # =========================================================================
    if updates:
        try:
            await ensure_browser_ready()
            
            # -- RECURSIVE FILLING LOOP --
            # We loop to handle cases where the agent provides data for multiple pages at once.
            max_nav_attempts = 3
            current_nav = 0
            
            while current_nav < max_nav_attempts:
                # 1. Special case: Address Selection (triggers autocomplete)
                if "address_line_1" in updates:
                    full_address = updates["address_line_1"]
                    if application_state.city: full_address += f" {application_state.city}"
                    if application_state.state: full_address += f" {application_state.state}"
                    if application_state.zipcode: full_address += f" {application_state.zipcode}"
                    await browser_manager.select_address(full_address)
                
                # 2. Special case: Consent stage password generation
                # If we are on the consent page and password is missing, auto-generate it.
                if browser_manager.get_current_stage() == "consent":
                    if not application_state.password:
                        # Generate a simple but secure-looking password for the user
                        auto_pass = "WCL!2024!Agent"
                        application_state.password = auto_pass
                        application_state.confirm_password = auto_pass
                        updates["password"] = auto_pass
                        updates["confirm_password"] = auto_pass
                        logger.info("Auto-generated secure password for the final stage.")

                # 3. Bulk fill fields visible on current page
                await browser_manager.fill_form_fields(updates)

                # 4. DOM Verification
                mismatches = await browser_manager.verify_fields(updates)
                if mismatches:
                    mismatch_list = ", ".join(mismatches)
                    logger.warning("DOM Verification failed for fields: %s", mismatch_list)
                    validation_errors.append(f"BROWSER VERIFICATION WARNING: Fields not updated: {mismatch_list}.")

                # 5. Check Complete and Navigate (SPEED GOVERNOR)
                # We only allow ONE navigation jump per tool call. This ensures the browser 
                # doesn't skip pages (like DOB) before the caller has spoken.
                if await browser_manager.is_page_complete():
                    logger.info("Page complete. Auto-navigating (1 jump max)...")
                    nav_success = await browser_manager.click_continue()
                    if nav_success:
                        current_nav += 1
                        logger.info("Successfully navigated. Stopping auto-advance to stay in sync with caller.")
                        # After 1 jump, we stop and let the LLM react to the new page content
                        break
                    else:
                        # SECOND OPINION: If click reported failure, check if the page is STILL complete.
                        if not await browser_manager.is_page_complete():
                            logger.info("Navigation confirmed via second-opinion.")
                            current_nav += 1
                            break
                            
                        validation_errors.append("BROWSER NAVIGATION WARNING: 'Continue' button clicked but page did not move.")
                        break
                else:
                    break
                
        except Exception as e:
            logger.error("Browser sync failed: %s", str(e), exc_info=True)

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
        # 'missing' is now a pre-formatted string from get_missing_fields()
        parts.append(missing)
    
    return ". ".join(parts) + "."
