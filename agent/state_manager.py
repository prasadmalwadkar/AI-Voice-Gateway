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
"""

import logging
from typing import Annotated, Optional

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
# Loopback Tool — The LLM calls this after extracting data from user speech.
#
# How it works:
# 1. User says: "I live at 742 Evergreen Terrace in Springfield, Illinois"
# 2. LLM extracts: address_line_1="742 Evergreen Terrace", city="Springfield",
#    state="IL"
# 3. LLM calls this tool with those 3 fields (other params stay None)
# 4. Tool updates state, returns: "Updated: street address, city, state.
#    Still needed: zipcode, ownership type, ..."
# 5. LLM reads the return value and asks the caller for the next missing item.
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
    annual_pre_tax_income: Annotated[Optional[int], "Total annual pre-tax income in whole dollars"] = None,
    other_income: Annotated[Optional[int], "Additional income (rental, investment, etc.)"] = None,
    primary_source_of_income: Annotated[Optional[IncomeSource], "Primary source of annual income"] = None,
    first_name: Annotated[Optional[str], "User's legal first name"] = None,
    middle_name: Annotated[Optional[str], "User's legal middle name or initial"] = None,
    last_name: Annotated[Optional[str], "User's legal last name"] = None,
    suffix: Annotated[Optional[str], "Name suffix (Jr, Sr, III, etc.)"] = None,
    date_of_birth: Annotated[Optional[str], "Date of birth as MM/DD/YYYY"] = None,
    mobile_phone_number: Annotated[Optional[str], "10-digit mobile phone number"] = None,
    email_address: Annotated[Optional[str], "Email address"] = None,
    agreed_to_privacy_policy: Annotated[Optional[bool], "User agrees to Privacy Policy and Terms of Service"] = None,
    agreed_to_credit_pull: Annotated[Optional[bool], "User consents to Electronic Communications Policy and soft credit pull"] = None,
) -> str:
    """
    Save extracted application data from the user's speech. Call this tool
    whenever the user provides any information relevant to their HELOC
    application. You can update one or many fields at a time — only pass
    the fields you have new information for.
    """
    global application_state
    
    # Collect which fields were actually provided in this call
    updates = {}
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
    
    for field_name, value in local_vars.items():
        if value is not None:
            setattr(application_state, field_name, value)
            updates[field_name] = value
    
    if not updates:
        return "No fields were updated. Please extract information from the user's speech before calling this tool."
    
    # Log the update
    updated_names = ", ".join(updates.keys())
    logger.info("State updated — fields: %s", updated_names)
    logger.debug("Current state: %s", application_state.model_dump(exclude_none=True))
    
    # Build the feedback message for the LLM
    missing = get_missing_fields()
    
    if not missing:
        return (
            f"Updated: {updated_names}. "
            "ALL required fields are now collected! "
            "Please confirm the information with the caller and proceed to submit."
        )
    
    return (
        f"Updated: {updated_names}. "
        f"Still needed from the caller: {', '.join(missing)}."
    )
