import re

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Optional, Literal

# Enum types based on the specific dropdowns in the West Capital UI
OccupancyType = Literal["Primary property", "Secondary property", "Investment property"]
OwnershipType = Literal["Sole owner", "Joint owner"] # Assuming Joint is the alternative to Sole
IncomeSource = Literal[
    "Alimony", 
    "Child support", 
    "Employed full-time", 
    "Employed part-time", 
    "Public assistance program", 
    "Retired", 
    "Self-employed", 
    "Separate maintenance payments", 
    "Unemployed / furloughed"
]

class HelocApplicationState(BaseModel):
    """Tracks the state of the user's HELOC application during the call."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    # 1. Property Details
    address_line_1: Optional[str] = Field(None, description="The street address of the property (e.g., 325 W Adams Blvd).")
    apt_suite: Optional[str] = Field(None, description="Apartment or suite number, if applicable. Optional.")
    city: Optional[str] = Field(None, description="City of the property.")
    state: Optional[str] = Field(None, description="2-letter state abbreviation (e.g., CA, NY, IL).")
    zipcode: Optional[str] = Field(None, description="5-digit zipcode of the property.")
    ownership_type: Optional[OwnershipType] = Field(None, description="Whether they are the sole owner or a joint owner.")
    occupancy_type: Optional[OccupancyType] = Field(None, description="How the property is used (Primary, Secondary, or Investment).")
    
    # 2. For Sale Status
    is_currently_for_sale: Optional[bool] = Field(None, description="True if the property is currently listed for sale, False otherwise.")
    
    # 3. Income Details
    annual_pre_tax_income: Optional[int] = Field(None, description="Total annual pre-tax income in whole dollars.")
    other_income: Optional[int] = Field(None, description="Any additional income (rental, investment). Optional.")
    primary_source_of_income: Optional[IncomeSource] = Field(None, description="The primary source of their annual income.")
    
    # 4. Legal Name
    first_name: Optional[str] = Field(None, description="User's legal first name.")
    middle_name: Optional[str] = Field(None, description="User's legal middle name or initial. Optional.")
    last_name: Optional[str] = Field(None, description="User's legal last name.")
    suffix: Optional[str] = Field(None, description="Name suffix (e.g., Jr, Sr, III). Optional.")
    
    # 5. Birthday
    date_of_birth: Optional[str] = Field(None, description="User's date of birth formatted as MM/DD/YYYY.")
    
    # 6. Contact Info
    mobile_phone_number: Optional[str] = Field(None, description="User's 10-digit mobile phone number.")
    email_address: Optional[str] = Field(None, description="User's email address.")

    # 7. Legal & Consents
    agreed_to_privacy_policy: Optional[bool] = Field(None, description="Must be True. User verbally agrees to Privacy Policy and Terms of Service.")
    agreed_to_commerce_communication: Optional[bool] = Field(None, description="Optional. User agrees to marketing/commerce communications.")
    agreed_to_credit_pull: Optional[bool] = Field(None, description="Must be True. User verbally consents to the Electronic Communications Policy and soft credit pull.")
    
    # 8. Account Security (Auto-generated)
    password: Optional[str] = Field(None, description="Auto-generated account password.")
    confirm_password: Optional[str] = Field(None, description="Confirmation of the auto-generated password.")

    # =========================================================================
    # Field Validators — catch bad data before it enters the state.
    # When validation fails, Pydantic raises ValidationError which the
    # state_manager catches and returns as guidance to the LLM.
    # =========================================================================

    @field_validator("mobile_phone_number", mode="before")
    @classmethod
    def validate_phone(cls, v: Optional[str]) -> Optional[str]:
        """Strip formatting and ensure exactly 10 digits."""
        if v is None:
            return v
        # Strip common speech-to-text artifacts: dashes, parens, spaces, dots
        digits = re.sub(r"[^\d]", "", v)
        # Remove leading "1" country code if user said it (e.g., "1-213-555-0100")
        if len(digits) == 11 and digits.startswith("1"):
            digits = digits[1:]
        if len(digits) != 10:
            raise ValueError(
                f"Phone number must be exactly 10 digits, got {len(digits)} digits from '{v}'."
            )
        return digits

    @field_validator("zipcode", mode="before")
    @classmethod
    def validate_zipcode(cls, v: Optional[str]) -> Optional[str]:
        """Ensure exactly 5 digits."""
        if v is None:
            return v
        digits = re.sub(r"[^\d]", "", v)
        if len(digits) != 5:
            raise ValueError(
                f"Zipcode must be exactly 5 digits, got {len(digits)} digits from '{v}'."
            )
        return digits

    @field_validator("date_of_birth", mode="before")
    @classmethod
    def validate_dob(cls, v: Optional[str]) -> Optional[str]:
        """Ensure MM/DD/YYYY format."""
        if v is None:
            return v
        if not re.match(r"^\d{2}/\d{2}/\d{4}$", v):
            raise ValueError(
                f"Date of birth must be in MM/DD/YYYY format, got '{v}'."
            )
        month, day, year = int(v[:2]), int(v[3:5]), int(v[6:])
        if not (1 <= month <= 12):
            raise ValueError(f"Invalid month: {month}. Must be 01-12.")
        if not (1 <= day <= 31):
            raise ValueError(f"Invalid day: {day}. Must be 01-31.")
        if not (1900 <= year <= 2010):
            raise ValueError(f"Invalid year: {year}. Must be between 1900-2010.")
        return v

    @field_validator("email_address", mode="before")
    @classmethod
    def validate_email(cls, v: Optional[str]) -> Optional[str]:
        """Basic email format check."""
        if v is None:
            return v
        v = v.strip().lower()
        if not re.match(r"^[\w.+-]+@[\w-]+\.[\w.-]+$", v):
            raise ValueError(
                f"Invalid email format: '{v}'. Expected format: user@example.com."
            )
        return v

    @field_validator("state", mode="before")
    @classmethod
    def validate_state(cls, v: Optional[str]) -> Optional[str]:
        """Normalize to 2-letter uppercase abbreviation."""
        if v is None:
            return v
        v = v.strip().upper()
        valid_states = {
            "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
            "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
            "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
            "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
            "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
            "DC",
        }
        if v not in valid_states:
            raise ValueError(
                f"'{v}' is not a valid US state abbreviation."
            )
        return v

