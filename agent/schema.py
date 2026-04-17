from pydantic import BaseModel, ConfigDict, Field
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
    state: Optional[str] = Field(None, description="State of the property.")
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
    agreed_to_credit_pull: Optional[bool] = Field(None, description="Must be True. User verbally consents to the Electronic Communications Policy and soft credit pull.")

    def generate_password(self, pincode: str) -> str:
        """
        Dynamically generates a secure password meeting the 12-character requirement
        (Uppercase, lowercase, number, and symbol). 
        Format: firstname#pincode. If length < 12, uses firstname_lastname#pincode.
        """
        if not self.first_name:
            return "" # Cannot generate without a name
            
        first = self.first_name.capitalize()
        last = self.last_name.capitalize() if self.last_name else ""
        
        # Base attempt: Firstname#Pincode
        attempt_1 = f"{first}#{pincode}"
        
        if len(attempt_1) >= 12:
            return attempt_1
            
        # Fallback: Firstname_Lastname#Pincode
        attempt_2 = f"{first}_{last}#{pincode}"
        
        # If it's STILL under 12 chars (e.g., user is "Ed Wu"), pad it with an extra symbol and year
        if len(attempt_2) < 12:
             return f"{first}_{last}#{pincode}!2026"
             
        return attempt_2
