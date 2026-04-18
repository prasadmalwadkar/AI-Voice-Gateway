import sys
import os

# Add the project root to sys.path so we can import from 'agent'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.schema import HelocApplicationState

def test_heloc_schema():
    # 1. Test basic data assignment
    state = HelocApplicationState(
        first_name="John",
        last_name="Smith",
        address_line_1="742 Evergreen Terrace",
        city="Springfield",
        state="IL",
        zipcode="62704",
        occupancy_type="Primary property" # Must match Literal exactly
    )
    
    print(f"Name: {state.first_name} {state.last_name}")
    print(f"Address: {state.address_line_1}, {state.city}, {state.state} {state.zipcode}")
    print(f"Occupancy: {state.occupancy_type}")

    # 2. Test the Password Generation Logic (The 12-char rule)
    # Scenario A: Normal name (John#62704 = 10 chars) -> Should trigger firstname_lastname fallback
    pass_1 = state.generate_password("62704")
    print(f"\nPassword 1 (John_Smith#62704): {pass_1} | Length: {len(pass_1)}")

    # Scenario B: Very short name (Ed#62704 = 8 chars) -> Should trigger padding fallback
    state.first_name = "Ed"
    state.last_name = "Wu"
    pass_2 = state.generate_password("62704")
    print(f"Password 2 (Ed_Wu#62704!2026): {pass_2} | Length: {len(pass_2)}")

    # Scenario C: Long name -> Should pass on first attempt
    state.first_name = "Alexander"
    state.last_name = "Hamilton"
    pass_3 = state.generate_password("62704")
    print(f"Password 3 (Alexander#62704): {pass_3} | Length: {len(pass_3)}")

    # 3. Test Validation Error (Try an invalid dropdown value)
    try:
        state.occupancy_type = "My house" # Not in our Literal list
    except ValueError as e:
        print("\nValidation Success: Correctly rejected invalid occupancy type.")

    # =========================================================================
    # 4. Field Validator Tests
    # =========================================================================
    print("\n--- Field Validator Tests ---")
    
    # 4a. Phone number: valid 10-digit
    state.mobile_phone_number = "213-555-0100"
    print(f"✅ Phone (stripped): {state.mobile_phone_number}")
    
    # 4b. Phone number: with country code prefix (should strip leading 1)
    state.mobile_phone_number = "1-213-555-0100"
    print(f"✅ Phone (stripped country code): {state.mobile_phone_number}")
    
    # 4c. Phone number: too short (should fail)
    try:
        state.mobile_phone_number = "555-1234"
    except (ValueError, Exception):
        print("✅ Phone rejected: 7 digits correctly caught.")
    
    # 4d. Zipcode: valid
    state.zipcode = "90007"
    print(f"✅ Zipcode: {state.zipcode}")
    
    # 4e. Zipcode: invalid (too short)
    try:
        state.zipcode = "900"
    except (ValueError, Exception):
        print("✅ Zipcode rejected: 3 digits correctly caught.")
    
    # 4f. Date of birth: valid
    state.date_of_birth = "03/15/1990"
    print(f"✅ DOB: {state.date_of_birth}")
    
    # 4g. Date of birth: invalid format
    try:
        state.date_of_birth = "March 15 1990"
    except (ValueError, Exception):
        print("✅ DOB rejected: bad format correctly caught.")
    
    # 4h. Email: valid
    state.email_address = "john.smith@example.com"
    print(f"✅ Email: {state.email_address}")
    
    # 4i. Email: invalid
    try:
        state.email_address = "not-an-email"
    except (ValueError, Exception):
        print("✅ Email rejected: no @ sign correctly caught.")
    
    # 4j. State: valid
    state.state = "ca"  # should normalize to CA
    print(f"✅ State (normalized): {state.state}")
    
    # 4k. State: invalid
    try:
        state.state = "XX"
    except (ValueError, Exception):
        print("✅ State rejected: 'XX' correctly caught.")

if __name__ == "__main__":
    test_heloc_schema()
