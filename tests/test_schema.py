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

if __name__ == "__main__":
    test_heloc_schema()
