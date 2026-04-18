import asyncio
import sys
import os
import logging

# Ensure project root is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.state_manager import update_application_state, application_state, shutdown_browser
from agent.schema import OccupancyType, OwnershipType, IncomeSource

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("tests.cli_sync")

async def interactive_cli():
    print("\n" + "="*60)
    print("   HELOC Orchestration Glue — CLI-to-Browser Sync Test")
    print("="*60)
    print("Enter data in the format: field=value (e.g., first_name=John)")
    print("To update multiple fields: field1=val1, field2=val2")
    print("Special fields: address, ownership, occupancy, income, name, dob, phone, email")
    print("Type 'exit' to quit.\n")

    try:
        while True:
            user_input = input(">> ").strip()
            if not user_input or user_input.lower() == 'exit':
                break

            updates = {}
            # Split by comma for multiple updates
            parts = [p.strip() for p in user_input.split(',')]
            
            for part in parts:
                if '=' not in part:
                    print(f"  [Error] Invalid format: '{part}'. Use field=value.")
                    continue
                
                key, val = part.split('=', 1)
                key = key.strip().lower()
                val = val.strip()

                # Map shortcuts to actual schema keys if needed
                key_map = {
                    "address": "address_line_1",
                    "fname": "first_name",
                    "lname": "last_name",
                    "dob": "date_of_birth",
                    "phone": "mobile_phone_number",
                    "email": "email_address",
                    "income": "annual_pre_tax_income",
                    "ownership": "ownership_type",
                    "occupancy": "occupancy_type",
                    "primary_income": "primary_source_of_income",
                    "source": "primary_source_of_income",
                    "for_sale": "is_currently_for_sale",
                    "terms": "agreed_to_privacy_policy",
                    "privacy": "agreed_to_privacy_policy",
                    "credit": "agreed_to_credit_pull",
                    "consent": "agreed_to_credit_pull",
                    "birthdate": "date_of_birth",
                }
                key = key_map.get(key, key)

                # Type casting for boolean and int
                if val.lower() in ['true', 'yes', 'y']:
                    val = True
                elif val.lower() in ['false', 'no', 'n']:
                    val = False
                elif val.isdigit():
                    val = int(val)
                
                updates[key] = val

            if updates:
                print(f"  [Sync] Calling update_application_state with: {updates}")
                try:
                    # Note: ctx is None as we're not running in a full agent context
                    # Access .func because the decorator returns a ToolDescriptor
                    result = await update_application_state.func(
                        None, 
                        **updates, 
                        confidence_score=1.0, 
                        raw_user_input=user_input
                    )
                    print(f"  [Result] {result}")
                    print(f"  [State] Current State: {application_state.model_dump(exclude_none=True)}")
                except TypeError as e:
                    print(f"  [Error] Parameter mismatch: {e}")
                    print(f"  [Tip] Use the exact Pydantic field names or supported shortcuts.")
                except Exception as e:
                    print(f"  [Error] Sync failed: {e}")
            
    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down browser...")
        await shutdown_browser()
        print("Done.")

if __name__ == "__main__":
    asyncio.run(interactive_cli())
