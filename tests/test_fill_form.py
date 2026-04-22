import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent.browser_manager import BrowserManager


TEST_ADDRESS = "325 W Adams Blvd Los Angeles CA 90007"

PAGE_1_DATA = {
    "ownership_type": "Sole owner",
    "occupancy_type": "Primary property",
}
PAGE_2_DATA = { "is_currently_for_sale": False }
PAGE_3_DATA = { 
    "annual_pre_tax_income": "100000",
    "other_income": "1000",
    "primary_source_of_income": "Employed full-time"
}
PAGE_4_DATA = {
    "first_name": "Test",
    "middle_name": "M",
    "last_name": "User",
    "suffix": "Jr"
}
PAGE_5_DATA = { "date_of_birth": "01/01/1990" } 
PAGE_6_DATA = {
    "mobile_phone_number": "2135550100",
    "email_address": "test@example.com"
}
PAGE_7_DATA = {
    "password": "Test_User#90007",
    "confirm_password": "Test_User#90007",
    "agreed_to_privacy_policy": True,
    "agreed_to_credit_pull": True,
}

@pytest.mark.browser
@pytest.mark.skipif(
    os.getenv("RUN_BROWSER_TESTS") != "1",
    reason="Set RUN_BROWSER_TESTS=1 to run headed Playwright form test",
)
async def test_form_filling():
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("--- Form Filling Test ---\n")
    manager = BrowserManager()
    
    pages = [
        ("Ownership & Occupancy", PAGE_1_DATA),
        ("For Sale Status", PAGE_2_DATA),
        ("Annual Income", PAGE_3_DATA),
        ("Legal Name", PAGE_4_DATA),
        ("Birthdate", PAGE_5_DATA),
        ("Phone & Email", PAGE_6_DATA),
        ("Password & Terms", PAGE_7_DATA),
    ]

    try:
        print("1. Launching browser (demo mode)...")
        await manager.start(is_demo=True)
        await manager.navigate_to_start()
        
        print(f"2. Typing address: '{TEST_ADDRESS}'")
        await manager.select_address(TEST_ADDRESS)

        for i, (page_name, page_data) in enumerate(pages, start=3):
            print(f"\n{i}. Filling '{page_name}' on current page...")
            await manager.fill_form_fields(page_data)
            print(f"   ✓ '{page_name}' fields processed!")
            
            print(f"   Pausing 3s for visual confirm...")
            await asyncio.sleep(3)
            
            if i < 9: # We want to click continue on everything except the final Password/Submit page
                print(f"   Clicking continue...")
                await manager.click_continue()

        print("\nAll pages processed! Pausing 5 seconds...")
        await asyncio.sleep(5)
        await asyncio.sleep(5)

    except Exception as e:
        print(f"\n✗ Test FAILED: {e}")
    finally:
        print("\n9. Shutting down browser...")
        await manager.shutdown()
        print("   Done.")


if __name__ == "__main__":
    asyncio.run(test_form_filling())
