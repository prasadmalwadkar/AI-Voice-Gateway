"""
Smoke test for select_address() — launches a headed browser,
types a full address, and verifies the autocomplete fills
City / State / Zip automatically.

Usage:
    python tests/test_address.py
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent.browser_manager import BrowserManager


# Test address — use a complete street address so autocomplete narrows correctly.
TEST_ADDRESS = "325 W Adams Blvd Los Angeles CA 90007"


async def test_select_address():
    print("--- Address Autocomplete Test ---\n")
    manager = BrowserManager()

    try:
        # 1. Launch in demo mode so we can watch it
        print("1. Launching browser (demo mode)...")
        await manager.start(is_demo=True)

        # 2. Navigate to the HELOC form
        print("2. Navigating to registration page...")
        await manager.navigate_to_start()

        title = await manager.page.title()
        print(f"   Page loaded: '{title}'\n")

        # 3. Run select_address with the full address
        print(f"3. Typing address: '{TEST_ADDRESS}'")
        print("   Waiting for autocomplete dropdown...")
        await manager.select_address(TEST_ADDRESS)
        print("   ✓ Address selected!\n")

        # 4. Read back the auto-populated fields to verify
        print("4. Verifying auto-populated fields:")
        page = manager.page

        # Give the UI a moment to settle after networkidle
        await asyncio.sleep(1)

        # Try to read city/state/zip values from the form inputs
        # These selectors may need adjustment based on the actual form
        fields_to_check = {
            "Address": "input[placeholder='Search for your address']",
            "City": "input[name='city'], input[placeholder*='City'], #city",
            "State": "input[name='state'], select[name='state'], #state",
            "Zipcode": "input[name='zipcode'], input[name='zip'], #zipcode, #zip",
        }

        for label, selector in fields_to_check.items():
            try:
                el = page.locator(selector).first
                value = await el.input_value() if await el.count() > 0 else "N/A"
                status = "✓" if value and value != "N/A" else "✗ (empty)"
                print(f"   {label}: '{value}' {status}")
            except Exception:
                print(f"   {label}: (could not read)")

        # 5. Pause so the user can visually verify
        print("\n5. Pausing 8 seconds for visual confirmation...")
        await asyncio.sleep(8)

    except Exception as e:
        print(f"\n✗ Test FAILED: {e}")
    finally:
        print("\n6. Shutting down browser...")
        await manager.shutdown()
        print("   Done.")


if __name__ == "__main__":
    asyncio.run(test_select_address())
