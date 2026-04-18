import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent.browser_manager import BrowserManager


async def test_lifecycle():
    print("--- Starting Browser Smoke Test ---")
    manager = BrowserManager()
    
    try:
        # 1. Test Browser Start (Headed for visual verification)
        print("Launching browser in Demo Mode...")
        await manager.start(is_demo=True)
        
        # 2. Test Navigation
        print("Navigating to West Capital Lending...")
        await manager.navigate_to_start()
        
        # 3. Validation
        title = await manager.page.title()
        print(f"Successfully reached: '{title}'")
        
        # Keep it open for 5 seconds so you can see it
        print("Waiting 5 seconds for visual confirmation...")
        await asyncio.sleep(5)
        
    except Exception as e:
        print(f"Test Failed with error: {e}")
    finally:
        # 4. Test Shutdown
        print("Closing browser...")
        await manager.shutdown()
        print("Test Complete.")

if __name__ == "__main__":
    asyncio.run(test_lifecycle())