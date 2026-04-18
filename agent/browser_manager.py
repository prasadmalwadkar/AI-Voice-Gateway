"""
Browser Automation Engine — Playwright-based BrowserManager
=============================================================

Provides a production-grade, async BrowserManager class that controls a
Chromium instance for automating the West Capital Lending HELOC application.

Design principles:
- **Decoupled**: The Voice Agent sends commands to this module without
  any browser logic leaking into the voice layer.
- **Single Context**: One BrowserContext per session preserves cookies and
  session state across the multi-page form.
- **Demo Mode**: When ``is_demo=True``, the browser launches headed with
  ``slow_mo=500`` so human observers can follow the automation in real time.
- **Async Everywhere**: Every public method is ``async``.
- **Defensive**: Navigation and lifecycle methods are wrapped in try/except
  with structured logging — a failed page load never crashes the caller.

Usage::

    from agent.browser_manager import BrowserManager

    manager = BrowserManager()
    await manager.start(is_demo=True)
    await manager.navigate_to_start()
    # Now integrated with state_manager for real-time extraction sync!
    await manager.shutdown()
"""

import logging
import asyncio
import re
from typing import Optional

from playwright.async_api import (
    async_playwright,
    Browser,
    BrowserContext,
    Page,
    Playwright,
)

logger = logging.getLogger("voip.browser")

# The entry-point URL for the HELOC registration form.
HELOC_REGISTER_URL = "https://heloc.westcapitallending.com/account/heloc/register"

# ---------------------------------------------------------------------------
# CSS Selectors — captured from the live West Capital Lending form.
# Centralised here so a future UI change only requires one update.
# ---------------------------------------------------------------------------
SEL_ADDRESS_INPUT = "input[placeholder='Search for your address']"
SEL_AUTOCOMPLETE_LIST = "ul[role='listbox']"
# Skips the "Address not listed" option at the top of the dropdown.
SEL_FIRST_SUGGESTION = "li[role='option']:not([id*='not-listed'])"


class BrowserManager:
    """Manages the full lifecycle of a Chromium browser for HELOC form automation.

    Attributes:
        _playwright: The Playwright instance that owns the browser process.
        _browser: The Chromium browser instance.
        _context: A single BrowserContext that retains cookies / session state.
        _page: The active Page used for all interactions.
        _is_running: Guard flag to prevent double-start or use-after-shutdown.
    """

    # Selector map mapping Pydantic keys to West Capital CSS selectors
    FIELD_MAP = {
        "address_line_1": "input[name='address_line_1']",
        "apt_suite": "input[data-test-id='property-address-street2']",
        "city": "input[name='city']",
        "state": "input[name='state']",
        "zipcode": "input[name='zipcode']",
        "ownership_type": "select[name='ownership']",
        "occupancy_type": "select[name='occupancy_type']",
        "is_currently_for_sale": "input[name='isListedForSale']",
        "annual_pre_tax_income": "input[data-test-id='income-amount']",
        "other_income": "input[data-test-id='other-income-amount']",
        "primary_source_of_income": "select[name='employmentType']",
        "first_name": "input[data-test-id='first-name']",
        "middle_name": "input[data-test-id='middle-name']",
        "last_name": "input[data-test-id='last-name']",
        "suffix": "select[name='suffix']",
        "date_of_birth": "custom",
        "mobile_phone_number": "input[data-test-id='phone-number']",
        "email_address": "input[data-test-id='email']",
        "password": "input[data-test-id='password']",
        "confirm_password": "input[data-test-id='confirm-password']",
        "agreed_to_privacy_policy": "input[data-test-id='consent-1']",
        "agreed_to_commerce_communication": "input[data-test-id='consent-3']",
        "agreed_to_credit_pull": "input[data-test-id='consent-2']",
    }

    # Fields that are visible but not required for navigation ("Continue")
    OPTIONAL_FIELDS = ["apt_suite", "other_income", "middle_name", "suffix", "agreed_to_commerce_communication"]

    def __init__(self) -> None:
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._is_running: bool = False

    # ------------------------------------------------------------------
    # Properties — safe accessors for the active page
    # ------------------------------------------------------------------

    @property
    def page(self) -> Page:
        """Return the active page, raising if the browser is not running."""
        if self._page is None or not self._is_running:
            raise RuntimeError(
                "BrowserManager is not running. Call start() first."
            )
        return self._page

    @property
    def is_running(self) -> bool:
        """Whether the browser is currently active."""
        return self._is_running

    # ------------------------------------------------------------------
    # Lifecycle — start / navigate / shutdown
    # ------------------------------------------------------------------

    async def start(self, is_demo: bool = False) -> None:
        """Launch a Chromium browser and create a single persistent context.

        Args:
            is_demo: If ``True``, launches in **headed** mode with
                ``slow_mo=500`` and a maximized viewport so a human can
                observe the automation in real time. If ``False``, launches
                headless for server / CI environments.

        Raises:
            RuntimeError: If the browser is already running.
        """
        if self._is_running:
            logger.warning("BrowserManager.start() called but browser is already running.")
            return

        try:
            self._playwright = await async_playwright().start()

            # Build launch kwargs based on demo flag
            launch_kwargs: dict = {
                "headless": not is_demo,
            }
            if is_demo:
                launch_kwargs["slow_mo"] = 500
                # --start-maximized opens the window at full screen size
                launch_kwargs["args"] = ["--start-maximized"]

            self._browser = await self._playwright.chromium.launch(**launch_kwargs)

            # Single context — maintains cookies & session across pages.
            # In demo mode, use no_viewport so the window fills the screen.
            context_kwargs: dict = {}
            if is_demo:
                context_kwargs["no_viewport"] = True

            self._context = await self._browser.new_context(**context_kwargs)
            self._page = await self._context.new_page()

            self._is_running = True
            logger.info(
                "Browser started — mode=%s, headless=%s",
                "demo" if is_demo else "headless",
                not is_demo,
            )

        except Exception:
            logger.exception("Failed to start browser.")
            # Clean up partial state so the caller can retry
            await self._teardown()
            raise

    async def navigate_to_start(self) -> None:
        """Navigate the active page to the HELOC registration form.

        Uses ``wait_until="networkidle"`` to ensure the page (including
        the Address field) is fully loaded before returning.

        Raises:
            RuntimeError: If the browser is not running.
        """
        page = self.page  # will raise if not running

        try:
            response = await page.goto(
                HELOC_REGISTER_URL,
                wait_until="networkidle",
                timeout=30_000,
            )

            status = response.status if response else "no response"
            title = await page.title()

            logger.info(
                "Navigation successful — url=%s, status=%s, title='%s'",
                HELOC_REGISTER_URL,
                status,
                title,
            )

        except Exception:
            logger.exception(
                "Navigation failed — url=%s. The page may be down or unreachable.",
                HELOC_REGISTER_URL,
            )
            raise

    # ------------------------------------------------------------------
    # Form Interaction — address autocomplete
    # ------------------------------------------------------------------

    async def select_address(self, address: str) -> None:
        """Type an address into the search field and select the first autocomplete match.

        Workflow:
        1. Simulate human-speed typing (``delay=100``) into the address
           input to trigger the site's autocomplete API.
        2. Wait up to 3 seconds for the suggestion dropdown to appear.
        3. Click the first real suggestion (skipping "Address not listed").
        4. Wait for ``networkidle`` so City, State, and Zipcode fields are
           auto-populated by the site.

        If the dropdown never appears (API timeout, network issue, etc.),
        the method falls back to ``page.fill()`` and logs a warning so
        the caller can handle City/State/Zip manually.

        Args:
            address: The street address string to type (e.g. "325 W Adams Blvd").

        Raises:
            RuntimeError: If the browser is not running.
        """
        page = self.page  # will raise if not running

        logger.info("Selecting address — typing '%s' into autocomplete field.", address)

        try:
            # 1. Clear any previous value and type slowly to trigger autocomplete
            await page.click(SEL_ADDRESS_INPUT)
            await page.fill(SEL_ADDRESS_INPUT, "")  # clear existing text
            await page.type(SEL_ADDRESS_INPUT, address, delay=100)
            logger.debug("Typed '%s' into address field.", address)

            # 2. Wait for the autocomplete dropdown to appear (3s timeout)
            try:
                await page.wait_for_selector(
                    SEL_AUTOCOMPLETE_LIST,
                    state="visible",
                    timeout=3_000,
                )
                logger.debug("Autocomplete dropdown visible.")

                # 3. Click the first real suggestion
                first_option = page.locator(SEL_FIRST_SUGGESTION).first
                suggestion_text = await first_option.text_content()
                await first_option.click()
                logger.info(
                    "Selected autocomplete suggestion — '%s'",
                    (suggestion_text or "").strip(),
                )

                # 4. Wait for the site to auto-populate City / State / Zip
                await page.wait_for_load_state("networkidle")
                logger.info(
                    "Address fields auto-populated after selecting '%s'.",
                    (suggestion_text or "").strip(),
                )

            except Exception:
                # ---------- SAFETY FALLBACK ----------
                # Dropdown never appeared — fill the field directly.
                logger.warning(
                    "Autocomplete dropdown did not appear within 3s for '%s'. "
                    "Falling back to direct fill — City/State/Zip may need manual entry.",
                    address,
                )
                await page.fill(SEL_ADDRESS_INPUT, address)

        except Exception:
            logger.exception("Failed during address selection for '%s'.", address)
            raise

    # ------------------------------------------------------------------
    # Generic Form Interaction
    # ------------------------------------------------------------------

    async def fill_form_fields(self, data: dict) -> None:
        """Fill form fields dynamically based on the provided data dictionary.

        Iterates through the data dictionary, looks up each key in FIELD_MAP,
        determines the element type (input, select, or checkbox), and interacts
        with the page accordingly.
        """
        page = self.page  # raises if not running

        for key, value in data.items():
            selector = self.FIELD_MAP.get(key)
            if not selector or value is None:
                continue

            try:
                # -------------------------------------------------------------
                # Special Handlers for Custom UI Components
                # -------------------------------------------------------------
                if key == "date_of_birth" and value:
                    val_str = str(value)
                    if len(val_str) == 8 and "/" not in val_str:
                        val_str = f"{val_str[:2]}/{val_str[2:4]}/{val_str[4:]}"
                        
                    parts = val_str.split("/")
                    if len(parts) == 3:
                        month, day, year = parts
                        
                        for part_val, test_id in [(month, 'dateOfBirth_month'), (day, 'dateOfBirth_day'), (year, 'dateOfBirth_year')]:
                            loc = page.locator(f"div[data-test-id='{test_id}']").first
                            # FORCE click to bypass 'pointer-events-none' on the spinbutton divs
                            await loc.click(force=True)
                            await loc.focus()
                            await page.keyboard.type(part_val, delay=50)
                            
                        logger.info("Filled bespoke spinbutton complex for date_of_birth")
                    continue
                if key in ["occupancy_type"]:
                    # Custom clickable cards: "Primary property", etc.
                    card_locator = page.locator(f"text='{value}'").last
                    await card_locator.wait_for(state="visible", timeout=3_000)
                    await card_locator.click()
                    logger.info("Clicked card option: '%s' for '%s'", value, key)
                    continue

                if key == "is_currently_for_sale":
                    val_str = "YES" if value else "NO"
                    radio_loc = page.locator(f"input[name='isListedForSale'][value='{val_str}']").first
                    await radio_loc.check(timeout=3_000, force=True)
                    logger.info("Checked native radio button: '%s' for is_currently_for_sale", val_str)
                    continue


                # -------------------------------------------------------------
                # Standard HTML Elements
                # -------------------------------------------------------------
                locator = page.locator(selector).first
                
                # FIRST: check if it's even on the page right now.
                # We use 'attached' with a short timeout because these checkboxes 
                # are often 'size-0' (not visible to Playwright) but still interactable.
                try:
                    await locator.wait_for(state="attached", timeout=1000)
                except Exception:
                    # Field not on this page, move to next
                    continue

                # 1. Wait for tag info to determine interaction type
                tag_name = await locator.evaluate("el => el.tagName.toLowerCase()")
                type_attr = await locator.evaluate("el => el.type ? el.type.toLowerCase() : ''")

                if tag_name == "select":
                    # Use label=str(value) to match the visible option text exactly
                    # rather than the internal value attribute! Added strict timeout 
                    # to prevent 30-second silent hangs if the option label slightly mismatches.
                    await locator.select_option(label=str(value), timeout=3000)
                elif tag_name == "input" and type_attr == "radio":
                    is_yes = str(value).lower() in ["true", "yes", "1", "on"]
                    target_val = "YES" if is_yes else "NO"
                    radio_loc = page.locator(f"{selector}[value='{target_val}']").first
                    await radio_loc.click(force=True)
                elif tag_name == "input" and type_attr == "checkbox":
                    is_checked = await locator.is_checked()
                    should_be_checked = str(value).lower() in ["true", "yes", "1", "on"]
                    
                    if should_be_checked != is_checked:
                        logger.info(f"Dispatching click event to checkbox: {key} (target state: {should_be_checked})")
                        # Dispatching a click event is more reliable than a click() for 
                        # checkboxes that might have pointer-events-none or nested links 
                        # in their associated labels.
                        await locator.dispatch_event("click")
                        await asyncio.sleep(0.3) # Wait for UI state sync
                else:
                    await locator.fill(str(value))

                logger.info("Filled '%s' with '%s'", key, value)

            except Exception as e:
                logger.warning("Failed to fill field '%s' (selector: %s): %s", key, selector, str(e))

    async def click_continue(self) -> bool:
        """Locate the 'Continue' or 'Check my rate' button, click it, and verify navigation.
        
        Returns:
            True if navigation (URL change) or view transition was detected, False otherwise.
        """
        page = self.page
        old_url = page.url

        logger.info("Clicking continue button...")
        try:
            # Re-locate to ensure freshness
            button_locator = page.locator("button:has-text('Continue'), input[type='submit'][value='Continue'], button:has-text('Check my rate'), button:has-text('Next')").first
            await button_locator.click(timeout=5_000)
            
            # 1. Short wait for immediate URL change
            try:
                await page.wait_for_url(lambda u: u != old_url, timeout=2_000)
                logger.info("Navigation confirmed via immediate URL change: %s -> %s", old_url, page.url)
                return True
            except:
                pass

            # 2. Polling loop with 'Success Indicators' (fields for other pages)
            # This is key! If we don't see a URL change, check if fields from SUCCESSOR stages appear.
            for _ in range(25): # 5 seconds total polling
                await asyncio.sleep(0.2)
                
                # Check 1: Did the URL change?
                if page.url.split('?')[0] != old_url.split('?')[0]:
                    logger.info("Navigation confirmed via URL change: %s -> %s", old_url, page.url)
                    # Short wait for DOM to settle
                    await asyncio.sleep(0.5)
                    return True
                
                # Check 2: Is the button gone? (Most reliable for view transitions)
                if await button_locator.count() == 0 or not await button_locator.is_visible():
                    logger.info("View transition confirmed (Continue button is gone).")
                    await asyncio.sleep(0.5)
                    return True

            # 3. Check for specific UI validation errors (red text)
            err = await page.evaluate("""() => {
                const sel = ['[role="alert"]', '.text-error', '.text-red-500', '.text-red-600', '.errorMessage', '.error-message'];
                for (const s of sel) {
                    const el = document.querySelectorAll(s);
                    // Filter for visible elements with text
                    const visible = Array.from(el).find(e => e.innerText.trim() && e.offsetParent !== null);
                    if (visible) return visible.innerText.trim();
                }
                return null;
            }""")
            if err:
                logger.warning(f"Navigation blocked by UI error: {err}")
                return False
                
            logger.warning("Navigation verification failed: URL and view appear unchanged after 5s polling.")
            return False

        except Exception as e:
            logger.warning("Failed to click continue: %s", str(e))
            return False

    # ------------------------------------------------------------------
    # DOM Verification & Intelligence (Orchestration Glue)
    # ------------------------------------------------------------------

    async def verify_fields(self, data: dict) -> list[str]:
        """Verify that the browser DOM actually reflects the provided data.
        
        Performs a 'DOM Check' by reading the values back from the page.
        Returns a list of field names that do NOT match the expected value.
        """
        page = self.page
        mismatches = []

        for key, expected_value in data.items():
            selector = self.FIELD_MAP.get(key)
            if not selector or expected_value is None:
                continue

            try:
                locator = page.locator(selector).first
                if not await locator.is_visible():
                    # If field isn't visible, we can't verify it right now
                    continue

                actual_value = None
                tag_name = await locator.evaluate("el => el.tagName.toLowerCase()")
                type_attr = await locator.evaluate("el => el.type ? el.type.toLowerCase() : ''")

                if tag_name == "select":
                    # For selects, we check the selected label
                    actual_value = await locator.evaluate("el => el.options[el.selectedIndex].text")
                elif tag_name == "input" and type_attr in ["checkbox", "radio"]:
                    actual_value = await locator.is_checked()
                    # Normalize expected value to bool for comparison
                    expected_value = str(expected_value).lower() in ["true", "yes", "1", "on"]
                else:
                    actual_value = await locator.input_value()

                # Basic normalization for comparison
                def normalize(v, is_numeric=False):
                    if v is None: return ""
                    s = str(v).strip().lower()
                    if is_numeric:
                        # Strip everything that isn't a digit for numbers/phones
                        return re.sub(r"\D", "", s)
                    return s

                # Determine if this is a numeric/phone field for better comparison
                is_num = any(x in key for x in ["income", "phone", "zipcode", "amount", "monthly"])
                
                if normalize(actual_value, is_num) != normalize(expected_value, is_num):
                    logger.warning("DOM Mismatch for '%s': expected='%s', actual='%s'", key, expected_value, actual_value)
                    mismatches.append(key)

            except Exception as e:
                logger.debug("Verification skipped for '%s': %s", key, str(e))
        
        return mismatches

    async def is_page_complete(self) -> bool:
        """Scan the current visible DOM to see if all required fields are filled."""
        page = self.page
        
        # 1. Map URL fragments to fields we expect on those pages for speed
        current_url = page.url
        # Order matters! 'register' must be last because it's the prefix for all other pages.
        url_targets = {
            "for-sale": ["is_currently_for_sale"],
            "annual-income": ["annual_pre_tax_income", "primary_source_of_income"],
            "legal-name": ["first_name", "last_name"],
            "birthdate": ["date_of_birth"],
            "phone-and-email": ["mobile_phone_number", "email_address"],
            # We only require the two main legal checkboxes to proceed
            "consent": ["agreed_to_privacy_policy", "agreed_to_credit_pull", "password", "confirm_password"],
            "register": ["address_line_1", "ownership_type", "occupancy_type"]
        }
        
        target_fields = []
        for pattern, fields in url_targets.items():
            if pattern in current_url:
                target_fields = fields
                break
        
        if not target_fields:
            # Fallback to all fields if URL is unknown
            target_fields = list(self.FIELD_MAP.keys())

        visible_fields_found = 0
        try:
            if await page.locator(".loading-spinner, .progress-bar-animated").count() > 0:
                return False

            for key in target_fields:
                selector = self.FIELD_MAP.get(key)
                if not selector: continue
                # Skip the DOB composite fields which are handled specially
                if key == "date_of_birth":
                    # Check the first spinbutton component for visibility
                    loc = page.locator("div[data-test-id='dateOfBirth_month']").first
                    if await loc.is_visible():
                        visible_fields_found += 1
                        val = await loc.text_content()
                        
                        # CRITICAL: Ignore placeholders! If it says 'Month', 'MM', etc, it's NOT filled.
                        # This prevents 'Double Advance' skipping from the Income page.
                        placeholders = ["Month", "Day", "Year", "MM", "DD", "YYYY"]
                        if not val or val.strip() == "" or any(p in val for p in placeholders):
                             return False
                    continue

                locator = page.locator(selector).first
                if await locator.is_visible():
                    visible_fields_found += 1
                    
                    # Optional fields never block navigation
                    if key in self.OPTIONAL_FIELDS:
                        continue
                        
                    tag_name = await locator.evaluate("el => el.tagName.toLowerCase()")
                    type_attr = await locator.evaluate("el => el.type ? el.type.toLowerCase() : ''")

                    if tag_name == "select":
                        val = await locator.input_value()
                        if not val: return False
                    elif tag_name == "input" and type_attr in ["checkbox", "radio"]:
                        if type_attr == "radio":
                            name = await locator.get_attribute("name")
                            group = page.locator(f"input[name='{name}']")
                            checked_count = await group.evaluate_all("els => els.filter(el => el.checked).length")
                            if checked_count == 0: return False
                        else:
                            if not await locator.is_checked(): return False
                    else:
                        val = await locator.input_value()
                        if not val or val.strip() == "": return False

            # If we found at least one visible field and all were filled, 
            # the page is likely complete.
            return visible_fields_found > 0

        except Exception as e:
            logger.debug("Page completeness check failed: %s", str(e))
            return False

    def get_current_stage(self) -> str:
        """Return a logical stage name based on the current URL.
        
        This allows the system to determine exactly where in the funnel
        the browser is currently located.
        """
        if not self._page: return "unknown"
        url = self._page.url
        
        if "for-sale" in url: return "for-sale"
        if "annual-income" in url: return "income"
        if "legal-name" in url: return "name"
        if "birthdate" in url: return "birthday"
        if "phone-and-email" in url: return "contact"
        if "set-password" in url: return "consent"
        # Check 'register' last as it is a parent path
        if "register" in url and url.endswith("register"): return "property"
        
        return "start"

    async def wait_for_stage(self, stage: str, timeout_ms: int = 15000) -> bool:
        """Wait until the browser reaches the specified logical stage.
        
        Uses a polling loop to verify the URL contains the expected stage identifier.
        """
        logger.info("Waiting for browser to reach stage: %s...", stage)
        start_time = asyncio.get_event_loop().time()
        
        while (asyncio.get_event_loop().time() - start_time) < (timeout_ms / 1000):
            current = self.get_current_stage()
            if current == stage:
                logger.info("Reached stage: %s", stage)
                # Wait a tiny bit for animations to settle
                await asyncio.sleep(0.5)
                return True
            await asyncio.sleep(0.5)
            
        logger.warning("Timed out waiting for stage: %s (current: %s)", stage, self.get_current_stage())
        return False

    async def shutdown(self) -> None:
        """Gracefully close the browser, context, and Playwright process.

        Safe to call multiple times — subsequent calls are no-ops.
        """
        if not self._is_running:
            logger.debug("BrowserManager.shutdown() called but not running — no-op.")
            return

        logger.info("Shutting down browser…")
        await self._teardown()
        logger.info("Browser shutdown complete.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _teardown(self) -> None:
        """Release all Playwright resources in the correct order.

        Called by both ``shutdown()`` and the error path of ``start()``.
        Each step is wrapped individually so a failure in one does not
        prevent the others from being cleaned up.
        """
        # 1. Close the page
        if self._page is not None:
            try:
                await self._page.close()
            except Exception:
                logger.debug("Page close failed (may already be closed).", exc_info=True)
            finally:
                self._page = None

        # 2. Close the context
        if self._context is not None:
            try:
                await self._context.close()
            except Exception:
                logger.debug("Context close failed.", exc_info=True)
            finally:
                self._context = None

        # 3. Close the browser
        if self._browser is not None:
            try:
                await self._browser.close()
            except Exception:
                logger.debug("Browser close failed.", exc_info=True)
            finally:
                self._browser = None

        # 4. Stop Playwright
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception:
                logger.debug("Playwright stop failed.", exc_info=True)
            finally:
                self._playwright = None

        self._is_running = False

    # ------------------------------------------------------------------
    # Context manager support (optional convenience)
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "BrowserManager":
        """Allow ``async with BrowserManager() as mgr:`` usage."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.shutdown()
