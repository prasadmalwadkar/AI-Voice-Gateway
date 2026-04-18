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
    # ... automation steps ...
    await manager.shutdown()
"""

import logging
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
        "apt_suite": "input[name='apt_suite']",
        "city": "input[name='city']",
        "state": "input[name='state']",
        "zipcode": "input[name='zipcode']",
        "ownership_type": "select[name='ownership']",
        "occupancy_type": "select[name='occupancy_type']",
        "is_currently_for_sale": "input[name='is_currently_for_sale']",
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
        "agreed_to_credit_pull": "input[data-test-id='consent-2']",
    }

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
                            await loc.focus()
                            await page.keyboard.type(part_val, delay=100)
                            
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
                # Wait briefly for exactly this field
                await locator.wait_for(state="attached", timeout=2_000)

                tag_name = await locator.evaluate("el => el.tagName.toLowerCase()")
                type_attr = await locator.evaluate("el => el.type ? el.type.toLowerCase() : ''")

                if tag_name == "select":
                    # Use label=str(value) to match the visible option text exactly
                    # rather than the internal value attribute! Added strict timeout 
                    # to prevent 30-second silent hangs if the option label slightly mismatches.
                    await locator.select_option(label=str(value), timeout=3000)
                elif tag_name == "input" and type_attr in ["checkbox", "radio"]:
                    # Coerce truthy values for checkboxes
                    if str(value).lower() in ["true", "yes", "1", "on"]:
                        await locator.check(force=True)
                    else:
                        await locator.uncheck(force=True)
                else:
                    await locator.fill(str(value))

                logger.info("Filled '%s' with '%s'", key, value)

            except Exception as e:
                logger.warning("Failed to fill field '%s' (selector: %s): %s", key, selector, str(e))

    async def click_continue(self) -> None:
        """Locate the 'Continue' or 'Check my rate' button, click it, and wait for networkidle."""
        page = self.page

        logger.info("Clicking continue button...")
        try:
            button_locator = page.locator("button:has-text('Continue'), input[type='submit'][value='Continue'], button:has-text('Check my rate')").first
            await button_locator.click(timeout=5_000)
            await page.wait_for_load_state("networkidle")
            logger.info("Successfully clicked continue and waited for networkidle.")
        except Exception as e:
            logger.warning("Failed to click continue: %s", str(e))

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
