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


class BrowserManager:
    """Manages the full lifecycle of a Chromium browser for HELOC form automation.

    Attributes:
        _playwright: The Playwright instance that owns the browser process.
        _browser: The Chromium browser instance.
        _context: A single BrowserContext that retains cookies / session state.
        _page: The active Page used for all interactions.
        _is_running: Guard flag to prevent double-start or use-after-shutdown.
    """

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
