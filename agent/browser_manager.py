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
- **Demo Mode**: When ``is_demo=True``, the browser launches headed (optional
  ``BROWSER_SLOW_MO_MS`` for slowed replay).
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

import difflib
import logging
import asyncio
import os
import re
import time
from typing import Any, ClassVar, Optional

from playwright.async_api import (
    async_playwright,
    Browser,
    BrowserContext,
    Page,
    Playwright,
)

from agent.schema import HelocApplicationState

logger = logging.getLogger("voip.browser")
_SENSITIVE_FIELDS = frozenset({"password", "confirm_password"})

# The entry-point URL for the HELOC registration form.
HELOC_REGISTER_URL = "https://heloc.westcapitallending.com/account/heloc/register"

# ---------------------------------------------------------------------------
# CSS Selectors — captured from the live West Capital Lending form.
# Centralised here so a future UI change only requires one update.
# Register step: street is the React combobox (type text, listbox, then pick) —
# same control fills or shows selected line; city/state/zip populate after pick.
# ---------------------------------------------------------------------------
SEL_PROPERTY_STREET1 = "input[data-test-id='property-address-street1']"
# WCL may add placeholder; older builds used name= only — OR chain for one locator
SEL_STREET_LINE = (
    f"{SEL_PROPERTY_STREET1}, input[name='address_line_1']"
)
# Typing for autocomplete: same combobox, or placeholder-only (legacy)
SEL_ADDRESS_INPUT = (
    f"{SEL_PROPERTY_STREET1}, input[placeholder='Search for your address']"
)
SEL_AUTOCOMPLETE_LIST = "ul[role='listbox']"
# Skips the "Address not listed" option at the top of the dropdown.
SEL_FIRST_SUGGESTION = "li[role='option']:not([id*='not-listed'])"

# WCL /register/home-address — primary residence combobox (secondary / investment HELOC).
SEL_HOME_ADDRESS_STREET1 = "input[data-test-id='home-address-street1']"
SEL_HOME_STREET_LINE = (
    f"{SEL_HOME_ADDRESS_STREET1}, input[name='address_line_1']"
)
SEL_HOME_ADDRESS_INPUT = (
    f"{SEL_HOME_ADDRESS_STREET1}, input[placeholder='Search for your address']"
)

# WCL /register: occupancy is a radio group name=occupancyType; cards are <label
# class="...card-select"> wrapping <input type="radio" name="occupancyType" value="HOME">…
SEL_OCCUPANCY_RADIOS = 'input[type="radio"][name="occupancyType"]'
# Confirmed: Primary = HOME; other values often SECONDARY + INVESTMENT (verify in prod if form changes)
OCCUPANCY_ENUM_TO_WCL: dict[str, str] = {
    "Primary property": "HOME",
    "Secondary property": "SECONDARY",
    "Investment property": "INVESTMENT",
}


async def _poll_autocomplete_option_count(page: Page) -> int:
    """Wait until listbox is visible and at least one real row has text (API may lag list shell)."""
    try:
        max_wait_ms = int(os.getenv("WCL_ADDRESS_OPTIONS_WAIT_MS", "12000").strip() or "12000")
    except ValueError:
        max_wait_ms = 12000
    max_wait_ms = max(2000, min(max_wait_ms, 45000))
    try:
        poll_sec = float(os.getenv("WCL_ADDRESS_OPTIONS_POLL_SEC", "0.18").strip() or "0.18")
    except ValueError:
        poll_sec = 0.18
    poll_sec = max(0.08, min(poll_sec, 0.75))

    deadline = time.monotonic() + max_wait_ms / 1000.0
    opts = page.locator(SEL_FIRST_SUGGESTION)

    while time.monotonic() < deadline:
        remaining_ms = max(150, int((deadline - time.monotonic()) * 1000))
        try:
            await page.wait_for_selector(
                SEL_AUTOCOMPLETE_LIST,
                state="visible",
                timeout=min(1800, remaining_ms),
            )
        except Exception:
            await asyncio.sleep(poll_sec)
            continue
        try:
            n = await opts.count()
        except Exception:
            n = 0
        if n <= 0:
            await asyncio.sleep(poll_sec)
            continue
        try:
            txt = (await opts.first.text_content(timeout=2500) or "").strip()
        except Exception:
            txt = ""
        if txt:
            return n
        await asyncio.sleep(poll_sec)

    return 0


def _format_full_address_for_autocomplete(state: HelocApplicationState) -> str:
    parts: list[str] = []
    if state.address_line_1:
        parts.append(state.address_line_1.strip())
    if state.city:
        parts.append(state.city.strip())
    if state.state:
        parts.append(state.state.strip())
    if state.zipcode:
        parts.append(state.zipcode.strip())
    return " ".join(parts)


def _street_city_state_key(state: HelocApplicationState) -> str:
    """Stable key for autocomplete dedupe — omits zip so adding zip does not re-run search."""
    parts: list[str] = []
    if state.address_line_1:
        parts.append(state.address_line_1.strip().lower())
    if state.city:
        parts.append(state.city.strip().lower())
    if state.state:
        parts.append(state.state.strip().upper())
    return "|".join(parts)


def _format_full_address_primary_home(state: HelocApplicationState) -> str:
    parts: list[str] = []
    if state.primary_home_address_line_1:
        parts.append(state.primary_home_address_line_1.strip())
    if state.primary_home_city:
        parts.append(state.primary_home_city.strip())
    if state.primary_home_state:
        parts.append(state.primary_home_state.strip())
    if state.primary_home_zipcode:
        parts.append(str(state.primary_home_zipcode).strip())
    return " ".join(parts)


def _street_city_state_key_primary_home(state: HelocApplicationState) -> str:
    parts: list[str] = []
    if state.primary_home_address_line_1:
        parts.append(state.primary_home_address_line_1.strip().lower())
    if state.primary_home_city:
        parts.append(state.primary_home_city.strip().lower())
    if state.primary_home_state:
        parts.append(state.primary_home_state.strip().upper())
    return "|".join(parts)


def _normalize_address_match(s: str) -> str:
    """Uppercase alphanumerics + spaces for fuzzy compare."""
    s = re.sub(r"[^\w\s]", " ", (s or "").upper())
    return re.sub(r"\s+", " ", s).strip()


def _first_street_number(s: str) -> Optional[str]:
    m = re.search(r"\b(\d{1,7})\b", (s or "").strip())
    return m.group(1) if m else None


def _score_address_suggestion(
    suggestion: str,
    typed_full: str,
    ranking_hints: list[str],
) -> float:
    """Higher = better match; combines fuzzy ratio + street-number agreement."""
    sug_n = _normalize_address_match(suggestion)
    if not sug_n:
        return 0.0
    probes: list[str] = [_normalize_address_match(typed_full)]
    for h in ranking_hints:
        h = (h or "").strip()
        if h:
            probes.append(_normalize_address_match(h))
    probes = [p for p in probes if p]
    base = max((difflib.SequenceMatcher(None, sug_n, p).ratio() for p in probes), default=0.0)
    bonus = 0.0
    sn = _first_street_number(suggestion)
    if sn:
        for probe in [typed_full] + ranking_hints:
            if not probe:
                continue
            pn = _first_street_number(str(probe))
            if pn and sn == pn:
                bonus = 0.22
                break
    return min(1.0, base + bonus)


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
        "address_line_1": (
            f"{SEL_PROPERTY_STREET1}, input[name='address_line_1']"
        ),
        "apt_suite": "input[data-test-id='property-address-street2']",
        "city": "input[name='city']",
        "state": "input[name='state']",
        "zipcode": "input[name='zipcode']",
        "ownership_type": "select[name='ownership']",
        "occupancy_type": (
            "input[name='occupancyType'], select[name='occupancy_type']"
        ),
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
        # consent-1: privacy + terms + soft credit (single checkbox).
        # consent-2: electronic communications only.
        "agreed_to_privacy_policy": "input[data-test-id='consent-1']",
        "agreed_to_commerce_communication": "input[data-test-id='consent-2']",
        "primary_home_address_line_1": (
            f"{SEL_HOME_ADDRESS_STREET1}, input[name='address_line_1']"
        ),
        "primary_home_apt_suite": "input[data-test-id='home-address-street2']",
        "primary_home_city": "input[name='city']",
        "primary_home_state": "input[name='state']",
        "primary_home_zipcode": "input[name='zipcode']",
    }

    # Fields that are visible but not required for navigation ("Continue")
    OPTIONAL_FIELDS = [
        "apt_suite",
        "primary_home_apt_suite",
        "other_income",
        "middle_name",
        "suffix",
    ]

    # Ordered URL substring → required fields for that step (first match wins).
    # ``register`` must be last — it appears as a substring on several paths.
    _URL_PATTERN_TO_FIELDS: ClassVar[list[tuple[str, list[str]]]] = [
        ("for-sale", ["is_currently_for_sale"]),
        ("annual-income", ["annual_pre_tax_income", "primary_source_of_income"]),
        ("legal-name", ["first_name", "last_name"]),
        ("birthdate", ["date_of_birth"]),
        ("phone-and-email", ["mobile_phone_number", "email_address"]),
        ("set-password", ["agreed_to_privacy_policy", "agreed_to_commerce_communication", "password", "confirm_password"]),
        ("consent", ["agreed_to_privacy_policy", "agreed_to_commerce_communication", "password", "confirm_password"]),
        # Must match before bare ``register`` (substring) on …/register/home-address.
        ("home-address", [
            "primary_home_address_line_1",
            "primary_home_city",
            "primary_home_state",
            "primary_home_zipcode",
        ]),
        # WCL /register: dropdown address pick fills these inputs in the live browser;
        # we hydrate/verify so Continue enables—user never types them in voice.
        # https://heloc.westcapitallending.com/account/heloc/register
        ("register", [
            "address_line_1", "city", "state", "zipcode",
            "ownership_type", "occupancy_type",
        ]),
    ]

    # Optional fields to hydrate when syncing DOM from state (stage from get_current_stage()).
    _OPTIONAL_HYDRATION_BY_STAGE: ClassVar[dict[str, tuple[str, ...]]] = {
        "property": ("apt_suite",),
        "primary-home": ("primary_home_apt_suite",),
        "for-sale": (),
        "income": ("other_income",),
        "name": ("middle_name", "suffix"),
        "birthday": (),
        "contact": (),
        "consent": (),
    }

    def __init__(self) -> None:
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._is_running: bool = False
        # Dedupe autocomplete on street+city+state only (not zip — site fills zip from suggestion).
        self._last_address_street_city_state_key: Optional[str] = None
        self.address_not_found: bool = False
        self._last_primary_home_street_city_state_key: Optional[str] = None
        self.address_not_found_primary_home: bool = False

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
            is_demo: If ``True``, launches in **headed** mode with a maximized
                viewport. Playwright ``slow_mo`` is **off** by default; set
                ``BROWSER_SLOW_MO_MS`` (e.g. 500) to slow steps for debugging.
                If ``False``, launches headless for server / CI environments.

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
            # Default 0: demo used to force slow_mo=500ms per action (~10s+ on address).
            # Set BROWSER_SLOW_MO_MS=400–500 only when debugging visually.
            try:
                _smo = int(os.getenv("BROWSER_SLOW_MO_MS", "0").strip() or "0")
            except ValueError:
                _smo = 0
            if _smo > 0:
                launch_kwargs["slow_mo"] = _smo
            if is_demo:
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

        Uses ``wait_until="load"`` then waits for the address field — ``networkidle``
        is often very slow on SPAs.

        Raises:
            RuntimeError: If the browser is not running.
        """
        page = self.page  # will raise if not running

        try:
            response = await page.goto(
                HELOC_REGISTER_URL,
                wait_until="load",
                timeout=30_000,
            )
            try:
                await page.wait_for_selector(
                    SEL_STREET_LINE,
                    state="visible",
                    timeout=20_000,
                )
            except Exception as e:
                logger.debug("Address field not visible after load: %s", e)

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

    def _field_keys_for_url(self, current_url: str) -> list[str]:
        """Return required field names for the funnel step implied by ``current_url``."""
        for pattern, fields in self._URL_PATTERN_TO_FIELDS:
            if pattern in current_url:
                return list(fields)
        return list(self.FIELD_MAP.keys())

    def build_hydration_payload(self, state: HelocApplicationState) -> dict[str, Any]:
        """
        Build a fill dict from application state for the **current** URL only.

        Used to sync the live DOM when the browser lags behind conversation (e.g. user
        fixes an earlier page while state already holds later-page values). This turn's
        tool ``updates`` should be merged on top (caller merges; this has no updates).
        """
        if not self._is_running or self._page is None:
            return {}

        keys = self._field_keys_for_url(self._page.url)
        dump = state.model_dump(exclude_none=True)
        out: dict[str, Any] = {}
        for k in keys:
            # Never hydrate email from state alone — only explicit tool sync_updates.
            # Prevents phantom addresses (e.g. LLM guess) typing into the live form.
            if k == "email_address":
                continue
            if k in dump:
                out[k] = dump[k]

        stage = self.get_current_stage()
        for ok in self._OPTIONAL_HYDRATION_BY_STAGE.get(stage, ()):
            if ok in dump and ok not in out:
                out[ok] = dump[ok]

        return out

    _ZIP_FIELD_SELECTORS: ClassVar[tuple[str, ...]] = (
        "input[name='zipcode']",
        "input[name='postalCode']",
        "input[data-test-id*='zip' i]",
        "input[id*='zipcode' i]",
    )

    async def _read_zip_from_dom(self, page: Page) -> str:
        """Zip inputs are sometimes slow or React-controlled; try JS .value + fallbacks."""
        for css in self._ZIP_FIELD_SELECTORS:
            try:
                loc = page.locator(css).first
                if await loc.count() == 0:
                    continue
                raw = (await loc.input_value() or "").strip()
                if not raw:
                    raw = (
                        await loc.evaluate("el => (el && 'value' in el ? String(el.value || '').trim() : '')")
                        or ""
                    ).strip()
                digits = re.sub(r"\D", "", raw)
                if len(digits) >= 5:
                    return digits[:5]
            except Exception as e:
                logger.debug("read zip from %s: %s", css, e)
        return ""

    async def pull_property_fields_from_dom_into_state(self, state: HelocApplicationState) -> list[str]:
        """Copy street/city/state/zip from the property page into state (e.g. after autocomplete)."""
        if self.get_current_stage() != "property":
            return []
        page = self.page
        # Give React a tick to commit zip after autocomplete.
        await asyncio.sleep(0.25)
        changed: list[str] = []
        mapping = [
            ("address_line_1", SEL_STREET_LINE),
            ("city", "input[name='city']"),
            ("state", "input[name='state']"),
        ]
        for field, css in mapping:
            try:
                loc = page.locator(css).first
                if await loc.count() == 0:
                    continue
                raw = (await loc.input_value() or "").strip()
                if not raw:
                    continue
                if field == "state" and len(raw) == 2:
                    raw = raw.upper()
                cur = getattr(state, field, None)
                if cur is not None and str(cur).strip() == str(raw).strip():
                    continue
                setattr(state, field, raw)
                changed.append(field)
            except Exception as e:
                logger.debug("pull_property_fields_from_dom_into_state: %s: %s", field, e)

        z = await self._read_zip_from_dom(page)
        if z:
            cur_z = getattr(state, "zipcode", None)
            if cur_z is None or str(cur_z).strip() != z:
                state.zipcode = z
                changed.append("zipcode")
        if changed:
            logger.info("Synced property fields from DOM into state: %s", changed)
        return changed

    async def sync_property_address_from_state(
        self,
        state: HelocApplicationState,
        *,
        user_updated_address: bool,
        address_ranking_hints: Optional[list[str]] = None,
        autocomplete_query: Optional[str] = None,
    ) -> None:
        if self.get_current_stage() != "property":
            return
        if not state.address_line_1:
            return

        if not user_updated_address:
            await self.pull_property_fields_from_dom_into_state(state)
            return

        full = (autocomplete_query or "").strip() or _format_full_address_for_autocomplete(state)
        if not full.strip():
            return

        dedupe_key = _street_city_state_key(state) + "|t:" + full.strip().lower()
        should_search = bool(dedupe_key) and (
            user_updated_address or dedupe_key != self._last_address_street_city_state_key
        )

        if should_search:
            suggestion = await self.select_address(
                full,
                ranking_hints=address_ranking_hints or [],
            )
            if suggestion == "":
                state.address_line_1 = None
                state.city = None
                state.state = None
                state.zipcode = None
                self._last_address_street_city_state_key = ""
                self.address_not_found = True
                logger.warning("Address not found in autocomplete — cleared state for re-ask.")
                return
            self.address_not_found = False
            self._last_address_street_city_state_key = dedupe_key

        await self.pull_property_fields_from_dom_into_state(state)

    async def pull_primary_home_fields_from_dom_into_state(
        self, state: HelocApplicationState
    ) -> list[str]:
        """Copy primary-home street/city/state/zip from /register/home-address into state."""
        if self.get_current_stage() != "primary-home":
            return []
        page = self.page
        await asyncio.sleep(0.25)
        changed: list[str] = []
        mapping = [
            ("primary_home_address_line_1", SEL_HOME_STREET_LINE),
            ("primary_home_city", "input[name='city']"),
            ("primary_home_state", "input[name='state']"),
        ]
        for field, css in mapping:
            try:
                loc = page.locator(css).first
                if await loc.count() == 0:
                    continue
                raw = (await loc.input_value() or "").strip()
                if not raw:
                    continue
                if field == "primary_home_state" and len(raw) == 2:
                    raw = raw.upper()
                cur = getattr(state, field, None)
                if cur is not None and str(cur).strip() == str(raw).strip():
                    continue
                setattr(state, field, raw)
                changed.append(field)
            except Exception as e:
                logger.debug("pull_primary_home_fields_from_dom_into_state: %s: %s", field, e)

        z = await self._read_zip_from_dom(page)
        if z:
            cur_z = getattr(state, "primary_home_zipcode", None)
            if cur_z is None or str(cur_z).strip() != z:
                state.primary_home_zipcode = z
                changed.append("primary_home_zipcode")
        if changed:
            logger.info("Synced primary home fields from DOM into state: %s", changed)
        return changed

    async def sync_primary_home_address_from_state(
        self,
        state: HelocApplicationState,
        *,
        user_updated_address: bool,
        address_ranking_hints: Optional[list[str]] = None,
        autocomplete_query: Optional[str] = None,
    ) -> None:
        if self.get_current_stage() != "primary-home":
            return
        if not state.primary_home_address_line_1:
            return

        if not user_updated_address:
            await self.pull_primary_home_fields_from_dom_into_state(state)
            return

        full = (autocomplete_query or "").strip() or _format_full_address_primary_home(state)
        if not full.strip():
            return

        dedupe_key = (
            _street_city_state_key_primary_home(state) + "|t:" + full.strip().lower()
        )
        should_search = bool(dedupe_key) and (
            user_updated_address
            or dedupe_key != self._last_primary_home_street_city_state_key
        )

        if should_search:
            suggestion = await self.select_address(
                full,
                ranking_hints=address_ranking_hints or [],
                address_input_selector=SEL_HOME_ADDRESS_INPUT,
            )
            if suggestion == "":
                state.primary_home_address_line_1 = None
                state.primary_home_city = None
                state.primary_home_state = None
                state.primary_home_zipcode = None
                self._last_primary_home_street_city_state_key = ""
                self.address_not_found_primary_home = True
                logger.warning(
                    "Primary home address not found in autocomplete — cleared state for re-ask."
                )
                return
            self.address_not_found_primary_home = False
            self._last_primary_home_street_city_state_key = dedupe_key

        await self.pull_primary_home_fields_from_dom_into_state(state)

    async def _wait_for_city_populated(self, page: Page, timeout_ms: int = 4000) -> None:
        """After picking a listbox row, city/state/zip fill async. Avoid ``networkidle`` (SPA)."""
        steps = max(1, timeout_ms // 50)
        loc = page.locator("input[name='city']").first
        for _ in range(steps):
            try:
                if await loc.count() == 0:
                    await asyncio.sleep(0.05)
                    continue
                c = await loc.input_value()
                if c and str(c).strip():
                    return
            except Exception:
                pass
            await asyncio.sleep(0.05)
        logger.debug("City field still empty after suggestion pick (timeout=%sms).", timeout_ms)

    # ------------------------------------------------------------------
    # Form Interaction — address autocomplete
    # ------------------------------------------------------------------

    async def select_address(
        self,
        address: str,
        *,
        ranking_hints: Optional[list[str]] = None,
        address_input_selector: Optional[str] = None,
    ) -> Optional[str]:
        """Type address, pick best-ranked suggestion (or first if ranking off), wait for fields."""
        page = self.page  # will raise if not running
        input_sel = address_input_selector or SEL_ADDRESS_INPUT

        hints = list(ranking_hints or [])
        use_rank = os.getenv("WCL_ADDRESS_RANK_SUGGESTIONS", "1").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )

        logger.info("Selecting address — typing '%s' into autocomplete field.", address)

        try:
            try:
                _td = int(os.getenv("WCL_ADDRESS_TYPE_DELAY_MS", "35").strip() or "35")
            except ValueError:
                _td = 35
            _td = max(0, min(_td, 150))
            # 1. Clear any previous value and type to trigger autocomplete (lower delay = faster)
            await page.click(input_sel)
            await page.fill(input_sel, "")  # clear existing text
            await page.type(input_sel, address, delay=_td)
            logger.debug("Typed '%s' into address field (delay_ms=%s).", address, _td)

            try:
                try:
                    _after = float(os.getenv("WCL_ADDRESS_AFTER_TYPE_SEC", "0.35").strip() or "0.35")
                except ValueError:
                    _after = 0.35
                await asyncio.sleep(max(0.0, min(_after, 2.5)))

                # Listbox shell can appear before React paints rows — poll until real options exist.
                n = await _poll_autocomplete_option_count(page)
                if n <= 0:
                    raise RuntimeError("no autocomplete rows populated")

                opts = page.locator(SEL_FIRST_SUGGESTION)

                if use_rank:
                    best_i = -1
                    suggestion_text = ""
                    best_score = -1.0
                    for i in range(n):
                        raw_t = (await opts.nth(i).text_content(timeout=2_000) or "").strip()
                        if not raw_t:
                            continue
                        sc = _score_address_suggestion(raw_t, address, hints)
                        if sc > best_score:
                            best_score = sc
                            best_i = i
                            suggestion_text = raw_t
                    if best_i < 0:
                        raise RuntimeError("no scored suggestion rows")
                    pick = opts.nth(best_i)
                    logger.info(
                        "Autocomplete ranked %s options (best score %.3f) — picking '%s'",
                        n,
                        best_score,
                        suggestion_text[:120],
                    )
                    await pick.click(timeout=5_000)
                else:
                    first_option = opts.first
                    suggestion_text = (await first_option.text_content(timeout=5_000) or "").strip()
                    await first_option.click(timeout=5_000)
                    logger.info(
                        "Selected autocomplete suggestion — '%s'",
                        suggestion_text,
                    )

                await self._wait_for_city_populated(page, timeout_ms=4_000)
                await asyncio.sleep(0.1)
                logger.info(
                    "Address fields auto-populated after selecting '%s'.",
                    suggestion_text,
                )
                return suggestion_text

            except Exception as exc:
                logger.warning(
                    "No valid autocomplete suggestion for %r (%s). Clearing field for re-entry.",
                    address,
                    exc,
                )
                await page.fill(input_sel, "")
                return ""

        except Exception:
            logger.exception("Failed during address selection for '%s'.", address)
            raise

    # ------------------------------------------------------------------
    # Generic Form Interaction
    # ------------------------------------------------------------------

    async def _click_occupancy_card(self, page: Page, value: str) -> bool:
        """WCL: card <label> + input[name=occupancyType] value=HOME|…; else <select> or ARIA fallbacks."""
        val = str(value).strip()
        short = re.sub(r"\s+property\s*$", "", val, flags=re.IGNORECASE).strip()

        if await page.locator(SEL_OCCUPANCY_RADIOS).count() > 0:
            wclv = OCCUPANCY_ENUM_TO_WCL.get(val)
            try:
                card = page.locator("label:has(input[name='occupancyType'])").filter(
                    has=page.get_by_text(val, exact=True)
                )
                if await card.count() > 0:
                    c0 = card.first
                    await c0.scroll_into_view_if_needed()
                    await c0.click(timeout=4_000, force=True, no_wait_after=True)
                    logger.info("WCL occupancy: clicked card label (exact) %r", val)
            except Exception as e:
                logger.debug("WCL card label %r: %s", val, e)
            # React may not commit :checked on label only; re-filling other fields can unmount
            # cards if occupancy ran first — fill flow must set occupancy last. Also force <input>.
            await asyncio.sleep(0.2)
            chk = page.locator('input[name="occupancyType"]:checked')
            if wclv and await chk.count() > 0:
                act = (await chk.first.get_attribute("value") or "").strip()
                if act == wclv:
                    return True
            if wclv:
                rloc = page.locator(
                    f'input[type="radio"][name="occupancyType"][value="{wclv}"]'
                )
                if await rloc.count() > 0:
                    try:
                        await rloc.first.check(timeout=3_000, force=True)
                        await asyncio.sleep(0.1)
                    except Exception as e:
                        logger.debug("WCL radio check %r: %s", wclv, e)
                    try:
                        if await page.locator('input[name="occupancyType"]:checked').count() > 0:
                            act2 = (
                                await page.locator('input[name="occupancyType"]:checked')
                                .first.get_attribute("value")
                                or ""
                            ).strip()
                            if wclv and act2 == wclv:
                                logger.info("WCL occupancy: native radio selected value=%r", wclv)
                                return True
                    except Exception as e:
                        logger.debug("WCL post-check: %s", e)
                    try:
                        lab2 = page.locator(
                            f'label:has(input[name="occupancyType"][value="{wclv}"])'
                        )
                        if await lab2.count() > 0:
                            await lab2.first.scroll_into_view_if_needed()
                            await lab2.first.click(
                                timeout=3_000, force=True, no_wait_after=True
                            )
                            await asyncio.sleep(0.15)
                            if await page.locator('input[name="occupancyType"]:checked').count() > 0:
                                logger.info("WCL occupancy: label click for value=%r", wclv)
                                return True
                    except Exception as e:
                        logger.debug("WCL label for value: %s", e)

        # Legacy or alternate: native <select name=occupancy_type>
        vmap_select: dict[str, str] = {
            "Primary property": "PRIMARY",
            "Secondary property": "SECONDARY",
            "Investment property": "INVESTMENT",
        }
        sel = page.locator("select[name='occupancy_type']").first
        if await sel.count() > 0:
            for label in (val, short):
                if not label:
                    continue
                try:
                    await sel.select_option(label=label, timeout=3_000)
                    logger.info("Selected occupancy_type <select> label=%r", label)
                    return True
                except Exception:
                    pass
            code = vmap_select.get(val)
            if code:
                try:
                    await sel.select_option(value=code, timeout=2_000)
                    logger.info("Selected occupancy_type <select> value=%r", code)
                    return True
                except Exception:
                    pass
            try:
                h = await sel.element_handle()
                if h is not None:
                    await h.evaluate(
                        """(el) => {
                        if (el && el.classList) el.classList.remove('sr-only');
                        if (el) el.removeAttribute('hidden');
                    }"""
                    )
            except Exception:
                pass
            for label in (val, short):
                if not label:
                    continue
                try:
                    await sel.select_option(label=label, timeout=2_000)
                    logger.info(
                        "Selected occupancy_type <select> after unhide: %r", label
                    )
                    return True
                except Exception:
                    pass

        if await page.locator(SEL_OCCUPANCY_RADIOS).count() == 0:
            for label in (val, short):
                if not label:
                    continue
                for role in ("button", "radio"):
                    try:
                        b = page.get_by_role(
                            role,  # type: ignore[arg-type]
                            name=label,
                            exact=True,
                        ).first
                        await b.wait_for(state="visible", timeout=2_500)
                        await b.scroll_into_view_if_needed()
                        await b.click(timeout=3_000, force=True)
                        logger.info("Clicked occupancy (role=%s) %r", role, label)
                        return True
                    except Exception:
                        pass

        for label in (val,):
            if not label:
                continue
            try:
                loc = page.get_by_text(label, exact=True)
                n = min(await loc.count(), 12)
                for i in range(n):
                    t = loc.nth(i)
                    if not await t.is_visible():
                        continue
                    await t.scroll_into_view_if_needed()
                    try:
                        await t.click(timeout=1_500)
                    except Exception:
                        hop = t.locator(
                            "xpath=ancestor::button[1] | ancestor::div[@role='button'][1] |"
                            " ancestor::label[1] |"
                            " ancestor::div[contains(concat(' ',@class,' '), ' cursor-pointer ')]"
                        ).first
                        if await hop.count() == 0:
                            continue
                        if not await hop.is_visible():
                            continue
                        await hop.click(timeout=1_500, force=True, no_wait_after=True)
                    logger.info("Clicked occupancy get_by_text %r", label)
                    return True
            except Exception as e:
                logger.debug("occupancy get_by_text %r: %s", label, e)
        if short and short != val and len(short) > 6:
            try:
                loc = page.get_by_text(short, exact=True)
                n = min(await loc.count(), 12)
                for i in range(1, n):
                    t = loc.nth(i)
                    if not await t.is_visible():
                        continue
                    await t.scroll_into_view_if_needed()
                    try:
                        await t.click(timeout=1_500)
                    except Exception:
                        continue
                    logger.info("Clicked occupancy get_by_text (short) %r", short)
                    return True
            except Exception as e:
                logger.debug("occupancy get_by_text short %r: %s", short, e)
        return False

    async def fill_form_fields(self, data: dict) -> None:
        """Fill form fields dynamically based on the provided data dictionary.

        Iterates through the data dictionary, looks up each key in FIELD_MAP,
        determines the element type (input, select, or checkbox), and interacts
        with the page accordingly.
        """
        page = self.page  # raises if not running

        # WCL: filling address/ownership after occupancy clears the card radio in React.
        on_property = self.get_current_stage() == "property"
        on_primary_home = self.get_current_stage() == "primary-home"
        occ_value = data.get("occupancy_type") if on_property else None

        for key, value in data.items():
            if on_property and key == "occupancy_type":
                continue
            if on_property and key in ("city", "state", "zipcode"):
                # Filled by WCL after address dropdown pick; never overwrite from hydration.
                continue
            if on_primary_home and key in (
                "primary_home_city",
                "primary_home_state",
                "primary_home_zipcode",
            ):
                continue
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
                if key == "address_line_1" and self.get_current_stage() == "property":
                    if await locator.count() > 0:
                        try:
                            cur = (await locator.input_value() or "").strip()
                        except Exception:
                            cur = ""
                        if (
                            len(cur) >= 28
                            and re.search(r"\b\d{5}\b", cur)
                            and ("," in cur)
                        ):
                            # Site already has the resolved autocomplete line; avoid clobber
                            # with a shorter Pydantic street and DOM mismatch on verify.
                            logger.info(
                                "Skip fill for address_line_1 (combobox already has resolved line)."
                            )
                            continue
                if key == "primary_home_address_line_1" and on_primary_home:
                    if await locator.count() > 0:
                        try:
                            cur = (await locator.input_value() or "").strip()
                        except Exception:
                            cur = ""
                        if (
                            len(cur) >= 28
                            and re.search(r"\b\d{5}\b", cur)
                            and ("," in cur)
                        ):
                            logger.info(
                                "Skip fill for primary_home_address_line_1 "
                                "(combobox already has resolved line)."
                            )
                            continue
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

                safe_value = "***REDACTED***" if key in _SENSITIVE_FIELDS else value
                logger.info("Filled '%s' with '%s'", key, safe_value)

            except Exception as e:
                logger.warning("Failed to fill field '%s' (selector: %s): %s", key, selector, str(e))

        if occ_value is not None:
            try:
                await asyncio.sleep(0.35)
                if await self._click_occupancy_card(page, str(occ_value)):
                    logger.info("Set occupancy_type: %r", occ_value)
            except Exception as e:
                logger.warning("Failed to set occupancy_type: %s", e)

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
            await button_locator.wait_for(state="visible", timeout=5_000)
            # React often disables Continue until the last required field is valid; wait
            # briefly for enablement to avoid a no-op click and silent stuck state.
            for _ in range(30):
                try:
                    disabled = await button_locator.get_attribute("disabled")
                    aria_dis = await button_locator.get_attribute("aria-disabled")
                except Exception:
                    disabled, aria_dis = None, None
                if (
                    disabled in (None, "false")
                    and (aria_dis is None or str(aria_dis).lower() not in ("true", "1"))
                ):
                    break
                await asyncio.sleep(0.15)
            else:
                logger.warning("Continue button still disabled after wait — clicking anyway (force)")

            await button_locator.click(timeout=5_000, force=True)
            
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
            if (
                key == "occupancy_type"
                and expected_value is not None
                and self.get_current_stage() == "property"
            ):
                wexp = OCCUPANCY_ENUM_TO_WCL.get(str(expected_value))
                chk = page.locator('input[name="occupancyType"]:checked')
                if await chk.count() > 0:
                    if wexp:
                        wact = (await chk.first.get_attribute("value") or "").strip()
                        if wact != wexp:
                            logger.warning(
                                "DOM Mismatch for 'occupancy_type': expected WCL value %r, actual %r",
                                wexp,
                                wact,
                            )
                            mismatches.append("occupancy_type")
                    continue
                if await self._occupancy_field_filled(page):
                    continue
                mismatches.append("occupancy_type")
                continue
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
                is_num = any(
                    x in key for x in ["income", "phone", "zipcode", "amount", "monthly"]
                ) or "primary_home_zip" in key

                if (
                    key == "address_line_1"
                    and self.get_current_stage() == "property"
                    and actual_value is not None
                    and expected_value is not None
                ):
                    a = re.sub(r"\W+", "", str(actual_value).lower())[:200]
                    e = re.sub(r"\W+", "", str(expected_value).lower())[:200]
                    if e in a or a in e or (len(e) > 4 and e[:6] in a):
                        continue

                if (
                    key == "primary_home_address_line_1"
                    and self.get_current_stage() == "primary-home"
                    and actual_value is not None
                    and expected_value is not None
                ):
                    a = re.sub(r"\W+", "", str(actual_value).lower())[:200]
                    e = re.sub(r"\W+", "", str(expected_value).lower())[:200]
                    if e in a or a in e or (len(e) > 4 and e[:6] in a):
                        continue

                if normalize(actual_value, is_num) != normalize(expected_value, is_num):
                    safe_expected = "***REDACTED***" if key in _SENSITIVE_FIELDS else expected_value
                    safe_actual = "***REDACTED***" if key in _SENSITIVE_FIELDS else actual_value
                    logger.warning(
                        "DOM Mismatch for '%s': expected='%s', actual='%s'",
                        key,
                        safe_expected,
                        safe_actual,
                    )
                    mismatches.append(key)

            except Exception as e:
                logger.debug("Verification skipped for '%s': %s", key, str(e))
        
        return mismatches

    async def _field_selected_in_cards_or_select(
        self, page: Page, select_name: str, card_labels: tuple[str, ...]
    ) -> bool:
        """WCL: native select can be hidden; also detect selection via same card text as fill."""
        try:
            sel = page.locator(f"select[name='{select_name}']").first
            if await sel.count() > 0:
                val = (await sel.input_value() or "").strip()
                if val:
                    return True
            # Use role=button to avoid false positives from headings / footers that
            # contain the same string as a card (e.g. "Primary property").
            for label in card_labels:
                loc = page.get_by_role("button", name=label, exact=True)
                n = await loc.count()
                for i in range(n):
                    try:
                        btn = loc.nth(i)
                        if not await btn.is_visible():
                            continue
                    except Exception:
                        continue
                    picked = await btn.evaluate(
                        """el => {
                        if (el.getAttribute('aria-pressed') === 'true' ||
                            el.getAttribute('aria-selected') === 'true') return true;
                        const ds = el.getAttribute && el.getAttribute('data-state');
                        if (ds === 'checked' || ds === 'active' || ds === 'on') return true;
                        if (el.classList && [...el.classList].some(c => c.includes('ring-') || c.includes('border-primary')))
                          return true;
                        return false;
                    }"""
                    )
                    if picked:
                        return True
            return False
        except Exception as e:
            logger.debug("card/select completeness check failed (%s): %s", select_name, e)
            return False

    async def _occupancy_field_filled(self, page: Page) -> bool:
        """WCL: radio name=occupancyType; or hidden select; or ARIA card buttons."""
        r = page.locator('input[name="occupancyType"]:checked')
        if await r.count() > 0 and (await r.first.get_attribute("value") or "").strip():
            return True
        return await self._field_selected_in_cards_or_select(
            page, "occupancy_type", ("Primary property", "Secondary property", "Investment property")
        )

    async def _wcl_register_street_satisfied(self, page: Page) -> bool:
        """WCL /register: street = combobox (data-test-id=property-address-street1) or
        hidden named input; after pick, value may be in the combobox. Else infer from
        city+state+zip5 filled by the same selection.
        """
        line = page.locator(SEL_STREET_LINE).first
        if await line.count() > 0 and (await line.input_value() or "").strip():
            return True
        search = page.locator(SEL_ADDRESS_INPUT).first
        if await search.count() > 0 and len(
            (await search.input_value() or "").strip()
        ) >= 4:
            return True
        c = page.locator("input[name='city']").first
        st = page.locator("input[name='state']").first
        if await c.count() == 0 or await st.count() == 0:
            return False
        cval = (await c.input_value() or "").strip()
        stval = (await st.input_value() or "").strip()
        zd = re.sub(r"\D", "", await self._read_zip_from_dom(page) or "")
        return bool(cval and stval and len(zd) >= 5)

    async def _wcl_primary_home_street_satisfied(self, page: Page) -> bool:
        """WCL /register/home-address: same completion pattern as property street combobox."""
        line = page.locator(SEL_HOME_STREET_LINE).first
        if await line.count() > 0 and (await line.input_value() or "").strip():
            return True
        search = page.locator(SEL_HOME_ADDRESS_INPUT).first
        if await search.count() > 0 and len(
            (await search.input_value() or "").strip()
        ) >= 4:
            return True
        c = page.locator("input[name='city']").first
        st = page.locator("input[name='state']").first
        if await c.count() == 0 or await st.count() == 0:
            return False
        cval = (await c.input_value() or "").strip()
        stval = (await st.input_value() or "").strip()
        zd = re.sub(r"\D", "", await self._read_zip_from_dom(page) or "")
        return bool(cval and stval and len(zd) >= 5)

    async def is_page_complete(self) -> bool:
        """True only when every required field for this URL is filled (including card-based occupancy)."""
        page = self.page

        current_url = page.url
        target_fields = self._field_keys_for_url(current_url)
        if not target_fields:
            return False

        try:
            if await page.locator(".loading-spinner, .progress-bar-animated").count() > 0:
                return False

            for key in target_fields:
                if key in self.OPTIONAL_FIELDS:
                    continue

                if key == "address_line_1":
                    if self.get_current_stage() == "property":
                        if not await self._wcl_register_street_satisfied(page):
                            return False
                    else:
                        loc = page.locator(SEL_STREET_LINE).first
                        if await loc.count() == 0:
                            return False
                        if not (await loc.input_value() or "").strip():
                            return False
                    continue

                if key == "primary_home_address_line_1":
                    if self.get_current_stage() == "primary-home":
                        if not await self._wcl_primary_home_street_satisfied(page):
                            return False
                    else:
                        loc = page.locator(SEL_HOME_STREET_LINE).first
                        if await loc.count() == 0:
                            return False
                        if not (await loc.input_value() or "").strip():
                            return False
                    continue

                if key in ("city", "state", "primary_home_city", "primary_home_state"):
                    dom_name = (
                        "city"
                        if key in ("city", "primary_home_city")
                        else "state"
                    )
                    loc = page.locator(f"input[name='{dom_name}']").first
                    if await loc.count() == 0:
                        logger.debug("is_page_complete: no input for %s", key)
                        return False
                    val = (await loc.input_value() or "").strip()
                    if not val:
                        logger.debug("is_page_complete: empty value for %s", key)
                        return False
                    continue

                if key in ("zipcode", "primary_home_zipcode"):
                    z = await self._read_zip_from_dom(page)
                    digits = re.sub(r"\D", "", z or "")
                    if len(digits) >= 5:
                        continue
                    logger.debug("is_page_complete: missing or short zip in DOM: %r", z)
                    return False

                if key == "date_of_birth":
                    loc = page.locator("div[data-test-id='dateOfBirth_month']").first
                    if not await loc.is_visible():
                        return False
                    val = await loc.text_content()
                    placeholders = ["Month", "Day", "Year", "MM", "DD", "YYYY"]
                    if not val or val.strip() == "" or any(p in val for p in placeholders):
                        return False
                    continue

                if key == "ownership_type":
                    if not await self._field_selected_in_cards_or_select(
                        page, "ownership", ("Sole owner", "Joint owner")
                    ):
                        return False
                    continue

                if key == "occupancy_type":
                    if not await self._occupancy_field_filled(page):
                        return False
                    continue

                selector = self.FIELD_MAP.get(key)
                if not selector:
                    continue

                locator = page.locator(selector).first
                if await locator.count() == 0 or not await locator.is_visible():
                    return False

                tag_name = await locator.evaluate("el => el.tagName.toLowerCase()")
                type_attr = await locator.evaluate("el => el.type ? el.type.toLowerCase() : ''")

                if tag_name == "select":
                    val = await locator.input_value()
                    if not val or not str(val).strip():
                        return False
                elif tag_name == "input" and type_attr in ["checkbox", "radio"]:
                    if type_attr == "radio":
                        name = await locator.get_attribute("name")
                        group = page.locator(f"input[name='{name}']")
                        checked_count = await group.evaluate_all(
                            "els => els.filter(el => el.checked).length"
                        )
                        if checked_count == 0:
                            return False
                    else:
                        if not await locator.is_checked():
                            return False
                else:
                    val = await locator.input_value()
                    if not val or str(val).strip() == "":
                        return False

            return True

        except Exception as e:
            logger.debug("Page completeness check failed: %s", str(e))
            return False

    async def path2_debug_register_snapshot(self) -> dict[str, Any]:
        """For PATH2_DEBUG_DETAILED: one dict of what the WCL register form looks like in Playwright (no PII in keys)."""
        if not self._is_running or self._page is None:
            return {"ok": False, "reason": "browser not running"}
        page = self._page
        out: dict[str, Any] = {
            "url": page.url,
            "stage": self.get_current_stage(),
        }
        st = self.get_current_stage()
        if st not in ("property", "primary-home"):
            out["note"] = "not on property or primary-home register step"
            return out
        try:
            out["loading_spinnerish"] = await page.locator(
                ".loading-spinner, .progress-bar-animated"
            ).count()
            if st == "primary-home":
                for name in (
                    "primary_home_address_line_1",
                    "primary_home_city",
                    "primary_home_state",
                    "primary_home_zipcode",
                ):
                    if name == "primary_home_zipcode":
                        out["zip_raw"] = await self._read_zip_from_dom(page)
                        z = re.sub(r"\D", "", out.get("zip_raw") or "")
                        out["zip_digits"] = len(z)
                        continue
                    if name == "primary_home_address_line_1":
                        loc = page.locator(SEL_HOME_STREET_LINE).first
                    else:
                        dom = "city" if name == "primary_home_city" else "state"
                        loc = page.locator(f"input[name='{dom}']").first
                    c = await loc.count()
                    if c:
                        v = (await loc.input_value() or "").strip()
                        out[name] = {"count": 1, "len": len(v), "set": bool(v)}
                    else:
                        out[name] = {"count": 0, "set": False}
                out["wcl_primary_home_street_satisfied"] = (
                    await self._wcl_primary_home_street_satisfied(page)
                )
                out["is_page_complete"] = await self.is_page_complete()
                return out

            for name in ("address_line_1", "city", "state", "zipcode"):
                if name == "zipcode":
                    out["zip_raw"] = await self._read_zip_from_dom(page)
                    z = re.sub(r"\D", "", out.get("zip_raw") or "")
                    out["zip_digits"] = len(z)
                    continue
                if name == "address_line_1":
                    loc = page.locator(SEL_STREET_LINE).first
                else:
                    loc = page.locator(f"input[name='{name}']").first
                c = await loc.count()
                if c:
                    v = (await loc.input_value() or "").strip()
                    out[name] = {"count": 1, "len": len(v), "set": bool(v)}
                else:
                    out[name] = {"count": 0, "set": False}
            own = page.locator("select[name='ownership']").first
            if await own.count():
                ovv = (await own.input_value() or "").strip()
                out["ownership"] = {"select_value_len": len(ovv), "set": bool(ovv)}
            else:
                out["ownership"] = {"select_present": False}
            ocr = page.locator('input[name="occupancyType"]:checked').first
            if await ocr.count() > 0:
                w = (await ocr.get_attribute("value") or "").strip()
                out["occupancy_radio"] = {
                    "value": w,
                    "set": bool(w),
                }
            else:
                out["occupancy_radio"] = {"set": False}
            occ = page.locator("select[name='occupancy_type']").first
            if await occ.count():
                ocv = (await occ.input_value() or "").strip()
                out["occupancy_select"] = {
                    "select_value_len": len(ocv),
                    "set": bool(ocv),
                }
            else:
                out["occupancy_select"] = {"select_present": False}
            out["occupancy_cards_satisfied"] = await self._occupancy_field_filled(page)
            out["ownership_cards_satisfied"] = await self._field_selected_in_cards_or_select(
                page, "ownership", ("Sole owner", "Joint owner")
            )
            out["wcl_street_satisfied"] = await self._wcl_register_street_satisfied(page)
            out["is_page_complete"] = await self.is_page_complete()
        except Exception as e:
            out["error"] = str(e)[:200]
        return out

    def get_current_stage(self) -> str:
        """Return a logical stage name based on the current URL.
        
        This allows the system to determine exactly where in the funnel
        the browser is currently located.
        """
        if not self._page: return "unknown"
        url = self._page.url
        
        if "for-sale" in url: return "for-sale"
        if "home-address" in url: return "primary-home"
        if "annual-income" in url: return "income"
        if "legal-name" in url: return "name"
        if "birthdate" in url: return "birthday"
        if "phone-and-email" in url: return "contact"
        if "set-password" in url: return "consent"
        # Check 'register' last as it is a parent path (handles /register and /register/)
        path = url.split("?")[0].rstrip("/")
        if "register" in url and path.endswith("register"):
            return "property"

        return "start"

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
        self._last_address_street_city_state_key = None
        self.address_not_found = False
        self._last_primary_home_street_city_state_key = None
        self.address_not_found_primary_home = False
