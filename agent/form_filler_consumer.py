"""
Form Filler Consumer — Standalone Queue Consumer Process
==========================================================

Picks up validated HELOC intake payloads from the queue and fills the
West Capital Lending form using the existing BrowserManager.

Architecture:
    - Runs as a **separate process** from the voice server
    - Polls the queue for pending payloads
    - Launches a Playwright browser per payload
    - Fills the form, navigates through all pages
    - ACKs on success, retries up to N times, then DLQ
    - Sends status callbacks (webhook) on completion/failure

Usage::

    # As a standalone process:
    python -m agent.form_filler_consumer

    # With env configuration:
    INTAKE_QUEUE_BACKEND=filesystem python -m agent.form_filler_consumer

Requirements for separate process (production):
    - Same Python environment as the main server (shared deps)
    - Access to the same queue directory (filesystem) or Redis instance
    - Playwright installed: ``playwright install chromium``
    - Can run on a different machine if using Redis Streams
    - No Twilio/Cartesia/LLM deps needed — only Playwright + queue
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Optional

# Ensure project root is on path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.browser_manager import BrowserManager
from agent.queue_publisher import (
    FilesystemQueue,
    IntakePayload,
    IntakeQueue,
    get_queue,
    send_status_callback,
)
from agent.schema import HelocApplicationState

# Configuration
POLL_INTERVAL_SEC = float(os.getenv("CONSUMER_POLL_INTERVAL_SEC", "5.0"))
HEADLESS = os.getenv("CONSUMER_BROWSER_HEADLESS", "1").strip().lower() in (
    "1", "true", "yes",
)
AUTO_PASSWORD = os.getenv("CONSUMER_AUTO_PASSWORD", "WCL!2026!Agent")
HEALTH_PORT = int(os.getenv("CONSUMER_HEALTH_PORT", "8001"))
STALE_THRESHOLD_SEC = int(os.getenv("CONSUMER_STALE_THRESHOLD_SEC", "1800"))
JSON_LOGS = os.getenv("CONSUMER_JSON_LOGS", "1").strip().lower() in ("1", "true", "yes")


# =============================================================================
# JSON Logging
# =============================================================================

class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        })


def _configure_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    if JSON_LOGS:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        ))
    root.handlers = [handler]


logger = logging.getLogger("voip.consumer")


# =============================================================================
# Health Check HTTP Server
# =============================================================================

_health_state: dict = {
    "last_processed_at": None,
    "processed_count": 0,
}


def _make_health_handler():
    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/health":
                self.send_response(404)
                self.end_headers()
                return
            body = json.dumps({
                "status": "ok",
                "last_processed_at": _health_state["last_processed_at"],
                "processed_count": _health_state["processed_count"],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass  # suppress access logs

    return HealthHandler


def _start_health_server() -> None:
    server = HTTPServer(("0.0.0.0", HEALTH_PORT), _make_health_handler())
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("Health check listening | port=%d path=/health", HEALTH_PORT)


# =============================================================================
# Consumer
# =============================================================================

class FormFillerConsumer:
    """
    Continuously polls the intake queue and processes payloads.

    Lifecycle per payload:
    1. consume_one() → get next pending payload
    2. Hydrate HelocApplicationState from payload
    3. Launch BrowserManager, navigate to WCL register
    4. Fill form fields page by page
    5. On success: ACK + status callback
    6. On failure: NACK (retry or DLQ) + status callback
    """

    def __init__(self, queue: Optional[IntakeQueue] = None) -> None:
        self.queue = queue or get_queue()
        self._running = False
        self._current_browser: Optional[BrowserManager] = None

    async def run(self) -> None:
        """Main consumer loop — poll queue, process payloads."""
        self._running = True
        logger.info(
            "Form filler consumer started | poll_interval=%.1fs headless=%s",
            POLL_INTERVAL_SEC,
            HEADLESS,
        )

        await self._recover_stale_processing()

        while self._running:
            try:
                payload = await self.queue.consume_one()
                if payload is None:
                    await asyncio.sleep(POLL_INTERVAL_SEC)
                    continue

                logger.info(
                    "Processing payload | message_id=%s call_sid=%s attempt=%d/%d",
                    payload.message_id,
                    payload.call_sid,
                    payload.attempt,
                    payload.max_retries,
                )

                try:
                    await self._fill_form(payload)
                    await self.queue.ack(payload.message_id)
                    payload.status = "completed"
                    payload.completed_at = datetime.now(timezone.utc).isoformat()
                    _health_state["last_processed_at"] = payload.completed_at
                    _health_state["processed_count"] += 1
                    logger.info(
                        "Form filled successfully | message_id=%s call_sid=%s",
                        payload.message_id,
                        payload.call_sid,
                    )
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {str(e)[:500]}"
                    await self.queue.nack(payload.message_id, error_msg)
                    payload.status = "failed"
                    logger.error(
                        "Form fill failed | message_id=%s attempt=%d error=%s",
                        payload.message_id,
                        payload.attempt,
                        error_msg,
                        exc_info=True,
                    )

                # Send status callback regardless of success/failure
                try:
                    await send_status_callback(payload)
                except Exception as cb_err:
                    logger.warning("Status callback error: %s", cb_err)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Consumer loop error: %s", e, exc_info=True)
                await asyncio.sleep(POLL_INTERVAL_SEC)

        logger.info("Form filler consumer stopped.")

    async def stop(self) -> None:
        """Graceful async shutdown — waits for in-flight browser job to finish."""
        self._running = False
        if self._current_browser and self._current_browser.is_running:
            try:
                await self._current_browser.shutdown()
            except Exception:
                pass

    def stop_sync(self) -> None:
        """Thread-safe stop — schedules stop() on the running event loop."""
        self._running = False

    async def _recover_stale_processing(self) -> None:
        """On startup, move processing/ files older than STALE_THRESHOLD_SEC back to pending/."""
        if not isinstance(self.queue, FilesystemQueue):
            return
        now = time.time()
        recovered = 0
        for path in self.queue.processing_dir.glob("*.json"):
            try:
                if now - path.stat().st_mtime > STALE_THRESHOLD_SEC:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    data.setdefault("errors", []).append(
                        f"recovered from stale processing at {datetime.now(timezone.utc).isoformat()}"
                    )
                    dest = self.queue.pending_dir / path.name
                    dest.write_text(json.dumps(data, indent=2), encoding="utf-8")
                    path.unlink()
                    recovered += 1
                    logger.info("Recovered stale payload | file=%s", path.name)
            except Exception as e:
                logger.warning("Failed to recover stale file %s: %s", path.name, e)
        if recovered:
            logger.info("Stale recovery complete | count=%d", recovered)

    async def _fill_form(self, payload: IntakePayload) -> None:
        """
        Fill the WCL HELOC form using BrowserManager.

        Steps:
        1. Hydrate HelocApplicationState from payload
        2. Start browser → navigate to register page
        3. Fill property page → click Continue
        4. Fill primary home (if needed) → click Continue
        5. Fill for-sale → click Continue
        6. Fill income → click Continue
        7. Fill name → click Continue
        8. Fill DOB → click Continue
        9. Fill contact → click Continue
        10. Fill consent + password → submit
        """
        state = self._hydrate_state(payload)
        browser = BrowserManager()
        self._current_browser = browser

        try:
            await browser.start(is_demo=not HEADLESS)
            await browser.navigate_to_start()

            await self._fill_property_page(browser, state)

            if state.occupancy_type in ("Secondary property", "Investment property"):
                await self._fill_primary_home_page(browser, state)

            await self._fill_for_sale_page(browser, state)
            await self._fill_income_page(browser, state)
            await self._fill_name_page(browser, state)
            await self._fill_birthday_page(browser, state)
            await self._fill_contact_page(browser, state)
            await self._fill_consent_page(browser, state)

            logger.info("All pages filled successfully | call_sid=%s", payload.call_sid)

        finally:
            self._current_browser = None
            if browser.is_running:
                await browser.shutdown()

    def _hydrate_state(self, payload: IntakePayload) -> HelocApplicationState:
        """Build HelocApplicationState from payload data."""
        state = HelocApplicationState()
        for key, value in payload.application_data.items():
            if hasattr(state, key) and value is not None:
                try:
                    setattr(state, key, value)
                except Exception as e:
                    logger.warning("Failed to set %s=%r: %s", key, value, e)
        return state

    async def _fill_and_navigate(
        self,
        browser: BrowserManager,
        state: HelocApplicationState,
        fields: dict,
        page_name: str,
    ) -> None:
        """Fill form fields and click Continue."""
        hydration = browser.build_hydration_payload(state)
        merged = {**hydration, **fields}

        await browser.fill_form_fields(merged)

        # React forms need time to register events before is_page_complete() reads DOM.
        # Retry up to 4 times with increasing wait so transient hydration delays don't DLQ.
        _WAIT_SECS = (1.0, 2.0, 3.0, 4.0)
        complete = False
        for attempt_n, wait in enumerate(_WAIT_SECS, 1):
            await asyncio.sleep(wait)
            mismatches = await browser.verify_fields(merged)
            if mismatches:
                logger.warning(
                    "[%s] attempt %d/%d — field mismatches: %s",
                    page_name, attempt_n, len(_WAIT_SECS), mismatches,
                )
                await browser.fill_form_fields({k: merged[k] for k in mismatches if k in merged})
            complete = await browser.is_page_complete()
            if complete:
                break
            logger.info(
                "[%s] attempt %d/%d — is_page_complete=False, retrying…",
                page_name, attempt_n, len(_WAIT_SECS),
            )

        if complete:
            nav_ok = await browser.click_continue()
            if nav_ok:
                logger.info("[%s] Page complete — navigated.", page_name)
                await asyncio.sleep(1.5)
            else:
                raise RuntimeError(
                    f"[{page_name}] click_continue() returned False — page may not have advanced."
                )
        else:
            raise RuntimeError(
                f"[{page_name}] is_page_complete() returned False after {len(_WAIT_SECS)} retries."
            )

    async def _fill_property_page(
        self, browser: BrowserManager, state: HelocApplicationState
    ) -> None:
        if state.address_line_1:
            await browser.sync_property_address_from_state(
                state,
                user_updated_address=True,
                address_ranking_hints=[state.address_line_1],
            )
            await browser.pull_property_fields_from_dom_into_state(state)

        fields = {}
        if state.ownership_type:
            fields["ownership_type"] = state.ownership_type
        if state.occupancy_type:
            fields["occupancy_type"] = state.occupancy_type
        if state.apt_suite:
            fields["apt_suite"] = state.apt_suite

        await self._fill_and_navigate(browser, state, fields, "property")

    async def _fill_primary_home_page(
        self, browser: BrowserManager, state: HelocApplicationState
    ) -> None:
        if state.primary_home_address_line_1:
            await browser.sync_primary_home_address_from_state(
                state,
                user_updated_address=True,
                address_ranking_hints=[state.primary_home_address_line_1],
            )
            await browser.pull_primary_home_fields_from_dom_into_state(state)

        fields = {}
        if state.primary_home_apt_suite:
            fields["primary_home_apt_suite"] = state.primary_home_apt_suite

        await self._fill_and_navigate(browser, state, fields, "primary-home")

    async def _fill_for_sale_page(
        self, browser: BrowserManager, state: HelocApplicationState
    ) -> None:
        fields = {}
        if state.is_currently_for_sale is not None:
            fields["is_currently_for_sale"] = state.is_currently_for_sale
        await self._fill_and_navigate(browser, state, fields, "for-sale")

    async def _fill_income_page(
        self, browser: BrowserManager, state: HelocApplicationState
    ) -> None:
        fields = {}
        if state.annual_pre_tax_income is not None:
            fields["annual_pre_tax_income"] = state.annual_pre_tax_income
        if state.other_income is not None:
            fields["other_income"] = state.other_income
        if state.primary_source_of_income:
            fields["primary_source_of_income"] = state.primary_source_of_income
        await self._fill_and_navigate(browser, state, fields, "income")

    async def _fill_name_page(
        self, browser: BrowserManager, state: HelocApplicationState
    ) -> None:
        fields = {}
        if state.first_name:
            fields["first_name"] = state.first_name
        if state.middle_name:
            fields["middle_name"] = state.middle_name
        if state.last_name:
            fields["last_name"] = state.last_name
        if state.suffix:
            fields["suffix"] = state.suffix
        await self._fill_and_navigate(browser, state, fields, "name")

    async def _fill_birthday_page(
        self, browser: BrowserManager, state: HelocApplicationState
    ) -> None:
        fields = {}
        if state.date_of_birth:
            fields["date_of_birth"] = state.date_of_birth
        await self._fill_and_navigate(browser, state, fields, "birthday")

    async def _fill_contact_page(
        self, browser: BrowserManager, state: HelocApplicationState
    ) -> None:
        fields = {}
        if state.mobile_phone_number:
            fields["mobile_phone_number"] = state.mobile_phone_number
        if state.email_address:
            fields["email_address"] = state.email_address
        await self._fill_and_navigate(browser, state, fields, "contact")

    async def _fill_consent_page(
        self, browser: BrowserManager, state: HelocApplicationState
    ) -> None:
        state.password = AUTO_PASSWORD
        state.confirm_password = AUTO_PASSWORD

        fields = {
            "password": AUTO_PASSWORD,
            "confirm_password": AUTO_PASSWORD,
        }
        if state.agreed_to_privacy_policy:
            fields["agreed_to_privacy_policy"] = True
        if state.agreed_to_commerce_communication:
            fields["agreed_to_commerce_communication"] = True

        # Retry consent in place (same browser, same page) up to 3 extra times before
        # raising — avoids NACKing the whole payload and restarting all 8 form pages
        # just because of a transient checkbox render delay on the last page.
        _CONSENT_PAGE_RETRIES = 3
        last_err: Exception = RuntimeError("[consent] unreachable")
        for attempt in range(_CONSENT_PAGE_RETRIES):
            try:
                await self._fill_and_navigate(browser, state, fields, "consent")
                return
            except RuntimeError as e:
                last_err = e
                if attempt < _CONSENT_PAGE_RETRIES - 1:
                    logger.warning(
                        "[consent] page-level retry %d/%d: %s",
                        attempt + 1,
                        _CONSENT_PAGE_RETRIES,
                        e,
                    )
                    await asyncio.sleep(1.0)
        raise last_err


# =============================================================================
# CLI entry point
# =============================================================================

def main() -> None:
    _configure_logging()
    _start_health_server()

    consumer = FormFillerConsumer()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _on_signal(sig: int, _frame) -> None:
        logger.info("Signal %d received — requesting graceful shutdown...", sig)
        loop.call_soon_threadsafe(consumer.stop_sync)

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        loop.run_until_complete(consumer.run())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
