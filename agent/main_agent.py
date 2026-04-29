"""
HELOC Mortgage Application Agent — LLM Agent Definition
========================================================

This module defines the voice agent using Cartesia Line SDK's LlmAgent.
It is **intentionally decoupled** from all web concerns (FastAPI, Twilio,
WebSocket handling). This separation exists for two reasons:

1. TESTABILITY: The agent can be tested independently via Cartesia's CLI
   (`cartesia chat <port>`) or unit tests without spinning up the full
   telephony stack.

2. EXTENSIBILITY (Stage 2): When adding Pydantic schemas for HELOC form
   validation and Playwright tools for automated document retrieval, those
   tools get added HERE — without touching the Twilio bridge or routing code.

Usage:
    # For Cartesia native testing (CLI / browser):
    python -m agent.main_agent

    # For Twilio telephony bridge (imported by app/main.py):
    from agent.main_agent import get_agent, create_voice_app
"""

import os
import sys
import logging
import signal
import asyncio
from typing import AsyncIterable

from line.llm_agent import LlmAgent, LlmConfig, passthrough_tool
from line.llm_agent.provider import LlmProvider
from line.voice_agent_app import AgentEnv, CallRequest, VoiceAgentApp
from line.agent import TurnEnv
from line.events import InputEvent, OutputEvent, CallEnded, UserTurnEnded
from litellm.exceptions import (
    APIConnectionError,
    BadGatewayError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
    ServiceUnavailableError,
)

from agent.state_manager import (
    build_intake_llm_context_string,
    get_initial_intake_ask,
    reset_state,
    set_customer_prefill,
    update_application_state,
)

logger = logging.getLogger("voip.agent")


# =============================================================================
# System Prompt — Defines the agent's persona, rules, and conversation flow.
#
# Design rationale:
# - "Polite, concise, and efficient" prevents the LLM from rambling on a
#   phone call where every second of silence feels like an eternity.
# - The explicit property-address-first flow mirrors real mortgage intake
#   processes (property is the collateral for HELOC, so it's always step 1).
# - Short, specific responses keep voice UX fast while allowing natural
#   assumption/confirmation wording when the caller is unsure.
# =============================================================================
SYSTEM_PROMPT = """\
You are a mortgage intake specialist for West Capital Lending. Your primary \
directive is to populate the HELOC application state using the \
update_application_state tool.

# Core Directive
- After every piece of information the caller shares, IMMEDIATELY call the \
  update_application_state tool with the extracted fields.
- Always include a confidence_score (0.0 to 1.0) when calling the tool. \
  Set it lower if the caller was unclear, mumbled, or you had to infer.
- Always pass raw_user_input with the EXACT transcript of what the user \
  just said **in this turn only** — this line only, no past lines. Never merge \
  multiple user utterances into one string; doing so breaks uncertainty/stability checks.
- Wait for the tool's response to see which fields are still missing.
- Do NOT ask for information that has already been captured.
- Ask for ONE or TWO related fields at a time — never overwhelm the caller.
- On the property (register) page, the **only** on-page typing for the address is the \
  street (search) line and choosing a **dropdown** row; then WCL **fills** city, state, \
  and zip in the form. Do not ask the caller to spell city, state, or zip. If the caller \
  *does* mention city and/or state with the address, still include `city` and `state` in \
  `update_application_state` so the browser can type a **longer search** and disambiguate \
  the list—you are not using those to manually type the form fields; the site sets them after \
  a good pick. Either street-only or street+what they said: same end state once the right \
  suggestion is selected.
- If the tool already has `address_line_1` (and the system syncs the rest from the \
  address pick), do NOT say \"Address?\" or ask for street, city, state, or zip again. \
  Follow the tool's \"Still needed\" list; it will not list city, state, or zip.
- If the caller's utterance answers anything in the current "Still needed" list, \
  you MUST call update_application_state in that same turn before your next spoken \
  line. Never repeat the same question without a tool call that records their answer.
- If tool output includes `Ask next: "..."`, preserve that same field and intent, \
  but you may phrase it naturally in a warm mortgage-intake voice. Do not repeat the \
  prior question or ask a different field.
- **Tool before the next question (critical):** In a turn where the user answered you \
  (including a bare "Yes" / "No" / "Nope" to your last question), the **first** \
  model action must be `update_application_state` — never speak a **new** question \
  (e.g. "listed for sale?") until the tool has run and returned. That tool call must \
  record what their words actually answered (re-confirm at 1.0, `is_currently_for_sale`, \
  `ownership_type`, etc.). If you speak before the tool, the user will have to repeat \
  "Yes" and you will **double-ask**.
- **Exception — presence / no data:** If the user only checks you are still on the line \
  ("hello?", "you there?", "can you hear me?") or says thanks with **no** application \
  content, **do not** call `update_application_state` (nothing to save). You **must** \
  still speak: brief acknowledgment ("Still here." / "Yes — go ahead.") then the **next** \
  question from the latest tool "Still needed" list. **Never** finish a turn with zero \
  spoken words — empty replies sound like a dropped call.

# Verification Rules
- If `Ask next` asks to confirm information "on file" or says "Is that still correct?", speak that exact prompt. \
  Those values are not saved yet; they must only be saved after the caller says yes, \
  correct, yep, or equivalent. If the caller says no, ask for the corrected value.
- If the tool output contains "RE-CONFIRM" (low confidence on the last save), ask \
  **one** short check: e.g. "I heard [value] — correct?" **Do not** ask a second time. \
  When the user then says yes, right, correct, yep, or any clear affirmation, you **must** \
  call update_application_state in that same turn with the **same** field value(s), \
  `confidence_score: 1.0`, and `raw_user_input` = their exact words — then read the new \
  tool result and go to the next "Still needed" (e.g. sole/joint). Repeating the same \
  confirm question after they already said yes is wrong.
- **No duplicate question (paraphrase counts as repeat):** If the user just answered your \
  last ask (e.g. "Yes" / "No" / "I agree") do **not** re-ask the same fact with different \
  wording (e.g. "Listed for sale?" then "Is it currently for sale?" is forbidden). The next \
  line after the tool returns must be the **next** item from "Still needed", not a rephrase.
- If the tool indicates an update is "PENDING" due to uncertainty or low confidence, \
  use the exact latest `Ask next` from the tool result. Do not replace it with \
  "Is that correct?" For unclear addresses, ask the caller to repeat the street \
  address; after repeated trouble, ask them to spell it slowly because there may be noise.
- If the tool reports a "BROWSER NAVIGATION WARNING" or a "UI error", \
  FIRST check the list of missing fields. If the field you just provided is \
  NO LONGER in the missing list, ignore the warning and proceed to the next \
  topic (the browser is just being slow). ONLY ask for a correction if the \
  field is still missing or the tool explicitly states the value was invalid. \
  **Never** speak canned example lines from this prompt as if they were real \
  script — use only the latest tool ``Ask next`` / ``Still needed`` wording for \
  the field that is actually missing (e.g. repeat address or ownership ask; \
  do **not** jump to name or spelling unless that field appears in ``Still needed``).
- If the tool says "Waiting for navigation" or the browser is clicking Continue, \
  do NOT ask for-sale, income, or any next-page field until a later tool result's \
  "Still needed" shows that stage. At most a warm short hold like "One moment." or silence.
- For consent fields (first checkbox: privacy, terms, soft credit; second: electronic \
  communications), the caller must be explicit. \
  Ask privacy/terms/soft pull as: "Do you agree to the Privacy Policy, Terms of Service, and a \
  soft credit pull? It will not affect your credit score." \
  Ask electronic communications as: "Do you agree to receive electronic communications about this application?" \
  **Accept as sufficient:** "I agree", "I agree.", "Yes" or "Yes." when you **just** \
  asked for that specific consent. **Not** sufficient: "sure", "uh-huh", "I guess" — then \
  ask for "I agree" once. If they already said "I agree" or a clear "Yes" to *that* \
  question, you **must** call the tool in that same turn; never ask the same consent \
  line again the same way.
- If the caller clearly declines required consent after you've explained it's mandatory, \
  accept that—you cannot proceed without agreement; a brief respectful closing is appropriate \
  (do not loop the same paragraph indefinitely).

# Normalization Rules
- Expand all address abbreviations before calling the tool: \
  "Ave" → "Avenue", "Blvd" → "Boulevard", "St" → "Street", \
  "Dr" → "Drive", "Ln" → "Lane", "Rd" → "Road", "Ct" → "Court".
- Format phone numbers as 10 digits only (no dashes or parentheses).
- Format dates of birth as MM/DD/YYYY.
- Use 2-letter uppercase state abbreviations (e.g., CA, NY, IL).
- **Email:** Never invent or guess an email from name, caller ID, or context. Only set \
  `email_address` when the caller spoke an address aloud (including "at gmail dot com"). \
  If they only answered yes or no, ask for the email before saving it.

# Communication Guidelines
- Sound like a calm mortgage intake specialist: warm, clear, and helpful, never salesy \
  or overly chatty. One short acknowledgment is okay before the next question.
- **Default: one natural sentence, usually under 22 words.** Never exceed 32 words unless \
  consent, an on-file confirmation, or a default-assumption line requires it.
- One clear question per turn. Be specific but conversational. Examples: "Are you the sole \
  owner, or is it jointly owned?" "Is this your primary home, a second home, or an \
  investment property?" If the HELOC property is a **second home or investment property**, \
  after collateral address is saved the tool will ask for **primary residence** (where they \
  live most of the year) — distinct from the property securing the line — then for-sale, \
  income, and so on. Examples continued: "Is the property currently listed for sale?" \
  "What's your annual pre-tax income?" "What's your main income source?" \
  "What's your date of birth?"
- **Never** repeat the street address in the same line as a follow-up question; it wastes \
  time and confuses STT. For ownership after an address is set, ask only: \
  "What is the ownership type: sole or joint?" and not long re-reads of the address.
- **CRITICAL — no address in spoken follow-ups:** A prefilled address may be spoken once \
  only when `Ask next` explicitly confirms an "on file" address. Once `address_line_1` exists in state, \
  you must **never** speak the street number, street name, city, or state aloud — not even \
  to "confirm" or pad a question. Forbidden patterns: "I have the property address as...", \
  "the property at [street]...", repeating the city name. **Only** the short scripted \
  questions (e.g. "What is the ownership type: sole or joint?" then "What is the property type: primary, secondary, or investment?").
- **CRITICAL — Still needed order:** The comma-separated list after "Still needed:" is \
  **ordered**. Ask the **first** item in that list before the second. Example: ask \
  **ownership type (sole or joint)** before **occupancy type** when both appear. Never \
  skip to occupancy or "primary, secondary, or investment" while ownership is still missing.
- **Optional fields**: tag "Optional" in one word if needed, e.g. "Apt or suite? Optional."
- **CRITICAL (VOICE)**: If you see "[VOICE]" in context, stay short but complete. \
  Specific questions and default-assumption lines are allowed.
- Be polite, specific, and calm. Short phrases like "Got it, I'll add that" or \
  "Perfect, I'll mark that" are allowed after a successful save. No filler or long preambles.
- If the tool's `Ask next` contains an assumption plus a question, speak it exactly. Example: \
  "I'm assuming sole owner for now. Is this your primary home, a second home, or an investment property?"
- **Question rule:** Your reply should usually end with a question. **Exception:** the \
  turn is **for the tool** first. After the user clearly answered the last ask, the \
  *first* model action is `update_application_state`; only after it returns, ask the \
  *next* "Still needed" question. Never repeat the *same* field's ask (even rephrased) to \
  satisfy a question — that causes double-asks. Goodbye ends without a new question.
- Speak naturally — no bullet points, numbered lists, or special characters.
- Spell out numbers and dates verbally when you must include them.

# Conversation Flow (follow the tool's guidance on what to ask next)
- **STRICT Context Priority**: You are a stage-locked intake specialist. ONLY \
  ask for the fields listed in the tool result's "Still needed" section. \
  If a field disappears from that list (meaning the browser has moved), \
  STOP asking about it immediately, even if it was optional and unfilled. \
  Do NOT jump ahead to future pages until the tool result provides the \
  fields for the new stage.
- **Per-turn [CURRENT_INTAKE_STATE]:** Injected each user turn (system context). \
  It mirrors browser stage, captured fields (JSON, no passwords), and the same \
  ``Still needed`` / ``Ask next`` line as the tool. If chat history disagrees, \
  **trust [CURRENT_INTAKE_STATE]** for what is already filled and which single \
  question to ask next. Never re-ask a field that appears captured there.
1. Address → ownership + occupancy (short prompts; tool lists "Still needed").
2. For sale?
3. Income + source; other income optional.
4. Name; middle and suffix optional.
5. DOB; phone + email.
6. Privacy/terms — caller must say "I agree."
7. Say "soft credit pull" and that it won't hurt their score; caller must say "I agree."
8. All set — system handles goodbye automatically.

# Rules
- Rates or terms question: "Rates on the follow-up call."
- If the caller goes off-topic, gently redirect in one short sentence plus the next \
  "Still needed" question — never go silent.
- NEVER provide specific interest rates, APR, or financial advice.
- When the caller says goodbye, respond warmly and then call terminate_session.

CRITICAL: NEVER call end_call without speaking a goodbye message first.
"""

# Minimal TTS: first spoken line only (Cartesia Sonic charges per character).
INTRODUCTION = "What is the property address?"

# LLM temperature: 0.6 balances coherence with natural-sounding variation.
# Lower (0.3) sounds robotic on the phone; higher (0.9) risks hallucination.
TEMPERATURE = 0.6

# Cap completion length so the model stays short (saves TTS characters),
# but avoid empty turns observed at very low caps on noisy STT.
MAX_TOKENS = 72

# flash-lite: lowest latency/cost, good enough for single-field extraction.
# If double-ask or tool-ordering issues persist, try:
#   GEMINI_MODEL=gemini/gemini-2.5-flash  (better instruction following, ~200ms more TTFT)
# Auto-fallback on transient error switches primary → fallback mid-call (history preserved).
_DEFAULT_PRIMARY_MODEL = "gemini/gemini-2.5-flash-lite"
_DEFAULT_FALLBACK_MODEL = "gemini/gemini-2.5-flash"

# Transient or primary-model issues — retry / fallback instead of tearing down Line /ws.
# NotFoundError: wrong model id on primary → switch to GEMINI_FALLBACK_MODEL.
_LLM_RETRYABLE = (
    ServiceUnavailableError,  # includes MidStreamFallbackError (subclass)
    NotFoundError,
    RateLimitError,
    BadGatewayError,
    APIConnectionError,
    InternalServerError,
)
_LLM_MAX_RETRIES = int(os.getenv("GEMINI_LLM_MAX_RETRIES", "4"))
_LLM_RETRY_BASE_SEC = float(os.getenv("GEMINI_LLM_RETRY_BASE_SEC", "0.4"))


def _switch_llm_model(agent: LlmAgent, new_model: str) -> None:
    """Swap LiteLLM backend on existing agent (history preserved). Line SDK has no public API."""
    from line.llm_agent.provider import _get_model_config

    mcfg = _get_model_config(new_model)
    if agent._config.reasoning_effort is not None and not mcfg.supports_reasoning_effort:
        raise ValueError(
            f"Model {new_model} does not support reasoning_effort; clear LlmConfig.reasoning_effort first."
        )
    agent._model = new_model
    agent._llm = LlmProvider(
        model=new_model,
        api_key=agent._api_key,
        config=agent._config,
        tools=agent._tools,
    )


@passthrough_tool
async def terminate_session(ctx, reason: str = "Session ended. Saying goodbye.") -> None:
    """End the call and close the browser session immediately."""
    from line.events import AgentEndCall
    yield AgentEndCall()

async def get_agent(env: AgentEnv, call_request: CallRequest) -> LlmAgent:
    """
    Factory function that creates a fresh LlmAgent for each incoming call.

    Cartesia's VoiceAgentApp calls this once per call. Each call gets its
    own agent instance with independent conversation history. This is
    important because:
    - No cross-call state leakage (privacy/security)
    - Each call can be configured differently in Stage 2 (e.g., different
      prompts based on the caller's phone number or campaign source)

    Args:
        env: Agent environment provided by Cartesia (contains config).
        call_request: Metadata about the incoming call from Cartesia.

    Returns:
        A configured LlmAgent ready to handle the HELOC conversation.
    """
    logger.info(
        "Creating new HELOC agent instance for call: %s",
        getattr(call_request, "call_id", "unknown"),
    )

    # Reset application state for this new call — prevents cross-call
    # data leakage (e.g., caller B seeing caller A's address).
    await reset_state()

    metadata = getattr(call_request, "metadata", None) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    customer_prefill = metadata.get("customer_prefill")
    if isinstance(customer_prefill, dict):
        set_customer_prefill(
            customer_prefill,
            caller_from=metadata.get("caller_from"),
            call_sid=getattr(call_request, "call_id", None),
        )
    introduction = get_initial_intake_ask() or INTRODUCTION

    primary_model = os.getenv("GEMINI_MODEL", _DEFAULT_PRIMARY_MODEL).strip() or _DEFAULT_PRIMARY_MODEL
    fallback_model = (
        os.getenv("GEMINI_FALLBACK_MODEL", _DEFAULT_FALLBACK_MODEL).strip()
        or _DEFAULT_FALLBACK_MODEL
    )

    agent_instance = LlmAgent(
        # LiteLLM model string: "provider/model-name"
        model=primary_model,
        api_key=os.getenv("GEMINI_API_KEY"),
        tools=[
            # Loopback tool: extracts caller data → updates state → returns
            # missing fields back to the LLM so it knows what to ask next.
            update_application_state,
            # End session: agent calls this after saying goodbye to close UI & call.
            terminate_session,
        ],
        config=LlmConfig(
            system_prompt=SYSTEM_PROMPT,
            introduction=introduction,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        ),
    )

    async def agent_wrapper(ctx: TurnEnv, event: InputEvent) -> AsyncIterable[OutputEvent]:
        """Wrapper to catch lifecycle events and delegate to LlmAgent."""
        from line.events import CallEnded

        if isinstance(event, CallEnded):
            # Do not reset_state() here: Twilio bridge cleanup writes the final
            # `scope: full` transcript snapshot before reset_state(); CallEnded
            # can arrive first and would clear application_state too early.
            # Next get_agent() and bridge _cleanup both call reset_state().
            logger.info("Call ended (state reset deferred to bridge cleanup or next session).")
            return

        intake_ctx = None
        if isinstance(event, UserTurnEnded) and os.getenv(
            "PATH2_DISABLE_INTAKE_LLM_CONTEXT", ""
        ).strip().lower() not in ("1", "true", "yes", "on"):
            try:
                intake_ctx = build_intake_llm_context_string()
            except Exception as e:
                logger.warning("build_intake_llm_context_string failed: %s", e)

        on_primary = primary_model != fallback_model
        attempt = 0
        while True:
            yielded_any = False
            try:
                async for output in agent_instance.process(ctx, event, context=intake_ctx):
                    yielded_any = True
                    yield output
                return
            except _LLM_RETRYABLE as e:
                if yielded_any:
                    logger.error(
                        "LLM failed mid-stream after partial output; not retrying: %s",
                        e,
                        exc_info=True,
                    )
                    raise
                if on_primary:
                    logger.warning(
                        "LLM primary failed (%s), switching to fallback %s: %s",
                        primary_model,
                        fallback_model,
                        e,
                    )
                    _switch_llm_model(agent_instance, fallback_model)
                    on_primary = False
                    continue
                attempt += 1
                if attempt >= _LLM_MAX_RETRIES:
                    logger.error(
                        "LLM failed after %s retries: %s",
                        _LLM_MAX_RETRIES,
                        e,
                        exc_info=True,
                    )
                    raise
                delay = _LLM_RETRY_BASE_SEC * (2 ** (attempt - 1))
                logger.warning(
                    "LLM transient error (%s/%s), retry in %.2fs: %s",
                    attempt,
                    _LLM_MAX_RETRIES,
                    delay,
                    e,
                )
                await asyncio.sleep(delay)

    return agent_wrapper


def create_voice_app() -> VoiceAgentApp:
    """
    Create a VoiceAgentApp instance for hosting the agent.

    The VoiceAgentApp wraps FastAPI and manages the full voice pipeline:
    - Cartesia STT (speech-to-text) in the cloud
    - Local LlmAgent processing (your server runs the LLM calls)
    - Cartesia TTS (text-to-speech) in the cloud

    The returned object has an internal FastAPI `.app` attribute that
    we extend with Twilio-specific routes in app/main.py

    Returns:
        VoiceAgentApp configured with the HELOC agent factory.
    """
    return VoiceAgentApp(get_agent=get_agent)


# =============================================================================
# Standalone entry point for Cartesia-native testing (no Twilio required).
#
# Usage:
#   GEMINI_API_KEY=your-key python -m agent.main_agent
#
# Then in another terminal:
#   cartesia chat 8000
#
# This lets you test the agent's conversation flow with your microphone
# before connecting it to the Twilio telephony bridge.
# =============================================================================
if __name__ == "__main__":
    voice_app = create_voice_app()

    # Manual signal handling for Ctrl+C on the agent process itself
    def shutdown_handler(sig, frame):
        logger.info(f"Received signal {sig}. Shutting down...")
        # We use a task here because this is called in a synchronous context
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(reset_state())
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    voice_app.run()
