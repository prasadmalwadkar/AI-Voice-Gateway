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
import logging

from line.llm_agent import LlmAgent, LlmConfig, end_call
from line.voice_agent_app import AgentEnv, CallRequest, VoiceAgentApp

from agent.state_manager import update_application_state, reset_state

logger = logging.getLogger("voip.agent")


# =============================================================================
# System Prompt — Defines the agent's persona, rules, and conversation flow.
#
# Design rationale:
# - "Polite, concise, and efficient" prevents the LLM from rambling on a
#   phone call where every second of silence feels like an eternity.
# - The explicit property-address-first flow mirrors real mortgage intake
#   processes (property is the collateral for HELOC, so it's always step 1).
# - The 35-word limit per response is critical for voice UX — long LLM
#   responses cause the caller to zone out or interrupt mid-sentence.
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
  just said. This is mandatory — the system uses it to detect uncertainty.
- Wait for the tool's response to see which fields are still missing.
- Do NOT ask for information that has already been captured.
- Ask for ONE or TWO related fields at a time — never overwhelm the caller.

# Verification Rules
- If the tool says a field "NEEDS VERBAL RE-CONFIRMATION", do NOT move to \
  the next topic. Confirm the current value first by asking: "I heard \
  [value], is that correct?"
- If the tool indicates an update is "PENDING" due to uncertainty, \
  apologize for the confusion and ask for a clear confirmation. Example: \
  "I want to make sure I get this right — is your address 325 West Adams \
  Boulevard?" Wait for a clear 'Yes', 'No', or corrected value.
- For consent fields (privacy policy, credit pull), the caller must \
  explicitly say "I agree." Vague responses like "sure" or "I guess" are \
  not sufficient — ask them to say "I agree" clearly.

# Normalization Rules
- Expand all address abbreviations before calling the tool: \
  "Ave" → "Avenue", "Blvd" → "Boulevard", "St" → "Street", \
  "Dr" → "Drive", "Ln" → "Lane", "Rd" → "Road", "Ct" → "Court".
- Format phone numbers as 10 digits only (no dashes or parentheses).
- Format dates of birth as MM/DD/YYYY.
- Use 2-letter uppercase state abbreviations (e.g., CA, NY, IL).

# Communication Guidelines
- Keep responses under 35 words. Phone callers lose patience with long answers.
- Be polite, concise, and efficient. Never ramble.
- Always end your response with a question to keep the conversation moving, \
  except when saying goodbye.
- Speak naturally — no bullet points, numbered lists, or special characters.
- Spell out numbers and dates verbally: "January fifteenth" not "1/15".

# Conversation Flow (follow the tool's guidance on what to ask next)
1. Property address (street, city, state, zipcode).
2. Ownership type (sole or joint) and occupancy (primary, secondary, investment).
3. Whether the property is currently for sale.
4. Annual pre-tax income and primary source of income.
5. Legal name (first, middle, last, suffix).
6. Date of birth.
7. Mobile phone number and email address.
8. Verbal consent to Privacy Policy / Terms of Service.
9. Verbal consent to Electronic Communications Policy and soft credit pull.
10. Once ALL fields are collected, confirm the details back to the caller.

# Rules
- If the caller asks about rates or terms, say: "Our loan officers can \
  discuss specific rates and terms during your follow-up call."
- If the caller goes off-topic, gently redirect back to the application.
- NEVER provide specific interest rates, APR, or financial advice.
- When the caller says goodbye, respond warmly and then call end_call.

CRITICAL: NEVER call end_call without speaking a goodbye message first.
"""

INTRODUCTION = (
    "Hi, I'm your West Capital assistant. "
    "I can help you start your HELOC application today. "
    "To get started, what is the address of the property "
    "you are looking to finance?"
)

# LLM temperature: 0.6 balances coherence with natural-sounding variation.
# Lower (0.3) sounds robotic on the phone; higher (0.9) risks hallucination.
TEMPERATURE = 0.6

# Max tokens per response: 200 keeps responses short for voice delivery.
# A 200-token response is roughly 15-20 seconds of speech — already long
# for a phone interaction.
MAX_TOKENS = 200


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
    reset_state()

    agent = LlmAgent(
        # LiteLLM model string: "provider/model-name"
        # gemini/gemini-2.5-flash is cost-effective with fast inference,
        # ideal for real-time voice where latency = bad UX.
        model="gemini/gemini-2.5-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        tools=[
            # Loopback tool: extracts caller data → updates state → returns
            # missing fields back to the LLM so it knows what to ask next.
            update_application_state,
            # End call: agent calls this after saying goodbye.
            end_call,
        ],
        config=LlmConfig(
            system_prompt=SYSTEM_PROMPT,
            introduction=INTRODUCTION,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        ),
    )

    return agent


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
    voice_app.run()
