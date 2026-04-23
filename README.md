# AI Voice Gateway

Voice bridge and agent for **West Capital Lending** HELOC (Home Equity Line of Credit) phone intake. Twilio Media Streams connect callers to a **Cartesia Line**–based voice agent; **Google Gemini** (via LiteLLM) powers the **LlmAgent**. Application state and **Playwright** keep the live **West Capital** web form in sync with what the caller says.

## Where things stand (current implementation)

- **Default phone path — Path2** (`TWILIO_VOICE_MODE=path2`, or `line` / `local`): Twilio **→** Cartesia **Ink STT** (8 kHz µ-law) **→** local **Line `/ws`** session **→** **LlmAgent** + tools **`update_application_state`** **→** **Gemini** **→** **Cartesia Sonic TTS** **→** Twilio. **Playwright** drives the HELOC UI when state updates run browser sync ([`agent/state_manager.py`](agent/state_manager.py), [`agent/browser_manager.py`](agent/browser_manager.py)).
- **Alternate phone path — Cartesia Calls API**: If `TWILIO_VOICE_MODE` is **not** `path2` / `line` / `local`, [`app/twilio_bridge.py`](app/twilio_bridge.py) streams **µ-law** to Cartesia’s **hosted** agent WebSocket (`CARTESIA_AGENT_WS_URL` or bundled default). **No** local `LlmAgent`, **no** this-repo state/Playwright loop — different product surface.
- **Local full stack (no Twilio phone) — recommended**: with the gateway up, run [`python agent/orchestrator.py`](agent/orchestrator.py). It **mimics Twilio** over **`ws://localhost:8000/media-stream`** (mic/speaker, optional VAD, optional **Path2 transcript debug**). Same **Path2** pipeline as a real call: **Ink STT** → **Line `/ws`** → **LlmAgent** → **Sonic TTS** — see **§5** for deps and commands.
- **Local agent only (Cartesia CLI)**: [`python -m agent.main_agent`](agent/main_agent.py) and **`cartesia chat <port>`** — **Line** STT/TTS from Cartesia’s client (same **LlmAgent**, not the **TwilioLineBridge** Ink/Sonic path).
- **Unified server** (prerequisite for orchestrator and Twilio): [`python run.py`](run.py) serves **TwiML** (`GET /voice`), **Twilio** WebSocket (`/media-stream`), **health** (`GET /health`), FastAPI **docs**, Cartesia **Line** routes (e.g. **`/ws`**), and optional **Path2 transcript debug** (`WS /debug/path2-transcripts`) when enabled.

## Architecture (high level)

| Layer | Role |
| --- | --- |
| **FastAPI + Uvicorn** ([`app/main.py`](app/main.py)) | Single process: Line **`VoiceAgentApp`** + Twilio routes. |
| **Twilio** | PSTN; **TwiML** opens **`wss://…/media-stream`**. |
| **Path2** ([`app/twilio_line_bridge.py`](app/twilio_line_bridge.py)) | Twilio audio **↔** Ink STT, Line client **↔** **`/ws`**, Sonic TTS **↔** Twilio; echo guard, barge-in, pre-line heuristics. |
| **Agent** ([`agent/main_agent.py`](agent/main_agent.py)) | **LlmAgent**, system prompt, tools **`update_application_state`**, **`terminate_session`**. |
| **State** ([`agent/state_manager.py`](agent/state_manager.py)) | Pydantic **`HelocApplicationState`**, confidence/uncertainty, optional async **browser sync**. |
| **Browser** ([`agent/browser_manager.py`](agent/browser_manager.py)) | **Playwright** against the WCL HELOC funnel. |
| **Calls API bridge** ([`app/twilio_bridge.py`](app/twilio_bridge.py)) | Optional: Twilio **↔** cloud agent only; **mulaw_8000** passthrough. |

## Prerequisites

1. **Python 3.9+** (match your team standard; 3.11+ recommended).
2. **Cartesia API key** — STT/TTS (Path2) and/or cloud agent (Calls path).
3. **Google AI Studio / Gemini API key** — `GEMINI_API_KEY` for the local **LlmAgent**.
4. **Twilio** — for real phone calls (number + webhook).
5. **Playwright browsers** — install once if you use form sync: `playwright install chromium` (or your CI equivalent).
6. **Orchestrator (local Path2)** — **PyAudio** + **webrtcvad** (pip install when using [`python agent/orchestrator.py`](agent/orchestrator.py)); **websockets** comes from `requirements.txt`.

## Getting started

### 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment

Create a **`.env`** in the project root (see [`.gitignore`](.gitignore); do not commit secrets). Typical variables:

| Variable | Purpose |
| --- | --- |
| `GEMINI_API_KEY` | Gemini for **LlmAgent** ([`agent/main_agent.py`](agent/main_agent.py)). |
| `CARTESIA_API_KEY` | Ink STT + Sonic TTS (Path2) or cloud agent (Calls path). |
| `CARTESIA_TTS_SPEED` | Path2 only — Sonic-3 **speech rate** `0.6` (slow) … `1.5` (fast); default **0.75** (see [`app/cartesia_tts_mulaw.py`](app/cartesia_tts_mulaw.py)). |
| `CARTESIA_TTS_VOLUME` | Path2 only — Sonic-3 **volume** `0.5` (quiet) … `2.0` (loud); default **1.5** for PSTN audibility. |
| `CARTESIA_STT_MAX_SILENCE_SECS` | Ink STT — seconds of silence before an utterance is **final**; higher = wait for longer phrases (default **0.45**; lower for snappier local mic / orchestrator). |
| `TWILIO_VOICE_MODE` | `path2` (default), `line`, `local` → **Path2**; anything else → **Calls API** bridge. |
| `CARTESIA_AGENT_WS_URL` | Optional; used when **not** in Path2 mode ([`app/main.py`](app/main.py)). |
| `HOST`, `PORT` | Server bind ([`run.py`](run.py)); default `0.0.0.0:8000`. |

`load_dotenv()` runs in [`run.py`](run.py) before app import.

### 3. Run the phone gateway

```bash
python run.py
```

Endpoints (see also [`run.py`](run.py) docstring):

- **`GET /voice`** — TwiML → Twilio opens **`/media-stream`**
- **`WS /media-stream`** — bidirectional **µ-law** audio
- **`GET /health`** — liveness JSON
- **`GET /docs`** — OpenAPI
- **`WS /ws`** (and other Line SDK routes) — **LlmAgent** / Line protocol
- **`WS /debug/path2-transcripts`** — optional Path2 STT/agent debug fan-out

### 4. Expose with ngrok (dev)

```bash
ngrok http 8000
```

In Twilio, set the voice webhook to:

`https://<your-subdomain>.ngrok-free.app/voice`

Twilio requires **HTTPS** / **`wss://`** for media streams.

### 5. Local end-to-end (orchestrator — no Twilio)

Use this to exercise the **same Path2 path** as production (**Ink STT**, **Line**, **LlmAgent**, **Sonic TTS**, optional **transcript debug**), with your **mic and speakers** (headphones help avoid echo).

**Terminal 1 — gateway (must be running first):**

```bash
python run.py
```

**Terminal 2 — Twilio protocol mimic:**

```bash
python agent/orchestrator.py
```

In the CLI, type **`/voice`** to connect and speak; use **`/chat`** to leave voice mode, **`exit`** to quit. The script sends Twilio-style **`start` / `media`** frames to **`/media-stream`** and plays returned audio. You need **PyAudio**, **webrtcvad**, and **websockets** (not all are in `requirements.txt` — install locally for the orchestrator). On **Python 3.13+**, use **`audioop-lts`** (see `requirements.txt`). Optional env: `ORCHESTRATOR_VAD_MODE`, `ORCHESTRATOR_SHOW_STT_PARTIALS`, etc. — see [`agent/orchestrator.py`](agent/orchestrator.py).

With **`PATH2_TRANSCRIPT_DEBUG_WS`** enabled, the orchestrator can open a side connection to **`/debug/path2-transcripts`** for live STT/agent events (see app env for flags).

### 6. Local agent only (`cartesia chat`)

Alternative for prompt/tool testing without the full Twilio wire protocol:

**Terminal 1:**

```bash
python -m agent.main_agent
```

**Terminal 2** (Cartesia CLI, e.g. from [Cartesia](https://cartesia.ai/)):

```bash
cartesia chat 8000
```

This uses Cartesia’s **Line** client audio — **not** the orchestrator’s **`/media-stream`** mimic.

## Twilio setup guide (inbound calls)

This app answers **PSTN** calls to **your** Twilio number and streams audio to **Path2** (default). You **do not** call Twilio from code for this flow — **you dial your Twilio number** from a normal phone; Twilio then hits your server.

### 1. Twilio account and number

1. Create or log in to [Twilio](https://www.twilio.com/try-twilio) (trial is fine to start; trial accounts may only call **verified** caller IDs — verify your mobile in the console if outbound/callback is restricted; **inbound** to your purchased number still works for testing in most cases).
2. In the [Console](https://console.twilio.com/), go to **Phone Numbers** → **Buy a number** (or **Manage** → **Active numbers** if you already have one).
3. Pick a number that has **Voice** capability for the country/region you need. Complete purchase / assign to your project.

**Save:** the E.164 number (e.g. `+1 555 123 4567`) — that is what you **call** to test.

### 2. App + tunnel running (local dev)

1. **`.env`** loaded with at least `CARTESIA_API_KEY`, `GEMINI_API_KEY`, and `TWILIO_VOICE_MODE=path2` (or unset — default is Path2).
2. Start the app: `python run.py` (listens on `PORT`, default **8000**).
3. In another terminal, start a tunnel to that port, e.g. `ngrok http 8000`. If `ngrok` is not found, install it (e.g. `brew install ngrok/ngrok/ngrok` on macOS) and run `ngrok config add-authtoken <token>` once from the [ngrok dashboard](https://dashboard.ngrok.com/).
4. Copy the **HTTPS** forwarding URL from ngrok, e.g. `https://abc-xyz.ngrok-free.app` (no path yet).

5. **Sanity check** in a browser: open `https://<your-ngrok-host>/health` — you should see a JSON `healthy` response. If not, fix the app or tunnel before Twilio.

### 3. Configure the phone number in Twilio Console

1. Open **Phone Numbers** → **Manage** → **Active numbers**.
2. **Click the number** you want to use (the one you will call).
3. Scroll to the **Voice** section (name varies slightly: **Configure** / **Voice & Fax** on older UI).
4. Under **A CALL COMES IN** (incoming voice call):
   - **Configure with:** **Webhook** (or **Webhooks / TwiML** — not a static TwiML Bin only, unless you paste the same URL there; simplest is **Webhook**).
   - **URL:** `https://<your-ngrok-host>/voice`  
     Example: `https://abc-xyz.ngrok-free.app/voice`
   - **HTTP:** **GET** — must match `GET /voice` in [`app/main.py`](app/main.py). If you set **POST**, Twilio will not hit your route as written.
5. **Save** configuration.

Optional: **Status callback** URL — not required for basic audio; useful later for call lifecycle logging.

### 4. What happens on a test call

1. **You** place a **normal phone call** **to** your Twilio number.
2. Twilio sends **GET** `https://<your-host>/voice` and receives **TwiML** with `<Stream url="wss://<Host>/media-stream" />` (the `Host` header matches the hostname Twilio used, e.g. your ngrok host).
3. Twilio opens a **WebSocket** to `wss://<same-host>/media-stream` and streams **µ-law** audio. Your **TwilioLineBridge** runs **Path2** (STT → Line → LlmAgent → TTS) and sends audio back.

Watch logs: [`logs/voip.log`](logs/voip.log) (if enabled) and your terminal running `run.py` / ngrok’s [inspector](http://127.0.0.1:4040) for HTTP requests.

### 5. Troubleshooting

| Issue | What to check |
| --- | --- |
| Webhook never fires | URL typo; **HTTPS**; ngrok and `run.py` both running; **GET** not POST. |
| 404 on `/voice` | App running; path exactly **`/voice`**. |
| Call connects, no audio / errors | `CARTESIA_API_KEY` / `GEMINI_API_KEY`; `TWILIO_VOICE_MODE`; Twilio [Debugger](https://console.twilio.com/us1/monitor/logs/debugger) for webhook errors. |
| `wss` fails | Twilio needs **TLS**; ngrok **https** URL ensures `wss://` to same host. |
| ngrok URL changed | Free tunnels get a new host each session — **update the Twilio webhook** every time you restart ngrok (or use a **reserved** ngrok domain on a paid plan). |

### 6. Production (no ngrok)

Deploy `run.py` (or gunicorn/uvicorn behind HTTPS) to a public host with a stable DNS name, e.g. `https://voice.example.com`. Set the voice webhook to **`https://voice.example.com/voice`** (GET). **Do not** use ngrok in production for that host.

## Scripts and tests

- **Replay / E2E** (needs server running): [`scripts/voice_agent_e2e_test.py`](scripts/voice_agent_e2e_test.py), [`scripts/voice_soak_test.py`](scripts/voice_soak_test.py); fixtures under [`scripts/fixtures/voice_replays/`](scripts/fixtures/voice_replays/).
- **Unit / integration tests**: [`tests/`](tests/) — e.g. `pytest tests/`.
- **Orchestrator** — primary **local** driver for Path2: [`agent/orchestrator.py`](agent/orchestrator.py) (see **§5**).

## Logging

Structured JSON logs under the `voip` logger; optional rotating file `logs/voip.log` unless `LOG_FILE_DISABLE=1` (see [`app/logging_config.py`](app/logging_config.py)).
