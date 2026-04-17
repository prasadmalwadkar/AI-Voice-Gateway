# AI Voice Gateway
This is the voice bridge and agent application powering the HELOC (Home Equity Line of Credit) onboarding flow for West Capital Lending. It bridges Twilio's incoming/outgoing phone calls to a Cartesia low-latency Voice Agent, backed by Google's Gemini 2.5 Flash for conversational reasoning.

## Architecture Let's Look At A Glance

- **FastAPI / Uvicorn Server**: Handles HTTP/WebSocket requests.
- **Twilio Media Streams**: Streams live G.711 µ-law audio from the caller over WebSocket.
- **`twilio_bridge.py`**: Performs real-time transcoding between telephony µ-law (8kHz) and Cartesia's raw Linear PCM formats (16kHz).
- **Cartesia Line SDK**: Powers Speech-to-Text (STT) and Text-to-Speech (TTS).
- **Google Gemini (2.5 Flash)**: Instructed to play a highly specific persona to ingest the HELOC address, perceived value, requested credit line, and contact info.

## Prerequisites

1. Python 3.9+ 
2. A Cartesia API Key
3. A Google AI Studio Key (Gemini)
4. A Twilio Account (for actual phone calls)

## Getting Started

### 1. Installation 

Clone the repository and set up your virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment Variables

Make a copy of the example environment file:
```bash
cp .env.example .env
```
Fill out `.env` with your API keys.

### 3. Testing Locally (Without Twilio)

You can chat directly with your LLM reasoning agent right in your terminal, entirely free, without deploying to Twilio!

Start the standalone Agent:
```bash
python -m agent.main_agent
```

Then, in a second terminal, utilize the Cartesia CLI. *If you haven't installed it, run: `curl -fsSL https://cartesia.sh | sh`*.
```bash
cartesia chat 8000
```
Use your microphone and it will transcribe your responses, and print/speak back!

### 4. Running for Twilio 

To serve real phone calls:
```bash
python run.py
```

Then, use `ngrok` or another tunnel to expose port 8000 to the web:
```bash
ngrok http 8000
```

Copy the `.ngrok-free.app` URL and set it as your webhook in the Twilio Phone Numbers console for inbound calls. Make sure you point the webhook to the correct path:
`https://<your_subdomain>.ngrok-free.app/voice`
