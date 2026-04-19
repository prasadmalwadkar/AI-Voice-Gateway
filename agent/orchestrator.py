import asyncio
import sys
import os
import json
import logging
import base64
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.state_manager import update_application_state

try:
    import pyaudio
    import webrtcvad
    import websockets
    import audioop
    AUDIO_READY = True
except ImportError:
    AUDIO_READY = False

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("orchestrator")

LOCAL_RATE = 16000
CHANNELS = 1
CHUNK_DURATION_MS = 20
CHUNK_SIZE = int(LOCAL_RATE * CHUNK_DURATION_MS / 1000)
BRIDGE_RATE = 8000

# Seconds to keep mic muted after last agent audio chunk,
# prevents echo from triggering barge-in.
ECHO_GUARD_SECONDS = 0.4


class TwilioMimicManager:
    """
    Mimics Twilio's Media Stream protocol to test the bridge locally.
    Connects to /media-stream with local mic/speaker.
    """
    def __init__(self, ws_url="ws://localhost:8000/media-stream"):
        self.ws_url = ws_url
        self.audio = None
        self.mic = None
        self.speaker = None
        self.vad = None
        self.ws = None
        self.active = False
        self.stream_sid = "cli-stream"

        self.up_state = None
        self.down_state = None

        # Echo suppression: purely time-based. Mic is muted whenever
        # agent audio arrived recently (within ECHO_GUARD_SECONDS).
        self._last_agent_audio_time = 0.0
        # Block mic until first agent audio arrives (let introduction play)
        self._heard_agent = False

    async def start(self):
        if not AUDIO_READY:
            print("\n[Error] Dependencies missing (pyaudio, webrtcvad, audioop-lts).")
            return False

        try:
            self.audio = pyaudio.PyAudio()
            self.vad = webrtcvad.Vad(3)  # Most aggressive filtering

            self.mic = self.audio.open(
                format=pyaudio.paInt16, channels=CHANNELS, rate=LOCAL_RATE,
                input=True, frames_per_buffer=CHUNK_SIZE
            )
            self.speaker = self.audio.open(
                format=pyaudio.paInt16, channels=CHANNELS, rate=LOCAL_RATE,
                output=True
            )

            self.ws = await websockets.connect(self.ws_url)

            await self.ws.send(json.dumps({
                "event": "start",
                "start": {
                    "streamSid": self.stream_sid,
                    "callSid": "cli-call-" + os.urandom(4).hex()
                }
            }))

            self.active = True

            asyncio.create_task(self._mic_loop())
            asyncio.create_task(self._speaker_loop())

            return True
        except Exception as e:
            logger.error(f"Bridge connection failed: {e}")
            await self.stop()
            return False

    async def stop(self):
        self.active = False
        if self.ws:
            await self.ws.close()
            self.ws = None
        if self.mic:
            self.mic.stop_stream(); self.mic.close(); self.mic = None
        if self.speaker:
            self.speaker.stop_stream(); self.speaker.close(); self.speaker = None
        if self.audio:
            self.audio.terminate(); self.audio = None

    def _mic_is_muted(self) -> bool:
        """Mic muted until agent speaks once, then muted whenever agent
        audio arrived within the last ECHO_GUARD_SECONDS."""
        if not self._heard_agent:
            return True
        return (time.monotonic() - self._last_agent_audio_time) < ECHO_GUARD_SECONDS

    async def _mic_loop(self):
        """Record (16k PCM) -> Transcode (8k mu-law) -> Send (Bridge)."""
        try:
            while self.active:
                data = self.mic.read(CHUNK_SIZE, exception_on_overflow=False)

                if not self._mic_is_muted() and self.vad.is_speech(data, LOCAL_RATE):
                    downsampled, self.down_state = audioop.ratecv(
                        data, 2, 1, LOCAL_RATE, BRIDGE_RATE, self.down_state
                    )
                    mulaw_data = audioop.lin2ulaw(downsampled, 2)

                    await self.ws.send(json.dumps({
                        "event": "media",
                        "media": {
                            "payload": base64.b64encode(mulaw_data).decode('utf-8')
                        }
                    }))

                await asyncio.sleep(0)
        except Exception as e:
            if self.active: logger.debug(f"Mic error: {e}")

    async def _speaker_loop(self):
        """Receive from bridge -> decode -> play through speaker."""
        try:
            async for message in self.ws:
                data = json.loads(message)

                if data.get("event") == "media":
                    if not self._heard_agent:
                        self._heard_agent = True
                        print("\r  [Agent speaking...]        ", end="", flush=True)
                    self._last_agent_audio_time = time.monotonic()

                    mulaw_payload = base64.b64decode(data["media"]["payload"])
                    pcm_8k = audioop.ulaw2lin(mulaw_payload, 2)
                    pcm_8k = audioop.mul(pcm_8k, 2, 2.0)

                    pcm_16k, self.up_state = audioop.ratecv(
                        pcm_8k, 2, 1, BRIDGE_RATE, LOCAL_RATE, self.up_state
                    )

                    self.speaker.write(pcm_16k)

                elif data.get("event") == "clear":
                    print("\r  [Listening...]             ", end="", flush=True)

        except Exception as e:
            if self.active: print(f"\n[Bridge Connection Error] {e}")


class Orchestrator:
    def __init__(self):
        self.mode = "CLI"
        self.bridge_manager = TwilioMimicManager()
        self.running = True

    async def run(self):
        print("\n" + "="*60)
        print("   AI Voice Gateway — Bridge Orchestrator")
        print("="*60)
        print("Commands: /voice  /chat  exit\n")

        while self.running:
            if self.mode == "CLI":
                try:
                    user_input = await asyncio.get_event_loop().run_in_executor(None, input, ">> ")
                    user_input = user_input.strip()
                    if not user_input: continue
                    if user_input.lower() == "/voice":
                        if await self.bridge_manager.start():
                            self.mode = "VOICE"
                            print("\n[VOICE MODE] Speak now. Use headphones to avoid echo.")
                            print("  Type /chat to return to CLI mode.\n")
                        continue
                    if user_input.lower() == "exit":
                        self.running = False; break
                    await self._process_text(user_input)
                except EOFError: break
            else:
                user_input = await asyncio.get_event_loop().run_in_executor(None, input, "")
                if user_input.strip().lower() == "/chat":
                    await self.bridge_manager.stop()
                    self.mode = "CLI"
                    print("\n[CLI MODE]")

    async def _process_text(self, text: str):
        updates = {}
        if "=" in text:
            for part in text.split(','):
                if '=' in part:
                    k, v = part.split('=', 1)
                    updates[k.strip()] = v.strip()
        result = await update_application_state.func(None, **updates, confidence_score=1.0, raw_user_input=text)
        print(f"\n[System] {result}")

if __name__ == "__main__":
    orchestrator = Orchestrator()
    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt: pass
