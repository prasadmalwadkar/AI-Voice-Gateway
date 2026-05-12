import asyncio
import sys
import os
import json
import logging
import base64
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.state_manager import update_application_state
from app.path2_stt_noise import path2_stt_final_is_drop_noise

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

# webrtcvad aggressiveness 0–3 (3 = fewest false speech detections; 2 is better for quiet mics).
_VAD_MODE = int(os.getenv("ORCHESTRATOR_VAD_MODE", "2"))
# Set to 1 to send all non-muted mic frames (no VAD) — noisy; debug only.
_VAD_OFF = os.getenv("ORCHESTRATOR_VAD_OFF", "").strip() in ("1", "true", "yes")
# Live STT partials on transcript debug WS (server must send them).
_SHOW_STT_PARTIALS = os.getenv("ORCHESTRATOR_SHOW_STT_PARTIALS", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)
_SHOW_STT_NOISE_DROPS = os.getenv("ORCHESTRATOR_SHOW_STT_NOISE_DROPS", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)


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
        self._last_you_print = 0.0
        self._transcript_task: asyncio.Task | None = None
        self._ui_state = "init"
        self._session_log_file = None

    @staticmethod
    def _ts() -> str:
        return time.strftime("%H:%M:%S")

    def _emit(self, label: str, message: str, *, newline: bool = True) -> None:
        prefix = f"[{self._ts()}] [{label}] "
        end = "\n" if newline else ""
        line = f"{prefix}{message}"
        print(line, end=end, flush=True)
        self._append_session_log(line)

    def _open_session_log(self) -> None:
        if self._session_log_file:
            return
        logs_dir = os.path.join(
            os.path.dirname(__file__),
            "..",
            "scripts",
            "fixtures",
            "voice_replays",
            "orchestrator_runs",
        )
        os.makedirs(logs_dir, exist_ok=True)
        fname = datetime.now().strftime("session_%Y_%m_%d_%H_%M_%S.log")
        fpath = os.path.join(logs_dir, fname)
        self._session_log_file = open(fpath, "a", encoding="utf-8")
        self._append_session_log(f"[{self._ts()}] [System] session_log={fpath}")

    def _close_session_log(self) -> None:
        if self._session_log_file:
            try:
                self._session_log_file.close()
            except Exception:
                pass
            self._session_log_file = None

    def _append_session_log(self, line: str) -> None:
        if not self._session_log_file:
            return
        try:
            self._session_log_file.write(line + "\n")
            self._session_log_file.flush()
        except Exception:
            pass

    def _set_ui_state(self, new_state: str, detail: str = "") -> None:
        if self._ui_state == new_state:
            return
        now = time.monotonic()
        suppress = (
            new_state in ("waiting_playback", "listening")
            and self._ui_state in ("waiting_playback", "listening")
            and (now - getattr(self, "_last_ui_ts", 0)) < 1.5
        )
        self._ui_state = new_state
        self._last_ui_ts = now
        if suppress:
            return
        msg = f"state={new_state}"
        if detail:
            msg += f" | {detail}"
        self._emit("Status", msg)

    def _transcript_debug_url(self) -> str:
        url = self.ws_url.replace("/media-stream", "/debug/path2-transcripts", 1)
        token = os.getenv("PATH2_TRANSCRIPT_DEBUG_TOKEN", "").strip()
        if token:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}token={token}"
        return url

    async def _transcript_debug_loop(self) -> None:
        url = self._transcript_debug_url()
        try:
            async with websockets.connect(url) as dbg:
                async for raw in dbg:
                    if not self.active:
                        break
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    k = data.get("kind", "")
                    t = data.get("text", "")
                    if k == "user_final":
                        if path2_stt_final_is_drop_noise(t):
                            continue
                        self._set_ui_state("processing", "stt_final_received")
                        self._emit("STT final", t)
                    elif k == "user_final_noise_dropped" and _SHOW_STT_NOISE_DROPS:
                        self._emit("STT noise-drop", t)
                    elif k == "user_partial" and _SHOW_STT_PARTIALS:
                        self._set_ui_state("listening", "partial_transcript")
                        self._emit("STT partial", t[:120], newline=False)
                    elif k == "user_final_echo_skip":
                        self._emit("STT echo-skip", t)
                    elif k == "agent_tts":
                        self._set_ui_state("agent_speaking", "tts_text_enqueued")
                        self._emit("Agent → TTS", t)
                    elif k == "agent_fallback":
                        self._emit("Agent fallback", t)
                    elif k == "call_ended":
                        self._emit("System", f"call ended ({t}) — stopping mimic")
                        self.active = False
                        ws = self.ws
                        if ws:
                            try:
                                await ws.close()
                            except Exception:
                                pass
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if self.active:
                logger.info(
                    "Transcript debug WS not available (%s). "
                    "Path2 server needs PATH2_TRANSCRIPT_DEBUG_WS=1 (default).",
                    e,
                )

    async def start(self):
        if not AUDIO_READY:
            print("\n[Error] Dependencies missing (pyaudio, webrtcvad, audioop-lts).")
            return False

        try:
            self.audio = pyaudio.PyAudio()
            self.vad = webrtcvad.Vad(_VAD_MODE)

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
            self._open_session_log()
            self._set_ui_state(
                "listening",
                f"mic_on=True vad_mode={_VAD_MODE} vad_off={_VAD_OFF}",
            )

            asyncio.create_task(self._mic_loop())
            asyncio.create_task(self._speaker_loop())
            self._transcript_task = asyncio.create_task(self._transcript_debug_loop())

            return True
        except Exception as e:
            logger.error(f"Bridge connection failed: {e}")
            await self.stop()
            return False

    async def stop(self):
        self.active = False
        if self._transcript_task:
            self._transcript_task.cancel()
            try:
                await self._transcript_task
            except asyncio.CancelledError:
                pass
            self._transcript_task = None
        if self.ws:
            await self.ws.close()
            self.ws = None
        if self.mic:
            self.mic.stop_stream(); self.mic.close(); self.mic = None
        if self.speaker:
            self.speaker.stop_stream(); self.speaker.close(); self.speaker = None
        if self.audio:
            self.audio.terminate(); self.audio = None
        self._close_session_log()

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

                speech = _VAD_OFF or self.vad.is_speech(data, LOCAL_RATE)
                mic_muted = self._mic_is_muted()
                if mic_muted:
                    self._set_ui_state("waiting_playback", "mic_muted_by_echo_guard")
                elif speech:
                    self._set_ui_state("capturing", "sending_audio_to_bridge")
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
                    now = time.monotonic()
                    if now - self._last_you_print >= 0.7:
                        self._last_you_print = now
                        self._emit("Mic", "You → bridge / STT", newline=True)
                else:
                    self._set_ui_state("listening", "mic_open_waiting_for_speech")

                await asyncio.sleep(0)
        except Exception as e:
            if self.active: logger.debug(f"Mic error: {e}")

    async def _speaker_loop(self):
        """Receive from bridge -> decode -> play through speaker."""
        try:
            async for message in self.ws:
                if not self.active:
                    break
                data = json.loads(message)

                if data.get("event") == "media":
                    if not self._heard_agent:
                        self._heard_agent = True
                        self._set_ui_state("agent_speaking", "first_audio_chunk")
                    self._last_agent_audio_time = time.monotonic()

                    mulaw_payload = base64.b64decode(data["media"]["payload"])
                    pcm_8k = audioop.ulaw2lin(mulaw_payload, 2)
                    pcm_8k = audioop.mul(pcm_8k, 2, 2.0)

                    pcm_16k, self.up_state = audioop.ratecv(
                        pcm_8k, 2, 1, BRIDGE_RATE, LOCAL_RATE, self.up_state
                    )

                    self.speaker.write(pcm_16k)

                elif data.get("event") == "clear":
                    self._emit("Clear", "server sent clear — mic reopening")
                    if self.active:
                        self._set_ui_state("listening", "tts_done_ready_for_user")

        except Exception as e:
            if self.active: print(f"\n[Bridge Connection Error] {e}")
        finally:
            self.active = False
            self._set_ui_state("idle", "bridge_ws_closed")


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
                            print("  STT finals print as [STT final] when Path2 + debug WS enabled.")
                            print("  ORCHESTRATOR_SHOW_STT_PARTIALS=1 for partials. Type /chat to exit.\n")
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
