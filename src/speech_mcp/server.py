import sys
import os
import json
import time
import threading
import tempfile
import subprocess
import psutil
import importlib.util
from importlib import resources as pkg_resources
from typing import Dict, List, Union, Optional, Callable, Any
from pathlib import Path
import numpy as np
import soundfile as sf
import socket
import random
import string
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Tuple
import queue as _queue

from mcp.server.fastmcp import FastMCP
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData, INTERNAL_ERROR, INVALID_PARAMS
from mcp_ui_server import create_UIResource
from mcp_ui_server.core import UIResource
# Import the centralized logger
from speech_mcp.utils.logger import get_logger

# Get a logger for this module
logger = get_logger(__name__, component="server")

# =============================================================
# SSE sidecar (loopback HTTP server for UI state streaming)
# =============================================================
_sse_server: ThreadingHTTPServer | None = None
_sse_thread: threading.Thread | None = None
_sse_port: int | None = None
_sse_token: str | None = None
_sse_clients: list[tuple[_queue.Queue, object]] = []  # (queue, wfile)
_sse_lock = threading.Lock()

def _random_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

def _random_token(n: int = 24) -> str:
    return ''.join(random.choices(string.ascii_letters + string.digits, k=n))

def _sse_broadcast(event: str, data: dict) -> None:
    payload = f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8")
    with _sse_lock:
        stale: list[int] = []
        for idx, (q, wfile) in enumerate(_sse_clients):
            try:
                q.put(payload, block=False)
            except Exception:
                stale.append(idx)
        for idx in reversed(stale):
            _sse_clients.pop(idx)

class _SSEHandler(BaseHTTPRequestHandler):
    server_version = "SpeechSSE/1.0"
    protocol_version = "HTTP/1.1"

    def _set_cors(self):
        # CORS for iframe-origin requests
        origin = self.headers.get('Origin') or '*'
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Headers", "*, content-type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        # Private Network Access (Chromium/Electron)
        if self.headers.get('Access-Control-Request-Private-Network') == 'true':
            self.send_header("Access-Control-Allow-Private-Network", "true")

    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204)
        self._set_cors()
        self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()

    def do_GET(self):  # noqa: N802
        global _sse_token
        parsed = urlparse(self.path)
        if parsed.path == "/events":
            params = parse_qs(parsed.query)
            token = (params.get("token") or [""])[0]
            if not _sse_token or token != _sse_token:
                self.send_response(403)
                self._set_cors()
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self._set_cors()
            self.end_headers()
            q: _queue.Queue = _queue.Queue()
            with _sse_lock:
                _sse_clients.append((q, self.wfile))
            # Initial state
            try:
                state = state_manager.get_state()
                init = {"listening": bool(state.get("listening")), "speaking": bool(state.get("speaking")), "voice": state.get("voice_preference")}
                self.wfile.write(f"event: state\ndata: {json.dumps(init)}\n\n".encode("utf-8"))
                self.wfile.flush()
            except Exception:
                pass
            try:
                while True:
                    try:
                        payload = q.get(timeout=15)
                        self.wfile.write(payload)
                        self.wfile.flush()
                    except _queue.Empty:
                        try:
                            self.wfile.write(b":\n\n")
                            self.wfile.flush()
                        except Exception:
                            break
            except Exception:
                pass
            finally:
                with _sse_lock:
                    for idx, (qq, wf) in enumerate(list(_sse_clients)):
                        if qq is q:
                            _sse_clients.pop(idx)
                            break
            return
        elif parsed.path == "/state":
            st = state_manager.get_state()
            body = json.dumps({
                "listening": bool(st.get("listening")),
                "speaking": bool(st.get("speaking")),
                "voice": st.get("voice_preference"),
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._set_cors()
            self.end_headers()
            self.wfile.write(body)
            return
        elif parsed.path == "/speak":
            # GET fallback: run TTS asynchronously and return immediately
            params = parse_qs(parsed.query)
            text = (params.get("text") or [""])[0]
            def _bg():
                try:
                    _sse_broadcast("state", {"speaking": True, "listening": bool(speech_state.get("listening"))})
                    speak_text(text)
                finally:
                    _sse_broadcast("state", {"speaking": False, "listening": bool(speech_state.get("listening"))})
                    state_manager.update_state({"last_response": text}, persist=False)
            threading.Thread(target=_bg, daemon=True).start()
            body = json.dumps({"ok": True, "accepted": True}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self._set_cors()
            self.end_headers()
            try:
                self.wfile.write(body); self.wfile.flush()
            except Exception:
                pass
            return
        elif parsed.path == "/listen":
            # GET fallback: start one-shot listen in background and return accepted
            def _bg():
                try:
                    speech_state["listening"] = True
                    save_speech_state(speech_state, False)
                    _sse_broadcast("state", {"listening": True, "speaking": bool(speech_state.get("speaking"))})
                    text = listen_for_speech() or ""
                finally:
                    speech_state["listening"] = False
                    save_speech_state(speech_state, False)
                    _sse_broadcast("state", {"listening": False, "speaking": bool(speech_state.get("speaking"))})
                state_manager.update_state({"last_transcript": text}, persist=False)
                _sse_broadcast("transcriptFinal", {"text": text})
            threading.Thread(target=_bg, daemon=True).start()
            body = json.dumps({"ok": True, "accepted": True}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self._set_cors()
            self.end_headers()
            try:
                self.wfile.write(body); self.wfile.flush()
            except Exception:
                pass
            return
        elif parsed.path == "/stop":
            speech_state["listening"] = False
            speech_state["speaking"] = False
            save_speech_state(speech_state, False)
            _sse_broadcast("state", {"listening": False, "speaking": False})
            body = json.dumps({"ok": True}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self._set_cors()
            self.end_headers()
            try:
                self.wfile.write(body); self.wfile.flush()
            except Exception:
                pass
            return
        elif parsed.path == "/voice":
            params = parse_qs(parsed.query)
            voice = (params.get("voice") or [""])[0].strip()
            if voice:
                state_manager.update_state({"voice_preference": voice}, persist=True)
                try:
                    if tts_engine and hasattr(tts_engine, "set_voice"):
                        tts_engine.set_voice(voice)
                except Exception:
                    pass
            body = json.dumps({"ok": True, "voice": voice}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self._set_cors()
            self.end_headers()
            try:
                self.wfile.write(body); self.wfile.flush()
            except Exception:
                pass
            return
        else:
            self.send_response(404)
            self._set_cors()
            self.end_headers()

    def do_POST(self):  # noqa: N802
        parsed = urlparse(self.path)
        length = int(self.headers.get('Content-Length', '0') or '0')
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            payload = {}
        logger.info(f"HTTP POST {parsed.path}")
        if parsed.path == "/listen":
            try:
                speech_state["listening"] = True
                save_speech_state(speech_state, False)
                _sse_broadcast("state", {"listening": True, "speaking": bool(speech_state.get("speaking"))})
                text = listen_for_speech() or ""
            finally:
                speech_state["listening"] = False
                save_speech_state(speech_state, False)
                _sse_broadcast("state", {"listening": False, "speaking": bool(speech_state.get("speaking"))})
            state_manager.update_state({"last_transcript": text}, persist=False)
            _sse_broadcast("transcriptFinal", {"text": text})
            body = json.dumps({"transcript": text}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self._set_cors()
            self.end_headers()
            self.wfile.write(body)
            try:
                self.wfile.flush()
            except Exception:
                pass
            return
        if parsed.path == "/speak":
            # Async TTS: accept and return immediately
            text = str(payload.get("text") or "")
            def _bg():
                try:
                    _sse_broadcast("state", {"speaking": True, "listening": bool(speech_state.get("listening"))})
                    speak_text(text)
                finally:
                    _sse_broadcast("state", {"speaking": False, "listening": bool(speech_state.get("listening"))})
                state_manager.update_state({"last_response": text}, persist=False)
            threading.Thread(target=_bg, daemon=True).start()
            body = json.dumps({"ok": True, "accepted": True}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self._set_cors()
            self.end_headers()
            self.wfile.write(body)
            try:
                self.wfile.flush()
            except Exception:
                pass
            return
        if parsed.path == "/voice":
            voice = str(payload.get("voice") or "").strip()
            if voice:
                state_manager.update_state({"voice_preference": voice}, persist=True)
                try:
                    if tts_engine and hasattr(tts_engine, "set_voice"):
                        tts_engine.set_voice(voice)
                except Exception:
                    pass
            body = json.dumps({"ok": True, "voice": voice}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self._set_cors()
            self.end_headers()
            try:
                self.wfile.write(body)
                self.wfile.flush()
            except Exception:
                pass
            return
        if parsed.path == "/stop":
            speech_state["listening"] = False
            speech_state["speaking"] = False
            save_speech_state(speech_state, False)
            _sse_broadcast("state", {"listening": False, "speaking": False})
            body = json.dumps({"ok": True}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self._set_cors()
            self.end_headers()
            try:
                self.wfile.write(body)
                self.wfile.flush()
            except Exception:
                pass
            return
        self.send_response(404)
        self._set_cors()
        self.end_headers()

def _ensure_sse_sidecar() -> Tuple[str, str]:
    """Start SSE sidecar if not running. Returns (base_url, token)."""
    global _sse_server, _sse_thread, _sse_port, _sse_token
    if _sse_server is not None and _sse_port and _sse_token:
        base = f"http://127.0.0.1:{_sse_port}"
        # Log on subsequent calls as well so users can discover values
        logger.info(f"SSE sidecar already running: BASE={base} TOKEN={_sse_token}")
        return (base, _sse_token)
    _sse_port = _random_port()
    _sse_token = _random_token()
    _sse_server = ThreadingHTTPServer(("127.0.0.1", _sse_port), _SSEHandler)
    _sse_thread = threading.Thread(target=_sse_server.serve_forever, daemon=True)
    _sse_thread.start()
    base = f"http://127.0.0.1:{_sse_port}"
    logger.info(f"Started SSE sidecar at {base}")
    # Print explicit BASE/TOKEN lines for easy grep/curl
    logger.info(f"SSE BASE={base}")
    logger.info(f"SSE TOKEN={_sse_token}")
    return (base, _sse_token)

# Import centralized constants
from speech_mcp.constants import (
    SERVER_LOG_FILE,
    TRANSCRIPTION_FILE,
    SPEECH_TIMEOUT, ENV_TTS_VOICE
)

# Import state manager
from speech_mcp.state_manager import StateManager

# Import shared audio processor and speech recognition
from speech_mcp.audio_processor import AudioProcessor
from speech_mcp.speech_recognition import (
    initialize_speech_recognition as init_speech_recognition,
    transcribe_audio as transcribe_audio_file,
    start_streaming_transcription,
    add_streaming_audio_chunk,
    stop_streaming_transcription,
    get_current_streaming_transcription,
    is_streaming_active
)

mcp = FastMCP("speech")

# Define TTS engine variable
tts_engine = None

# Define initialize_kokoro_tts function before it's used
def initialize_kokoro_tts():
    """Initialize Kokoro TTS specifically"""
    global tts_engine
    
    try:
        # Import the Kokoro TTS adapter
        from speech_mcp.tts_adapters import KokoroTTS
        
        # Try to get voice preference from config or environment
        voice = None
        try:
            from speech_mcp.config import get_setting, get_env_setting
            
            # First check environment variable
            env_voice = get_env_setting(ENV_TTS_VOICE)
            if env_voice:
                voice = env_voice
            else:
                # Then check config file
                config_voice = get_setting("tts", "voice", None)
                if config_voice:
                    voice = config_voice
        except ImportError:
            pass
        
        # Initialize Kokoro with default or saved voice settings
        if voice:
            tts_engine = KokoroTTS(voice=voice, lang_code="a", speed=1.0)
        else:
            tts_engine = KokoroTTS(voice="af_heart", lang_code="a", speed=1.0)
        
        if tts_engine.is_initialized and tts_engine.kokoro_available:
            logger.info("Kokoro TTS initialized successfully")
            return True
        else:
            # If Kokoro initialization failed, set tts_engine to None so we'll try fallback later
            tts_engine = None
            logger.warning("Kokoro TTS initialization failed, will use fallback")
            return False
            
    except ImportError as e:
        logger.error(f"Kokoro TTS import error: {e}")
        return False
    except Exception as e:
        logger.error(f"Kokoro TTS initialization error: {e}")
        return False

# Initialize Kokoro TTS on server start (asynchronously)
logger.info("Starting asynchronous Kokoro TTS initialization...")
# Use a thread-safe variable to track initialization status
import threading
kokoro_init_lock = threading.Lock()
kokoro_init_status = {"initialized": False, "in_progress": True}

def async_kokoro_init():
    """Initialize Kokoro TTS in a background thread"""
    global kokoro_init_status
    try:
        # Attempt to initialize Kokoro
        result = initialize_kokoro_tts()
        
        # Update status with thread safety
        with kokoro_init_lock:
            kokoro_init_status["initialized"] = result
            kokoro_init_status["in_progress"] = False
        
        if result:
            logger.info("Async Kokoro TTS initialization completed successfully")
        else:
            logger.warning("Async Kokoro TTS initialization failed, will use fallback when needed")
    except Exception as e:
        # Update status with thread safety
        with kokoro_init_lock:
            kokoro_init_status["initialized"] = False
            kokoro_init_status["in_progress"] = False
        logger.error(f"Error during async Kokoro TTS initialization: {e}")

# Start the initialization in a background thread
kokoro_init_thread = threading.Thread(target=async_kokoro_init)
kokoro_init_thread.daemon = True
kokoro_init_thread.start()

# State management has been moved to StateManager class

# Save speech state using StateManager
def save_speech_state(state, create_response_file=False):
    try:
        # Update state in StateManager
        state_manager.update_state(state, persist=True)
        # UI signaling via files is deprecated; MCP UI clients should reflect state directly
    except Exception as e:
        logger.error(f"Error saving speech state: {e}")
        pass

# Initialize state manager
state_manager = StateManager.get_instance()
speech_state = state_manager.get_state()  # Get a copy of the current state

def initialize_speech_recognition():
    """Initialize speech recognition"""
    try:
        # Use the centralized speech recognition module
        try:
            # Get stt settings from config
            from speech_mcp.config import get_setting

            model_name = get_setting("stt", "model", "base")
            device = get_setting("stt", "device", "cpu")
            compute_type = get_setting("stt", "compute_type", "int8")
        except ImportError:
            model_name = "base"
            device = "cpu"
            compute_type = "int8"

        result = init_speech_recognition(model_name=model_name, device=device, compute_type=compute_type)
        return result
    except Exception:
        return False

def initialize_tts():
    """Initialize text-to-speech"""
    global tts_engine, kokoro_init_status
    
    if tts_engine is not None:
        return True
    
    # Check if Kokoro initialization is still in progress
    kokoro_in_progress = False
    with kokoro_init_lock:
        kokoro_in_progress = kokoro_init_status["in_progress"]
        kokoro_initialized = kokoro_init_status["initialized"]
    
    # If Kokoro initialization completed successfully in the background,
    # but tts_engine is not set yet, we need to initialize it now
    if not kokoro_in_progress and kokoro_initialized and tts_engine is None:
        logger.info("Kokoro was initialized asynchronously, but tts_engine is not set. Reinitializing...")
        if initialize_kokoro_tts():
            return True
    
    try:
        # Try to import the TTS adapters
        try:
            # First try to use the new adapter system
            from speech_mcp.tts_adapters import KokoroTTS, Pyttsx3TTS
            
            # Try to get voice preference from config or environment
            voice = None
            try:
                from speech_mcp.config import get_setting, get_env_setting
                
                # First check environment variable
                env_voice = get_env_setting(ENV_TTS_VOICE)
                if env_voice:
                    voice = env_voice
                else:
                    # Then check config file
                    config_voice = get_setting("tts", "voice", None)
                    if config_voice:
                        voice = config_voice
            except ImportError:
                pass
            
            # First try Kokoro (our primary TTS engine)
            try:
                # Only try Kokoro if it's not still initializing
                if not kokoro_in_progress:
                    # Initialize with default or saved voice settings
                    if voice:
                        tts_engine = KokoroTTS(voice=voice, lang_code="a", speed=1.0)
                    else:
                        tts_engine = KokoroTTS(voice="af_heart", lang_code="a", speed=1.0)
                    
                    if tts_engine.is_initialized:
                        return True
            except ImportError:
                pass
            except Exception:
                pass
            
            # Fall back to pyttsx3 adapter
            try:
                # Initialize with default or saved voice settings
                if voice and voice.startswith("pyttsx3:"):
                    tts_engine = Pyttsx3TTS(voice=voice, lang_code="en", speed=1.0)
                else:
                    tts_engine = Pyttsx3TTS(lang_code="en", speed=1.0)
                
                if tts_engine.is_initialized:
                    return True
            except ImportError:
                pass
            except Exception:
                pass
        
        except ImportError:
            pass
        
        # Direct fallback to pyttsx3 if adapters are not available
        try:
            import pyttsx3
            tts_engine = pyttsx3.init()
            return True
        except ImportError:
            return False
        except Exception:
            return False
            
    except Exception:
        return False

def record_audio():
    """Record audio from the microphone and return the audio data"""
    try:
        # Create an instance of the shared AudioProcessor
        audio_processor = AudioProcessor()
        
        # Use the AudioProcessor to record audio
        audio_file_path = audio_processor.record_audio()
        
        if not audio_file_path:
            raise Exception("Failed to record audio")
        
        return audio_file_path
    
    except Exception as e:
        raise Exception(f"Error recording audio: {str(e)}")

def record_audio_streaming():
    """Record audio using streaming transcription and return the transcription"""
    try:
        # Create AudioProcessor instance
        audio_processor = AudioProcessor()
        
        # Initialize speech recognition
        if not initialize_speech_recognition():
            raise Exception("Failed to initialize speech recognition")
        
        # Set up result storage and synchronization
        transcription_result = {"text": "", "metadata": {}}
        transcription_ready = threading.Event()
        
        # Define callbacks for streaming transcription
        def on_partial_transcription(text):
            # Log partial transcription
            logger.debug(f"Partial transcription: {text}")
            # Update state with partial transcription
            speech_state["last_transcript"] = text
            save_speech_state(speech_state, False)
        
        def on_final_transcription(text, metadata):
            # Log final transcription
            logger.info(f"Final transcription: {text}")
            # Store result and signal completion
            transcription_result["text"] = text
            transcription_result["metadata"] = metadata
            transcription_ready.set()
        
        # Start streaming transcription
        if not start_streaming_transcription(
            language="en",
            on_partial_transcription=on_partial_transcription,
            on_final_transcription=on_final_transcription
        ):
            raise Exception("Failed to start streaming transcription")
        
        # Start audio recording in streaming mode
        if not audio_processor.start_listening(
            streaming_mode=True,
            on_audio_chunk=add_streaming_audio_chunk
        ):
            stop_streaming_transcription()
            raise Exception("Failed to start audio recording")
        
        # Wait for transcription to complete (with timeout)
        max_wait_time = 600  # 10 minutes maximum
        if not transcription_ready.wait(max_wait_time):
            logger.warning("Transcription timeout reached")
        
        # Stop audio recording
        audio_processor.stop_listening()
        
        # If streaming is still active, stop it
        if is_streaming_active():
            text, metadata = stop_streaming_transcription()
            if not transcription_result["text"]:
                transcription_result["text"] = text
                transcription_result["metadata"] = metadata
        
        # Return the transcription
        return transcription_result["text"]
        
    except Exception as e:
        logger.error(f"Error in streaming audio recording: {str(e)}")
        raise Exception(f"Error recording audio: {str(e)}")

def transcribe_audio(audio_file_path):
    """Transcribe audio file using the speech recognition module"""
    try:
        logger.info(f"Starting transcription for audio file: {audio_file_path}")
        
        if not initialize_speech_recognition():
            logger.error("Failed to initialize speech recognition")
            raise Exception("Failed to initialize speech recognition")
        
        logger.info("Speech recognition initialized successfully")
        
        # Use the centralized speech recognition module
        try:
            logger.info("Calling transcribe_audio_file...")
            result = transcribe_audio_file(audio_file_path)
            logger.info(f"transcribe_audio_file returned: {type(result)}")
            logger.debug(f"transcribe_audio_file full result: {result}")
            
            if isinstance(result, tuple):
                transcription, metadata = result
                logger.info(f"Unpacked tuple result - transcription type: {type(transcription)}, metadata type: {type(metadata)}")
            else:
                transcription = result
                logger.info(f"Single value result - transcription type: {type(transcription)}")
        except Exception as e:
            logger.error(f"Error during transcribe_audio_file call: {str(e)}", exc_info=True)
            raise
        
        if not transcription:
            logger.error("Transcription failed or returned empty result")
            raise Exception("Transcription failed or returned empty result")
        
        logger.info(f"Transcription successful, length: {len(transcription)}")
        
        # Clean up the temporary file
        try:
            os.unlink(audio_file_path)
            logger.info(f"Cleaned up temporary audio file: {audio_file_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary audio file: {str(e)}")
            pass
        
        return transcription
    
    except Exception as e:
        logger.error(f"Error in transcribe_audio: {str(e)}", exc_info=True)
        raise Exception(f"Error transcribing audio: {str(e)}")

def speak_text(text):
    """Speak text using TTS engine"""
    global tts_engine, kokoro_init_status
    
    if not text:
        raise McpError(
            ErrorData(
                INVALID_PARAMS,
                "No text provided to speak."
            )
        )
    
    # Set speaking state
    speech_state["speaking"] = True
    speech_state["last_response"] = text
    
    # Save state but don't create response file - we'll handle TTS directly
    save_speech_state(speech_state, False)
    
    try:
        # Check if Kokoro initialization is in progress
        kokoro_in_progress = False
        with kokoro_init_lock:
            kokoro_in_progress = kokoro_init_status["in_progress"]
            kokoro_initialized = kokoro_init_status["initialized"]
        
        # If Kokoro is still initializing and we don't have a TTS engine yet,
        # we'll use a fallback immediately rather than waiting
        if kokoro_in_progress and tts_engine is None:
            logger.info("Kokoro initialization still in progress, using fallback TTS for now")
            # Try to initialize a fallback TTS engine
            try:
                from speech_mcp.tts_adapters import Pyttsx3TTS
                tts_engine = Pyttsx3TTS(lang_code="en", speed=1.0)
            except Exception:
                # If fallback initialization fails, we'll simulate speech
                pass
        
        # Use the already initialized TTS engine or initialize if needed
        if tts_engine is None:
            # First check if Kokoro initialization completed successfully
            if not kokoro_in_progress and kokoro_initialized:
                # Kokoro was initialized successfully in the background
                logger.info("Using Kokoro TTS that was initialized asynchronously")
                # No need to initialize again, the global tts_engine should be set
            else:
                # If Kokoro initialization failed or is still in progress, try the general TTS initialization
                if not initialize_tts():
                    # If all TTS initialization fails, simulate speech with a delay
                    speaking_duration = len(text) * 0.05  # 50ms per character
                    time.sleep(speaking_duration)
                    
                    # Update state
                    speech_state["speaking"] = False
                    save_speech_state(speech_state, False)
                    return f"Simulated speaking: {text}"
        
        # Use TTS engine to speak text directly without going through the UI
        tts_start = time.time()
        
        # Use the appropriate method based on the TTS engine type
        if hasattr(tts_engine, 'speak'):
            # Use the speak method (our adapter system or Kokoro adapter)
            result = tts_engine.speak(text)
        elif hasattr(tts_engine, 'say'):
            # Use pyttsx3 directly
            tts_engine.say(text)
            tts_engine.runAndWait()
        else:
            # Simulate speech as fallback
            speaking_duration = len(text) * 0.05  # 50ms per character
            time.sleep(speaking_duration)
        
        # Update state
        speech_state["speaking"] = False
        save_speech_state(speech_state, False)
        
        return f"Spoke: {text}"
    
    except Exception as e:
        # Update state on error
        speech_state["speaking"] = False
        save_speech_state(speech_state, False)
        
        # Simulate speech with a delay as fallback
        speaking_duration = len(text) * 0.05  # 50ms per character
        time.sleep(speaking_duration)
        
        return f"Error speaking text: {str(e)}"

def listen_for_speech() -> str:
    """Listen for speech and return transcription"""
    global speech_state
    
    # Set listening state
    speech_state["listening"] = True
    save_speech_state(speech_state, False)
    
    try:
        # Use streaming transcription
        transcription = record_audio_streaming()
        
        # Update state
        speech_state["listening"] = False
        speech_state["last_transcript"] = transcription
        save_speech_state(speech_state, False)
        
        return transcription
    
    except Exception as e:
        # Update state on error
        speech_state["listening"] = False
        save_speech_state(speech_state, False)
        
        raise McpError(
            ErrorData(
                INTERNAL_ERROR,
                f"Error during speech recognition: {str(e)}"
            )
        )

class VoiceInstance:
    """Manages a single Kokoro TTS voice instance"""
    def __init__(self, voice_id: str):
        from speech_mcp.tts_adapters import KokoroTTS
        self.engine = None
        self.fallback_engine = None
        self.voice_id = voice_id
        
        logger.info(f"Initializing VoiceInstance for voice: {voice_id}")
        
        # Try to initialize Kokoro
        try:
            logger.info("Attempting to initialize Kokoro TTS...")
            self.engine = KokoroTTS(voice=voice_id, lang_code="a", speed=1.0)
            logger.info(f"Kokoro TTS initialized: is_initialized={self.engine.is_initialized}, kokoro_available={self.engine.kokoro_available}")
            if self.engine.is_initialized and self.engine.kokoro_available:
                logger.info("Kokoro TTS initialization successful")
                return  # Successfully initialized
            else:
                logger.warning("Kokoro TTS initialization incomplete")
        except Exception as e:
            logger.warning(f"Failed to initialize Kokoro TTS for voice {voice_id}: {str(e)}")
            self.engine = None
            
        # Try fallback if Kokoro failed
        try:
            logger.info("Attempting to initialize fallback TTS...")
            from speech_mcp.tts_adapters import Pyttsx3TTS
            self.fallback_engine = Pyttsx3TTS(lang_code="en", speed=1.0)
            logger.info(f"Fallback TTS initialized: is_initialized={self.fallback_engine.is_initialized}")
            if self.fallback_engine.is_initialized:
                logger.info("Fallback TTS initialization successful")
                return  # Successfully initialized fallback
            else:
                logger.warning("Fallback TTS initialization incomplete")
        except Exception as e:
            logger.warning(f"Failed to initialize fallback TTS: {str(e)}")
            self.fallback_engine = None
        
        # If both Kokoro and fallback failed, raise exception
        if self.engine is None and self.fallback_engine is None:
            logger.error("Both Kokoro and fallback TTS initialization failed")
            raise Exception(f"Failed to initialize any TTS engine for voice {voice_id}")
        
        # If initialization succeeded but voice isn't available, show available voices
        available_voices = []
        if self.engine and self.engine.kokoro_available:
            available_voices = self.engine.get_available_voices()
            logger.info(f"Got {len(available_voices)} available voices from Kokoro")
        elif self.fallback_engine:
            available_voices = self.fallback_engine.get_available_voices()
            logger.info(f"Got {len(available_voices)} available voices from fallback")
            
        if available_voices and voice_id not in available_voices:
            logger.warning(f"Requested voice {voice_id} not found in available voices")
            voice_categories = {
                "American Female": [v for v in available_voices if v.startswith("af_")],
                "American Male": [v for v in available_voices if v.startswith("am_")],
                "British Female": [v for v in available_voices if v.startswith("bf_")],
                "British Male": [v for v in available_voices if v.startswith("bm_")],
                "Other English": [v for v in available_voices if v.startswith(("ef_", "em_"))],
                "French": [v for v in available_voices if v.startswith("ff_")],
                "Hindi": [v for v in available_voices if v.startswith(("hf_", "hm_"))],
                "Italian": [v for v in available_voices if v.startswith(("if_", "im_"))],
                "Japanese": [v for v in available_voices if v.startswith(("jf_", "jm_"))],
                "Portuguese": [v for v in available_voices if v.startswith(("pf_", "pm_"))],
                "Chinese": [v for v in available_voices if v.startswith(("zf_", "zm_"))]
            }
            
            # Build error message
            error_msg = [f"Voice '{voice_id}' not found. Available voices:"]
            for category, voices in voice_categories.items():
                if voices:  # Only show categories that have voices
                    error_msg.append(f"\n{category}:")
                    error_msg.append("  " + ", ".join(sorted(voices)))
            
            raise Exception("\n".join(error_msg))
        
    def generate_audio(self, text: str, output_path: str) -> bool:
        """Generate audio for the given text"""
        logger.info(f"Generating audio for text: '{text[:50]}...' with voice {self.voice_id}")
        
        # Try Kokoro first
        if self.engine and self.engine.kokoro_available:
            try:
                logger.info("Attempting to generate audio with Kokoro TTS...")
                result = self.engine.save_to_file(text, output_path)
                if result:
                    logger.info("Successfully generated audio with Kokoro TTS")
                    return True
                else:
                    logger.warning("Kokoro TTS save_to_file returned False")
            except Exception as e:
                logger.warning(f"Kokoro TTS failed to generate audio: {str(e)}")
        else:
            logger.info("Kokoro TTS not available for audio generation")
        
        # Try fallback if available
        if self.fallback_engine:
            try:
                logger.info("Attempting to generate audio with fallback TTS...")
                result = self.fallback_engine.save_to_file(text, output_path)
                if result:
                    logger.info("Successfully generated audio with fallback TTS")
                    return True
                else:
                    logger.warning("Fallback TTS save_to_file returned False")
            except Exception as e:
                logger.warning(f"Fallback TTS failed to generate audio: {str(e)}")
        else:
            logger.info("Fallback TTS not available for audio generation")
        
        # If both failed, raise exception with available voices info
        logger.error("Both Kokoro and fallback TTS failed to generate audio")
        
        # Get available voices from whichever engine is working
        available_voices = []
        if self.engine and self.engine.kokoro_available:
            available_voices = self.engine.get_available_voices()
            logger.info(f"Got {len(available_voices)} available voices from Kokoro")
        elif self.fallback_engine:
            available_voices = self.fallback_engine.get_available_voices()
            logger.info(f"Got {len(available_voices)} available voices from fallback")
        
        voice_categories = {
            "American Female": [v for v in available_voices if v.startswith("af_")],
            "American Male": [v for v in available_voices if v.startswith("am_")],
            "British Female": [v for v in available_voices if v.startswith("bf_")],
            "British Male": [v for v in available_voices if v.startswith("bm_")],
            "Other English": [v for v in available_voices if v.startswith(("ef_", "em_"))],
            "French": [v for v in available_voices if v.startswith("ff_")],
            "Hindi": [v for v in available_voices if v.startswith(("hf_", "hm_"))],
            "Italian": [v for v in available_voices if v.startswith(("if_", "im_"))],
            "Japanese": [v for v in available_voices if v.startswith(("jf_", "jm_"))],
            "Portuguese": [v for v in available_voices if v.startswith(("pf_", "pm_"))],
            "Chinese": [v for v in available_voices if v.startswith(("zf_", "zm_"))]
        }
        
        error_msg = [f"Failed to generate audio with voice '{self.voice_id}'. Available voices:"]
        for category, voices in voice_categories.items():
            if voices:
                error_msg.append(f"\n{category}:")
                error_msg.append("  " + ", ".join(sorted(voices)))
        
        raise Exception("\n".join(error_msg))

class VoiceManager:
    """Manages multiple voice instances"""
    def __init__(self):
        self._voices: Dict[str, VoiceInstance] = {}
        
    def get_voice(self, voice_id: str) -> VoiceInstance:
        if voice_id not in self._voices:
            self._voices[voice_id] = VoiceInstance(voice_id)
        return self._voices[voice_id]

# Global voice manager
voice_manager = VoiceManager()


@mcp.tool()
def panel_ui() -> list[UIResource]:
    """
    UI tool: returns the unified Speech panel UI.
    All actions (listen/speak/voice) are driven inside the iframe via the loopback sidecar.
    """
    html = _full_panel_html("")
    panel = _create_ui_resource("ui://speech/panel", html, min_height=342)
    return [panel]

@mcp.tool()
def listen() -> str:
    """
    Start listening via streaming recognition and return the final transcript.
    """
    try:
        text = listen_for_speech()
        return text or ""
    except Exception as e:
        return f"ERROR: {e}"


@mcp.tool()
def speak(text: str) -> str:
    """
    Speak the provided text using the configured TTS engine.
    """
    try:
        return speak_text(text or "")
    except Exception as e:
        return f"ERROR: {e}"


@mcp.tool()
def set_voice(voice: str) -> str:
    """
    Set the preferred voice for TTS.
    """
    try:
        voice = (voice or "").strip()
        state_manager.update_state({"voice_preference": voice}, persist=True)
        try:
            if tts_engine and hasattr(tts_engine, "set_voice"):
                tts_engine.set_voice(voice)
        except Exception:
            pass
        return f"OK: voice={voice}"
    except Exception as e:
        return f"ERROR: {e}"

 

 

def _create_ui_resource(uri: str, html_string: str, min_height: int = 342) -> Dict[str, Any]:
    """Create a UIResource-like dictionary consumable by MCP UI clients.

    The structure mirrors the example in mcp-ui, returning a dict with keys:
    - uri: unique resource URI (e.g., ui://speech/controls)
    - content: { type: "rawHtml", htmlString: "..." }
    - encoding: "text"
    """
    res = create_UIResource({
            "uri": uri,
            "content": {
                "type": "rawHtml",
                "htmlString": html_string,
                # Ensure the client allocates at least this much vertical space
                "height": min_height,
            },
            "encoding": "text",
        })
    return res

def _full_panel_html(extra_inline_js: str = "") -> str:
    from importlib import resources as pkg
    s = state_manager.get_state()
    v = {
        "listening": "true" if s.get("listening") else "false",
        "speaking": "true" if s.get("speaking") else "false",
        "voice_pref": s.get("voice_preference") or "(default)",
        "last_transcript": (s.get("last_transcript") or "").replace("<", "&lt;"),
        "last_response": (s.get("last_response") or "").replace("<", "&lt;"),
    }
    # Prefer bundled HTML if present, else inject CSS/JS into template
    bundle_path = pkg.files("speech_mcp.resources.ui").joinpath("panel.bundled.html")
    if bundle_path.is_file():
        html = bundle_path.read_text(encoding="utf-8")
    else:
        tpl = pkg.files("speech_mcp.resources.ui").joinpath("panel.html").read_text(encoding="utf-8")
        css = pkg.files("speech_mcp.resources.ui").joinpath("panel.css").read_text(encoding="utf-8")
        js = pkg.files("speech_mcp.resources.ui").joinpath("panel.js").read_text(encoding="utf-8")
        html = tpl.replace("{{CSS}}", f"<style>{css}</style>").replace("{{JS}}", f"<script>{js}</script>")
    # If extra JS provided, inject it just before closing </script> or append a new script tag
    injected = (html
        .replace("{{listening}}", v["listening"]) 
        .replace("{{speaking}}", v["speaking"]) 
        .replace("{{voice_pref}}", v["voice_pref"]) 
        .replace("{{last_transcript}}", v["last_transcript"]) 
        .replace("{{last_response}}", v["last_response"]) )
    # SSE sidecar injection disabled (MCP-UI host should manage state/updates)
    if extra_inline_js.strip():
        if "</script>\n </div>\n" in injected:
            injected = injected.replace("</script>\n </div>\n", f"{extra_inline_js}\n  </script>\n </div>\n")
        else:
            injected = injected.replace("</div>\n", f"  <script>\n{extra_inline_js}\n  </script>\n</div>\n")
    return injected

def _listen_panel_html(extra_inline_js: str = "") -> str:
    from importlib import resources as pkg
    s = state_manager.get_state()
    v = {
        "last_transcript": (s.get("last_transcript") or "").replace("<", "&lt;"),
    }
    tpl_path = pkg.files("speech_mcp.resources.ui").joinpath("listen_panel.bundled.html")
    if tpl_path.is_file():
        tpl = tpl_path.read_text(encoding="utf-8")
    else:
        # Fallback to base
        tpl = _full_panel_html()
    html = tpl.replace("{{last_transcript}}", v["last_transcript"]) \
             .replace("{{CSS}}", "") \
             .replace("{{JS}}", "")
    if extra_inline_js.strip():
        if "{{JS}}" in tpl:
            html = tpl.replace("{{JS}}", f"<script>{extra_inline_js}</script>")
        else:
            html = html + f"\n<script>\n{extra_inline_js}\n</script>\n"
    return html

def _speak_panel_html(extra_inline_js: str = "") -> str:
    from importlib import resources as pkg
    s = state_manager.get_state()
    v = {
        "last_response": (s.get("last_response") or "").replace("<", "&lt;"),
    }
    tpl_path = pkg.files("speech_mcp.resources.ui").joinpath("speak_panel.bundled.html")
    if tpl_path.is_file():
        tpl = tpl_path.read_text(encoding="utf-8")
    else:
        # Fallback to base
        tpl = _full_panel_html()
    html = tpl.replace("{{last_response}}", v["last_response"]) \
             .replace("{{CSS}}", "") \
             .replace("{{JS}}", "")
    if extra_inline_js.strip():
        if "{{JS}}" in tpl:
            html = tpl.replace("{{JS}}", f"<script>{extra_inline_js}</script>")
        else:
            html = html + f"\n<script>\n{extra_inline_js}\n</script>\n"
    return html

def _render_status_html() -> str:
    """Render current status from state into a small HTML panel."""
    state = state_manager.get_state()
    listening = "true" if state.get("listening") else "false"
    speaking = "true" if state.get("speaking") else "false"
    last_transcript = (state.get("last_transcript") or "").replace("<", "&lt;")
    last_response = (state.get("last_response") or "").replace("<", "&lt;")
    voice_pref = state.get("voice_preference") or "(default)"
    return f"""
    <div style=\"padding: 16px; font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial;\">
      <h3 style=\"margin: 0 0 8px;\">Speech Status</h3>
      <div style=\"font-size: 14px; line-height: 1.5;\">
        <div><strong>Listening:</strong> {listening}</div>
        <div><strong>Speaking:</strong> {speaking}</div>
        <div><strong>Voice:</strong> {voice_pref}</div>
        <div style=\"margin-top: 8px;\"><strong>Last transcript</strong></div>
        <pre style=\"white-space: pre-wrap; background:#f8f9fa; padding:8px; border:1px solid #e9ecef; border-radius:6px;\">{last_transcript}</pre>
        <div style=\"margin-top: 8px;\"><strong>Last response</strong></div>
        <pre style=\"white-space: pre-wrap; background:#f8f9fa; padding:8px; border:1px solid #e9ecef; border-radius:6px;\">{last_response}</pre>
      </div>
    </div>
    """

def _controls_html() -> str:
    """Interactive controls panel with intent buttons/dropdowns.

    The client should forward postMessage({ type: 'intent', payload: { intent, params } })
    to the server tool `ui_intent(intent: str, params: dict)`.
    """
    return """
    <div style=\"padding: 16px; font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial;\">
      <h2 style=\"margin: 0 0 8px;\">Speech Controls</h2>
      <p style=\"margin:0 0 12px; color:#444;\">Use these controls to start listening, speak text, and set the voice.</p>

      <div style=\"display:flex; gap:8px; flex-wrap: wrap;\">
        <button onclick=\"sendIntent('start_listening', {})\" style=\"background:#007cba; color:#fff; padding:8px 12px; border:none; border-radius:6px; cursor:pointer;\">Start Listening</button>
        <button onclick=\"promptSpeak()\" style=\"background:#28a745; color:#fff; padding:8px 12px; border:none; border-radius:6px; cursor:pointer;\">Speak Text…</button>
        <button onclick=\"sendIntent('stop', {})\" style=\"background:#6c757d; color:#fff; padding:8px 12px; border:none; border-radius:6px; cursor:pointer;\">Stop</button>
      </div>

      <div style=\"margin-top:12px;\">
        <label for=\"voiceSelect\" style=\"font-size: 14px;\">Voice</label><br />
        <select id=\"voiceSelect\" onchange=\"onVoiceChange(this.value)\" style=\"margin-top:4px; padding:6px 8px; border-radius:6px; border:1px solid #ced4da;\">
          <option value=\"af_heart\">af_heart (default)</option>
          <option value=\"am_michael\">am_michael</option>
          <option value=\"bm_daniel\">bm_daniel</option>
          <option value=\"bf_emma\">bf_emma</option>
          <option value=\"ff_siwis\">ff_siwis</option>
        </select>
      </div>

      <div id=\"uiStatus\" style=\"margin-top:12px; font-size:13px; color:#555;\"></div>
    </div>

    <script>
      function sendIntent(intent, params) {
        const status = document.getElementById('uiStatus');
        if (status) {
          status.textContent = `Intent: ${intent}  Params: ${JSON.stringify(params)}`;
        }
        if (window.parent) {
          window.parent.postMessage({ type: 'intent', payload: { intent, params } }, '*');
        }
      }
      function onVoiceChange(voice) { sendIntent('set_voice', { voice }); }
      function promptSpeak() {
        const text = window.prompt('Text to speak');
        if (text && text.trim()) {
          sendIntent('speak', { text: text.trim() });
        }
      }
    </script>
    """

def _mini_widget_html() -> str:
    """Render a compact status + quick controls widget.

    Shows listening/speaking state and offers quick Start/Stop and Speak actions.
    Includes size-change postMessages for host autosizing.
    """
    state = state_manager.get_state()
    listening = "true" if state.get("listening") else "false"
    speaking = "true" if state.get("speaking") else "false"
    voice_pref = state.get("voice_preference") or "(default)"
    return f"""
    <div style=\"font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; display:flex; align-items:center; gap:10px; padding:8px 10px; border:1px solid #e5e7eb; border-radius:10px; background:#fff; color:#111827;\">
      <div style=\"display:flex; align-items:center; gap:8px;\">
        <span style=\"display:inline-flex; align-items:center; gap:6px;\">
          <span title=\"Listening\" style=\"width:10px; height:10px; border-radius:50%; background:{'#10b981' if listening=='true' else '#d1d5db'}; display:inline-block;\"></span>
          <span style=\"font-size:12px; color:#374151;\">Listening</span>
        </span>
        <span style=\"display:inline-flex; align-items:center; gap:6px;\">
          <span title=\"Speaking\" style=\"width:10px; height:10px; border-radius:50%; background:{'#3b82f6' if speaking=='true' else '#d1d5db'}; display:inline-block;\"></span>
          <span style=\"font-size:12px; color:#374151;\">Speaking</span>
        </span>
        <span style=\"font-size:12px; color:#6b7280;\">Voice: {voice_pref}</span>
      </div>
      <div style=\"margin-left:auto; display:flex; gap:6px;\">
        <button onclick=\"sendIntent('start_listening', {{}})\" style=\"background:#10b981; color:#fff; border:none; border-radius:8px; padding:6px 10px; font-size:12px; cursor:pointer;\">Listen</button>
        <button onclick=\"promptSpeak()\" style=\"background:#3b82f6; color:#fff; border:none; border-radius:8px; padding:6px 10px; font-size:12px; cursor:pointer;\">Speak…</button>
        <button onclick=\"sendIntent('stop', {{}})\" style=\"background:#6b7280; color:#fff; border:none; border-radius:8px; padding:6px 10px; font-size:12px; cursor:pointer;\">Stop</button>
      </div>
      <div id=\"uiStatus\" style=\"display:none\"></div>
    </div>
    <script>
      function postSize() {{
        const h = document.documentElement.scrollHeight;
        const w = document.documentElement.scrollWidth;
        const payload = {{ height: h, width: w }};
        if (window.parent) {{
          window.parent.postMessage({{ type: 'ui-size-change', payload }}, '*');
        }}
      }}
      let rafScheduled = false;
      function scheduleSize() {{
        if (rafScheduled) return;
        rafScheduled = true;
        requestAnimationFrame(() => {{ rafScheduled = false; postSize(); }});
      }}
      if ('ResizeObserver' in window) {{
        const ro = new ResizeObserver(() => scheduleSize());
        ro.observe(document.documentElement);
        ro.observe(document.body);
      }} else {{ window.addEventListener('resize', scheduleSize); }}
      document.addEventListener('DOMContentLoaded', scheduleSize);
      window.addEventListener('load', scheduleSize);
      setTimeout(scheduleSize, 0);

      function sendIntent(intent, params) {{
        const status = document.getElementById('uiStatus');
        if (status) {{ status.textContent = `Intent: ${'{'}intent{'}'} — ${'{'}JSON.stringify(params){'}'}`; }}
        if (window.parent) {{
          window.parent.postMessage({{ type: 'intent', payload: {{ intent, params }} }}, '*');
        }}
      }}
      function promptSpeak() {{
        const text = window.prompt('Text to speak');
        if (text && text.trim()) {{ sendIntent('speak', {{ text: text.trim() }}); }}
      }}
    </script>
    """

 

 


 

@mcp.tool()
def transcribe(file_path: str, include_timestamps: bool = False, detect_speakers: bool = False) -> str:
    """
    Transcribe an audio or video file to text.
    
    This tool uses faster-whisper to transcribe speech from audio/video files.
    Supports various formats including mp3, wav, mp4, etc.
    
    The transcription is saved to two files:
    - {input_name}.transcript.txt: Contains the transcription text (with timestamps/speakers if requested)
    - {input_name}.metadata.json: Contains metadata about the transcription process
    
    Args:
        file_path: Path to the audio or video file to transcribe
        include_timestamps: Whether to include word-level timestamps (default: False)
        detect_speakers: Whether to attempt speaker detection (default: False)
        
    Returns:
        A message indicating where the transcription was saved
    """
    try:
        # Initialize speech recognition if not already done
        import os
        import json
        from pathlib import Path
        
        # Check if file exists
        if not os.path.exists(file_path):
            return "ERROR: File not found."
            
        # Get file extension and create output paths
        input_path = Path(file_path)
        transcript_path = input_path.with_suffix('.transcript.txt')
        metadata_path = input_path.with_suffix('.metadata.json')
        
        # Get file extension
        ext = input_path.suffix.lower()
        
        # List of supported formats
        audio_formats = {'.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg'}
        video_formats = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
        
        if ext not in audio_formats and ext not in video_formats:
            return f"ERROR: Unsupported file format '{ext}'. Supported formats: {', '.join(sorted(audio_formats | video_formats))}"
        
        # For video files, we'll extract the audio first
        temp_audio = None
        if ext in video_formats:
            try:
                import tempfile
                from subprocess import run, PIPE
                
                # Create temporary file for audio
                temp_dir = tempfile.gettempdir()
                temp_audio = os.path.join(temp_dir, 'temp_audio.wav')
                
                # Use ffmpeg to extract audio with progress and higher priority
                logger.info(f"Extracting audio from video file: {file_path}")
                cmd = ['nice', '-n', '-10', 'ffmpeg', '-i', str(file_path), '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-y', '-v', 'warning', '-stats', '-threads', str(os.cpu_count()), temp_audio]
                logger.info(f"Running ffmpeg command: {' '.join(cmd)}")
                
                start_time = time.time()
                result = run(cmd, stdout=PIPE, stderr=PIPE)
                duration = time.time() - start_time
                
                logger.info(f"Audio extraction completed in {duration:.2f}s")
                
                if result.returncode != 0:
                    error = result.stderr.decode()
                    logger.error(f"ffmpeg error: {error}")
                    return f"ERROR: Failed to extract audio from video: {error}"
                    
                # Get the size of the extracted audio
                audio_size = os.path.getsize(temp_audio)
                logger.info(f"Extracted audio size: {audio_size / 1024 / 1024:.2f}MB")
                
                # Update file_path to use the extracted audio
                file_path = temp_audio
                
            except Exception as e:
                return f"ERROR: Failed to process video file: {str(e)}"
        
        if not initialize_speech_recognition():
            return "ERROR: Failed to initialize speech recognition."
            
        # Use the centralized speech recognition module
        try:
            # First try without timestamps/speakers for compatibility
            transcription, metadata = transcribe_audio_file(file_path)
            
            # If that worked and user requested timestamps/speakers, try again with those options
            if transcription and (include_timestamps or detect_speakers):
                try:
                    enhanced_transcription, enhanced_metadata = transcribe_audio_file(
                        file_path, 
                        include_timestamps=include_timestamps,
                        detect_speakers=detect_speakers
                    )
                    if enhanced_transcription:
                        transcription = enhanced_transcription
                        metadata = enhanced_metadata
                except Exception:
                    # If enhanced transcription fails, we'll keep the basic transcription
                    pass
        except Exception as e:
            return f"ERROR: Transcription failed: {str(e)}"
        
        # Clean up temporary audio file if it was created
        if temp_audio and os.path.exists(temp_audio):
            try:
                os.remove(temp_audio)
            except Exception:
                pass
        
        if not transcription:
            return "ERROR: Transcription failed or returned empty result."
        
        # Save the transcription and metadata
        try:
            # Save transcription
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(transcription)
            
            # Save metadata
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # Return success message with file locations
            msg = f"Transcription complete!\n\n"
            msg += f"Transcript saved to: {transcript_path}\n"
            msg += f"Metadata saved to: {metadata_path}\n\n"
            
            if detect_speakers and metadata.get('speakers'):
                msg += "The transcript includes speaker detection and timestamps.\n"
                msg += f"Detected {len(metadata.get('speakers', {}))} speakers\n"
                msg += f"Speaker changes: {metadata.get('speaker_changes', 0)}\n"
            elif include_timestamps and metadata.get('timestamps'):
                msg += "The transcript includes timestamps for each segment.\n"
            
            # Add some metadata to the message
            msg += f"\nDuration: {metadata.get('duration', 'unknown')} seconds\n"
            if metadata.get('language'):
                msg += f"Language: {metadata.get('language', 'unknown')} "
                if metadata.get('language_probability'):
                    msg += f"(probability: {metadata.get('language_probability', 0):.2f})\n"
            if metadata.get('time_taken'):
                msg += f"Processing time: {metadata.get('time_taken', 0):.2f} seconds"
            
            return msg
            
        except Exception as e:
            return f"ERROR: Failed to save transcription files: {str(e)}"
            
    except Exception as e:
        return f"ERROR: Failed to transcribe file: {str(e)}"

@mcp.tool()
def narrate(text: Optional[str] = None, text_file_path: Optional[str] = None, output_path: str = None) -> str:
    """
    Convert text to speech and save as an audio file.
    
    This will use the configured TTS engine to generate speech from text
    and save it to the specified output path.
    
    Args:
        text: The text to convert to speech (optional if text_file_path is provided)
        text_file_path: Path to a text file containing the text to narrate (optional if text is provided)
        output_path: Path where to save the audio file (.wav)
        
    Returns:
        A message indicating success or failure of the operation.
    """
    import os
    global tts_engine

    try:
        # Parameter validation
        if not output_path:
            return "ERROR: output_path is required"
        
        if text is None and text_file_path is None:
            return "ERROR: Either text or text_file_path must be provided"
        
        if text is not None and text_file_path is not None:
            return "ERROR: Cannot provide both text and text_file_path"
        
        # If text_file_path is provided, read the text from file
        if text_file_path is not None:
            try:
                if not os.path.exists(text_file_path):
                    return f"ERROR: Text file not found: {text_file_path}"
                with open(text_file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except Exception as e:
                return f"ERROR: Failed to read text file: {str(e)}"

        # Initialize TTS if needed
        if tts_engine is None and not initialize_tts():
            return "ERROR: Failed to initialize text-to-speech engine."

        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Use the adapter's save_to_file method
        if tts_engine.save_to_file(text, output_path):
            # Verify the file was created
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                if file_size > 0:
                    return f"Successfully saved speech to {output_path} ({file_size} bytes)"
                else:
                    os.unlink(output_path)
                    return f"ERROR: Generated file is empty: {output_path}"
            else:
                return f"ERROR: Failed to generate speech file: {output_path} was not created"
        else:
            # If save_to_file failed, clean up any partial file
            if os.path.exists(output_path):
                os.unlink(output_path)
            return "ERROR: Failed to save speech to file"

    except Exception as e:
        # Clean up any partial file
        try:
            if os.path.exists(output_path):
                os.unlink(output_path)
        except Exception:
            pass
        return f"ERROR: Failed to generate speech file: {str(e)}"

def parse_markdown_script(script: str) -> List[Dict]:
    """Parse the markdown-format script into segments"""
    segments = []
    current_segment = None
    
    for line in script.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('[') and line.endswith(']'):
            # New speaker definition
            if current_segment:
                current_segment["text"] = current_segment["text"].strip()
                segments.append(current_segment)
            
            # Parse speaker and voice
            content = line[1:-1]
            speaker, voice = content.split(':')
            current_segment = {
                "speaker": speaker.strip(),
                "voice": voice.strip(),
                "text": ""
            }
        elif line.startswith('{pause:') and line.endswith('}'):
            # Parse pause duration
            pause = float(line[7:-1])
            if current_segment:
                current_segment["pause_after"] = pause
        elif current_segment is not None:
            # Add text to current segment
            current_segment["text"] += line + "\n"
    
    # Add final segment
    if current_segment:
        current_segment["text"] = current_segment["text"].strip()
        segments.append(current_segment)
    
    return segments

@mcp.tool()
def narrate_conversation(
    script: Union[str, Dict],
    output_path: str,
    script_format: str = "json",
    temp_dir: Optional[str] = None
) -> str:
    """
    Generate a multi-speaker conversation audio file using multiple Kokoro TTS instances.
    
    Args:
        script: Either a JSON string/dict, a path to a script file, or a markdown-formatted script
        output_path: Path where to save the final audio file (.wav)
        script_format: Format of the script ("json" or "markdown")
        temp_dir: Optional directory for temporary files (default: system temp)
    
    Script Format Examples:
    
    JSON:
    {
        "conversation": [
            {
                "speaker": "narrator",
                "voice": "en_joe",
                "text": "Once upon a time...",
                "pause_after": 1.0
            },
            {
                "speaker": "alice", 
                "voice": "en_rachel",
                "text": "Hello there!",
                "pause_after": 0.5
            }
        ]
    }
    
    Markdown:
    [narrator:en_joe]
    Once upon a time...
    {pause:1.0}

    [alice:en_rachel]
    Hello there!
    {pause:0.5}
    
    Returns:
        A message indicating success or failure of the operation.
    """
    try:
        # Create temp directory if not provided
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp()
        temp_dir = Path(temp_dir)
        
        # Handle script input
        if isinstance(script, str):
            # Check if it's a file path
            script_path = Path(os.path.expanduser(script))
            if script_path.exists():
                with open(script_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if script_format == "json":
                        script = json.loads(content)
                    else:
                        script = content
            elif script_format == "json" and (script.startswith('{') or script.startswith('[')):
                # It's a JSON string
                script = json.loads(script)
        
        # Parse the script
        if script_format == "json":
            if isinstance(script, str):
                conversation = json.loads(script)
            else:
                conversation = script
            segments = conversation["conversation"]
        else:
            segments = parse_markdown_script(script)
        
        # Expand output path
        output_path = os.path.expanduser(output_path)
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Track sample rate from first segment for consistency
        sample_rate = None
        
        # Generate individual audio segments
        audio_segments = []
        for i, segment in enumerate(segments):
            voice_id = segment["voice"]
            
            # Get or create voice instance
            voice = voice_manager.get_voice(voice_id)
            
            # Generate temp filename
            temp_file = temp_dir / f"segment_{i}.wav"
            
            # Generate audio for this segment
            success = voice.generate_audio(segment["text"], str(temp_file))
            if not success:
                raise Exception(f"Failed to generate audio for segment {i} with voice {voice_id}")
            
            # Load the audio data
            audio_data, sr = sf.read(str(temp_file))
            
            # Set sample rate from first segment
            if sample_rate is None:
                sample_rate = sr
            elif sr != sample_rate:
                raise Exception(f"Inconsistent sample rates: {sr} != {sample_rate}")
            
            # Add pause after segment if specified
            if "pause_after" in segment:
                pause_samples = int(segment["pause_after"] * sample_rate)
                pause = np.zeros(pause_samples)
                audio_data = np.concatenate([audio_data, pause])
            
            audio_segments.append(audio_data)
            
            # Clean up temp file
            temp_file.unlink()
        
        # Combine all segments
        final_audio = np.concatenate(audio_segments)
        
        # Save final audio
        sf.write(output_path, final_audio, sample_rate)
        
        # Clean up temp directory
        if temp_dir != Path(tempfile.gettempdir()):
            temp_dir.rmdir()
        
        # Generate summary of the conversation
        summary = "\nConversation Summary:\n"
        for i, segment in enumerate(segments, 1):
            summary += f"{i}. {segment['speaker']} ({segment['voice']}): {segment['text'][:50]}...\n"
        
        return f"Successfully generated conversation audio at {output_path}\n{summary}"
        
    except Exception as e:
        # Clean up temp directory on error
        if temp_dir and temp_dir != Path(tempfile.gettempdir()):
            try:
                for file in temp_dir.glob("*.wav"):
                    file.unlink()
                temp_dir.rmdir()
            except Exception:
                pass
        return f"ERROR: Failed to generate conversation: {str(e)}"

@mcp.resource(uri="mcp://speech/usage_guide")
def usage_guide() -> str:
    """
    Return the usage guide for the Speech MCP.
    """
    return """
    # Speech MCP Usage Guide
    
    This MCP extension provides voice interaction capabilities with a simplified interface.
    
    ## How to Use
    
    1. Use an MCP UI client to connect to this server for visual feedback.
       
    2. Start a conversation:
       ```
       user_input = start_conversation()
       ```
       This initializes the speech recognition system and immediately starts listening for user input.
       Note: The first time you run this, it will download the faster-whisper model which may take a moment.
    
    3. Reply to the user and get their response:
       ```
       user_response = reply("Your response text here")
       ```
       This speaks your response and then listens for the user's reply.
       
    4. Speak without waiting for a response:
       ```
       reply("This is just an announcement", wait_for_response=False)
       ```
       This speaks the text but doesn't listen for a response, useful for announcements or confirmations.
       
    5. Close the UI from your MCP UI client when done.
       
    6. Transcribe audio/video files:
       ```
       transcription = transcribe("/path/to/media.mp4")
       ```
       This converts speech from media files to text. Supports various formats:
       - Audio: mp3, wav, m4a, flac, aac, ogg
       - Video: mp4, mov, avi, mkv, webm
       For video files, the audio track is automatically extracted for transcription.
       
    7. Generate speech audio files:
       ```
       narrate("Your text to convert to speech", "/path/to/output.wav")
       ```
       This converts text to speech and saves it as a WAV file using the configured TTS engine.
       Note: Requires a TTS engine that supports saving to file (like Kokoro).
    
    ## Typical Workflow
    
    1. Start the conversation to get the initial user input
    2. Process the transcribed speech
    3. Use the reply function to respond and get the next user input
    4. Repeat steps 2-3 for a continuous conversation
    
    ## Example Conversation Flow
    
    ```python
    # Start the conversation
    user_input = start_conversation()
    
    # Process the input and generate a response
    # ...
    
    # Reply to the user and get their response
    follow_up = reply("Here's my response to your question.")
    
    # Process the follow-up and reply again
    reply("I understand your follow-up question. Here's my answer.")
    
    # Make an announcement without waiting for a response
    reply("I'll notify you when the process is complete.", wait_for_response=False)
    
    # Close the UI when done with voice interaction
    close_ui()
    ```
    
    ## File Processing Examples
    
    ```python
    # Transcribe an audio file
    transcript = transcribe("recording.mp3")
    print("Transcription:", transcript)
    
    # Generate a speech file
    narrate("This text will be converted to speech", "output.wav")
    ```
    
    ## Tips
    
    - For best results, use a quiet environment and speak clearly
    - Kokoro TTS is automatically initialized on server start for faster response times
    - Use an MCP UI client (like mcp-ui) to view microphone status and activity:
      - A blue pulsing circle indicates active listening
      - A green circle indicates the system is speaking
      - Voice selection is available in the UI dropdown
      - Only one UI instance can run at a time (prevents duplicates)
    - The system automatically detects silence to know when you've finished speaking
      - Silence detection waits for 5 seconds of quiet before stopping recording
      - This allows for natural pauses in speech without cutting off
    - The overall listening timeout is set to 10 minutes to allow for extended thinking time or long pauses
    - For file transcription, use high-quality audio for best results
    - When generating speech files, ensure the output path ends with .wav extension
    """

@mcp.resource(uri="mcp://speech/kokoro_tts")
def kokoro_tts_guide() -> str:
    """
    Return information about the Kokoro TTS adapter.
    """
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "kokoro_tts_adapter.md"), 'r') as f:
            return f.read()
    except Exception:
        return """
        # Kokoro TTS Adapter
        
        Kokoro is a high-quality neural text-to-speech engine that can be used with speech-mcp.
        
        To install Kokoro, run:
        ```
        python scripts/install_kokoro.py
        ```
        
        For more information, see the documentation in the speech-mcp repository.
        """

@mcp.resource(uri="mcp://speech/transcription_guide")
def transcription_guide() -> str:
    """
    Return the transcription guide for speech-mcp.
    """
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "transcription_guide.md"), 'r') as f:
            return f.read()
    except Exception:
        return """
        # Speech Transcription Guide
        
        For detailed documentation on speech transcription features including timestamps
        and speaker detection, please see the transcription_guide.md file in the
        speech-mcp repository.
        """
