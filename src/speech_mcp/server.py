import sys
import os
import json
import time
import threading
import tempfile
import subprocess
import psutil
import importlib.util
from typing import Dict, List, Union, Optional, Callable
from pathlib import Path
import numpy as np
import soundfile as sf

from mcp.server.fastmcp import FastMCP
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData, INTERNAL_ERROR, INVALID_PARAMS

# Import the centralized logger
from speech_mcp.utils.logger import get_logger

# Get a logger for this module
logger = get_logger(__name__, component="server")

# Import centralized constants
from speech_mcp.constants import (
    SERVER_LOG_FILE,
    TRANSCRIPTION_FILE, RESPONSE_FILE, COMMAND_FILE,
    CMD_LISTEN, CMD_SPEAK, CMD_IDLE, CMD_UI_READY, CMD_UI_CLOSED,
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
        
        # Only create response file if specifically requested
        if create_response_file:
            # Create or update response file for UI communication
            # This helps ensure the UI is properly notified of state changes
            if state.get("speaking", False):
                # If speaking, write the response to the file for the UI to pick up
                logger.debug(f"Creating response file with text: {state.get('last_response', '')[:30]}...")
                with open(RESPONSE_FILE, 'w') as f:
                    f.write(state.get("last_response", ""))
        
        # Create a special command file to signal state changes to the UI
        command = ""
        if state.get("listening", False):
            command = CMD_LISTEN
        elif state.get("speaking", False):
            command = CMD_SPEAK
        else:
            command = CMD_IDLE
        
        logger.debug(f"Writing command {command} to {COMMAND_FILE}")
        with open(COMMAND_FILE, 'w') as f:
            f.write(command)
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

def ensure_ui_is_running():
    """Ensure the PyQt UI process is running"""
    global speech_state
    
    # Check if UI is already active
    if speech_state.get("ui_active", False) and speech_state.get("ui_process_id"):
        # Check if the process is actually running
        try:
            process_id = speech_state["ui_process_id"]
            if psutil.pid_exists(process_id):
                process = psutil.Process(process_id)
                if process.status() != psutil.STATUS_ZOMBIE:
                    return True
        except Exception:
            pass
    
    # Check for any existing UI processes by looking for Python processes running speech_mcp.ui
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if cmdline and len(cmdline) >= 3:
                    # Look specifically for PyQt UI processes
                    if 'python' in cmdline[0].lower() and '-m' in cmdline[1] and 'speech_mcp.ui' in cmdline[2]:
                        # Found an existing PyQt UI process
                        
                        # Update our state to track this process
                        speech_state["ui_active"] = True
                        speech_state["ui_process_id"] = proc.info['pid']
                        save_speech_state(speech_state, False)
                        
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except Exception:
        pass
    
    # No UI process found, we'll need to start one using the launch_ui tool
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

def cleanup_ui_process():
    """Clean up the PyQt UI process when the server shuts down"""
    global speech_state
    
    if speech_state.get("ui_active", False) and speech_state.get("ui_process_id"):
        try:
            process_id = speech_state["ui_process_id"]
            if psutil.pid_exists(process_id):
                process = psutil.Process(process_id)
                process.terminate()
                try:
                    process.wait(timeout=3)
                except psutil.TimeoutExpired:
                    process.kill()
            
            # Update state
            speech_state["ui_active"] = False
            speech_state["ui_process_id"] = None
            save_speech_state(speech_state, False)
            
            # Write a UI_CLOSED command to the command file
            try:
                with open(COMMAND_FILE, 'w') as f:
                    f.write(CMD_UI_CLOSED)
            except Exception:
                pass
        except Exception:
            pass

# Register cleanup function to be called on exit
import atexit
atexit.register(cleanup_ui_process)

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
def launch_ui() -> str:
    """
    Launch the speech UI.
    
    This will start the speech UI window that shows the microphone status and speech visualization.
    The UI is required for visual feedback during speech recognition.
    
    Returns:
        A message indicating whether the UI was successfully launched.
    """
    global speech_state
    
    # Check if UI is already running
    if ensure_ui_is_running():
        return "Speech UI is already running."
    
    # Check if a voice preference is saved
    has_voice_preference = False
    try:
        # Import config module if available
        if importlib.util.find_spec("speech_mcp.config") is not None:
            from speech_mcp.config import get_setting, get_env_setting
            
            # Check environment variable
            env_voice = get_env_setting(ENV_TTS_VOICE)
            if env_voice:
                has_voice_preference = True
            else:
                # Check config file
                config_voice = get_setting("tts", "voice", None)
                if config_voice:
                    has_voice_preference = True
    except Exception:
        pass
    
    # Start a new UI process
    try:
        # Check for any existing UI processes first to prevent duplicates
        existing_ui = False
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if cmdline and len(cmdline) >= 3:
                    # Look specifically for PyQt UI processes
                    if 'python' in cmdline[0].lower() and '-m' in cmdline[1] and 'speech_mcp.ui' in cmdline[2]:
                        # Found an existing PyQt UI process
                        existing_ui = True
                        
                        # Update our state to track this process
                        speech_state["ui_active"] = True
                        speech_state["ui_process_id"] = proc.info['pid']
                        save_speech_state(speech_state, False)
                        
                        return f"Speech PyQt UI is already running with PID {proc.info['pid']}."
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        
        # Start a new UI process if none exists
        if not existing_ui:
            # Clear any existing command file
            try:
                if os.path.exists(COMMAND_FILE):
                    os.remove(COMMAND_FILE)
            except Exception:
                pass
            
            # Start the UI process
            ui_process = subprocess.Popen(
                [sys.executable, "-m", "speech_mcp.ui"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Update the speech state
            speech_state["ui_active"] = True
            speech_state["ui_process_id"] = ui_process.pid
            save_speech_state(speech_state, False)
            
            # Wait for UI to fully initialize by checking for the UI_READY command
            max_wait_time = 10  # Maximum wait time in seconds
            wait_interval = 0.2  # Check every 200ms
            waited_time = 0
            ui_ready = False
            
            while waited_time < max_wait_time:
                # Check if the process is still running
                if not psutil.pid_exists(ui_process.pid):
                    return "ERROR: PyQt UI process terminated unexpectedly."
                
                # Check if the command file exists and contains UI_READY
                if os.path.exists(COMMAND_FILE):
                    try:
                        with open(COMMAND_FILE, 'r') as f:
                            command = f.read().strip()
                            if command == CMD_UI_READY:
                                ui_ready = True
                                break
                    except Exception:
                        pass
                
                # Wait before checking again
                time.sleep(wait_interval)
                waited_time += wait_interval
            
            if ui_ready:
                # Check if we have a voice preference
                if has_voice_preference:
                    return f"PyQt Speech UI launched successfully with PID {ui_process.pid} and is ready."
                else:
                    return f"PyQt Speech UI launched successfully with PID {ui_process.pid}. Please select a voice to continue."
            else:
                return f"PyQt Speech UI launched with PID {ui_process.pid}, but readiness state is unknown."
    except Exception as e:
        return f"ERROR: Failed to launch PyQt Speech UI: {str(e)}"

@mcp.tool()
def start_conversation() -> str:
    """
    Start a voice conversation by beginning to listen.
    
    This will initialize the speech recognition system and immediately start listening for user input.
    
    Returns:
        The transcription of the user's speech.
    """
    global speech_state
    
    # Force reset the state
    state_manager.update_state({
        "listening": False,
        "speaking": False,
        "last_transcript": "",
        "last_response": "",
        "ui_active": False,
        "ui_process_id": None,
        "error": None
    })
    
    # Initialize speech recognition if not already done
    if not initialize_speech_recognition():
        return "ERROR: Failed to initialize speech recognition."
    
    # Check if UI is running but don't launch it automatically
    ensure_ui_is_running()
    
    # Start listening
    try:
        # Set listening state before starting to ensure UI shows the correct state
        speech_state["listening"] = True
        save_speech_state(speech_state, False)
        
        # Create a special command file to signal LISTEN state to the UI
        # This ensures the audio blips are played
        try:
            with open(COMMAND_FILE, 'w') as f:
                f.write(CMD_LISTEN)
        except Exception:
            pass
        
        # Use a queue to get the result from the thread
        import queue
        result_queue = queue.Queue()
        
        def listen_and_queue():
            try:
                result = listen_for_speech()
                result_queue.put(result)
            except Exception as e:
                result_queue.put(f"ERROR: {str(e)}")
        
        # Start the thread
        listen_thread = threading.Thread(target=listen_and_queue)
        listen_thread.daemon = True
        listen_thread.start()
        
        # Wait for the result with a timeout
        try:
            transcription = result_queue.get(timeout=SPEECH_TIMEOUT)
            
            # Signal that we're done listening
            speech_state["listening"] = False
            save_speech_state(speech_state, False)
            
            # Create a special command file to signal IDLE state to the UI
            # This ensures the audio blips are played
            try:
                with open(COMMAND_FILE, 'w') as f:
                    f.write(CMD_IDLE)
            except Exception:
                pass
            
            return transcription
        except queue.Empty:
            # Update state to stop listening
            speech_state["listening"] = False
            save_speech_state(speech_state, False)
            
            # Signal that we're done listening
            try:
                with open(COMMAND_FILE, 'w') as f:
                    f.write(CMD_IDLE)
            except Exception:
                pass
            
            # Create an emergency transcription
            emergency_message = f"ERROR: Timeout waiting for speech transcription after {SPEECH_TIMEOUT} seconds."
            return emergency_message
    
    except Exception as e:
        # Update state to stop listening
        speech_state["listening"] = False
        save_speech_state(speech_state, False)
        
        # Signal that we're done listening
        try:
            with open(COMMAND_FILE, 'w') as f:
                f.write(CMD_IDLE)
        except Exception:
            pass
        
        # Return an error message instead of raising an exception
        error_message = f"ERROR: Failed to start conversation: {str(e)}"
        return error_message

@mcp.tool()
def reply(text: str, wait_for_response: bool = True) -> str:
    """
    Speak the provided text and optionally listen for a response.
    
    This will speak the given text and then immediately start listening for user input
    if wait_for_response is True. If wait_for_response is False, it will just speak
    the text without listening for a response.
    
    Args:
        text: The text to speak to the user
        wait_for_response: Whether to wait for and return the user's response (default: True)
        
    Returns:
        If wait_for_response is True: The transcription of the user's response.
        If wait_for_response is False: A confirmation message that the text was spoken.
    """
    global speech_state
    
    # Reset listening and speaking states to ensure we're in a clean state
    speech_state["listening"] = False
    speech_state["speaking"] = False
    save_speech_state(speech_state, False)
    
    # Clear any existing response file to prevent double-speaking
    try:
        if os.path.exists(RESPONSE_FILE):
            os.remove(RESPONSE_FILE)
    except Exception:
        pass
    
    # Speak the text
    try:
        speak_text(text)
        
        # Add a small delay to ensure speaking is complete
        time.sleep(0.5)
    except Exception as e:
        return f"ERROR: Failed to speak text: {str(e)}"
    
    # If we don't need to wait for a response, return now
    if not wait_for_response:
        return f"Spoke: {text}"
    
    # Check if UI is running but don't launch it automatically
    ensure_ui_is_running()
    
    # Start listening for response
    try:
        # Use a queue to get the result from the thread
        import queue
        result_queue = queue.Queue()
        
        def listen_and_queue():
            try:
                result = listen_for_speech()
                result_queue.put(result)
            except Exception as e:
                result_queue.put(f"ERROR: {str(e)}")
        
        # Start the thread
        listen_thread = threading.Thread(target=listen_and_queue)
        listen_thread.daemon = True
        listen_thread.start()
        
        # Wait for the result with a timeout
        try:
            transcription = result_queue.get(timeout=SPEECH_TIMEOUT)
            return transcription
        except queue.Empty:
            # Update state to stop listening
            speech_state["listening"] = False
            save_speech_state(speech_state, False)
            
            # Create an emergency transcription
            emergency_message = f"ERROR: Timeout waiting for speech transcription after {SPEECH_TIMEOUT} seconds."
            return emergency_message
    
    except Exception as e:
        # Update state to stop listening
        speech_state["listening"] = False
        save_speech_state(speech_state, False)
        
        # Return an error message instead of raising an exception
        error_message = f"ERROR: Failed to listen for response: {str(e)}"
        return error_message

@mcp.tool()
def close_ui() -> str:
    """
    Close the speech UI window.
    
    This will gracefully shut down the speech UI window if it's currently running.
    Use this when you're done with voice interaction to clean up resources.
    
    Returns:
        A message indicating whether the UI was successfully closed.
    """
    global speech_state
    
    # Check if UI is running
    if speech_state.get("ui_active", False) and speech_state.get("ui_process_id"):
        try:
            process_id = speech_state["ui_process_id"]
            if psutil.pid_exists(process_id):
                # Check if it's actually our UI process (not just a reused PID)
                try:
                    process = psutil.Process(process_id)
                    cmdline = process.cmdline()
                    if not any('speech_mcp.ui' in cmd for cmd in cmdline):
                        # Update state since this isn't our process
                        speech_state["ui_active"] = False
                        speech_state["ui_process_id"] = None
                        save_speech_state(speech_state, False)
                        return "No active Speech UI found to close (PID was reused by another process)."
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                
                # First try to gracefully close the UI by writing a UI_CLOSED command
                try:
                    with open(COMMAND_FILE, 'w') as f:
                        f.write(CMD_UI_CLOSED)
                    
                    # Give the UI a moment to close gracefully
                    time.sleep(1.0)
                except Exception:
                    pass
                
                # Now check if the process is still running
                if psutil.pid_exists(process_id):
                    # Process is still running, terminate it
                    process = psutil.Process(process_id)
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except psutil.TimeoutExpired:
                        process.kill()
            
            # Update state
            speech_state["ui_active"] = False
            speech_state["ui_process_id"] = None
            save_speech_state(speech_state, False)
            
            return "Speech UI was closed successfully."
        except Exception as e:
            return f"ERROR: Failed to close Speech UI: {str(e)}"
    else:
        return "No active Speech UI found to close."

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
    
    1. Launch the speech UI for visual feedback (optional but recommended):
       ```
       launch_ui()
       ```
       This starts the visual interface that shows when the microphone is active.
       
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
       
    5. Close the speech UI when done:
       ```
       close_ui()
       ```
       This gracefully closes the speech UI window when you're finished with voice interaction.
       
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
    - Use the `launch_ui()` function to start the visual PyQt interface:
      - The PyQt UI shows when the microphone is active and listening
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
