"""
Centralized constants for speech-mcp.

This module provides constants used throughout the speech-mcp extension.
It eliminates duplication by centralizing all shared constants in one place.
"""

import os
import sys
import pyaudio
from pathlib import Path

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSCRIPTION_FILE = os.path.join(BASE_DIR, "transcription.txt")
RESPONSE_FILE = os.path.join(BASE_DIR, "response.txt")
COMMAND_FILE = os.path.join(BASE_DIR, "ui_command.txt")

# Log files
SERVER_LOG_FILE = os.path.join(BASE_DIR, "speech-mcp-server.log")
UI_LOG_FILE = os.path.join(BASE_DIR, "speech-mcp-ui.log")
MAIN_LOG_FILE = os.path.join(BASE_DIR, "speech-mcp.log")

# Audio parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Audio notification files
AUDIO_DIR = os.path.join(BASE_DIR, "resources", "audio")
START_LISTENING_SOUND = os.path.join(AUDIO_DIR, "start_listening.wav")
STOP_LISTENING_SOUND = os.path.join(AUDIO_DIR, "stop_listening.wav")

# Default speech state has been moved to state_manager.py

# Configuration paths
CONFIG_DIR = os.path.join(str(Path.home()), '.config', 'speech-mcp')
CONFIG_FILE = os.path.join(CONFIG_DIR, 'config.json')

# Environment variable names
ENV_TTS_VOICE = "SPEECH_MCP_TTS_VOICE"

# Default configuration values
DEFAULT_CONFIG = {
    'tts': {
        'engine': 'kokoro',
        'voice': 'af_heart',
        'speed': 1.0,
        'lang_code': 'a'
    },
    'stt': {
        'engine': 'faster-whisper',
        'model': 'base',
        'device': 'cpu',
        'compute_type': 'int8',
        'preferred_devices': []
    },
    'ui': {
        'theme': 'dark'
    }
}

# UI Commands
CMD_LISTEN = "LISTEN"
CMD_SPEAK = "SPEAK"
CMD_IDLE = "IDLE"
CMD_UI_READY = "UI_READY"
CMD_UI_CLOSED = "UI_CLOSED"

# Speech recognition parameters
SILENCE_THRESHOLD = 0.02  # Threshold for detecting silence (higher = less sensitive)
MAX_SILENCE_DURATION = 3.0  # 3 seconds of silence to stop recording
SILENCE_CHECK_INTERVAL = 0.1  # Check every 100ms
SPEECH_TIMEOUT = 600  # 10 minutes timeout for speech recognition

# Streaming transcription parameters
STREAMING_END_SILENCE_DURATION = 4.0  # 4 seconds without new words to end streaming
STREAMING_INITIAL_WAIT = 10.0  # 10 seconds initial wait before first silence check
STREAMING_PROCESSING_INTERVAL = 0.1   # Process streaming audio every 100ms
STREAMING_BUFFER_SIZE = 10  # Number of chunks to buffer before processing (about 0.5 seconds)
STREAMING_MAX_BUFFER_SIZE = 100  # Maximum buffer size to prevent memory issues
STREAMING_MIN_WORDS = 2  # Minimum number of words before considering end of speech