# Speech MCP

A Goose MCP extension for voice interaction with audio visualization.

## Overview

Speech MCP provides a voice interface for Goose, allowing users to interact through speech rather than text. It includes:

- Real-time audio processing for speech recognition
- Local speech-to-text using faster-whisper (a faster implementation of OpenAI's Whisper model)
- Text-to-speech capabilities 
- Simple command-line interface for voice interaction

## Features

- **Voice Input**: Capture and transcribe user speech using faster-whisper
- **Voice Output**: Convert agent responses to speech
- **Continuous Conversation**: Automatically listen for user input after agent responses
- **Silence Detection**: Automatically stops recording when the user stops speaking
- **Robust Error Handling**: Graceful recovery from common failure modes

## System Requirements

Before installing, ensure you have the required system dependencies:

### macOS
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install PortAudio (required for PyAudio)
brew install portaudio
```

### Linux (Debian/Ubuntu)
```bash
sudo apt-get update
sudo apt-get install python3-dev portaudio19-dev
```

### Linux (Fedora)
```bash
sudo dnf install python3-devel portaudio-devel
```

### Windows
- No additional system dependencies required
- PyAudio wheels are available for direct installation

## Installation

### Option 1: Quick Install (One-Click)

Click the link below if you have Goose installed:

`goose://extension?cmd=uvx&arg=speech-mcp&id=speech_mcp&name=Speech%20Interface&description=Voice%20interaction%20with%20audio%20visualization%20for%20Goose`

### Option 2: Using Goose CLI (recommended)

Start Goose with your extension enabled:

```bash
# First, check system dependencies
speech-mcp-check

# If you installed via PyPI
goose session --with-extension "speech-mcp"

# Or if you want to use a local development version
goose session --with-extension "python -m speech_mcp"
```

### Option 3: Manual setup in Goose

1. Run `goose configure`
2. Select "Add Extension" from the menu
3. Choose "Command-line Extension"
4. Enter a name (e.g., "Speech Interface")
5. For the command, enter: `speech-mcp`
6. Follow the prompts to complete the setup

### Option 4: Manual Installation

1. Clone this repository
2. Install system dependencies (see System Requirements above)
3. Install Python dependencies:
   ```bash
   # Check system dependencies first
   python -m speech_mcp.install_check
   
   # Then install the package
   uv pip install -e .
   ```

## Dependencies

### System Dependencies
- PortAudio (required for audio capture)
  - macOS: Install via `brew install portaudio`
  - Linux: Install development packages (see System Requirements)
  - Windows: No additional requirements

### Python Dependencies
- Python 3.10+
- PyAudio (for audio capture)
- faster-whisper (for speech-to-text)
- NumPy (for audio processing)
- Pydub (for audio processing)
- pyttsx3 (for text-to-speech)
- psutil (for process management)

### Optional Dependencies

- **Kokoro TTS**: For high-quality text-to-speech with multiple voices
  - To install Kokoro, you can use pip with optional dependencies:
    ```bash
    pip install speech-mcp[kokoro]     # Basic Kokoro support with English
    pip install speech-mcp[ja]         # Add Japanese support
    pip install speech-mcp[zh]         # Add Chinese support
    pip install speech-mcp[all]        # All languages and features
    ```
  - Alternatively, run the installation script: `python scripts/install_kokoro.py`
  - See [Kokoro TTS Guide](docs/kokoro-tts-guide.md) for more information

## Usage

To use this MCP with Goose, you can:

1. Start a conversation:
   ```python
   user_input = start_conversation()
   ```

2. Reply to the user and get their response:
   ```python
   user_response = reply("Your response text here")
   ```

## Typical Workflow

```python
# Start the conversation
user_input = start_conversation()

# Process the input and generate a response
# ...

# Reply to the user and get their response
follow_up = reply("Here's my response to your question.")

# Process the follow-up and reply again
reply("I understand your follow-up question. Here's my answer.")
```

## Troubleshooting

### Common Issues

1. **PyAudio Installation Fails**
   - Make sure you've installed the system dependencies first (see System Requirements)
   - On macOS: Run `brew install portaudio` before installing PyAudio
   - On Linux: Install the appropriate development packages for your distribution

2. **Audio Device Issues**
   - Check if your microphone is properly connected and recognized
   - Verify microphone permissions in your system settings
   - Try running `python -m sounddevice` to list available audio devices

3. **Extension Freezing**
   - Check the logs in `src/speech_mcp/` for detailed error messages
   - Try deleting `src/speech_mcp/speech_state.json` or setting all states to `false`
   - Use `speech-mcp` directly instead of `uv run speech-mcp`

### Log Files
Look for detailed error messages in:
- `src/speech_mcp/speech-mcp.log`
- `src/speech_mcp/speech-mcp-server.log`
- `src/speech_mcp/speech-mcp-ui.log`

## Recent Fixes

- **Kokoro TTS integration**: Added support for high-quality neural text-to-speech
- **Improved error handling**: Better recovery from common failure modes
- **Timeout management**: Reduced timeouts and added fallback mechanisms
- **Process management**: Better handling of UI process startup and termination
- **State consistency**: Added state reset mechanisms to avoid getting stuck
- **Fallback transcription**: Added emergency transcription when UI process fails
- **Debugging output**: Enhanced logging and console output for troubleshooting
- **Installation checks**: Added system dependency verification during installation

## Technical Details

### Speech-to-Text

The MCP uses faster-whisper for speech recognition:
- Uses the "base" model for a good balance of accuracy and speed
- Processes audio locally without sending data to external services
- Automatically detects when the user has finished speaking
- Provides improved performance over the original Whisper implementation

### Text-to-Speech

The MCP supports multiple text-to-speech engines:

#### Default: pyttsx3
- Uses system voices available on your computer
- Works out of the box without additional setup
- Limited voice quality and customization

#### Optional: Kokoro TTS
- High-quality neural text-to-speech with multiple voices
- Lightweight model (82M parameters) that runs efficiently on CPU
- Multiple voice styles: casual, serious, robot, bright, etc.
- Supports multiple languages (English, Japanese, Chinese, Spanish, etc.)
- To install: `python scripts/install_kokoro.py`

## License

[MIT License](LICENSE)