# Speech MCP

A Goose MCP extension for voice interaction with modern audio visualization.


https://github.com/user-attachments/assets/f10f29d9-8444-43fb-a919-c80b9e0a12c8



## Overview

Speech MCP provides a voice interface for [Goose](https://github.com/block/goose), allowing users to interact through speech rather than text. It includes:

- Real-time audio processing for speech recognition
- Local speech-to-text using faster-whisper (a faster implementation of OpenAI's Whisper model)
- High-quality text-to-speech with multiple voice options
- Modern PyQt-based UI with audio visualization
- Simple command-line interface for voice interaction

## Features

- **Modern UI**: Sleek PyQt-based interface with audio visualization and dark theme
- **Voice Input**: Capture and transcribe user speech using faster-whisper
- **Voice Output**: Convert agent responses to speech with 54+ voice options
- **Voice Persistence**: Remembers your preferred voice between sessions
- **Continuous Conversation**: Automatically listen for user input after agent responses
- **Silence Detection**: Automatically stops recording when the user stops speaking
- **Robust Error Handling**: Graceful recovery from common failure modes

## Installation
> **Important Note**: After installation, the first time you use the speech interface, it may take several minutes to download the Kokoro voice models (approximately 523 KB per voice). During this initial setup period, the system will use a more robotic-sounding fallback voice. Once the Kokoro voices are downloaded, the high-quality voices will be used automatically.

## ⚠️ IMPORTANT PREREQUISITES ⚠️

Before installing Speech MCP, you **MUST** install PortAudio on your system. PortAudio is required for PyAudio to capture audio from your microphone.

### PortAudio Installation Instructions

**macOS:**
```bash
brew install portaudio
export LDFLAGS="-L/usr/local/lib"
export CPPFLAGS="-I/usr/local/include"
```

**Linux (Debian/Ubuntu):**
```bash
sudo apt-get update
sudo apt-get install portaudio19-dev python3-dev
```

**Linux (Fedora/RHEL/CentOS):**
```bash
sudo dnf install portaudio-devel
```

**Windows:**
For Windows, PortAudio is included in the PyAudio wheel file, so no separate installation is required when installing PyAudio with pip.

> **Note**: If you skip this step, PyAudio installation will fail with "portaudio.h file not found" errors and the extension will not work.

### Option 1: Quick Install (One-Click)

Click the link below if you have Goose installed:

[goose://extension?cmd=uvx&&arg=-p&arg=3.10.14arg=speech-mcp@latest&id=speech_mcp&name=Speech%20Interface&description=Voice%20interaction%20with%20audio%20visualization%20for%20Goose](goose://extension?cmd=uvx&arg=-p&arg=3.10.14&arg=speech-mcp@latest&id=speech_mcp&name=Speech%20Interface&description=Voice%20interaction%20with%20audio%20visualization%20for%20Goose)

### Option 2: Using Goose CLI (recommended)

Start Goose with your extension enabled:

```bash
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

1. Install PortAudio (see [Prerequisites](#prerequisites) section)
2. Clone this repository
3. Install dependencies:
   ```
   uv pip install -e .
   ```
   
   Or for a complete installation including Kokoro TTS:
   ```
   uv pip install -e .[all]
   ```

## Dependencies

- Python 3.10+
- PyQt5 (for modern UI)
- PyAudio (for audio capture)
- faster-whisper (for speech-to-text)
- NumPy (for audio processing)
- Pydub (for audio processing)
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

To use this MCP with Goose, simply ask Goose to talk to you or start a voice conversation:

1. Start a conversation by saying something like:
   ```
   "Let's talk using voice"
   "Can we have a voice conversation?"
   "I'd like to speak instead of typing"
   ```

2. Goose will automatically launch the speech interface and start listening for your voice input.

3. When Goose responds, it will speak the response aloud and then automatically listen for your next input.

4. The conversation continues naturally with alternating speaking and listening, just like talking to a person.

No need to call specific functions or use special commands - just ask Goose to talk and start speaking naturally.

## UI Features

The new PyQt-based UI includes:

- **Modern Dark Theme**: Sleek, professional appearance
- **Audio Visualization**: Dynamic visualization of audio input
- **Voice Selection**: Choose from 54+ voice options
- **Voice Persistence**: Your voice preference is saved between sessions
- **Animated Effects**: Smooth animations and visual feedback
- **Status Indicators**: Clear indication of system state (ready, listening, processing)

## Configuration

User preferences are stored in `~/.config/speech-mcp/config.json` and include:

- Selected TTS voice
- TTS engine preference
- Voice speed
- Language code
- UI theme settings

You can also set preferences via environment variables, such as:
- `SPEECH_MCP_TTS_VOICE` - Set your preferred voice
- `SPEECH_MCP_TTS_ENGINE` - Set your preferred TTS engine

## Troubleshooting

If you encounter issues with the extension freezing or not responding:

1. **Check the logs**: Look at the log files in `src/speech_mcp/` for detailed error messages.
2. **Reset the state**: If the extension seems stuck, try deleting `src/speech_mcp/speech_state.json` or setting all states to `false`.
3. **Use the direct command**: Instead of `uv run speech-mcp`, use the installed package with `speech-mcp` directly.
4. **Check audio devices**: Ensure your microphone is properly configured and accessible to Python.
5. **Verify dependencies**: Make sure all required dependencies are installed correctly.

### Common PortAudio Issues

#### "PyAudio installation failed" or "portaudio.h file not found"

This typically means PortAudio is not installed or not found in your system:

- **macOS**: 
  ```bash
  brew install portaudio
  export LDFLAGS="-L/usr/local/lib"
  export CPPFLAGS="-I/usr/local/include"
  pip install pyaudio
  ```

- **Linux**:
  Make sure you have the development packages:
  ```bash
  # For Debian/Ubuntu
  sudo apt-get install portaudio19-dev python3-dev
  pip install pyaudio
  
  # For Fedora
  sudo dnf install portaudio-devel
  pip install pyaudio
  ```

#### "Audio device not found" or "No Default Input Device Available"

- Check if your microphone is properly connected
- Verify your system recognizes the microphone in your sound settings
- Try selecting a specific device index in the code if you have multiple audio devices

## Changelog

For a detailed list of recent improvements and version history, please see the [Changelog](docs/CHANGELOG.md).

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
- Multiple voice styles and languages
- To install: `python scripts/install_kokoro.py`

**Note about Voice Models**: The voice models are `.pt` files (PyTorch models) that are loaded by Kokoro. Each voice model is approximately 523 KB in size and is automatically downloaded when needed.

**Voice Persistence**: The selected voice is automatically saved to a configuration file (`~/.config/speech-mcp/config.json`) and will be remembered between sessions. This allows users to set their preferred voice once and have it used consistently.

##### Available Kokoro Voices

Speech MCP supports 54+ high-quality voice models through Kokoro TTS. For a complete list of available voices and language options, please visit the [Kokoro GitHub repository](https://github.com/hexgrad/kokoro).

## License

[MIT License](LICENSE)
