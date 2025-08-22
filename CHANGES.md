# Speech MCP Changes

## Latest Changes

### 8. 2.0.0 - Switch to external MCP UI
- Removed internal PyQt UI and file-based signaling
- Deprecated `launch_ui()` and `close_ui()` tools; MCP UI is handled by external clients
- Updated usage guide and README to reference MCP UI clients
- Removed `PyQt5` dependency
- Bumped major version to 2.0.0 for breaking changes

### 7. Kokoro TTS Integration
- Added support for Kokoro TTS, a high-quality neural text-to-speech engine
- Created a modular TTS adapter system to support multiple TTS engines
- Added a Kokoro adapter that falls back to pyttsx3 if Kokoro is not available
- Added pip optional dependencies for easy installation: `pip install speech-mcp[kokoro]`
- Added an installation script for Kokoro and its dependencies
- Updated documentation with information about Kokoro TTS
- Enhanced the UI to use Kokoro when available
- Added support for multiple voice styles and languages

### 6. Freezing and State Management Issues
- Fixed issue where `start_conversation()` would freeze indefinitely
- Added file system permission testing to verify transcription file can be created
- Added fallback transcription mechanism when UI process fails to create transcription file
- Reduced timeout from 10 minutes to 30 seconds for better responsiveness
- Added emergency transcription when timeout occurs
- Added more detailed debug output to console
- Added psutil dependency for better process management
- Enhanced state management to avoid getting stuck in listening state
- Added force reset of speech state at the beginning of `start_conversation()`
- Modified error handling to return error messages instead of raising exceptions
- Updated documentation with troubleshooting information

### 5. Migrated to faster-whisper
- Replaced openai-whisper with faster-whisper for improved performance
- Updated transcription processing to work with faster-whisper's API
- Updated documentation to reflect the new dependency
- Configured faster-whisper to use CPU with int8 quantization for better compatibility

## Previous Fixes

### 1. Listen Function Timeout Issue
- Increased the timeout from 60 seconds to 10 minutes
- Added progress messages during listening to show that the system is still waiting
- Reduced silence detection threshold from 0.01 to 0.005 to make it less sensitive
- Increased maximum silence duration from 1.5 to 2.0 seconds before ending recording

### 2. Speak Function Not Producing Audio
- Added pyttsx3 as a dependency for text-to-speech functionality
- Implemented actual speech synthesis using pyttsx3 instead of just simulating speech with delays
- Added fallback to simulation if text-to-speech fails

### 3. UI Not Opening
- Enhanced error handling and logging in the UI startup process
- Added detailed logging of UI process output to help diagnose issues
- Increased the startup wait time from 1 to 2 seconds
- Added checks to verify if the UI process is still running after startup
- Improved process termination handling for existing UI processes
- **Fixed path issue:** Changed UI process startup to use Python module import (`python -m speech_mcp.ui`) instead of direct file path
- **Fixed log output:** Improved log output formatting to clean up error messages
- **Added minimal UI:** Implemented a simple status window that shows when the system is listening or speaking

### 4. Simplified API
- Reduced the API to just two main functions:
  - `start_conversation()`: Launches the UI and immediately starts listening
  - `reply(text)`: Speaks the provided text and then listens for a response
- Removed separate `start_voice_mode()`, `listen()`, and `speak()` functions
- Simplified the workflow for voice conversations

## How to Test

1. Make sure all dependencies are installed:
   ```
   source .venv/bin/activate
   uv pip install -e .
   ```

2. (Optional) Install Kokoro TTS:
   ```
   python scripts/install_kokoro.py
   ```

3. Run the speech-mcp server:
   ```
   speech-mcp
   ```

4. Start a conversation:
   ```
   user_input = start_conversation()
   ```

5. Reply to the user and get their response:
   ```
   user_response = reply("This is a test of the speech synthesis system")
   ```

## Troubleshooting

If you encounter issues:

1. Check the log files:
   - `/Users/mnovich/Development/speech-mcp/src/speech_mcp/speech-mcp.log`
   - `/Users/mnovich/Development/speech-mcp/src/speech_mcp/speech-mcp-server.log`
   - `/Users/mnovich/Development/speech-mcp/src/speech_mcp/speech-mcp-ui.log`

2. Make sure the UI process is running:
   ```
   ps aux | grep speech_mcp
   ```

3. If the UI is not running, check for error messages in the logs and try restarting the server.

4. If the extension seems stuck, try deleting or resetting the state file:
   ```
   echo '{"ui_active": false, "listening": false, "speaking": false, "last_transcript": "", "last_response": ""}' > src/speech_mcp/speech_state.json
   ```

5. Use the direct command instead of `uv run speech-mcp`:
   ```
   goose session --with-extension "speech-mcp"
   ```
