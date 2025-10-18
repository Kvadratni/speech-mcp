"""
Audio processing module for speech-mcp.

This module provides centralized audio processing functionality including:
- Audio device selection
- Audio recording (with both traditional and streaming modes)
- Audio playback
- Audio level visualization
- Silence detection (in traditional mode)

The module supports two recording modes:
1. Traditional mode: Uses silence detection to automatically stop recording
2. Streaming mode: Continuously streams audio chunks to a callback function
"""

import os
import time
import tempfile
import threading
import wave
import numpy as np
import pyaudio
from typing import Optional, List, Tuple, Callable, Any, Dict

# Import the centralized logger
from speech_mcp.utils.logger import get_logger

# Get a logger for this module
logger = get_logger(__name__, component="stt")

# Import centralized constants
from speech_mcp.constants import (
    CHUNK, FORMAT, CHANNELS, RATE,
    SILENCE_THRESHOLD, MAX_SILENCE_DURATION, SILENCE_CHECK_INTERVAL,
    START_LISTENING_SOUND, STOP_LISTENING_SOUND
)

class AudioProcessor:
    """
    Core audio processing class that handles device selection, recording, and playback.
    
    This class provides the shared audio functionality used by both the server and UI components.
    """
    
    def __init__(self, on_audio_level: Optional[Callable[[float], None]] = None):
        """
        Initialize the audio processor.
        
        Args:
            on_audio_level: Optional callback function that receives audio level updates (0.0 to 1.0)
        """
        self.pyaudio = None
        self.stream = None
        self.selected_device_index = None
        self.is_listening = False
        self.audio_frames = []
        self.on_audio_level = on_audio_level
        self._on_recording_complete = None
        self._on_audio_chunk = None  # Callback for streaming mode
        self._streaming_mode = False  # Flag for streaming mode
        self._setup_audio()
    
    def _setup_audio(self) -> None:
        """Set up audio capture and processing."""
        try:
            logger.info("Setting up audio processing")
            self.pyaudio = pyaudio.PyAudio()
            
            # Log audio device information
            logger.info(f"PyAudio version: {pyaudio.get_portaudio_version()}")
            
            # Get all available audio devices
            info = self.pyaudio.get_host_api_info_by_index(0)
            numdevices = info.get('deviceCount')
            logger.info(f"Found {numdevices} audio devices:")
            
            # Check for preferred devices in config
            try:
                # Check for preferred devices in config
                from speech_mcp.config import get_setting

                preferred_devices = [name.lower() for name in get_setting("stt", "preferred_devices", [])]
            except ImportError:
                preferred_devices = []

            if preferred_devices:
                logger.info(f"Preferred input device list: {preferred_devices}")
            else:
                logger.info(f"No preferred input devices configured")

            # Build a list of all available input devices
            available_devices = self.get_available_devices()

            matched_preferred_device = None
            for preferred_device_name in preferred_devices:
                matched_preferred_device = next((device for device in available_devices if preferred_device_name.lower() in device['name'].lower()), None)
                if matched_preferred_device:
                    self.selected_device_index = matched_preferred_device['index']
                    logger.info(f"Selected input device from config: {matched_preferred_device['name']} (index {self.selected_device_index})")
                    break

            # No matched preferred device
            # Prefer non-default devices as they're often external mics
            if self.selected_device_index is None:
                logger.info("No preferred device matched, using fallback logic")
                selected_device = next((device for device in available_devices if 'default' not in device['name'].lower()), None)
                if selected_device:
                    self.selected_device_index = selected_device['index']
                    logger.info(f"Selected input device: {selected_device['name']} (index {self.selected_device_index})")

            if self.selected_device_index is None:
                logger.warning("No suitable input device found, using default")
            
        except Exception as e:
            logger.error(f"Error setting up audio: {e}")
    
    def start_listening(self, 
                   callback: Optional[Callable] = None, 
                   on_recording_complete: Optional[Callable[[str], None]] = None,
                   streaming_mode: bool = True,  # Force streaming mode to True
                   on_audio_chunk: Optional[Callable[[bytes], None]] = None) -> bool:
        """
        Start listening for audio input.
        
        Args:
            callback: Optional callback function to call when audio data is received
            on_recording_complete: Optional callback function to call when recording is complete,
                                  receives the path to the recorded audio file as an argument
            streaming_mode: Whether to use streaming mode (no silence detection) - Currently forced to True
            on_audio_chunk: Optional callback function to receive audio chunks in streaming mode
            
        Returns:
            bool: True if listening started successfully, False otherwise
        """
        if self.is_listening:
            logger.debug("Already listening, ignoring start_listening call")
            return True
            
        self.is_listening = True
        self.audio_frames = []
        self._on_recording_complete = on_recording_complete
        self._streaming_mode = streaming_mode
        self._on_audio_chunk = on_audio_chunk
        
        # Play start listening notification sound
        threading.Thread(target=self.play_audio_file, args=(START_LISTENING_SOUND,), daemon=True).start()
        
        try:
            logger.info("Starting audio recording")
            
            def audio_callback(in_data, frame_count, time_info, status):
                try:
                    # Check for audio status flags
                    if status:
                        status_flags = []
                        if status & pyaudio.paInputUnderflow:
                            status_flags.append("Input Underflow")
                        if status & pyaudio.paInputOverflow:
                            status_flags.append("Input Overflow")
                        if status & pyaudio.paOutputUnderflow:
                            status_flags.append("Output Underflow")
                        if status & pyaudio.paOutputOverflow:
                            status_flags.append("Output Overflow")
                        if status & pyaudio.paPrimingOutput:
                            status_flags.append("Priming Output")
                        
                        if status_flags:
                            logger.warning(f"Audio callback status flags: {', '.join(status_flags)}")
                    
                    # Store audio data for processing
                    self.audio_frames.append(in_data)
                    
                    # Process audio for visualization
                    self._process_audio_for_visualization(in_data)
                    
                    # Call streaming callback if in streaming mode
                    if self._streaming_mode and self._on_audio_chunk:
                        self._on_audio_chunk(in_data)
                    
                    # Call user-provided callback if available
                    if callback:
                        callback(in_data)
                    
                    return (in_data, pyaudio.paContinue)
                    
                except Exception as e:
                    logger.error(f"Error in audio callback: {e}")
                    return (in_data, pyaudio.paContinue)  # Try to continue despite errors
            
            # Start the audio stream with the selected device
            logger.debug(f"Opening audio stream with FORMAT={FORMAT}, CHANNELS={CHANNELS}, RATE={RATE}, CHUNK={CHUNK}, DEVICE={self.selected_device_index}")
            self.stream = self.pyaudio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=self.selected_device_index,
                frames_per_buffer=CHUNK,
                stream_callback=audio_callback
            )
            
            # Verify stream is active and receiving audio
            if not self.stream.is_active():
                logger.error("Stream created but not active")
                self.is_listening = False
                return False
            
            logger.info("Audio stream initialized and receiving data")
            
            # Start silence detection thread only if not in streaming mode
            if not self._streaming_mode:
                def silence_detection_thread():
                    self._detect_silence()
                    # If recording completed and callback is provided, get the audio path and call the callback
                    if self._on_recording_complete and not self.is_listening:
                        audio_path = self.get_recorded_audio_path()
                        self._on_recording_complete(audio_path)
                
                threading.Thread(target=silence_detection_thread, daemon=True).start()
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting audio stream: {e}")
            self.is_listening = False
            return False
    
    def _process_audio_for_visualization(self, audio_data: bytes) -> None:
        """
        Process audio data for visualization.
        
        Args:
            audio_data: Raw audio data from PyAudio
        """
        try:
            # Convert to numpy array
            data = np.frombuffer(audio_data, dtype=np.int16)
            
            # Normalize the data to range [-1, 1]
            normalized = data.astype(float) / 32768.0
            
            # Take absolute value to get amplitude
            amplitude = np.abs(normalized).mean()
            
            # Apply amplification factor to make the visualization more prominent
            # Increase the factor from 1.0 to 5.0 to make the visualization more visible
            amplification_factor = 5.0
            amplified_amplitude = min(amplitude * amplification_factor, 1.0)  # Clamp to 1.0 max
            
            # Call the audio level callback if provided
            if self.on_audio_level:
                self.on_audio_level(amplified_amplitude)
            
        except Exception:
            pass
    
    def _detect_silence(self) -> None:
        """
        Detect when the user stops speaking and end recording.
        
        This method runs in a separate thread and monitors audio levels to detect
        when the user has stopped speaking.
        """
        try:
            # Wait for initial audio to accumulate
            time.sleep(0.5)
            
            # Initialize silence detection parameters
            silence_duration = 0
            
            while self.is_listening and self.stream and silence_duration < MAX_SILENCE_DURATION:
                if not self.audio_frames or len(self.audio_frames) < 2:
                    time.sleep(SILENCE_CHECK_INTERVAL)
                    continue
                
                # Get the latest audio frame
                latest_frame = self.audio_frames[-1]
                audio_data = np.frombuffer(latest_frame, dtype=np.int16)
                normalized = audio_data.astype(float) / 32768.0
                current_amplitude = np.abs(normalized).mean()
                
                if current_amplitude < SILENCE_THRESHOLD:
                    silence_duration += SILENCE_CHECK_INTERVAL
                else:
                    silence_duration = 0
                
                time.sleep(SILENCE_CHECK_INTERVAL)
            
            # If we exited because of silence detection
            if self.is_listening and self.stream:
                self.stop_listening()
            
        except Exception:
            pass
    
    def stop_listening(self) -> None:
        """
        Stop listening for audio input.
        
        Returns:
            None
        """
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
                
                # Play stop listening notification sound
                threading.Thread(target=self.play_audio_file, args=(STOP_LISTENING_SOUND,), daemon=True).start()
            
            self.is_listening = False
            
        except Exception:
            self.is_listening = False
    
    def get_recorded_audio_path(self) -> Optional[str]:
        """
        Save the recorded audio to a temporary WAV file and return the path.
        
        Returns:
            str: Path to the temporary WAV file, or None if an error occurred
        """
        if not self.audio_frames:
            return None
        
        try:
            # Check if we have enough audio data
            total_audio_time = len(self.audio_frames) * (CHUNK / RATE)
            
            # Save the recorded audio to a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
                
                # Create a WAV file from the recorded frames
                wf = wave.open(temp_audio_path, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.pyaudio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(self.audio_frames))
                wf.close()
            
            return temp_audio_path
            
        except Exception:
            return None
    
    def record_audio(self, streaming_mode: bool = True, on_audio_chunk: Optional[Callable[[bytes], None]] = None) -> Optional[str]:
        """
        Record audio from the microphone and return the path to the audio file.
        
        This is a blocking method that handles the entire recording process including
        starting recording and streaming transcription. Silence detection is disabled.
        
        Args:
            streaming_mode: Whether to use streaming mode (forced to True)
            on_audio_chunk: Optional callback function to receive audio chunks in streaming mode
            
        Returns:
            str: Path to the recorded audio file, or None if an error occurred
        """
        if not self.start_listening(streaming_mode=streaming_mode, on_audio_chunk=on_audio_chunk):
            return None
        
        # Wait for recording to complete (silence detection will stop it in non-streaming mode)
        while self.is_listening:
            time.sleep(0.1)
        
        # Get the recorded audio file path
        return self.get_recorded_audio_path()
    
    def play_audio_file(self, file_path: str) -> bool:
        """
        Play an audio file using PyAudio.
        
        Args:
            file_path: Path to the audio file to play
            
        Returns:
            bool: True if the file was played successfully, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                return False
            
            # Open the wave file
            with wave.open(file_path, 'rb') as wf:
                # Create PyAudio instance
                p = pyaudio.PyAudio()
                
                # Open stream
                stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                                channels=wf.getnchannels(),
                                rate=wf.getframerate(),
                                output=True)
                
                # Read data in chunks and play
                chunk_size = 1024
                data = wf.readframes(chunk_size)
                
                while data:
                    stream.write(data)
                    data = wf.readframes(chunk_size)
                
                # Close stream and PyAudio
                stream.stop_stream()
                stream.close()
                p.terminate()
                
                return True
        except Exception:
            return False
    
    def get_available_devices(self) -> List[Dict[str, Any]]:
        """
        Get a list of available audio input devices.
        
        Returns:
            List of dictionaries containing device information
        """
        devices = []
        
        try:
            if not self.pyaudio:
                self._setup_audio()
                
            if not self.pyaudio:
                return devices
                
            # Get all available audio devices
            info = self.pyaudio.get_host_api_info_by_index(0)
            numdevices = info.get('deviceCount')
            
            for i in range(numdevices):
                try:
                    device_info = self.pyaudio.get_device_info_by_host_api_device_index(0, i)
                    device_name = device_info.get('name')
                    max_input_channels = device_info.get('maxInputChannels')

                    logger.info(f"Device {i}: {device_name}")
                    logger.info(f"  Max Input Channels: {max_input_channels}")
                    logger.info(f"  Default Sample Rate: {device_info.get('defaultSampleRate')}")
                    
                    # Only include input devices
                    if max_input_channels > 0:
                        logger.info(f"Found input device: {device_name}")

                        devices.append({
                            'index': i,
                            'name': device_info.get('name'),
                            'channels': max_input_channels,
                            'sample_rate': device_info.get('defaultSampleRate')
                        })
                except Exception as e:
                    logger.warning(f"Error checking device {i}: {e}")
                    
            return devices
            
        except Exception as e:
            logger.warning(f"Error getting all devices: {e}")
            return devices
    
    def set_device_index(self, device_index: int) -> bool:
        """
        Set the audio input device by index.
        
        Args:
            device_index: Index of the audio device to use
            
        Returns:
            bool: True if the device was set successfully, False otherwise
        """
        try:
            # Check if the device exists
            if not self.pyaudio:
                self._setup_audio()
                
            if not self.pyaudio:
                return False
                
            try:
                device_info = self.pyaudio.get_device_info_by_host_api_device_index(0, device_index)
                if device_info.get('maxInputChannels') > 0:
                    self.selected_device_index = device_index
                    return True
                else:
                    return False
            except Exception:
                return False
                
        except Exception:
            return False
    
    def cleanup(self) -> None:
        """
        Clean up resources used by the audio processor.
        
        This should be called when the audio processor is no longer needed.
        """
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
                
            if self.pyaudio:
                self.pyaudio.terminate()
                self.pyaudio = None
                
        except Exception:
            pass
