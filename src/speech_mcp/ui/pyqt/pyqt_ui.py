"""
Main PyQt UI implementation for the Speech UI.

This module provides the main PyQt window for the speech interface.
"""

import os
import sys
import time
import logging
import threading
import random
import math
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
    QWidget, QLabel, QComboBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal

# Import centralized constants
from speech_mcp.constants import (
    STATE_FILE, TRANSCRIPTION_FILE, RESPONSE_FILE, COMMAND_FILE,
    CMD_LISTEN, CMD_SPEAK, CMD_IDLE, CMD_UI_READY, CMD_UI_CLOSED,
    ENV_TTS_VOICE
)

# Import UI components
from speech_mcp.ui.components import (
    AudioVisualizer, 
    AnimatedButton, 
    TTSAdapter, 
    AudioProcessorUI
)

# Import configuration module for voice preferences
from speech_mcp.config import get_env_setting, get_setting, set_setting, set_env_setting

# Setup logging
logger = logging.getLogger(__name__)

class PyQtSpeechUI(QMainWindow):
    """
    Main speech UI window implemented with PyQt.
    """
    # Signal for when components are fully loaded
    components_ready = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Goose Speech Interface")
        self.resize(500, 300)
        
        # Set initial loading state
        self.tts_ready = False
        self.stt_ready = False
        self.audio_ready = False
        
        # Add a watchdog timer to ensure UI responsiveness
        self.watchdog_timer = QTimer(self)
        self.watchdog_timer.timeout.connect(self.check_ui_responsiveness)
        self.watchdog_timer.start(5000)  # Check every 5 seconds
        
        # Create UI first (will be in loading state)
        self.setup_ui()
        
        # Create a command file to indicate UI is visible (but not fully ready)
        try:
            with open(COMMAND_FILE, 'w') as f:
                f.write("UI_READY")
            logger.info("Created initial UI_READY command file (UI is visible)")
        except Exception as e:
            logger.error(f"Error creating initial command file: {e}")
        
        # Start checking for server commands
        self.command_check_timer = QTimer(self)
        self.command_check_timer.timeout.connect(self.check_for_commands)
        self.command_check_timer.start(100)  # Check every 100ms
        
        # Start checking for response files
        self.response_check_timer = QTimer(self)
        self.response_check_timer.timeout.connect(self.check_for_responses)
        self.response_check_timer.start(100)  # Check every 100ms
        
        # Connect the components_ready signal to update UI
        self.components_ready.connect(self.on_components_ready)
        
        # Initialize components in background threads
        QTimer.singleShot(100, self.initialize_components)
        
    def setup_ui(self):
        """Set up the UI components."""
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Create a layout for the visualizer labels
        label_layout = QHBoxLayout()
        
        # User label
        user_label = QLabel("User")
        user_label.setAlignment(Qt.AlignCenter)
        user_label.setStyleSheet("""
            font-size: 14px;
            color: #00c8ff;
            font-weight: bold;
        """)
        label_layout.addWidget(user_label, 1)
        
        # Agent label
        agent_label = QLabel("Agent")
        agent_label.setAlignment(Qt.AlignCenter)
        agent_label.setStyleSheet("""
            font-size: 14px;
            color: #00ff64;
            font-weight: bold;
        """)
        label_layout.addWidget(agent_label, 1)
        
        # Add the label layout to the main layout
        main_layout.addLayout(label_layout)
        
        # Create a layout for the visualizers
        visualizer_layout = QHBoxLayout()
        
        # User audio visualizer (blue)
        self.user_visualizer = AudioVisualizer(mode="user", width_factor=1.0)
        visualizer_layout.addWidget(self.user_visualizer, 1)  # Equal ratio
        
        # Agent audio visualizer (green)
        self.agent_visualizer = AudioVisualizer(mode="agent", width_factor=1.0)
        visualizer_layout.addWidget(self.agent_visualizer, 1)  # Equal ratio
        
        # Add the visualizer layout to the main layout
        main_layout.addLayout(visualizer_layout)
        
        # Transcription display
        self.transcription_label = QLabel("Ready for voice interaction")
        self.transcription_label.setAlignment(Qt.AlignCenter)
        self.transcription_label.setWordWrap(True)
        self.transcription_label.setStyleSheet("""
            font-size: 14px;
            color: #ffffff;
            background-color: #2a2a2a;
            border-radius: 5px;
            padding: 10px;
        """)
        main_layout.addWidget(self.transcription_label)
        
        # Voice selection
        voice_layout = QHBoxLayout()
        voice_label = QLabel("Voice:")
        voice_label.setStyleSheet("color: #ffffff;")
        self.voice_combo = QComboBox()
        self.voice_combo.setStyleSheet("""
            background-color: #2a2a2a;
            color: #ffffff;
            border: 1px solid #3a3a3a;
            border-radius: 3px;
            padding: 5px;
        """)
        
        # Add loading placeholder
        self.voice_combo.addItem("Loading voices...")
        self.voice_combo.setEnabled(False)
        self.voice_combo.currentIndexChanged.connect(self.on_voice_changed)
        
        voice_layout.addWidget(voice_label)
        voice_layout.addWidget(self.voice_combo, 1)  # 1 = stretch factor
        main_layout.addLayout(voice_layout)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        # Add Select Voice button
        self.select_voice_button = AnimatedButton("Save Voice")
        self.select_voice_button.clicked.connect(self.save_selected_voice)
        self.select_voice_button.setEnabled(True)
        self.select_voice_button.setMinimumWidth(120)
        self.select_voice_button.set_style("""
            background-color: #9b59b6;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 8px 16px;
            font-weight: bold;
        """)
        
        # Use AnimatedButton for Test Voice button
        self.speak_button = AnimatedButton("Test Voice")
        self.speak_button.clicked.connect(self.test_voice)
        self.speak_button.setEnabled(True)
        self.speak_button.setMinimumWidth(120)
        self.speak_button.set_style("""
            background-color: #27ae60;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 8px 16px;
            font-weight: bold;
        """)
        
        # Use AnimatedButton for Close button
        self.close_button = AnimatedButton("Close")
        self.close_button.clicked.connect(self.close)
        self.close_button.setMinimumWidth(120)
        self.close_button.set_style("""
            background-color: #e74c3c;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 8px 16px;
            font-weight: bold;
        """)
        
        # Add buttons to layout with equal spacing
        button_layout.addStretch(1)
        button_layout.addWidget(self.select_voice_button)
        button_layout.addSpacing(10)
        button_layout.addWidget(self.speak_button)
        button_layout.addSpacing(10)
        button_layout.addWidget(self.close_button)
        button_layout.addStretch(1)
        
        main_layout.addLayout(button_layout)
        
        # Set the main widget
        self.setCentralWidget(main_widget)
        
        # Apply dark theme
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #121212;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
            }
        """)
        
        # Initialize visualizers to inactive state
        self.set_user_visualizer_active(False)
        self.set_agent_visualizer_active(False)
    
    def set_user_visualizer_active(self, active):
        """Set the user visualizer as active or inactive."""
        self.user_visualizer.set_active(active)
    
    def set_agent_visualizer_active(self, active):
        """Set the agent visualizer as active or inactive."""
        self.agent_visualizer.set_active(active)
    
    def update_voice_list(self):
        """Update the voice selection combo box"""
        # Skip if TTS adapter is not ready yet
        if not hasattr(self, 'tts_adapter') or not self.tts_adapter:
            logger.warning("Cannot update voice list - TTS adapter not ready")
            return
            
        self.voice_combo.clear()
        voices = self.tts_adapter.get_available_voices()
        current_voice = self.tts_adapter.get_current_voice()
        
        if not voices:
            self.voice_combo.addItem("No voices available")
            self.voice_combo.setEnabled(False)
            return
        
        # Add all available voices
        selected_index = 0
        for i, voice in enumerate(voices):
            # Format the voice name for display
            if voice.startswith("pyttsx3:"):
                # For pyttsx3 voices, try to get a more readable name
                voice_id = voice.split(":", 1)[1]
                if hasattr(self.tts_adapter.tts_engine, 'getProperty'):
                    for v in self.tts_adapter.tts_engine.getProperty('voices'):
                        if v.id == voice_id:
                            display_name = f"{v.name} (pyttsx3)"
                            self.voice_combo.addItem(display_name, voice)
                            break
                    else:
                        self.voice_combo.addItem(voice, voice)
                else:
                    self.voice_combo.addItem(voice, voice)
            else:
                # For Kokoro voices, use the voice name directly
                self.voice_combo.addItem(voice, voice)
            
            # Select the current voice
            if voice == current_voice:
                selected_index = i
        
        # Enable the combo box now that it has real data
        self.voice_combo.setEnabled(True)
        
        # Set the current selection
        self.voice_combo.setCurrentIndex(selected_index)
        logger.info(f"Voice combo initialized with {len(voices)} voices, selected: {current_voice}")
        print(f"Voice combo initialized with {len(voices)} voices, selected: {current_voice}")
    
    def initialize_components(self):
        """Initialize components in background threads"""
        logger.info("Starting background initialization of components")
        
        # Start background threads for initialization
        threading.Thread(target=self.initialize_audio_processor, daemon=True).start()
        threading.Thread(target=self.initialize_tts_adapter, daemon=True).start()
    
    def initialize_audio_processor(self):
        """Initialize audio processor in background thread"""
        try:
            logger.info("Initializing audio processor in background")
            self.audio_processor = AudioProcessorUI()
            self.audio_processor.audio_level_updated.connect(self.update_audio_level)
            self.audio_processor.transcription_ready.connect(self.handle_transcription)
            self.audio_ready = True
            logger.info("Audio processor initialization complete")
            self.check_all_components_ready()
        except Exception as e:
            logger.error(f"Error initializing audio processor: {e}")
    
    def initialize_tts_adapter(self):
        """Initialize TTS adapter in background thread"""
        try:
            logger.info("Initializing TTS adapter in background")
            self.tts_adapter = TTSAdapter()
            self.tts_adapter.speaking_started.connect(self.on_speaking_started)
            self.tts_adapter.speaking_finished.connect(self.on_speaking_finished)
            
            # Connect audio level signal to agent visualizer
            self.tts_adapter.audio_level.connect(self.update_agent_audio_level)
            
            # Create audio level timer if it doesn't exist yet
            if not hasattr(self.tts_adapter, 'audio_level_timer'):
                self.tts_adapter.audio_level_timer = QTimer()
                self.tts_adapter.audio_level_timer.timeout.connect(self.tts_adapter.emit_audio_level)
                logger.info("Created audio level timer for TTS visualization")
            
            self.tts_ready = True
            logger.info("TTS adapter initialization complete")
            
            # Update voice list when TTS is ready - use QTimer to call from main thread
            QTimer.singleShot(0, self.update_voice_list)
            
            self.check_all_components_ready()
        except Exception as e:
            logger.error(f"Error initializing TTS adapter: {e}")
    
    def check_all_components_ready(self):
        """Check if all components are ready and emit signal if they are"""
        if self.audio_ready and self.tts_ready:
            logger.info("All components initialized successfully")
            # Use QTimer to safely emit signal from background thread
            QTimer.singleShot(0, lambda: self.components_ready.emit())
    
    def on_components_ready(self):
        """Called when all components are ready"""
        logger.info("All components are ready, updating UI")
        
        # Clear initialization message from transcription label
        self.transcription_label.setText("Ready for voice interaction")
        
        # Check for any pending commands
        if os.path.exists(COMMAND_FILE):
            try:
                with open(COMMAND_FILE, 'r') as f:
                    command = f.read().strip()
                    if command == "LISTEN" and self.has_saved_voice_preference():
                        # Start listening since we have a saved voice preference
                        self.start_listening()
            except Exception as e:
                logger.error(f"Error reading command file: {e}")
        
        # If no voice preference is saved, show guidance message
        if not self.has_saved_voice_preference():
            self.transcription_label.setText("Please select a voice from the dropdown and click 'Save Voice' to continue")
            # Wait a moment before speaking to ensure UI is fully ready
            QTimer.singleShot(500, self.play_guidance_message)
    
    def has_saved_voice_preference(self):
        """Check if a voice preference has been saved"""
        try:
            # First check environment variable
            env_voice = get_env_setting(ENV_TTS_VOICE)
            if env_voice:
                logger.info(f"Found voice preference in environment variable: {env_voice}")
                return True
                
            # Then check config file
            config_voice = get_setting("tts", "voice", None)
            if config_voice:
                logger.info(f"Found voice preference in config file: {config_voice}")
                return True
                
            logger.info("No saved voice preference found")
            return False
        except ImportError:
            logger.warning("Config module not available, assuming no voice preference")
            return False
        except Exception as e:
            logger.error(f"Error checking for saved voice preference: {e}")
            return False
    
    def save_voice_preference(self, voice):
        """Save the selected voice preference to config"""
        try:
            # Save to config file
            result = set_setting("tts", "voice", voice)
            
            # Also set environment variable for current session
            set_env_setting(ENV_TTS_VOICE, voice)
            
            logger.info(f"Voice preference saved: {voice}")
            return result
        except ImportError:
            logger.error("Config module not available, cannot save voice preference")
            return False
        except Exception as e:
            logger.error(f"Error saving voice preference: {e}")
            return False
    
    def save_selected_voice(self):
        """Save the selected voice and switch to listen mode"""
        # Get the currently selected voice
        index = self.voice_combo.currentIndex()
        if index < 0:
            logger.warning("No voice selected")
            self.transcription_label.setText("Please select a voice from the dropdown")
            return
        
        voice = self.voice_combo.itemData(index)
        if not voice:
            logger.warning("Invalid voice selection")
            self.transcription_label.setText("Please select a valid voice from the dropdown")
            return
        
        logger.info(f"Saving voice preference: {voice}")
        
        # Save the voice preference
        if self.save_voice_preference(voice):
            logger.info("Voice preference saved successfully")
            self.transcription_label.setText(f"Voice '{voice}' saved as your preference")
            
            # Create a UI_READY command file to signal back to the server
            try:
                with open(COMMAND_FILE, 'w') as f:
                    f.write(CMD_UI_READY)
                logger.info("Created UI_READY command file after voice selection")
            except Exception as e:
                logger.error(f"Error creating command file: {e}")
            
            # Test the voice to confirm
            QTimer.singleShot(1000, lambda: self.tts_adapter.speak("Voice preference saved. You can now start listening."))
        else:
            logger.error("Failed to save voice preference")
            self.transcription_label.setText("Failed to save voice preference. Please try again.")
    
    def play_guidance_message(self):
        """Play a guidance message for first-time users"""
        if hasattr(self, 'tts_adapter') and self.tts_adapter:
            # Add a highlight effect to the Select Voice button
            original_style = self.select_voice_button.styleSheet()
            highlight_style = """
                background-color: #e74c3c;
                color: white;
                border: 2px solid #f39c12;
                border-radius: 5px;
                padding: 8px 16px;
                font-weight: bold;
            """
            self.select_voice_button.setStyleSheet(highlight_style)
            
            # Speak the guidance message
            self.tts_adapter.speak("Please select a voice from the dropdown menu and click Save Voice to continue.")
            logger.info("Played guidance message for first-time user")
            
            # Restore the original style after a delay
            QTimer.singleShot(3000, lambda: self.select_voice_button.setStyleSheet(original_style))
    
    
    def on_voice_changed(self, index):
        """Handle voice selection change"""
        # Skip if TTS adapter is not ready yet
        if not hasattr(self, 'tts_adapter') or not self.tts_adapter:
            return
            
        if index < 0:
            return
        
        voice = self.voice_combo.itemData(index)
        if not voice:
            return
        
        logger.info(f"Voice selection changed to: {voice}")
        self.tts_adapter.set_voice(voice)
    
    def test_voice(self):
        """Test the selected voice"""
        print("Test voice button clicked!")
        logger.info("Test voice button clicked!")
        
        # Skip if TTS adapter is not ready yet
        if not hasattr(self, 'tts_adapter') or not self.tts_adapter:
            logger.warning("Test voice button clicked but TTS not ready yet")
            self.transcription_label.setText("TTS not ready yet. Please wait...")
            return
            
        logger.info(f"TTS adapter exists: {self.tts_adapter is not None}")
        logger.info(f"TTS engine exists: {self.tts_adapter.tts_engine is not None}")
        logger.info(f"Is speaking: {self.tts_adapter.is_speaking}")
        
        if self.tts_adapter.is_speaking:
            logger.warning("Already speaking, ignoring test request")
            return
        
        # Update the transcription label to show we're testing the voice
        self.transcription_label.setText("Testing voice...")
        
        # Log the current voice being tested
        current_voice = self.tts_adapter.get_current_voice()
        logger.info(f"Testing voice: {current_voice}")
        print(f"Testing voice: {current_voice}")
        
        # Start the agent animation timer before speaking
        if not hasattr(self, 'agent_animation_timer'):
            self.agent_animation_timer = QTimer(self)
            self.agent_animation_timer.timeout.connect(self.animate_agent_visualizer)
        self.agent_animation_timer.start(50)  # Update every 50ms
        
        # Activate agent visualizer
        self.set_agent_visualizer_active(True)
        self.set_user_visualizer_active(False)
        
        # Speak a test message
        try:
            logger.info("Attempting to speak test message")
            result = self.tts_adapter.speak("This is a test of the selected voice. Hello, I am Goose!")
            logger.info(f"TTS speak result: {result}")
            print(f"TTS speak result: {result}")
            
            if not result:
                logger.error("Failed to start speaking test message")
                self.transcription_label.setText("Error: Failed to test voice")
                QTimer.singleShot(2000, lambda: self.transcription_label.setText("Select a voice and click 'Test Voice' to hear it"))
                
                # Stop the animation if speaking failed
                if hasattr(self, 'agent_animation_timer') and self.agent_animation_timer.isActive():
                    self.agent_animation_timer.stop()
                    self.agent_visualizer.update_level(0.0)
        except Exception as e:
            logger.error(f"Exception during test voice: {e}", exc_info=True)
            print(f"Exception during test voice: {e}")
            self.transcription_label.setText(f"Error: {str(e)}")
            QTimer.singleShot(3000, lambda: self.transcription_label.setText("Select a voice and click 'Test Voice' to hear it"))
            
            # Stop the animation if an exception occurred
            if hasattr(self, 'agent_animation_timer') and self.agent_animation_timer.isActive():
                self.agent_animation_timer.stop()
                self.agent_visualizer.update_level(0.0)
    
    def update_audio_level(self, level):
        """Update the user audio level visualization."""
        self.user_visualizer.update_level(level)
    
    def update_agent_audio_level(self, level):
        """Update the agent audio level visualization."""
        self.agent_visualizer.update_level(level)
    
    def handle_transcription(self, text):
        """Handle new transcription text."""
        self.transcription_label.setText(f"You: {text}")
        logger.info(f"New transcription: {text}")
    
    def start_listening(self):
        """Start listening mode."""
        # Skip if audio processor is not ready yet
        if not hasattr(self, 'audio_processor') or not self.audio_processor:
            self.transcription_label.setText("Speech recognition not ready yet")
            return
            
        self.audio_processor.start_listening()
        
        # Activate user visualizer, deactivate agent visualizer
        self.set_user_visualizer_active(True)
        self.set_agent_visualizer_active(False)
    
    def stop_listening(self):
        """Stop listening mode."""
        # Skip if audio processor is not ready yet
        if not hasattr(self, 'audio_processor') or not self.audio_processor:
            return
            
        self.audio_processor.stop_listening()
        
        # Deactivate user visualizer
        self.set_user_visualizer_active(False)
    
    def on_speaking_started(self):
        """Called when speaking starts."""
        self.speak_button.setEnabled(False)
        
        # Record when speaking started for the watchdog timer
        self._speaking_start_time = time.time()
        
        # Activate agent visualizer, deactivate user visualizer
        self.set_agent_visualizer_active(True)
        self.set_user_visualizer_active(False)
        
        # Start a timer to animate the agent visualizer
        if not hasattr(self, 'agent_animation_timer'):
            self.agent_animation_timer = QTimer(self)
            self.agent_animation_timer.timeout.connect(self.animate_agent_visualizer)
        self.agent_animation_timer.start(50)  # Update every 50ms
        
    def on_speaking_finished(self):
        """Called when speaking finishes."""
        self.speak_button.setEnabled(True)
        
        # Clear the speaking start time
        if hasattr(self, '_speaking_start_time'):
            del self._speaking_start_time
        
        # Stop the agent animation timer
        if hasattr(self, 'agent_animation_timer') and self.agent_animation_timer.isActive():
            self.agent_animation_timer.stop()
        
        # Deactivate agent visualizer
        self.set_agent_visualizer_active(False)
            
    def animate_agent_visualizer(self):
        """Animate the agent visualizer with dynamic levels"""
        # Create a dynamic wave pattern
        t = time.time() * 5.0
        base_level = 0.5 + 0.3 * math.sin(t * 1.5)
        variation = 0.2 * random.random()
        level = base_level + variation
        
        # Ensure level stays within bounds
        level = max(0.1, min(0.95, level))
        
        # Update the agent visualizer
        self.agent_visualizer.update_level(level)
    
    def check_for_commands(self):
        """Check for commands from the server."""
        if os.path.exists(COMMAND_FILE):
            try:
                with open(COMMAND_FILE, 'r') as f:
                    command = f.read().strip()
                
                # Process the command
                if command == CMD_LISTEN:
                    logger.info("Received LISTEN command")
                    # If components are not ready, store the command to process later
                    if not hasattr(self, 'audio_processor') or not self.audio_processor:
                        logger.info("Components not ready yet, will process LISTEN command when ready")
                        # Command will be processed in on_components_ready
                        return
                    
                    # Only start listening if we have a saved voice preference
                    if self.has_saved_voice_preference():
                        self.start_listening()
                    else:
                        logger.info("Ignoring LISTEN command because no voice preference is saved")
                        # Show guidance message instead
                        self.transcription_label.setText("Please select a voice from the dropdown and click 'Select Voice' to continue")
                        # Wait a moment before speaking to ensure UI is fully ready
                        QTimer.singleShot(500, self.play_guidance_message)
                        
                elif command == CMD_IDLE and hasattr(self, 'audio_processor') and self.audio_processor and self.audio_processor.is_listening:
                    logger.info("Received IDLE command")
                    self.stop_listening()
                elif command == CMD_SPEAK:
                    logger.info("Received SPEAK command")
                    # We'll handle speaking in check_for_responses
                    if hasattr(self, 'tts_adapter') and self.tts_adapter:
                        # Activate agent visualizer
                        self.set_agent_visualizer_active(True)
                        self.set_user_visualizer_active(False)
            except Exception as e:
                logger.error(f"Error reading command file: {e}")
    
    def check_for_responses(self):
        """Check for response files to speak."""
        if os.path.exists(RESPONSE_FILE):
            try:
                # Read the response
                with open(RESPONSE_FILE, 'r') as f:
                    response = f.read().strip()
                
                logger.info(f"Found response to speak: {response[:50]}{'...' if len(response) > 50 else ''}")
                
                # Delete the file immediately to prevent duplicate processing
                try:
                    os.remove(RESPONSE_FILE)
                except Exception as e:
                    logger.warning(f"Error removing response file: {e}")
                
                # If TTS is not ready yet, show a message and return
                if not hasattr(self, 'tts_adapter') or not self.tts_adapter:
                    logger.warning("TTS not ready yet, cannot speak response")
                    self.transcription_label.setText("Response received but TTS not ready yet")
                    return
                
                # Display the response text in the transcription label
                self.transcription_label.setText(f"Agent: {response}")
                
                # Start the agent animation timer before speaking
                # This ensures the visualization works even if the TTS signal connection fails
                if not hasattr(self, 'agent_animation_timer'):
                    self.agent_animation_timer = QTimer(self)
                    self.agent_animation_timer.timeout.connect(self.animate_agent_visualizer)
                self.agent_animation_timer.start(50)  # Update every 50ms
                
                # Speak the response using the TTS adapter
                if response:
                    self.tts_adapter.speak(response)
                
            except Exception as e:
                logger.error(f"Error processing response file: {e}")
                self.transcription_label.setText(f"Error processing response: {str(e)}")
                QTimer.singleShot(3000, lambda: self.transcription_label.setText("Ready for voice interaction"))
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Stop audio processor if it exists
        if hasattr(self, 'audio_processor') and self.audio_processor:
            self.audio_processor.stop_listening()
        
        # Write a UI_CLOSED command to the command file
        try:
            with open(COMMAND_FILE, 'w') as f:
                f.write(CMD_UI_CLOSED)
            logger.info("Created UI_CLOSED command file")
        except Exception as e:
            logger.error(f"Error creating command file: {e}")
        
        super().closeEvent(event)
    
    def check_ui_responsiveness(self):
        """Check if UI is responsive and reset state if needed."""
        # Check if TTS adapter is in a stuck state
        if hasattr(self, 'tts_adapter') and self.tts_adapter:
            # Use the lock to safely check the speaking state
            with self.tts_adapter._speaking_lock:
                is_speaking = self.tts_adapter.is_speaking
            
            # If speaking state has been active for too long, reset it
            if is_speaking and hasattr(self, '_speaking_start_time'):
                duration = time.time() - self._speaking_start_time
                if duration > 30:  # 30 seconds max for speaking
                    logger.warning(f"Speaking state active for {duration:.1f}s, resetting")
                    with self.tts_adapter._speaking_lock:
                        self.tts_adapter.is_speaking = False
                    self.on_speaking_finished()
            elif is_speaking:
                # Record when speaking started
                self._speaking_start_time = time.time()
            else:
                # Clear the timestamp when not speaking
                if hasattr(self, '_speaking_start_time'):
                    del self._speaking_start_time


def run_ui():
    """Run the PyQt speech UI."""
    app = QApplication(sys.argv)
    window = PyQtSpeechUI()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    # Run the UI
    sys.exit(run_ui())