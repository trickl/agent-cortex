"""
LLMFlow - A powerful framework for building AI agents based on GAME methodology
(Goals, Actions, Memory, Environment).

Text-to-Speech Tool - Converts text to speech with support for multiple languages, voices, and audio output formats.
"""

import os
import tempfile
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Try importing required libraries with fallbacks
TTS_AVAILABLE = False
SPEECH_RECOGNITION_AVAILABLE = False
AUDIO_PLAYBACK_AVAILABLE = False

try:
    import pyttsx3
    from gtts import gTTS
    TTS_AVAILABLE = True
except ImportError:
    pyttsx3 = None
    gTTS = None

try:
    import pygame
    AUDIO_PLAYBACK_AVAILABLE = True
except ImportError:
    pygame = None

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    sr = None

# Try importing Whisper (optional, more advanced STT)
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    whisper = None
    WHISPER_AVAILABLE = False

class TextToSpeechSpeechToTextTool:
    """Tool for text-to-speech and speech-to-text conversion."""
    
    def __init__(self):
        """Initialize the tool."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize TTS engines
        self.pyttsx3_engine = None
        if TTS_AVAILABLE and pyttsx3:
            try:
                self.pyttsx3_engine = pyttsx3.init()
                # Configure voice settings
                self.pyttsx3_engine.setProperty('rate', 150)  # Speed of speech
                self.pyttsx3_engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)
            except Exception as e:
                self.logger.warning(f"Failed to initialize pyttsx3: {e}")
                self.pyttsx3_engine = None
        
        # Initialize pygame for audio playback
        if AUDIO_PLAYBACK_AVAILABLE and pygame:
            try:
                pygame.mixer.init()
            except Exception as e:
                self.logger.warning(f"Failed to initialize pygame mixer: {e}")
        
        # Initialize speech recognition
        self.recognizer = None
        if SPEECH_RECOGNITION_AVAILABLE and sr:
            try:
                self.recognizer = sr.Recognizer()
            except Exception as e:
                self.logger.warning(f"Failed to initialize speech recognizer: {e}")
        
        # Initialize Whisper model (if available)
        self.whisper_model = None
        if WHISPER_AVAILABLE:
            try:
                self.whisper_model = whisper.load_model("base")
            except Exception as e:
                self.logger.warning(f"Failed to load Whisper model: {e}")
        
    def text_to_speech(self, text: str, language: str = "en", save_file: Optional[str] = None, 
                      engine: str = "gtts") -> Dict[str, Any]:
        """Convert text to speech and optionally save to file.
        
        Args:
            text: Text to convert to speech
            language: Language code (e.g., 'en', 'ru', 'es')
            save_file: Path to save audio file (optional)
            engine: TTS engine to use ('gtts', 'pyttsx3')
        """
        try:
            if not text.strip():
                return {
                    "success": False,
                    "error": "Empty text provided"
                }
            
            if engine == "gtts" and TTS_AVAILABLE and gTTS:
                return self._gtts_convert(text, language, save_file)
            elif engine == "pyttsx3" and self.pyttsx3_engine:
                return self._pyttsx3_convert(text, save_file)
            else:
                # Fallback to mock if libraries not available
                return self._mock_tts(text, save_file)
                
        except Exception as e:
            self.logger.error(f"Error in text_to_speech: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _gtts_convert(self, text: str, language: str, save_file: Optional[str]) -> Dict[str, Any]:
        """Convert text to speech using Google TTS."""
        try:
            # Create temporary file if no save file specified
            if not save_file:
                temp_dir = tempfile.gettempdir()
                save_file = os.path.join(temp_dir, f"tts_output_{os.getpid()}.mp3")
            
            # Create TTS object and save
            tts = gTTS(text=text, lang=language, slow=False)
            tts.save(save_file)
            
            # Play the audio if pygame is available
            if AUDIO_PLAYBACK_AVAILABLE and pygame and pygame.mixer.get_init():
                try:
                    pygame.mixer.music.load(save_file)
                    pygame.mixer.music.play()
                    
                    # Wait for playback to finish
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                        
                    playback_status = "Audio played successfully"
                except Exception as e:
                    playback_status = f"Audio saved but playback failed: {e}"
            else:
                playback_status = "Audio saved (no playback available)"
            
            return {
                "success": True,
                "message": f"Text converted to speech using Google TTS. {playback_status}",
                "audio_file": save_file,
                "engine": "gtts",
                "language": language,
                "text_length": len(text)
            }
            
        except Exception as e:
            raise Exception(f"gTTS conversion failed: {e}")
    
    def _pyttsx3_convert(self, text: str, save_file: Optional[str]) -> Dict[str, Any]:
        """Convert text to speech using pyttsx3."""
        try:
            if save_file:
                # Save to file
                self.pyttsx3_engine.save_to_file(text, save_file)
                self.pyttsx3_engine.runAndWait()
                message = f"Text converted to speech and saved to {save_file}"
            else:
                # Speak directly
                self.pyttsx3_engine.say(text)
                self.pyttsx3_engine.runAndWait()
                message = "Text spoken using pyttsx3"
                save_file = "direct_speech"
            
            return {
                "success": True,
                "message": message,
                "audio_file": save_file,
                "engine": "pyttsx3",
                "text_length": len(text)
            }
            
        except Exception as e:
            raise Exception(f"pyttsx3 conversion failed: {e}")
    
    def _mock_tts(self, text: str, save_file: Optional[str]) -> Dict[str, Any]:
        """Fallback mock implementation when libraries are not available."""
        return {
            "success": True,
            "message": f"Mock TTS: Would convert '{text[:50]}...' to speech (libraries not available)",
            "audio_file": save_file or "/tmp/mock_audio.mp3",
            "engine": "mock",
            "note": "Install 'gtts' and 'pygame' for real TTS functionality"
        }
    
    def speech_to_text(self, audio_file: str = None, language: str = "en", 
                      engine: str = "google") -> Dict[str, Any]:
        """Convert speech to text from microphone or audio file.
        
        Args:
            audio_file: Path to audio file (if None, uses microphone)
            language: Language code for recognition
            engine: STT engine to use ('google', 'whisper')
        """
        try:
            if audio_file:
                return self._transcribe_file(audio_file, language, engine)
            else:
                return self._transcribe_microphone(language, engine)
                
        except Exception as e:
            self.logger.error(f"Error in speech_to_text: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _transcribe_file(self, audio_file: str, language: str, engine: str) -> Dict[str, Any]:
        """Transcribe audio file to text."""
        if not os.path.exists(audio_file):
            return {
                "success": False,
                "error": f"Audio file not found: {audio_file}"
            }
        
        try:
            if engine == "whisper" and self.whisper_model:
                # Use Whisper for transcription
                result = self.whisper_model.transcribe(audio_file)
                return {
                    "success": True,
                    "text": result["text"].strip(),
                    "engine": "whisper",
                    "language": result.get("language", language),
                    "confidence": "high"
                }
            
            elif engine == "google" and SPEECH_RECOGNITION_AVAILABLE and self.recognizer:
                # Use SpeechRecognition with Google service
                with sr.AudioFile(audio_file) as source:
                    audio = self.recognizer.record(source)
                    text = self.recognizer.recognize_google(audio, language=language)
                    
                return {
                    "success": True,
                    "text": text,
                    "engine": "google",
                    "language": language
                }
            
            else:
                # Fallback mock
                return {
                    "success": True,
                    "text": f"Mock transcription of {Path(audio_file).name}",
                    "engine": "mock",
                    "note": "Install 'openai-whisper' or 'SpeechRecognition' for real STT"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Transcription failed: {e}"
            }
    
    def _transcribe_microphone(self, language: str, engine: str) -> Dict[str, Any]:
        """Transcribe from microphone input."""
        if not SPEECH_RECOGNITION_AVAILABLE or not self.recognizer:
            return {
                "success": False,
                "error": "SpeechRecognition library not available for microphone input"
            }
        
        try:
            with sr.Microphone() as source:
                self.logger.info("Listening... Speak now!")
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                # Listen for audio
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                
            # Transcribe
            if engine == "google":
                text = self.recognizer.recognize_google(audio, language=language)
            else:
                # Default to Google
                text = self.recognizer.recognize_google(audio, language=language)
            
            return {
                "success": True,
                "text": text,
                "engine": engine,
                "language": language,
                "source": "microphone"
            }
            
        except sr.WaitTimeoutError:
            return {
                "success": False,
                "error": "Listening timeout - no speech detected"
            }
        except sr.UnknownValueError:
            return {
                "success": False,
                "error": "Could not understand the speech"
            }
        except sr.RequestError as e:
            return {
                "success": False,
                "error": f"Recognition service error: {e}"
            }
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the tool's schema for the LLM."""
        return {
            "name": "text_to_speech_tool",
            "description": "A tool for converting text to speech and speech to text with multiple engine options",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["text_to_speech", "speech_to_text", "get_capabilities"],
                        "description": "The operation to perform"
                    },
                    "text": {
                        "type": "string",
                        "description": "The text to convert to speech (for text_to_speech operation)"
                    },
                    "language": {
                        "type": "string",
                        "description": "The language code (e.g., 'en' for English, 'ru' for Russian)",
                        "default": "en"
                    },
                    "audio_file": {
                        "type": "string",
                        "description": "Path to the audio file (for speech_to_text operation)"
                    },
                    "save_file": {
                        "type": "string",
                        "description": "Path to save the generated audio file (optional)"
                    },
                    "engine": {
                        "type": "string",
                        "description": "Engine to use: 'gtts'/'pyttsx3' for TTS, 'google'/'whisper' for STT",
                        "default": "gtts"
                    }
                },
                "required": ["operation"]
            }
        }
    
    def get_capabilities(self) -> Dict[str, bool]:
        """Get current capabilities based on available libraries."""
        return {
            "tts_gtts": TTS_AVAILABLE and gTTS is not None,
            "tts_pyttsx3": TTS_AVAILABLE and self.pyttsx3_engine is not None,
            "audio_playback": AUDIO_PLAYBACK_AVAILABLE and pygame is not None,
            "stt_google": SPEECH_RECOGNITION_AVAILABLE and self.recognizer is not None,
            "stt_whisper": WHISPER_AVAILABLE and self.whisper_model is not None,
            "microphone_input": SPEECH_RECOGNITION_AVAILABLE and self.recognizer is not None
        } 