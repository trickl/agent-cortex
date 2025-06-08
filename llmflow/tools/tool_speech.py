"""
LLMFlow - A powerful framework for building AI agents based on GAME methodology
(Goals, Actions, Memory, Environment).

Speech Tool - Handles speech recognition and processing with support for multiple languages and audio formats.
"""

import os
import io
import json
import time
import wave
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
import numpy as np
import logging
from dataclasses import dataclass, asdict
import hashlib
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Core audio libraries
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    from pydub import AudioSegment
    from pydub.playback import play
    from pydub.effects import normalize, compress_dynamic_range
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

from ..core.base import BaseTool, ToolResponse

@dataclass
class TTSConfig:
    """Configuration for Text-to-Speech."""
    engine: str = 'pyttsx3'  # pyttsx3, gtts, azure, aws, custom
    voice_id: Optional[str] = None
    language: str = 'en'
    speed: float = 150  # words per minute
    pitch: float = 0    # -50 to 50
    volume: float = 0.9  # 0.0 to 1.0
    output_format: str = 'wav'  # wav, mp3, ogg
    quality: str = 'medium'  # low, medium, high
    ssml_enabled: bool = False

@dataclass
class STTConfig:
    """Configuration for Speech-to-Text."""
    engine: str = 'speech_recognition'  # speech_recognition, whisper, azure, aws, custom
    language: str = 'en-US'
    model_size: str = os.getenv('WHISPER_MODEL', 'base')  # tiny, base, small, medium, large (for whisper)
    timeout: float = 5.0
    phrase_timeout: float = 1.0
    energy_threshold: int = 300
    dynamic_energy_threshold: bool = True
    pause_threshold: float = 0.8

@dataclass
class AudioMetadata:
    """Audio file metadata."""
    filename: str
    duration: float
    sample_rate: int
    channels: int
    bit_depth: int
    format: str
    file_size: int
    created_at: str 

class TextToSpeechSpeechToTextTool(BaseTool):
    """Text-to-Speech and Speech-to-Text tool implementation."""
    
    def __init__(self):
        super().__init__()
        self.output_dir = Path('speech_output')
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize engines
        self.tts_engine = None
        self.stt_engine = None
        
        # Load API keys
        self.google_speech_api_key = os.getenv('GOOGLE_SPEECH_API_KEY')
        self.azure_speech_key = os.getenv('AZURE_SPEECH_KEY')
        self.azure_speech_region = os.getenv('AZURE_SPEECH_REGION')
        
        if PYTTSX3_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
            except Exception as e:
                self.logger.warning(f"Failed to initialize pyttsx3: {e}")
        
        if SPEECH_RECOGNITION_AVAILABLE:
            try:
                self.stt_engine = sr.Recognizer()
            except Exception as e:
                self.logger.warning(f"Failed to initialize speech recognition: {e}")
    
    def execute(self, operation: str, **kwargs) -> ToolResponse:
        """Execute a speech processing operation."""
        start_time = time.time()
        
        try:
            if operation == 'text_to_speech':
                if not self.tts_engine and not GTTS_AVAILABLE:
                    raise ValueError("No TTS engine available")
                
                text = kwargs.get('text')
                if not text:
                    raise ValueError("Text parameter is required")
                
                # Generate output path
                output_path = self.output_dir / f"tts_{int(time.time())}.wav"
                
                # Convert text to speech
                if self.tts_engine:
                    self.tts_engine.save_to_file(text, str(output_path))
                    self.tts_engine.runAndWait()
                else:
                    tts = gTTS(text=text, lang='en')
                    tts.save(str(output_path))
                
                return ToolResponse(
                    success=True,
                    result={'output_path': str(output_path)},
                    execution_time=time.time() - start_time
                )
            
            elif operation == 'speech_to_text':
                if not self.stt_engine and not WHISPER_AVAILABLE:
                    raise ValueError("No STT engine available")
                
                audio_path = kwargs.get('audio_path')
                if not audio_path:
                    raise ValueError("Audio path parameter is required")
                
                # Convert speech to text
                text = ""
                if self.stt_engine:
                    with sr.AudioFile(audio_path) as source:
                        audio = self.stt_engine.record(source)
                        text = self.stt_engine.recognize_google(audio)
                elif WHISPER_AVAILABLE:
                    model = whisper.load_model("base")
                    result = model.transcribe(audio_path)
                    text = result["text"]
                
                return ToolResponse(
                    success=True,
                    result={'text': text},
                    execution_time=time.time() - start_time
                )
            
            else:
                raise ValueError(f"Unsupported operation: {operation}")
            
        except Exception as e:
            self.logger.error(f"Error in speech processing: {str(e)}")
            return ToolResponse(
                success=False,
                result=None,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    @staticmethod
    def get_schema() -> Dict[str, Any]:
        """Get the tool's JSON schema."""
        return {
            'type': 'function',
            'function': {
                'name': 'speech_processing',
                'description': 'Process text-to-speech and speech-to-text conversions',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'operation': {
                            'type': 'string',
                            'enum': ['text_to_speech', 'speech_to_text'],
                            'description': 'The speech processing operation to perform'
                        },
                        'text': {
                            'type': 'string',
                            'description': 'Text to convert to speech (for text_to_speech operation)'
                        },
                        'audio_path': {
                            'type': 'string',
                            'description': 'Path to the audio file (for speech_to_text operation)'
                        }
                    },
                    'required': ['operation']
                }
            }
        }

# Agent framework integration
class TextToSpeechSpeechToTextAgent:
    """
    Agent wrapper for the TTS/STT tool.
    """
    
    def __init__(self):
        self.tool = TextToSpeechSpeechToTextTool()
        self.capabilities = [
            'text_to_speech',
            'speech_to_text',
            'record_audio',
            'play_audio',
            'convert_audio',
            'enhance_audio',
            'batch_tts',
            'batch_stt',
            'configure_tts',
            'configure_stt',
            'get_voices',
            'get_languages',
            'get_stats'
        ]
    
    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        """Execute a specific audio operation."""
        try:
            if action == 'text_to_speech':
                return self.tool.text_to_speech(**kwargs)
            elif action == 'speech_to_text':
                return self.tool.speech_to_text(**kwargs)
            elif action == 'record_audio':
                return self.tool.record_audio(**kwargs)
            elif action == 'play_audio':
                return self.tool.play_audio(**kwargs)
            elif action == 'convert_audio':
                return self.tool.convert_audio_format(**kwargs)
            elif action == 'enhance_audio':
                return self.tool.enhance_audio(**kwargs)
            elif action == 'batch_tts':
                return self.tool.batch_text_to_speech(**kwargs)
            elif action == 'batch_stt':
                return self.tool.batch_speech_to_text(**kwargs)
            elif action == 'configure_tts':
                return self.tool.configure_tts(**kwargs)
            elif action == 'configure_stt':
                return self.tool.configure_stt(**kwargs)
            elif action == 'get_voices':
                return self.tool.get_available_voices()
            elif action == 'get_languages':
                return self.tool.get_supported_languages()
            elif action == 'get_stats':
                return self.tool.get_processing_stats()
            elif action == 'get_history':
                return self.tool.get_history(**kwargs)
            elif action == 'cleanup':
                return self.tool.cleanup_temp_files()
            else:
                return {'status': 'error', 'message': f'Unknown action: {action}'}
                
        except Exception as e:
            return {'status': 'error', 'message': f'Error executing {action}: {str(e)}'}
    
    def get_capabilities(self) -> List[str]:
        """Return list of available capabilities."""
        return self.capabilities.copy()


# Quick utility functions
def quick_tts(text: str, output_file: str = None, voice: str = None, 
              speed: float = 150) -> Dict[str, Any]:
    """Quick text-to-speech conversion."""
    tool = TextToSpeechSpeechToTextTool()
    
    config_params = {'speed': speed}
    if voice:
        config_params['voice_id'] = voice
    
    tool.configure_tts(**config_params)
    return tool.text_to_speech(text, output_file)

def quick_stt(audio_file: str, engine: str = 'speech_recognition') -> Dict[str, Any]:
    """Quick speech-to-text conversion."""
    tool = TextToSpeechSpeechToTextTool()
    tool.configure_stt(engine=engine)
    return tool.speech_to_text(audio_file)

def quick_record_and_transcribe(duration: float = 5.0) -> Dict[str, Any]:
    """Quick record and transcribe function."""
    tool = TextToSpeechSpeechToTextTool()
    
    # Record audio
    record_result = tool.record_audio(duration=duration)
    if record_result['status'] != 'success':
        return record_result
    
    # Transcribe
    transcribe_result = tool.speech_to_text(record_result['output_file'])
    
    return {
        'status': 'success',
        'recording': record_result,
        'transcription': transcribe_result,
        'final_text': transcribe_result.get('text', '')
    } 