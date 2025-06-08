"""
Whisper Transcription Tool for LLM Agents
=========================================

A Python tool for transcribing audio files using OpenAI's Whisper model.
Designed to be used as a tool in LLM agent frameworks.

Requirements:
    - openai-whisper (install with: pip install openai-whisper)
    - ffmpeg (for audio format conversion)
    - Python 3.7+

Example Usage:
    >>> transcriber = WhisperTranscriber()
    >>> result = transcriber.transcribe("audio.mp3")
    >>> print(result.text)

Author: AI Assistant
Version: 1.0.0
"""

import os
import json
import logging
import hashlib
import tempfile
import subprocess
import sys
import glob
from typing import Dict, Optional, Union, List, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import time

try:
    from .tool_decorator import register_tool
except ImportError:
    # If tool_decorator doesn't exist, create a dummy decorator
    def register_tool(tags=None):
        def decorator(func):
            return func
        return decorator

# Check if whisper is installed
try:
    import whisper
except ImportError:
    print("openai-whisper is not installed. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai-whisper"])
    import whisper

# Check if torch is available for GPU support
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False


class WhisperModel(Enum):
    """Available Whisper model sizes."""
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"


class OutputFormat(Enum):
    """Output format options for transcription."""
    TEXT = "text"
    JSON = "json"
    SRT = "srt"
    VTT = "vtt"
    TSV = "tsv"


@dataclass
class TranscriptionOptions:
    """Configuration options for audio transcription."""
    model_size: WhisperModel = WhisperModel.MEDIUM
    language: Optional[str] = None  # Auto-detect if None
    task: str = "transcribe"  # "transcribe" or "translate"
    temperature: float = 0.0
    compression_ratio_threshold: float = 2.4
    logprob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    condition_on_previous_text: bool = True
    initial_prompt: Optional[str] = None
    word_timestamps: bool = False
    prepend_punctuations: str = "\"'([{-"
    append_punctuations: str = "\"'.,:;)]}?"
    output_format: OutputFormat = OutputFormat.TEXT
    verbose: bool = False
    fp16: bool = True  # Use FP16 for faster inference
    threads: int = 0  # 0 = use all available cores
    beam_size: int = 5
    best_of: int = 5
    patience: float = 1.0
    length_penalty: float = 1.0
    suppress_tokens: str = "-1"
    suppress_blank: bool = True
    without_timestamps: bool = False
    max_initial_timestamp: float = 1.0
    highlight_words: bool = False
    max_line_width: int = 0
    max_line_count: int = 0
    max_words_per_line: int = 0


@dataclass
class TranscriptionSegment:
    """Represents a segment of transcribed text."""
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float


@dataclass
class TranscriptionResult:
    """Result of a transcription operation."""
    success: bool
    text: Optional[str] = None
    segments: List[TranscriptionSegment] = field(default_factory=list)
    language: Optional[str] = None
    duration: Optional[float] = None
    audio_file: Optional[str] = None
    model_used: Optional[str] = None
    processing_time: Optional[float] = None
    output_file: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AudioConverter:
    """Handles audio format conversion using ffmpeg."""
    
    SUPPORTED_FORMATS = {'.mp3', '.mp4', '.wav', '.m4a', '.flac', '.aac', 
                        '.ogg', '.opus', '.oga', '.webm', '.wma', '.ac3',
                        '.aiff', '.amr', '.ape', '.au', '.dts', '.m4b',
                        '.mka', '.mp2', '.mpc', '.ra', '.tta', '.voc', '.wv'}
    
    @staticmethod
    def check_ffmpeg() -> bool:
        """Check if ffmpeg is available."""
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    @staticmethod
    def get_audio_duration(file_path: str) -> Optional[float]:
        """Get duration of audio file in seconds."""
        try:
            cmd = [
                'ffprobe', '-v', 'error', '-show_entries',
                'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
                file_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception:
            pass
        return None
    
    @staticmethod
    def convert_to_wav(input_file: str, output_file: str) -> bool:
        """Convert audio file to WAV format for Whisper."""
        try:
            cmd = [
                'ffmpeg', '-i', input_file, '-ar', '16000',
                '-ac', '1', '-c:a', 'pcm_s16le', '-y', output_file
            ]
            result = subprocess.run(cmd, 
                                  stdout=subprocess.DEVNULL, 
                                  stderr=subprocess.DEVNULL)
            return result.returncode == 0
        except Exception:
            return False


class ModelManager:
    """Manages Whisper model downloading and caching."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize model manager with cache directory."""
        self.cache_dir = cache_dir or os.path.join(
            os.path.expanduser("~"), ".cache", "whisper"
        )
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        self.loaded_models = {}
    
    def get_model_path(self, model_size: WhisperModel) -> str:
        """Get the expected path for a model file."""
        return os.path.join(self.cache_dir, f"{model_size.value}.pt")
    
    def is_model_cached(self, model_size: WhisperModel) -> bool:
        """Check if a model is already downloaded."""
        model_path = self.get_model_path(model_size)
        return os.path.exists(model_path)
    
    def load_model(self, model_size: WhisperModel, 
                   device: Optional[str] = None,
                   download_root: Optional[str] = None) -> Any:
        """Load a Whisper model, downloading if necessary."""
        model_name = model_size.value
        
        # Return cached loaded model if available
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        # Determine device
        if device is None:
            device = "cuda" if CUDA_AVAILABLE else "cpu"
        
        # Load model
        try:
            model = whisper.load_model(
                model_name, 
                device=device,
                download_root=download_root or self.cache_dir
            )
            self.loaded_models[model_name] = model
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {str(e)}")
    
    def unload_model(self, model_size: WhisperModel) -> None:
        """Remove a model from memory (but not from disk)."""
        model_name = model_size.value
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            if CUDA_AVAILABLE:
                torch.cuda.empty_cache()


class WhisperTranscriber:
    """
    A Whisper transcription tool for LLM agents.
    
    This class provides a high-level interface for transcribing audio files
    using OpenAI's Whisper model with automatic format conversion.
    
    Attributes:
        options (TranscriptionOptions): Configuration options for transcription
        logger (logging.Logger): Logger instance for this class
        model_manager (ModelManager): Manages model loading and caching
        converter (AudioConverter): Handles audio format conversion
    """
    
    def __init__(
        self,
        options: Optional[TranscriptionOptions] = None,
        logger: Optional[logging.Logger] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the Whisper transcriber.
        
        Args:
            options: Transcription configuration options
            logger: Logger instance (creates default if None)
            cache_dir: Directory for caching models
        """
        self.options = options or TranscriptionOptions()
        self.logger = logger or self._setup_logger()
        self.model_manager = ModelManager(cache_dir)
        self.converter = AudioConverter()
        
        # Check ffmpeg availability
        if not self.converter.check_ffmpeg():
            self.logger.warning(
                "ffmpeg not found. Audio format conversion will be limited."
            )
    
    def _setup_logger(self) -> logging.Logger:
        """Set up default logger configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _prepare_audio_file(self, audio_file: str) -> Tuple[str, bool]:
        """
        Prepare audio file for transcription, converting if necessary.
        
        Args:
            audio_file: Path to input audio file
            
        Returns:
            Tuple of (prepared_file_path, needs_cleanup)
        """
        file_ext = Path(audio_file).suffix.lower()
        
        # Check if file needs conversion
        if file_ext not in AudioConverter.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported audio format: {file_ext}")
        
        # Whisper works best with WAV files
        if file_ext != '.wav' and self.converter.check_ffmpeg():
            # Create temporary WAV file
            temp_wav = tempfile.NamedTemporaryFile(
                suffix='.wav', delete=False
            ).name
            
            self.logger.info(f"Converting {file_ext} to WAV format...")
            if self.converter.convert_to_wav(audio_file, temp_wav):
                return temp_wav, True
            else:
                os.unlink(temp_wav)
                self.logger.warning(
                    f"Failed to convert to WAV, using original file"
                )
        
        return audio_file, False
    
    def _format_output(
        self,
        result: Dict[str, Any],
        format: OutputFormat
    ) -> str:
        """Format transcription output according to specified format."""
        if format == OutputFormat.TEXT:
            return result["text"]
        
        elif format == OutputFormat.JSON:
            return json.dumps(result, indent=2, ensure_ascii=False)
        
        elif format == OutputFormat.SRT:
            return self._generate_srt(result["segments"])
        
        elif format == OutputFormat.VTT:
            return self._generate_vtt(result["segments"])
        
        elif format == OutputFormat.TSV:
            return self._generate_tsv(result["segments"])
        
        else:
            return result["text"]
    
    def _generate_srt(self, segments: List[Dict]) -> str:
        """Generate SRT subtitle format."""
        srt_lines = []
        for i, segment in enumerate(segments, 1):
            start = self._format_timestamp(segment["start"], srt=True)
            end = self._format_timestamp(segment["end"], srt=True)
            text = segment["text"].strip()
            srt_lines.append(f"{i}\n{start} --> {end}\n{text}\n")
        return "\n".join(srt_lines)
    
    def _generate_vtt(self, segments: List[Dict]) -> str:
        """Generate WebVTT subtitle format."""
        vtt_lines = ["WEBVTT\n"]
        for segment in segments:
            start = self._format_timestamp(segment["start"])
            end = self._format_timestamp(segment["end"])
            text = segment["text"].strip()
            vtt_lines.append(f"{start} --> {end}\n{text}\n")
        return "\n".join(vtt_lines)
    
    def _generate_tsv(self, segments: List[Dict]) -> str:
        """Generate TSV format."""
        tsv_lines = ["start\tend\ttext"]
        for segment in segments:
            start = int(segment["start"] * 1000)
            end = int(segment["end"] * 1000)
            text = segment["text"].strip().replace("\t", " ")
            tsv_lines.append(f"{start}\t{end}\t{text}")
        return "\n".join(tsv_lines)
    
    def _format_timestamp(self, seconds: float, srt: bool = False) -> str:
        """Format timestamp for subtitles."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        
        if srt:
            return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")
        else:
            return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    
    def transcribe(
        self,
        audio_file: str,
        options: Optional[TranscriptionOptions] = None,
        output_file: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Transcribe an audio file.
        
        Args:
            audio_file: Path to audio file
            options: Optional transcription options (overrides instance options)
            output_file: Optional path to save formatted output
            
        Returns:
            TranscriptionResult object with transcription information
        """
        # Check if file exists
        if not os.path.exists(audio_file):
            return TranscriptionResult(
                success=False,
                audio_file=audio_file,
                error_message=f"Audio file not found: {audio_file}"
            )
        
        # Use provided options or instance options
        opts = options or self.options
        
        # Record start time
        start_time = time.time()
        
        prepared_file = None
        needs_cleanup = False
        
        try:
            # Prepare audio file
            prepared_file, needs_cleanup = self._prepare_audio_file(audio_file)
            
            # Get audio duration
            duration = self.converter.get_audio_duration(prepared_file)
            
            # Load model
            self.logger.info(f"Loading Whisper {opts.model_size.value} model...")
            model = self.model_manager.load_model(opts.model_size)
            
            # Prepare decode options
            decode_options = {
                "task": opts.task,
                "language": opts.language,
                "temperature": opts.temperature,
                "compression_ratio_threshold": opts.compression_ratio_threshold,
                "logprob_threshold": opts.logprob_threshold,
                "no_speech_threshold": opts.no_speech_threshold,
                "condition_on_previous_text": opts.condition_on_previous_text,
                "initial_prompt": opts.initial_prompt,
                "word_timestamps": opts.word_timestamps,
                "prepend_punctuations": opts.prepend_punctuations,
                "append_punctuations": opts.append_punctuations,
                "verbose": opts.verbose,
                "fp16": opts.fp16 and CUDA_AVAILABLE,
                "beam_size": opts.beam_size,
                "best_of": opts.best_of,
                "patience": opts.patience,
                "length_penalty": opts.length_penalty,
                "suppress_tokens": opts.suppress_tokens,
                "suppress_blank": opts.suppress_blank,
                "without_timestamps": opts.without_timestamps,
                "max_initial_timestamp": opts.max_initial_timestamp,
            }
            
            # Remove None values
            decode_options = {k: v for k, v in decode_options.items() if v is not None}
            
            # Transcribe
            self.logger.info(f"Transcribing audio file: {audio_file}")
            result = model.transcribe(prepared_file, **decode_options)
            
            # Format output
            formatted_output = self._format_output(result, opts.output_format)
            
            # Save output if requested
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(formatted_output)
                self.logger.info(f"Transcription saved to: {output_file}")
            
            # Create segments
            segments = []
            for seg in result.get("segments", []):
                segments.append(TranscriptionSegment(
                    id=seg["id"],
                    seek=seg["seek"],
                    start=seg["start"],
                    end=seg["end"],
                    text=seg["text"],
                    tokens=seg["tokens"],
                    temperature=seg["temperature"],
                    avg_logprob=seg["avg_logprob"],
                    compression_ratio=seg["compression_ratio"],
                    no_speech_prob=seg["no_speech_prob"]
                ))
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            self.logger.info(
                f"Transcription completed in {processing_time:.2f} seconds"
            )
            
            return TranscriptionResult(
                success=True,
                text=result["text"],
                segments=segments,
                language=result.get("language"),
                duration=duration,
                audio_file=audio_file,
                model_used=opts.model_size.value,
                processing_time=processing_time,
                output_file=output_file,
                metadata=result
            )
            
        except Exception as e:
            self.logger.error(f"Transcription error: {str(e)}")
            return TranscriptionResult(
                success=False,
                audio_file=audio_file,
                error_message=f"Transcription error: {str(e)}"
            )
        
        finally:
            # Cleanup temporary file
            if needs_cleanup and prepared_file and os.path.exists(prepared_file):
                os.unlink(prepared_file)
    
    def transcribe_with_timestamps(
        self,
        audio_file: str,
        segment_level: bool = True,
        word_level: bool = False
    ) -> TranscriptionResult:
        """
        Transcribe with detailed timestamp information.
        
        Args:
            audio_file: Path to audio file
            segment_level: Include segment-level timestamps
            word_level: Include word-level timestamps
            
        Returns:
            TranscriptionResult with timestamp information
        """
        options = TranscriptionOptions(
            model_size=self.options.model_size,
            word_timestamps=word_level,
            output_format=OutputFormat.JSON if segment_level else OutputFormat.TEXT
        )
        
        return self.transcribe(audio_file, options)
    
    def detect_language(self, audio_file: str) -> Optional[str]:
        """
        Detect the language of an audio file.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Detected language code or None if detection fails
        """
        try:
            prepared_file, needs_cleanup = self._prepare_audio_file(audio_file)
            model = self.model_manager.load_model(self.options.model_size)
            
            # Load audio and detect language
            audio = whisper.load_audio(prepared_file)
            audio = whisper.pad_or_trim(audio)
            
            mel = whisper.log_mel_spectrogram(audio).to(model.device)
            _, probs = model.detect_language(mel)
            
            detected_language = max(probs, key=probs.get)
            
            if needs_cleanup and os.path.exists(prepared_file):
                os.unlink(prepared_file)
            
            return detected_language
            
        except Exception as e:
            self.logger.error(f"Language detection error: {str(e)}")
            return None


class WhisperTranscriberTool:
    """
    LLM Agent Tool wrapper for Whisper Transcriber.
    
    This class provides a simplified interface designed for use in LLM agent
    frameworks like LangChain, AutoGPT, etc.
    """
    
    def __init__(self, default_options: Optional[TranscriptionOptions] = None):
        """
        Initialize the tool.
        
        Args:
            default_options: Default transcription options
        """
        self.transcriber = WhisperTranscriber(default_options)
        self.name = "whisper_transcriber"
        self.description = (
            "Transcribe audio files to text using OpenAI's Whisper model. "
            "Input should be a path to an audio file. "
            "Returns transcribed text and metadata."
        )
    
    def run(self, audio_file: str, **kwargs) -> Dict[str, Any]:
        """
        Run the tool with an audio file.
        
        Args:
            audio_file: Path to audio file
            **kwargs: Additional options (language, output_format, etc.)
            
        Returns:
            Dictionary with transcription results
        """
        # Parse kwargs into options
        options = TranscriptionOptions()
        
        if kwargs.get('language'):
            options.language = kwargs['language']
        
        if kwargs.get('translate'):
            options.task = 'translate'
        
        if kwargs.get('output_format'):
            format_map = {
                'text': OutputFormat.TEXT,
                'json': OutputFormat.JSON,
                'srt': OutputFormat.SRT,
                'vtt': OutputFormat.VTT,
                'tsv': OutputFormat.TSV
            }
            options.output_format = format_map.get(
                kwargs['output_format'],
                OutputFormat.TEXT
            )
        
        if kwargs.get('model_size'):
            model_map = {
                'tiny': WhisperModel.TINY,
                'base': WhisperModel.BASE,
                'small': WhisperModel.SMALL,
                'medium': WhisperModel.MEDIUM,
                'large': WhisperModel.LARGE,
                'large-v2': WhisperModel.LARGE_V2,
                'large-v3': WhisperModel.LARGE_V3
            }
            options.model_size = model_map.get(
                kwargs['model_size'],
                WhisperModel.MEDIUM
            )
        
        # Transcribe the audio
        result = self.transcriber.transcribe(
            audio_file,
            options,
            kwargs.get('output_file')
        )
        
        # Convert to dictionary for agent compatibility
        return {
            'success': result.success,
            'text': result.text,
            'language': result.language,
            'duration': result.duration,
            'model_used': result.model_used,
            'processing_time': result.processing_time,
            'segments_count': len(result.segments),
            'error': result.error_message
        }
    
    async def arun(self, audio_file: str, **kwargs) -> Dict[str, Any]:
        """Async version of run (currently just calls sync version)."""
        return self.run(audio_file, **kwargs)
    
    def detect_language(self, audio_file: str) -> Dict[str, Any]:
        """
        Detect the language of an audio file.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Dictionary with language detection result
        """
        language = self.transcriber.detect_language(audio_file)
        
        return {
            'success': language is not None,
            'language': language,
            'error': None if language else 'Language detection failed'
        }


@register_tool(
    tags=["audio", "transcription", "whisper", "speech-to-text"]
)
def transcribe_audio(
    audio_file: str,
    output_file: Optional[str] = None,
    language: Optional[str] = None,
    translate: bool = False,
    model_size: str = "medium"
) -> Dict[str, Any]:
    """Transcribes an audio file to text using OpenAI's Whisper model.
    
    Args:
        audio_file: Path to the audio file to transcribe
        output_file: Optional path to save the transcription
        language: Optional language code (auto-detect if None)
        translate: If True, translate to English
        model_size: Model size (tiny, base, small, medium, large, large-v2, large-v3)
        
    Returns:
        Dictionary with transcription results
    """
    options = TranscriptionOptions(
        model_size=WhisperModel(model_size),
        language=language,
        task="translate" if translate else "transcribe"
    )
    
    transcriber = WhisperTranscriber(options)
    result = transcriber.transcribe(audio_file, output_file=output_file)
    
    return {
        "success": result.success,
        "text": result.text,
        "language": result.language,
        "duration": result.duration,
        "error_message": result.error_message
    }


# Example usage and testing
if __name__ == "__main__":
    print("=== Whisper Transcription Tool Test ===\n")
    
    # List audio files in current directory
    audio_extensions = ['*.mp3', '*.wav', '*.m4a', '*.mp4', '*.flac', '*.aac', '*.ogg', '*.opus', '*.webm']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(glob.glob(ext))
    
    if not audio_files:
        print("No audio files found in current directory.")
        print("Supported formats: mp3, wav, m4a, mp4, flac, aac, ogg, opus, webm")
        sys.exit(1)
    
    print("Audio files found in current directory:")
    for i, file in enumerate(audio_files, 1):
        file_size = os.path.getsize(file) / (1024 * 1024)  # Size in MB
        print(f"{i}. {file} ({file_size:.2f} MB)")
    
    # Get user input
    print("\nEnter the number of the file to transcribe (or 'q' to quit):")
    choice = input().strip()
    
    if choice.lower() == 'q':
        sys.exit(0)
    
    try:
        file_index = int(choice) - 1
        if 0 <= file_index < len(audio_files):
            selected_file = audio_files[file_index]
        else:
            print("Invalid selection.")
            sys.exit(1)
    except ValueError:
        # Try direct filename
        if os.path.exists(choice):
            selected_file = choice
        else:
            print("Invalid input.")
            sys.exit(1)
    
    print(f"\nSelected file: {selected_file}")
    
    # Ask for options
    print("\nOptions:")
    print("1. Basic transcription")
    print("2. Transcription with translation to English")
    print("3. Detect language only")
    print("4. Transcription with SRT subtitles")
    print("5. Transcription with word timestamps")
    
    option = input("Select option (1-5): ").strip()
    
    # Initialize transcriber
    print("\nInitializing Whisper transcriber...")
    transcriber = WhisperTranscriber()
    
    if option == "1":
        # Basic transcription
        print("Transcribing audio...")
        result = transcriber.transcribe(selected_file)
        if result.success:
            print(f"\n=== Transcription ===")
            print(result.text)
            print(f"\nLanguage: {result.language}")
            print(f"Duration: {result.duration:.2f}s")
            print(f"Processing time: {result.processing_time:.2f}s")
        else:
            print(f"Error: {result.error_message}")
    
    elif option == "2":
        # Translation
        print("Transcribing and translating to English...")
        options = TranscriptionOptions(task="translate")
        result = transcriber.transcribe(selected_file, options)
        if result.success:
            print(f"\n=== Translation ===")
            print(result.text)
            print(f"\nOriginal language: {result.language}")
            print(f"Duration: {result.duration:.2f}s")
        else:
            print(f"Error: {result.error_message}")
    
    elif option == "3":
        # Language detection
        print("Detecting language...")
        language = transcriber.detect_language(selected_file)
        if language:
            print(f"\nDetected language: {language}")
        else:
            print("Failed to detect language")
    
    elif option == "4":
        # SRT subtitles
        output_file = selected_file.rsplit('.', 1)[0] + '.srt'
        print(f"Creating SRT subtitles: {output_file}")
        options = TranscriptionOptions(output_format=OutputFormat.SRT)
        result = transcriber.transcribe(selected_file, options, output_file)
        if result.success:
            print(f"\nSubtitles saved to: {output_file}")
            print(f"Total segments: {len(result.segments)}")
            print(f"Duration: {result.duration:.2f}s")
        else:
            print(f"Error: {result.error_message}")
    
    elif option == "5":
        # Word timestamps
        print("Transcribing with word timestamps...")
        result = transcriber.transcribe_with_timestamps(selected_file, word_level=True)
        if result.success:
            print(f"\n=== Transcription with timestamps ===")
            for segment in result.segments[:5]:  # Show first 5 segments
                print(f"[{segment.start:.2f}s - {segment.end:.2f}s]: {segment.text.strip()}")
            if len(result.segments) > 5:
                print(f"... and {len(result.segments) - 5} more segments")
        else:
            print(f"Error: {result.error_message}")
    
    else:
        print("Invalid option")
        sys.exit(1)
    
    # Ask to save full transcription
    if option in ["1", "2", "5"] and 'result' in locals() and result.success:
        save = input("\nSave transcription to file? (y/n): ").strip().lower()
        if save == 'y':
            output_file = selected_file.rsplit('.', 1)[0] + '_transcript.txt'
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result.text)
            print(f"Transcription saved to: {output_file}")