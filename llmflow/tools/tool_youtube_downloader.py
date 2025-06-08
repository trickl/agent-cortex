"""
YouTube Video Downloader Tool for LLM Agents
============================================

A Python tool for downloading YouTube videos with support for various formats
and quality options. Designed to be used as a tool in LLM agent frameworks.

Requirements:
    - yt-dlp (install with: pip install yt-dlp)
    - Python 3.7+

Example Usage:
    >>> downloader = YouTubeDownloader()
    >>> result = downloader.download("https://www.youtube.com/watch?v=VIDEO_ID")
    >>> print(result)

Author: AI Assistant
Version: 1.0.0
"""

import os
import json
import logging
from typing import Dict, Optional, Union, List, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import subprocess
import sys

from .tool_decorator import register_tool

# Check if yt-dlp is installed
try:
    import yt_dlp
except ImportError:
    print("yt-dlp is not installed. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp"])
    import yt_dlp


class VideoQuality(Enum):
    """Enum for video quality options."""
    BEST = "best"
    WORST = "worst"
    HD1080 = "1080"
    HD720 = "720"
    SD480 = "480"
    SD360 = "360"


class AudioFormat(Enum):
    """Enum for audio format options."""
    MP3 = "mp3"
    M4A = "m4a"
    WAV = "wav"
    OPUS = "opus"
    AAC = "aac"


@dataclass
class DownloadOptions:
    """Configuration options for video downloading."""
    output_path: str = "./downloads"
    video_quality: VideoQuality = VideoQuality.BEST
    audio_only: bool = False
    audio_format: AudioFormat = AudioFormat.MP3
    subtitle: bool = False
    subtitle_lang: str = "en"
    quiet: bool = False
    no_warnings: bool = True
    extract_info_only: bool = False
    playlist: bool = False
    format_id: Optional[str] = None
    cookies_file: Optional[str] = None
    user_agent: Optional[str] = None
    proxy: Optional[str] = None
    rate_limit: Optional[str] = None
    max_downloads: Optional[int] = None
    age_limit: Optional[int] = None
    geo_bypass: bool = False
    prefer_free_formats: bool = True
    concurrent_fragments: int = 5
    retries: int = 3
    fragment_retries: int = 3
    skip_unavailable_fragments: bool = True
    keep_video: bool = True
    write_thumbnail: bool = False
    write_description: bool = False
    write_info_json: bool = False
    write_comments: bool = False
    embed_metadata: bool = True
    embed_thumbnail: bool = False
    embed_subs: bool = False
    embed_chapters: bool = True
    sponsorblock_remove: List[str] = field(default_factory=lambda: ["sponsor"])


@dataclass
class DownloadResult:
    """Result of a download operation."""
    success: bool
    video_id: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None
    filename: Optional[str] = None
    format: Optional[str] = None
    filesize: Optional[int] = None
    duration: Optional[int] = None
    upload_date: Optional[str] = None
    uploader: Optional[str] = None
    view_count: Optional[int] = None
    like_count: Optional[int] = None
    description: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class YouTubeDownloader:
    """
    A YouTube video downloader tool for LLM agents.
    
    This class provides a high-level interface for downloading YouTube videos
    with various options and error handling.
    
    Attributes:
        options (DownloadOptions): Configuration options for downloading
        logger (logging.Logger): Logger instance for this class
    """
    
    def __init__(
        self,
        options: Optional[DownloadOptions] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the YouTube downloader.
        
        Args:
            options: Download configuration options
            logger: Logger instance (creates default if None)
        """
        self.options = options or DownloadOptions()
        self.logger = logger or self._setup_logger()
        self._ensure_output_directory()
    
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
    
    def _ensure_output_directory(self) -> None:
        """Ensure the output directory exists."""
        Path(self.options.output_path).mkdir(parents=True, exist_ok=True)
    
    def _get_ydl_options(self) -> Dict[str, Any]:
        """
        Get yt-dlp options based on current configuration.
        
        Returns:
            Dictionary of yt-dlp options
        """
        ydl_opts = {
            'outtmpl': os.path.join(self.options.output_path, '%(id)s.%(ext)s'),
            'quiet': self.options.quiet,
            'no_warnings': self.options.no_warnings,
            'extract_flat': 'in_playlist' if self.options.playlist else False,
            'playlistend': self.options.max_downloads,
            'age_limit': self.options.age_limit,
            'geo_bypass': self.options.geo_bypass,
            'prefer_free_formats': self.options.prefer_free_formats,
            'concurrent_fragment_downloads': self.options.concurrent_fragments,
            'retries': self.options.retries,
            'fragment_retries': self.options.fragment_retries,
            'skip_unavailable_fragments': self.options.skip_unavailable_fragments,
            'keepvideo': self.options.keep_video,
            'writethumbnail': self.options.write_thumbnail,
            'writedescription': self.options.write_description,
            'writeinfojson': self.options.write_info_json,
            'writecomments': self.options.write_comments,
            'embedmetadata': self.options.embed_metadata,
            'embedthumbnail': self.options.embed_thumbnail,
            'embedsubtitles': self.options.embed_subs,
            'embedchapters': self.options.embed_chapters,
        }
        
        # Add sponsorblock options
        if self.options.sponsorblock_remove:
            ydl_opts['sponsorblock_remove'] = self.options.sponsorblock_remove
        
        # Add authentication options
        if self.options.cookies_file:
            ydl_opts['cookiefile'] = self.options.cookies_file
        
        if self.options.user_agent:
            ydl_opts['user_agent'] = self.options.user_agent
        
        if self.options.proxy:
            ydl_opts['proxy'] = self.options.proxy
        
        if self.options.rate_limit:
            ydl_opts['ratelimit'] = self.options.rate_limit
        
        # Handle format selection
        if self.options.audio_only:
            ydl_opts['format'] = 'bestaudio/best'
            ydl_opts['postprocessors'] = [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': self.options.audio_format.value,
                'preferredquality': '192',
            }]
        elif self.options.format_id:
            ydl_opts['format'] = self.options.format_id
        else:
            # Quality-based format selection
            if self.options.video_quality == VideoQuality.BEST:
                ydl_opts['format'] = 'bestvideo+bestaudio/best'
            elif self.options.video_quality == VideoQuality.WORST:
                ydl_opts['format'] = 'worstvideo+worstaudio/worst'
            else:
                # Specific quality selection
                quality = self.options.video_quality.value
                ydl_opts['format'] = (
                    f'bestvideo[height<={quality}]+bestaudio/best[height<={quality}]'
                )
        
        # Handle subtitles
        if self.options.subtitle:
            ydl_opts['writesubtitles'] = True
            ydl_opts['subtitleslangs'] = [self.options.subtitle_lang]
            ydl_opts['writeautomaticsub'] = True
        
        return ydl_opts
    
    def get_video_info(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Get video information without downloading.
        
        Args:
            url: YouTube video URL
            
        Returns:
            Dictionary with video information or None if error
        """
        ydl_opts = self._get_ydl_options()
        ydl_opts['skip_download'] = True
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return info
        except Exception as e:
            self.logger.error(f"Error getting video info: {str(e)}")
            return None
    
    def download(
        self,
        url: str,
        options: Optional[DownloadOptions] = None
    ) -> DownloadResult:
        """
        Download a YouTube video.
        
        Args:
            url: YouTube video URL
            options: Optional download options (overrides instance options)
            
        Returns:
            DownloadResult object with download information
        """
        # Use provided options or instance options
        if options:
            original_options = self.options
            self.options = options
            self._ensure_output_directory()
        
        try:
            if self.options.extract_info_only:
                # Only extract information
                info = self.get_video_info(url)
                if info:
                    return DownloadResult(
                        success=True,
                        video_id=info.get('id'),
                        title=info.get('title'),
                        url=url,
                        format=info.get('format'),
                        duration=info.get('duration'),
                        upload_date=info.get('upload_date'),
                        uploader=info.get('uploader'),
                        view_count=info.get('view_count'),
                        like_count=info.get('like_count'),
                        description=info.get('description'),
                        metadata=info
                    )
                else:
                    return DownloadResult(
                        success=False,
                        url=url,
                        error_message="Failed to extract video information"
                    )
            
            # Perform actual download
            ydl_opts = self._get_ydl_options()
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                self.logger.info(f"Starting download: {url}")
                info = ydl.extract_info(url, download=True)
                
                # Determine filename
                if self.options.audio_only:
                    # For audio only, yt-dlp might use a different extension than preferred_codec
                    # Get the actual extension from info or default
                    ext = info.get('ext', self.options.audio_format.value)
                else:
                    ext = info.get('ext', 'mp4')
                
                # Construct a simplified filename using the video ID and determined extension
                video_id = info.get('id', 'unknown_video')
                simplified_filename = f"{video_id}.{ext}"
                full_simplified_path = os.path.join(self.options.output_path, simplified_filename)

                # yt-dlp saves to outtmpl, which we set to %(id)s.%(ext)s
                # We need to make sure we return the path that yt-dlp actually used, which should be our simplified path
                # Confirm the file exists at the simplified path
                if os.path.exists(full_simplified_path):
                    filename_to_return = full_simplified_path
                else:
                    # Fallback: try to determine the actual filename used by yt-dlp
                    self.logger.warning(f"Simplified path {full_simplified_path} not found after download. Attempting to find actual filename.")
                    # This is a best guess based on yt-dlp's default naming if outtmpl is complex
                    # This part might still be prone to issues with complex filenames
                    filename_to_return = ydl.prepare_filename(info)
                    if not os.path.exists(filename_to_return):
                        self.logger.error(f"Actual filename {filename_to_return} not found either.")
                        filename_to_return = None # Indicate failure to get a valid path

                # Get file size if available
                filesize = None
                if filename_to_return and os.path.exists(filename_to_return):
                    filesize = os.path.getsize(filename_to_return)
                else:
                    self.logger.warning("Could not determine file size as filename is not valid.")

                self.logger.info(f"Download completed. Reporting filename: {filename_to_return}")
                
                return DownloadResult(
                    success=True,
                    video_id=info.get('id'),
                    title=info.get('title'),
                    url=url,
                    filename=filename_to_return, # Return the simplified or found filename
                    format=info.get('format'),
                    filesize=filesize,
                    duration=info.get('duration'),
                    upload_date=info.get('upload_date'),
                    uploader=info.get('uploader'),
                    view_count=info.get('view_count'),
                    like_count=info.get('like_count'),
                    description=info.get('description'),
                    metadata=info
                )
                
        except yt_dlp.utils.DownloadError as e:
            self.logger.error(f"Download error: {str(e)}")
            return DownloadResult(
                success=False,
                url=url,
                error_message=f"Download error: {str(e)}"
            )
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            return DownloadResult(
                success=False,
                url=url,
                error_message=f"Unexpected error: {str(e)}"
            )
        finally:
            # Restore original options if they were overridden
            if options:
                self.options = original_options
    
    def download_playlist(
        self,
        url: str,
        max_videos: Optional[int] = None
    ) -> List[DownloadResult]:
        """
        Download a YouTube playlist.
        
        Args:
            url: YouTube playlist URL
            max_videos: Maximum number of videos to download
            
        Returns:
            List of DownloadResult objects
        """
        original_playlist = self.options.playlist
        original_max = self.options.max_downloads
        
        self.options.playlist = True
        if max_videos:
            self.options.max_downloads = max_videos
        
        results = []
        
        try:
            # Get playlist info
            ydl_opts = self._get_ydl_options()
            ydl_opts['extract_flat'] = True
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                playlist_info = ydl.extract_info(url, download=False)
                
                if 'entries' not in playlist_info:
                    return [DownloadResult(
                        success=False,
                        url=url,
                        error_message="Not a valid playlist"
                    )]
                
                # Download each video
                for i, entry in enumerate(playlist_info['entries']):
                    if max_videos and i >= max_videos:
                        break
                    
                    video_url = f"https://www.youtube.com/watch?v={entry['id']}"
                    self.logger.info(
                        f"Downloading video {i+1}/{len(playlist_info['entries'])}: "
                        f"{entry.get('title', 'Unknown')}"
                    )
                    
                    result = self.download(video_url)
                    results.append(result)
                
        finally:
            self.options.playlist = original_playlist
            self.options.max_downloads = original_max
        
        return results
    
    def download_with_retry(
        self,
        url: str,
        max_retries: int = 3,
        options: Optional[DownloadOptions] = None
    ) -> DownloadResult:
        """
        Download with automatic retry on failure.

    Args:
            url: YouTube video URL
            max_retries: Maximum number of retry attempts
            options: Optional download options

    Returns:
            DownloadResult object
        """
        last_error = None
        
        for attempt in range(max_retries + 1):
            if attempt > 0:
                self.logger.info(f"Retry attempt {attempt}/{max_retries}")
            
            result = self.download(url, options)
            
            if result.success:
                return result
            
            last_error = result.error_message
            
            # Don't retry on certain errors
            if last_error and any(msg in last_error.lower() for msg in [
                "private video", "deleted video", "unavailable"
            ]):
                break
        
        return DownloadResult(
            success=False,
            url=url,
            error_message=f"Failed after {max_retries} retries. Last error: {last_error}"
        )


class YouTubeDownloaderTool:
    """
    LLM Agent Tool wrapper for YouTube Downloader.
    
    This class provides a simplified interface designed for use in LLM agent
    frameworks like LangChain, AutoGPT, etc.
    """
    
    def __init__(self, default_options: Optional[DownloadOptions] = None):
        """
        Initialize the tool.
        
        Args:
            default_options: Default download options
        """
        self.downloader = YouTubeDownloader(default_options)
        self.name = "youtube_downloader"
        self.description = (
            "Download YouTube videos or audio. "
            "Input should be a YouTube URL. "
            "Returns download status and file information."
        )
    
    def run(self, url: str, **kwargs) -> Dict[str, Any]:
        """
        Run the tool with a YouTube URL.
        
        Args:
            url: YouTube video URL
            **kwargs: Additional options (audio_only, quality, etc.)
            
        Returns:
            Dictionary with download results
        """
        # Parse kwargs into options
        options = DownloadOptions()
        
        if kwargs.get('audio_only'):
            options.audio_only = True
            
        if kwargs.get('quality'):
            quality_map = {
                '1080': VideoQuality.HD1080,
                '720': VideoQuality.HD720,
                '480': VideoQuality.SD480,
                '360': VideoQuality.SD360,
                'best': VideoQuality.BEST,
                'worst': VideoQuality.WORST
            }
            options.video_quality = quality_map.get(
                kwargs['quality'],
                VideoQuality.BEST
            )
        
        if kwargs.get('output_path'):
            options.output_path = kwargs['output_path']
        
        # Download the video
        result = self.downloader.download(url, options)
        
        # Convert to dictionary for agent compatibility
        return {
            'success': result.success,
            'video_id': result.video_id,
            'title': result.title,
            'filename': result.filename,
            'filesize': result.filesize,
            'duration': result.duration,
            'error': result.error_message
        }
    
    async def arun(self, url: str, **kwargs) -> Dict[str, Any]:
        """Async version of run (currently just calls sync version)."""
        return self.run(url, **kwargs)
    
    def get_info(self, url: str) -> Dict[str, Any]:
        """
        Get video information without downloading.
        
        Args:
            url: YouTube video URL
            
        Returns:
            Dictionary with video information
        """
        info = self.downloader.get_video_info(url)
        
        if info:
            return {
                'success': True,
                'video_id': info.get('id'),
                'title': info.get('title'),
                'duration': info.get('duration'),
                'uploader': info.get('uploader'),
                'view_count': info.get('view_count'),
                'description': info.get('description')
            }
        else:
            return {
                'success': False,
                'error': 'Failed to get video information'
            }


@register_tool(
    tags=["media", "download", "youtube"]
)
def download_youtube_video(
    url: str,
    output_path: str = "./downloads",
    audio_only: bool = False,
    quality: str = "best"
) -> Dict[str, Any]:
    """Downloads a video from YouTube given its URL. Returns information about the downloaded video.
    
    Args:
        url: The YouTube video URL to download
        output_path: Path where the video should be saved
        audio_only: If True, only downloads the audio
        quality: Video quality (best, worst, 1080, 720, 480, 360)
        
    Returns:
        Dictionary with download results
    """
    options = DownloadOptions(
        output_path=output_path,
        video_quality=VideoQuality(quality),
        audio_only=audio_only
    )
    
    downloader = YouTubeDownloader(options)
    result = downloader.download(url)
    
    return {
        "success": result.success,
        "filename": result.filename,
        "title": result.title,
        "url": result.url,
        "error_message": result.error_message
    }


# Example usage and testing
if __name__ == "__main__":
    # Example 1: Basic download
    downloader = YouTubeDownloader()
    result = downloader.download("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    print(f"Download result: {result}")
    
    # Example 2: Download audio only
    audio_options = DownloadOptions(
        audio_only=True,
        audio_format=AudioFormat.MP3
    )
    audio_result = downloader.download(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        options=audio_options
    )
    print(f"Audio download result: {audio_result}")
    
    # Example 3: Using as LLM tool
    tool = YouTubeDownloaderTool()
    tool_result = tool.run(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        audio_only=True
    )
    print(f"Tool result: {tool_result}")