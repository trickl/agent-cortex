# Tools and actions for the agent
# Collection of pre-defined tools

# Import tool modules to ensure they are registered
from . import tool_file_system  # For file system operations like read, write, list
from . import tool_search_duckduckgo  # For DuckDuckGo search etc.
from . import tool_control_tools  # For agent control tools like terminate
from . import tool_web_parser  # For web parsing
from . import tool_datetime  # For date and time operations
from . import tool_system_monitoring  # For system monitoring
from . import tool_shell  # For shell operations
from . import tool_cloud  # For cloud operations
from . import tool_messenger  # For messaging operations
from . import tool_calendar  # For calendar operations
from . import tool_email  # For email operations
from . import tool_embenddings  # For embeddings operations
from . import tool_text_to_speech  # For text-to-speech operations
from . import tool_speech  # For speech operations
from . import tool_image_processing  # For image processing
from . import tool_data_analysis  # For data analysis
from . import tool_sql_database  # For SQL database operations
from . import tool_mathematical  # For mathematical operations
from . import tool_code_execution  # For code execution
from . import tool_file_operations  # For file operations
from . import tool_decorator  # For tool decorators
from . import tool_youtube_downloader # For downloading YouTube videos
from . import tool_video_audio_transcriber # For video/audio transcription
from . import tool_video_processor # For video processing and merging
from . import tool_frame_extractor # For extracting frames from videos
from . import tool_video_preview_generator # For generating video previews
from . import tool_scientific_papers_search_arxiv # For arXiv paper search
from . import tool_download_pdf_from_url # For downloading PDFs from URL
# from . import string_tools # Example: if you add string_tools.py
# from . import math_tools   # Example: if you add math_tools.py 