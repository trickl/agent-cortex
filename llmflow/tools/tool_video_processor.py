import os
from typing import Dict, Any, Optional, List, Tuple
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
from llmflow.tools.tool_decorator import register_tool

# Define a directory to save processed videos
PROCESSED_DIR = os.path.join(os.path.expanduser("~"), "Downloads", "llmflow_processed_videos")
os.makedirs(PROCESSED_DIR, exist_ok=True)

@register_tool(tags=["media", "video", "processing"])
def process_video(
    video_path: str,
    output_filename: str = None,
    trim_start: float = None,
    trim_end: float = None,
    resize_width: int = None,
    resize_height: int = None,
    speed_factor: float = None,
    extract_audio: bool = False,
    remove_audio: bool = False,
    rotate_degrees: int = None,
) -> Dict[str, Any]:
    """
    Process a video file with various operations like trimming, resizing, speed adjustment, etc.

    Args:
        video_path: Path to the input video file
        output_filename: Optional custom filename for the output (without extension)
        trim_start: Start time in seconds to trim from
        trim_end: End time in seconds to trim to
        resize_width: New width in pixels
        resize_height: New height in pixels
        speed_factor: Speed multiplier (e.g., 0.5 for half speed, 2 for double speed)
        extract_audio: If True, extracts audio to a separate file
        remove_audio: If True, removes audio from the video
        rotate_degrees: Degrees to rotate the video (90, 180, or 270)

    Returns:
        Dictionary containing the status and paths to the processed files
    """
    try:
        if not os.path.exists(video_path):
            return {
                "status": "error",
                "message": f"Input video file not found: {video_path}"
            }

        # Load the video
        video = VideoFileClip(video_path)
        
        # Store original duration for validation
        original_duration = video.duration

        # Process the video according to parameters
        if trim_start is not None or trim_end is not None:
            start = trim_start if trim_start is not None else 0
            end = trim_end if trim_end is not None else video.duration
            if start >= end or start < 0 or end > video.duration:
                return {
                    "status": "error",
                    "message": f"Invalid trim parameters. Valid range is 0 to {video.duration} seconds"
                }
            video = video.subclip(start, end)

        if resize_width is not None or resize_height is not None:
            current_width, current_height = video.size
            new_width = resize_width if resize_width is not None else current_width
            new_height = resize_height if resize_height is not None else current_height
            video = video.resize((new_width, new_height))

        if speed_factor is not None:
            if speed_factor <= 0:
                return {
                    "status": "error",
                    "message": "Speed factor must be positive"
                }
            video = video.speedx(speed_factor)

        if rotate_degrees is not None:
            if rotate_degrees not in [90, 180, 270]:
                return {
                    "status": "error",
                    "message": "Rotation must be 90, 180, or 270 degrees"
                }
            video = video.rotate(rotate_degrees)

        if remove_audio and video.audio is not None:
            video = video.without_audio()

        # Prepare output filename
        if output_filename is None:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_filename = f"{base_name}_processed"

        # Prepare result paths
        video_output_path = os.path.join(PROCESSED_DIR, f"{output_filename}.mp4")
        audio_output_path = os.path.join(PROCESSED_DIR, f"{output_filename}.mp3") if extract_audio else None

        # Write the processed video
        video.write_videofile(video_output_path, codec='libx264', audio_codec='aac')

        # Extract audio if requested
        if extract_audio and video.audio is not None:
            video.audio.write_audiofile(audio_output_path)

        # Clean up
        video.close()

        result = {
            "status": "success",
            "message": "Video processing completed successfully",
            "processed_video": video_output_path,
        }

        if extract_audio and audio_output_path:
            result["extracted_audio"] = audio_output_path

        return result

    except Exception as e:
        return {
            "status": "error",
            "message": f"An error occurred during video processing: {str(e)}"
        }

@register_tool(tags=["media", "video", "merge"])
def merge_videos(
    video_paths: List[str],
    output_filename: str = None,
) -> Dict[str, Any]:
    """
    Merge multiple videos into a single video file.

    Args:
        video_paths: List of paths to input video files
        output_filename: Optional custom filename for the output (without extension)

    Returns:
        Dictionary containing the status and path to the merged video
    """
    try:
        if not video_paths:
            return {
                "status": "error",
                "message": "No input videos provided"
            }

        # Check if all input files exist
        for path in video_paths:
            if not os.path.exists(path):
                return {
                    "status": "error",
                    "message": f"Input video file not found: {path}"
                }

        # Load all videos
        clips = [VideoFileClip(path) for path in video_paths]

        # Concatenate videos
        final_clip = concatenate_videoclips(clips)

        # Prepare output filename
        if output_filename is None:
            output_filename = "merged_video"

        # Prepare output path
        output_path = os.path.join(PROCESSED_DIR, f"{output_filename}.mp4")

        # Write the merged video
        final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

        # Clean up
        for clip in clips:
            clip.close()
        final_clip.close()

        return {
            "status": "success",
            "message": "Videos merged successfully",
            "merged_video": output_path
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"An error occurred while merging videos: {str(e)}"
        }

# Example usage:
# if __name__ == "__main__":
#     # Process a single video
#     result = process_video(
#         "input.mp4",
#         output_filename="processed",
#         trim_start=10,
#         trim_end=30,
#         resize_width=1280,
#         resize_height=720,
#         speed_factor=1.5,
#         extract_audio=True
#     )
#     print(result)
#
#     # Merge multiple videos
#     result = merge_videos(
#         ["video1.mp4", "video2.mp4", "video3.mp4"],
#         output_filename="merged"
#     )
#     print(result) 