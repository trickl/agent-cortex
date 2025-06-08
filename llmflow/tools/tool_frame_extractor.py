import os
from typing import Dict, Any, List, Optional
from moviepy.editor import VideoFileClip
from PIL import Image
import numpy as np
from llmflow.tools.tool_decorator import register_tool

# Define a directory to save extracted frames
FRAMES_DIR = os.path.join(os.path.expanduser("~"), "Downloads", "llmflow_frames")
os.makedirs(FRAMES_DIR, exist_ok=True)

@register_tool(tags=["media", "video", "frames", "image"])
def extract_frames(
    video_path: str,
    output_prefix: str = None,
    start_time: float = None,
    end_time: float = None,
    frame_interval: float = 1.0,
    max_frames: int = None,
    frame_quality: int = 95,
    frame_format: str = "jpg",
    resize_width: int = None,
    resize_height: int = None,
) -> Dict[str, Any]:
    """
    Extract frames from a video file with various options.

    Args:
        video_path: Path to the input video file
        output_prefix: Optional prefix for output frame filenames
        start_time: Start time in seconds to begin extraction (default: start of video)
        end_time: End time in seconds to stop extraction (default: end of video)
        frame_interval: Time interval between frames in seconds (default: 1.0)
        max_frames: Maximum number of frames to extract (default: no limit)
        frame_quality: JPEG quality for saved frames, 1-100 (default: 95)
        frame_format: Output format for frames ('jpg' or 'png', default: 'jpg')
        resize_width: Optional width to resize frames to
        resize_height: Optional height to resize frames to

    Returns:
        Dictionary containing the status and paths to the extracted frames
    """
    try:
        if not os.path.exists(video_path):
            return {
                "status": "error",
                "message": f"Input video file not found: {video_path}"
            }

        if frame_format.lower() not in ['jpg', 'png']:
            return {
                "status": "error",
                "message": "Frame format must be 'jpg' or 'png'"
            }

        if frame_quality < 1 or frame_quality > 100:
            return {
                "status": "error",
                "message": "Frame quality must be between 1 and 100"
            }

        # Load the video
        video = VideoFileClip(video_path)
        
        # Validate and adjust time parameters
        video_duration = video.duration
        start = start_time if start_time is not None else 0
        end = end_time if end_time is not None else video_duration

        if start < 0 or start >= video_duration:
            return {
                "status": "error",
                "message": f"Invalid start time. Valid range is 0 to {video_duration} seconds"
            }

        if end <= start or end > video_duration:
            return {
                "status": "error",
                "message": f"Invalid end time. Valid range is {start} to {video_duration} seconds"
            }

        # Prepare output filename prefix
        if output_prefix is None:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_prefix = f"{base_name}_frame"

        # Calculate frame times
        frame_times = np.arange(start, end, frame_interval)
        if max_frames is not None:
            frame_times = frame_times[:max_frames]

        extracted_frames = []
        
        # Extract frames
        for i, t in enumerate(frame_times):
            # Get the frame
            frame = video.get_frame(t)
            
            # Convert to PIL Image
            image = Image.fromarray(frame)
            
            # Resize if requested
            if resize_width is not None or resize_height is not None:
                current_width, current_height = image.size
                new_width = resize_width if resize_width is not None else current_width
                new_height = resize_height if resize_height is not None else current_height
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Prepare frame filename
            frame_filename = f"{output_prefix}_{i:04d}.{frame_format.lower()}"
            frame_path = os.path.join(FRAMES_DIR, frame_filename)
            
            # Save the frame
            if frame_format.lower() == 'jpg':
                image.save(frame_path, 'JPEG', quality=frame_quality)
            else:  # png
                image.save(frame_path, 'PNG')
            
            extracted_frames.append(frame_path)

        # Clean up
        video.close()

        return {
            "status": "success",
            "message": f"Successfully extracted {len(extracted_frames)} frames",
            "frames": extracted_frames,
            "frames_dir": FRAMES_DIR,
            "frame_count": len(extracted_frames)
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"An error occurred while extracting frames: {str(e)}"
        }

@register_tool(tags=["media", "video", "frames", "image"])
def extract_keyframes(
    video_path: str,
    output_prefix: str = None,
    threshold: float = 0.1,
    max_frames: int = None,
    frame_quality: int = 95,
    frame_format: str = "jpg",
    resize_width: int = None,
    resize_height: int = None,
) -> Dict[str, Any]:
    """
    Extract keyframes from a video based on scene changes.

    Args:
        video_path: Path to the input video file
        output_prefix: Optional prefix for output frame filenames
        threshold: Difference threshold for detecting scene changes (0.0-1.0, default: 0.1)
        max_frames: Maximum number of keyframes to extract (default: no limit)
        frame_quality: JPEG quality for saved frames, 1-100 (default: 95)
        frame_format: Output format for frames ('jpg' or 'png', default: 'jpg')
        resize_width: Optional width to resize frames to
        resize_height: Optional height to resize frames to

    Returns:
        Dictionary containing the status and paths to the extracted keyframes
    """
    try:
        if not os.path.exists(video_path):
            return {
                "status": "error",
                "message": f"Input video file not found: {video_path}"
            }

        if frame_format.lower() not in ['jpg', 'png']:
            return {
                "status": "error",
                "message": "Frame format must be 'jpg' or 'png'"
            }

        if frame_quality < 1 or frame_quality > 100:
            return {
                "status": "error",
                "message": "Frame quality must be between 1 and 100"
            }

        if threshold < 0.0 or threshold > 1.0:
            return {
                "status": "error",
                "message": "Threshold must be between 0.0 and 1.0"
            }

        # Load the video
        video = VideoFileClip(video_path)

        # Prepare output filename prefix
        if output_prefix is None:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_prefix = f"{base_name}_keyframe"

        # Initialize variables for keyframe detection
        prev_frame = None
        keyframes = []
        keyframe_times = []

        # Sample frames at 1 FPS for efficiency
        for t in range(int(video.duration)):
            frame = video.get_frame(t)
            
            if prev_frame is not None:
                # Calculate frame difference
                diff = np.mean(np.abs(frame.astype(float) - prev_frame.astype(float))) / 255.0
                
                # If difference exceeds threshold, consider it a keyframe
                if diff > threshold:
                    keyframe_times.append(t)
            
            prev_frame = frame

            # Check if we've reached the maximum number of keyframes
            if max_frames is not None and len(keyframe_times) >= max_frames:
                break

        extracted_frames = []

        # Extract and save keyframes
        for i, t in enumerate(keyframe_times):
            # Get the keyframe
            frame = video.get_frame(t)
            
            # Convert to PIL Image
            image = Image.fromarray(frame)
            
            # Resize if requested
            if resize_width is not None or resize_height is not None:
                current_width, current_height = image.size
                new_width = resize_width if resize_width is not None else current_width
                new_height = resize_height if resize_height is not None else current_height
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Prepare frame filename
            frame_filename = f"{output_prefix}_{i:04d}.{frame_format.lower()}"
            frame_path = os.path.join(FRAMES_DIR, frame_filename)
            
            # Save the frame
            if frame_format.lower() == 'jpg':
                image.save(frame_path, 'JPEG', quality=frame_quality)
            else:  # png
                image.save(frame_path, 'PNG')
            
            extracted_frames.append(frame_path)

        # Clean up
        video.close()

        return {
            "status": "success",
            "message": f"Successfully extracted {len(extracted_frames)} keyframes",
            "frames": extracted_frames,
            "frames_dir": FRAMES_DIR,
            "frame_count": len(extracted_frames),
            "frame_times": keyframe_times
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"An error occurred while extracting keyframes: {str(e)}"
        }

# Example usage:
# if __name__ == "__main__":
#     # Extract frames at regular intervals
#     result = extract_frames(
#         "input.mp4",
#         output_prefix="scene",
#         frame_interval=0.5,  # Extract every 0.5 seconds
#         max_frames=100,
#         frame_quality=95,
#         resize_width=1280,
#         resize_height=720
#     )
#     print(result)
#
#     # Extract keyframes based on scene changes
#     result = extract_keyframes(
#         "input.mp4",
#         output_prefix="keyframe",
#         threshold=0.1,
#         max_frames=50,
#         frame_quality=95
#     )
#     print(result) 