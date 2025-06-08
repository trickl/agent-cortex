import os
from typing import Dict, Any, List, Optional, Tuple
from moviepy.editor import VideoFileClip, ImageClip, concatenate_videoclips
from PIL import Image
import numpy as np
from llmflow.tools.tool_decorator import register_tool

# Define a directory to save generated previews
PREVIEWS_DIR = os.path.join(os.path.expanduser("~"), "Downloads", "llmflow_previews")
os.makedirs(PREVIEWS_DIR, exist_ok=True)

def create_frame_grid(frames: List[Image.Image], grid_size: Tuple[int, int], spacing: int = 10, background_color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """
    Create a grid of frames with specified dimensions and spacing.
    
    Args:
        frames: List of PIL Image objects
        grid_size: Tuple of (rows, columns)
        spacing: Pixels between frames
        background_color: RGB color tuple for background
        
    Returns:
        PIL Image containing the frame grid
    """
    rows, cols = grid_size
    n_frames = len(frames)
    
    # Calculate frame size (assume all frames are the same size)
    frame_width, frame_height = frames[0].size
    
    # Calculate grid dimensions
    grid_width = cols * frame_width + (cols - 1) * spacing
    grid_height = rows * frame_height + (rows - 1) * spacing
    
    # Create background
    grid_image = Image.new('RGB', (grid_width, grid_height), background_color)
    
    # Place frames in grid
    for idx, frame in enumerate(frames[:rows * cols]):  # Limit to grid capacity
        row = idx // cols
        col = idx % cols
        
        x = col * (frame_width + spacing)
        y = row * (frame_height + spacing)
        
        grid_image.paste(frame, (x, y))
    
    return grid_image

@register_tool(tags=["media", "video", "preview", "gif"])
def create_gif_preview(
    video_path: str,
    output_filename: str = None,
    start_time: float = None,
    end_time: float = None,
    frame_interval: float = 0.5,
    max_frames: int = 10,
    resize_width: int = None,
    resize_height: int = None,
    fps: int = 2,
) -> Dict[str, Any]:
    """
    Create an animated GIF preview from a video.

    Args:
        video_path: Path to the input video file
        output_filename: Optional custom filename for output (without extension)
        start_time: Start time in seconds (default: start of video)
        end_time: End time in seconds (default: end of video)
        frame_interval: Time interval between frames in seconds (default: 0.5)
        max_frames: Maximum number of frames to include (default: 10)
        resize_width: Optional width to resize frames to
        resize_height: Optional height to resize frames to
        fps: Frames per second in the output GIF (default: 2)

    Returns:
        Dictionary containing the status and path to the generated GIF
    """
    try:
        if not os.path.exists(video_path):
            return {
                "status": "error",
                "message": f"Input video file not found: {video_path}"
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

        # Prepare output filename
        if output_filename is None:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_filename = f"{base_name}_preview"

        # Calculate frame times
        frame_times = np.arange(start, end, frame_interval)
        if max_frames is not None:
            frame_times = frame_times[:max_frames]

        frames = []
        
        # Extract frames
        for t in frame_times:
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
            
            frames.append(image)

        # Prepare output path
        output_path = os.path.join(PREVIEWS_DIR, f"{output_filename}.gif")
        
        # Save as GIF
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=1000/fps,  # Duration in milliseconds
            loop=0
        )

        # Clean up
        video.close()

        return {
            "status": "success",
            "message": "Successfully created GIF preview",
            "preview": output_path,
            "frame_count": len(frames)
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"An error occurred while creating GIF preview: {str(e)}"
        }

@register_tool(tags=["media", "video", "preview", "grid"])
def create_grid_preview(
    video_path: str,
    output_filename: str = None,
    start_time: float = None,
    end_time: float = None,
    frame_interval: float = None,
    grid_rows: int = 3,
    grid_cols: int = 3,
    spacing: int = 10,
    frame_quality: int = 95,
    resize_width: int = None,
    resize_height: int = None,
    background_color: Tuple[int, int, int] = (255, 255, 255),
) -> Dict[str, Any]:
    """
    Create a grid preview image from video frames.

    Args:
        video_path: Path to the input video file
        output_filename: Optional custom filename for output (without extension)
        start_time: Start time in seconds (default: start of video)
        end_time: End time in seconds (default: end of video)
        frame_interval: Time interval between frames (default: auto-calculated)
        grid_rows: Number of rows in the grid (default: 3)
        grid_cols: Number of columns in the grid (default: 3)
        spacing: Pixels between frames (default: 10)
        frame_quality: JPEG quality for output image, 1-100 (default: 95)
        resize_width: Optional width to resize frames to
        resize_height: Optional height to resize frames to
        background_color: RGB color tuple for grid background (default: white)

    Returns:
        Dictionary containing the status and path to the generated preview
    """
    try:
        if not os.path.exists(video_path):
            return {
                "status": "error",
                "message": f"Input video file not found: {video_path}"
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

        # Calculate frame interval if not provided
        total_frames = grid_rows * grid_cols
        if frame_interval is None:
            frame_interval = (end - start) / (total_frames - 1) if total_frames > 1 else 0

        # Prepare output filename
        if output_filename is None:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_filename = f"{base_name}_grid"

        # Calculate frame times
        frame_times = np.linspace(start, end, total_frames)

        frames = []
        
        # Extract frames
        for t in frame_times:
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
            
            frames.append(image)

        # Create grid
        grid_image = create_frame_grid(
            frames,
            (grid_rows, grid_cols),
            spacing,
            background_color
        )

        # Prepare output path
        output_path = os.path.join(PREVIEWS_DIR, f"{output_filename}.jpg")
        
        # Save the grid
        grid_image.save(output_path, 'JPEG', quality=frame_quality)

        # Clean up
        video.close()

        return {
            "status": "success",
            "message": "Successfully created grid preview",
            "preview": output_path,
            "frame_count": len(frames),
            "grid_size": (grid_rows, grid_cols)
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"An error occurred while creating grid preview: {str(e)}"
        }

@register_tool(tags=["media", "video", "preview", "montage"])
def create_video_montage(
    video_path: str,
    output_filename: str = None,
    clips_duration: float = 1.0,
    clips_interval: float = 5.0,
    max_clips: int = 6,
    resize_width: int = None,
    resize_height: int = None,
    crossfade_duration: float = 0.5,
) -> Dict[str, Any]:
    """
    Create a video montage preview from clips of the original video.

    Args:
        video_path: Path to the input video file
        output_filename: Optional custom filename for output (without extension)
        clips_duration: Duration of each clip in seconds (default: 1.0)
        clips_interval: Time interval between clip starts in seconds (default: 5.0)
        max_clips: Maximum number of clips to include (default: 6)
        resize_width: Optional width to resize video to
        resize_height: Optional height to resize video to
        crossfade_duration: Duration of crossfade between clips in seconds (default: 0.5)

    Returns:
        Dictionary containing the status and path to the generated montage
    """
    try:
        if not os.path.exists(video_path):
            return {
                "status": "error",
                "message": f"Input video file not found: {video_path}"
            }

        # Load the video
        video = VideoFileClip(video_path)
        
        # Validate parameters
        video_duration = video.duration
        
        if clips_duration <= 0:
            return {
                "status": "error",
                "message": "Clips duration must be positive"
            }

        if clips_interval < clips_duration:
            return {
                "status": "error",
                "message": "Clips interval must be greater than or equal to clips duration"
            }

        if crossfade_duration >= clips_duration:
            return {
                "status": "error",
                "message": "Crossfade duration must be less than clips duration"
            }

        # Prepare output filename
        if output_filename is None:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_filename = f"{base_name}_montage"

        # Calculate clip start times
        start_times = np.arange(0, video_duration - clips_duration, clips_interval)
        if max_clips is not None:
            start_times = start_times[:max_clips]

        clips = []
        
        # Extract clips
        for start in start_times:
            # Extract clip
            clip = video.subclip(start, start + clips_duration)
            
            # Resize if requested
            if resize_width is not None or resize_height is not None:
                clip = clip.resize(width=resize_width, height=resize_height)
            
            clips.append(clip)

        # Concatenate clips with crossfade
        final_clip = concatenate_videoclips(clips, method="compose", crossfadein=crossfade_duration)

        # Prepare output path
        output_path = os.path.join(PREVIEWS_DIR, f"{output_filename}.mp4")
        
        # Write the montage
        final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

        # Clean up
        video.close()
        final_clip.close()
        for clip in clips:
            clip.close()

        return {
            "status": "success",
            "message": "Successfully created video montage",
            "preview": output_path,
            "clip_count": len(clips),
            "duration": final_clip.duration
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"An error occurred while creating video montage: {str(e)}"
        }

# Example usage:
# if __name__ == "__main__":
#     # Create GIF preview
#     result = create_gif_preview(
#         "input.mp4",
#         output_filename="preview_gif",
#         frame_interval=0.5,
#         max_frames=10,
#         resize_width=480,
#         fps=2
#     )
#     print(result)
#
#     # Create grid preview
#     result = create_grid_preview(
#         "input.mp4",
#         output_filename="preview_grid",
#         grid_rows=3,
#         grid_cols=3,
#         resize_width=320,
#         frame_quality=95
#     )
#     print(result)
#
#     # Create video montage
#     result = create_video_montage(
#         "input.mp4",
#         output_filename="preview_montage",
#         clips_duration=1.0,
#         clips_interval=5.0,
#         max_clips=6,
#         resize_width=720,
#         crossfade_duration=0.5
#     )
#     print(result) 