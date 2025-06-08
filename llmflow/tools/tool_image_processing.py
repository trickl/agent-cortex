"""
LLMFlow - A powerful framework for building AI agents based on GAME methodology
(Goals, Actions, Memory, Environment).

Image Processing Tool - Handles image manipulation, analysis, and generation with support for various formats and processing operations.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from PIL.ExifTags import TAGS
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import base64
import json
import os
import io
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib
import tempfile
from pathlib import Path
import logging
import colorsys
import time

from ..core.base import BaseTool, ToolResponse

@dataclass
class ImageMetadata:
    """Data class for image metadata."""
    filename: str
    format: str
    size: Tuple[int, int]
    mode: str
    file_size: int
    created_at: str
    exif_data: Dict[str, Any]
    color_profile: Dict[str, Any]

class ImageProcessingTool(BaseTool):
    """Advanced image processing tool implementation."""
    
    def __init__(self):
        super().__init__()
        self.temp_dir = tempfile.mkdtemp()
        self.supported_formats = ['jpg', 'jpeg', 'png', 'bmp', 'gif']
        self.supported_operations = [
            'resize', 'crop', 'rotate', 'filter', 'enhance',
            'analyze', 'generate', 'convert'
        ]
    
    def execute(self, operation: str, **kwargs) -> ToolResponse:
        """Execute an image processing operation."""
        start_time = time.time()
        
        try:
            # Validate operation
            if operation not in self.supported_operations:
                return ToolResponse(
                    success=False,
                    result=None,
                    error=f"Unsupported operation: {operation}",
                    execution_time=time.time() - start_time
                )
            
            # Route to appropriate handler
            handler = getattr(self, f'_handle_{operation}', None)
            if not handler:
                return ToolResponse(
                    success=False,
                    result=None,
                    error=f"Handler not implemented for operation: {operation}",
                    execution_time=time.time() - start_time
                )
            
            result = handler(**kwargs)
            execution_time = time.time() - start_time
            
            return ToolResponse(
                success=True,
                result=result,
                execution_time=execution_time,
                metadata={'operation': operation}
            )
            
        except Exception as e:
            self.logger.error(f"Error in image processing: {str(e)}")
            return ToolResponse(
                success=False,
                result=None,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    def _handle_resize(self, image_path: str, width: int, height: int, 
                      maintain_aspect: bool = True) -> Dict[str, Any]:
        """Handle image resize operation."""
        try:
            img = Image.open(image_path)
            
            if maintain_aspect:
                img.thumbnail((width, height))
            else:
                img = img.resize((width, height))
            
            output_path = os.path.join(self.temp_dir, f'resized_{os.path.basename(image_path)}')
            img.save(output_path)
            
            return {
                'output_path': output_path,
                'new_size': img.size,
                'original_size': Image.open(image_path).size
            }
            
        except Exception as e:
            raise Exception(f"Error resizing image: {str(e)}")
    
    def _handle_analyze(self, image_path: str) -> Dict[str, Any]:
        """Handle image analysis operation."""
        try:
            img = Image.open(image_path)
            img_array = np.array(img)
            
            # Basic analysis
            analysis = {
                'size': img.size,
                'mode': img.mode,
                'format': img.format,
                'mean_color': np.mean(img_array, axis=(0,1)).tolist(),
                'std_color': np.std(img_array, axis=(0,1)).tolist()
            }
            
            # Add histogram data
            hist_data = {}
            if img.mode == 'RGB':
                for i, channel in enumerate(['red', 'green', 'blue']):
                    hist_data[channel] = np.histogram(img_array[:,:,i], bins=256)[0].tolist()
            else:
                hist_data['intensity'] = np.histogram(img_array, bins=256)[0].tolist()
            
            analysis['histogram'] = hist_data
            
            return analysis
            
        except Exception as e:
            raise Exception(f"Error analyzing image: {str(e)}")
    
    def _handle_generate(self, type: str, **kwargs) -> Dict[str, Any]:
        """Handle image generation operation."""
        try:
            if type == 'solid_color':
                color = kwargs.get('color', (255, 255, 255))
                size = kwargs.get('size', (100, 100))
                img = Image.new('RGB', size, color)
            
            elif type == 'gradient':
                start_color = kwargs.get('start_color', (0, 0, 0))
                end_color = kwargs.get('end_color', (255, 255, 255))
                size = kwargs.get('size', (100, 100))
                direction = kwargs.get('direction', 'horizontal')
                
                img = Image.new('RGB', size)
                draw = ImageDraw.Draw(img)
                
                for i in range(size[0] if direction == 'horizontal' else size[1]):
                    t = i / (size[0] if direction == 'horizontal' else size[1])
                    color = tuple(int(start * (1-t) + end * t) for start, end in zip(start_color, end_color))
                    
                    if direction == 'horizontal':
                        draw.line([(i, 0), (i, size[1])], fill=color)
                    else:
                        draw.line([(0, i), (size[0], i)], fill=color)
            
            else:
                raise ValueError(f"Unsupported generation type: {type}")
            
            output_path = os.path.join(self.temp_dir, f'generated_{int(time.time())}.png')
            img.save(output_path)
            
            return {
                'output_path': output_path,
                'size': img.size,
                'type': type
            }
            
        except Exception as e:
            raise Exception(f"Error generating image: {str(e)}")
    
    @staticmethod
    def get_schema() -> Dict[str, Any]:
        """Get the tool's JSON schema."""
        return {
            'type': 'function',
            'function': {
                'name': 'image_processing',
                'description': 'Process, analyze, and generate images',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'operation': {
                            'type': 'string',
                            'enum': ['resize', 'crop', 'rotate', 'filter', 'enhance', 'analyze', 'generate', 'convert'],
                            'description': 'The image processing operation to perform'
                        },
                        'image_path': {
                            'type': 'string',
                            'description': 'Path to the input image'
                        },
                        'width': {
                            'type': 'integer',
                            'description': 'Width in pixels for resize operation'
                        },
                        'height': {
                            'type': 'integer',
                            'description': 'Height in pixels for resize operation'
                        },
                        'type': {
                            'type': 'string',
                            'enum': ['solid_color', 'gradient'],
                            'description': 'Type of image to generate'
                        }
                    },
                    'required': ['operation']
                }
            }
        } 