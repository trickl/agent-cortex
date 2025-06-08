"""
LLMFlow - A powerful framework for building AI agents based on GAME methodology
(Goals, Actions, Memory, Environment).

Data Analysis Tool - Provides data processing, analysis, and visualization capabilities with support for various data formats and statistical operations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import warnings
import time
from pathlib import Path

from ..core.base import BaseTool, ToolResponse

warnings.filterwarnings('ignore')

class DataAnalysisTool(BaseTool):
    """Data analysis and visualization tool implementation."""
    
    def __init__(self):
        super().__init__()
        self.supported_operations = [
            'analyze', 'visualize', 'summarize',
            'correlate', 'predict', 'cluster'
        ]
        self.output_dir = Path('analysis_output')
        self.output_dir.mkdir(exist_ok=True)
    
    def execute(self, operation: str, **kwargs) -> ToolResponse:
        """Execute a data analysis operation."""
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
            self.logger.error(f"Error in data analysis: {str(e)}")
            return ToolResponse(
                success=False,
                result=None,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    def _handle_analyze(self, data_path: str, analysis_type: str = 'basic') -> Dict[str, Any]:
        """Handle data analysis operation."""
        try:
            # Load data
            df = pd.read_csv(data_path)
            
            if analysis_type == 'basic':
                # Basic statistics
                stats = {
                    'shape': df.shape,
                    'columns': df.columns.tolist(),
                    'dtypes': df.dtypes.astype(str).to_dict(),
                    'missing_values': df.isnull().sum().to_dict(),
                    'numeric_summary': df.describe().to_dict()
                }
                
                return stats
                
            elif analysis_type == 'detailed':
                # Detailed analysis
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                categorical_cols = df.select_dtypes(exclude=[np.number]).columns
                
                analysis = {
                    'shape': df.shape,
                    'columns': {
                        'numeric': numeric_cols.tolist(),
                        'categorical': categorical_cols.tolist()
                    },
                    'numeric_stats': df[numeric_cols].describe().to_dict(),
                    'categorical_stats': {
                        col: df[col].value_counts().to_dict()
                        for col in categorical_cols
                    },
                    'correlations': df[numeric_cols].corr().to_dict()
                }
                
                return analysis
                
            else:
                raise ValueError(f"Unsupported analysis type: {analysis_type}")
                
        except Exception as e:
            raise Exception(f"Error analyzing data: {str(e)}")
    
    def _handle_visualize(self, data_path: str, plot_type: str,
                         x_col: str = None, y_col: str = None,
                         **kwargs) -> Dict[str, Any]:
        """Handle data visualization operation."""
        try:
            # Load data
            df = pd.read_csv(data_path)
            
            # Create plot
            if plot_type == 'scatter':
                fig = px.scatter(df, x=x_col, y=y_col, **kwargs)
            elif plot_type == 'line':
                fig = px.line(df, x=x_col, y=y_col, **kwargs)
            elif plot_type == 'bar':
                fig = px.bar(df, x=x_col, y=y_col, **kwargs)
            elif plot_type == 'histogram':
                fig = px.histogram(df, x=x_col, **kwargs)
            else:
                raise ValueError(f"Unsupported plot type: {plot_type}")
            
            # Save plot
            output_path = self.output_dir / f"{plot_type}_{int(time.time())}.html"
            fig.write_html(str(output_path))
            
            return {
                'plot_type': plot_type,
                'output_path': str(output_path),
                'columns_used': {'x': x_col, 'y': y_col}
            }
            
        except Exception as e:
            raise Exception(f"Error creating visualization: {str(e)}")
    
    @staticmethod
    def get_schema() -> Dict[str, Any]:
        """Get the tool's JSON schema."""
        return {
            'type': 'function',
            'function': {
                'name': 'data_analysis',
                'description': 'Analyze and visualize data',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'operation': {
                            'type': 'string',
                            'enum': ['analyze', 'visualize', 'summarize', 'correlate', 'predict', 'cluster'],
                            'description': 'The data analysis operation to perform'
                        },
                        'data_path': {
                            'type': 'string',
                            'description': 'Path to the input data file'
                        },
                        'analysis_type': {
                            'type': 'string',
                            'enum': ['basic', 'detailed'],
                            'description': 'Type of analysis to perform'
                        },
                        'plot_type': {
                            'type': 'string',
                            'enum': ['scatter', 'line', 'bar', 'histogram'],
                            'description': 'Type of plot to create'
                        },
                        'x_col': {
                            'type': 'string',
                            'description': 'Column to use for x-axis'
                        },
                        'y_col': {
                            'type': 'string',
                            'description': 'Column to use for y-axis'
                        }
                    },
                    'required': ['operation', 'data_path']
                }
            }
        } 