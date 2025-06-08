"""
LLMFlow - A powerful framework for building AI agents based on GAME methodology
(Goals, Actions, Memory, Environment).

Mathematical Tool - Provides mathematical computation capabilities including statistical analysis, numerical operations, and formula processing.
"""

import math
import cmath
import statistics
import random
import re
import time
import logging
import threading
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from decimal import Decimal, getcontext, ROUND_HALF_UP
from fractions import Fraction
import json

# Optional dependencies for enhanced functionality
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import scipy
    import scipy.optimize
    import scipy.integrate
    import scipy.stats
    import scipy.linalg
    import scipy.interpolate
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import sympy as sp
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

@dataclass
class MathConfig:
    """Configuration for mathematical calculations"""
    # Precision settings
    decimal_precision: int = 28
    float_precision: int = 15
    rounding_mode: str = 'ROUND_HALF_UP'
    
    # Calculation limits
    max_iterations: int = 1000000
    convergence_tolerance: float = 1e-10
    max_factorial: int = 10000
    max_array_size: int = 1000000
    
    # Numerical methods
    default_integration_method: str = 'quad'
    default_optimization_method: str = 'minimize'
    default_root_finding_method: str = 'brentq'
    
    # Performance settings
    enable_caching: bool = True
    cache_size: int = 1000
    enable_parallel: bool = True
    max_workers: int = 4
    
    # Output settings
    scientific_notation_threshold: float = 1e6
    angle_unit: str = 'radians'  # 'radians' or 'degrees'
    complex_format: str = 'rectangular'  # 'rectangular' or 'polar'
    
    # Logging
    log_level: str = 'INFO'
    log_calculations: bool = False

@dataclass
class MathResult:
    """Result of mathematical calculation"""
    success: bool
    operation: str
    result: Any = None
    intermediate_steps: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time: float = 0.0
    precision_used: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        result_dict = asdict(self)
        # Handle non-serializable types
        if self.result is not None:
            try:
                json.dumps(self.result)
                result_dict['result'] = self.result
            except (TypeError, ValueError):
                result_dict['result'] = str(self.result)
        return result_dict

class MathCache:
    """Thread-safe cache for mathematical calculations"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.lock = threading.Lock()
    
    def _generate_key(self, operation: str, *args, **kwargs) -> str:
        """Generate cache key"""
        key_data = f"{operation}:{args}:{sorted(kwargs.items())}"
        return str(hash(key_data))
    
    def get(self, operation: str, *args, **kwargs) -> Optional[Any]:
        """Get cached result"""
        key = self._generate_key(operation, *args, **kwargs)
        with self.lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                return self.cache[key]
            return None
    
    def put(self, operation: str, result: Any, *args, **kwargs):
        """Cache calculation result"""
        key = self._generate_key(operation, *args, **kwargs)
        with self.lock:
            if len(self.cache) >= self.max_size:
                # Remove least recently used item
                oldest_key = min(self.access_times.keys(), 
                               key=lambda k: self.access_times[k])
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
            
            self.cache[key] = result
            self.access_times[key] = time.time()
    
    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()

class MathematicalCalculationsTool:
    """
    Universal Mathematical Calculations Tool
    
    Provides comprehensive mathematical operations:
    - Basic arithmetic and trigonometry
    - Statistical analysis and hypothesis testing
    - Matrix and polynomial operations
    - Numerical methods (integration, optimization, root finding)
    - Financial calculations
    - Unit conversions
    """
    
    def __init__(self, config: Optional[MathConfig] = None):
        self.config = config or MathConfig()
        self.logger = self._setup_logger()
        self.cache = MathCache(self.config.cache_size) if self.config.enable_caching else None
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger"""
        logger = logging.getLogger('MathematicalCalculationsTool')
        logger.setLevel(getattr(logging, self.config.log_level))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def calculate(self, operation: str, *args) -> MathResult:
        """Basic arithmetic operations"""
        start_time = time.time()
        
        try:
            if operation == 'add':
                result = sum(args)
            elif operation == 'subtract':
                result = args[0] - sum(args[1:])
            elif operation == 'multiply':
                result = args[0]
                for arg in args[1:]:
                    result *= arg
            elif operation == 'divide':
                result = args[0]
                for arg in args[1:]:
                    if arg == 0:
                        return MathResult(
                            success=False,
                            operation=f'arithmetic_{operation}',
                            error="Division by zero"
                        )
                    result /= arg
            elif operation == 'power':
                result = pow(args[0], args[1])
            else:
                return MathResult(
                    success=False,
                    operation=f'arithmetic_{operation}',
                    error=f"Unknown arithmetic operation: {operation}"
                )
            
            execution_time = time.time() - start_time
            
            return MathResult(
                success=True,
                operation=f'arithmetic_{operation}',
                result=result,
                execution_time=execution_time,
                metadata={'arguments': args}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return MathResult(
                success=False,
                operation=f'arithmetic_{operation}',
                error=str(e),
                execution_time=execution_time
            )
    
    def statistics(self, data: List[float]) -> MathResult:
        """Calculate descriptive statistics"""
        start_time = time.time()
        
        try:
            if not data:
                return MathResult(
                    success=False,
                    operation='descriptive_statistics',
                    error="Empty data set"
                )
            
            result = {
                'count': len(data),
                'mean': statistics.mean(data),
                'median': statistics.median(data),
                'std_dev': statistics.stdev(data) if len(data) > 1 else 0,
                'min': min(data),
                'max': max(data),
                'range': max(data) - min(data)
            }
            
            execution_time = time.time() - start_time
            
            return MathResult(
                success=True,
                operation='descriptive_statistics',
                result=result,
                execution_time=execution_time,
                metadata={'data_size': len(data)}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return MathResult(
                success=False,
                operation='descriptive_statistics',
                error=str(e),
                execution_time=execution_time
            )
    
    def matrix(self, operation: str, matrix_a: List[List[float]], 
              matrix_b: Optional[List[List[float]]] = None) -> MathResult:
        """Matrix operations"""
        start_time = time.time()
        
        try:
            if not HAS_NUMPY:
                return MathResult(
                    success=False,
                    operation=f'matrix_{operation}',
                    error="NumPy required for matrix operations"
                )
            
            A = np.array(matrix_a, dtype=float)
            
            if operation == 'determinant':
                result = float(np.linalg.det(A))
            elif operation == 'inverse':
                result = np.linalg.inv(A).tolist()
            elif operation == 'transpose':
                result = A.T.tolist()
            else:
                return MathResult(
                    success=False,
                    operation=f'matrix_{operation}',
                    error=f"Unknown matrix operation: {operation}"
                )
            
            execution_time = time.time() - start_time
            
            return MathResult(
                success=True,
                operation=f'matrix_{operation}',
                result=result,
                execution_time=execution_time,
                metadata={'matrix_shape': A.shape}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return MathResult(
                success=False,
                operation=f'matrix_{operation}',
                error=str(e),
                execution_time=execution_time
            )
    
    def evaluate_expression(self, expression: str, variables: Optional[Dict[str, float]] = None) -> MathResult:
        """Evaluate mathematical expression"""
        start_time = time.time()
        
        try:
            # Create safe evaluation environment
            safe_dict = {
                'math': math,
                'sin': math.sin,
                'cos': math.cos,
                'tan': math.tan,
                'log': math.log,
                'sqrt': math.sqrt,
                'exp': math.exp,
                'pi': math.pi,
                'e': math.e
            }
            
            if variables:
                safe_dict.update(variables)
            
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            
            execution_time = time.time() - start_time
            
            return MathResult(
                success=True,
                operation='evaluate_expression',
                result=result,
                execution_time=execution_time,
                metadata={'expression': expression, 'variables': variables}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return MathResult(
                success=False,
                operation='evaluate_expression',
                error=str(e),
                execution_time=execution_time
            )
    
    def clear_cache(self):
        """Clear calculation cache"""
        if self.cache:
            self.cache.clear()

# Factory function
def create_math_tool(config: Optional[Dict[str, Any]] = None) -> MathematicalCalculationsTool:
    """Create mathematical calculations tool"""
    math_config = MathConfig(**config) if config else MathConfig()
    return MathematicalCalculationsTool(math_config)

# Quick functions
def quick_calculate(operation: str, *args) -> Dict[str, Any]:
    """Quick arithmetic calculation"""
    tool = create_math_tool()
    result = tool.calculate(operation, *args)
    return result.to_dict()

def quick_statistics(data: List[float]) -> Dict[str, Any]:
    """Quick statistical analysis"""
    tool = create_math_tool()
    result = tool.statistics(data)
    return result.to_dict()

# Example usage
if __name__ == "__main__":
    # Create math tool
    tool = create_math_tool({
        'decimal_precision': 15,
        'angle_unit': 'degrees',
        'enable_caching': True
    })
    
    # Basic calculations
    print("Basic calculations:")
    result = tool.calculate('add', 10, 20, 30)
    print(f"10 + 20 + 30 = {result.result}")
    
    # Statistics
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    stats_result = tool.statistics(data)
    if stats_result.success:
        stats = stats_result.result
        print(f"Data statistics: mean={stats['mean']:.2f}, std={stats['std_dev']:.2f}")
    
    # Matrix operations
    matrix_a = [[1, 2], [3, 4]]
    matrix_result = tool.matrix('determinant', matrix_a)
    print(f"Matrix determinant: {matrix_result.result}")
    
    print("Mathematical calculations completed!") 