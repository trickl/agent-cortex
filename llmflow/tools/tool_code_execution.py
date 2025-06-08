"""
LLMFlow - A powerful framework for building AI agents based on GAME methodology
(Goals, Actions, Memory, Environment).

Code Execution Tool - Enables secure code execution in various programming languages with sandboxing, output capture, and error handling.
"""

import os
import sys
import subprocess
import tempfile
import time
import json
import threading
import logging
import traceback
import hashlib
import signal
import psutil
import shutil
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from io import StringIO
import concurrent.futures
import queue
import shlex

from llmflow.tools.tool_decorator import register_tool

# Optional dependencies for enhanced functionality
try:
    import docker
    HAS_DOCKER = True
except ImportError:
    HAS_DOCKER = False

try:
    import jupyter_client
    HAS_JUPYTER = True
except ImportError:
    HAS_JUPYTER = False

@dataclass
class CodeExecutionConfig:
    """Configuration for code execution"""
    # Security settings
    max_execution_time: int = 30  # seconds
    max_memory_mb: int = 512
    max_output_size: int = 10 * 1024 * 1024  # 10MB
    enable_networking: bool = False
    enable_file_system: bool = False
    allowed_imports: Optional[List[str]] = None
    blocked_imports: List[str] = field(default_factory=lambda: [
        'os', 'sys', 'subprocess', 'importlib', '__import__',
        'eval', 'exec', 'compile', 'open', 'file'
    ])
    
    # Language settings
    supported_languages: List[str] = field(default_factory=lambda: [
        'python', 'javascript', 'bash', 'powershell', 'r', 'sql', 'go', 'rust'
    ])
    language_commands: Dict[str, str] = field(default_factory=lambda: {
        'python': 'python3',
        'javascript': 'node',
        'bash': 'bash',
        'powershell': 'powershell',
        'r': 'Rscript',
        'sql': 'sqlite3',
        'go': 'go run',
        'rust': 'rustc'
    })
    
    # Execution environment
    use_containers: bool = False
    container_image: str = 'python:3.11-alpine'
    working_directory: Optional[str] = None
    environment_variables: Dict[str, str] = field(default_factory=dict)
    
    # Performance settings
    enable_cache: bool = True
    cache_ttl: int = 300  # 5 minutes
    max_concurrent_executions: int = 3
    
    # Logging and monitoring
    log_level: str = 'INFO'
    save_execution_history: bool = True
    max_history_size: int = 1000

@dataclass
class ExecutionResult:
    """Result of code execution"""
    success: bool
    language: str
    code: str
    stdout: str = ""
    stderr: str = ""
    return_value: Any = None
    execution_time: float = 0.0
    memory_used: int = 0  # MB
    exit_code: int = 0
    error_type: Optional[str] = None
    security_violations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        # Convert non-serializable objects
        if self.return_value is not None:
            try:
                json.dumps(self.return_value)
                result['return_value'] = self.return_value
            except (TypeError, ValueError):
                result['return_value'] = str(self.return_value)
        return result

class SecurityValidator:
    """Security validation for code execution"""
    
    def __init__(self, config: CodeExecutionConfig):
        self.config = config
        
        # Python security patterns
        self.dangerous_python_patterns = [
            r'__import__\s*\(',
            r'eval\s*\(',
            r'exec\s*\(',
            r'compile\s*\(',
            r'open\s*\(',
            r'file\s*\(',
            r'input\s*\(',
            r'raw_input\s*\(',
            r'globals\s*\(',
            r'locals\s*\(',
            r'vars\s*\(',
            r'dir\s*\(',
            r'getattr\s*\(',
            r'setattr\s*\(',
            r'delattr\s*\(',
            r'hasattr\s*\(',
            r'import\s+os',
            r'from\s+os',
            r'import\s+sys',
            r'from\s+sys',
            r'import\s+subprocess',
            r'from\s+subprocess',
        ]
        
        # JavaScript security patterns
        self.dangerous_js_patterns = [
            r'require\s*\(',
            r'process\.',
            r'global\.',
            r'eval\s*\(',
            r'Function\s*\(',
            r'setTimeout\s*\(',
            r'setInterval\s*\(',
            r'XMLHttpRequest',
            r'fetch\s*\(',
            r'import\s*\(',
            r'document\.',
            r'window\.',
        ]
        
        # Shell command patterns
        self.dangerous_shell_patterns = [
            r'rm\s+-rf',
            r'sudo\s+',
            r'su\s+',
            r'chmod\s+',
            r'chown\s+',
            r'wget\s+',
            r'curl\s+',
            r'nc\s+',
            r'netcat\s+',
            r'telnet\s+',
            r'ssh\s+',
            r'scp\s+',
            r'rsync\s+',
            r'dd\s+if=',
            r'mkfs\.',
            r'fdisk\s+',
            r'mount\s+',
            r'umount\s+',
        ]
    
    def validate_code(self, code: str, language: str) -> Tuple[bool, List[str]]:
        """Validate code for security issues"""
        violations = []
        import re
        
        if language == 'python':
            violations.extend(self._check_python_security(code))
        elif language == 'javascript':
            violations.extend(self._check_javascript_security(code))
        elif language in ['bash', 'powershell']:
            violations.extend(self._check_shell_security(code))
        
        # Check for blocked imports
        if self.config.blocked_imports:
            for blocked in self.config.blocked_imports:
                if blocked in code:
                    violations.append(f"Blocked import/function: {blocked}")
        
        # Check for networking if disabled
        if not self.config.enable_networking:
            network_patterns = [
                r'socket\.',
                r'urllib\.',
                r'requests\.',
                r'http\.',
                r'ftp\.',
                r'smtp\.',
            ]
            for pattern in network_patterns:
                if re.search(pattern, code):
                    violations.append(f"Network access not allowed: {pattern}")
        
        # Check for file system access if disabled
        if not self.config.enable_file_system:
            fs_patterns = [
                r'open\s*\(',
                r'file\s*\(',
                r'os\.path',
                r'pathlib\.',
                r'shutil\.',
                r'glob\.',
            ]
            for pattern in fs_patterns:
                if re.search(pattern, code):
                    violations.append(f"File system access not allowed: {pattern}")
        
        return len(violations) == 0, violations
    
    def _check_python_security(self, code: str) -> List[str]:
        """Check Python-specific security issues"""
        violations = []
        import re
        
        for pattern in self.dangerous_python_patterns:
            if re.search(pattern, code):
                violations.append(f"Dangerous Python pattern: {pattern}")
        
        return violations
    
    def _check_javascript_security(self, code: str) -> List[str]:
        """Check JavaScript-specific security issues"""
        violations = []
        import re
        
        for pattern in self.dangerous_js_patterns:
            if re.search(pattern, code):
                violations.append(f"Dangerous JavaScript pattern: {pattern}")
        
        return violations
    
    def _check_shell_security(self, code: str) -> List[str]:
        """Check shell script security issues"""
        violations = []
        import re
        
        for pattern in self.dangerous_shell_patterns:
            if re.search(pattern, code):
                violations.append(f"Dangerous shell pattern: {pattern}")
        
        return violations

class ExecutionCache:
    """Cache for code execution results"""
    
    def __init__(self, ttl: int = 300):
        self.cache = {}
        self.timestamps = {}
        self.ttl = ttl
        self.lock = threading.Lock()
    
    def get_cache_key(self, code: str, language: str, context: Dict) -> str:
        """Generate cache key"""
        content = f"{language}:{code}:{json.dumps(context, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[ExecutionResult]:
        """Get cached result"""
        with self.lock:
            if key in self.cache:
                if time.time() - self.timestamps[key] < self.ttl:
                    return self.cache[key]
                else:
                    del self.cache[key]
                    del self.timestamps[key]
            return None
    
    def put(self, key: str, result: ExecutionResult):
        """Cache execution result"""
        with self.lock:
            self.cache[key] = result
            self.timestamps[key] = time.time()
    
    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()

class ResourceMonitor:
    """Monitor resource usage during execution"""
    
    def __init__(self, max_memory_mb: int, max_time: int):
        self.max_memory_mb = max_memory_mb
        self.max_time = max_time
        self.process = None
        self.start_time = None
        self.peak_memory = 0
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self, process):
        """Start monitoring process"""
        self.process = process
        self.start_time = time.time()
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
    
    def _monitor_loop(self):
        """Monitor loop"""
        try:
            while self.monitoring and self.process:
                # Check execution time
                if time.time() - self.start_time > self.max_time:
                    self._terminate_process("Execution timeout")
                    break
                
                # Check memory usage
                try:
                    if self.process.poll() is None:  # Process still running
                        ps_process = psutil.Process(self.process.pid)
                        memory_mb = ps_process.memory_info().rss / 1024 / 1024
                        self.peak_memory = max(self.peak_memory, memory_mb)
                        
                        if memory_mb > self.max_memory_mb:
                            self._terminate_process("Memory limit exceeded")
                            break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
                
                time.sleep(0.1)  # Check every 100ms
        except Exception:
            pass
    
    def _terminate_process(self, reason: str):
        """Terminate process"""
        try:
            if self.process and self.process.poll() is None:
                # Try graceful termination first
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    # Force kill if needed
                    self.process.kill()
                    self.process.wait()
        except Exception:
            pass

class LanguageExecutor:
    """Base class for language-specific executors"""
    
    def __init__(self, config: CodeExecutionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def execute(self, code: str, context: Dict[str, Any]) -> ExecutionResult:
        """Execute code (to be implemented by subclasses)"""
        raise NotImplementedError
    
    def _prepare_execution_environment(self) -> Dict[str, Any]:
        """Prepare execution environment"""
        env = os.environ.copy()
        env.update(self.config.environment_variables)
        return env
    
    def _setup_working_directory(self) -> str:
        """Setup working directory"""
        return self.config.working_directory or tempfile.mkdtemp()
    
    def _cleanup_resources(self, temp_dir: Optional[str] = None):
        """Cleanup temporary resources"""
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except:
                pass

class PythonExecutor(LanguageExecutor):
    """Python code executor with safety features"""
    
    def execute(self, code: str, context: Dict[str, Any]) -> ExecutionResult:
        """Execute Python code safely"""
        start_time = time.time()
        
        try:
            # Prepare safe execution environment
            safe_globals = self._create_safe_globals()
            safe_locals = context.get('variables', {}).copy()
            
            # Capture output
            stdout_capture = StringIO()
            stderr_capture = StringIO()
            
            result_value = None
            
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                try:
                    # Try to evaluate as expression first
                    try:
                        result_value = eval(code, safe_globals, safe_locals)
                    except SyntaxError:
                        # If not an expression, execute as statements
                        exec(code, safe_globals, safe_locals)
                        
                        # Try to get result from last line if it's an expression
                        lines = code.strip().split('\n')
                        if lines:
                            last_line = lines[-1].strip()
                            if last_line and not any(last_line.startswith(kw) for kw in 
                                                   ['if', 'for', 'while', 'def', 'class', 'try', 'with', 'import']):
                                try:
                                    result_value = eval(last_line, safe_globals, safe_locals)
                                except:
                                    pass
                
                except Exception as e:
                    execution_time = time.time() - start_time
                    return ExecutionResult(
                        success=False,
                        language='python',
                        code=code,
                        stderr=str(e),
                        execution_time=execution_time,
                        error_type=type(e).__name__
                    )
            
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                success=True,
                language='python',
                code=code,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                return_value=result_value,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ExecutionResult(
                success=False,
                language='python',
                code=code,
                stderr=str(e),
                execution_time=execution_time,
                error_type=type(e).__name__
            )
    
    def _create_safe_globals(self) -> Dict[str, Any]:
        """Create safe global environment for Python execution"""
        safe_builtins = {
            'abs', 'all', 'any', 'bin', 'bool', 'chr', 'complex',
            'dict', 'divmod', 'enumerate', 'filter', 'float', 'format',
            'frozenset', 'hex', 'int', 'iter', 'len', 'list', 'map',
            'max', 'min', 'oct', 'ord', 'pow', 'range', 'reversed',
            'round', 'set', 'slice', 'sorted', 'str', 'sum', 'tuple',
            'type', 'zip', 'print'
        }
        
        # Create restricted builtins
        restricted_builtins = {}
        for name in safe_builtins:
            if hasattr(__builtins__, name):
                restricted_builtins[name] = getattr(__builtins__, name)
        
        # Add safe modules
        safe_globals = {
            '__builtins__': restricted_builtins,
            'math': __import__('math'),
            'random': __import__('random'),
            'datetime': __import__('datetime'),
            'json': __import__('json'),
            'time': __import__('time'),
            'statistics': __import__('statistics'),
        }
        
        # Add allowed imports
        if self.config.allowed_imports:
            for module_name in self.config.allowed_imports:
                try:
                    safe_globals[module_name] = __import__(module_name)
                except ImportError:
                    pass
        
        return safe_globals 

class ExternalLanguageExecutor(LanguageExecutor):
    """Executor for external languages (JavaScript, Go, etc.)"""
    
    def execute(self, code: str, context: Dict[str, Any]) -> ExecutionResult:
        """Execute code in external language"""
        start_time = time.time()
        language = context.get('language', 'javascript')
        
        try:
            if language not in self.config.language_commands:
                return ExecutionResult(
                    success=False,
                    language=language,
                    code=code,
                    stderr=f"Unsupported language: {language}",
                    error_type="UnsupportedLanguage"
                )
            
            # Create temporary file
            file_extension = self._get_file_extension(language)
            temp_file = tempfile.NamedTemporaryFile(
                mode='w', 
                suffix=file_extension, 
                delete=False,
                encoding='utf-8'
            )
            
            # Add language-specific setup
            full_code = self._prepare_code(code, language, context)
            temp_file.write(full_code)
            temp_file.close()
            
            try:
                # Execute code
                result = self._execute_external_command(
                    language, 
                    temp_file.name, 
                    context
                )
                
                execution_time = time.time() - start_time
                result.execution_time = execution_time
                
                return result
                
            finally:
                # Cleanup
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
                    
        except Exception as e:
            execution_time = time.time() - start_time
            return ExecutionResult(
                success=False,
                language=language,
                code=code,
                stderr=str(e),
                execution_time=execution_time,
                error_type=type(e).__name__
            )
    
    def _get_file_extension(self, language: str) -> str:
        """Get file extension for language"""
        extensions = {
            'javascript': '.js',
            'python': '.py',
            'bash': '.sh',
            'powershell': '.ps1',
            'r': '.R',
            'sql': '.sql',
            'go': '.go',
            'rust': '.rs',
            'java': '.java',
            'cpp': '.cpp',
            'c': '.c'
        }
        return extensions.get(language, '.txt')
    
    def _prepare_code(self, code: str, language: str, context: Dict[str, Any]) -> str:
        """Prepare code with language-specific setup"""
        if language == 'javascript':
            # Add console capture for Node.js
            setup = """
const originalLog = console.log;
const outputs = [];
console.log = (...args) => {
    outputs.push(args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : String(arg)).join(' '));
    originalLog(...args);
};

try {
"""
            teardown = """
} catch (error) {
    console.error('Error:', error.message);
    process.exit(1);
}
"""
            return setup + code + teardown
        
        elif language == 'go':
            # Add package main if not present
            if 'package main' not in code:
                return f"""package main

import "fmt"

func main() {{
{code}
}}"""
            return code
        
        elif language == 'rust':
            # Add main function if not present
            if 'fn main()' not in code:
                return f"""fn main() {{
{code}
}}"""
            return code
        
        return code
    
    def _execute_external_command(self, language: str, file_path: str, 
                                 context: Dict[str, Any]) -> ExecutionResult:
        """Execute external command"""
        command = self.config.language_commands[language]
        
        # Build command
        if language == 'go':
            cmd = ['go', 'run', file_path]
        elif language == 'rust':
            # Compile and run Rust
            exe_path = file_path.replace('.rs', '')
            compile_result = subprocess.run(
                ['rustc', file_path, '-o', exe_path],
                capture_output=True,
                text=True,
                timeout=self.config.max_execution_time
            )
            if compile_result.returncode != 0:
                return ExecutionResult(
                    success=False,
                    language=language,
                    code="",
                    stderr=compile_result.stderr,
                    exit_code=compile_result.returncode,
                    error_type="CompilationError"
                )
            cmd = [exe_path]
        else:
            cmd = command.split() + [file_path]
        
        # Setup environment
        env = os.environ.copy()
        env.update(self.config.environment_variables)
        
        # Set working directory
        cwd = self.config.working_directory or os.path.dirname(file_path)
        
        # Execute with monitoring
        monitor = ResourceMonitor(
            self.config.max_memory_mb,
            self.config.max_execution_time
        )
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=cwd
            )
            
            monitor.start_monitoring(process)
            
            try:
                stdout, stderr = process.communicate(timeout=self.config.max_execution_time)
                exit_code = process.returncode
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                exit_code = -1
                stderr += "\nExecution timed out"
            
            monitor.stop_monitoring()
            
            return ExecutionResult(
                success=exit_code == 0,
                language=language,
                code="",
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                memory_used=int(monitor.peak_memory)
            )
            
        except FileNotFoundError:
            return ExecutionResult(
                success=False,
                language=language,
                code="",
                stderr=f"Language runtime not found: {command}",
                error_type="RuntimeNotFound"
            )

class DockerExecutor(LanguageExecutor):
    """Docker-based code executor for enhanced security"""
    
    def __init__(self, config: CodeExecutionConfig):
        super().__init__(config)
        if not HAS_DOCKER:
            raise ImportError("Docker package not installed")
        
        try:
            self.client = docker.from_env()
        except Exception as e:
            raise RuntimeError(f"Cannot connect to Docker: {e}")
    
    def execute(self, code: str, context: Dict[str, Any]) -> ExecutionResult:
        """Execute code in Docker container"""
        start_time = time.time()
        language = context.get('language', 'python')
        
        try:
            # Prepare code file
            file_extension = self._get_file_extension(language)
            code_content = self._prepare_code(code, language)
            
            # Create container
            container_config = {
                'image': self._get_image_for_language(language),
                'command': self._get_command_for_language(language),
                'detach': True,
                'mem_limit': f"{self.config.max_memory_mb}m",
                'network_disabled': not self.config.enable_networking,
                'remove': True,
                'stdin_open': True,
                'environment': self.config.environment_variables
            }
            
            container = self.client.containers.run(**container_config)
            
            try:
                # Send code to container
                self._send_code_to_container(container, code_content, language)
                
                # Wait for execution with timeout
                exit_code = container.wait(timeout=self.config.max_execution_time)
                
                # Get output
                stdout = container.logs(stdout=True, stderr=False).decode('utf-8')
                stderr = container.logs(stdout=False, stderr=True).decode('utf-8')
                
                execution_time = time.time() - start_time
                
                return ExecutionResult(
                    success=exit_code['StatusCode'] == 0,
                    language=language,
                    code=code,
                    stdout=stdout,
                    stderr=stderr,
                    execution_time=execution_time,
                    exit_code=exit_code['StatusCode']
                )
                
            finally:
                # Cleanup container
                try:
                    container.stop(timeout=1)
                    container.remove()
                except:
                    pass
                    
        except Exception as e:
            execution_time = time.time() - start_time
            return ExecutionResult(
                success=False,
                language=language,
                code=code,
                stderr=str(e),
                execution_time=execution_time,
                error_type=type(e).__name__
            )
    
    def _get_image_for_language(self, language: str) -> str:
        """Get Docker image for language"""
        images = {
            'python': 'python:3.11-alpine',
            'javascript': 'node:18-alpine',
            'go': 'golang:1.21-alpine',
            'rust': 'rust:1.75-alpine',
            'java': 'openjdk:11-alpine',
            'ruby': 'ruby:3.2-alpine'
        }
        return images.get(language, self.config.container_image)
    
    def _get_command_for_language(self, language: str) -> List[str]:
        """Get command for language"""
        commands = {
            'python': ['python', '-c'],
            'javascript': ['node', '-e'],
            'go': ['go', 'run'],
            'rust': ['rustc'],
            'java': ['javac'],
            'ruby': ['ruby', '-e']
        }
        return commands.get(language, ['sh', '-c'])
    
    def _send_code_to_container(self, container, code: str, language: str):
        """Send code to container"""
        if language in ['python', 'javascript', 'ruby']:
            container.exec_run(['sh', '-c', f'echo "{code}" > /tmp/code.{language}'])
        else:
            container.exec_run(['sh', '-c', f'echo "{code}" > /tmp/code'])

class CodeExecutionTool:
    """
    Universal Code Execution Tool
    
    Features:
    - Multi-language support (Python, JavaScript, Go, Rust, etc.)
    - Security validation and sandboxing
    - Resource monitoring and limits
    - Caching for performance
    - Docker isolation (optional)
    """
    
    def __init__(self, config: Optional[CodeExecutionConfig] = None):
        self.config = config or CodeExecutionConfig()
        self.logger = self._setup_logger()
        
        # Initialize components
        self.security_validator = SecurityValidator(self.config)
        self.cache = ExecutionCache(self.config.cache_ttl) if self.config.enable_cache else None
        
        # Initialize executors
        self.executors = {
            'python': PythonExecutor(self.config),
        }
        
        # Add external language executor
        external_languages = ['javascript', 'bash', 'powershell', 'r', 'go', 'rust']
        for lang in external_languages:
            if lang in self.config.supported_languages:
                self.executors[lang] = ExternalLanguageExecutor(self.config)
        
        # Add Docker executor if available
        if self.config.use_containers and HAS_DOCKER:
            try:
                docker_executor = DockerExecutor(self.config)
                for lang in self.config.supported_languages:
                    self.executors[f"{lang}_docker"] = docker_executor
            except Exception as e:
                self.logger.warning(f"Docker executor not available: {e}")
        
        # Execution history
        self.execution_history = []
        self.execution_semaphore = threading.Semaphore(self.config.max_concurrent_executions)
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger"""
        logger = logging.getLogger('CodeExecutionTool')
        logger.setLevel(getattr(logging, self.config.log_level))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def execute_code(self, code: str, language: str, 
                    context: Optional[Dict[str, Any]] = None,
                    use_docker: bool = False) -> ExecutionResult:
        """
        Execute code in specified language
        
        Args:
            code: Source code to execute
            language: Programming language
            context: Execution context (variables, options)
            use_docker: Use Docker for execution (if available)
        
        Returns:
            ExecutionResult with execution details
        """
        start_time = time.time()
        context = context or {}
        
        # Validate language support
        if language not in self.config.supported_languages:
            return ExecutionResult(
                success=False,
                language=language,
                code=code,
                stderr=f"Unsupported language: {language}",
                error_type="UnsupportedLanguage"
            )
        
        # Security validation
        is_safe, violations = self.security_validator.validate_code(code, language)
        if not is_safe:
            return ExecutionResult(
                success=False,
                language=language,
                code=code,
                stderr="Security validation failed",
                security_violations=violations,
                error_type="SecurityViolation"
            )
        
        # Check cache
        if self.cache:
            cache_key = self.cache.get_cache_key(code, language, context)
            cached_result = self.cache.get(cache_key)
            if cached_result:
                self.logger.info(f"Cache hit for {language} code execution")
                return cached_result
        
        # Execute with concurrency control
        with self.execution_semaphore:
            try:
                # Choose executor
                executor_key = language
                if use_docker and f"{language}_docker" in self.executors:
                    executor_key = f"{language}_docker"
                
                if executor_key not in self.executors:
                    return ExecutionResult(
                        success=False,
                        language=language,
                        code=code,
                        stderr=f"No executor available for {language}",
                        error_type="ExecutorNotFound"
                    )
                
                # Execute code
                context['language'] = language
                executor = self.executors[executor_key]
                result = executor.execute(code, context)
                
                # Cache successful results
                if self.cache and result.success:
                    cache_key = self.cache.get_cache_key(code, language, context)
                    self.cache.put(cache_key, result)
                
                # Add to history
                if self.config.save_execution_history:
                    self._add_to_history(result)
                
                self.logger.info(f"Code execution completed: {language}, success={result.success}, time={result.execution_time:.3f}s")
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                error_result = ExecutionResult(
                    success=False,
                    language=language,
                    code=code,
                    stderr=str(e),
                    execution_time=execution_time,
                    error_type=type(e).__name__
                )
                
                if self.config.save_execution_history:
                    self._add_to_history(error_result)
                
                return error_result
    
    def execute_multi_language(self, code_blocks: List[Dict[str, str]], 
                             context: Optional[Dict[str, Any]] = None) -> List[ExecutionResult]:
        """
        Execute multiple code blocks in different languages
        
        Args:
            code_blocks: List of {'code': str, 'language': str} dictionaries
            context: Shared execution context
        
        Returns:
            List of ExecutionResults
        """
        results = []
        shared_context = context or {}
        
        for block in code_blocks:
            code = block.get('code', '')
            language = block.get('language', 'python')
            block_context = shared_context.copy()
            block_context.update(block.get('context', {}))
            
            result = self.execute_code(code, language, block_context)
            results.append(result)
            
            # Pass successful results to next block
            if result.success and result.return_value is not None:
                shared_context[f'result_{len(results)}'] = result.return_value
        
        return results
    
    def validate_syntax(self, code: str, language: str) -> Dict[str, Any]:
        """
        Validate code syntax without executing
        
        Args:
            code: Source code to validate
            language: Programming language
        
        Returns:
            Validation result
        """
        try:
            if language == 'python':
                compile(code, '<string>', 'exec')
                return {'valid': True, 'errors': []}
            
            elif language == 'javascript':
                # Basic JavaScript syntax check using Node.js
                temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False)
                temp_file.write(f"try {{ {code} }} catch(e) {{ console.error('Syntax Error:', e.message); process.exit(1); }}")
                temp_file.close()
                
                try:
                    result = subprocess.run(
                        ['node', '--check', temp_file.name],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    
                    return {
                        'valid': result.returncode == 0,
                        'errors': [result.stderr] if result.stderr else []
                    }
                finally:
                    os.unlink(temp_file.name)
            
            else:
                return {'valid': True, 'errors': ['Syntax validation not available for this language']}
                
        except SyntaxError as e:
            return {
                'valid': False,
                'errors': [f"Syntax Error: {e.msg} at line {e.lineno}"]
            }
        except Exception as e:
            return {
                'valid': False,
                'errors': [str(e)]
            }
    
    def get_execution_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get execution history"""
        history = self.execution_history
        if limit:
            history = history[-limit:]
        
        return [result.to_dict() for result in history]
    
    def clear_cache(self):
        """Clear execution cache"""
        if self.cache:
            self.cache.clear()
    
    def clear_history(self):
        """Clear execution history"""
        self.execution_history.clear()
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return self.config.supported_languages.copy()
    
    def get_language_info(self, language: str) -> Dict[str, Any]:
        """Get information about a specific language"""
        if language not in self.config.supported_languages:
            return {'supported': False, 'error': 'Language not supported'}
        
        info = {
            'supported': True,
            'command': self.config.language_commands.get(language, 'unknown'),
            'executor_available': language in self.executors,
            'docker_available': f"{language}_docker" in self.executors
        }
        
        # Check if runtime is available
        if language in self.config.language_commands:
            try:
                cmd = self.config.language_commands[language].split()[0]
                result = subprocess.run(['which', cmd], capture_output=True)
                info['runtime_available'] = result.returncode == 0
            except:
                info['runtime_available'] = False
        
        return info
    
    def _add_to_history(self, result: ExecutionResult):
        """Add execution result to history"""
        self.execution_history.append(result)
        
        # Limit history size
        if len(self.execution_history) > self.config.max_history_size:
            self.execution_history = self.execution_history[-self.config.max_history_size:]

class CodeExecutionAgent:
    """Agent interface for code execution"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        exec_config = CodeExecutionConfig(**config) if config else CodeExecutionConfig()
        self.executor = CodeExecutionTool(exec_config)
    
    def execute(self, operation: str, **params) -> Dict[str, Any]:
        """
        Execute code operation
        
        Operations:
        - execute_code: Execute single code block
        - execute_multi: Execute multiple code blocks
        - validate_syntax: Validate code syntax
        - get_history: Get execution history
        - get_languages: Get supported languages
        - get_language_info: Get language information
        """
        try:
            if operation == 'execute_code':
                result = self.executor.execute_code(
                    params['code'],
                    params['language'],
                    params.get('context'),
                    params.get('use_docker', False)
                )
                return result.to_dict()
            
            elif operation == 'execute_multi':
                results = self.executor.execute_multi_language(
                    params['code_blocks'],
                    params.get('context')
                )
                return {
                    'success': True,
                    'operation': operation,
                    'results': [r.to_dict() for r in results]
                }
            
            elif operation == 'validate_syntax':
                validation = self.executor.validate_syntax(
                    params['code'],
                    params['language']
                )
                return {
                    'success': True,
                    'operation': operation,
                    'result': validation
                }
            
            elif operation == 'get_history':
                history = self.executor.get_execution_history(params.get('limit'))
                return {
                    'success': True,
                    'operation': operation,
                    'result': history
                }
            
            elif operation == 'get_languages':
                languages = self.executor.get_supported_languages()
                return {
                    'success': True,
                    'operation': operation,
                    'result': languages
                }
            
            elif operation == 'get_language_info':
                info = self.executor.get_language_info(params['language'])
                return {
                    'success': True,
                    'operation': operation,
                    'result': info
                }
            
            elif operation == 'clear_cache':
                self.executor.clear_cache()
                return {'success': True, 'operation': operation}
            
            elif operation == 'clear_history':
                self.executor.clear_history()
                return {'success': True, 'operation': operation}
            
            else:
                return {'success': False, 'error': f'Unknown operation: {operation}'}
                
        except Exception as e:
            return {'success': False, 'operation': operation, 'error': str(e)}

# Factory function
def create_code_executor(config: Optional[Dict[str, Any]] = None) -> CodeExecutionAgent:
    """Create code execution agent"""
    return CodeExecutionAgent(config)

# Quick functions
def execute_python(code: str, variables: Optional[Dict] = None) -> Dict[str, Any]:
    """Quick Python execution"""
    agent = create_code_executor()
    return agent.execute('execute_code', 
                        code=code, 
                        language='python', 
                        context={'variables': variables or {}})

def execute_javascript(code: str) -> Dict[str, Any]:
    """Quick JavaScript execution"""
    agent = create_code_executor()
    return agent.execute('execute_code', code=code, language='javascript')

# Example usage
if __name__ == "__main__":
    # Create code executor
    executor = create_code_executor({
        'max_execution_time': 30,
        'max_memory_mb': 256,
        'supported_languages': ['python', 'javascript', 'bash'],
        'enable_cache': True,
        'save_execution_history': True
    })
    
    # Execute Python code
    python_result = executor.execute('execute_code',
                                   code='print("Hello from Python!"); result = 2 + 2',
                                   language='python')
    print("Python result:", python_result)
    
    # Execute JavaScript code  
    js_result = executor.execute('execute_code',
                               code='console.log("Hello from JavaScript!"); const result = 2 + 2; console.log("Result:", result);',
                               language='javascript')
    print("JavaScript result:", js_result)
    
    # Multi-language execution
    code_blocks = [
        {'code': 'x = 10', 'language': 'python'},
        {'code': 'y = x * 2; print(f"Result: {y}")', 'language': 'python'},
        {'code': 'console.log("JavaScript can access:", process.env);', 'language': 'javascript'}
    ]
    
    multi_result = executor.execute('execute_multi', code_blocks=code_blocks)
    print("Multi-language result:", multi_result['success'])
    
    # Get supported languages
    languages = executor.execute('get_languages')
    print("Supported languages:", languages['result'])
    
    print("Code execution examples completed!") 