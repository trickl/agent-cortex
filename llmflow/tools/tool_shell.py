"""
LLMFlow - A powerful framework for building AI agents based on GAME methodology
(Goals, Actions, Memory, Environment).

Shell Tool - Provides secure shell command execution capabilities with process management, monitoring, and cross-platform support.
"""

import os
import sys
import subprocess
import threading
import time
import signal
import logging
import shlex
import tempfile
import json
import psutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import uuid
import platform
import re

# Import LLMFlow registration decorator
from llmflow.tools.tool_decorator import register_tool

@dataclass
class CommandResult:
    """Result of command execution."""
    command: str
    success: bool
    return_code: int
    stdout: str
    stderr: str
    execution_time: float
    pid: Optional[int]
    start_time: datetime
    end_time: datetime
    working_directory: str
    environment_vars: Dict[str, str]

@dataclass
class ProcessInfo:
    """Information about a running process."""
    pid: int
    command: str
    status: str
    cpu_percent: float
    memory_percent: float
    create_time: datetime
    working_directory: str
    environment: Dict[str, str]

@dataclass
class ShellConfig:
    """Configuration for shell command execution."""
    timeout: int = 30
    max_output_size: int = 1024 * 1024  # 1MB
    working_directory: str = None
    environment_vars: Dict[str, str] = None
    shell: bool = True
    capture_output: bool = True
    log_commands: bool = True
    allowed_commands: List[str] = None
    blocked_commands: List[str] = None
    safe_mode: bool = True

class ShellCommandExecutor:
    """
    Shell command executor with security and monitoring features.
    """
    
    def __init__(self, config: ShellConfig = None):
        self.config = config or ShellConfig()
        self.running_processes = {}
        self.command_history = []
        self.stats = {
            'commands_executed': 0,
            'successful_commands': 0,
            'failed_commands': 0,
            'total_execution_time': 0,
            'processes_started': 0,
            'processes_killed': 0
        }
        
        # Default blocked commands for security
        self.default_blocked_commands = [
            'rm -rf /', 'rm -rf *', 'format', 'del /f /s /q',
            'shutdown', 'reboot', 'halt', 'poweroff',
            'mkfs', 'fdisk', 'parted', 'dd if=', 'sudo rm',
            'chmod 777', 'chown -R', ':(){ :|:& };:',  # Fork bomb
            'wget http', 'curl http', 'nc -l', 'netcat -l',
            'whisper'  # Block direct whisper command execution
        ]
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def _is_command_safe(self, command: str) -> Tuple[bool, str]:
        """Check if command is safe to execute."""
        if not self.config.safe_mode:
            return True, "Safe mode disabled"
        
        # Check blocked commands
        blocked_commands = self.config.blocked_commands or self.default_blocked_commands
        for blocked in blocked_commands:
            if blocked.lower() in command.lower():
                return False, f"Command contains blocked pattern: {blocked}"
        
        # Check allowed commands if specified
        if self.config.allowed_commands:
            command_parts = shlex.split(command)
            if command_parts:
                main_command = command_parts[0]
                if main_command not in self.config.allowed_commands:
                    return False, f"Command not in allowed list: {main_command}"
        
        # Additional security checks
        dangerous_patterns = [
            r'>\s*/dev/',  # Writing to device files
            r'chmod\s+[0-7]{3,4}',  # Chmod with octal permissions
            r'chown\s+.*:.*',  # Changing ownership
            r'sudo\s+.*',  # Sudo commands
            r'su\s+.*',  # Switch user
            r'eval\s+.*',  # Code evaluation
            r'exec\s+.*',  # Code execution
            r'\$\(.*\)',  # Command substitution
            r'`.*`',  # Backticks command substitution
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return False, f"Command contains dangerous pattern: {pattern}"
        
        return True, "Command appears safe"
    
    def execute_command(self, command: str, timeout: int = None, 
                       working_directory: str = None, 
                       environment_vars: Dict[str, str] = None,
                       capture_output: bool = None) -> CommandResult:
        """
        Execute a shell command with comprehensive monitoring.
        
        Args:
            command: Command to execute
            timeout: Execution timeout in seconds
            working_directory: Working directory for command
            environment_vars: Additional environment variables
            capture_output: Whether to capture stdout/stderr
        
        Returns:
            CommandResult with execution details
        """
        start_time = datetime.now()
        
        # Security check
        is_safe, safety_message = self._is_command_safe(command)
        if not is_safe:
            return CommandResult(
                command=command,
                success=False,
                return_code=-1,
                stdout="",
                stderr=f"Security check failed: {safety_message}",
                execution_time=0,
                pid=None,
                start_time=start_time,
                end_time=start_time,
                working_directory=working_directory or os.getcwd(),
                environment_vars=environment_vars or {}
            )
        
        # Prepare execution parameters
        timeout = timeout or self.config.timeout
        working_directory = working_directory or self.config.working_directory or os.getcwd()
        capture_output = capture_output if capture_output is not None else self.config.capture_output
        
        # Prepare environment
        env = os.environ.copy()
        if self.config.environment_vars:
            env.update(self.config.environment_vars)
        if environment_vars:
            env.update(environment_vars)
        
        try:
            # Log command execution
            if self.config.log_commands:
                self.logger.info(f"Executing command: {command}")
            
            # Execute command
            process = subprocess.Popen(
                command,
                shell=self.config.shell,
                stdout=subprocess.PIPE if capture_output else None,
                stderr=subprocess.PIPE if capture_output else None,
                text=True,
                cwd=working_directory,
                env=env,
                preexec_fn=None if platform.system() == 'Windows' else os.setsid
            )
            
            # Monitor process
            self.running_processes[process.pid] = {
                'process': process,
                'command': command,
                'start_time': start_time
            }
            
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                return_code = process.returncode
                
                # Limit output size
                if capture_output and self.config.max_output_size:
                    if len(stdout) > self.config.max_output_size:
                        stdout = stdout[:self.config.max_output_size] + "\n... (output truncated)"
                    if len(stderr) > self.config.max_output_size:
                        stderr = stderr[:self.config.max_output_size] + "\n... (output truncated)"
                
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                return_code = -9
                stderr = f"Command timed out after {timeout} seconds\n" + (stderr or "")
            
            finally:
                # Remove from running processes
                if process.pid in self.running_processes:
                    del self.running_processes[process.pid]
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Create result
            result = CommandResult(
                command=command,
                success=return_code == 0,
                return_code=return_code,
                stdout=stdout or "",
                stderr=stderr or "",
                execution_time=execution_time,
                pid=process.pid,
                start_time=start_time,
                end_time=end_time,
                working_directory=working_directory,
                environment_vars=environment_vars or {}
            )
            
            # Update statistics
            self.stats['commands_executed'] += 1
            self.stats['total_execution_time'] += execution_time
            if result.success:
                self.stats['successful_commands'] += 1
            else:
                self.stats['failed_commands'] += 1
            
            # Add to history
            self.command_history.append(result)
            
            return result
            
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            return CommandResult(
                command=command,
                success=False,
                return_code=-1,
                stdout="",
                stderr=f"Execution error: {str(e)}",
                execution_time=execution_time,
                pid=None,
                start_time=start_time,
                end_time=end_time,
                working_directory=working_directory,
                environment_vars=environment_vars or {}
            )
    
    def execute_background(self, command: str, working_directory: str = None,
                          environment_vars: Dict[str, str] = None) -> Dict[str, Any]:
        """Execute command in background."""
        # Security check
        is_safe, safety_message = self._is_command_safe(command)
        if not is_safe:
            return {
                'status': 'error',
                'message': f'Security check failed: {safety_message}',
                'pid': None
            }
        
        try:
            working_directory = working_directory or self.config.working_directory or os.getcwd()
            
            # Prepare environment
            env = os.environ.copy()
            if self.config.environment_vars:
                env.update(self.config.environment_vars)
            if environment_vars:
                env.update(environment_vars)
            
            process = subprocess.Popen(
                command,
                shell=self.config.shell,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=working_directory,
                env=env,
                preexec_fn=None if platform.system() == 'Windows' else os.setsid
            )
            
            # Store process info
            self.running_processes[process.pid] = {
                'process': process,
                'command': command,
                'start_time': datetime.now()
            }
            
            self.stats['processes_started'] += 1
            
            return {
                'status': 'success',
                'message': f'Process started in background',
                'pid': process.pid,
                'command': command
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to start background process: {str(e)}',
                'pid': None
            }
    
    def get_process_status(self, pid: int) -> Dict[str, Any]:
        """Get status of a running process."""
        try:
            if pid in self.running_processes:
                process_info = self.running_processes[pid]
                process = process_info['process']
                
                # Check if process is still running
                if process.poll() is None:
                    # Process is running
                    try:
                        ps_process = psutil.Process(pid)
                        return {
                            'status': 'running',
                            'pid': pid,
                            'command': process_info['command'],
                            'cpu_percent': ps_process.cpu_percent(),
                            'memory_percent': ps_process.memory_percent(),
                            'start_time': process_info['start_time'].isoformat(),
                            'working_directory': ps_process.cwd() if hasattr(ps_process, 'cwd') else 'unknown'
                        }
                    except psutil.NoSuchProcess:
                        return {
                            'status': 'terminated',
                            'pid': pid,
                            'message': 'Process no longer exists'
                        }
                else:
                    # Process has finished
                    return_code = process.returncode
                    return {
                        'status': 'finished',
                        'pid': pid,
                        'command': process_info['command'],
                        'return_code': return_code,
                        'success': return_code == 0,
                        'start_time': process_info['start_time'].isoformat()
                    }
            else:
                return {
                    'status': 'not_found',
                    'pid': pid,
                    'message': 'Process not found in tracking list'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'pid': pid,
                'message': f'Error checking process status: {str(e)}'
            }
    
    def kill_process(self, pid: int, force: bool = False) -> Dict[str, Any]:
        """Kill a running process."""
        try:
            if pid in self.running_processes:
                process = self.running_processes[pid]['process']
                
                if force:
                    process.kill()
                else:
                    process.terminate()
                
                # Wait for process to terminate
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if not force:
                        process.kill()
                        process.wait(timeout=5)
                
                # Remove from tracking
                del self.running_processes[pid]
                self.stats['processes_killed'] += 1
                
                return {
                    'status': 'success',
                    'message': f'Process {pid} terminated',
                    'pid': pid
                }
            else:
                # Try to kill system process
                try:
                    ps_process = psutil.Process(pid)
                    if force:
                        ps_process.kill()
                    else:
                        ps_process.terminate()
                    
                    return {
                        'status': 'success',
                        'message': f'System process {pid} terminated',
                        'pid': pid
                    }
                except psutil.NoSuchProcess:
                    return {
                        'status': 'error',
                        'message': f'Process {pid} not found',
                        'pid': pid
                    }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error killing process {pid}: {str(e)}',
                'pid': pid
            }
    
    def list_running_processes(self) -> List[Dict[str, Any]]:
        """List all tracked running processes."""
        processes = []
        
        for pid, process_info in self.running_processes.items():
            try:
                status = self.get_process_status(pid)
                processes.append(status)
            except Exception as e:
                processes.append({
                    'status': 'error',
                    'pid': pid,
                    'message': f'Error getting process info: {str(e)}'
                })
        
        return processes
    
    def execute_script(self, script_content: str, script_type: str = 'bash',
                      timeout: int = None, cleanup: bool = True) -> CommandResult:
        """
        Execute a script from content.
        
        Args:
            script_content: Script content to execute
            script_type: Type of script (bash, python, powershell, etc.)
            timeout: Execution timeout
            cleanup: Whether to delete script file after execution
        """
        # Create temporary script file
        script_extensions = {
            'bash': '.sh',
            'sh': '.sh',
            'python': '.py',
            'powershell': '.ps1',
            'batch': '.bat',
            'cmd': '.cmd'
        }
        
        extension = script_extensions.get(script_type.lower(), '.sh')
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix=extension, delete=False) as script_file:
                script_file.write(script_content)
                script_path = script_file.name
            
            # Make script executable on Unix systems
            if platform.system() != 'Windows' and script_type in ['bash', 'sh']:
                os.chmod(script_path, 0o755)
            
            # Determine execution command
            if script_type.lower() == 'python':
                command = f'python "{script_path}"'
            elif script_type.lower() == 'powershell':
                command = f'powershell -ExecutionPolicy Bypass -File "{script_path}"'
            elif script_type.lower() in ['batch', 'cmd']:
                command = f'"{script_path}"'
            else:  # bash/sh
                if platform.system() == 'Windows':
                    command = f'bash "{script_path}"'
                else:
                    command = f'bash "{script_path}"'
            
            # Execute script
            result = self.execute_command(command, timeout=timeout)
            
            # Add script info to result
            result.command = f"Script execution ({script_type}): {script_content[:100]}..."
            
            return result
            
        except Exception as e:
            return CommandResult(
                command=f"Script execution ({script_type})",
                success=False,
                return_code=-1,
                stdout="",
                stderr=f"Script execution error: {str(e)}",
                execution_time=0,
                pid=None,
                start_time=datetime.now(),
                end_time=datetime.now(),
                working_directory=os.getcwd(),
                environment_vars={}
            )
        finally:
            # Cleanup script file
            if cleanup and 'script_path' in locals():
                try:
                    os.unlink(script_path)
                except Exception:
                    pass
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        try:
            return {
                'platform': platform.platform(),
                'system': platform.system(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'python_version': platform.python_version(),
                'cpu_count': os.cpu_count(),
                'current_directory': os.getcwd(),
                'environment_variables': dict(os.environ),
                'path_separator': os.pathsep,
                'line_separator': os.linesep,
                'user': os.getenv('USER') or os.getenv('USERNAME', 'unknown'),
                'home_directory': os.path.expanduser('~'),
                'temp_directory': tempfile.gettempdir()
            }
        except Exception as e:
            return {
                'error': f'Failed to get system info: {str(e)}'
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            'stats': self.stats.copy(),
            'running_processes': len(self.running_processes),
            'command_history_count': len(self.command_history),
            'config': asdict(self.config)
        }

# Initialize global executor
shell_executor = ShellCommandExecutor()

# Registered tool functions
@register_tool(tags=["shell", "system", "command", "execute"])
def execute_shell_command(command: str, timeout: int = 30, 
                         working_directory: str = None,
                         environment_vars: Dict[str, str] = None) -> Dict[str, Any]:
    """
    Execute a shell command with security checks and monitoring.
    
    Args:
        command: Shell command to execute
        timeout: Maximum execution time in seconds
        working_directory: Directory to execute command in
        environment_vars: Additional environment variables
    
    Returns:
        Dictionary with execution results
    """
    try:
        result = shell_executor.execute_command(
            command=command,
            timeout=timeout,
            working_directory=working_directory,
            environment_vars=environment_vars
        )
        
        return {
            'status': 'success' if result.success else 'error',
            'command': result.command,
            'return_code': result.return_code,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'execution_time': result.execution_time,
            'pid': result.pid,
            'working_directory': result.working_directory
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Command execution failed: {str(e)}',
            'command': command
        }

@register_tool(tags=["shell", "system", "background", "process"])
def execute_background_command(command: str, working_directory: str = None,
                              environment_vars: Dict[str, str] = None) -> Dict[str, Any]:
    """
    Execute a command in the background.
    
    Args:
        command: Shell command to execute
        working_directory: Directory to execute command in
        environment_vars: Additional environment variables
    
    Returns:
        Dictionary with process information
    """
    return shell_executor.execute_background(
        command=command,
        working_directory=working_directory,
        environment_vars=environment_vars
    )

@register_tool(tags=["shell", "system", "process", "status"])
def get_process_status(pid: int) -> Dict[str, Any]:
    """
    Get the status of a running process.
    
    Args:
        pid: Process ID to check
    
    Returns:
        Dictionary with process status information
    """
    return shell_executor.get_process_status(pid)

@register_tool(tags=["shell", "system", "process", "kill", "terminate"])
def kill_process(pid: int, force: bool = False) -> Dict[str, Any]:
    """
    Kill a running process.
    
    Args:
        pid: Process ID to kill
        force: Whether to force kill (SIGKILL vs SIGTERM)
    
    Returns:
        Dictionary with operation result
    """
    return shell_executor.kill_process(pid, force)

@register_tool(tags=["shell", "system", "process", "list"])
def list_running_processes() -> List[Dict[str, Any]]:
    """
    List all tracked running processes.
    
    Returns:
        List of process information dictionaries
    """
    return shell_executor.list_running_processes()

@register_tool(tags=["shell", "system", "script", "execute"])
def execute_script(script_content: str, script_type: str = 'bash',
                  timeout: int = 60) -> Dict[str, Any]:
    """
    Execute a script from content.
    
    Args:
        script_content: Script content to execute
        script_type: Type of script (bash, python, powershell, etc.)
        timeout: Maximum execution time in seconds
    
    Returns:
        Dictionary with execution results
    """
    try:
        result = shell_executor.execute_script(
            script_content=script_content,
            script_type=script_type,
            timeout=timeout
        )
        
        return {
            'status': 'success' if result.success else 'error',
            'return_code': result.return_code,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'execution_time': result.execution_time,
            'script_type': script_type
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Script execution failed: {str(e)}',
            'script_type': script_type
        }

@register_tool(tags=["shell", "system", "info", "platform"])
def get_system_info() -> Dict[str, Any]:
    """
    Get comprehensive system information.
    
    Returns:
        Dictionary with system information
    """
    return shell_executor.get_system_info()

@register_tool(tags=["shell", "system", "stats", "monitoring"])
def get_shell_execution_stats() -> Dict[str, Any]:
    """
    Get shell execution statistics and monitoring information.
    
    Returns:
        Dictionary with execution statistics
    """
    return shell_executor.get_stats()

@register_tool(tags=["shell", "system", "directory", "navigation"])
def change_working_directory(path: str) -> Dict[str, Any]:
    """
    Change the working directory for shell operations.
    
    Args:
        path: New working directory path
    
    Returns:
        Dictionary with operation result
    """
    try:
        if not os.path.exists(path):
            return {
                'status': 'error',
                'message': f'Directory does not exist: {path}',
                'current_directory': os.getcwd()
            }
        
        if not os.path.isdir(path):
            return {
                'status': 'error',
                'message': f'Path is not a directory: {path}',
                'current_directory': os.getcwd()
            }
        
        # Update global executor config
        shell_executor.config.working_directory = os.path.abspath(path)
        
        return {
            'status': 'success',
            'message': f'Working directory changed to: {path}',
            'new_directory': os.path.abspath(path),
            'previous_directory': os.getcwd()
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Failed to change directory: {str(e)}',
            'current_directory': os.getcwd()
        }

@register_tool(tags=["shell", "system", "environment", "variables"])
def set_environment_variable(name: str, value: str, persistent: bool = False) -> Dict[str, Any]:
    """
    Set an environment variable for shell operations.
    
    Args:
        name: Environment variable name
        value: Environment variable value
        persistent: Whether to make it persistent (for current session)
    
    Returns:
        Dictionary with operation result
    """
    try:
        # Set in current process
        os.environ[name] = value
        
        # Set in executor config if persistent
        if persistent:
            if not shell_executor.config.environment_vars:
                shell_executor.config.environment_vars = {}
            shell_executor.config.environment_vars[name] = value
        
        return {
            'status': 'success',
            'message': f'Environment variable {name} set to: {value}',
            'variable_name': name,
            'variable_value': value,
            'persistent': persistent
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Failed to set environment variable: {str(e)}',
            'variable_name': name
        }

@register_tool(tags=["shell", "system", "environment", "variables"])
def get_environment_variables() -> Dict[str, Any]:
    """
    Get all environment variables.
    
    Returns:
        Dictionary with all environment variables
    """
    try:
        return {
            'status': 'success',
            'environment_variables': dict(os.environ),
            'count': len(os.environ)
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Failed to get environment variables: {str(e)}'
        }

@register_tool(tags=["shell", "system", "quick", "command"])
def quick_shell_command(command: str) -> str:
    """
    Quick shell command execution returning just the output.
    
    Args:
        command: Shell command to execute
    
    Returns:
        Command output as string
    """
    try:
        result = shell_executor.execute_command(command, timeout=15)
        if result.success:
            return result.stdout.strip()
        else:
            return f"Error: {result.stderr.strip()}"
    except Exception as e:
        return f"Execution failed: {str(e)}"

@register_tool(tags=["shell", "system", "config", "security"])
def configure_shell_security(safe_mode: bool = True, 
                           allowed_commands: List[str] = None,
                           blocked_commands: List[str] = None,
                           timeout: int = 30) -> Dict[str, Any]:
    """
    Configure shell execution security settings.
    
    Args:
        safe_mode: Enable/disable safe mode
        allowed_commands: List of allowed commands (if specified, only these are allowed)
        blocked_commands: List of additional blocked commands
        timeout: Default timeout for commands
    
    Returns:
        Dictionary with configuration result
    """
    try:
        # Update global executor config
        shell_executor.config.safe_mode = safe_mode
        shell_executor.config.timeout = timeout
        
        if allowed_commands is not None:
            shell_executor.config.allowed_commands = allowed_commands
        
        if blocked_commands is not None:
            if shell_executor.config.blocked_commands:
                shell_executor.config.blocked_commands.extend(blocked_commands)
            else:
                shell_executor.config.blocked_commands = blocked_commands
        
        return {
            'status': 'success',
            'message': 'Shell security configuration updated',
            'safe_mode': safe_mode,
            'timeout': timeout,
            'allowed_commands_count': len(allowed_commands) if allowed_commands else 0,
            'blocked_commands_count': len(shell_executor.config.blocked_commands or [])
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Failed to configure security: {str(e)}'
        } 