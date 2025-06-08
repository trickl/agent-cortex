"""
LLMFlow - A powerful framework for building AI agents based on GAME methodology
(Goals, Actions, Memory, Environment).

System Monitoring Tool - Provides system resource monitoring, process tracking, and performance analysis with alerting capabilities.
"""

import os
import sys
import time
import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import statistics
import platform

# Import LLMFlow registration decorator
from .tool_decorator import register_tool

# Optional imports with fallbacks
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Some monitoring features will be limited.")

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

@dataclass
class SystemMetrics:
    """System metrics snapshot."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used: int
    memory_total: int
    disk_usage_percent: float
    disk_used: int
    disk_total: int
    network_bytes_sent: int
    network_bytes_recv: int
    load_average: List[float]
    boot_time: datetime
    uptime_seconds: float

@dataclass
class ProcessMetrics:
    """Process metrics snapshot."""
    pid: int
    name: str
    cpu_percent: float
    memory_percent: float
    memory_rss: int
    memory_vms: int
    status: str
    create_time: datetime
    num_threads: int
    username: str
    cmdline: List[str]

@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    metric: str
    condition: str  # '>', '<', '>=', '<=', '==', '!='
    threshold: float
    enabled: bool = True
    notification_count: int = 0
    last_triggered: Optional[datetime] = None
    cooldown_minutes: int = 5

@dataclass
class MonitoringConfig:
    """Configuration for system monitoring."""
    collection_interval: int = 5  # seconds
    history_retention: int = 1000  # number of snapshots
    enable_alerts: bool = True
    log_metrics: bool = True
    monitor_processes: bool = True
    monitor_network: bool = True
    alert_rules: List[AlertRule] = field(default_factory=list)

class SystemMonitor:
    """
    Comprehensive system monitoring with metrics collection and alerting.
    """
    
    def __init__(self, config: MonitoringConfig = None):
        self.config = config or MonitoringConfig()
        self.metrics_history = deque(maxlen=self.config.history_retention)
        self.process_history = defaultdict(lambda: deque(maxlen=100))
        self.alert_rules = self.config.alert_rules.copy()
        self.monitoring_active = False
        self.monitor_thread = None
        self.logger = logging.getLogger(__name__)
        
        # Performance counters
        self.last_network_io = None
        self.last_disk_io = None
        
        # Statistics
        self.stats = {
            'monitoring_sessions': 0,
            'alerts_triggered': 0,
            'snapshots_collected': 0,
            'processes_tracked': 0
        }
        
        if not PSUTIL_AVAILABLE:
            self.logger.warning("psutil not available - limited monitoring capabilities")
    
    def get_current_metrics(self) -> SystemMetrics:
        """Get current system metrics snapshot."""
        if not PSUTIL_AVAILABLE:
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used=0,
                memory_total=0,
                disk_usage_percent=0.0,
                disk_used=0,
                disk_total=0,
                network_bytes_sent=0,
                network_bytes_recv=0,
                load_average=[0.0, 0.0, 0.0],
                boot_time=datetime.now(),
                uptime_seconds=0.0
            )
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Network metrics
            network = psutil.net_io_counters()
            
            # Load average (Unix only)
            try:
                load_avg = list(os.getloadavg())
            except (OSError, AttributeError):
                load_avg = [0.0, 0.0, 0.0]
            
            # Boot time and uptime
            boot_time = datetime.fromtimestamp(psutil.boot_time())
            uptime = time.time() - psutil.boot_time()
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used=memory.used,
                memory_total=memory.total,
                disk_usage_percent=disk.percent,
                disk_used=disk.used,
                disk_total=disk.total,
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                load_average=load_avg,
                boot_time=boot_time,
                uptime_seconds=uptime
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            raise
    
    def get_process_metrics(self, limit: int = 10) -> List[ProcessMetrics]:
        """Get metrics for top processes by CPU usage."""
        if not PSUTIL_AVAILABLE:
            return []
        
        try:
            processes = []
            
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent',
                                           'memory_info', 'status', 'create_time', 
                                           'num_threads', 'username', 'cmdline']):
                try:
                    pinfo = proc.info
                    
                    processes.append(ProcessMetrics(
                        pid=pinfo['pid'],
                        name=pinfo['name'] or 'Unknown',
                        cpu_percent=pinfo['cpu_percent'] or 0.0,
                        memory_percent=pinfo['memory_percent'] or 0.0,
                        memory_rss=pinfo['memory_info'].rss if pinfo['memory_info'] else 0,
                        memory_vms=pinfo['memory_info'].vms if pinfo['memory_info'] else 0,
                        status=pinfo['status'] or 'unknown',
                        create_time=datetime.fromtimestamp(pinfo['create_time']) if pinfo['create_time'] else datetime.now(),
                        num_threads=pinfo['num_threads'] or 0,
                        username=pinfo['username'] or 'unknown',
                        cmdline=pinfo['cmdline'] or []
                    ))
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            
            # Sort by CPU usage and return top processes
            processes.sort(key=lambda x: x.cpu_percent, reverse=True)
            return processes[:limit]
            
        except Exception as e:
            self.logger.error(f"Error collecting process metrics: {e}")
            return []
    
    def check_alerts(self, metrics: SystemMetrics) -> List[Dict[str, Any]]:
        """Check alert rules against current metrics."""
        triggered_alerts = []
        
        if not self.config.enable_alerts:
            return triggered_alerts
        
        current_time = datetime.now()
        
        for rule in self.alert_rules:
            if not rule.enabled:
                continue
            
            # Check cooldown
            if (rule.last_triggered and 
                current_time - rule.last_triggered < timedelta(minutes=rule.cooldown_minutes)):
                continue
            
            # Get metric value
            metric_value = getattr(metrics, rule.metric, None)
            if metric_value is None:
                continue
            
            # Check condition
            triggered = False
            if rule.condition == '>':
                triggered = metric_value > rule.threshold
            elif rule.condition == '<':
                triggered = metric_value < rule.threshold
            elif rule.condition == '>=':
                triggered = metric_value >= rule.threshold
            elif rule.condition == '<=':
                triggered = metric_value <= rule.threshold
            elif rule.condition == '==':
                triggered = metric_value == rule.threshold
            elif rule.condition == '!=':
                triggered = metric_value != rule.threshold
            
            if triggered:
                rule.last_triggered = current_time
                rule.notification_count += 1
                
                alert = {
                    'rule_name': rule.name,
                    'metric': rule.metric,
                    'current_value': metric_value,
                    'threshold': rule.threshold,
                    'condition': rule.condition,
                    'timestamp': current_time,
                    'notification_count': rule.notification_count
                }
                
                triggered_alerts.append(alert)
                self.stats['alerts_triggered'] += 1
                
                self.logger.warning(f"Alert triggered: {rule.name} - {rule.metric} {rule.condition} {rule.threshold} (current: {metric_value})")
        
        return triggered_alerts
    
    def start_monitoring(self, interval: int = None) -> Dict[str, Any]:
        """Start continuous monitoring in background thread."""
        if self.monitoring_active:
            return {'status': 'error', 'message': 'Monitoring already active'}
        
        interval = interval or self.config.collection_interval
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    # Collect metrics
                    metrics = self.get_current_metrics()
                    self.metrics_history.append(metrics)
                    self.stats['snapshots_collected'] += 1
                    
                    # Check alerts
                    alerts = self.check_alerts(metrics)
                    
                    # Log metrics if enabled
                    if self.config.log_metrics:
                        self.logger.info(f"System metrics - CPU: {metrics.cpu_percent:.1f}%, "
                                       f"Memory: {metrics.memory_percent:.1f}%, "
                                       f"Disk: {metrics.disk_usage_percent:.1f}%")
                    
                    # Collect process metrics
                    if self.config.monitor_processes:
                        processes = self.get_process_metrics(limit=5)
                        for proc in processes:
                            self.process_history[proc.pid].append(proc)
                        self.stats['processes_tracked'] = len(self.process_history)
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(interval)
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.stats['monitoring_sessions'] += 1
        
        return {
            'status': 'success',
            'message': f'Monitoring started with {interval}s interval',
            'interval': interval
        }
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop continuous monitoring."""
        if not self.monitoring_active:
            return {'status': 'error', 'message': 'Monitoring not active'}
        
        self.monitoring_active = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        return {
            'status': 'success',
            'message': 'Monitoring stopped',
            'snapshots_collected': self.stats['snapshots_collected']
        }
    
    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get summary statistics for recent metrics."""
        if not self.metrics_history:
            return {'status': 'error', 'message': 'No metrics data available'}
        
        # Filter recent metrics
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {'status': 'error', 'message': f'No metrics data for last {hours} hours'}
        
        # Calculate statistics
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        disk_values = [m.disk_usage_percent for m in recent_metrics]
        
        return {
            'status': 'success',
            'period_hours': hours,
            'snapshot_count': len(recent_metrics),
            'cpu_stats': {
                'average': statistics.mean(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values),
                'current': cpu_values[-1] if cpu_values else 0
            },
            'memory_stats': {
                'average': statistics.mean(memory_values),
                'max': max(memory_values),
                'min': min(memory_values),
                'current': memory_values[-1] if memory_values else 0
            },
            'disk_stats': {
                'average': statistics.mean(disk_values),
                'max': max(disk_values),
                'min': min(disk_values),
                'current': disk_values[-1] if disk_values else 0
            },
            'first_snapshot': recent_metrics[0].timestamp.isoformat(),
            'last_snapshot': recent_metrics[-1].timestamp.isoformat()
        }
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get detailed current resource usage."""
        if not PSUTIL_AVAILABLE:
            return {'status': 'error', 'message': 'psutil not available'}
        
        try:
            # CPU information
            cpu_info = {
                'percent': psutil.cpu_percent(interval=1),
                'count_logical': psutil.cpu_count(),
                'count_physical': psutil.cpu_count(logical=False),
                'per_cpu': psutil.cpu_percent(interval=1, percpu=True),
                'times': psutil.cpu_times()._asdict()
            }
            
            # Memory information
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            memory_info = {
                'virtual': {
                    'total': memory.total,
                    'available': memory.available,
                    'used': memory.used,
                    'free': memory.free,
                    'percent': memory.percent
                },
                'swap': {
                    'total': swap.total,
                    'used': swap.used,
                    'free': swap.free,
                    'percent': swap.percent
                }
            }
            
            # Disk information
            disk_info = {}
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_info[partition.device] = {
                        'mountpoint': partition.mountpoint,
                        'fstype': partition.fstype,
                        'total': usage.total,
                        'used': usage.used,
                        'free': usage.free,
                        'percent': (usage.used / usage.total) * 100 if usage.total > 0 else 0
                    }
                except PermissionError:
                    continue
            
            # Network information
            network = psutil.net_io_counters()
            network_info = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv,
                'err_in': network.errin,
                'err_out': network.errout,
                'drop_in': network.dropin,
                'drop_out': network.dropout
            }
            
            return {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'cpu': cpu_info,
                'memory': memory_info,
                'disk': disk_info,
                'network': network_info
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error getting resource usage: {str(e)}'
            }
    
    def get_top_processes(self, limit: int = 10, sort_by: str = 'cpu') -> Dict[str, Any]:
        """Get top processes sorted by specified metric."""
        if not PSUTIL_AVAILABLE:
            return {'status': 'error', 'message': 'psutil not available'}
        
        try:
            processes = self.get_process_metrics(limit=limit * 2)  # Get more for sorting
            
            # Sort by specified metric
            if sort_by == 'cpu':
                processes.sort(key=lambda x: x.cpu_percent, reverse=True)
            elif sort_by == 'memory':
                processes.sort(key=lambda x: x.memory_percent, reverse=True)
            elif sort_by == 'name':
                processes.sort(key=lambda x: x.name.lower())
            else:
                return {'status': 'error', 'message': f'Invalid sort criteria: {sort_by}'}
            
            # Convert to dictionaries
            process_list = []
            for proc in processes[:limit]:
                process_list.append({
                    'pid': proc.pid,
                    'name': proc.name,
                    'cpu_percent': proc.cpu_percent,
                    'memory_percent': proc.memory_percent,
                    'memory_rss_mb': proc.memory_rss / (1024 * 1024),
                    'status': proc.status,
                    'num_threads': proc.num_threads,
                    'username': proc.username,
                    'create_time': proc.create_time.isoformat(),
                    'cmdline': ' '.join(proc.cmdline[:3]) if proc.cmdline else ''
                })
            
            return {
                'status': 'success',
                'processes': process_list,
                'count': len(process_list),
                'sorted_by': sort_by,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error getting top processes: {str(e)}'
            }
    
    def add_alert_rule(self, name: str, metric: str, condition: str, 
                      threshold: float, cooldown_minutes: int = 5) -> Dict[str, Any]:
        """Add a new alert rule."""
        # Validate inputs
        valid_metrics = ['cpu_percent', 'memory_percent', 'disk_usage_percent']
        valid_conditions = ['>', '<', '>=', '<=', '==', '!=']
        
        if metric not in valid_metrics:
            return {
                'status': 'error',
                'message': f'Invalid metric. Valid options: {valid_metrics}'
            }
        
        if condition not in valid_conditions:
            return {
                'status': 'error',
                'message': f'Invalid condition. Valid options: {valid_conditions}'
            }
        
        # Check if rule already exists
        for rule in self.alert_rules:
            if rule.name == name:
                return {
                    'status': 'error',
                    'message': f'Alert rule "{name}" already exists'
                }
        
        # Create new rule
        new_rule = AlertRule(
            name=name,
            metric=metric,
            condition=condition,
            threshold=threshold,
            cooldown_minutes=cooldown_minutes
        )
        
        self.alert_rules.append(new_rule)
        
        return {
            'status': 'success',
            'message': f'Alert rule "{name}" added successfully',
            'rule': {
                'name': name,
                'metric': metric,
                'condition': condition,
                'threshold': threshold,
                'cooldown_minutes': cooldown_minutes
            }
        }
    
    def list_alert_rules(self) -> Dict[str, Any]:
        """List all alert rules."""
        rules_list = []
        
        for rule in self.alert_rules:
            rules_list.append({
                'name': rule.name,
                'metric': rule.metric,
                'condition': rule.condition,
                'threshold': rule.threshold,
                'enabled': rule.enabled,
                'cooldown_minutes': rule.cooldown_minutes,
                'notification_count': rule.notification_count,
                'last_triggered': rule.last_triggered.isoformat() if rule.last_triggered else None
            })
        
        return {
            'status': 'success',
            'alert_rules': rules_list,
            'count': len(rules_list)
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        if not PSUTIL_AVAILABLE:
            return {
                'status': 'limited',
                'message': 'psutil not available - basic info only',
                'platform': platform.platform(),
                'python_version': platform.python_version()
            }
        
        try:
            # Basic system info
            boot_time = datetime.fromtimestamp(psutil.boot_time())
            uptime = datetime.now() - boot_time
            
            info = {
                'status': 'success',
                'platform': {
                    'system': platform.system(),
                    'release': platform.release(),
                    'version': platform.version(),
                    'machine': platform.machine(),
                    'processor': platform.processor(),
                    'platform': platform.platform(),
                    'python_version': platform.python_version()
                },
                'boot_time': boot_time.isoformat(),
                'uptime': {
                    'total_seconds': uptime.total_seconds(),
                    'days': uptime.days,
                    'hours': uptime.seconds // 3600,
                    'minutes': (uptime.seconds % 3600) // 60
                },
                'users': [user._asdict() for user in psutil.users()],
                'cpu': {
                    'physical_cores': psutil.cpu_count(logical=False),
                    'logical_cores': psutil.cpu_count(),
                    'max_frequency': psutil.cpu_freq().max if psutil.cpu_freq() else None,
                    'current_frequency': psutil.cpu_freq().current if psutil.cpu_freq() else None
                }
            }
            
            return info
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error getting system info: {str(e)}'
            }
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        return {
            'status': 'success',
            'monitoring_active': self.monitoring_active,
            'metrics_history_count': len(self.metrics_history),
            'processes_tracked': len(self.process_history),
            'alert_rules_count': len(self.alert_rules),
            'stats': self.stats.copy(),
            'config': {
                'collection_interval': self.config.collection_interval,
                'history_retention': self.config.history_retention,
                'enable_alerts': self.config.enable_alerts,
                'monitor_processes': self.config.monitor_processes,
                'monitor_network': self.config.monitor_network
            }
        }

# Initialize global monitor
system_monitor = SystemMonitor()

# Registered tool functions
@register_tool(tags=["system", "monitoring", "metrics", "current"])
def get_current_system_metrics() -> Dict[str, Any]:
    """
    Get current system metrics including CPU, memory, disk, and network usage.
    
    Returns:
        Dictionary with current system metrics
    """
    try:
        metrics = system_monitor.get_current_metrics()
        return {
            'status': 'success',
            'timestamp': metrics.timestamp.isoformat(),
            'cpu_percent': metrics.cpu_percent,
            'memory_percent': metrics.memory_percent,
            'memory_used_gb': metrics.memory_used / (1024**3),
            'memory_total_gb': metrics.memory_total / (1024**3),
            'disk_usage_percent': metrics.disk_usage_percent,
            'disk_used_gb': metrics.disk_used / (1024**3),
            'disk_total_gb': metrics.disk_total / (1024**3),
            'network_bytes_sent': metrics.network_bytes_sent,
            'network_bytes_recv': metrics.network_bytes_recv,
            'load_average': metrics.load_average,
            'uptime_seconds': metrics.uptime_seconds
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Error getting current metrics: {str(e)}'
        }

@register_tool(tags=["system", "monitoring", "resources", "detailed"])
def get_detailed_resource_usage() -> Dict[str, Any]:
    """
    Get detailed resource usage information for CPU, memory, disk, and network.
    
    Returns:
        Dictionary with detailed resource usage information
    """
    return system_monitor.get_resource_usage()

@register_tool(tags=["system", "monitoring", "processes", "top"])
def get_top_processes(limit: int = 10, sort_by: str = 'cpu') -> Dict[str, Any]:
    """
    Get top processes sorted by CPU usage, memory usage, or name.
    
    Args:
        limit: Maximum number of processes to return
        sort_by: Sort criteria ('cpu', 'memory', or 'name')
    
    Returns:
        Dictionary with top processes information
    """
    return system_monitor.get_top_processes(limit=limit, sort_by=sort_by)

@register_tool(tags=["system", "monitoring", "start", "continuous"])
def start_system_monitoring(interval: int = 5) -> Dict[str, Any]:
    """
    Start continuous system monitoring in background.
    
    Args:
        interval: Collection interval in seconds
    
    Returns:
        Dictionary with monitoring start result
    """
    return system_monitor.start_monitoring(interval=interval)

@register_tool(tags=["system", "monitoring", "stop", "continuous"])
def stop_system_monitoring() -> Dict[str, Any]:
    """
    Stop continuous system monitoring.
    
    Returns:
        Dictionary with monitoring stop result
    """
    return system_monitor.stop_monitoring()

@register_tool(tags=["system", "monitoring", "summary", "statistics"])
def get_monitoring_summary(hours: int = 1) -> Dict[str, Any]:
    """
    Get summary statistics for recent monitoring data.
    
    Args:
        hours: Number of hours to include in summary
    
    Returns:
        Dictionary with monitoring summary statistics
    """
    return system_monitor.get_metrics_summary(hours=hours)

@register_tool(tags=["system", "monitoring", "alerts", "add"])
def add_system_alert(name: str, metric: str, condition: str, 
                    threshold: float, cooldown_minutes: int = 5) -> Dict[str, Any]:
    """
    Add a new system monitoring alert rule.
    
    Args:
        name: Alert rule name
        metric: Metric to monitor ('cpu_percent', 'memory_percent', 'disk_usage_percent')
        condition: Comparison operator ('>', '<', '>=', '<=', '==', '!=')
        threshold: Threshold value for alert
        cooldown_minutes: Minutes to wait before re-triggering alert
    
    Returns:
        Dictionary with alert rule creation result
    """
    return system_monitor.add_alert_rule(
        name=name,
        metric=metric,
        condition=condition,
        threshold=threshold,
        cooldown_minutes=cooldown_minutes
    )

@register_tool(tags=["system", "monitoring", "alerts", "list"])
def list_system_alerts() -> Dict[str, Any]:
    """
    List all configured system monitoring alert rules.
    
    Returns:
        Dictionary with list of alert rules
    """
    return system_monitor.list_alert_rules()

@register_tool(tags=["system", "monitoring", "info", "system"])
def get_comprehensive_system_info() -> Dict[str, Any]:
    """
    Get comprehensive system information including platform, uptime, and hardware details.
    
    Returns:
        Dictionary with comprehensive system information
    """
    return system_monitor.get_system_info()

@register_tool(tags=["system", "monitoring", "stats", "monitoring"])
def get_monitoring_statistics() -> Dict[str, Any]:
    """
    Get monitoring system statistics and configuration.
    
    Returns:
        Dictionary with monitoring statistics
    """
    return system_monitor.get_monitoring_stats()

@register_tool(tags=["system", "monitoring", "health", "check"])
def system_health_check() -> Dict[str, Any]:
    """
    Perform a comprehensive system health check.
    
    Returns:
        Dictionary with system health status and recommendations
    """
    try:
        metrics = system_monitor.get_current_metrics()
        
        # Health assessment
        health_issues = []
        warnings = []
        
        # Check CPU usage
        if metrics.cpu_percent > 90:
            health_issues.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        elif metrics.cpu_percent > 75:
            warnings.append(f"Elevated CPU usage: {metrics.cpu_percent:.1f}%")
        
        # Check memory usage
        if metrics.memory_percent > 90:
            health_issues.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        elif metrics.memory_percent > 80:
            warnings.append(f"Elevated memory usage: {metrics.memory_percent:.1f}%")
        
        # Check disk usage
        if metrics.disk_usage_percent > 95:
            health_issues.append(f"Disk nearly full: {metrics.disk_usage_percent:.1f}%")
        elif metrics.disk_usage_percent > 85:
            warnings.append(f"High disk usage: {metrics.disk_usage_percent:.1f}%")
        
        # Determine overall health
        if health_issues:
            health_status = "critical"
        elif warnings:
            health_status = "warning"
        else:
            health_status = "healthy"
        
        return {
            'status': 'success',
            'health_status': health_status,
            'timestamp': metrics.timestamp.isoformat(),
            'issues': health_issues,
            'warnings': warnings,
            'metrics_summary': {
                'cpu_percent': metrics.cpu_percent,
                'memory_percent': metrics.memory_percent,
                'disk_usage_percent': metrics.disk_usage_percent,
                'uptime_hours': metrics.uptime_seconds / 3600
            },
            'recommendations': [
                "Monitor resource usage regularly",
                "Set up alerts for critical thresholds",
                "Consider system maintenance if issues persist"
            ]
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Error performing health check: {str(e)}'
        }

@register_tool(tags=["system", "monitoring", "performance", "analysis"])
def analyze_system_performance() -> Dict[str, Any]:
    """
    Analyze system performance based on historical data.
    
    Returns:
        Dictionary with performance analysis
    """
    try:
        if len(system_monitor.metrics_history) < 10:
            return {
                'status': 'insufficient_data',
                'message': 'Need at least 10 data points for analysis',
                'current_snapshots': len(system_monitor.metrics_history)
            }
        
        # Get recent metrics
        recent_metrics = list(system_monitor.metrics_history)[-50:]  # Last 50 snapshots
        
        # Calculate trends
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        
        # Simple trend analysis (increasing/decreasing/stable)
        def analyze_trend(values):
            if len(values) < 5:
                return "insufficient_data"
            
            first_half = statistics.mean(values[:len(values)//2])
            second_half = statistics.mean(values[len(values)//2:])
            
            diff_percent = ((second_half - first_half) / first_half) * 100 if first_half > 0 else 0
            
            if diff_percent > 10:
                return "increasing"
            elif diff_percent < -10:
                return "decreasing"
            else:
                return "stable"
        
        cpu_trend = analyze_trend(cpu_values)
        memory_trend = analyze_trend(memory_values)
        
        return {
            'status': 'success',
            'analysis_period': f'{len(recent_metrics)} snapshots',
            'cpu_analysis': {
                'trend': cpu_trend,
                'average': statistics.mean(cpu_values),
                'peak': max(cpu_values),
                'minimum': min(cpu_values),
                'volatility': statistics.stdev(cpu_values) if len(cpu_values) > 1 else 0
            },
            'memory_analysis': {
                'trend': memory_trend,
                'average': statistics.mean(memory_values),
                'peak': max(memory_values),
                'minimum': min(memory_values),
                'volatility': statistics.stdev(memory_values) if len(memory_values) > 1 else 0
            },
            'recommendations': [
                f"CPU trend is {cpu_trend}",
                f"Memory trend is {memory_trend}",
                "Monitor trends over longer periods for better insights"
            ]
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Error analyzing performance: {str(e)}'
        } 