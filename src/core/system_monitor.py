"""System resource monitoring for memory and CPU usage."""

import psutil
import asyncio
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import structlog

from src.core.config import get_config

logger = structlog.get_logger(__name__)


@dataclass
class SystemMetrics:
    """System resource metrics snapshot."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    memory_total_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    process_count: int
    load_average: Optional[List[float]] = None  # Unix only


@dataclass
class ProcessMetrics:
    """Process-specific resource metrics."""
    timestamp: float
    pid: int
    cpu_percent: float
    memory_percent: float
    memory_rss_mb: float  # Resident Set Size
    memory_vms_mb: float  # Virtual Memory Size
    num_threads: int
    num_fds: int  # File descriptors (Unix only)
    status: str


class SystemMonitor:
    """Monitor system and process resource usage."""
    
    def __init__(self, collection_interval: float = 60.0, history_size: int = 100):
        """Initialize system monitor.
        
        Args:
            collection_interval: Seconds between metric collections
            history_size: Number of historical metrics to keep
        """
        self.collection_interval = collection_interval
        self.history_size = history_size
        
        self._system_history: List[SystemMetrics] = []
        self._process_history: List[ProcessMetrics] = []
        self._monitoring_task: Optional[asyncio.Task] = None
        self._process = psutil.Process()
        self._is_monitoring = False
        
    async def start_monitoring(self) -> None:
        """Start continuous system monitoring."""
        if self._is_monitoring:
            return
            
        self._is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("System monitoring started", interval=self.collection_interval)
        
    async def stop_monitoring(self) -> None:
        """Stop system monitoring."""
        self._is_monitoring = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("System monitoring stopped")
        
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._is_monitoring:
            try:
                # Collect metrics
                system_metrics = await self._collect_system_metrics()
                process_metrics = await self._collect_process_metrics()
                
                # Store in history
                self._add_to_history(self._system_history, system_metrics)
                self._add_to_history(self._process_history, process_metrics)
                
                # Log if resource usage is high
                await self._check_resource_alerts(system_metrics, process_metrics)
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
                await asyncio.sleep(self.collection_interval)
                
    def _add_to_history(self, history: List, metrics) -> None:
        """Add metrics to history with size limit."""
        history.append(metrics)
        if len(history) > self.history_size:
            history.pop(0)
            
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system-wide resource metrics."""
        # CPU usage (1 second average)
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        
        # Disk usage for current directory
        disk = psutil.disk_usage('.')
        
        # Process count
        process_count = len(psutil.pids())
        
        # Load average (Unix only)
        load_avg = None
        try:
            load_avg = list(psutil.getloadavg())
        except AttributeError:
            # Windows doesn't have load average
            pass
            
        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            memory_available_mb=memory.available / (1024 * 1024),
            memory_total_mb=memory.total / (1024 * 1024),
            disk_usage_percent=disk.percent,
            disk_free_gb=disk.free / (1024 * 1024 * 1024),
            process_count=process_count,
            load_average=load_avg
        )
        
    async def _collect_process_metrics(self) -> ProcessMetrics:
        """Collect process-specific resource metrics."""
        try:
            # Refresh process info
            self._process.cpu_percent()  # First call returns 0
            await asyncio.sleep(0.1)  # Short wait for accurate CPU measurement
            
            cpu_percent = self._process.cpu_percent()
            memory_info = self._process.memory_info()
            memory_percent = self._process.memory_percent()
            
            # File descriptors (Unix only)
            num_fds = 0
            try:
                num_fds = self._process.num_fds()
            except AttributeError:
                # Windows doesn't have file descriptors
                pass
                
            return ProcessMetrics(
                timestamp=time.time(),
                pid=self._process.pid,
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_rss_mb=memory_info.rss / (1024 * 1024),
                memory_vms_mb=memory_info.vms / (1024 * 1024),
                num_threads=self._process.num_threads(),
                num_fds=num_fds,
                status=self._process.status()
            )
            
        except psutil.NoSuchProcess:
            # Process might have been restarted
            self._process = psutil.Process()
            return await self._collect_process_metrics()
            
    async def _check_resource_alerts(
        self, 
        system_metrics: SystemMetrics,
        process_metrics: ProcessMetrics
    ) -> None:
        """Check for high resource usage and log alerts."""
        config = get_config()
        
        # System alerts
        if system_metrics.cpu_percent > 80:
            logger.warning(
                "High system CPU usage",
                cpu_percent=system_metrics.cpu_percent
            )
            
        if system_metrics.memory_percent > 85:
            logger.warning(
                "High system memory usage",
                memory_percent=system_metrics.memory_percent,
                available_mb=system_metrics.memory_available_mb
            )
            
        if system_metrics.disk_usage_percent > 90:
            logger.warning(
                "High disk usage",
                disk_percent=system_metrics.disk_usage_percent,
                free_gb=system_metrics.disk_free_gb
            )
            
        # Process alerts
        if process_metrics.cpu_percent > 50:
            logger.warning(
                "High process CPU usage",
                pid=process_metrics.pid,
                cpu_percent=process_metrics.cpu_percent
            )
            
        if process_metrics.memory_percent > 20:
            logger.warning(
                "High process memory usage",
                pid=process_metrics.pid,
                memory_percent=process_metrics.memory_percent,
                memory_rss_mb=process_metrics.memory_rss_mb
            )
            
    def get_current_system_metrics(self) -> Optional[SystemMetrics]:
        """Get the most recent system metrics."""
        return self._system_history[-1] if self._system_history else None
        
    def get_current_process_metrics(self) -> Optional[ProcessMetrics]:
        """Get the most recent process metrics."""
        return self._process_history[-1] if self._process_history else None
        
    def get_system_history(self, limit: int = 10) -> List[SystemMetrics]:
        """Get recent system metrics history."""
        return self._system_history[-limit:] if self._system_history else []
        
    def get_process_history(self, limit: int = 10) -> List[ProcessMetrics]:
        """Get recent process metrics history."""
        return self._process_history[-limit:] if self._process_history else []
        
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get a summary of current resource usage."""
        current_system = self.get_current_system_metrics()
        current_process = self.get_current_process_metrics()
        
        if not current_system or not current_process:
            return {"status": "no_data", "message": "Monitoring not started or no data available"}
            
        # Calculate averages over last 5 minutes
        recent_system = self.get_system_history(5)
        recent_process = self.get_process_history(5)
        
        avg_cpu_system = sum(m.cpu_percent for m in recent_system) / len(recent_system) if recent_system else 0
        avg_cpu_process = sum(m.cpu_percent for m in recent_process) / len(recent_process) if recent_process else 0
        avg_memory_system = sum(m.memory_percent for m in recent_system) / len(recent_system) if recent_system else 0
        avg_memory_process = sum(m.memory_percent for m in recent_process) / len(recent_process) if recent_process else 0
        
        return {
            "timestamp": current_system.timestamp,
            "system": {
                "cpu_percent": current_system.cpu_percent,
                "cpu_percent_avg_5min": round(avg_cpu_system, 2),
                "memory_percent": current_system.memory_percent,
                "memory_percent_avg_5min": round(avg_memory_system, 2),
                "memory_used_mb": round(current_system.memory_used_mb, 2),
                "memory_available_mb": round(current_system.memory_available_mb, 2),
                "disk_usage_percent": current_system.disk_usage_percent,
                "disk_free_gb": round(current_system.disk_free_gb, 2),
                "process_count": current_system.process_count,
                "load_average": current_system.load_average
            },
            "process": {
                "pid": current_process.pid,
                "cpu_percent": current_process.cpu_percent,
                "cpu_percent_avg_5min": round(avg_cpu_process, 2),
                "memory_percent": current_process.memory_percent,
                "memory_percent_avg_5min": round(avg_memory_process, 2),
                "memory_rss_mb": round(current_process.memory_rss_mb, 2),
                "memory_vms_mb": round(current_process.memory_vms_mb, 2),
                "num_threads": current_process.num_threads,
                "num_fds": current_process.num_fds,
                "status": current_process.status
            },
            "status": "healthy" if (
                current_system.cpu_percent < 80 and
                current_system.memory_percent < 85 and
                current_process.cpu_percent < 50 and
                current_process.memory_percent < 20
            ) else "warning"
        }


# Global system monitor instance
_system_monitor: Optional[SystemMonitor] = None


async def get_system_monitor() -> SystemMonitor:
    """Get the global system monitor instance."""
    global _system_monitor
    if _system_monitor is None:
        config = get_config()
        collection_interval = getattr(config.performance, 'monitoring_interval', 60.0)
        _system_monitor = SystemMonitor(collection_interval=collection_interval)
    return _system_monitor


async def start_system_monitoring() -> None:
    """Start global system monitoring."""
    monitor = await get_system_monitor()
    await monitor.start_monitoring()


async def stop_system_monitoring() -> None:
    """Stop global system monitoring."""
    global _system_monitor
    if _system_monitor:
        await _system_monitor.stop_monitoring()


async def get_resource_stats() -> Dict[str, Any]:
    """Get current resource statistics."""
    monitor = await get_system_monitor()
    return monitor.get_resource_summary()