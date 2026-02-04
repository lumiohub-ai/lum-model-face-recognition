"""GPU memory monitoring utilities."""

import torch
from loguru import logger
from typing import Optional


class GPUMemoryMonitor:
    """Monitor GPU memory usage and log statistics.
    
    This class helps track GPU memory allocation and provides warnings
    when memory usage is high, helping to identify memory leaks early.
    """
    
    def __init__(self, device: str = 'cuda:0', warning_threshold_mb: float = 2000.0):
        """Initialize GPU memory monitor.
        
        Args:
            device: CUDA device to monitor (e.g., 'cuda:0')
            warning_threshold_mb: Memory threshold in MB for warnings
        """
        self.device = device
        self.enabled = torch.cuda.is_available()
        self.peak_memory_mb = 0.0
        self.warning_threshold_mb = warning_threshold_mb
        self.last_log_time = 0.0
        
        if self.enabled:
            logger.info(f"GPU Memory Monitor initialized: device={device}, threshold={warning_threshold_mb}MB")
        else:
            logger.info("GPU Memory Monitor: CUDA not available, monitoring disabled")
    
    def get_memory_stats(self) -> dict:
        """Get current GPU memory statistics.
        
        Returns:
            Dictionary with memory stats (allocated, reserved, peak)
        """
        if not self.enabled:
            return {
                'allocated_mb': 0.0,
                'reserved_mb': 0.0,
                'peak_mb': 0.0,
                'enabled': False
            }
        
        allocated = torch.cuda.memory_allocated(self.device) / 1024**2  # MB
        reserved = torch.cuda.memory_reserved(self.device) / 1024**2    # MB
        
        if allocated > self.peak_memory_mb:
            self.peak_memory_mb = allocated
        
        return {
            'allocated_mb': allocated,
            'reserved_mb': reserved,
            'peak_mb': self.peak_memory_mb,
            'enabled': True
        }
    
    def log_memory_stats(self, context: str = "", force: bool = False):
        """Log current GPU memory usage.
        
        Args:
            context: Optional context string for logging
            force: Force logging even if recently logged
        """
        if not self.enabled:
            return
        
        import time
        current_time = time.time()
        
        # Throttle logging to once per 10 seconds unless forced
        if not force and (current_time - self.last_log_time) < 10.0:
            return
        
        self.last_log_time = current_time
        stats = self.get_memory_stats()
        
        context_str = f" [{context}]" if context else ""
        logger.debug(
            f"GPU Memory{context_str}: "
            f"allocated={stats['allocated_mb']:.1f}MB, "
            f"reserved={stats['reserved_mb']:.1f}MB, "
            f"peak={stats['peak_mb']:.1f}MB"
        )
        
        # Warning if memory usage is high
        if stats['allocated_mb'] > self.warning_threshold_mb:
            logger.warning(
                f"⚠️  High GPU memory usage: {stats['allocated_mb']:.1f}MB allocated "
                f"(threshold: {self.warning_threshold_mb}MB)"
            )
    
    def reset_peak_memory(self):
        """Reset peak memory tracker.
        
        Call this periodically (e.g., every hour) to track peak memory
        in recent time windows rather than absolute peak.
        """
        if self.enabled:
            torch.cuda.reset_peak_memory_stats(self.device)
            self.peak_memory_mb = 0.0
            logger.debug("GPU peak memory stats reset")
    
    def clear_cache(self):
        """Clear CUDA cache to free fragmented memory.
        
        This is a lightweight operation that helps reduce memory fragmentation.
        """
        if self.enabled:
            torch.cuda.empty_cache()
    
    def get_memory_summary(self) -> str:
        """Get formatted memory summary string.
        
        Returns:
            Human-readable memory summary
        """
        if not self.enabled:
            return "GPU monitoring disabled (CUDA not available)"
        
        stats = self.get_memory_stats()
        return (
            f"GPU Memory: {stats['allocated_mb']:.1f}MB allocated, "
            f"{stats['reserved_mb']:.1f}MB reserved, "
            f"{stats['peak_mb']:.1f}MB peak"
        )
