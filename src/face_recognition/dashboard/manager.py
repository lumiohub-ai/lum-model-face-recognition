"""Dashboard manager for the face recognition system.

This module provides a clean interface for managing the FastAPI dashboard,
including starting, stopping, and health checking the dashboard server.
"""

import os
import threading
from typing import Optional
from loguru import logger


class DashboardManager:
    """Manager for the FastAPI dashboard server.

    This class handles the lifecycle of the dashboard server, running it in
    a background thread to avoid blocking the main face recognition pipeline.
    """

    def __init__(self, enabled: bool = True, host: str = "0.0.0.0", port: int = 5001):
        """Initialize the dashboard manager.

        Args:
            enabled: Whether the dashboard should be enabled
            host: Host address for the dashboard server
            port: Port number for the dashboard server
        """
        self.enabled = enabled
        self.host = host
        self.port = port
        self.thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> bool:
        """Start the dashboard server in a background thread.

        Returns:
            True if dashboard started successfully, False otherwise
        """
        if not self.enabled:
            logger.info("Dashboard is disabled in configuration")
            return False

        if self._running:
            logger.warning("Dashboard is already running")
            return True

        try:
            # Import dashboard server
            from .backend import run_server

            # Start in daemon thread so it doesn't block shutdown
            self.thread = threading.Thread(
                target=run_server,
                daemon=True,
                name="DashboardThread"
            )
            self.thread.start()
            self._running = True

            logger.info(f"Dashboard started on http://{self.host}:{self.port}")
            logger.info(f"Dashboard thread: {self.thread.name} (daemon={self.thread.daemon})")

            return True

        except Exception as e:
            logger.error(f"Failed to start dashboard: {e}", exc_info=True)
            logger.warning("Face recognition will continue without dashboard")
            return False

    def stop(self) -> None:
        """Stop the dashboard server.

        Note: Since the dashboard runs in a daemon thread, it will automatically
        stop when the main program exits. This method is primarily for explicit
        shutdown scenarios.
        """
        if not self._running:
            logger.debug("Dashboard is not running, nothing to stop")
            return

        logger.info("Stopping dashboard server...")
        self._running = False

        # Daemon threads will automatically stop when main program exits
        # If we need graceful shutdown, we would implement it here
        if self.thread and self.thread.is_alive():
            logger.debug("Dashboard thread will terminate with main program (daemon thread)")

    def is_running(self) -> bool:
        """Check if the dashboard is currently running.

        Returns:
            True if dashboard is running, False otherwise
        """
        return self._running and (self.thread is not None and self.thread.is_alive())

    def get_url(self) -> Optional[str]:
        """Get the dashboard URL.

        Returns:
            Dashboard URL if running, None otherwise
        """
        if self._running:
            return f"http://{self.host}:{self.port}"
        return None

    @classmethod
    def from_env(cls) -> 'DashboardManager':
        """Create a DashboardManager from environment variables.

        Environment variables:
            - DASHBOARD_ENABLED: Enable/disable dashboard (default: true)
            - DASHBOARD_HOST: Dashboard host (default: 0.0.0.0)
            - DASHBOARD_PORT: Dashboard port (default: 5001)

        Returns:
            Configured DashboardManager instance
        """
        enabled = os.getenv("DASHBOARD_ENABLED", "true").lower() == "true"
        host = os.getenv("DASHBOARD_HOST", "0.0.0.0")
        port = int(os.getenv("DASHBOARD_PORT", "5001"))

        return cls(enabled=enabled, host=host, port=port)


def create_dashboard_manager(
    enabled: Optional[bool] = None,
    host: Optional[str] = None,
    port: Optional[int] = None
) -> DashboardManager:
    """Factory function to create a dashboard manager.

    Args:
        enabled: Whether dashboard should be enabled (defaults to env var)
        host: Dashboard host (defaults to env var)
        port: Dashboard port (defaults to env var)

    Returns:
        Configured DashboardManager instance

    Example:
        >>> manager = create_dashboard_manager(port=8080)
        >>> manager.start()
        >>> print(manager.get_url())
        http://0.0.0.0:8080
    """
    # Start with environment defaults
    manager = DashboardManager.from_env()

    # Override with explicit parameters
    if enabled is not None:
        manager.enabled = enabled
    if host is not None:
        manager.host = host
    if port is not None:
        manager.port = port

    return manager
