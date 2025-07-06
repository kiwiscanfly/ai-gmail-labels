"""
Centralized logging wrapper for MCP server.

This module ensures all logging output goes to stderr to avoid interfering
with the MCP JSON-RPC protocol on stdout.
"""

import sys
import logging
import structlog
from typing import Any, Dict, Optional
from pathlib import Path


class MCPLogger:
    """
    Centralized logger wrapper that ensures all output goes to stderr.
    
    This prevents logging from interfering with the MCP JSON-RPC protocol
    which requires stdout to contain only protocol messages.
    """
    
    _configured = False
    _loggers: Dict[str, Any] = {}
    
    @classmethod
    def configure(cls, level: str = "INFO", format_type: str = "dev") -> None:
        """Configure logging for MCP server environment."""
        if cls._configured:
            return
            
        # Configure standard library logging to stderr
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            stream=sys.stderr,
            force=True
        )
        
        # Configure structlog to stderr
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
        ]
        
        if format_type == "dev":
            processors.append(structlog.dev.ConsoleRenderer())
        else:
            processors.append(structlog.processors.JSONRenderer())
        
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        # Redirect any remaining stdout logging to stderr
        cls._redirect_logging_to_stderr()
        
        cls._configured = True
    
    @classmethod
    def _redirect_logging_to_stderr(cls) -> None:
        """Ensure all logging handlers use stderr."""
        root_logger = logging.getLogger()
        
        # Remove any existing handlers that might use stdout
        for handler in root_logger.handlers[:]:
            if hasattr(handler, 'stream') and handler.stream == sys.stdout:
                root_logger.removeHandler(handler)
        
        # Ensure we have at least one stderr handler
        if not any(hasattr(h, 'stream') and h.stream == sys.stderr for h in root_logger.handlers):
            stderr_handler = logging.StreamHandler(sys.stderr)
            stderr_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            root_logger.addHandler(stderr_handler)
    
    @classmethod
    def get_logger(cls, name: str) -> Any:
        """Get a logger instance that outputs to stderr."""
        if not cls._configured:
            cls.configure()
        
        if name not in cls._loggers:
            # Use structlog for consistent formatting
            cls._loggers[name] = structlog.get_logger(name)
        
        return cls._loggers[name]
    
    @classmethod
    def silence_noisy_loggers(cls) -> None:
        """Silence or redirect commonly noisy third-party loggers."""
        # Silence specific noisy loggers
        noisy_loggers = [
            'urllib3',
            'httpx',
            'httpcore',
            'googleapiclient.discovery',
            'googleapiclient.discovery_cache',
            'google.auth.transport.requests',
            'google_auth_httplib2',
            'oauth2client',
            'google.auth',
            'google_auth_oauthlib',
        ]
        
        for logger_name in noisy_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.WARNING)
            
            # Ensure handlers use stderr
            for handler in logger.handlers:
                if hasattr(handler, 'stream') and handler.stream == sys.stdout:
                    handler.stream = sys.stderr
        
        # Also redirect any root logger handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            if hasattr(handler, 'stream') and handler.stream == sys.stdout:
                handler.stream = sys.stderr
    
    @classmethod
    def mcp_safe_print(cls, message: str, level: str = "info") -> None:
        """
        Print a message safely without interfering with MCP protocol.
        Always goes to stderr.
        """
        logger = cls.get_logger("mcp_server")
        getattr(logger, level.lower())(message)
    
    @classmethod
    def suppress_all_stdout(cls) -> None:
        """
        Emergency method to suppress all stdout output.
        Use only if other methods don't work.
        """
        # Redirect stdout to stderr temporarily
        original_stdout = sys.stdout
        sys.stdout = sys.stderr
        return original_stdout
    
    @classmethod
    def restore_stdout(cls, original_stdout) -> None:
        """Restore original stdout."""
        if original_stdout:
            sys.stdout = original_stdout


# Convenience functions for easy migration
def get_mcp_logger(name: str) -> Any:
    """Get an MCP-safe logger."""
    return MCPLogger.get_logger(name)


def configure_mcp_logging(level: str = "INFO") -> None:
    """Configure MCP-safe logging."""
    MCPLogger.configure(level=level)
    MCPLogger.silence_noisy_loggers()


def mcp_print(message: str, level: str = "info") -> None:
    """Print safely for MCP environment."""
    MCPLogger.mcp_safe_print(message, level)