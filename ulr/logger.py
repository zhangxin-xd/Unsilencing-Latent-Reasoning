"""
Custom logger module with colored debug output
"""
import logging
import sys
import os

# Try to import colorama for cross-platform color support
try:
    from colorama import Fore, Style, init
    init(autoreset=True)
    USE_COLORAMA = True
except ImportError:
    # Fallback to ANSI codes if colorama is not available
    USE_COLORAMA = False

    class Fore:
        BLUE = '\033[94m'

    class Style:
        RESET_ALL = '\033[0m'


class ColoredFormatter(logging.Formatter):
    """Custom formatter that colors debug messages in blue"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_format = f"{Fore.BLUE}{self._fmt}{Style.RESET_ALL}"

    def format(self, record):
        if record.levelno == logging.DEBUG:
            # Use blue color for debug messages
            original_fmt = self._fmt
            self._fmt = self.debug_format
            result = super().format(record)
            self._fmt = original_fmt
            return result
        return super().format(record)


def _resolve_log_level(default_level):
    """Resolve log level from LOG_LEVEL env var, falling back to default_level.

    Accepts level names like DEBUG/INFO/WARNING/ERROR/CRITICAL or a numeric value.
    """
    env_value = os.getenv("LOG_LEVEL")
    if not env_value:
        return default_level
    candidate = env_value.strip().upper()
    # Try named levels first
    level_from_name = getattr(logging, candidate, None)
    if isinstance(level_from_name, int):
        return level_from_name
    # Fallback: try numeric level
    try:
        return int(candidate)
    except (TypeError, ValueError):
        return default_level


def setup_logger(name="", level=logging.INFO):
    """
    Setup a logger with colored debug output
    
    Args:
        name: logger name
        level: default logging level for handler (overridden by LOG_LEVEL env var if set)
    
    Returns:
        configured logger
    """
    logger = logging.getLogger(name)
    # Set logger level to DEBUG so it can show debug messages
    logger.setLevel(logging.DEBUG)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Allow environment override via LOG_LEVEL
    effective_level = _resolve_log_level(level)

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    # Handler level controls what actually gets displayed
    handler.setLevel(effective_level)

    # Create formatter
    formatter = ColoredFormatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


# Create default logger instance
log = setup_logger()
