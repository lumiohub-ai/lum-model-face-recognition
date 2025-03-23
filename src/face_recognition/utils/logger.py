import logging
import sys

class ColorLogger:
    """A simple, colored logger that can be instantiated once and used anywhere."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[41m',  # Red background
        'RESET': '\033[0m'       # Reset
    }
    
    # Singleton instance
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Ensure only one instance of the logger exists."""
        if cls._instance is None:
            cls._instance = super(ColorLogger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, level=logging.INFO, name="color_logger", 
                format_str="%(asctime)s - %(levelname)s - %(message)s"):
        """Initialize the logger if it hasn't been initialized yet."""
        if self._initialized:
            return
            
        self._initialized = True
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False
        
        # Clear any existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Create console handler with formatter
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = self._create_colored_formatter(format_str)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def _create_colored_formatter(self, format_str):
        """Create a formatter that adds colors to messages."""
        formatter = logging.Formatter(format_str)
        formatter.format = self._add_color_to_format(formatter.format)
        return formatter
    
    def _add_color_to_format(self, original_format_func):
        """Wrap the formatter's format function to add colors."""
        def format_with_color(record):
            log_message = original_format_func(record)
            levelname = record.levelname
            if levelname in self.COLORS:
                return f"{self.COLORS[levelname]}{log_message}{self.COLORS['RESET']}"
            return log_message
        return format_with_color
    
    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)
    
    def set_level(self, level):
        """Change the logging level."""
        self.logger.setLevel(level)
    
    @staticmethod
    def get_logger(level=None, name=None, format_str=None):
        """Static method to get the logger instance with optional reconfiguration."""
        logger_instance = ColorLogger()
        
        if level is not None:
            logger_instance.set_level(level)
            
        return logger_instance