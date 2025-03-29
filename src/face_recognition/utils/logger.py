import logging
import sys

class ColorLogger:
    """A minimal colored logger that prints only the message, with color based on level."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[41m',  # Red background
        'RESET': '\033[0m'
    }

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ColorLogger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, level=logging.INFO, name="color_logger", format_str="%(message)s"):
        if self._initialized:
            return

        self._initialized = True
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False

        if self.logger.handlers:
            self.logger.handlers.clear()

        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = self._create_colored_formatter(format_str)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

    def _create_colored_formatter(self, format_str):
        formatter = logging.Formatter(format_str)
        formatter.format = self._add_color_to_format(formatter.format)
        return formatter

    def _add_color_to_format(self, original_format_func):
        def format_with_color(record):
            message = original_format_func(record)
            level = record.levelname
            color = self.COLORS.get(level, "")
            return f"{color}{message}{self.COLORS['RESET']}"
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
        self.logger.setLevel(level)

    @staticmethod
    def get_logger(level=None, name=None, format_str=None):
        logger_instance = ColorLogger()
        if level is not None:
            logger_instance.set_level(level)
        return logger_instance
