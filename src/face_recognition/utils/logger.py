import logging
import sys
import os

class ColorLogger:
    """A minimal colored logger that prints only the message with color to the console,
    and can optionally log messages to a file (without color).
    """
    
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

    def __init__(self, 
                 level=logging.INFO, 
                 name="color_logger", 
                 format_str="%(message)s",
                 log_file=None):
        """
        :param level: log level (e.g. logging.DEBUG, logging.INFO, etc.)
        :param name: name of the logger
        :param format_str: string format for console logs
        :param log_file: path to an optional file to write logs (no color)
        """
        if self._initialized:
            return

        self._initialized = True
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False

        # Clear existing handlers, if any, before adding new ones
        if self.logger.handlers:
            self.logger.handlers.clear()

        # ---- Console Handler (Color) ----
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = self._create_colored_formatter(format_str)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # ---- Optional File Handler (No Color) ----
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            # You can customize the file log format as desired:
            # e.g., timestamp, level, and the message
            file_format_str = "%(asctime)s - %(levelname)s - %(message)s"
            file_formatter = logging.Formatter(file_format_str)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

    def _create_colored_formatter(self, format_str):
        formatter = logging.Formatter(format_str)
        # Override its "format" method to insert color codes:
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
    def get_logger(level=None, name=None, format_str=None, log_file=None):
        """
        A convenience factory method that ensures we only have one instance.
        """
        logger_instance = ColorLogger(
            level=level if level is not None else logging.INFO,
            name=name if name is not None else "color_logger",
            format_str=format_str if format_str is not None else "%(message)s",
            log_file=log_file
        )
        return logger_instance
