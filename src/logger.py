
import logging
import sys
import json
from typing import Any

class JSONFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings after parsing the LogRecord.
    """

    def __init__(self, fmt=None, datefmt=None, style='%'):
        super().__init__(fmt, datefmt, style)

    def format(self, record: logging.LogRecord) -> str:
        record_dict = record.__dict__.copy()
        
        # Extract relevant fields
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "lineNo": record.lineno,
        }

        # Add extra fields if available (structured logging args)
        if hasattr(record, "props") and isinstance(record.props, dict):
            log_record.update(record.props)

        # Handle exception info
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record)


def setup_logger(name: str = "voice_agent", level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a logger with JSON formatting.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding multiple handlers if setup is called multiple times
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    
    return logger

# Create a default logger instance
logger = setup_logger()
