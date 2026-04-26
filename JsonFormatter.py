import json
import logging
import warnings
from datetime import datetime, timezone
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "ts":      datetime.now(timezone.utc).isoformat(),
            "level":   record.levelname,
            "msg":     record.getMessage(),
        }
        # attach any extra context passed via `extra={}`
        for k, v in record.__dict__.items():
            if k not in {
                "name","msg","args","levelname","levelno","pathname",
                "filename","module","exc_info","exc_text","stack_info",
                "lineno","funcName","created","msecs","relativeCreated",
                "thread","threadName","processName","process","message",
            }:
                base[k] = v
        if record.exc_info:
            base["exc"] = self.formatException(record.exc_info)
        return json.dumps(base, ensure_ascii=False)


def _make_logger(name: str = "ingest") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger

log = _make_logger()