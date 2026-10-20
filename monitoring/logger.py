"""JSON logs with size rotation and age-based archive retention."""
from __future__ import annotations
import json
import logging
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

FIELDS = dict(regime='unknown', probability=0.0, equity=0.0, positions={}, daily_pnl=0.0)


class JsonFormatter(logging.Formatter):
    def format(self, record):
        data = dict(FIELDS, timestamp=datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
                    level=record.levelname, logger=record.name, message=record.getMessage())
        data.update(getattr(record, 'context', {}))
        data.update({k: getattr(record, k) for k in FIELDS if hasattr(record, k)})
        if record.exc_info:
            data['exception'] = self.formatException(record.exc_info)
        return json.dumps(data, default=str)


class RetentionHandler(RotatingFileHandler):
    retention_days = 30

    def __init__(self, filename):
        # No count cap: archives are retained by age, even on busy trading days.
        super().__init__(filename, maxBytes=10*1024*1024, backupCount=0, encoding='utf-8')
        self.prune()

    def prune(self):
        for path in Path(self.baseFilename).parent.glob(Path(self.baseFilename).name + '.*'):
            if path.is_file() and path.stat().st_mtime < time.time()-self.retention_days*86400:
                path.unlink()

    def doRollover(self):
        if self.stream:
            self.stream.close()
            self.stream = None
        source = Path(self.baseFilename)
        if source.exists():
            source.rename(str(source) + '.' + str(time.time_ns()))
        self.prune()
        self.stream = self._open()

    def emit(self, record):
        self.prune()
        super().emit(record)


def setup_logging(level='INFO', log_dir='logs', json_file=True):
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        if getattr(handler, '_monitoring', False):
            root.removeHandler(handler)
            handler.close()
    if not json_file:
        return
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    for category in ('main', 'trades', 'alerts', 'regime'):
        handler = RetentionHandler(Path(log_dir)/f'{category}.log')
        handler._monitoring = True
        handler.setFormatter(JsonFormatter())
        handler.addFilter(lambda r, c=category: c == 'main' or getattr(r, 'category', '') == c
                          or r.name.startswith('regime_trader.' + c))
        root.addHandler(handler)


def get_logger(name):
    return logging.getLogger('regime_trader.' + name)


def log_event(logger, event, **fields):
    logger.info(event, extra={'context': fields, 'category': fields.get('category', '')})
