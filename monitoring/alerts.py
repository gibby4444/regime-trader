"""Email and webhook alerts for critical events.

Examples: drawdown halt/kill switch, order rejections, broker disconnects, regime
changes. Repeated alerts of the same key are suppressed for
``monitoring.alert_rate_limit_minutes``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class Alert:
    """A single alert message."""

    key: str
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    context: dict[str, Any] = field(default_factory=dict)


class AlertManager:
    """Dispatches alerts to configured channels with per-key rate limiting."""

    def __init__(
        self,
        rate_limit_minutes: int = 15,
        email_config: dict[str, Any] | None = None,
        webhook_url: str | None = None,
        *, clock=None, console=None,
    ) -> None:
        """Configure channels; either may be None to disable it."""
        from rich.console import Console
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.console = console or Console()
        self.rate_limit_minutes = rate_limit_minutes
        self.email_config = email_config
        self.webhook_url = webhook_url
        self._last_sent: dict[str, datetime] = {}

    def should_send(self, alert: Alert) -> bool:
        """False if an alert with the same key was sent within the rate-limit window."""
        previous = self._last_sent.get(alert.key)
        return previous is None or (self.clock()-previous).total_seconds() >= self.rate_limit_minutes*60

    def send(self, alert: Alert) -> bool:
        """Send to all channels if not rate-limited; return True if dispatched."""
        from monitoring.logger import get_logger, log_event
        if not self.should_send(alert):
            return False
        self._last_sent[alert.key] = self.clock()
        for deliver in (self.console_delivery, self.log_delivery, self.send_email, self.send_webhook):
            try:
                deliver(alert)
            except Exception:
                # Transport exceptions can contain secrets; never log their text.
                log_event(get_logger('alerts'), 'Alert delivery failed', event_type=alert.key, **alert.context)
        return True

    def console_delivery(self, alert):
        self.console.print(f'{alert.severity.value.upper()}: {alert.title}: {alert.message}', markup=False)

    def log_delivery(self, alert):
        from monitoring.logger import get_logger, log_event
        log_event(get_logger('alerts'), alert.title, event_type=alert.key,
                  severity=alert.severity.value, detail=alert.message, **alert.context)

    def send_email(self, alert):
        import os
        import smtplib
        from email.message import EmailMessage
        config = self.email_config or {}
        if not config.get('enabled', False):
            return
        message = EmailMessage()
        message['Subject'] = alert.title
        message['From'] = config['sender']
        message['To'] = config['recipient']
        message.set_content(alert.message)
        with smtplib.SMTP(config['host'], config.get('port', 587), timeout=5) as smtp:
            smtp.starttls()
            if config.get('username'):
                smtp.login(config['username'], os.environ[config.get('password_env', 'ALERT_SMTP_PASSWORD')])
            smtp.send_message(message)

    def send_webhook(self, alert):
        import json
        from urllib.request import Request, urlopen
        if not self.webhook_url:
            return
        payload = dict(text=f'{alert.title}: {alert.message}', event_type=alert.key,
                       timestamp=alert.timestamp.isoformat(), context=alert.context)
        request = Request(self.webhook_url, data=json.dumps(payload, default=str).encode(),
                          headers={'Content-Type': 'application/json'}, method='POST')
        with urlopen(request, timeout=5):
            pass
