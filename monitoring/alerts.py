"""Email and webhook alerts for critical events.

Examples: drawdown halt/kill switch, order rejections, broker disconnects, regime
changes. Repeated alerts of the same key are suppressed for
``monitoring.alert_rate_limit_minutes``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
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
    ) -> None:
        """Configure channels; either may be None to disable it."""
        self.rate_limit_minutes = rate_limit_minutes
        self.email_config = email_config
        self.webhook_url = webhook_url
        self._last_sent: dict[str, datetime] = {}

    def should_send(self, alert: Alert) -> bool:
        """False if an alert with the same key was sent within the rate-limit window."""
        raise NotImplementedError

    def send(self, alert: Alert) -> bool:
        """Send to all channels if not rate-limited; return True if dispatched."""
        raise NotImplementedError

    def send_email(self, alert: Alert) -> None:
        """Send via SMTP (STARTTLS) using ``email_config``."""
        raise NotImplementedError

    def send_webhook(self, alert: Alert) -> None:
        """POST a JSON payload to ``webhook_url``."""
        raise NotImplementedError
