#!/usr/bin/env python3
"""
QNTI Notification System - Critical Events & Alerts
Comprehensive notification system for trading events, risk alerts, and system status
"""

import asyncio
import smtplib
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
import threading
import time
import requests

logger = logging.getLogger(__name__)

class NotificationLevel(Enum):
    """Notification severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class NotificationChannel(Enum):
    """Available notification channels"""
    EMAIL = "email"
    WEBHOOK = "webhook"
    DESKTOP = "desktop"
    SMS = "sms"
    DISCORD = "discord"
    SLACK = "slack"
    TELEGRAM = "telegram"

@dataclass
class NotificationEvent:
    """Notification event data structure"""
    id: str
    title: str
    message: str
    level: NotificationLevel
    category: str
    timestamp: datetime
    data: Dict[str, Any]
    channels: List[NotificationChannel]
    acknowledged: bool = False
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class NotificationConfig:
    """Notification configuration"""
    enabled: bool = True
    channels: Dict[str, Dict] = None
    rules: Dict[str, Dict] = None
    rate_limits: Dict[str, int] = None
    escalation_rules: List[Dict] = None

class QNTINotificationSystem:
    """Advanced notification system for QNTI trading platform"""
    
    def __init__(self, config_file: str = "notification_config.json"):
        self.config_file = config_file
        self.config = self._load_config()
        self.event_queue = []
        self.handlers = {}
        self.rate_limit_tracker = {}
        self.running = False
        self.worker_thread = None
        
        # Initialize handlers
        self._initialize_handlers()
        
        # Start background worker
        self.start_worker()
        
        logger.info("QNTI Notification System initialized")
    
    def _load_config(self) -> NotificationConfig:
        """Load notification configuration"""
        try:
            if Path(self.config_file).exists():
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                return NotificationConfig(**config_data)
            else:
                # Create default config
                default_config = {
                    "enabled": True,
                    "channels": {
                        "email": {
                            "enabled": False,
                            "smtp_server": "smtp.gmail.com",
                            "smtp_port": 587,
                            "username": "",
                            "password": "",
                            "recipients": []
                        },
                        "webhook": {
                            "enabled": False,
                            "url": "",
                            "headers": {}
                        },
                        "discord": {
                            "enabled": False,
                            "webhook_url": ""
                        },
                        "slack": {
                            "enabled": False,
                            "webhook_url": ""
                        }
                    },
                    "rules": {
                        "trading_loss": {
                            "threshold": 500.0,
                            "level": "warning",
                            "channels": ["email", "webhook"]
                        },
                        "connection_lost": {
                            "level": "critical",
                            "channels": ["email", "discord", "webhook"]
                        },
                        "high_drawdown": {
                            "threshold": 0.10,
                            "level": "critical",
                            "channels": ["email", "discord"]
                        },
                        "ea_stopped": {
                            "level": "warning",
                            "channels": ["webhook"]
                        }
                    },
                    "rate_limits": {
                        "email": 10,  # per hour
                        "webhook": 60,  # per hour
                        "discord": 30   # per hour
                    }
                }
                
                with open(self.config_file, 'w') as f:
                    json.dump(default_config, f, indent=2)
                
                return NotificationConfig(**default_config)
        except Exception as e:
            logger.error(f"Error loading notification config: {e}")
            return NotificationConfig()
    
    def _initialize_handlers(self):
        """Initialize notification channel handlers"""
        self.handlers = {
            NotificationChannel.EMAIL: self._send_email,
            NotificationChannel.WEBHOOK: self._send_webhook,
            NotificationChannel.DISCORD: self._send_discord,
            NotificationChannel.SLACK: self._send_slack,
            NotificationChannel.DESKTOP: self._send_desktop
        }
    
    def start_worker(self):
        """Start background notification worker"""
        if not self.running:
            self.running = True
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            logger.info("Notification worker started")
    
    def stop_worker(self):
        """Stop background notification worker"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        logger.info("Notification worker stopped")
    
    def _worker_loop(self):
        """Background worker to process notification queue"""
        while self.running:
            try:
                if self.event_queue:
                    event = self.event_queue.pop(0)
                    self._process_event(event)
                
                time.sleep(1)  # Check every second
            except Exception as e:
                logger.error(f"Error in notification worker: {e}")
                time.sleep(5)
    
    def _process_event(self, event: NotificationEvent):
        """Process a notification event"""
        try:
            for channel in event.channels:
                if self._check_rate_limit(channel):
                    handler = self.handlers.get(channel)
                    if handler:
                        success = handler(event)
                        if success:
                            self._update_rate_limit(channel)
                            logger.info(f"Notification sent via {channel.value}: {event.title}")
                        else:
                            self._handle_failed_notification(event, channel)
                    else:
                        logger.warning(f"No handler for channel: {channel.value}")
                else:
                    logger.warning(f"Rate limit exceeded for channel: {channel.value}")
        except Exception as e:
            logger.error(f"Error processing notification event: {e}")
    
    def _handle_failed_notification(self, event: NotificationEvent, channel: NotificationChannel):
        """Handle failed notification delivery"""
        event.retry_count += 1
        if event.retry_count < event.max_retries:
            # Re-queue for retry
            self.event_queue.append(event)
            logger.warning(f"Notification failed, retrying ({event.retry_count}/{event.max_retries})")
        else:
            logger.error(f"Notification failed permanently: {event.title}")
    
    def _check_rate_limit(self, channel: NotificationChannel) -> bool:
        """Check if channel is within rate limits"""
        if not self.config.rate_limits:
            return True
        
        channel_limit = self.config.rate_limits.get(channel.value, 100)
        current_hour = datetime.now().hour
        key = f"{channel.value}_{current_hour}"
        
        count = self.rate_limit_tracker.get(key, 0)
        return count < channel_limit
    
    def _update_rate_limit(self, channel: NotificationChannel):
        """Update rate limit counter"""
        current_hour = datetime.now().hour
        key = f"{channel.value}_{current_hour}"
        self.rate_limit_tracker[key] = self.rate_limit_tracker.get(key, 0) + 1
        
        # Clean old entries
        keys_to_remove = [k for k in self.rate_limit_tracker.keys() 
                         if not k.endswith(str(current_hour))]
        for key in keys_to_remove:
            del self.rate_limit_tracker[key]
    
    def send_notification(self, title: str, message: str, level: NotificationLevel, 
                         category: str, data: Dict = None, channels: List[str] = None):
        """Send a notification"""
        if not self.config.enabled:
            return
        
        # Generate event ID
        event_id = f"qnti_{int(datetime.now().timestamp())}"
        
        # Determine channels based on rules
        if not channels:
            rule = self.config.rules.get(category, {})
            channels = rule.get('channels', ['webhook'])
        
        # Convert string channels to enum
        channel_enums = []
        for ch in channels:
            try:
                channel_enums.append(NotificationChannel(ch))
            except ValueError:
                logger.warning(f"Unknown notification channel: {ch}")
        
        # Create event
        event = NotificationEvent(
            id=event_id,
            title=title,
            message=message,
            level=level,
            category=category,
            timestamp=datetime.now(),
            data=data or {},
            channels=channel_enums
        )
        
        # Add to queue
        self.event_queue.append(event)
        logger.info(f"Notification queued: {title}")
    
    def _send_email(self, event: NotificationEvent) -> bool:
        """Send email notification"""
        try:
            email_config = self.config.channels.get('email', {})
            if not email_config.get('enabled', False):
                return False
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = email_config['username']
            msg['To'] = ', '.join(email_config['recipients'])
            msg['Subject'] = f"[QNTI {event.level.value.upper()}] {event.title}"
            
            # Email body
            body = f"""
QNTI Trading System Alert

Level: {event.level.value.upper()}
Category: {event.category}
Time: {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

Message:
{event.message}

Additional Data:
{json.dumps(event.data, indent=2)}

---
Quantum Nexus Trading Intelligence
            """.strip()
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
                server.starttls()
                server.login(email_config['username'], email_config['password'])
                server.send_message(msg)
            
            return True
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
            return False
    
    def _send_webhook(self, event: NotificationEvent) -> bool:
        """Send webhook notification"""
        try:
            webhook_config = self.config.channels.get('webhook', {})
            if not webhook_config.get('enabled', False):
                return False
            
            payload = {
                'id': event.id,
                'title': event.title,
                'message': event.message,
                'level': event.level.value,
                'category': event.category,
                'timestamp': event.timestamp.isoformat(),
                'data': event.data
            }
            
            headers = webhook_config.get('headers', {})
            headers['Content-Type'] = 'application/json'
            
            response = requests.post(
                webhook_config['url'],
                json=payload,
                headers=headers,
                timeout=10
            )
            
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error sending webhook notification: {e}")
            return False
    
    def _send_discord(self, event: NotificationEvent) -> bool:
        """Send Discord notification"""
        try:
            discord_config = self.config.channels.get('discord', {})
            if not discord_config.get('enabled', False):
                return False
            
            # Color based on level
            colors = {
                NotificationLevel.INFO: 0x3498db,      # Blue
                NotificationLevel.WARNING: 0xf39c12,  # Orange
                NotificationLevel.CRITICAL: 0xe74c3c, # Red
                NotificationLevel.EMERGENCY: 0x8e44ad # Purple
            }
            
            embed = {
                "title": f"🚨 QNTI Alert: {event.title}",
                "description": event.message,
                "color": colors.get(event.level, 0x95a5a6),
                "fields": [
                    {"name": "Level", "value": event.level.value.upper(), "inline": True},
                    {"name": "Category", "value": event.category, "inline": True},
                    {"name": "Time", "value": event.timestamp.strftime('%Y-%m-%d %H:%M:%S'), "inline": True}
                ],
                "footer": {"text": "Quantum Nexus Trading Intelligence"},
                "timestamp": event.timestamp.isoformat()
            }
            
            # Add data fields if present
            if event.data:
                for key, value in event.data.items():
                    if len(embed["fields"]) < 25:  # Discord limit
                        embed["fields"].append({
                            "name": key.replace('_', ' ').title(),
                            "value": str(value),
                            "inline": True
                        })
            
            payload = {"embeds": [embed]}
            
            response = requests.post(
                discord_config['webhook_url'],
                json=payload,
                timeout=10
            )
            
            return response.status_code == 204
        except Exception as e:
            logger.error(f"Error sending Discord notification: {e}")
            return False
    
    def _send_slack(self, event: NotificationEvent) -> bool:
        """Send Slack notification"""
        try:
            slack_config = self.config.channels.get('slack', {})
            if not slack_config.get('enabled', False):
                return False
            
            # Emoji based on level
            emojis = {
                NotificationLevel.INFO: "ℹ️",
                NotificationLevel.WARNING: "⚠️",
                NotificationLevel.CRITICAL: "🚨",
                NotificationLevel.EMERGENCY: "🆘"
            }
            
            payload = {
                "text": f"{emojis.get(event.level, '📢')} *QNTI Alert: {event.title}*",
                "attachments": [
                    {
                        "color": "warning" if event.level == NotificationLevel.WARNING else "danger",
                        "fields": [
                            {"title": "Message", "value": event.message, "short": False},
                            {"title": "Level", "value": event.level.value.upper(), "short": True},
                            {"title": "Category", "value": event.category, "short": True},
                            {"title": "Time", "value": event.timestamp.strftime('%Y-%m-%d %H:%M:%S'), "short": True}
                        ],
                        "footer": "Quantum Nexus Trading Intelligence",
                        "ts": int(event.timestamp.timestamp())
                    }
                ]
            }
            
            response = requests.post(
                slack_config['webhook_url'],
                json=payload,
                timeout=10
            )
            
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
            return False
    
    def _send_desktop(self, event: NotificationEvent) -> bool:
        """Send desktop notification (placeholder)"""
        try:
            # This would require platform-specific implementation
            # For now, just log the notification
            logger.info(f"Desktop notification: {event.title} - {event.message}")
            return True
        except Exception as e:
            logger.error(f"Error sending desktop notification: {e}")
            return False
    
    # Convenience methods for common trading events
    def notify_trade_opened(self, symbol: str, lot_size: float, trade_type: str, price: float):
        """Notify when a trade is opened"""
        self.send_notification(
            title="Trade Opened",
            message=f"New {trade_type} trade opened for {symbol}",
            level=NotificationLevel.INFO,
            category="trade_opened",
            data={
                "symbol": symbol,
                "lot_size": lot_size,
                "type": trade_type,
                "price": price
            }
        )
    
    def notify_trade_closed(self, symbol: str, profit: float, trade_type: str):
        """Notify when a trade is closed"""
        level = NotificationLevel.WARNING if profit < 0 else NotificationLevel.INFO
        self.send_notification(
            title="Trade Closed",
            message=f"{trade_type} trade closed for {symbol} with {profit:+.2f} profit",
            level=level,
            category="trade_closed",
            data={
                "symbol": symbol,
                "profit": profit,
                "type": trade_type
            }
        )
    
    def notify_high_loss(self, current_loss: float, threshold: float):
        """Notify when losses exceed threshold"""
        self.send_notification(
            title="High Loss Alert",
            message=f"Current loss ${current_loss:.2f} exceeds threshold ${threshold:.2f}",
            level=NotificationLevel.CRITICAL,
            category="trading_loss",
            data={
                "current_loss": current_loss,
                "threshold": threshold,
                "percentage": (current_loss / threshold) * 100
            }
        )
    
    def notify_connection_status(self, status: str, details: str = ""):
        """Notify about connection status changes"""
        level = NotificationLevel.CRITICAL if status == "disconnected" else NotificationLevel.INFO
        self.send_notification(
            title=f"MT5 Connection {status.title()}",
            message=f"MT5 connection status changed to {status}. {details}",
            level=level,
            category="connection_lost" if status == "disconnected" else "connection_restored",
            data={"status": status, "details": details}
        )
    
    def notify_ea_status(self, ea_name: str, status: str, reason: str = ""):
        """Notify about EA status changes"""
        level = NotificationLevel.WARNING if status == "stopped" else NotificationLevel.INFO
        self.send_notification(
            title=f"EA {status.title()}",
            message=f"Expert Advisor '{ea_name}' is now {status}. {reason}",
            level=level,
            category="ea_stopped" if status == "stopped" else "ea_started",
            data={"ea_name": ea_name, "status": status, "reason": reason}
        )
    
    def notify_high_drawdown(self, current_dd: float, threshold: float):
        """Notify about high drawdown"""
        self.send_notification(
            title="High Drawdown Alert",
            message=f"Account drawdown {current_dd:.1%} exceeds threshold {threshold:.1%}",
            level=NotificationLevel.CRITICAL,
            category="high_drawdown",
            data={
                "current_drawdown": current_dd,
                "threshold": threshold,
                "drawdown_percentage": current_dd * 100
            }
        )
    
    def get_status(self) -> Dict:
        """Get notification system status"""
        return {
            "enabled": self.config.enabled,
            "running": self.running,
            "queue_length": len(self.event_queue),
            "configured_channels": list(self.config.channels.keys()) if self.config.channels else [],
            "rate_limits": self.rate_limit_tracker
        }

# Example usage and testing
if __name__ == "__main__":
    # Create notification system
    notifier = QNTINotificationSystem()
    
    # Test notifications
    notifier.notify_trade_opened("EURUSD", 0.1, "BUY", 1.0845)
    notifier.notify_high_loss(750.0, 500.0)
    notifier.notify_connection_status("disconnected", "Network error")
    
    # Wait a bit for processing
    time.sleep(3)
    
    # Show status
    print(json.dumps(notifier.get_status(), indent=2))
    
    # Stop system
    notifier.stop_worker()