#!/usr/bin/env python3
"""
QNTI Audit Logger - Comprehensive Security Audit Logging System
Advanced audit logging with multiple storage backends and real-time monitoring
"""

import json
import logging
import hashlib
import time
import uuid
import threading
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from functools import wraps
from flask import request, g
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import os
import gzip
import shutil
from pathlib import Path
import ipaddress
from user_agents import parse

logger = logging.getLogger('QNTI_AUDIT')

class AuditLevel(Enum):
    """Audit log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    SECURITY = "SECURITY"

class AuditAction(Enum):
    """Audit action types"""
    # Authentication actions
    LOGIN_SUCCESS = "LOGIN_SUCCESS"
    LOGIN_FAILED = "LOGIN_FAILED"
    LOGOUT = "LOGOUT"
    PASSWORD_CHANGE = "PASSWORD_CHANGE"
    MFA_ENABLE = "MFA_ENABLE"
    MFA_DISABLE = "MFA_DISABLE"
    
    # Authorization actions
    ACCESS_GRANTED = "ACCESS_GRANTED"
    ACCESS_DENIED = "ACCESS_DENIED"
    PERMISSION_CHANGE = "PERMISSION_CHANGE"
    ROLE_CHANGE = "ROLE_CHANGE"
    
    # API actions
    API_REQUEST = "API_REQUEST"
    API_RESPONSE = "API_RESPONSE"
    API_ERROR = "API_ERROR"
    API_KEY_CREATE = "API_KEY_CREATE"
    API_KEY_DELETE = "API_KEY_DELETE"
    API_KEY_USE = "API_KEY_USE"
    
    # Trading actions
    TRADE_EXECUTE = "TRADE_EXECUTE"
    TRADE_MODIFY = "TRADE_MODIFY"
    TRADE_CLOSE = "TRADE_CLOSE"
    ORDER_PLACE = "ORDER_PLACE"
    ORDER_CANCEL = "ORDER_CANCEL"
    
    # Account actions
    ACCOUNT_ACCESS = "ACCOUNT_ACCESS"
    ACCOUNT_MODIFY = "ACCOUNT_MODIFY"
    BALANCE_CHANGE = "BALANCE_CHANGE"
    
    # System actions
    SYSTEM_START = "SYSTEM_START"
    SYSTEM_STOP = "SYSTEM_STOP"
    CONFIG_CHANGE = "CONFIG_CHANGE"
    DATA_EXPORT = "DATA_EXPORT"
    DATA_IMPORT = "DATA_IMPORT"
    
    # Security actions
    SECURITY_VIOLATION = "SECURITY_VIOLATION"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    ENCRYPTION_ERROR = "ENCRYPTION_ERROR"
    SUSPICIOUS_ACTIVITY = "SUSPICIOUS_ACTIVITY"

class AuditResource(Enum):
    """Audit resource types"""
    USER = "USER"
    SESSION = "SESSION"
    API_KEY = "API_KEY"
    TRADE = "TRADE"
    ORDER = "ORDER"
    ACCOUNT = "ACCOUNT"
    SYSTEM = "SYSTEM"
    CONFIG = "CONFIG"
    DATA = "DATA"
    SECURITY = "SECURITY"

@dataclass
class AuditEvent:
    """Audit event data structure"""
    id: str
    timestamp: datetime
    level: AuditLevel
    action: AuditAction
    resource: AuditResource
    user_id: Optional[str] = None
    username: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    status_code: Optional[int] = None
    resource_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    success: bool = True
    duration: Optional[float] = None
    request_size: Optional[int] = None
    response_size: Optional[int] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)

class AuditStorage:
    """Base class for audit storage backends"""
    
    def store_event(self, event: AuditEvent):
        """Store audit event"""
        raise NotImplementedError
    
    def query_events(self, filters: Dict[str, Any]) -> List[AuditEvent]:
        """Query audit events"""
        raise NotImplementedError
    
    def cleanup_old_events(self, retention_days: int):
        """Clean up old audit events"""
        raise NotImplementedError

class SQLiteAuditStorage(AuditStorage):
    """SQLite-based audit storage"""
    
    def __init__(self, db_path: str = "qnti_data/audit.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_events (
                    id TEXT PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    level TEXT NOT NULL,
                    action TEXT NOT NULL,
                    resource TEXT NOT NULL,
                    user_id TEXT,
                    username TEXT,
                    session_id TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    endpoint TEXT,
                    method TEXT,
                    status_code INTEGER,
                    resource_id TEXT,
                    details TEXT,
                    success BOOLEAN NOT NULL,
                    duration REAL,
                    request_size INTEGER,
                    response_size INTEGER,
                    error_message TEXT
                )
            ''')
            
            # Create indexes for better query performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON audit_events(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_action ON audit_events(action)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ip_address ON audit_events(ip_address)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_level ON audit_events(level)')
            
            conn.commit()
    
    def store_event(self, event: AuditEvent):
        """Store audit event in SQLite"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO audit_events (
                    id, timestamp, level, action, resource, user_id, username, session_id,
                    ip_address, user_agent, endpoint, method, status_code, resource_id,
                    details, success, duration, request_size, response_size, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.id, event.timestamp, event.level.value, event.action.value,
                event.resource.value, event.user_id, event.username, event.session_id,
                event.ip_address, event.user_agent, event.endpoint, event.method,
                event.status_code, event.resource_id, json.dumps(event.details) if event.details else None,
                event.success, event.duration, event.request_size, event.response_size,
                event.error_message
            ))
            conn.commit()
    
    def query_events(self, filters: Dict[str, Any]) -> List[AuditEvent]:
        """Query audit events from SQLite"""
        query = "SELECT * FROM audit_events WHERE 1=1"
        params = []
        
        if filters.get('start_time'):
            query += " AND timestamp >= ?"
            params.append(filters['start_time'])
        
        if filters.get('end_time'):
            query += " AND timestamp <= ?"
            params.append(filters['end_time'])
        
        if filters.get('user_id'):
            query += " AND user_id = ?"
            params.append(filters['user_id'])
        
        if filters.get('action'):
            query += " AND action = ?"
            params.append(filters['action'])
        
        if filters.get('level'):
            query += " AND level = ?"
            params.append(filters['level'])
        
        if filters.get('ip_address'):
            query += " AND ip_address = ?"
            params.append(filters['ip_address'])
        
        query += " ORDER BY timestamp DESC"
        
        if filters.get('limit'):
            query += " LIMIT ?"
            params.append(filters['limit'])
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            events = []
            for row in rows:
                event = AuditEvent(
                    id=row[0],
                    timestamp=datetime.fromisoformat(row[1]),
                    level=AuditLevel(row[2]),
                    action=AuditAction(row[3]),
                    resource=AuditResource(row[4]),
                    user_id=row[5],
                    username=row[6],
                    session_id=row[7],
                    ip_address=row[8],
                    user_agent=row[9],
                    endpoint=row[10],
                    method=row[11],
                    status_code=row[12],
                    resource_id=row[13],
                    details=json.loads(row[14]) if row[14] else None,
                    success=bool(row[15]),
                    duration=row[16],
                    request_size=row[17],
                    response_size=row[18],
                    error_message=row[19]
                )
                events.append(event)
            
            return events
    
    def cleanup_old_events(self, retention_days: int):
        """Clean up old audit events"""
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM audit_events WHERE timestamp < ?', (cutoff_date,))
            deleted_count = cursor.rowcount
            conn.commit()
            
            logger.info(f"Cleaned up {deleted_count} old audit events")

class FileAuditStorage(AuditStorage):
    """File-based audit storage with log rotation"""
    
    def __init__(self, log_dir: str = "logs/audit", max_file_size: int = 10*1024*1024):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_file_size = max_file_size
        self.current_file = None
        self.file_lock = threading.Lock()
    
    def _get_current_log_file(self) -> Path:
        """Get current log file path"""
        today = datetime.now().strftime('%Y-%m-%d')
        return self.log_dir / f"audit_{today}.log"
    
    def _rotate_log_file(self, log_file: Path):
        """Rotate log file if it's too large"""
        if log_file.exists() and log_file.stat().st_size > self.max_file_size:
            # Compress and archive old file
            timestamp = datetime.now().strftime('%H%M%S')
            archived_file = log_file.with_suffix(f'.{timestamp}.log.gz')
            
            with open(log_file, 'rb') as f_in:
                with gzip.open(archived_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove original file
            log_file.unlink()
            
            logger.info(f"Rotated audit log file to {archived_file}")
    
    def store_event(self, event: AuditEvent):
        """Store audit event in file"""
        with self.file_lock:
            log_file = self._get_current_log_file()
            self._rotate_log_file(log_file)
            
            log_entry = {
                'id': event.id,
                'timestamp': event.timestamp.isoformat(),
                'level': event.level.value,
                'action': event.action.value,
                'resource': event.resource.value,
                'user_id': event.user_id,
                'username': event.username,
                'session_id': event.session_id,
                'ip_address': event.ip_address,
                'user_agent': event.user_agent,
                'endpoint': event.endpoint,
                'method': event.method,
                'status_code': event.status_code,
                'resource_id': event.resource_id,
                'details': event.details,
                'success': event.success,
                'duration': event.duration,
                'request_size': event.request_size,
                'response_size': event.response_size,
                'error_message': event.error_message
            }
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
    
    def query_events(self, filters: Dict[str, Any]) -> List[AuditEvent]:
        """Query audit events from files"""
        events = []
        
        # Get list of log files to search
        log_files = list(self.log_dir.glob('audit_*.log'))
        log_files.extend(self.log_dir.glob('audit_*.log.gz'))
        
        for log_file in sorted(log_files):
            if log_file.suffix == '.gz':
                # Read compressed file
                with gzip.open(log_file, 'rt', encoding='utf-8') as f:
                    lines = f.readlines()
            else:
                # Read regular file
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            
            for line in lines:
                try:
                    data = json.loads(line.strip())
                    event = AuditEvent(
                        id=data['id'],
                        timestamp=datetime.fromisoformat(data['timestamp']),
                        level=AuditLevel(data['level']),
                        action=AuditAction(data['action']),
                        resource=AuditResource(data['resource']),
                        user_id=data.get('user_id'),
                        username=data.get('username'),
                        session_id=data.get('session_id'),
                        ip_address=data.get('ip_address'),
                        user_agent=data.get('user_agent'),
                        endpoint=data.get('endpoint'),
                        method=data.get('method'),
                        status_code=data.get('status_code'),
                        resource_id=data.get('resource_id'),
                        details=data.get('details'),
                        success=data.get('success', True),
                        duration=data.get('duration'),
                        request_size=data.get('request_size'),
                        response_size=data.get('response_size'),
                        error_message=data.get('error_message')
                    )
                    
                    # Apply filters
                    if self._matches_filters(event, filters):
                        events.append(event)
                
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.warning(f"Invalid audit log entry: {e}")
        
        # Sort by timestamp descending
        events.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Apply limit
        if filters.get('limit'):
            events = events[:filters['limit']]
        
        return events
    
    def _matches_filters(self, event: AuditEvent, filters: Dict[str, Any]) -> bool:
        """Check if event matches filters"""
        if filters.get('start_time') and event.timestamp < filters['start_time']:
            return False
        
        if filters.get('end_time') and event.timestamp > filters['end_time']:
            return False
        
        if filters.get('user_id') and event.user_id != filters['user_id']:
            return False
        
        if filters.get('action') and event.action.value != filters['action']:
            return False
        
        if filters.get('level') and event.level.value != filters['level']:
            return False
        
        if filters.get('ip_address') and event.ip_address != filters['ip_address']:
            return False
        
        return True
    
    def cleanup_old_events(self, retention_days: int):
        """Clean up old audit files"""
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        for log_file in self.log_dir.glob('audit_*.log*'):
            try:
                # Extract date from filename
                date_str = log_file.stem.split('_')[1]
                if '.' in date_str:
                    date_str = date_str.split('.')[0]
                
                file_date = datetime.strptime(date_str, '%Y-%m-%d')
                
                if file_date < cutoff_date:
                    log_file.unlink()
                    logger.info(f"Deleted old audit log file: {log_file}")
            
            except (ValueError, IndexError) as e:
                logger.warning(f"Could not parse date from file {log_file}: {e}")

class QNTIAuditLogger:
    """Main audit logging system"""
    
    def __init__(self, storage_backends: List[AuditStorage] = None):
        self.storage_backends = storage_backends or [
            SQLiteAuditStorage(),
            FileAuditStorage()
        ]
        self.event_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._process_events, daemon=True)
        self.worker_thread.start()
        
        # IP geolocation cache
        self.ip_cache = {}
        self.ip_cache_lock = threading.Lock()
        
        # Suspicious activity detection
        self.suspicious_ips = set()
        self.failed_login_attempts = {}
        self.rate_limit_violations = {}
    
    def _process_events(self):
        """Process audit events in background thread"""
        while True:
            try:
                event = self.event_queue.get(timeout=1)
                if event is None:
                    break
                
                # Store event in all backends
                for backend in self.storage_backends:
                    try:
                        backend.store_event(event)
                    except Exception as e:
                        logger.error(f"Error storing audit event in {backend.__class__.__name__}: {e}")
                
                # Check for suspicious activity
                self._check_suspicious_activity(event)
                
                self.event_queue.task_done()
            
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing audit event: {e}")
    
    def _check_suspicious_activity(self, event: AuditEvent):
        """Check for suspicious activity patterns"""
        if not event.ip_address:
            return
        
        current_time = datetime.now()
        
        # Check for failed login attempts
        if event.action == AuditAction.LOGIN_FAILED:
            if event.ip_address not in self.failed_login_attempts:
                self.failed_login_attempts[event.ip_address] = []
            
            self.failed_login_attempts[event.ip_address].append(current_time)
            
            # Remove old attempts (older than 1 hour)
            cutoff_time = current_time - timedelta(hours=1)
            self.failed_login_attempts[event.ip_address] = [
                attempt for attempt in self.failed_login_attempts[event.ip_address]
                if attempt > cutoff_time
            ]
            
            # Check if too many failed attempts
            if len(self.failed_login_attempts[event.ip_address]) >= 10:
                self.suspicious_ips.add(event.ip_address)
                self.log_event(
                    level=AuditLevel.SECURITY,
                    action=AuditAction.SUSPICIOUS_ACTIVITY,
                    resource=AuditResource.SECURITY,
                    ip_address=event.ip_address,
                    details={
                        'reason': 'Multiple failed login attempts',
                        'count': len(self.failed_login_attempts[event.ip_address]),
                        'time_window': '1 hour'
                    }
                )
        
        # Check for rate limit violations
        if event.action == AuditAction.RATE_LIMIT_EXCEEDED:
            if event.ip_address not in self.rate_limit_violations:
                self.rate_limit_violations[event.ip_address] = []
            
            self.rate_limit_violations[event.ip_address].append(current_time)
            
            # Remove old violations (older than 1 hour)
            cutoff_time = current_time - timedelta(hours=1)
            self.rate_limit_violations[event.ip_address] = [
                violation for violation in self.rate_limit_violations[event.ip_address]
                if violation > cutoff_time
            ]
            
            # Check if too many violations
            if len(self.rate_limit_violations[event.ip_address]) >= 5:
                self.suspicious_ips.add(event.ip_address)
                self.log_event(
                    level=AuditLevel.SECURITY,
                    action=AuditAction.SUSPICIOUS_ACTIVITY,
                    resource=AuditResource.SECURITY,
                    ip_address=event.ip_address,
                    details={
                        'reason': 'Multiple rate limit violations',
                        'count': len(self.rate_limit_violations[event.ip_address]),
                        'time_window': '1 hour'
                    }
                )
    
    def _get_ip_info(self, ip_address: str) -> Dict[str, Any]:
        """Get IP address information"""
        with self.ip_cache_lock:
            if ip_address in self.ip_cache:
                return self.ip_cache[ip_address]
            
            info = {}
            try:
                # Check if it's a private IP
                ip_obj = ipaddress.ip_address(ip_address)
                info['is_private'] = ip_obj.is_private
                info['is_loopback'] = ip_obj.is_loopback
                info['is_multicast'] = ip_obj.is_multicast
                info['version'] = ip_obj.version
                
                # Add to cache
                self.ip_cache[ip_address] = info
                
            except ValueError:
                info['is_private'] = False
                info['is_loopback'] = False
                info['is_multicast'] = False
                info['version'] = None
            
            return info
    
    def log_event(self, level: AuditLevel, action: AuditAction, resource: AuditResource,
                  user_id: str = None, username: str = None, session_id: str = None,
                  ip_address: str = None, user_agent: str = None, endpoint: str = None,
                  method: str = None, status_code: int = None, resource_id: str = None,
                  details: Dict[str, Any] = None, success: bool = True, duration: float = None,
                  request_size: int = None, response_size: int = None, error_message: str = None):
        """Log audit event"""
        
        # Get additional IP information
        ip_info = {}
        if ip_address:
            ip_info = self._get_ip_info(ip_address)
        
        # Parse user agent
        user_agent_info = {}
        if user_agent:
            try:
                parsed_ua = parse(user_agent)
                user_agent_info = {
                    'browser': parsed_ua.browser.family,
                    'browser_version': parsed_ua.browser.version_string,
                    'os': parsed_ua.os.family,
                    'os_version': parsed_ua.os.version_string,
                    'device': parsed_ua.device.family,
                    'is_mobile': parsed_ua.is_mobile,
                    'is_tablet': parsed_ua.is_tablet,
                    'is_pc': parsed_ua.is_pc,
                    'is_bot': parsed_ua.is_bot
                }
            except Exception:
                user_agent_info = {'raw': user_agent}
        
        # Combine details
        combined_details = details or {}
        if ip_info:
            combined_details['ip_info'] = ip_info
        if user_agent_info:
            combined_details['user_agent_info'] = user_agent_info
        
        # Create audit event
        event = AuditEvent(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            level=level,
            action=action,
            resource=resource,
            user_id=user_id,
            username=username,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            resource_id=resource_id,
            details=combined_details if combined_details else None,
            success=success,
            duration=duration,
            request_size=request_size,
            response_size=response_size,
            error_message=error_message
        )
        
        # Queue event for processing
        self.event_queue.put(event)
    
    def query_events(self, filters: Dict[str, Any]) -> List[AuditEvent]:
        """Query audit events"""
        # Use first storage backend for queries
        if self.storage_backends:
            return self.storage_backends[0].query_events(filters)
        return []
    
    def get_user_activity(self, user_id: str, hours: int = 24) -> List[AuditEvent]:
        """Get user activity for specified time period"""
        start_time = datetime.now() - timedelta(hours=hours)
        filters = {
            'user_id': user_id,
            'start_time': start_time,
            'limit': 1000
        }
        return self.query_events(filters)
    
    def get_suspicious_activities(self, hours: int = 24) -> List[AuditEvent]:
        """Get suspicious activities"""
        start_time = datetime.now() - timedelta(hours=hours)
        filters = {
            'level': AuditLevel.SECURITY.value,
            'start_time': start_time,
            'limit': 1000
        }
        return self.query_events(filters)
    
    def get_failed_logins(self, hours: int = 24) -> List[AuditEvent]:
        """Get failed login attempts"""
        start_time = datetime.now() - timedelta(hours=hours)
        filters = {
            'action': AuditAction.LOGIN_FAILED.value,
            'start_time': start_time,
            'limit': 1000
        }
        return self.query_events(filters)
    
    def cleanup_old_events(self, retention_days: int = 90):
        """Clean up old audit events"""
        for backend in self.storage_backends:
            try:
                backend.cleanup_old_events(retention_days)
            except Exception as e:
                logger.error(f"Error cleaning up old events in {backend.__class__.__name__}: {e}")
    
    def is_suspicious_ip(self, ip_address: str) -> bool:
        """Check if IP address is marked as suspicious"""
        return ip_address in self.suspicious_ips
    
    def shutdown(self):
        """Shutdown audit logger"""
        self.event_queue.put(None)
        self.worker_thread.join()

# Global audit logger instance
audit_logger = None

def get_audit_logger() -> QNTIAuditLogger:
    """Get global audit logger instance"""
    global audit_logger
    if audit_logger is None:
        audit_logger = QNTIAuditLogger()
    return audit_logger

def audit_log(level: AuditLevel = AuditLevel.INFO, action: AuditAction = AuditAction.API_REQUEST,
              resource: AuditResource = AuditResource.API_KEY, **kwargs):
    """Decorator for audit logging"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs_inner):
            logger = get_audit_logger()
            start_time = time.time()
            
            # Get request information
            user_id = getattr(request, 'user', {}).get('user_id')
            username = getattr(request, 'user', {}).get('username')
            session_id = getattr(request, 'user', {}).get('session_id')
            ip_address = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
            user_agent = request.headers.get('User-Agent')
            endpoint = request.endpoint
            method = request.method
            
            request_size = None
            if request.content_length:
                request_size = request.content_length
            
            try:
                # Execute function
                result = f(*args, **kwargs_inner)
                
                # Calculate duration
                duration = time.time() - start_time
                
                # Determine response size and status code
                response_size = None
                status_code = 200
                
                if hasattr(result, 'content_length') and result.content_length:
                    response_size = result.content_length
                if hasattr(result, 'status_code'):
                    status_code = result.status_code
                
                # Log successful event
                logger.log_event(
                    level=level,
                    action=action,
                    resource=resource,
                    user_id=user_id,
                    username=username,
                    session_id=session_id,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    endpoint=endpoint,
                    method=method,
                    status_code=status_code,
                    success=True,
                    duration=duration,
                    request_size=request_size,
                    response_size=response_size,
                    **kwargs
                )
                
                return result
                
            except Exception as e:
                # Calculate duration
                duration = time.time() - start_time
                
                # Log error event
                logger.log_event(
                    level=AuditLevel.ERROR,
                    action=action,
                    resource=resource,
                    user_id=user_id,
                    username=username,
                    session_id=session_id,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    endpoint=endpoint,
                    method=method,
                    status_code=500,
                    success=False,
                    duration=duration,
                    request_size=request_size,
                    error_message=str(e),
                    **kwargs
                )
                
                raise
        
        return decorated_function
    return decorator

def audit_api_request():
    """Decorator for API request audit logging"""
    return audit_log(
        level=AuditLevel.INFO,
        action=AuditAction.API_REQUEST,
        resource=AuditResource.API_KEY
    )

def audit_trading_action():
    """Decorator for trading action audit logging"""
    return audit_log(
        level=AuditLevel.INFO,
        action=AuditAction.TRADE_EXECUTE,
        resource=AuditResource.TRADE
    )

def audit_auth_action():
    """Decorator for authentication action audit logging"""
    return audit_log(
        level=AuditLevel.SECURITY,
        action=AuditAction.LOGIN_SUCCESS,
        resource=AuditResource.SESSION
    )

class AuditMiddleware:
    """Flask middleware for automatic audit logging"""
    
    def __init__(self, app=None, audit_logger=None):
        self.app = app
        self.audit_logger = audit_logger or get_audit_logger()
        
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize middleware with Flask app"""
        app.before_request(self.before_request)
        app.after_request(self.after_request)
    
    def before_request(self):
        """Log request start"""
        g.audit_start_time = time.time()
    
    def after_request(self, response):
        """Log request completion"""
        try:
            duration = time.time() - g.audit_start_time
            
            # Get request information
            user_id = getattr(request, 'user', {}).get('user_id')
            username = getattr(request, 'user', {}).get('username')
            session_id = getattr(request, 'user', {}).get('session_id')
            ip_address = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
            user_agent = request.headers.get('User-Agent')
            
            # Determine audit level based on status code
            if response.status_code >= 500:
                level = AuditLevel.ERROR
            elif response.status_code >= 400:
                level = AuditLevel.WARNING
            else:
                level = AuditLevel.INFO
            
            # Log the request
            self.audit_logger.log_event(
                level=level,
                action=AuditAction.API_REQUEST,
                resource=AuditResource.API_KEY,
                user_id=user_id,
                username=username,
                session_id=session_id,
                ip_address=ip_address,
                user_agent=user_agent,
                endpoint=request.endpoint,
                method=request.method,
                status_code=response.status_code,
                success=response.status_code < 400,
                duration=duration,
                request_size=request.content_length,
                response_size=response.content_length
            )
            
        except Exception as e:
            logger.error(f"Error in audit middleware: {e}")
        
        return response