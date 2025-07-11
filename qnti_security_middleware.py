#!/usr/bin/env python3
"""
QNTI Security Middleware - Comprehensive Security Middleware System
Advanced middleware for Flask applications with enterprise-grade security features
"""

import os
import json
import logging
import time
import hashlib
import ipaddress
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from functools import wraps
from flask import Flask, request, jsonify, g, current_app, session
from werkzeug.exceptions import BadRequest, Unauthorized, Forbidden, TooManyRequests
import re
import bleach
from urllib.parse import quote, unquote
import user_agents

# Import QNTI security components
from qnti_security_framework import get_security_framework, SecurityConfig
from qnti_auth_system import get_auth_system, Permission, UserRole
from qnti_rate_limiter import get_rate_limiter, RateLimitScope
from qnti_encryption import get_encryption_system, EncryptionMethod
from qnti_audit_logger import get_audit_logger, AuditLevel, AuditAction, AuditResource

logger = logging.getLogger('QNTI_SECURITY_MIDDLEWARE')

class SecurityMiddleware:
    """Comprehensive security middleware for Flask applications"""
    
    def __init__(self, app: Flask = None, config: SecurityConfig = None):
        self.app = app
        self.config = config or SecurityConfig()
        self.security_framework = get_security_framework()
        self.auth_system = get_auth_system()
        self.rate_limiter = get_rate_limiter()
        self.encryption_system = get_encryption_system()
        self.audit_logger = get_audit_logger()
        
        # Security state tracking
        self.request_start_time = {}
        self.failed_requests = {}
        self.suspicious_patterns = {}
        
        if app:
            self.init_app(app)
    
    def init_app(self, app: Flask):
        """Initialize middleware with Flask app"""
        self.app = app
        
        # Register middleware in correct order
        app.before_request(self.before_request)
        app.after_request(self.after_request)
        app.teardown_request(self.teardown_request)
        
        # Register error handlers
        self._register_error_handlers()
        
        logger.info("Security middleware initialized")
    
    def before_request(self):
        """Security processing before request"""
        g.security_context = {}
        g.request_start_time = time.time()
        
        # Step 1: IP and request validation
        if not self._validate_request_basics():
            return self._security_response('Invalid request', 400)
        
        # Step 2: Rate limiting
        if not self._check_rate_limits():
            return self._security_response('Rate limit exceeded', 429)
        
        # Step 3: Authentication and authorization
        if not self._authenticate_request():
            return self._security_response('Authentication failed', 401)
        
        # Step 4: Input validation and sanitization
        if not self._validate_input():
            return self._security_response('Invalid input', 400)
        
        # Step 5: Security headers validation
        if not self._validate_security_headers():
            return self._security_response('Security headers validation failed', 400)
        
        # Step 6: Suspicious activity detection
        if not self._check_suspicious_activity():
            return self._security_response('Suspicious activity detected', 403)
        
        # Step 7: Session security
        if not self._validate_session_security():
            return self._security_response('Session security validation failed', 401)
    
    def after_request(self, response):
        """Security processing after request"""
        try:
            # Add security headers
            response = self._add_security_headers(response)
            
            # Add audit information
            response = self._add_audit_headers(response)
            
            # Encrypt sensitive responses
            response = self._encrypt_sensitive_response(response)
            
            # Log successful request
            self._log_successful_request(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in after_request security processing: {e}")
            return response
    
    def teardown_request(self, exception):
        """Security cleanup after request"""
        try:
            # Log any exceptions
            if exception:
                self._log_request_exception(exception)
            
            # Clean up security context
            if hasattr(g, 'security_context'):
                del g.security_context
            
            # Update security metrics
            self._update_security_metrics()
            
        except Exception as e:
            logger.error(f"Error in teardown_request: {e}")
    
    def _validate_request_basics(self) -> bool:
        """Validate basic request properties"""
        try:
            # Get client IP
            client_ip = self._get_client_ip()
            g.security_context['client_ip'] = client_ip
            
            # IP validation
            if not self._is_valid_ip(client_ip):
                self._log_security_event(
                    AuditLevel.WARNING,
                    AuditAction.SECURITY_VIOLATION,
                    {'reason': 'Invalid IP address', 'ip': client_ip}
                )
                return False
            
            # IP blacklist check
            if self._is_ip_blacklisted(client_ip):
                self._log_security_event(
                    AuditLevel.SECURITY,
                    AuditAction.ACCESS_DENIED,
                    {'reason': 'IP blacklisted', 'ip': client_ip}
                )
                return False
            
            # Request size validation
            max_size = self.config.get('input_validation.max_request_size', 10485760)
            if request.content_length and request.content_length > max_size:
                self._log_security_event(
                    AuditLevel.WARNING,
                    AuditAction.SECURITY_VIOLATION,
                    {'reason': 'Request too large', 'size': request.content_length, 'max_size': max_size}
                )
                return False
            
            # HTTPS enforcement
            if self.config.get('encryption.require_https', True) and not self._is_https_request():
                self._log_security_event(
                    AuditLevel.WARNING,
                    AuditAction.SECURITY_VIOLATION,
                    {'reason': 'HTTPS required', 'protocol': request.scheme}
                )
                return False
            
            # User agent validation
            user_agent = request.headers.get('User-Agent', '')
            if not self._is_valid_user_agent(user_agent):
                self._log_security_event(
                    AuditLevel.WARNING,
                    AuditAction.SECURITY_VIOLATION,
                    {'reason': 'Invalid user agent', 'user_agent': user_agent}
                )
                return False
            
            g.security_context['user_agent'] = user_agent
            return True
            
        except Exception as e:
            logger.error(f"Error in basic request validation: {e}")
            return False
    
    def _check_rate_limits(self) -> bool:
        """Check rate limits"""
        try:
            client_ip = g.security_context.get('client_ip')
            
            # IP-based rate limiting
            is_limited, result = self.rate_limiter.is_rate_limited(client_ip, RateLimitScope.IP, request.endpoint)
            if is_limited:
                self._log_security_event(
                    AuditLevel.WARNING,
                    AuditAction.RATE_LIMIT_EXCEEDED,
                    {'ip': client_ip, 'endpoint': request.endpoint, 'limit': result.limit}
                )
                return False
            
            # Global rate limiting
            is_limited, result = self.rate_limiter.is_rate_limited('global', RateLimitScope.GLOBAL, request.endpoint)
            if is_limited:
                self._log_security_event(
                    AuditLevel.WARNING,
                    AuditAction.RATE_LIMIT_EXCEEDED,
                    {'scope': 'global', 'endpoint': request.endpoint, 'limit': result.limit}
                )
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in rate limit check: {e}")
            return True  # Fail open for rate limiting
    
    def _authenticate_request(self) -> bool:
        """Authenticate request"""
        try:
            # Skip authentication for public endpoints
            if self._is_public_endpoint():
                return True
            
            # Check for authentication token
            auth_header = request.headers.get('Authorization')
            api_key = request.headers.get('X-API-Key')
            
            if auth_header and auth_header.startswith('Bearer '):
                # JWT authentication
                token = auth_header.split(' ')[1]
                result = self.auth_system.verify_token(token)
                
                if result['valid']:
                    request.user = result['payload']
                    g.security_context['user_id'] = result['payload'].get('user_id')
                    g.security_context['auth_method'] = 'jwt'
                    return True
                else:
                    self._log_security_event(
                        AuditLevel.WARNING,
                        AuditAction.LOGIN_FAILED,
                        {'reason': 'Invalid JWT token', 'error': result.get('error')}
                    )
                    return False
            
            elif api_key:
                # API key authentication
                result = self.auth_system.verify_api_key(api_key)
                
                if result['valid']:
                    request.user = {
                        'user_id': result['user_id'],
                        'username': result['username'],
                        'role': result['role']
                    }
                    g.security_context['user_id'] = result['user_id']
                    g.security_context['auth_method'] = 'api_key'
                    return True
                else:
                    self._log_security_event(
                        AuditLevel.WARNING,
                        AuditAction.LOGIN_FAILED,
                        {'reason': 'Invalid API key', 'error': result.get('error')}
                    )
                    return False
            
            else:
                # No authentication provided
                self._log_security_event(
                    AuditLevel.WARNING,
                    AuditAction.LOGIN_FAILED,
                    {'reason': 'No authentication provided'}
                )
                return False
            
        except Exception as e:
            logger.error(f"Error in authentication: {e}")
            return False
    
    def _validate_input(self) -> bool:
        """Validate and sanitize input"""
        try:
            if not self.config.get('input_validation.enabled', True):
                return True
            
            # Validate JSON payload
            if request.is_json:
                try:
                    data = request.get_json()
                    if data:
                        # Check for malicious patterns
                        if self._contains_malicious_patterns(data):
                            return False
                        
                        # Sanitize data
                        sanitized_data = self._sanitize_json_data(data)
                        g.security_context['sanitized_data'] = sanitized_data
                        
                except Exception as e:
                    logger.warning(f"JSON validation failed: {e}")
                    return False
            
            # Validate query parameters
            if request.args:
                for key, value in request.args.items():
                    if self._contains_malicious_content(value):
                        self._log_security_event(
                            AuditLevel.WARNING,
                            AuditAction.SECURITY_VIOLATION,
                            {'reason': 'Malicious query parameter', 'parameter': key, 'value': value}
                        )
                        return False
            
            # Validate form data
            if request.form:
                for key, value in request.form.items():
                    if self._contains_malicious_content(value):
                        self._log_security_event(
                            AuditLevel.WARNING,
                            AuditAction.SECURITY_VIOLATION,
                            {'reason': 'Malicious form data', 'field': key, 'value': value}
                        )
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in input validation: {e}")
            return False
    
    def _validate_security_headers(self) -> bool:
        """Validate security headers"""
        try:
            # Check for required headers
            required_headers = ['User-Agent', 'Accept']
            for header in required_headers:
                if header not in request.headers:
                    self._log_security_event(
                        AuditLevel.WARNING,
                        AuditAction.SECURITY_VIOLATION,
                        {'reason': f'Missing required header: {header}'}
                    )
                    return False
            
            # Validate Content-Type for POST/PUT requests
            if request.method in ['POST', 'PUT', 'PATCH']:
                content_type = request.headers.get('Content-Type', '')
                if not content_type:
                    self._log_security_event(
                        AuditLevel.WARNING,
                        AuditAction.SECURITY_VIOLATION,
                        {'reason': 'Missing Content-Type header'}
                    )
                    return False
            
            # Check for suspicious headers
            suspicious_headers = ['X-Forwarded-Host', 'X-Original-URL', 'X-Rewrite-URL']
            for header in suspicious_headers:
                if header in request.headers:
                    self._log_security_event(
                        AuditLevel.WARNING,
                        AuditAction.SECURITY_VIOLATION,
                        {'reason': f'Suspicious header present: {header}', 'value': request.headers[header]}
                    )
            
            return True
            
        except Exception as e:
            logger.error(f"Error in security headers validation: {e}")
            return True  # Fail open for header validation
    
    def _check_suspicious_activity(self) -> bool:
        """Check for suspicious activity patterns"""
        try:
            client_ip = g.security_context.get('client_ip')
            
            # Check for rapid requests from same IP
            if self._is_rapid_requests(client_ip):
                self._log_security_event(
                    AuditLevel.SECURITY,
                    AuditAction.SUSPICIOUS_ACTIVITY,
                    {'reason': 'Rapid requests detected', 'ip': client_ip}
                )
                return False
            
            # Check for unusual request patterns
            if self._has_unusual_patterns():
                self._log_security_event(
                    AuditLevel.SECURITY,
                    AuditAction.SUSPICIOUS_ACTIVITY,
                    {'reason': 'Unusual request patterns', 'ip': client_ip}
                )
                return False
            
            # Check for known attack patterns
            if self._has_attack_patterns():
                self._log_security_event(
                    AuditLevel.SECURITY,
                    AuditAction.SUSPICIOUS_ACTIVITY,
                    {'reason': 'Attack patterns detected', 'ip': client_ip}
                )
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in suspicious activity check: {e}")
            return True  # Fail open for suspicious activity detection
    
    def _validate_session_security(self) -> bool:
        """Validate session security"""
        try:
            # Skip for public endpoints
            if self._is_public_endpoint():
                return True
            
            # Check session timeout
            if hasattr(request, 'user') and request.user:
                session_id = request.user.get('session_id')
                if session_id:
                    # Validate session is still active
                    if not self.auth_system._is_session_active(session_id):
                        self._log_security_event(
                            AuditLevel.WARNING,
                            AuditAction.SESSION_EXPIRED,
                            {'session_id': session_id}
                        )
                        return False
                    
                    # Update session activity
                    self.auth_system._update_session_activity(session_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error in session validation: {e}")
            return True  # Fail open for session validation
    
    def _add_security_headers(self, response):
        """Add security headers to response"""
        try:
            # Apply security headers from framework
            response = self.security_framework.security_headers.apply_headers(response)
            
            # Add additional security headers
            response.headers['X-Security-Middleware'] = 'QNTI-v1.0'
            response.headers['X-Request-ID'] = g.security_context.get('request_id', 'unknown')
            
            # Add rate limit headers
            client_ip = g.security_context.get('client_ip')
            if client_ip:
                results = self.rate_limiter.check_limits(client_ip, RateLimitScope.IP, request.endpoint)
                headers = self.rate_limiter.get_rate_limit_headers(results)
                for key, value in headers.items():
                    response.headers[key] = value
            
            return response
            
        except Exception as e:
            logger.error(f"Error adding security headers: {e}")
            return response
    
    def _add_audit_headers(self, response):
        """Add audit information to response headers"""
        try:
            # Add audit trace ID
            if hasattr(g, 'audit_trace_id'):
                response.headers['X-Audit-Trace-ID'] = g.audit_trace_id
            
            # Add security context information (non-sensitive)
            response.headers['X-Security-Context'] = json.dumps({
                'auth_method': g.security_context.get('auth_method'),
                'request_validated': True,
                'timestamp': datetime.now().isoformat()
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error adding audit headers: {e}")
            return response
    
    def _encrypt_sensitive_response(self, response):
        """Encrypt sensitive response data"""
        try:
            # Check if encryption is enabled and requested
            if not self.config.get('encryption.enabled', True):
                return response
            
            encryption_support = request.headers.get('X-Encryption-Support')
            if not encryption_support:
                return response
            
            # Encrypt sensitive endpoints
            sensitive_endpoints = self.config.get('encryption.sensitive_endpoints', [])
            if request.endpoint in sensitive_endpoints:
                if response.is_json:
                    data = response.get_json()
                    if data:
                        encrypted_data = self._encrypt_response_data(data)
                        response.data = json.dumps(encrypted_data)
                        response.headers['X-Encryption-Used'] = 'true'
            
            return response
            
        except Exception as e:
            logger.error(f"Error encrypting response: {e}")
            return response
    
    def _log_successful_request(self, response):
        """Log successful request"""
        try:
            duration = time.time() - g.request_start_time
            
            self.audit_logger.log_event(
                level=AuditLevel.INFO,
                action=AuditAction.API_REQUEST,
                resource=AuditResource.API_KEY,
                user_id=g.security_context.get('user_id'),
                username=getattr(request, 'user', {}).get('username'),
                session_id=getattr(request, 'user', {}).get('session_id'),
                ip_address=g.security_context.get('client_ip'),
                user_agent=g.security_context.get('user_agent'),
                endpoint=request.endpoint,
                method=request.method,
                status_code=response.status_code,
                success=response.status_code < 400,
                duration=duration,
                request_size=request.content_length,
                response_size=response.content_length,
                details={
                    'auth_method': g.security_context.get('auth_method'),
                    'security_validated': True
                }
            )
            
        except Exception as e:
            logger.error(f"Error logging successful request: {e}")
    
    def _log_request_exception(self, exception):
        """Log request exception"""
        try:
            duration = time.time() - g.request_start_time
            
            self.audit_logger.log_event(
                level=AuditLevel.ERROR,
                action=AuditAction.API_ERROR,
                resource=AuditResource.API_KEY,
                user_id=g.security_context.get('user_id'),
                username=getattr(request, 'user', {}).get('username'),
                session_id=getattr(request, 'user', {}).get('session_id'),
                ip_address=g.security_context.get('client_ip'),
                user_agent=g.security_context.get('user_agent'),
                endpoint=request.endpoint,
                method=request.method,
                status_code=500,
                success=False,
                duration=duration,
                request_size=request.content_length,
                error_message=str(exception),
                details={
                    'exception_type': type(exception).__name__,
                    'auth_method': g.security_context.get('auth_method')
                }
            )
            
        except Exception as e:
            logger.error(f"Error logging request exception: {e}")
    
    def _update_security_metrics(self):
        """Update security metrics"""
        try:
            # Update request counters
            # This could be implemented to update metrics in Redis or other storage
            pass
            
        except Exception as e:
            logger.error(f"Error updating security metrics: {e}")
    
    def _get_client_ip(self) -> str:
        """Get client IP address"""
        # Check for forwarded headers
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            # Get the first IP in the chain
            return forwarded_for.split(',')[0].strip()
        
        forwarded = request.headers.get('X-Forwarded')
        if forwarded:
            return forwarded.split(',')[0].strip()
        
        real_ip = request.headers.get('X-Real-IP')
        if real_ip:
            return real_ip.strip()
        
        return request.remote_addr or '127.0.0.1'
    
    def _is_valid_ip(self, ip_address: str) -> bool:
        """Validate IP address"""
        try:
            ipaddress.ip_address(ip_address)
            return True
        except ValueError:
            return False
    
    def _is_ip_blacklisted(self, ip_address: str) -> bool:
        """Check if IP is blacklisted"""
        blacklist = self.config.get('brute_force_protection.blacklist_ips', [])
        return ip_address in blacklist
    
    def _is_https_request(self) -> bool:
        """Check if request is HTTPS"""
        return request.is_secure or request.headers.get('X-Forwarded-Proto') == 'https'
    
    def _is_valid_user_agent(self, user_agent: str) -> bool:
        """Validate user agent"""
        if not user_agent:
            return False
        
        # Check for suspicious user agents
        suspicious_patterns = [
            r'sqlmap',
            r'nmap',
            r'nikto',
            r'dirb',
            r'gobuster',
            r'wpscan'
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, user_agent, re.IGNORECASE):
                return False
        
        return True
    
    def _is_public_endpoint(self) -> bool:
        """Check if endpoint is public"""
        public_endpoints = [
            'static',
            'health',
            'security_health',
            'auth.login',
            'auth.register'
        ]
        
        return request.endpoint in public_endpoints
    
    def _contains_malicious_patterns(self, data: Any) -> bool:
        """Check for malicious patterns in data"""
        if isinstance(data, dict):
            for key, value in data.items():
                if self._contains_malicious_content(str(key)) or self._contains_malicious_patterns(value):
                    return True
        elif isinstance(data, list):
            for item in data:
                if self._contains_malicious_patterns(item):
                    return True
        elif isinstance(data, str):
            return self._contains_malicious_content(data)
        
        return False
    
    def _contains_malicious_content(self, content: str) -> bool:
        """Check for malicious content in string"""
        # SQL injection patterns
        sql_patterns = self.config.get('input_validation.sql_injection_patterns', [])
        for pattern in sql_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        
        # XSS patterns
        xss_patterns = self.config.get('input_validation.xss_patterns', [])
        for pattern in xss_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        
        return False
    
    def _sanitize_json_data(self, data: Any) -> Any:
        """Sanitize JSON data"""
        if isinstance(data, dict):
            return {self._sanitize_string(str(k)): self._sanitize_json_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_json_data(item) for item in data]
        elif isinstance(data, str):
            return self._sanitize_string(data)
        else:
            return data
    
    def _sanitize_string(self, value: str) -> str:
        """Sanitize string value"""
        # HTML sanitization
        if self.config.get('input_validation.sanitize_html', True):
            value = bleach.clean(value, tags=[], attributes={}, strip=True)
        
        # URL encode/decode to prevent injection
        value = unquote(quote(value, safe=''))
        
        return value
    
    def _is_rapid_requests(self, ip_address: str) -> bool:
        """Check for rapid requests from IP"""
        current_time = time.time()
        
        if ip_address not in self.request_start_time:
            self.request_start_time[ip_address] = []
        
        # Clean old entries (older than 1 minute)
        self.request_start_time[ip_address] = [
            req_time for req_time in self.request_start_time[ip_address]
            if current_time - req_time < 60
        ]
        
        # Add current request
        self.request_start_time[ip_address].append(current_time)
        
        # Check if too many requests in last minute
        return len(self.request_start_time[ip_address]) > 100
    
    def _has_unusual_patterns(self) -> bool:
        """Check for unusual request patterns"""
        # Check for unusual request sizes
        if request.content_length and request.content_length > 1000000:  # 1MB
            return True
        
        # Check for unusual number of headers
        if len(request.headers) > 50:
            return True
        
        # Check for unusual query parameters
        if len(request.args) > 20:
            return True
        
        return False
    
    def _has_attack_patterns(self) -> bool:
        """Check for known attack patterns"""
        # Check URL for attack patterns
        attack_patterns = [
            r'\.\./',
            r'%2e%2e%2f',
            r'<script',
            r'javascript:',
            r'vbscript:',
            r'onload=',
            r'onerror=',
            r'union.*select',
            r'insert.*into',
            r'update.*set',
            r'delete.*from'
        ]
        
        url = request.url.lower()
        for pattern in attack_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return True
        
        return False
    
    def _encrypt_response_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive response data"""
        sensitive_fields = self.config.get('encryption.sensitive_fields', [])
        
        def encrypt_field(key: str, value: Any) -> Any:
            if key.lower() in [field.lower() for field in sensitive_fields]:
                encrypted = self.encryption_system.encrypt(value, EncryptionMethod.AES_256_GCM)
                return {
                    'encrypted': True,
                    'method': 'aes_256_gcm',
                    'data': self.encryption_system.serialize_encrypted_data(encrypted)
                }
            return value
        
        def process_dict(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: process_dict(encrypt_field(k, v)) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [process_dict(item) for item in obj]
            else:
                return obj
        
        return process_dict(data)
    
    def _security_response(self, message: str, status_code: int) -> Tuple[Dict[str, Any], int]:
        """Generate security error response"""
        response = {
            'error': message,
            'timestamp': datetime.now().isoformat(),
            'request_id': g.security_context.get('request_id', 'unknown')
        }
        
        return jsonify(response), status_code
    
    def _log_security_event(self, level: AuditLevel, action: AuditAction, details: Dict[str, Any]):
        """Log security event"""
        self.audit_logger.log_event(
            level=level,
            action=action,
            resource=AuditResource.SECURITY,
            user_id=g.security_context.get('user_id'),
            username=getattr(request, 'user', {}).get('username'),
            session_id=getattr(request, 'user', {}).get('session_id'),
            ip_address=g.security_context.get('client_ip'),
            user_agent=g.security_context.get('user_agent'),
            endpoint=request.endpoint,
            method=request.method,
            success=False,
            details=details
        )
    
    def _register_error_handlers(self):
        """Register error handlers"""
        @self.app.errorhandler(400)
        def bad_request(error):
            return self._security_response('Bad Request', 400)
        
        @self.app.errorhandler(401)
        def unauthorized(error):
            return self._security_response('Unauthorized', 401)
        
        @self.app.errorhandler(403)
        def forbidden(error):
            return self._security_response('Forbidden', 403)
        
        @self.app.errorhandler(429)
        def rate_limit_exceeded(error):
            return self._security_response('Rate Limit Exceeded', 429)
        
        @self.app.errorhandler(500)
        def internal_server_error(error):
            return self._security_response('Internal Server Error', 500)

# Global middleware instance
security_middleware = None

def get_security_middleware() -> SecurityMiddleware:
    """Get global security middleware instance"""
    global security_middleware
    if security_middleware is None:
        security_middleware = SecurityMiddleware()
    return security_middleware

def init_security_middleware(app: Flask, config: SecurityConfig = None):
    """Initialize security middleware with Flask app"""
    middleware = SecurityMiddleware(app, config)
    return middleware