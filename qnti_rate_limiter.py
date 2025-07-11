#!/usr/bin/env python3
"""
QNTI Rate Limiter - Comprehensive Rate Limiting System
Advanced rate limiting with Redis backend and multiple strategies
"""

import time
import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from functools import wraps
from flask import request, jsonify, g
from dataclasses import dataclass
from enum import Enum
import redis
import threading
from collections import defaultdict, deque

logger = logging.getLogger('QNTI_RATE_LIMITER')

class RateLimitType(Enum):
    """Rate limit types"""
    REQUESTS_PER_MINUTE = "requests_per_minute"
    REQUESTS_PER_HOUR = "requests_per_hour"
    REQUESTS_PER_DAY = "requests_per_day"
    BURST_LIMIT = "burst_limit"
    SLIDING_WINDOW = "sliding_window"

class RateLimitScope(Enum):
    """Rate limit scope"""
    GLOBAL = "global"
    USER = "user"
    IP = "ip"
    ENDPOINT = "endpoint"
    API_KEY = "api_key"

@dataclass
class RateLimit:
    """Rate limit configuration"""
    limit: int
    window: int  # seconds
    scope: RateLimitScope
    limit_type: RateLimitType
    burst_limit: Optional[int] = None
    message: Optional[str] = None

@dataclass
class RateLimitResult:
    """Rate limit check result"""
    allowed: bool
    limit: int
    remaining: int
    reset_time: datetime
    retry_after: Optional[int] = None
    message: Optional[str] = None

class RedisRateLimiter:
    """Redis-based rate limiter with sliding window"""
    
    def __init__(self, redis_client=None, key_prefix: str = "qnti_rate_limit"):
        self.redis = redis_client or redis.Redis(host='localhost', port=6379, db=1, decode_responses=True)
        self.key_prefix = key_prefix
        self.local_cache = {}
        self.cache_lock = threading.Lock()
    
    def _get_key(self, identifier: str, scope: RateLimitScope, endpoint: str = None) -> str:
        """Generate Redis key for rate limit"""
        if endpoint:
            return f"{self.key_prefix}:{scope.value}:{identifier}:{endpoint}"
        return f"{self.key_prefix}:{scope.value}:{identifier}"
    
    def check_rate_limit(self, identifier: str, rate_limit: RateLimit, 
                        endpoint: str = None) -> RateLimitResult:
        """Check if request is within rate limit"""
        try:
            key = self._get_key(identifier, rate_limit.scope, endpoint)
            current_time = time.time()
            window_start = current_time - rate_limit.window
            
            # Use sliding window algorithm
            pipe = self.redis.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count current requests
            pipe.zcard(key)
            
            # Add current request
            pipe.zadd(key, {str(current_time): current_time})
            
            # Set expiration
            pipe.expire(key, rate_limit.window + 1)
            
            results = pipe.execute()
            current_count = results[1]
            
            # Check burst limit if configured
            if rate_limit.burst_limit and current_count > rate_limit.burst_limit:
                return RateLimitResult(
                    allowed=False,
                    limit=rate_limit.burst_limit,
                    remaining=0,
                    reset_time=datetime.fromtimestamp(current_time + rate_limit.window),
                    retry_after=rate_limit.window,
                    message="Burst limit exceeded"
                )
            
            # Check main limit
            if current_count > rate_limit.limit:
                # Get oldest entry to calculate reset time
                oldest_entries = self.redis.zrange(key, 0, 0, withscores=True)
                if oldest_entries:
                    oldest_time = oldest_entries[0][1]
                    reset_time = datetime.fromtimestamp(oldest_time + rate_limit.window)
                    retry_after = int(oldest_time + rate_limit.window - current_time)
                else:
                    reset_time = datetime.fromtimestamp(current_time + rate_limit.window)
                    retry_after = rate_limit.window
                
                return RateLimitResult(
                    allowed=False,
                    limit=rate_limit.limit,
                    remaining=0,
                    reset_time=reset_time,
                    retry_after=max(0, retry_after),
                    message=rate_limit.message or "Rate limit exceeded"
                )
            
            return RateLimitResult(
                allowed=True,
                limit=rate_limit.limit,
                remaining=rate_limit.limit - current_count,
                reset_time=datetime.fromtimestamp(current_time + rate_limit.window)
            )
            
        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            # Fail open - allow request on error
            return RateLimitResult(
                allowed=True,
                limit=rate_limit.limit,
                remaining=rate_limit.limit,
                reset_time=datetime.fromtimestamp(time.time() + rate_limit.window)
            )
    
    def reset_rate_limit(self, identifier: str, scope: RateLimitScope, endpoint: str = None):
        """Reset rate limit for identifier"""
        key = self._get_key(identifier, scope, endpoint)
        self.redis.delete(key)
    
    def get_rate_limit_info(self, identifier: str, scope: RateLimitScope, 
                           endpoint: str = None) -> Dict[str, Any]:
        """Get current rate limit information"""
        key = self._get_key(identifier, scope, endpoint)
        current_time = time.time()
        
        # Get all entries
        entries = self.redis.zrangebyscore(key, current_time - 3600, current_time, withscores=True)
        
        return {
            "key": key,
            "current_count": len(entries),
            "entries": [(entry[0], datetime.fromtimestamp(entry[1])) for entry in entries[-10:]]
        }

class MemoryRateLimiter:
    """In-memory rate limiter for fallback"""
    
    def __init__(self, cleanup_interval: int = 60):
        self.buckets = defaultdict(deque)
        self.lock = threading.Lock()
        self.cleanup_interval = cleanup_interval
        self.last_cleanup = time.time()
    
    def _cleanup_old_entries(self):
        """Clean up old entries"""
        current_time = time.time()
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        with self.lock:
            for key, bucket in list(self.buckets.items()):
                # Remove entries older than 1 hour
                while bucket and current_time - bucket[0] > 3600:
                    bucket.popleft()
                
                # Remove empty buckets
                if not bucket:
                    del self.buckets[key]
            
            self.last_cleanup = current_time
    
    def check_rate_limit(self, identifier: str, rate_limit: RateLimit, 
                        endpoint: str = None) -> RateLimitResult:
        """Check rate limit using memory storage"""
        self._cleanup_old_entries()
        
        key = f"{rate_limit.scope.value}:{identifier}"
        if endpoint:
            key += f":{endpoint}"
        
        current_time = time.time()
        window_start = current_time - rate_limit.window
        
        with self.lock:
            bucket = self.buckets[key]
            
            # Remove old entries
            while bucket and bucket[0] < window_start:
                bucket.popleft()
            
            # Check limit
            if len(bucket) >= rate_limit.limit:
                oldest_time = bucket[0] if bucket else current_time
                reset_time = datetime.fromtimestamp(oldest_time + rate_limit.window)
                retry_after = int(oldest_time + rate_limit.window - current_time)
                
                return RateLimitResult(
                    allowed=False,
                    limit=rate_limit.limit,
                    remaining=0,
                    reset_time=reset_time,
                    retry_after=max(0, retry_after),
                    message=rate_limit.message or "Rate limit exceeded"
                )
            
            # Add current request
            bucket.append(current_time)
            
            return RateLimitResult(
                allowed=True,
                limit=rate_limit.limit,
                remaining=rate_limit.limit - len(bucket),
                reset_time=datetime.fromtimestamp(current_time + rate_limit.window)
            )

class QNTIRateLimiter:
    """Main rate limiter with multiple backends"""
    
    def __init__(self, redis_client=None, use_redis: bool = True):
        self.use_redis = use_redis
        self.redis_limiter = RedisRateLimiter(redis_client) if use_redis else None
        self.memory_limiter = MemoryRateLimiter()
        
        # Default rate limits
        self.default_limits = {
            "global": [
                RateLimit(1000, 60, RateLimitScope.GLOBAL, RateLimitType.REQUESTS_PER_MINUTE),
                RateLimit(10000, 3600, RateLimitScope.GLOBAL, RateLimitType.REQUESTS_PER_HOUR)
            ],
            "user": [
                RateLimit(100, 60, RateLimitScope.USER, RateLimitType.REQUESTS_PER_MINUTE),
                RateLimit(1000, 3600, RateLimitScope.USER, RateLimitType.REQUESTS_PER_HOUR),
                RateLimit(200, 60, RateLimitScope.USER, RateLimitType.BURST_LIMIT, burst_limit=50)
            ],
            "ip": [
                RateLimit(60, 60, RateLimitScope.IP, RateLimitType.REQUESTS_PER_MINUTE),
                RateLimit(600, 3600, RateLimitScope.IP, RateLimitType.REQUESTS_PER_HOUR)
            ],
            "api_key": [
                RateLimit(500, 60, RateLimitScope.API_KEY, RateLimitType.REQUESTS_PER_MINUTE),
                RateLimit(5000, 3600, RateLimitScope.API_KEY, RateLimitType.REQUESTS_PER_HOUR)
            ]
        }
        
        # Endpoint-specific limits
        self.endpoint_limits = {
            "/api/trade/execute": [
                RateLimit(10, 60, RateLimitScope.USER, RateLimitType.REQUESTS_PER_MINUTE),
                RateLimit(50, 3600, RateLimitScope.USER, RateLimitType.REQUESTS_PER_HOUR)
            ],
            "/api/auth/login": [
                RateLimit(5, 300, RateLimitScope.IP, RateLimitType.REQUESTS_PER_MINUTE),
                RateLimit(20, 3600, RateLimitScope.IP, RateLimitType.REQUESTS_PER_HOUR)
            ],
            "/api/data/market": [
                RateLimit(200, 60, RateLimitScope.USER, RateLimitType.REQUESTS_PER_MINUTE),
                RateLimit(2000, 3600, RateLimitScope.USER, RateLimitType.REQUESTS_PER_HOUR)
            ]
        }
    
    def _get_limiter(self) -> Union[RedisRateLimiter, MemoryRateLimiter]:
        """Get appropriate limiter backend"""
        if self.use_redis and self.redis_limiter:
            try:
                # Test Redis connection
                self.redis_limiter.redis.ping()
                return self.redis_limiter
            except Exception as e:
                logger.warning(f"Redis unavailable, falling back to memory limiter: {e}")
        
        return self.memory_limiter
    
    def check_limits(self, identifier: str, scope: RateLimitScope, 
                    endpoint: str = None) -> List[RateLimitResult]:
        """Check multiple rate limits"""
        limiter = self._get_limiter()
        results = []
        
        # Check default limits for scope
        if scope.value in self.default_limits:
            for rate_limit in self.default_limits[scope.value]:
                result = limiter.check_rate_limit(identifier, rate_limit, endpoint)
                results.append(result)
        
        # Check endpoint-specific limits
        if endpoint and endpoint in self.endpoint_limits:
            for rate_limit in self.endpoint_limits[endpoint]:
                result = limiter.check_rate_limit(identifier, rate_limit, endpoint)
                results.append(result)
        
        return results
    
    def is_rate_limited(self, identifier: str, scope: RateLimitScope, 
                       endpoint: str = None) -> Tuple[bool, RateLimitResult]:
        """Check if request should be rate limited"""
        results = self.check_limits(identifier, scope, endpoint)
        
        for result in results:
            if not result.allowed:
                return True, result
        
        return False, results[0] if results else RateLimitResult(
            allowed=True, limit=0, remaining=0, reset_time=datetime.now()
        )
    
    def add_custom_limit(self, endpoint: str, rate_limit: RateLimit):
        """Add custom rate limit for endpoint"""
        if endpoint not in self.endpoint_limits:
            self.endpoint_limits[endpoint] = []
        self.endpoint_limits[endpoint].append(rate_limit)
    
    def get_rate_limit_headers(self, results: List[RateLimitResult]) -> Dict[str, str]:
        """Generate rate limit headers"""
        headers = {}
        
        if results:
            # Use the most restrictive limit
            min_result = min(results, key=lambda r: r.remaining)
            
            headers.update({
                'X-RateLimit-Limit': str(min_result.limit),
                'X-RateLimit-Remaining': str(min_result.remaining),
                'X-RateLimit-Reset': str(int(min_result.reset_time.timestamp())),
                'X-RateLimit-Reset-After': str(int((min_result.reset_time - datetime.now()).total_seconds()))
            })
            
            if min_result.retry_after:
                headers['Retry-After'] = str(min_result.retry_after)
        
        return headers

# Global rate limiter instance
rate_limiter = None

def get_rate_limiter() -> QNTIRateLimiter:
    """Get global rate limiter instance"""
    global rate_limiter
    if rate_limiter is None:
        rate_limiter = QNTIRateLimiter()
    return rate_limiter

def get_request_identifier(scope: RateLimitScope) -> str:
    """Get identifier for rate limiting based on scope"""
    if scope == RateLimitScope.USER:
        return getattr(request, 'user', {}).get('user_id', 'anonymous')
    elif scope == RateLimitScope.IP:
        return request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
    elif scope == RateLimitScope.API_KEY:
        return request.headers.get('X-API-Key', 'no-key')
    elif scope == RateLimitScope.GLOBAL:
        return 'global'
    else:
        return 'unknown'

def rate_limit(scopes: List[RateLimitScope] = None, custom_limits: List[RateLimit] = None):
    """Rate limiting decorator"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            limiter = get_rate_limiter()
            endpoint = request.endpoint
            
            # Default scopes
            if scopes is None:
                check_scopes = [RateLimitScope.IP, RateLimitScope.USER]
            else:
                check_scopes = scopes
            
            # Add custom limits if provided
            if custom_limits:
                for custom_limit in custom_limits:
                    limiter.add_custom_limit(endpoint, custom_limit)
            
            # Check rate limits for each scope
            for scope in check_scopes:
                identifier = get_request_identifier(scope)
                is_limited, result = limiter.is_rate_limited(identifier, scope, endpoint)
                
                if is_limited:
                    headers = limiter.get_rate_limit_headers([result])
                    response = jsonify({
                        "error": "Rate limit exceeded",
                        "message": result.message or "Too many requests",
                        "retry_after": result.retry_after
                    })
                    response.status_code = 429
                    for key, value in headers.items():
                        response.headers[key] = value
                    return response
            
            # Add rate limit headers to successful responses
            @f.after_app_request
            def after_request(response):
                try:
                    # Get rate limit info for headers
                    for scope in check_scopes:
                        identifier = get_request_identifier(scope)
                        results = limiter.check_limits(identifier, scope, endpoint)
                        headers = limiter.get_rate_limit_headers(results)
                        for key, value in headers.items():
                            response.headers[key] = value
                        break  # Only add headers from first scope
                except Exception as e:
                    logger.error(f"Error adding rate limit headers: {e}")
                return response
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def strict_rate_limit(limit: int, window: int, scope: RateLimitScope = RateLimitScope.USER):
    """Strict rate limiting decorator for sensitive endpoints"""
    custom_limit = RateLimit(
        limit=limit,
        window=window,
        scope=scope,
        limit_type=RateLimitType.REQUESTS_PER_MINUTE,
        message=f"Strict rate limit: {limit} requests per {window} seconds"
    )
    return rate_limit(scopes=[scope], custom_limits=[custom_limit])

def trading_rate_limit():
    """Specialized rate limiting for trading endpoints"""
    return rate_limit(
        scopes=[RateLimitScope.USER, RateLimitScope.IP],
        custom_limits=[
            RateLimit(5, 60, RateLimitScope.USER, RateLimitType.REQUESTS_PER_MINUTE, 
                     message="Trading rate limit exceeded"),
            RateLimit(20, 3600, RateLimitScope.USER, RateLimitType.REQUESTS_PER_HOUR,
                     message="Trading hourly limit exceeded")
        ]
    )

def auth_rate_limit():
    """Specialized rate limiting for authentication endpoints"""
    return rate_limit(
        scopes=[RateLimitScope.IP],
        custom_limits=[
            RateLimit(5, 300, RateLimitScope.IP, RateLimitType.REQUESTS_PER_MINUTE,
                     message="Authentication rate limit exceeded"),
            RateLimit(20, 3600, RateLimitScope.IP, RateLimitType.REQUESTS_PER_HOUR,
                     message="Authentication hourly limit exceeded")
        ]
    )

class RateLimitMiddleware:
    """Flask middleware for automatic rate limiting"""
    
    def __init__(self, app=None, rate_limiter=None):
        self.app = app
        self.rate_limiter = rate_limiter or get_rate_limiter()
        
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize middleware with Flask app"""
        app.before_request(self.before_request)
        app.after_request(self.after_request)
    
    def before_request(self):
        """Check rate limits before request"""
        # Skip rate limiting for static files
        if request.endpoint and request.endpoint.startswith('static'):
            return
        
        # Get identifiers
        ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
        user_id = getattr(request, 'user', {}).get('user_id', 'anonymous')
        
        # Check IP rate limit
        is_limited, result = self.rate_limiter.is_rate_limited(ip, RateLimitScope.IP, request.endpoint)
        if is_limited:
            headers = self.rate_limiter.get_rate_limit_headers([result])
            response = jsonify({
                "error": "Rate limit exceeded",
                "message": result.message or "Too many requests from this IP",
                "retry_after": result.retry_after
            })
            response.status_code = 429
            for key, value in headers.items():
                response.headers[key] = value
            return response
        
        # Store rate limit info for after_request
        g.rate_limit_info = {
            'ip': ip,
            'user_id': user_id,
            'endpoint': request.endpoint
        }
    
    def after_request(self, response):
        """Add rate limit headers after request"""
        try:
            if hasattr(g, 'rate_limit_info'):
                info = g.rate_limit_info
                
                # Add rate limit headers
                results = self.rate_limiter.check_limits(info['ip'], RateLimitScope.IP, info['endpoint'])
                headers = self.rate_limiter.get_rate_limit_headers(results)
                
                for key, value in headers.items():
                    response.headers[key] = value
        except Exception as e:
            logger.error(f"Error adding rate limit headers: {e}")
        
        return response