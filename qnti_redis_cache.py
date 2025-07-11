# qnti_redis_cache.py
# Redis caching layer for QNTI system to improve performance

import json
import redis
import logging
from typing import Any, Optional, Dict, List, Callable
from datetime import datetime, timedelta
from functools import wraps
import hashlib
import pickle
import time

logger = logging.getLogger(__name__)

class QNTIRedisCache:
    """Redis cache manager for QNTI system"""
    
    def __init__(self, host='localhost', port=6379, db=0, decode_responses=False):
        """Initialize Redis connection with connection pooling"""
        self.pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            decode_responses=decode_responses,
            max_connections=50
        )
        self.redis_client = redis.Redis(connection_pool=self.pool)
        self.default_ttl = 300  # 5 minutes default
        
        # Cache key prefixes for different data types
        self.prefixes = {
            'account': 'qnti:account:',
            'trades': 'qnti:trades:',
            'ea_stats': 'qnti:ea:',
            'health': 'qnti:health:',
            'pnl': 'qnti:pnl:',
            'market': 'qnti:market:',
            'system': 'qnti:system:'
        }
        
        # TTL settings for different data types (in seconds)
        self.ttl_settings = {
            'account': 10,      # Account info updates frequently
            'trades': 5,        # Trades change rapidly
            'ea_stats': 60,     # EA stats can be cached longer
            'health': 30,       # System health moderate cache
            'pnl': 15,          # P&L calculations expensive but need freshness
            'market': 5,        # Market data very dynamic
            'system': 300       # System config rarely changes
        }
        
        self._test_connection()
    
    def _test_connection(self):
        """Test Redis connection"""
        try:
            if self.redis_client:
                self.redis_client.ping()
                logger.info("Redis connection established successfully")
        except (redis.ConnectionError, Exception) as e:
            logger.error(f"Failed to connect to Redis. Cache will be disabled. Error: {e}")
            self.redis_client = None
    
    def _generate_key(self, prefix: str, identifier: str) -> str:
        """Generate cache key with prefix"""
        return f"{self.prefixes.get(prefix, 'qnti:')}{identifier}"
    
    def _serialize(self, data: Any) -> str:
        """Serialize data for Redis storage"""
        try:
            # Try JSON first for simple types
            return json.dumps(data)
        except (TypeError, ValueError):
            # Fall back to pickle for complex objects
            return pickle.dumps(data).hex()
    
    def _deserialize(self, data: str) -> Any:
        """Deserialize data from Redis"""
        if not data:
            return None
        try:
            # Try JSON first
            return json.loads(data)
        except (json.JSONDecodeError, ValueError):
            # Fall back to pickle
            try:
                return pickle.loads(bytes.fromhex(data))
            except:
                return None
    
    def get(self, cache_type: str, identifier: str) -> Optional[Any]:
        """Get item from cache"""
        if not self.redis_client:
            return None
        
        try:
            key = self._generate_key(cache_type, identifier)
            data = self.redis_client.get(key)
            if data:
                logger.debug(f"Cache hit: {key}")
                return self._deserialize(data)
            else:
                logger.debug(f"Cache miss: {key}")
                return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(self, cache_type: str, identifier: str, data: Any, ttl: Optional[int] = None) -> bool:
        """Set item in cache with TTL"""
        if not self.redis_client:
            return False
        
        try:
            key = self._generate_key(cache_type, identifier)
            serialized_data = self._serialize(data)
            
            # Use cache_type specific TTL or provided TTL
            cache_ttl = ttl or self.ttl_settings.get(cache_type, self.default_ttl)
            
            self.redis_client.setex(key, cache_ttl, serialized_data)
            logger.debug(f"Cache set: {key} (TTL: {cache_ttl}s)")
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def delete(self, cache_type: str, identifier: str) -> bool:
        """Delete item from cache"""
        if not self.redis_client:
            return False
        
        try:
            key = self._generate_key(cache_type, identifier)
            result = self.redis_client.delete(key)
            logger.debug(f"Cache delete: {key}")
            return result > 0
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    def get_multi(self, keys: List[tuple]) -> Dict[str, Any]:
        """Get multiple items from cache"""
        if not self.redis_client:
            return {}
        
        try:
            cache_keys = [self._generate_key(cache_type, identifier) for cache_type, identifier in keys]
            values = self.redis_client.mget(cache_keys)
            
            result = {}
            for i, (cache_type, identifier) in enumerate(keys):
                if values[i]:
                    result[f"{cache_type}:{identifier}"] = self._deserialize(values[i])
            
            return result
        except Exception as e:
            logger.error(f"Cache get_multi error: {e}")
            return {}
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern"""
        if not self.redis_client:
            return 0
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"Invalidated {deleted} keys matching pattern: {pattern}")
                return deleted
            return 0
        except Exception as e:
            logger.error(f"Cache invalidate_pattern error: {e}")
            return 0
    
    def invalidate_trades(self):
        """Invalidate all trade-related caches"""
        patterns = [
            'qnti:trades:*',
            'qnti:pnl:*',
            'qnti:health:*',
            'qnti:ea:*'
        ]
        total_deleted = 0
        for pattern in patterns:
            total_deleted += self.invalidate_pattern(pattern)
        return total_deleted
    
    def clear_all(self) -> bool:
        """Clear all cache entries"""
        if not self.redis_client:
            return False
        
        try:
            self.redis_client.flushdb()
            logger.info("All cache entries cleared")
            return True
        except Exception as e:
            logger.error(f"Cache clear_all error: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.redis_client:
            return {'status': 'disabled'}
        
        try:
            info = self.redis_client.info()
            return {
                'status': 'active',
                'connected_clients': info.get('connected_clients', 0),
                'used_memory': info.get('used_memory_human', '0B'),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'uptime_in_seconds': info.get('uptime_in_seconds', 0)
            }
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def cache_decorator(self, cache_type: str, ttl: Optional[int] = None):
        """Decorator for automatic caching of function results"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Create cache key from function name and arguments
                cache_key = f"{func.__name__}:{hashlib.md5(str(args).encode() + str(kwargs).encode()).hexdigest()}"
                
                # Try to get from cache
                cached_result = self.get(cache_type, cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.set(cache_type, cache_key, result, ttl)
                return result
            return wrapper
        return decorator

# Global cache instance
cache = QNTIRedisCache()

# Cached wrapper classes
class CachedMT5Bridge:
    """Cached wrapper for MT5 bridge operations"""
    
    def __init__(self, mt5_bridge):
        self.mt5_bridge = mt5_bridge
    
    def get_account_info(self):
        """Get account info with caching"""
        cached = cache.get('account', 'info')
        if cached:
            return cached
        
        try:
            account_info = self.mt5_bridge.get_account_info()
            cache.set('account', 'info', account_info)
            return account_info
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}
    
    def get_open_positions(self):
        """Get open positions with caching"""
        cached = cache.get('trades', 'open_positions')
        if cached:
            return cached
        
        try:
            positions = self.mt5_bridge.get_open_positions()
            cache.set('trades', 'open_positions', positions)
            return positions
        except Exception as e:
            logger.error(f"Error getting open positions: {e}")
            return []
    
    def get_mt5_status(self):
        """Get MT5 status with caching"""
        cached = cache.get('system', 'mt5_status')
        if cached:
            return cached
        
        try:
            status = self.mt5_bridge.get_mt5_status()
            cache.set('system', 'mt5_status', status)
            return status
        except Exception as e:
            logger.error(f"Error getting MT5 status: {e}")
            return {'connected': False, 'error': str(e)}

class CachedTradeManager:
    """Cached wrapper for trade manager operations"""
    
    def __init__(self, trade_manager):
        self.trade_manager = trade_manager
    
    def get_active_trades(self):
        """Get active trades with caching"""
        cached = cache.get('trades', 'active')
        if cached:
            return cached
        
        try:
            active_trades = []
            for trade in self.trade_manager.trades.values():
                if trade.status.value == 'open':
                    trade_dict = {
                        "id": trade.trade_id,
                        "ticket": trade.trade_id.replace("MT5_", "") if trade.trade_id.startswith("MT5_") else trade.trade_id,
                        "ea_name": trade.ea_name,
                        "symbol": trade.symbol,
                        "type": trade.trade_type,
                        "volume": trade.lot_size,
                        "open_price": trade.open_price,
                        "entry_price": trade.open_price,
                        "current_price": trade.close_price or trade.open_price,
                        "profit": trade.profit or 0.0,
                        "swap": trade.swap or 0.0,
                        "commission": trade.commission or 0.0,
                        "open_time": trade.open_time.isoformat() if trade.open_time else None,
                        "status": trade.status.value,
                        "stop_loss": trade.stop_loss,
                        "take_profit": trade.take_profit,
                        "magic_number": trade.magic_number,
                        "comment": trade.notes or ""
                    }
                    active_trades.append(trade_dict)
            
            cache.set('trades', 'active', active_trades, ttl=10)  # Cache for 10 seconds
            return active_trades
        except Exception as e:
            logger.error(f"Error getting active trades: {e}")
            return []
    
    def calculate_statistics(self):
        """Calculate statistics with caching"""
        cached = cache.get('pnl', 'statistics')
        if cached:
            return cached
        
        try:
            stats = self.trade_manager.calculate_statistics()
            cache.set('pnl', 'statistics', stats)
            return stats
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            return {
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0
            }
    
    def get_ea_performance(self):
        """Get EA performance with caching"""
        cached = cache.get('ea_stats', 'performance')
        if cached:
            return cached
        
        try:
            performance = self.trade_manager.get_ea_performance()
            cache.set('ea_stats', 'performance', performance)
            return performance
        except Exception as e:
            logger.error(f"Error getting EA performance: {e}")
            return {}

# System health caching functions
def get_cached_system_health():
    """Get cached system health"""
    return cache.get('health', 'system')

def cache_system_health(health_data):
    """Cache system health data"""
    cache.set('health', 'system', health_data)

# Flask integration helpers
def integrate_cache_with_flask(app, main_system):
    """Integrate cache with Flask app"""
    
    @app.route('/api/cache/stats')
    def cache_stats():
        """Get cache statistics"""
        try:
            return cache.get_stats()
        except Exception as e:
            return {'error': str(e)}, 500
    
    @app.route('/api/cache/clear', methods=['POST'])
    def clear_cache():
        """Clear all cache"""
        try:
            cache.clear_all()
            return {'message': 'Cache cleared successfully'}
        except Exception as e:
            return {'error': str(e)}, 500
    
    @app.route('/api/cache/invalidate/trades', methods=['POST'])
    def invalidate_trades():
        """Invalidate trade-related caches"""
        try:
            deleted = cache.invalidate_trades()
            return {'message': f'Invalidated {deleted} cache entries'}
        except Exception as e:
            return {'error': str(e)}, 500

# Utility functions
def warm_cache(main_system):
    """Pre-warm cache with expensive operations"""
    logger.info("Warming cache...")
    
    try:
        # Create cached wrappers
        cached_mt5 = CachedMT5Bridge(main_system.mt5_bridge)
        cached_trade_manager = CachedTradeManager(main_system.trade_manager)
        
        # Pre-load expensive operations
        cached_mt5.get_account_info()
        cached_mt5.get_open_positions()
        cached_mt5.get_mt5_status()
        cached_trade_manager.calculate_statistics()
        cached_trade_manager.get_ea_performance()
        
        logger.info("Cache warmed successfully")
    except Exception as e:
        logger.error(f"Error warming cache: {e}")

def setup_cache_middleware(app):
    """Setup cache middleware for Flask"""
    from flask import request
    
    @app.before_request
    def before_request():
        """Set up request timing"""
        import time
        request.start_time = time.time()
    
    @app.after_request
    def after_request(response):
        """Log request performance"""
        if hasattr(request, 'start_time'):
            duration = time.time() - request.start_time
            if duration > 1.0:  # Log slow requests
                logger.warning(f"Slow request: {request.path} took {duration:.3f}s")
        return response

# Cache warming decorator
def cache_warm_on_startup(func):
    """Decorator to warm cache on application startup"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        # Warm cache after startup
        if hasattr(args[0], 'main_system'):
            warm_cache(args[0].main_system)
        return result
    return wrapper

logger.info("QNTI Redis Cache module loaded successfully") 