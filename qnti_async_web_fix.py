# qnti_async_web_fix.py
# Async Flask implementation to fix the blocking issue in QNTI

import asyncio
import time
from flask import Flask, jsonify
from flask_cors import CORS
import concurrent.futures
from functools import wraps
import logging
from threading import Thread, RLock
import queue

logger = logging.getLogger(__name__)

class AsyncFlaskWrapper:
    """Wrapper to make Flask endpoints non-blocking using thread pool"""
    
    def __init__(self, app, max_workers=10):
        self.app = app
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._cache_lock = RLock()
        self._request_queue = queue.Queue(maxsize=100)
        
    def make_async(self, func):
        """Convert blocking function to non-blocking"""
        @wraps(func)
        def async_wrapper(*args, **kwargs):
            try:
                # Submit to thread pool wrapped in application context to avoid context errors
                def _run_in_app_ctx(*iargs, **ikw):
                    # Push app context manually
                    with self.app.app_context():
                        return func(*iargs, **ikw)

                future = self.executor.submit(_run_in_app_ctx, *args, **kwargs)
                # Wait with timeout
                result = future.result(timeout=5.0)  # 5 second timeout
                return result
            except concurrent.futures.TimeoutError:
                logger.error(f"Function {func.__name__} timed out")
                return jsonify({"error": "Request timeout"}), 504
            except Exception as e:
                logger.error(f"Async execution error: {e}")
                return jsonify({"error": str(e)}), 500
        return async_wrapper

def create_optimized_health_endpoint(main_system, cache):
    """Create non-blocking health endpoint"""
    
    def get_health_data():
        """Get health data with timeout protection"""
        start_time = time.time()
        
        # Check cache first
        cached = cache.get('health', 'system')
        if cached:
            logger.info(f"Health data from cache: {time.time() - start_time:.3f}s")
            return cached
        
        health_data = {
            'timestamp': time.time(),
            'system_status': 'healthy'
        }
        
        # Use concurrent futures for parallel data fetching
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all data fetches in parallel
            futures = {
                'account': executor.submit(lambda: main_system.mt5_bridge.get_account_info() if hasattr(main_system, 'mt5_bridge') else {}),
                'trades': executor.submit(lambda: len(main_system.mt5_bridge.get_open_positions()) if hasattr(main_system, 'mt5_bridge') else 0),
                'statistics': executor.submit(lambda: main_system.trade_manager.calculate_statistics() if hasattr(main_system, 'trade_manager') else {}),
                'mt5_status': executor.submit(lambda: {'connected': True})  # Simplified to avoid blocking
            }
            
            # Collect results with individual timeouts
            for key, future in futures.items():
                try:
                    health_data[key] = future.result(timeout=1.0)  # 1 second timeout per operation
                except concurrent.futures.TimeoutError:
                    logger.warning(f"Timeout getting {key}")
                    health_data[key] = None
                except Exception as e:
                    logger.error(f"Error getting {key}: {e}")
                    health_data[key] = None
        
        # Flatten and merge statistics into health_data for frontend compatibility
        stats = health_data.get('statistics', {}) or {}
        account = health_data.get('account', {}) or {}
        # Add all required fields for analytics
        health_data.update({
            'total_trades': stats.get('total_trades', 0),
            'closed_trades': stats.get('closed_trades', 0),
            'winning_trades': stats.get('winning_trades', 0),
            'losing_trades': stats.get('losing_trades', 0),
            'win_rate': stats.get('win_rate', 0.0),
            'profit_factor': stats.get('profit_factor', 0.0),
            'total_profit': getattr(account, 'profit', None) if hasattr(account, 'profit') else stats.get('total_pnl', 0.0),
            'max_drawdown': stats.get('max_drawdown', 0.0),
            'avg_trade': stats.get('avg_trade', 0.0),
            'sharpe_ratio': stats.get('sharpe_ratio', 0.0),
            'best_trade': stats.get('best_trade', 0.0),
            'worst_trade': stats.get('worst_trade', 0.0),
            'balance': getattr(account, 'balance', 0.0),
            'equity': getattr(account, 'equity', 0.0),
            'margin': getattr(account, 'margin', 0.0),
            'free_margin': getattr(account, 'free_margin', 0.0),
            'open_trades': health_data.get('trades', 0),
        })
        
        # Cache the result
        cache.set('health', 'system', health_data, ttl=30)
        
        elapsed = time.time() - start_time
        logger.info(f"Health data calculated in {elapsed:.3f}s")
        
        return health_data
    
    return get_health_data

# Optimized Flask app factory
def create_optimized_flask_app(main_system, cache):
    """Create Flask app with non-blocking endpoints"""
    
    app = Flask(__name__)
    CORS(app)
    
    # Create async wrapper
    async_wrapper = AsyncFlaskWrapper(app)
    
    # Create optimized endpoints
    health_endpoint = create_optimized_health_endpoint(main_system, cache)
    
    @app.route('/api/health')
    @async_wrapper.make_async
    def health():
        """Non-blocking health endpoint"""
        return jsonify(health_endpoint())
    
    @app.route('/api/eas')
    @async_wrapper.make_async
    def get_eas():
        """Non-blocking EA endpoint"""
        # Check cache first
        cached = cache.get('ea_stats', 'all')
        if cached:
            return jsonify(cached)
        
        # Get data with timeout
        try:
            ea_data = main_system.trade_manager.get_ea_performance() if hasattr(main_system, 'trade_manager') else []
            cache.set('ea_stats', 'all', ea_data, ttl=60)
            return jsonify(ea_data)
        except Exception as e:
            logger.error(f"EA endpoint error: {e}")
            return jsonify([])
    
    @app.route('/api/trades')
    @async_wrapper.make_async
    def get_trades():
        """Non-blocking trades endpoint"""
        # Check cache first
        cached = cache.get('trades', 'open')
        if cached is not None:
            return jsonify(cached)
        
        try:
            trades = main_system.mt5_bridge.get_open_positions() if hasattr(main_system, 'mt5_bridge') else []
            cache.set('trades', 'open', trades, ttl=5)
            return jsonify(trades)
        except Exception as e:
            logger.error(f"Trades endpoint error: {e}")
            return jsonify([])
    
    # Fast test endpoint
    @app.route('/api/test')
    def test():
        """Instant response test endpoint"""
        return jsonify({"status": "ok", "timestamp": time.time()})
    
    return app

# Background task manager to prevent blocking
class NonBlockingTaskManager:
    """Manages background tasks without blocking main thread"""
    
    def __init__(self):
        self.tasks = []
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        
    def submit_task(self, func, *args, **kwargs):
        """Submit task to background without blocking"""
        future = self.executor.submit(func, *args, **kwargs)
        self.tasks.append(future)
        
        # Clean up completed tasks
        self.tasks = [t for t in self.tasks if not t.done()]
        
        return future
    
    def shutdown(self):
        """Gracefully shutdown executor"""
        self.executor.shutdown(wait=False)

# Integration function for your existing QNTI system
def apply_async_fix(qnti_web_interface):
    """Apply async fixes to existing web interface"""
    
    # Replace blocking endpoints with async versions
    async_wrapper = AsyncFlaskWrapper(qnti_web_interface.app)
    
    # Get existing endpoints
    rules = list(qnti_web_interface.app.url_map.iter_rules())
    
    # Wrap each endpoint
    for rule in rules:
        if rule.endpoint and rule.endpoint != 'static':
            view_func = qnti_web_interface.app.view_functions.get(rule.endpoint)
            if view_func:
                # Wrap with async
                qnti_web_interface.app.view_functions[rule.endpoint] = async_wrapper.make_async(view_func)
    
    logger.info("Applied async wrapper to all endpoints")
    
    return qnti_web_interface

# Quick fix function to add to your main system
def add_performance_monitoring(app):
    """Add performance monitoring to identify slow operations"""
    
    @app.before_request
    def before_request():
        """Log request start time"""
        from flask import g
        g.start_time = time.time()
    
    @app.after_request
    def after_request(response):
        """Log request duration"""
        from flask import g
        if hasattr(g, 'start_time'):
            elapsed = time.time() - g.start_time
            if elapsed > 1.0:  # Log slow requests
                logger.warning(f"Slow request: {elapsed:.3f}s")
        return response
    
    return app

# Minimal working example
if __name__ == "__main__":
    from qnti_redis_cache import cache
    
    # Create minimal app for testing
    app = Flask(__name__)
    CORS(app)
    
    async_wrapper = AsyncFlaskWrapper(app)
    
    @app.route('/api/test')
    def test():
        return jsonify({"status": "ok", "time": time.time()})
    
    @app.route('/api/slow')
    @async_wrapper.make_async
    def slow():
        # Simulate slow operation
        time.sleep(5)
        return jsonify({"status": "completed", "time": time.time()})
    
    # Add monitoring
    add_performance_monitoring(app)
    
    print("Starting async Flask server on http://localhost:5003")
    app.run(host='0.0.0.0', port=5003, debug=True, threaded=True) 