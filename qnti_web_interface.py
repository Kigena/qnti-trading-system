#!/usr/bin/env python3
"""
QNTI Web Interface - Flask Routes and WebSocket Handlers
Handles all web interface interactions for the QNTI system
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from flask import Flask, jsonify, request, render_template_string, render_template, send_from_directory
from flask_socketio import SocketIO, emit
import threading
import os
import dataclasses
import asyncio

# Redis caching imports
from qnti_redis_cache import (
    cache, 
    CachedMT5Bridge, 
    CachedTradeManager,
    get_cached_system_health,
    cache_system_health,
    integrate_cache_with_flask,
    warm_cache,
    setup_cache_middleware
)

# Import async fixes
from qnti_async_web_fix import AsyncFlaskWrapper, add_performance_monitoring
from qnti_advanced_analytics import QNTIAdvancedAnalytics

logger = logging.getLogger('QNTI_WEB')

class QNTIWebInterface:
    """QNTI Web Interface Handler"""
    
    def __init__(self, app: Flask, socketio: SocketIO, main_system):
        self.app = app
        self.socketio = socketio
        self.main_system = main_system
        self.start_time = time.time()  # Add start_time attribute for system health
        
        # Legacy cache management (will be replaced by Redis)
        self._ea_cache = None
        self._ea_cache_timestamp = 0
        self._cache_duration = 10  # seconds
        
        # Performance optimization caches
        self._profit_cache = None
        self._win_rate_cache = None
        
        # Redis cached wrappers for high performance
        self.cached_mt5 = CachedMT5Bridge(main_system.mt5_bridge) if main_system.mt5_bridge else None
        self.cached_trade_manager = CachedTradeManager(main_system.trade_manager) if main_system.trade_manager else None
        
        # Setup async wrapper for performance
        self.async_wrapper = AsyncFlaskWrapper(self.app, max_workers=15)
        
        # Setup cache middleware
        setup_cache_middleware(self.app)
        
        # Integrate cache with Flask (adds cache management routes)
        integrate_cache_with_flask(self.app, main_system)
        
        # Add performance monitoring
        add_performance_monitoring(self.app)
        
        # Initialize advanced analytics
        self.advanced_analytics = QNTIAdvancedAnalytics(main_system.trade_manager, main_system.mt5_bridge)
        
        self.setup_routes()
        self.setup_websocket_handlers()
        self.setup_ea_generation_routes()
        
        # Warm cache for faster initial responses
        if main_system:
            warm_cache(main_system)
    
    def _is_cache_valid(self, timestamp):
        """Check if cache is still valid"""
        if not timestamp:
            return False
        return (datetime.now() - timestamp).total_seconds() < self._cache_duration
    
    def _get_cached_account_info(self):
        """Get cached account info with fallback"""
        try:
            # Try cached MT5 account info first
            if self.cached_mt5:
                account_info = self.cached_mt5.get_account_info()
                if account_info:
                    return account_info
            
            # Fallback to direct MT5 bridge
            if self.main_system and self.main_system.mt5_bridge:
                mt5_status = self.main_system.mt5_bridge.get_mt5_status()
                return mt5_status.get('account_info', {})
            
            # Final fallback - return default values
            return {
                'balance': 2500.0,
                'equity': 2485.0,
                'margin': 150.0,
                'free_margin': 2335.0,
                'margin_level': 1656.67,
                'profit': -15.0
            }
        except Exception as e:
            logger.error(f"Error getting cached account info: {e}")
            return {
                'balance': 2500.0,
                'equity': 2485.0,
                'margin': 150.0,
                'free_margin': 2335.0,
                'margin_level': 1656.67,
                'profit': -15.0
            }
    
    def _get_ea_data_cached(self):
        """Get EA data with caching"""
        try:
            # Check cache validity
            if self._is_cache_valid(self._ea_cache_timestamp):
                return self._ea_cache
            
            # Cache expired or doesn't exist, refresh it with lightweight data
            ea_data = []
            
            if self.main_system and self.main_system.mt5_bridge:
                # Get only essential data for performance
                for ea_name, monitor in self.main_system.mt5_bridge.ea_monitors.items():
                    performance = self.main_system.trade_manager.ea_performances.get(ea_name)
                    
                    # Create lightweight EA info for list view
                    ea_info = {
                        "name": ea_name,
                        "magic_number": monitor.magic_number,
                        "symbol": monitor.symbol,
                        "status": "active" if monitor.is_active else "inactive",
                        "description": f"Magic: {monitor.magic_number} | Symbol: {monitor.symbol}"
                    }
                    
                    # Add performance data if available (simplified)
                    if performance:
                        # Handle infinity values in profit_factor
                        profit_factor = performance.profit_factor
                        if profit_factor == float('inf') or profit_factor == float('-inf'):
                            profit_factor = 999.99 if profit_factor == float('inf') else -999.99
                        
                        ea_info.update({
                            "total_trades": performance.total_trades,
                            "win_rate": round(performance.win_rate, 1),
                            "total_profit": round(performance.total_profit, 2),
                            "profit_factor": round(profit_factor, 2),
                            "max_drawdown": round(performance.max_drawdown, 2),
                            "risk_score": performance.risk_score,
                            "last_trade_time": performance.last_trade_time.isoformat() if performance.last_trade_time else None
                        })
                    else:
                        # Default values for EAs without performance data
                        ea_info.update({
                            "total_trades": 0,
                            "win_rate": 0.0,
                            "total_profit": 0.0,
                            "profit_factor": 0.0,
                            "max_drawdown": 0.0,
                            "risk_score": 0.0,
                            "last_trade_time": None
                        })
                    
                    ea_data.append(ea_info)
            
            # Update cache
            self._ea_cache = ea_data
            self._ea_cache_timestamp = datetime.now()
            logger.debug(f"Updated EA cache with {len(ea_data)} EAs (lightweight)")
            
            return ea_data
            
        except Exception as e:
            logger.error(f"Error getting EA data: {e}")
            return []
    
    def _invalidate_ea_cache(self):
        """Invalidate EA cache to force refresh"""
        self._ea_cache_timestamp = None
        self._ea_cache = {}
    
    def _load_ea_profiles_by_magic_number(self):
        """Load EA profiles and index them by magic number for quick lookup"""
        try:
            import json
            from pathlib import Path
            
            profiles_dir = Path("ea_profiles")
            if not profiles_dir.exists():
                return {}
            
            magic_to_profile = {}
            
            for profile_file in profiles_dir.glob("*.json"):
                try:
                    with open(profile_file, 'r', encoding='utf-8') as f:
                        profile_data = json.load(f)
                    
                    # Extract magic numbers and create mapping
                    magic_numbers = profile_data.get('magic_numbers', [])
                    if isinstance(magic_numbers, list) and magic_numbers:
                        for magic_number in magic_numbers:
                            magic_to_profile[magic_number] = {
                                'name': profile_data.get('name', 'Unknown EA'),
                                'timeframes': profile_data.get('timeframes', ['CURRENT']),
                                'symbols': profile_data.get('symbols', ['CURRENT']),
                                'description': profile_data.get('description', ''),
                                'is_portfolio': profile_data.get('is_portfolio', False),
                                'strategies': profile_data.get('strategies', [])
                            }
                    
                    # Also try magic_number field (single value)
                    if 'magic_number' in profile_data:
                        magic_number = profile_data['magic_number']
                        magic_to_profile[magic_number] = {
                            'name': profile_data.get('name', 'Unknown EA'),
                            'timeframes': profile_data.get('timeframes', ['CURRENT']),
                            'symbols': profile_data.get('symbols', ['CURRENT']),
                            'description': profile_data.get('description', ''),
                            'is_portfolio': profile_data.get('is_portfolio', False),
                            'strategies': profile_data.get('strategies', [])
                        }
                    
                except Exception as e:
                    logger.warning(f'Could not load EA profile {profile_file}: {e}')
                    continue
            
            logger.info(f"Loaded {len(magic_to_profile)} EA profiles indexed by magic number")
            return magic_to_profile
            
        except Exception as e:
            logger.error(f"Error loading EA profiles by magic number: {e}")
            return {}

    def setup_routes(self):
        """Setup all web routes"""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard page"""
            try:
                from pathlib import Path
                dashboard_file = Path("qnti_dashboard.html")
                if dashboard_file.exists():
                    with open(dashboard_file, 'r', encoding='utf-8') as f:
                        return f.read()
                else:
                    return f"<h1>Error: Dashboard not found</h1><p>Please ensure qnti_dashboard.html exists in the root directory.</p>", 404
            except Exception as e:
                return f"<h1>Error loading dashboard</h1><p>{str(e)}</p>", 500
        
        @self.app.route('/dashboard/main_dashboard.html')
        @self.app.route('/main_dashboard.html')
        @self.app.route('/overview')
        @self.app.route('/dashboard/overview')
        def main_dashboard_page():
            """Main dashboard page"""
            try:
                from pathlib import Path
                dashboard_file = Path("dashboard/main_dashboard.html")
                if dashboard_file.exists():
                    with open(dashboard_file, 'r', encoding='utf-8') as f:
                        return f.read()
                else:
                    return f"<h1>Error: main_dashboard.html not found</h1><p>Please ensure the file exists in the dashboard/ directory.</p>", 404
            except Exception as e:
                return f"<h1>Error loading main dashboard</h1><p>{str(e)}</p>", 500
        
        @self.app.route('/dashboard/trading_center.html')
        @self.app.route('/trading_center.html')
        def trading_center():
            """Trading center page"""
            try:
                from pathlib import Path
                dashboard_file = Path("dashboard/trading_center.html")
                if dashboard_file.exists():
                    with open(dashboard_file, 'r', encoding='utf-8') as f:
                        return f.read()
                else:
                    return f"<h1>Error: trading_center.html not found</h1><p>Please ensure the file exists in the dashboard/ directory.</p>", 404
            except Exception as e:
                return f"<h1>Error loading trading center</h1><p>{str(e)}</p>", 500
        
        @self.app.route('/dashboard/ea_management.html')
        @self.app.route('/ea_management.html')
        def ea_management():
            """EA management page"""
            try:
                from pathlib import Path
                dashboard_file = Path("dashboard/ea_management.html")
                if dashboard_file.exists():
                    with open(dashboard_file, 'r', encoding='utf-8') as f:
                        return f.read()
                else:
                    return f"<h1>Error: ea_management.html not found</h1><p>Please ensure the file exists in the dashboard/ directory.</p>", 404
            except Exception as e:
                return f"<h1>Error loading EA management</h1><p>{str(e)}</p>", 500
        
        @self.app.route('/dashboard/analytics_reports.html')
        @self.app.route('/analytics_reports.html')
        def analytics_reports():
            """Analytics reports page"""
            try:
                from pathlib import Path
                dashboard_file = Path("dashboard/analytics_reports.html")
                if dashboard_file.exists():
                    with open(dashboard_file, 'r', encoding='utf-8') as f:
                        return f.read()
                else:
                    return f"<h1>Error: analytics_reports.html not found</h1><p>Please ensure the file exists in the dashboard/ directory.</p>", 404
            except Exception as e:
                return f"<h1>Error loading analytics reports</h1><p>{str(e)}</p>", 500

        @self.app.route('/api/system/health')
        @self.app.route('/api/health')  # Add compatibility route
        @self.async_wrapper.make_async
        def system_health():
            """Get comprehensive system health status"""
            try:
                start_time = time.time()
                
                # Get cached account info (cached for 30 seconds)
                account_info = self._get_cached_account_info()
                
                # Get EA count from performance data
                ea_count = 0
                if self.cached_trade_manager:
                    ea_performance_data = self.cached_trade_manager.get_ea_performance()
                    ea_count = len(ea_performance_data) if ea_performance_data else 0
                elif hasattr(self.main_system, 'trade_manager') and self.main_system.trade_manager:
                    ea_count = len(self.main_system.trade_manager.ea_performances)
                
                # Get active trades count
                active_trades_count = 0
                if self.cached_trade_manager:
                    trades_data = self.cached_trade_manager.get_active_trades()
                    active_trades_count = len(trades_data) if trades_data else 0
                elif hasattr(self.main_system, 'trade_manager') and self.main_system.trade_manager:
                    active_trades_count = len([t for t in self.main_system.trade_manager.trades.values() 
                                             if t.status.value == 'open'])
                
                # System status
                system_status = {
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat(),
                    "uptime": time.time() - self.start_time,
                    "mt5_connection": True,  # Assume connected for now
                    "auto_trading": account_info.get('auto_trading', False),
                    "account": {
                        "balance": account_info.get('balance', 0.0),
                        "equity": account_info.get('equity', 0.0),
                        "margin": account_info.get('margin', 0.0),
                        "free_margin": account_info.get('free_margin', 0.0),
                        "margin_level": account_info.get('margin_level', 0.0),
                        "profit": account_info.get('profit', 0.0),
                        "currency": account_info.get('currency', 'USD'),
                        "server": account_info.get('server', 'Unknown'),
                        "leverage": account_info.get('leverage', 1),
                        "account_number": account_info.get('account_number', 0)
                    },
                    "statistics": {
                        "total_eas": ea_count,
                        "active_trades": active_trades_count,
                        "total_trades": self._get_total_trades_count(),
                        "daily_profit": self._get_daily_profit(),
                        "win_rate": self._get_win_rate(),
                        "best_trade": self._get_best_trade(),
                        "worst_trade": self._get_worst_trade(),
                        "avg_trade": self._get_avg_trade(),
                        "sharpe_ratio": self._get_sharpe_ratio()
                    },
                    "performance": {
                        "cpu_usage": 0.0,  # Could be added with psutil
                        "memory_usage": 0.0,  # Could be added with psutil
                        "disk_usage": 0.0   # Could be added with psutil
                    }
                }
                
                # Check for any issues
                if account_info.get('equity', 0) < account_info.get('balance', 0) * 0.8:
                    system_status["status"] = "warning"
                    system_status["warnings"] = ["High drawdown detected"]
                
                if account_info.get('margin_level', 1000) < 200:
                    system_status["status"] = "critical"
                    system_status["alerts"] = ["Low margin level - risk of margin call"]
                
                elapsed = time.time() - start_time
                logger.info(f"System health check completed in {elapsed:.3f}s")
                
                return jsonify(system_status)
                
            except Exception as e:
                logger.error(f"Error getting system health: {e}")
                return jsonify({
                    "status": "error",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                }), 500
        
        @self.app.route('/api/trades/active')
        @self.app.route('/api/trades')  # Add compatibility route
        @self.async_wrapper.make_async
        def active_trades():
            """Get active trades with enhanced caching"""
            try:
                start_time = time.time()
                
                # Try cached trades first (cached for 10 seconds)
                if self.cached_trade_manager:
                    trades_data = self.cached_trade_manager.get_active_trades()
                    
                    if trades_data:
                        elapsed = time.time() - start_time
                        logger.info(f"Retrieved {len(trades_data)} active trades from cache in {elapsed:.3f}s")
                        return jsonify(trades_data)
                
                # Fallback to original method if cache not available
                trades = []
                
                if hasattr(self.main_system, 'trade_manager') and self.main_system.trade_manager:
                    for trade in self.main_system.trade_manager.trades.values():
                        if trade.status.value == 'open':
                            trade_dict = {
                                "id": trade.id,
                                "ea_name": trade.ea_name,
                                "symbol": trade.symbol,
                                "type": trade.type.value,
                                "volume": trade.volume,
                                "open_price": trade.open_price,
                                "current_price": trade.current_price,
                                "profit": trade.profit,
                                "swap": trade.swap,
                                "commission": trade.commission,
                                "open_time": trade.open_time.isoformat() if trade.open_time else None,
                                "duration": str(trade.duration) if trade.duration else None,
                                "status": trade.status.value,
                                "stop_loss": trade.stop_loss,
                                "take_profit": trade.take_profit,
                                "magic_number": trade.magic_number,
                                "comment": trade.comment
                            }
                            trades.append(trade_dict)
                
                elapsed = time.time() - start_time
                logger.info(f"Retrieved {len(trades)} active trades (uncached) in {elapsed:.3f}s")
                
                return jsonify(trades)
                
            except Exception as e:
                logger.error(f"Error getting active trades: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/eas')
        @self.async_wrapper.make_async
        def get_eas():
            """Get comprehensive EA data with Redis caching (60-second cache) and profile integration"""
            try:
                start_time = time.time()
                
                # Load EA profiles indexed by magic number
                magic_to_profile = self._load_ea_profiles_by_magic_number()
                
                # Try cached EA performance first (cached for 60 seconds)
                if self.cached_trade_manager:
                    ea_performance_data = self.cached_trade_manager.get_ea_performance()
                    
                    # Convert to the expected format
                    ea_data = []
                    def _get(val, key, default=None):
                        # Helper to fetch attribute regardless of dict or object
                        if isinstance(val, dict):
                            return val.get(key, default)
                        return getattr(val, key, default)

                    for ea_name, performance in ea_performance_data.items():
                        # Handle infinity values in profit_factor
                        profit_factor_val = _get(performance, 'profit_factor', 0.0)
                        if profit_factor_val in (float('inf'), float('-inf')):
                            profit_factor_val = 999.99 if profit_factor_val == float('inf') else -999.99

                        # Get profile data if available
                        magic_number = _get(performance, 'magic_number', 0)
                        profile_data = magic_to_profile.get(magic_number, {})
                        
                        # Use profile name if available, otherwise use performance name
                        display_name = profile_data.get('name', ea_name)
                        
                        # Use profile timeframes if available, otherwise use 'CURRENT'
                        timeframes = profile_data.get('timeframes', ['CURRENT'])
                        timeframe_str = timeframes[0] if timeframes else 'CURRENT'
                        
                        # Enhanced description with timeframe
                        description = f"Magic: {magic_number} | Symbol: {_get(performance, 'symbol', 'UNKNOWN')}"
                        if timeframe_str != 'CURRENT':
                            description += f" | TF: {timeframe_str}"

                        ea_info = {
                            "name": display_name,
                            "original_name": ea_name,  # Keep original for compatibility
                            "symbol": _get(performance, 'symbol', 'UNKNOWN'),
                            "magic_number": magic_number,
                            "timeframe": timeframe_str,
                            "status": "active" if str(_get(performance, 'status', 'inactive')).lower() == "active" else "inactive",
                            "description": description,
                            "total_trades": _get(performance, 'total_trades', 0),
                            "win_rate": round(_get(performance, 'win_rate', 0.0), 1),
                            "total_profit": round(_get(performance, 'total_profit', 0.0), 2),
                            "total_loss": round(_get(performance, 'total_loss', 0.0), 2),
                            "profit_factor": round(profit_factor_val, 2),
                            "max_drawdown": round(_get(performance, 'max_drawdown', 0.0), 2),
                            "avg_trade": round(_get(performance, 'avg_trade', 0.0), 2),
                            "risk_score": _get(performance, 'risk_score', 0.0),
                            "last_trade_time": _get(performance, 'last_trade_time', None),
                            # Profile-specific fields
                            "is_portfolio": profile_data.get('is_portfolio', False),
                            "strategies": profile_data.get('strategies', [])
                        }
                        
                        # Convert datetime to ISO format
                        if ea_info["last_trade_time"] and hasattr(ea_info["last_trade_time"], 'isoformat'):
                            ea_info["last_trade_time"] = ea_info["last_trade_time"].isoformat()
                        elif ea_info["last_trade_time"]:
                            ea_info["last_trade_time"] = str(ea_info["last_trade_time"])
                        
                        ea_data.append(ea_info)
                    
                    elapsed = time.time() - start_time
                    logger.info(f"Retrieved {len(ea_data)} EAs from cached performance data with profile integration in {elapsed:.3f}s")
                    return jsonify(ea_data)
                
                else:
                    # Fallback to original method if cache not available
                    ea_data = []
                    
                    # Get EA performance data (this has 32 EAs)
                    if hasattr(self.main_system, 'trade_manager') and self.main_system.trade_manager:
                        for ea_name, performance in self.main_system.trade_manager.ea_performances.items():
                            # Handle infinity values in profit_factor
                            profit_factor = performance.profit_factor
                            if profit_factor == float('inf') or profit_factor == float('-inf'):
                                profit_factor = 999.99 if profit_factor == float('inf') else -999.99
                            
                            # Get profile data if available
                            magic_number = performance.magic_number
                            profile_data = magic_to_profile.get(magic_number, {})
                            
                            # Use profile name if available, otherwise use performance name
                            display_name = profile_data.get('name', ea_name)
                            
                            # Use profile timeframes if available, otherwise use 'CURRENT'
                            timeframes = profile_data.get('timeframes', ['CURRENT'])
                            timeframe_str = timeframes[0] if timeframes else 'CURRENT'
                            
                            # Enhanced description with timeframe
                            description = f"Magic: {magic_number} | Symbol: {performance.symbol}"
                            if timeframe_str != 'CURRENT':
                                description += f" | TF: {timeframe_str}"
                            
                            ea_info = {
                                "name": display_name,
                                "original_name": ea_name,  # Keep original for compatibility
                                "symbol": performance.symbol,
                                "magic_number": magic_number,
                                "timeframe": timeframe_str,
                                "status": "active" if performance.status.value == "active" else "inactive",
                                "description": description,
                                "total_trades": performance.total_trades,
                                "win_rate": round(performance.win_rate, 1),
                                "total_profit": round(performance.total_profit, 2),
                                "total_loss": round(performance.total_loss, 2),
                                "profit_factor": round(profit_factor, 2),
                                "max_drawdown": round(performance.max_drawdown, 2),
                                "risk_score": performance.risk_score,
                                "last_trade_time": performance.last_trade_time.isoformat() if performance.last_trade_time else None,
                                # Profile-specific fields
                                "is_portfolio": profile_data.get('is_portfolio', False),
                                "strategies": profile_data.get('strategies', [])
                            }
                            ea_data.append(ea_info)
                    
                    elapsed = time.time() - start_time
                    logger.info(f"Retrieved {len(ea_data)} EAs from performance data with profile integration (uncached) in {elapsed:.3f}s")
                    return jsonify(ea_data)
                    
            except Exception as e:
                logger.error(f"Error getting EAs: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/eas/register', methods=['POST'])
        def register_ea():
            """Register a new EA for monitoring"""
            try:
                data = request.get_json()
                
                # Validate required fields
                required_fields = ['name', 'magic_number', 'symbol']
                for field in required_fields:
                    if field not in data:
                        return jsonify({"error": f"Missing required field: {field}"}), 400
                
                ea_name = data['name']
                magic_number = int(data['magic_number'])
                symbol = data['symbol']
                timeframe = data.get('timeframe', 'M1')
                log_file = data.get('log_file', None)
                
                # Check if EA already exists
                if (self.main_system.mt5_bridge and 
                    ea_name in self.main_system.mt5_bridge.ea_monitors):
                    return jsonify({"error": "EA already registered"}), 400
                
                # Register EA
                if self.main_system.mt5_bridge:
                    self.main_system.mt5_bridge.register_ea_monitor(
                        ea_name, magic_number, symbol, timeframe, log_file
                    )
                    
                    logger.info(f"EA {ea_name} registered successfully")
                    return jsonify({
                        "message": f"EA {ea_name} registered successfully",
                        "ea_name": ea_name,
                        "magic_number": magic_number,
                        "symbol": symbol,
                        "timeframe": timeframe
                    })
                else:
                    return jsonify({"error": "MT5 bridge not available"}), 500
                    
            except Exception as e:
                logger.error(f"Error registering EA: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/eas/<ea_name>/control', methods=['POST'])
        def control_ea(ea_name):
            """Control EA (start, stop, pause, resume)"""
            try:
                data = request.get_json()
                action = data.get('action', '').lower()
                
                if action not in ['start', 'stop', 'pause', 'resume', 'restart']:
                    return jsonify({"error": "Invalid action"}), 400
                
                # Check if EA exists
                if (not self.main_system.mt5_bridge or 
                    ea_name not in self.main_system.mt5_bridge.ea_monitors):
                    return jsonify({"error": "EA not found"}), 404
                
                # Apply control action
                monitor = self.main_system.mt5_bridge.ea_monitors[ea_name]
                
                if action == 'start':
                    monitor.is_active = True
                    message = f"EA {ea_name} started"
                elif action == 'stop':
                    monitor.is_active = False
                    message = f"EA {ea_name} stopped"
                elif action == 'pause':
                    monitor.is_active = False
                    message = f"EA {ea_name} paused"
                elif action == 'resume':
                    monitor.is_active = True
                    message = f"EA {ea_name} resumed"
                elif action == 'restart':
                    monitor.is_active = False
                    time.sleep(1)  # Brief pause
                    monitor.is_active = True
                    message = f"EA {ea_name} restarted"
                
                logger.info(f"EA control: {message}")
                return jsonify({
                    "message": message,
                    "ea_name": ea_name,
                    "action": action,
                    "status": "active" if monitor.is_active else "inactive"
                })
                
            except Exception as e:
                logger.error(f"Error controlling EA {ea_name}: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/eas/bulk-control', methods=['POST'])
        def bulk_control_eas():
            """Apply optimization controls to multiple EAs based on filters"""
            try:
                data = request.get_json()
                action = data.get('action', '').lower()
                ea_filter = data.get('ea_filter', {})
                parameters = data.get('parameters', {})
                
                if not action:
                    return jsonify({"error": "Missing action parameter"}), 400
                
                # Get list of EAs that match the filter
                affected_eas = []
                
                if self.main_system.trade_manager:
                    for ea_name, performance in self.main_system.trade_manager.ea_performances.items():
                        # Apply filters
                        if ea_filter.get('status') and performance.status.value != ea_filter['status']:
                            continue
                        if ea_filter.get('symbol') and performance.symbol != ea_filter['symbol']:
                            continue
                        if ea_filter.get('type'):
                            # Simple strategy type detection based on EA name
                            ea_name_lower = ea_name.lower()
                            if ea_filter['type'] == 'high_frequency' and 'scalp' not in ea_name_lower and 'hf' not in ea_name_lower:
                                continue
                            if ea_filter['type'] == 'trend_following' and 'trend' not in ea_name_lower:
                                continue
                            if ea_filter['type'] == 'grid' and 'grid' not in ea_name_lower:
                                continue
                        
                        affected_eas.append(ea_name)
                
                # Apply the optimization action
                successful_applications = 0
                failed_applications = []
                
                for ea_name in affected_eas:
                    try:
                        if action == 'reduce_risk':
                            # Apply risk reduction parameters
                            control_params = {
                                'lot_multiplier': parameters.get('lot_multiplier', 0.7),
                                'max_spread': parameters.get('max_spread', 2.0),
                                'risk_level': 'reduced'
                            }
                            success = self.main_system.trade_manager.control_ea(ea_name, 'optimize', control_params)
                            
                        elif action == 'optimize_trend':
                            # Optimize trend following parameters
                            control_params = {
                                'trend_sensitivity': parameters.get('trend_sensitivity', 1.2),
                                'stop_loss_multiplier': parameters.get('stop_loss_multiplier', 1.1),
                                'optimization_type': 'trend'
                            }
                            success = self.main_system.trade_manager.control_ea(ea_name, 'optimize', control_params)
                            
                        elif action == 'adjust_grid':
                            # Adjust grid trading parameters
                            control_params = {
                                'grid_spacing_multiplier': parameters.get('grid_spacing_multiplier', 1.3),
                                'optimization_type': 'grid'
                            }
                            success = self.main_system.trade_manager.control_ea(ea_name, 'optimize', control_params)
                            
                        elif action == 'optimize_risk':
                            # Apply general risk optimization
                            control_params = {
                                'max_risk_per_trade': parameters.get('max_risk_per_trade', 0.02),
                                'max_total_risk': parameters.get('max_total_risk', 0.10),
                                'optimization_type': 'risk'
                            }
                            success = self.main_system.trade_manager.control_ea(ea_name, 'optimize', control_params)
                            
                        elif action == 'optimize_strategy':
                            # Optimize entry/exit strategy
                            control_params = {
                                'optimize_entry': parameters.get('optimize_entry', True),
                                'optimize_exit': parameters.get('optimize_exit', True),
                                'optimization_type': 'strategy'
                            }
                            success = self.main_system.trade_manager.control_ea(ea_name, 'optimize', control_params)
                            
                        elif action == 'general_optimize':
                            # General optimization
                            control_params = {
                                'auto_adjust': parameters.get('auto_adjust', True),
                                'market_adaptive': parameters.get('market_adaptive', True),
                                'optimization_type': 'general'
                            }
                            success = self.main_system.trade_manager.control_ea(ea_name, 'optimize', control_params)
                            
                        else:
                            # Unknown action, apply as-is
                            success = self.main_system.trade_manager.control_ea(ea_name, action, parameters)
                        
                        if success:
                            successful_applications += 1
                        else:
                            failed_applications.append(ea_name)
                            
                    except Exception as e:
                        logger.error(f"Error applying {action} to EA {ea_name}: {e}")
                        failed_applications.append(f"{ea_name} ({str(e)})")
                
                # Invalidate EA cache to ensure fresh data
                self._invalidate_ea_cache()
                
                logger.info(f"Bulk control '{action}' applied: {successful_applications} successful, {len(failed_applications)} failed")
                
                return jsonify({
                    "success": True,
                    "action": action,
                    "affected_eas": len(affected_eas),
                    "successful_applications": successful_applications,
                    "failed_applications": failed_applications,
                    "message": f"Applied {action} to {successful_applications} EAs"
                })
                
            except Exception as e:
                logger.error(f"Error in bulk EA control: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/eas/<ea_name>', methods=['DELETE'])
        def unregister_ea(ea_name):
            """Unregister an EA"""
            try:
                # Check if EA exists
                if (not self.main_system.mt5_bridge or 
                    ea_name not in self.main_system.mt5_bridge.ea_monitors):
                    return jsonify({"error": "EA not found"}), 404
                
                # Remove EA monitor
                del self.main_system.mt5_bridge.ea_monitors[ea_name]
                
                # Remove EA performance data
                if ea_name in self.main_system.trade_manager.ea_performances:
                    del self.main_system.trade_manager.ea_performances[ea_name]
                
                logger.info(f"EA {ea_name} unregistered successfully")
                return jsonify({"message": f"EA {ea_name} unregistered successfully"})
                
            except Exception as e:
                logger.error(f"Error unregistering EA {ea_name}: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/eas/<ea_name>/details')
        def get_ea_details(ea_name):
            """Get detailed EA information"""
            try:
                # Check if EA exists
                if (not self.main_system.mt5_bridge or 
                    ea_name not in self.main_system.mt5_bridge.ea_monitors):
                    return jsonify({"error": "EA not found"}), 404
                
                monitor = self.main_system.mt5_bridge.ea_monitors[ea_name]
                performance = self.main_system.trade_manager.ea_performances.get(ea_name)
                
                # Get EA trades
                ea_trades = [t for t in self.main_system.trade_manager.trades.values() 
                           if t.ea_name == ea_name]
                
                open_trades = [t for t in ea_trades if t.status.value == 'open']
                closed_trades = [t for t in ea_trades if t.status.value == 'closed']
                
                details = {
                    'name': ea_name,
                    'magic_number': monitor.magic_number,
                    'symbol': monitor.symbol,
                    'timeframe': monitor.timeframe,
                    'is_active': monitor.is_active,
                    'last_check': monitor.last_check.isoformat() if monitor.last_check else None,
                    'file_path': monitor.file_path,
                    'process_id': monitor.process_id,
                    'status': 'active' if monitor.is_active else 'inactive',
                    'trades': {
                        'total': len(ea_trades),
                        'open': len(open_trades),
                        'closed': len(closed_trades)
                    }
                }
                
                # Add performance data if available
                if performance:
                    # FIXED: Use property accessor for net_profit
                    try:
                        net_profit = getattr(performance, 'net_profit', performance.total_profit - performance.total_loss)
                    except AttributeError:
                        net_profit = performance.total_profit - performance.total_loss
                    
                    details['performance'] = {
                        'total_trades': performance.total_trades,
                        'winning_trades': performance.winning_trades,
                        'losing_trades': performance.losing_trades,
                        'win_rate': performance.win_rate,
                        'total_profit': performance.total_profit,
                        'total_loss': performance.total_loss,
                        'net_profit': net_profit,
                        'profit_factor': performance.profit_factor if performance.profit_factor != float('inf') else 999.99,
                        'max_drawdown': performance.max_drawdown,
                        'risk_score': performance.risk_score,
                        'last_trade_time': performance.last_trade_time.isoformat() if performance.last_trade_time else None
                    }
                else:
                    # Default performance values
                    details['performance'] = {
                        'total_trades': 0,
                        'winning_trades': 0,
                        'losing_trades': 0,
                        'win_rate': 0.0,
                        'total_profit': 0.0,
                        'total_loss': 0.0,
                        'net_profit': 0.0,
                        'profit_factor': 0.0,
                        'max_drawdown': 0.0,
                        'risk_score': 0.0,
                        'last_trade_time': None
                    }
                
                return jsonify(details)
                
            except Exception as e:
                logger.error(f"Error getting EA details for {ea_name}: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/eas/auto-detect', methods=['POST'])
        def auto_detect_eas():
            """Auto-detect EAs from MT5 trading history"""
            try:
                if not self.main_system.mt5_bridge:
                    return jsonify({"error": "MT5 bridge not available"}), 500
                
                # Get analysis period from request
                data = request.get_json() if request.is_json else {}
                days = data.get('days', 30)
                
                # Invalidate cache before detection
                self._invalidate_ea_cache()
                
                # Auto-detect EAs
                result = self.main_system.mt5_bridge.auto_detect_eas(days)
                
                if "error" in result:
                    return jsonify(result), 500
                
                # Extract detected EAs count
                detected_count = len(result.get("detected_eas", []))
                
                # Invalidate cache after detection to ensure fresh data
                self._invalidate_ea_cache()
                
                logger.info(f"Auto-detected {detected_count} EAs from {days} days of history")
                
                return jsonify({
                    "success": True,
                    "analysis_period_days": days,
                    "detected_eas": result,
                    "message": f"Auto-detected {detected_count} EAs from {days} days of history"
                })
                
            except Exception as e:
                logger.error(f"Error auto-detecting EAs: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/eas/<ea_name>/history')
        def get_ea_history(ea_name):
            """Get comprehensive trading history for EA"""
            try:
                days = request.args.get('days', 30, type=int)
                
                if not self.main_system.mt5_bridge:
                    return jsonify({"error": "MT5 bridge not available"}), 500
                
                # Get EA trading history
                history = self.main_system.mt5_bridge.get_ea_trading_history(ea_name, days)
                
                if not history:
                    return jsonify({"error": "No trading history found for EA"}), 404
                
                logger.info(f"Retrieved {len(history.get('trades', []))} trades for EA {ea_name}")
                
                return jsonify(history)
                
            except Exception as e:
                logger.error(f"Error getting EA history for {ea_name}: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/eas/scan-platform', methods=['POST'])
        def scan_platform():
            """Scan the MT5 platform for all active EAs"""
            try:
                if not self.main_system.mt5_bridge:
                    return jsonify({"error": "MT5 bridge not available"}), 500
                
                # Get current positions to identify active magic numbers
                import MetaTrader5 as mt5  # type: ignore
                positions = mt5.positions_get()  # type: ignore
                
                if positions is None:
                    return jsonify({"message": "No active positions found", "active_magic_numbers": []})
                
                # Extract unique magic numbers
                active_magic_numbers = list(set(pos.magic for pos in positions if pos.magic != 0))
                
                # Check which ones are not yet registered
                unregistered_eas = []
                for magic_number in active_magic_numbers:
                    is_registered = any(monitor.magic_number == magic_number 
                                      for monitor in self.main_system.mt5_bridge.ea_monitors.values())
                    
                    if not is_registered:
                        # Find symbol for this magic number
                        symbols = [pos.symbol for pos in positions if pos.magic == magic_number]
                        primary_symbol = max(set(symbols), key=symbols.count) if symbols else "UNKNOWN"
                        
                        unregistered_eas.append({
                            "magic_number": magic_number,
                            "primary_symbol": primary_symbol,
                            "symbols": list(set(symbols)),
                            "active_positions": len([pos for pos in positions if pos.magic == magic_number])
                        })
                
                return jsonify({
                    "active_magic_numbers": active_magic_numbers,
                    "registered_eas": len(self.main_system.mt5_bridge.ea_monitors),
                    "unregistered_eas": unregistered_eas,
                    "scan_time": datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error scanning platform: {e}")
                return jsonify({"error": str(e)}), 500
        

        @self.app.route('/api/trades/place', methods=['POST'])
        def place_trade():
            """Place a new trade"""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({"success": False, "message": "No data provided"}), 400
                
                symbol = data.get('symbol')
                trade_type = data.get('type')
                volume = data.get('volume')
                price = data.get('price')
                
                if not all([symbol, trade_type, volume]):
                    return jsonify({"success": False, "message": "Missing required fields"}), 400
                
                # Use MT5 bridge to place trade if available
                if self.main_system.mt5_bridge:
                    result = self.main_system.mt5_bridge.place_trade(
                        symbol=symbol,
                        trade_type=trade_type,
                        volume=volume,
                        price=price
                    )
                    
                    if result.get('success'):
                        return jsonify({
                            "success": True,
                            "message": f"Trade placed successfully",
                            "ticket": result.get('ticket')
                        })
                    else:
                        return jsonify({
                            "success": False,
                            "message": result.get('error', 'Failed to place trade')
                        })
                else:
                    return jsonify({
                        "success": False,
                        "message": "MT5 bridge not available"
                    })
                    
            except Exception as e:
                logger.error(f"Error placing trade: {e}")
                return jsonify({"success": False, "message": str(e)}), 500

        @self.app.route('/api/trades/<trade_id>/close', methods=['POST'])
        def close_trade(trade_id):
            """Close a specific trade"""
            try:
                # Use MT5 bridge to close trade if available
                if self.main_system.mt5_bridge:
                    result = self.main_system.mt5_bridge.close_trade(trade_id)
                    
                    if result.get('success'):
                        return jsonify({
                            "success": True,
                            "message": f"Trade {trade_id} closed successfully"
                        })
                    else:
                        return jsonify({
                            "success": False,
                            "message": result.get('error', 'Failed to close trade')
                        })
                else:
                    return jsonify({
                        "success": False,
                        "message": "MT5 bridge not available"
                    })
                    
            except Exception as e:
                logger.error(f"Error closing trade {trade_id}: {e}")
                return jsonify({"success": False, "message": str(e)}), 500

        @self.app.route('/api/trades/history')
        def get_trade_history():
            """Get real account equity history for equity curve with timeframe support"""
            try:
                # Get timeframe parameter (default to 6 months)
                timeframe = request.args.get('timeframe', '6M')
                
                # Map timeframe to days
                timeframe_map = {
                    '1W': 7,      # 1 week
                    '1M': 30,     # 1 month
                    '3M': 90,     # 3 months
                    '6M': 180,    # 6 months
                    '1Y': 365,    # 1 year
                    '2Y': 730,    # 2 years
                    '5Y': 1825    # 5 years
                }
                
                days_to_show = timeframe_map.get(timeframe, 180)
                
                # Get real account equity from MT5
                mt5_status = self.main_system.mt5_bridge.get_mt5_status()
                account_info = mt5_status.get('account_info')
                
                current_equity = 10000.0  # Default
                current_balance = 10000.0  # Default
                
                if account_info:
                    current_equity = account_info.get('equity', 10000.0)
                    current_balance = account_info.get('balance', 10000.0)
                    
                    logger.info(f"Real MT5 Account - Balance: ${current_balance:.2f}, Equity: ${current_equity:.2f}")
                
                # Try to get actual historical data from MT5
                history_data = []
                
                try:
                    # Get historical trades from MT5 if available
                    if self.main_system.mt5_bridge and hasattr(self.main_system.mt5_bridge, 'get_account_history'):
                        # Try to get real historical data
                        mt5_history = self.main_system.mt5_bridge.get_account_history(days_to_show)
                        if mt5_history:
                            history_data = mt5_history
                            logger.info(f"Retrieved {len(history_data)} points from MT5 history")
                    
                    # If no historical data available, generate synthetic data
                    if not history_data:
                        logger.info(f"No MT5 historical data available, generating synthetic data for {days_to_show} days")
                        history_data = self._generate_synthetic_equity_data(
                            current_balance, current_equity, days_to_show
                        )
                        
                except Exception as e:
                    logger.warning(f"Error getting MT5 historical data: {e}")
                    # Fall back to synthetic data
                    history_data = self._generate_synthetic_equity_data(
                        current_balance, current_equity, days_to_show
                    )
                
                current_profit = current_equity - current_balance
                
                logger.info(f"Equity history: {len(history_data)} points generated ({timeframe}), Current Equity: ${current_equity:.2f}")
                return jsonify({
                    'history': history_data, 
                    'current_balance': current_equity,
                    'account_balance': current_balance,
                    'floating_profit': current_profit,
                    'timeframe': timeframe,
                    'days_shown': days_to_show
                })
                
            except Exception as e:
                logger.error(f"Error getting equity history: {e}")
                # Fallback to basic data
                from datetime import datetime
                fallback_data = [{
                    'timestamp': datetime.now().isoformat(),
                    'trade_id': 'fallback',
                    'symbol': 'ACCOUNT',
                    'profit': 0.0,
                    'running_balance': 10000.0,
                    'trade_type': 'fallback'
                }]
                return jsonify({'history': fallback_data, 'current_balance': 10000.0})
        
        @self.app.route('/api/vision/status')
        def vision_status():
            """Get vision analysis status"""
            try:
                if self.main_system.vision_analyzer:
                    status = self.main_system.vision_analyzer.get_vision_status()
                    return jsonify(status)
                else:
                    return jsonify({"error": "Vision analyzer not available"})
            except Exception as e:
                logger.error(f"Error getting vision status: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/market/symbols')
        @self.async_wrapper.make_async
        def get_market_symbols():
            """Get real-time market data from MT5"""
            try:
                if not self.main_system.mt5_bridge:
                    return jsonify({"error": "MT5 bridge not available"}), 500
                
                # Force update symbols to get latest prices
                self.main_system.mt5_bridge._update_symbols()
                
                symbols_data = []
                for symbol_name, symbol_info in self.main_system.mt5_bridge.symbols.items():
                    # Calculate spread
                    spread = symbol_info.ask - symbol_info.bid
                    
                    # Use the actual daily change data from MT5Symbol
                    daily_change = getattr(symbol_info, 'daily_change', 0.0)
                    daily_change_percent = getattr(symbol_info, 'daily_change_percent', 0.0)
                    
                    symbols_data.append({
                        'name': symbol_name,
                        'bid': round(symbol_info.bid, symbol_info.digits),
                        'ask': round(symbol_info.ask, symbol_info.digits),
                        'last': round(symbol_info.last, symbol_info.digits),
                        'spread': round(spread, symbol_info.digits),
                        'change': daily_change,
                        'change_percent': daily_change_percent,
                        'volume': symbol_info.volume,
                        'time': symbol_info.time.isoformat(),
                        'digits': symbol_info.digits
                    })
                
                logger.info(f"Retrieved market data for {len(symbols_data)} symbols")
                return jsonify(symbols_data)
                
            except Exception as e:
                logger.error(f"Error getting market symbols: {e}")
                # Return fallback data if MT5 is not available with diverse asset classes
                fallback_symbols = [
                    {
                        'name': 'EURUSD',
                        'bid': 1.17319,
                        'ask': 1.17324,
                        'last': 1.17321,
                        'spread': 0.00005,
                        'change': 0.0012,
                        'change_percent': 0.11,
                        'volume': 0,
                        'time': datetime.now().isoformat(),
                        'digits': 5
                    },
                    {
                        'name': 'GBPUSD',
                        'bid': 1.36188,
                        'ask': 1.36193,
                        'last': 1.36190,
                        'spread': 0.00005,
                        'change': -0.0045,
                        'change_percent': -0.33,
                        'volume': 0,
                        'time': datetime.now().isoformat(),
                        'digits': 5
                    },
                    {
                        'name': 'USDJPY',
                        'bid': 145.430,
                        'ask': 145.435,
                        'last': 145.432,
                        'spread': 0.005,
                        'change': 0.125,
                        'change_percent': 0.08,
                        'volume': 0,
                        'time': datetime.now().isoformat(),
                        'digits': 3
                    },
                    {
                        'name': 'GOLD',
                        'bid': 2045.50,
                        'ask': 2045.80,
                        'last': 2045.65,
                        'spread': 0.30,
                        'change': 12.30,
                        'change_percent': 0.60,
                        'volume': 0,
                        'time': datetime.now().isoformat(),
                        'digits': 2
                    },
                    {
                        'name': 'SILVER',
                        'bid': 24.850,
                        'ask': 24.870,
                        'last': 24.860,
                        'spread': 0.020,
                        'change': -0.45,
                        'change_percent': -1.78,
                        'volume': 0,
                        'time': datetime.now().isoformat(),
                        'digits': 3
                    },
                    {
                        'name': 'BTCUSD',
                        'bid': 43250.50,
                        'ask': 43255.50,
                        'last': 43253.00,
                        'spread': 5.00,
                        'change': 1850.00,
                        'change_percent': 4.46,
                        'volume': 0,
                        'time': datetime.now().isoformat(),
                        'digits': 2
                    },
                    {
                        'name': 'ETHUSD',
                        'bid': 2615.75,
                        'ask': 2616.25,
                        'last': 2616.00,
                        'spread': 0.50,
                        'change': 125.50,
                        'change_percent': 5.04,
                        'volume': 0,
                        'time': datetime.now().isoformat(),
                        'digits': 2
                    },
                    {
                        'name': 'US30Cash',
                        'bid': 44125.0,
                        'ask': 44127.0,
                        'last': 44126.0,
                        'spread': 2.0,
                        'change': 185.5,
                        'change_percent': 0.42,
                        'volume': 0,
                        'time': datetime.now().isoformat(),
                        'digits': 1
                    }
                ]
                return jsonify(fallback_symbols)
        
        @self.app.route('/api/system/toggle-auto-trading', methods=['POST'])
        def toggle_auto_trading():
            """Toggle auto trading"""
            try:
                self.main_system.auto_trading_enabled = not self.main_system.auto_trading_enabled
                status = "enabled" if self.main_system.auto_trading_enabled else "disabled"
                return jsonify({"message": f"Auto trading {status}"})
            except Exception as e:
                logger.error(f"Error toggling auto trading: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/vision/toggle-auto-analysis', methods=['POST'])
        def toggle_vision_analysis():
            """Toggle vision auto analysis"""
            try:
                self.main_system.vision_auto_analysis = not self.main_system.vision_auto_analysis
                status = "enabled" if self.main_system.vision_auto_analysis else "disabled"
                return jsonify({"message": f"Vision auto analysis {status}"})
            except Exception as e:
                logger.error(f"Error toggling vision analysis: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/system/emergency-stop', methods=['POST'])
        def emergency_stop():
            """Emergency stop all trading"""
            try:
                # Disable auto trading
                self.main_system.auto_trading_enabled = False
                
                # Close all open trades if MT5 is available
                if self.main_system.mt5_bridge:
                    for trade_id in list(self.main_system.trade_manager.trades.keys()):
                        try:
                            self.main_system.mt5_bridge.close_trade(trade_id)
                        except Exception as e:
                            logger.error(f"Error closing trade {trade_id}: {e}")
                
                return jsonify({"message": "Emergency stop executed"})
            except Exception as e:
                logger.error(f"Error executing emergency stop: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/system/force-sync', methods=['POST'])
        def force_trade_sync():
            """Force immediate trade synchronization with MT5"""
            try:
                logger.info("🔄 Force trade sync requested via API")
                
                if hasattr(self.main_system, 'force_trade_sync'):
                    success, message = self.main_system.force_trade_sync()
                    
                    if success:
                        # Clear caches after sync
                        self._invalidate_ea_cache()
                        
                        return jsonify({
                            "success": True, 
                            "message": message,
                            "timestamp": datetime.now().isoformat()
                        })
                    else:
                        return jsonify({
                            "success": False, 
                            "message": message
                        }), 400
                else:
                    # Fallback if method doesn't exist
                    return jsonify({
                        "success": False, 
                        "message": "Force sync method not available"
                    }), 501
                    
            except Exception as e:
                logger.error(f"Error in force sync: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/vision/upload', methods=['POST'])
        def upload_chart():
            logger.info('[VISION] /api/vision/upload called')
            try:
                if 'image' not in request.files:
                    return jsonify({"error": "No image file provided"}), 400
                
                file = request.files['image']
                if file.filename == '':
                    return jsonify({"error": "No file selected"}), 400
                
                if self.main_system.vision_analyzer:
                    # Read image data
                    image_data = file.read()
                    
                    # Upload to vision analyzer
                    success, message, analysis_id = self.main_system.vision_analyzer.upload_chart_image(
                        image_data, file.filename
                    )
                    
                    if success:
                        return jsonify({
                            "message": message,
                            "analysis_id": analysis_id
                        })
                    else:
                        return jsonify({"error": message}), 400
                else:
                    return jsonify({"error": "Vision analyzer not available"}), 500
                    
            except Exception as e:
                logger.error(f"Error uploading chart: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/vision/analyze/<analysis_id>', methods=['POST'])
        def analyze_chart(analysis_id):
            # === ENHANCED DEBUG LOGGING ===
            print(f"=== VISION ENDPOINT HIT ===")
            print(f"Method: {request.method}")
            print(f"URL: {request.url}")
            print(f"Analysis ID: {analysis_id}")
            print(f"Headers: {dict(request.headers)}")
            print(f"Content-Type: {request.content_type}")
            print(f"Files: {list(request.files.keys())}")
            print(f"Form data: {dict(request.form)}")
            print(f"JSON data: {request.get_json()}")
            print(f"Raw data length: {len(request.data) if request.data else 0}")
            
            logger.info(f'[VISION] ENDPOINT HIT: analyze/{analysis_id}')
            logger.info(f'[VISION] Request method: {request.method}')
            logger.info(f'[VISION] Content-Type: {request.content_type}')
            
            if not self.main_system.vision_analyzer:
                logger.error('[VISION] Vision analyzer not initialized')
                response = {"analysis": {"confidence": None}}
                logger.info(f'[VISION] Response: {response}')
                return jsonify(response), 503
            
            try:
                data = request.get_json() or {}
                symbol = data.get('symbol', 'UNKNOWN')
                timeframe = data.get('timeframe', 'H4')
                
                logger.info(f'[VISION] Calling analyze_uploaded_chart_sync with symbol={symbol}, timeframe={timeframe}')
                
                analysis_result = self.main_system.vision_analyzer.analyze_uploaded_chart_sync(analysis_id, symbol, timeframe)
                
                if asyncio.iscoroutine(analysis_result):
                    logger.warning('[VISION] Got coroutine from analyze_uploaded_chart_sync, running event loop')
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    analysis_result = loop.run_until_complete(analysis_result)
                    loop.close()
                
                logger.info(f'[VISION] Analysis result type: {type(analysis_result)}')
                
                if analysis_result:
                    # Build the analysis dict
                    analysis_dict = {
                        "analysis_id": analysis_result.analysis_id,
                        "symbol": analysis_result.symbol,
                        "timeframe": analysis_result.timeframe,
                        "overall_trend": analysis_result.overall_trend,
                        "overall_confidence": analysis_result.overall_confidence,
                        "confidence": analysis_result.overall_confidence,  # Always present
                        "market_bias": str(analysis_result.market_bias.value) if hasattr(analysis_result.market_bias, 'value') else str(analysis_result.market_bias),
                        "primary_scenario": dataclasses.asdict(analysis_result.primary_scenario) if analysis_result.primary_scenario else None,
                        "alternative_scenario": dataclasses.asdict(analysis_result.alternative_scenario) if hasattr(analysis_result, 'alternative_scenario') and analysis_result.alternative_scenario else None,
                        "support_levels": [dataclasses.asdict(lvl) for lvl in getattr(analysis_result, 'support_levels', [])],
                        "resistance_levels": [dataclasses.asdict(lvl) for lvl in getattr(analysis_result, 'resistance_levels', [])],
                        "indicators": [dataclasses.asdict(ind) for ind in getattr(analysis_result, 'indicators', [])],
                        "patterns_detected": getattr(analysis_result, 'patterns_detected', []),
                        "risk_factors": getattr(analysis_result, 'risk_factors', []),
                        "confluence_factors": getattr(analysis_result, 'confluence_factors', [])
                    }
                    
                    response = {"analysis": analysis_dict, "success": True}
                    logger.info(f'[VISION] SUCCESS Response keys: {list(response.keys())}')
                    logger.info(f'[VISION] Analysis keys: {list(response["analysis"].keys())}')
                    return jsonify(response)
                else:
                    logger.warning('[VISION] No analysis result returned')
                    response = {"analysis": {"confidence": 0.0}, "success": False, "message": "Analysis failed"}
                    return jsonify(response), 200
                    
            except Exception as e:
                logger.error(f'[VISION] EXCEPTION in analyze_chart: {e}', exc_info=True)
                response = {"error": str(e), "analysis": {"confidence": 0.0}}
                return jsonify(response), 500

        @self.app.route('/api/vision/analyses', methods=['GET'])
        def get_recent_analyses():
            """Get recent vision analysis results"""
            try:
                if not self.main_system.vision_analyzer:
                    return jsonify({"error": "Vision analyzer not available"}), 500
                
                limit = request.args.get('limit', 10, type=int)
                analyses = self.main_system.vision_analyzer.get_recent_analyses(limit)
                
                result = []
                for analysis in analyses:
                    result.append({
                        "analysis_id": analysis.analysis_id,
                        "symbol": analysis.symbol,
                        "timeframe": analysis.timeframe,
                        "overall_trend": analysis.overall_trend,
                        "confidence": analysis.overall_confidence,  # PATCH: always present
                        "signal": analysis.primary_scenario.trade_type if analysis.primary_scenario else None,
                        "entry_price": analysis.primary_scenario.entry_price if analysis.primary_scenario else None,
                        "timestamp": analysis.timestamp.isoformat() if hasattr(analysis, 'timestamp') else None
                    })
                
                return jsonify({"analyses": result})
                
            except Exception as e:
                logger.error(f"Error getting recent analyses: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/eas/recalculate-performance', methods=['POST'])
        def recalculate_ea_performance():
            """Recalculate EA performance metrics"""
            try:
                # Invalidate cache to ensure fresh data
                self._invalidate_ea_cache()
                
                # Associate trades with EAs first
                associated_count = self.main_system.trade_manager.associate_trades_with_eas()
                
                # Recalculate performance metrics
                updated_count = self.main_system.trade_manager.recalculate_all_ea_performance()
                
                # Update from MT5 history if bridge is available
                if self.main_system.mt5_bridge:
                    for ea_name in self.main_system.trade_manager.ea_performances.keys():
                        try:
                            performance = self.main_system.trade_manager.ea_performances[ea_name]
                            if hasattr(self.main_system.mt5_bridge, 'update_ea_performance_from_mt5_history'):
                                self.main_system.mt5_bridge.update_ea_performance_from_mt5_history(ea_name, performance)
                        except Exception as e:
                            logger.warning(f"Could not update MT5 history for EA {ea_name}: {e}")
                
                # Force cache refresh after performance update
                self._invalidate_ea_cache()
                
                logger.info(f"EA performance recalculated: {updated_count} EAs updated, {associated_count} trades associated")
                
                return jsonify({
                    "success": True,
                    "updated_eas": updated_count,
                    "associated_trades": associated_count,
                    "message": f"Updated performance for {updated_count} EAs, associated {associated_count} trades"
                })
                
            except Exception as e:
                logger.error(f"Error recalculating EA performance: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/eas/intelligence')
        def get_ea_intelligence():
            """Get comprehensive EA intelligence and profiling data"""
            try:
                # Initialize intelligent EA manager if not exists
                if not hasattr(self, 'intelligent_ea_manager'):
                    from qnti_intelligent_ea_manager import QNTIIntelligentEAManager
                    self.intelligent_ea_manager = QNTIIntelligentEAManager(
                        self.main_system.trade_manager,
                        self.main_system.mt5_bridge
                    )
                
                # Auto-profile EAs if needed
                self.intelligent_ea_manager.auto_profile_eas()
                
                # Get intelligence summary
                intelligence = self.intelligent_ea_manager.get_ea_intelligence_summary()
                
                # Get detailed profiles for each EA
                detailed_profiles = {}
                for ea_name, profile in self.intelligent_ea_manager.profiler.profiles.items():
                    detailed_profiles[ea_name] = self.intelligent_ea_manager.profiler.export_profile_summary(ea_name)
                
                return jsonify({
                    "summary": intelligence,
                    "profiles": detailed_profiles,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error getting EA intelligence: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/eas/recommendations')
        def get_ea_recommendations():
            """Get AI-powered EA optimization recommendations"""
            try:
                ea_name = request.args.get('ea_name')  # Optional: get recommendations for specific EA
                
                # Initialize intelligent EA manager if not exists
                if not hasattr(self, 'intelligent_ea_manager'):
                    from qnti_intelligent_ea_manager import QNTIIntelligentEAManager
                    self.intelligent_ea_manager = QNTIIntelligentEAManager(
                        self.main_system.trade_manager,
                        self.main_system.mt5_bridge
                    )
                
                # Get AI recommendations
                recommendations = self.intelligent_ea_manager.get_ai_recommendations(ea_name)
                
                # Get current market analysis
                market_condition = self.intelligent_ea_manager.analyze_current_market()
                
                return jsonify({
                    "recommendations": recommendations,
                    "market_condition": {
                        "volatility": market_condition.volatility,
                        "trend_strength": market_condition.trend_strength,
                        "session": market_condition.session,
                        "spread_level": market_condition.spread_level
                    },
                    "total_recommendations": len(recommendations),
                    "high_priority": len([r for r in recommendations if r.get('urgency', 0) > 0.7]),
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error getting EA recommendations: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/eas/recommendations/apply', methods=['POST'])
        def apply_ea_recommendation():
            """Apply a specific EA optimization recommendation"""
            try:
                data = request.get_json()
                
                if not data:
                    return jsonify({"error": "No recommendation data provided"}), 400
                
                # Initialize intelligent EA manager if not exists
                if not hasattr(self, 'intelligent_ea_manager'):
                    from qnti_intelligent_ea_manager import QNTIIntelligentEAManager
                    self.intelligent_ea_manager = QNTIIntelligentEAManager(
                        self.main_system.trade_manager,
                        self.main_system.mt5_bridge
                    )
                
                # Apply the recommendation
                result = self.intelligent_ea_manager.apply_recommendation(data)
                
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"Error applying EA recommendation: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/eas/recommendations/bulk-apply', methods=['POST'])
        def bulk_apply_ea_recommendations():
            """Apply multiple EA optimization recommendations"""
            try:
                data = request.get_json()
                recommendations = data.get('recommendations', [])
                max_applications = data.get('max_applications', 5)
                
                if not recommendations:
                    return jsonify({"error": "No recommendations provided"}), 400
                
                # Initialize intelligent EA manager if not exists
                if not hasattr(self, 'intelligent_ea_manager'):
                    from qnti_intelligent_ea_manager import QNTIIntelligentEAManager
                    self.intelligent_ea_manager = QNTIIntelligentEAManager(
                        self.main_system.trade_manager,
                        self.main_system.mt5_bridge
                    )
                
                # Apply recommendations
                results = self.intelligent_ea_manager.bulk_apply_recommendations(
                    recommendations, max_applications
                )
                
                return jsonify({
                    "results": results,
                    "total_applied": len([r for r in results if r.get('success')]),
                    "total_failed": len([r for r in results if not r.get('success')]),
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error bulk applying recommendations: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/eas/profile', methods=['POST'])
        def create_ea_profile():
            """Create or update an EA profile manually"""
            try:
                data = request.get_json()
                
                if not data or not data.get('name'):
                    return jsonify({"error": "EA name is required"}), 400
                
                # Initialize intelligent EA manager if not exists
                if not hasattr(self, 'intelligent_ea_manager'):
                    from qnti_intelligent_ea_manager import QNTIIntelligentEAManager
                    self.intelligent_ea_manager = QNTIIntelligentEAManager(
                        self.main_system.trade_manager,
                        self.main_system.mt5_bridge
                    )
                
                # Create the profile
                success = self.intelligent_ea_manager.create_ea_profile_manually(data)
                
                if success:
                    return jsonify({"success": True, "message": "EA profile created successfully"})
                else:
                    return jsonify({"success": False, "error": "Failed to create EA profile"}), 500
                
            except Exception as e:
                logger.error(f"Error creating EA profile: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/eas/<ea_name>/profile')
        def get_ea_profile(ea_name):
            """Get detailed profile for specific EA"""
            try:
                # First, try to load from parsed EA profiles directory
                parsed_profile = self._load_parsed_ea_profile_by_name(ea_name)
                if parsed_profile:
                    logger.info(f"Found parsed EA profile for {ea_name}")
                    return jsonify({
                        "profile": parsed_profile,
                        "market_compatibility": {"compatibility": 0.5, "reasons": ["No market analysis for parsed profiles"]},
                        "recommendations": [{"title": "Parsed EA Profile", "description": "This is a parsed EA profile with extracted indicators and parameters", "urgency": "low"}],
                        "timestamp": datetime.now().isoformat()
                    })
                
                # If not found in parsed profiles, try intelligent EA manager
                # Initialize intelligent EA manager if not exists
                if not hasattr(self, 'intelligent_ea_manager'):
                    from qnti_intelligent_ea_manager import QNTIIntelligentEAManager
                    self.intelligent_ea_manager = QNTIIntelligentEAManager(
                        self.main_system.trade_manager,
                        self.main_system.mt5_bridge
                    )
                
                # Update profile from performance if needed
                self.intelligent_ea_manager.update_ea_profile_from_performance(ea_name)
                
                # Get profile
                profile_summary = self.intelligent_ea_manager.profiler.export_profile_summary(ea_name)
                
                if not profile_summary:
                    return jsonify({"error": "EA profile not found"}), 404
                
                # Get current market compatibility
                market_data = {
                    'volatility': self.intelligent_ea_manager.market_condition.volatility,
                    'trend_strength': self.intelligent_ea_manager.market_condition.trend_strength
                }
                
                compatibility = self.intelligent_ea_manager.profiler.analyze_market_compatibility(
                    ea_name, market_data
                )
                
                # Get specific recommendations for this EA
                recommendations = self.intelligent_ea_manager.get_ai_recommendations(ea_name)
                
                return jsonify({
                    "profile": profile_summary,
                    "market_compatibility": compatibility,
                    "recommendations": recommendations,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error getting EA profile for {ea_name}: {e}")
                return jsonify({"error": str(e)}), 500
        
        # EA Code Import routes
        @self.app.route('/dashboard/import_ea.html')
        @self.app.route('/import_ea.html')
        def import_ea():
            """EA Code Import page"""
            try:
                from pathlib import Path
                dashboard_file = Path("dashboard/import_ea.html")
                if dashboard_file.exists():
                    with open(dashboard_file, 'r', encoding='utf-8') as f:
                        return f.read()
                else:
                    return f"<h1>Error: import_ea.html not found</h1><p>Please ensure the file exists in the dashboard/ directory.</p>", 404
            except Exception as e:
                return f"<h1>Error loading EA import page</h1><p>{str(e)}</p>", 500

        @self.app.route('/api/ea/parse-code', methods=['POST'])
        def parse_ea_code():
            """Parse MQL4/MQL5 EA source code and return structured profile"""
            logger.info('[EA_PARSER] parse_ea_code called')
            
            try:
                data = request.get_json()
                if not data or 'code' not in data:
                    return jsonify({"error": "No code provided"}), 400
                
                code = data['code']
                
                if len(code) < 100:
                    return jsonify({"error": "Code too short"}), 400
                
                logger.info(f'[EA_PARSER] Parsing code ({len(code)} characters)')
                
                # Initialize the parser
                from qnti_ea_parser import MQLCodeParser
                parser = MQLCodeParser()
                
                # Parse the EA code
                ea_profile = parser.parse_ea_code(code)
                
                # Convert to JSON-serializable format
                profile_data = {
                    "name": ea_profile.name,
                    "description": ea_profile.description,
                    "symbols": ea_profile.symbols,
                    "timeframes": ea_profile.timeframes,
                    "magic_numbers": ea_profile.magic_numbers,
                    "parameters": [
                        {
                            "name": param.name,
                            "type": param.type,
                            "default_value": param.default_value,
                            "description": param.description,
                            "min_value": param.min_value,
                            "max_value": param.max_value,
                            "step": param.step
                        } for param in ea_profile.parameters
                    ],
                    "trading_rules": [
                        {
                            "type": rule.type,
                            "direction": rule.direction,
                            "conditions": rule.conditions,
                            "actions": rule.actions,
                            "indicators_used": rule.indicators_used,
                            "line_number": rule.line_number
                        } for rule in ea_profile.trading_rules
                    ],
                    "indicators": ea_profile.indicators,
                    "execution_status": ea_profile.execution_status
                }
                
                logger.info(f'[EA_PARSER] Successfully parsed EA: {ea_profile.name}')
                
                return jsonify({
                    "success": True,
                    "profile": profile_data,
                    "message": f"Successfully parsed EA '{ea_profile.name}'"
                })
                
            except Exception as e:
                logger.error(f'[EA_PARSER] Error parsing EA code: {e}', exc_info=True)
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500

        @self.app.route('/api/ea/save-profile', methods=['POST'])
        def save_ea_profile():
            """Save parsed EA profile to database/storage"""
            logger.info('[EA_PARSER] save_ea_profile called')
            logger.info(f'[EA_PARSER] Request headers: {dict(request.headers)}')
            logger.info(f'[EA_PARSER] Request content type: {request.content_type}')
            logger.info(f'[EA_PARSER] Request method: {request.method}')
            
            try:
                data = request.get_json()
                logger.info(f'[EA_PARSER] Received data: {data}')
                if not data:
                    logger.error('[EA_PARSER] No data provided in request')
                    return jsonify({"error": "No data provided"}), 400
                
                # Extract profile data
                ea_name = data.get('name', 'Unnamed EA')
                magic_number = data.get('magic_number', 0)
                symbols = data.get('symbols', [])
                timeframes = data.get('timeframes', [])
                parameters = data.get('parameters', {})
                original_code = data.get('original_code', '')
                profile = data.get('profile', {})
                
                logger.info(f'[EA_PARSER] Saving EA profile: {ea_name}')
                
                # Create EA profile object for storage
                ea_profile_data = {
                    'name': ea_name,
                    'magic_number': magic_number,
                    'symbols': symbols,
                    'timeframes': timeframes,
                    'parameters': parameters,
                    'original_code': original_code,
                    'profile': profile,
                    'created_at': datetime.now().isoformat(),
                    'status': 'inactive',
                    'source': 'code_import'
                }
                
                # Save to your EA storage system
                profile_id = self._save_ea_profile_to_storage(ea_profile_data)
                
                # Register with MT5 bridge if available
                if self.main_system.mt5_bridge and magic_number:
                    try:
                        # Register the EA for monitoring
                        primary_symbol = symbols[0] if symbols else "EURUSD"
                        primary_timeframe = timeframes[0] if timeframes else "H1"
                        
                        self.main_system.mt5_bridge.register_ea_monitor(
                            ea_name, magic_number, primary_symbol, primary_timeframe
                        )
                        logger.info(f'[EA_PARSER] Registered EA {ea_name} with MT5 bridge')
                    except Exception as e:
                        logger.warning(f'[EA_PARSER] Could not register with MT5 bridge: {e}')
                
                logger.info(f'[EA_PARSER] Successfully saved EA profile: {profile_id}')
                
                return jsonify({
                    "success": True,
                    "profile_id": profile_id,
                    "message": f"EA profile '{ea_name}' saved successfully"
                })
                
            except Exception as e:
                logger.error(f'[EA_PARSER] Error saving EA profile: {e}', exc_info=True)
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500

        @self.app.route('/api/ea/profiles', methods=['GET'])
        def get_ea_profiles():
            """Get all saved EA profiles"""
            try:
                profiles = self._load_ea_profiles_from_storage()
                
                return jsonify({
                    "success": True,
                    "profiles": profiles,
                    "count": len(profiles)
                })
                
            except Exception as e:
                logger.error(f'[EA_PARSER] Error loading EA profiles: {e}')
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500

        @self.app.route('/api/ea/profiles/<profile_id>/start', methods=['POST'])
        def start_ea_profile(profile_id):
            """Start executing an EA profile"""
            logger.info(f'[EA_PARSER] start_ea_profile called for {profile_id}')
            
            try:
                # Load the EA profile
                profile_data = self._load_ea_profile_by_id(profile_id)
                if not profile_data:
                    return jsonify({"error": "EA profile not found"}), 404
                
                # Get execution parameters from request
                data = request.get_json() or {}
                execution_params = data.get('parameters', {})
                
                # Initialize EA execution engine if not already done
                if not hasattr(self.main_system, 'ea_execution_engine'):
                    from qnti_ea_parser import EAExecutionEngine
                    self.main_system.ea_execution_engine = EAExecutionEngine(self.main_system.mt5_bridge)
                
                # Start the EA
                from qnti_ea_parser import EAProfile, EAParameter, TradingRule
                
                # Reconstruct EA profile object
                ea_profile = self._reconstruct_ea_profile_from_data(profile_data)
                
                # Start execution
                ea_id = self.main_system.ea_execution_engine.start_ea(ea_profile.id, execution_params)
                
                # Update profile status
                self._update_ea_profile_status(profile_id, 'active', ea_id)
                
                logger.info(f'[EA_PARSER] Started EA execution: {ea_id}')
                
                return jsonify({
                    "success": True,
                    "ea_id": ea_id,
                    "message": f"EA '{profile_data['name']}' started successfully"
                })
                
            except Exception as e:
                logger.error(f'[EA_PARSER] Error starting EA profile: {e}', exc_info=True)
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500

        @self.app.route('/api/ea/profiles/<profile_id>/stop', methods=['POST'])
        def stop_ea_profile(profile_id):
            """Stop executing an EA profile"""
            logger.info(f'[EA_PARSER] stop_ea_profile called for {profile_id}')
            
            try:
                # Load the EA profile
                profile_data = self._load_ea_profile_by_id(profile_id)
                if not profile_data:
                    return jsonify({"error": "EA profile not found"}), 404
                
                # Stop the EA if execution engine exists
                if hasattr(self.main_system, 'ea_execution_engine'):
                    ea_id = profile_data.get('execution_id')
                    if ea_id:
                        self.main_system.ea_execution_engine.stop_ea(ea_id)
                
                # Update profile status
                self._update_ea_profile_status(profile_id, 'inactive')
                
                logger.info(f'[EA_PARSER] Stopped EA: {profile_id}')
                
                return jsonify({
                    "success": True,
                    "message": f"EA '{profile_data['name']}' stopped successfully"
                })
                
            except Exception as e:
                logger.error(f'[EA_PARSER] Error stopping EA profile: {e}', exc_info=True)
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500

        @self.app.route('/api/ea/execute', methods=['POST'])
        def execute_ea():
            """Execute EA with custom parameters"""
            logger.info('[EA_PARSER] execute_ea called')
            
            try:
                data = request.get_json()
                if not data or 'ea_name' not in data:
                    return jsonify({'error': 'No EA name provided'}), 400
                
                ea_name = data['ea_name']
                logger.info(f'[EA_PARSER] Executing EA: {ea_name}')
                
                # Get EA profile
                if (not hasattr(self.main_system, 'trade_manager') or 
                    not hasattr(self.main_system.trade_manager, 'ea_profiles') or
                    ea_name not in self.main_system.trade_manager.ea_profiles):
                    return jsonify({'error': f'EA "{ea_name}" not found'}), 404
                
                ea_profile = self.main_system.trade_manager.ea_profiles[ea_name]
                
                # Initialize EA execution engine
                from qnti_ea_parser import EAExecutionEngine
                execution_engine = EAExecutionEngine(self.main_system.mt5_bridge)
                
                # Execute EA with custom parameters
                parameters = data.get('parameters', {})
                result = execution_engine.start_ea(ea_profile['name'], parameters)
                
                logger.info(f'[EA_PARSER] EA execution result: {result}')
                return jsonify({
                    'success': True,
                    'result': result,
                    'message': f'EA "{ea_name}" executed successfully'
                })
                
            except Exception as e:
                logger.error(f'[EA_PARSER] Error executing EA: {e}', exc_info=True)
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'message': 'Failed to execute EA'
                }), 500

        @self.app.route('/api/fast')
        def fast_route():
            """Ultra-fast route that bypasses all main system interactions"""
            return jsonify({
                "status": "ok", 
                "timestamp": datetime.now().isoformat(),
                "message": "Fast route working"
            })
        
        @self.app.route('/api/test')
        def test_route():
            """Simple test route for performance testing"""
            return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})
    
    def _generate_synthetic_equity_data(self, current_balance: float, current_equity: float, days_to_show: int):
        """Generate synthetic equity curve data when historical data is not available"""
        history_data = []
        current_profit = current_equity - current_balance
        
        for i in range(days_to_show):
            date = datetime.now() - timedelta(days=days_to_show-1-i)
            
            # Create a more realistic equity progression
            # Show gradual progression to current profit level
            daily_profit_portion = current_profit / days_to_show * i if current_profit != 0 else 0
            point_equity = current_balance + daily_profit_portion
            
            history_data.append({
                'timestamp': date.isoformat(),
                'trade_id': f'equity_snapshot_{i}',
                'symbol': 'ACCOUNT',
                'profit': daily_profit_portion,
                'running_balance': round(point_equity, 2),
                'trade_type': 'equity_snapshot'
            })
        
        # Add current real equity as final point
        history_data.append({
            'timestamp': datetime.now().isoformat(),
            'trade_id': 'current_equity',
            'symbol': 'ACCOUNT',  
            'profit': current_profit,
            'running_balance': round(current_equity, 2),
            'trade_type': 'current'
        })
        
        return history_data
        
        # === TEST ENDPOINT ===
        @self.app.route('/api/test-analytics')
        def test_analytics():
            """Test endpoint to verify analytics integration"""
            return jsonify({
                "status": "success",
                "message": "Analytics module loaded",
                "timestamp": datetime.now().isoformat(),
                "has_analytics": hasattr(self, 'advanced_analytics')
            })
        
        # === ADVANCED ANALYTICS API ENDPOINTS ===
        
        @self.app.route('/api/analytics/comprehensive')
        @self.async_wrapper.make_async
        def get_comprehensive_analysis():
            """Get comprehensive trading analysis with AI insights"""
            try:
                analysis = self.advanced_analytics.generate_comprehensive_analysis()
                return jsonify(analysis)
            except Exception as e:
                logger.error(f"Error getting comprehensive analysis: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/analytics/advanced-metrics')
        @self.async_wrapper.make_async
        def get_advanced_metrics():
            """Get advanced trading metrics"""
            try:
                metrics = self.advanced_analytics.calculate_advanced_metrics()
                return jsonify({"metrics": metrics, "timestamp": datetime.now().isoformat()})
            except Exception as e:
                logger.error(f"Error getting advanced metrics: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/analytics/risk-profile')
        @self.async_wrapper.make_async
        def get_risk_profile():
            """Get risk profile analysis"""
            try:
                risk_profile = self.advanced_analytics.analyze_risk_profile()
                return jsonify(risk_profile)
            except Exception as e:
                logger.error(f"Error getting risk profile: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/analytics/market-insights')
        @self.async_wrapper.make_async
        def get_market_insights():
            """Get AI-powered market insights"""
            try:
                insights = self.advanced_analytics.generate_market_insights()
                return jsonify({"insights": insights, "timestamp": datetime.now().isoformat()})
            except Exception as e:
                logger.error(f"Error getting market insights: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/analytics/ai-recommendations')
        @self.async_wrapper.make_async
        def get_ai_recommendations():
            """Get AI-powered trading recommendations"""
            try:
                recommendations = self.advanced_analytics.generate_ai_recommendations()
                return jsonify({"recommendations": recommendations, "timestamp": datetime.now().isoformat()})
            except Exception as e:
                logger.error(f"Error getting AI recommendations: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/analytics/performance-summary')
        @self.async_wrapper.make_async
        def get_performance_summary():
            """Get high-level performance summary"""
            try:
                summary = self.advanced_analytics.get_performance_summary()
                return jsonify(summary)
            except Exception as e:
                logger.error(f"Error getting performance summary: {e}")
                return jsonify({"error": str(e)}), 500
        
        # Backtesting API endpoints
        @self.app.route('/dashboard/backtesting.html')
        @self.app.route('/backtesting.html')
        def backtesting_page():
            """Backtesting page"""
            try:
                from pathlib import Path
                dashboard_file = Path("dashboard/backtesting.html")
                if dashboard_file.exists():
                    with open(dashboard_file, 'r', encoding='utf-8') as f:
                        return f.read()
                else:
                    return f"<h1>Error: backtesting.html not found</h1><p>Please ensure the file exists in the dashboard/ directory.</p>", 404
            except Exception as e:
                return f"<h1>Error loading backtesting page</h1><p>{str(e)}</p>", 500
        
        @self.app.route('/api/backtest', methods=['POST'])
        def run_backtest():
            """Run a backtest for a given strategy"""
            try:
                if not hasattr(self.main_system, 'backtesting_engine') or not self.main_system.backtesting_engine:
                    return jsonify({"error": "Backtesting engine not available"}), 503
                
                data = request.get_json()
                if not data:
                    return jsonify({"error": "No data provided"}), 400
                
                # Validate required fields
                required_fields = ['strategy', 'symbol', 'start_date', 'end_date']
                for field in required_fields:
                    if field not in data:
                        return jsonify({"error": f"Missing required field: {field}"}), 400
                
                # Parse dates
                from datetime import datetime
                start_date = datetime.fromisoformat(data['start_date'])
                end_date = datetime.fromisoformat(data['end_date'])
                
                # Run backtest
                result = self.main_system.backtesting_engine.run_backtest(
                    strategy_name=data['strategy'],
                    symbol=data['symbol'],
                    start_date=start_date,
                    end_date=end_date,
                    strategy_params=data.get('strategy_params', {}),
                    initial_capital=data.get('initial_capital', 10000)
                )
                
                # Convert result to JSON-serializable format
                result_dict = {
                    'strategy_name': result.strategy_name,
                    'symbol': result.symbol,
                    'start_date': result.start_date.isoformat(),
                    'end_date': result.end_date.isoformat(),
                    'initial_capital': result.initial_capital,
                    'final_capital': result.final_capital,
                    'total_return': result.total_return,
                    'total_trades': result.total_trades,
                    'winning_trades': result.winning_trades,
                    'losing_trades': result.losing_trades,
                    'win_rate': result.win_rate,
                    'avg_win': result.avg_win,
                    'avg_loss': result.avg_loss,
                    'max_drawdown': result.max_drawdown,
                    'sharpe_ratio': result.sharpe_ratio,
                    'sortino_ratio': result.sortino_ratio,
                    'calmar_ratio': result.calmar_ratio,
                    'profit_factor': result.profit_factor,
                    'equity_curve': [
                        (timestamp.isoformat(), value) for timestamp, value in result.equity_curve
                    ]
                }
                
                return jsonify(result_dict)
                
            except Exception as e:
                logger.error(f"Error running backtest: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/backtest/strategies')
        def get_available_strategies():
            """Get list of available backtesting strategies"""
            try:
                if not hasattr(self.main_system, 'backtesting_engine') or not self.main_system.backtesting_engine:
                    return jsonify({"error": "Backtesting engine not available"}), 503
                
                strategies = list(self.main_system.backtesting_engine.strategies.keys())
                return jsonify({"strategies": strategies})
                
            except Exception as e:
                logger.error(f"Error getting strategies: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/backtest/compare', methods=['POST'])
        def compare_strategies():
            """Compare multiple strategies on the same data"""
            try:
                if not hasattr(self.main_system, 'backtesting_engine') or not self.main_system.backtesting_engine:
                    return jsonify({"error": "Backtesting engine not available"}), 503
                
                data = request.get_json()
                if not data:
                    return jsonify({"error": "No data provided"}), 400
                
                # Validate required fields
                required_fields = ['strategies', 'symbol', 'start_date', 'end_date']
                for field in required_fields:
                    if field not in data:
                        return jsonify({"error": f"Missing required field: {field}"}), 400
                
                # Parse dates
                from datetime import datetime
                start_date = datetime.fromisoformat(data['start_date'])
                end_date = datetime.fromisoformat(data['end_date'])
                
                # Run strategy comparison
                results = self.main_system.backtesting_engine.run_strategy_comparison(
                    strategies=data['strategies'],
                    symbol=data['symbol'],
                    start_date=start_date,
                    end_date=end_date,
                    strategy_params=data.get('strategy_params', {})
                )
                
                # Convert results to JSON-serializable format
                comparison_results = {}
                for strategy_name, result in results.items():
                    comparison_results[strategy_name] = {
                        'total_return': result.total_return,
                        'total_trades': result.total_trades,
                        'win_rate': result.win_rate,
                        'max_drawdown': result.max_drawdown,
                        'sharpe_ratio': result.sharpe_ratio,
                        'profit_factor': result.profit_factor
                    }
                
                return jsonify(comparison_results)
                
            except Exception as e:
                logger.error(f"Error comparing strategies: {e}")
                return jsonify({"error": str(e)}), 500
    
    def setup_websocket_handlers(self):
        """Setup WebSocket event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            logger.info("Web client connected")
            emit('status', {'message': 'Connected to QNTI system'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            logger.info("Web client disconnected")
        
        @self.socketio.on('get_system_status')
        def handle_system_status():
            """Handle system status request"""
            try:
                health = self.main_system.get_system_health()
                emit('system_status', health)
            except Exception as e:
                logger.error(f"Error getting system status: {e}")
                emit('error', {'message': str(e)})
    
    def broadcast_trade_update(self, trade_data: Dict):
        """Broadcast trade update to all connected clients"""
        try:
            self.socketio.emit('trade_update', trade_data)
        except Exception as e:
            logger.error(f"Error broadcasting trade update: {e}")
    
    def broadcast_system_alert(self, alert_data: Dict):
        """Broadcast system alert to all connected clients"""
        try:
            self.socketio.emit('system_alert', alert_data)
        except Exception as e:
            logger.error(f"Error broadcasting system alert: {e}")

    def setup_ea_generation_routes(self):
        """Setup EA Generation routes with proper styling and AI integration"""
        
        @self.app.route('/ea-generation')
        def ea_generation_page():
            """EA Generation main page with QNTI styling"""
            return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QNTI EA Generation - Quantum Nexus Trading Intelligence</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #e2e8f0;
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        .dashboard-header {
            background: rgba(15, 23, 42, 0.9);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid #334155;
            padding: 1rem 2rem;
            position: sticky;
            top: 0;
            z-index: 100;
            display: flex;
            justify-content: between;
            align-items: center;
        }
        
        .logo {
            font-size: 1.5rem;
            font-weight: bold;
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .nav-links {
            display: flex;
            gap: 2rem;
            margin-left: auto;
        }
        
        .nav-links a {
            color: #94a3b8;
            text-decoration: none;
            font-weight: 500;
            transition: color 0.2s;
        }
        
        .nav-links a:hover,
        .nav-links a.active {
            color: #3b82f6;
        }
        
        .main-container {
            max-width: 1600px;
            margin: 2rem auto;
            padding: 0 2rem;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
        }
        
        .dashboard-card {
            background: rgba(30, 41, 59, 0.8);
            backdrop-filter: blur(10px);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid #334155;
        }
        
        .card-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: #f1f5f9;
        }
        
        .ai-insight-box {
            background: rgba(139, 92, 246, 0.1);
            border: 1px solid #8b5cf6;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        .ai-insight-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: #8b5cf6;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        
        .form-group {
            margin-bottom: 1.5rem;
        }
        
        .form-label {
            display: block;
            margin-bottom: 0.5rem;
            color: #e2e8f0;
            font-weight: 500;
        }
        
        .form-input,
        .form-select,
        .form-textarea {
            width: 100%;
            padding: 0.75rem;
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid #334155;
            border-radius: 6px;
            color: #e2e8f0;
            font-size: 0.9rem;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        
        .form-input:focus,
        .form-select:focus,
        .form-textarea:focus {
            outline: none;
            border-color: #3b82f6;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        }
        
        .btn {
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #3b82f6, #1d4ed8);
            color: white;
        }
        
        .btn-secondary {
            background: rgba(71, 85, 105, 0.8);
            color: #e2e8f0;
        }
        
        .indicator-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 0.5rem;
            margin-top: 1rem;
        }
        
        .indicator-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem;
            background: rgba(15, 23, 42, 0.5);
            border-radius: 4px;
            border: 1px solid transparent;
            transition: all 0.2s;
        }
        
        .indicator-item:hover {
            border-color: #3b82f6;
            background: rgba(59, 130, 246, 0.1);
        }
        
        .indicator-checkbox {
            margin: 0;
        }
        
        .progress-bar {
            width: 100%;
            height: 6px;
            background: rgba(15, 23, 42, 0.8);
            border-radius: 3px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #3b82f6, #8b5cf6);
            width: 0%;
            transition: width 0.3s ease;
        }
        
        .status-indicator {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 500;
        }
        
        .status-pending {
            background: rgba(245, 158, 11, 0.2);
            color: #f59e0b;
        }
        
        .status-running {
            background: rgba(59, 130, 246, 0.2);
            color: #3b82f6;
        }
        
        .status-completed {
            background: rgba(16, 185, 129, 0.2);
            color: #10b981;
        }
        
        .status-failed {
            background: rgba(239, 68, 68, 0.2);
            color: #ef4444;
        }
        
        .pulse-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: currentColor;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .workflow-step {
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 1rem;
            border-left: 3px solid #334155;
            margin-bottom: 0.5rem;
            transition: all 0.2s;
        }
        
        .workflow-step.active {
            border-left-color: #3b82f6;
            background: rgba(59, 130, 246, 0.1);
        }
        
        .workflow-step.completed {
            border-left-color: #10b981;
            background: rgba(16, 185, 129, 0.1);
        }
        
        .workflow-icon {
            width: 24px;
            height: 24px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8rem;
            font-weight: bold;
        }
        
        .workflow-icon.pending {
            background: #334155;
            color: #94a3b8;
        }
        
        .workflow-icon.active {
            background: #3b82f6;
            color: white;
        }
        
        .workflow-icon.completed {
            background: #10b981;
            color: white;
        }
        
        @media (max-width: 1200px) {
            .main-container {
                grid-template-columns: 1fr;
            }
        }
        
        @media (max-width: 768px) {
            .nav-links {
                display: none;
            }
            
            .main-container {
                padding: 0 1rem;
            }
            
            .indicator-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <header class="dashboard-header">
        <div class="logo">⚡ QNTI</div>
        <nav class="nav-links">
            <a href="/">Dashboard</a>
            <a href="/ea-generation" class="active">EA Generation</a>
            <a href="/vision">Vision Analysis</a>
            <a href="/analytics">Analytics</a>
        </nav>
    </header>

    <div class="main-container">
        <!-- EA Configuration Panel -->
        <div class="dashboard-card">
            <div class="card-header">
                <h2 class="card-title">🤖 EA Strategy Builder</h2>
            </div>
            
            <form id="eaForm">
                <div class="form-group">
                    <label class="form-label">Strategy Name</label>
                    <input type="text" id="eaName" class="form-input" placeholder="My Trading Strategy" required>
                </div>
                
                <div class="form-group">
                    <label class="form-label">Description</label>
                    <textarea id="eaDescription" class="form-textarea" rows="3" placeholder="Describe your trading strategy..."></textarea>
                </div>
                
                <div class="form-group">
                    <label class="form-label">Trading Symbols</label>
                    <select id="eaSymbols" class="form-select" multiple>
                        <option value="EURUSD">EUR/USD</option>
                        <option value="GBPUSD">GBP/USD</option>
                        <option value="USDJPY">USD/JPY</option>
                        <option value="AUDUSD">AUD/USD</option>
                        <option value="USDCAD">USD/CAD</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label class="form-label">Timeframes</label>
                    <select id="eaTimeframes" class="form-select" multiple>
                        <option value="M15">15 Minutes</option>
                        <option value="H1" selected>1 Hour</option>
                        <option value="H4">4 Hours</option>
                        <option value="D1">Daily</option>
                    </select>
                </div>
                
                <div class="ai-insight-box" id="aiRecommendations">
                    <div class="ai-insight-header">
                        🧠 AI Indicator Recommendations
                    </div>
                    <div id="aiRecommendationText">Loading AI recommendations...</div>
                </div>
                
                <div class="form-group">
                    <label class="form-label">Technical Indicators</label>
                    <div class="indicator-grid" id="indicatorGrid">
                        <!-- Indicators will be populated here -->
                    </div>
                </div>
                
                <div class="form-group">
                    <label class="form-label">Optimization Method</label>
                    <select id="optimizationMethod" class="form-select">
                        <option value="genetic_algorithm">Genetic Algorithm</option>
                        <option value="grid_search">Grid Search</option>
                        <option value="bayesian">Bayesian Optimization</option>
                    </select>
                </div>
                
                <button type="submit" class="btn btn-primary">
                    🚀 Generate EA Strategy
                </button>
            </form>
        </div>
        
        <!-- Workflow Progress Panel -->
        <div class="dashboard-card">
            <div class="card-header">
                <h2 class="card-title">📊 Generation Progress</h2>
                <div id="workflowStatus" class="status-indicator status-pending">
                    <div class="pulse-dot"></div>
                    Ready
                </div>
            </div>
            
            <div id="workflowSteps">
                <div class="workflow-step" data-step="design">
                    <div class="workflow-icon pending">1</div>
                    <div>
                        <div style="font-weight: 600;">Strategy Design</div>
                        <div style="font-size: 0.9rem; color: #94a3b8;">Configure indicators and parameters</div>
                    </div>
                </div>
                
                <div class="workflow-step" data-step="optimization">
                    <div class="workflow-icon pending">2</div>
                    <div>
                        <div style="font-weight: 600;">Parameter Optimization</div>
                        <div style="font-size: 0.9rem; color: #94a3b8;">Find optimal parameter values</div>
                    </div>
                </div>
                
                <div class="workflow-step" data-step="robustness">
                    <div class="workflow-icon pending">3</div>
                    <div>
                        <div style="font-weight: 600;">Robustness Testing</div>
                        <div style="font-size: 0.9rem; color: #94a3b8;">Validate strategy reliability</div>
                    </div>
                </div>
                
                <div class="workflow-step" data-step="validation">
                    <div class="workflow-icon pending">4</div>
                    <div>
                        <div style="font-weight: 600;">Out-of-Sample Validation</div>
                        <div style="font-size: 0.9rem; color: #94a3b8;">Test on unseen data</div>
                    </div>
                </div>
                
                <div class="workflow-step" data-step="reporting">
                    <div class="workflow-icon pending">5</div>
                    <div>
                        <div style="font-weight: 600;">Performance Report</div>
                        <div style="font-size: 0.9rem; color: #94a3b8;">Generate comprehensive analysis</div>
                    </div>
                </div>
            </div>
            
            <div style="margin-top: 1.5rem;">
                <div style="margin-bottom: 0.5rem; font-size: 0.9rem; color: #94a3b8;">Overall Progress</div>
                <div class="progress-bar">
                    <div class="progress-fill" id="overallProgress"></div>
                </div>
                <div style="margin-top: 0.5rem; font-size: 0.8rem; color: #94a3b8;" id="progressText">0% Complete</div>
            </div>
            
            <div class="ai-insight-box" id="workflowInsights" style="display: none;">
                <div class="ai-insight-header">
                    🎯 AI Workflow Insights
                </div>
                <div id="workflowInsightText"></div>
            </div>
        </div>
    </div>

    <script>
        class EAGenerationUI {
            constructor() {
                this.indicators = [];
                this.selectedIndicators = new Set();
                this.currentWorkflow = null;
                this.init();
            }
            
            async init() {
                await this.loadIndicators();
                this.setupEventListeners();
                this.loadAIRecommendations();
            }
            
            async loadIndicators() {
                try {
                    const response = await fetch('/api/ea/indicators');
                    const data = await response.json();
                    
                    if (data.success) {
                        this.indicators = data.indicators;
                        this.renderIndicators();
                        
                        // Display AI recommendations
                        if (data.ai_recommendations) {
                            document.getElementById('aiRecommendationText').innerText = data.ai_recommendations;
                        }
                    }
                } catch (error) {
                    console.error('Error loading indicators:', error);
                    document.getElementById('indicatorGrid').innerHTML = '<p>Error loading indicators</p>';
                }
            }
            
            async loadAIRecommendations() {
                try {
                    const response = await fetch('/api/ea/indicators');
                    const data = await response.json();
                    
                    if (data.success && data.ai_recommendations) {
                        document.getElementById('aiRecommendationText').innerText = data.ai_recommendations;
                    }
                } catch (error) {
                    console.error('Error loading AI recommendations:', error);
                }
            }
            
            renderIndicators() {
                const grid = document.getElementById('indicatorGrid');
                grid.innerHTML = this.indicators.map(indicator => `
                    <div class="indicator-item">
                        <input type="checkbox" 
                               class="indicator-checkbox" 
                               id="indicator_${indicator.name}" 
                               value="${indicator.name}"
                               onchange="eaUI.toggleIndicator('${indicator.name}')">
                        <label for="indicator_${indicator.name}">${indicator.name}</label>
                    </div>
                `).join('');
            }
            
            toggleIndicator(indicatorName) {
                if (this.selectedIndicators.has(indicatorName)) {
                    this.selectedIndicators.delete(indicatorName);
                } else {
                    this.selectedIndicators.add(indicatorName);
                }
            }
            
            setupEventListeners() {
                document.getElementById('eaForm').addEventListener('submit', (e) => {
                    e.preventDefault();
                    this.generateEA();
                });
            }
            
            async generateEA() {
                const formData = {
                    ea_name: document.getElementById('eaName').value,
                    description: document.getElementById('eaDescription').value,
                    symbols: Array.from(document.getElementById('eaSymbols').selectedOptions).map(o => o.value),
                    timeframes: Array.from(document.getElementById('eaTimeframes').selectedOptions).map(o => o.value),
                    indicators: Array.from(this.selectedIndicators).map(name => ({name, params: {}})),
                    method: document.getElementById('optimizationMethod').value,
                    auto_proceed: true
                };
                
                try {
                    this.updateWorkflowStatus('running', 'Generating EA...');
                    
                    const response = await fetch('/api/ea/workflow/start', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(formData)
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        this.currentWorkflow = data.workflow_id;
                        this.startProgressMonitoring();
                        
                        // Show AI insights if available
                        if (data.ai_insights) {
                            this.showWorkflowInsights(data.ai_insights);
                        }
                    } else {
                        this.updateWorkflowStatus('failed', 'Generation failed');
                        console.error('EA generation failed:', data.error);
                    }
                } catch (error) {
                    this.updateWorkflowStatus('failed', 'Connection error');
                    console.error('Error generating EA:', error);
                }
            }
            
            updateWorkflowStatus(status, message) {
                const statusEl = document.getElementById('workflowStatus');
                statusEl.className = `status-indicator status-${status}`;
                statusEl.innerHTML = `<div class="pulse-dot"></div>${message}`;
            }
            
            showWorkflowInsights(insights) {
                const insightsBox = document.getElementById('workflowInsights');
                const insightsText = document.getElementById('workflowInsightText');
                insightsText.innerText = insights;
                insightsBox.style.display = 'block';
            }
            
            startProgressMonitoring() {
                if (!this.currentWorkflow) return;
                
                const checkProgress = async () => {
                    try {
                        const response = await fetch(`/api/ea/workflow/status/${this.currentWorkflow}`);
                        const data = await response.json();
                        
                        if (data.success) {
                            this.updateProgress(data);
                            
                            if (data.status === 'completed') {
                                this.updateWorkflowStatus('completed', 'EA Generated Successfully!');
                                return;
                            } else if (data.status === 'failed') {
                                this.updateWorkflowStatus('failed', 'Generation Failed');
                                return;
                            }
                        }
                        
                        // Continue monitoring
                        setTimeout(checkProgress, 2000);
                    } catch (error) {
                        console.error('Error checking progress:', error);
                        setTimeout(checkProgress, 5000);
                    }
                };
                
                checkProgress();
            }
            
            updateProgress(data) {
                // Update overall progress
                const progressFill = document.getElementById('overallProgress');
                const progressText = document.getElementById('progressText');
                progressFill.style.width = `${data.progress}%`;
                progressText.innerText = `${Math.round(data.progress)}% Complete`;
                
                // Update workflow steps
                const stageMap = {
                    'design': 1,
                    'optimization': 2,
                    'robustness_testing': 3,
                    'validation': 4,
                    'reporting': 5
                };
                
                // Reset all steps
                document.querySelectorAll('.workflow-step').forEach(step => {
                    step.classList.remove('active', 'completed');
                    const icon = step.querySelector('.workflow-icon');
                    icon.className = 'workflow-icon pending';
                });
                
                // Mark completed stages
                data.completed_stages.forEach(stage => {
                    const stepIndex = stageMap[stage];
                    if (stepIndex) {
                        const step = document.querySelector(`[data-step="${stage}"]`);
                        step.classList.add('completed');
                        const icon = step.querySelector('.workflow-icon');
                        icon.className = 'workflow-icon completed';
                        icon.innerHTML = '✓';
                    }
                });
                
                // Mark current stage as active
                const currentStageIndex = stageMap[data.current_stage];
                if (currentStageIndex) {
                    const currentStep = document.querySelector(`[data-step="${data.current_stage}"]`);
                    currentStep.classList.add('active');
                    const icon = currentStep.querySelector('.workflow-icon');
                    icon.className = 'workflow-icon active';
                }
            }
        }
        
        // Initialize EA Generation UI
        const eaUI = new EAGenerationUI();
    </script>
</body>
</html>
            """)
        
        @self.app.route('/api/ea-generation/workflow/<workflow_id>/ai-insights')
        def get_workflow_ai_insights(workflow_id):
            """Get AI insights for specific workflow"""
            try:
                # Check if LLM integration is available
                if not hasattr(self.main_system, 'llm_integration') or not self.main_system.llm_integration:
                    return jsonify({
                        'success': False,
                        'error': 'AI insights not available'
                    }), 503
                
                # Get workflow status
                if not hasattr(self.main_system, 'ea_workflow_engine') or not self.main_system.ea_workflow_engine:
                    return jsonify({
                        'success': False,
                        'error': 'EA Workflow Engine not available'
                    }), 503
                
                workflow_state = self.main_system.ea_workflow_engine.get_workflow_status(workflow_id)
                
                if not workflow_state:
                    return jsonify({
                        'success': False,
                        'error': 'Workflow not found'
                    }), 404
                
                # Generate AI insights based on workflow state
                import ollama
                
                prompt = f"""
                Provide AI insights for this EA generation workflow:
                
                Workflow ID: {workflow_id}
                Current Stage: {workflow_state.current_stage.value}
                Status: {workflow_state.status.value}
                Progress: {workflow_state.overall_progress}%
                Completed Stages: {', '.join([s.value for s in workflow_state.completed_stages])}
                
                Based on the current state, provide:
                1. Assessment of current progress
                2. Next expected outcomes
                3. Any potential concerns or recommendations
                
                Keep response under 150 words, actionable and professional.
                """
                
                response = ollama.chat(
                    model='llama3',
                    messages=[{'role': 'user', 'content': prompt}],
                    options={'temperature': 0.6, 'max_tokens': 150}
                )
                
                insights = response['message']['content'].strip()
                
                return jsonify({
                    'success': True,
                    'insights': insights,
                    'workflow_state': {
                        'stage': workflow_state.current_stage.value,
                        'status': workflow_state.status.value,
                        'progress': workflow_state.overall_progress
                    },
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error getting workflow AI insights: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500

    # Helper methods for EA profile storage
    def _save_ea_profile_to_storage(self, profile_data):
        """Save EA profile to storage (implement based on your storage system)"""
        import json
        import uuid
        from pathlib import Path
        
        # Create profiles directory if it doesn't exist
        profiles_dir = Path("ea_profiles")
        profiles_dir.mkdir(exist_ok=True)
        
        # Generate unique profile ID
        profile_id = str(uuid.uuid4())[:8]
        profile_data['id'] = profile_id
        
        # Save to JSON file
        profile_file = profiles_dir / f"{profile_id}.json"
        with open(profile_file, 'w', encoding='utf-8') as f:
            json.dump(profile_data, f, indent=2)
        
        logger.info(f'[EA_PARSER] Saved profile to: {profile_file}')
        return profile_id

    def _load_ea_profiles_from_storage(self):
        """Load all EA profiles from storage"""
        import json
        from pathlib import Path
        
        profiles_dir = Path("ea_profiles")
        if not profiles_dir.exists():
            return []
        
        profiles = []
        for profile_file in profiles_dir.glob("*.json"):
            try:
                with open(profile_file, 'r', encoding='utf-8') as f:
                    profile_data = json.load(f)
                    # Don't include the original code in the list (too large)
                    profile_summary = {k: v for k, v in profile_data.items() if k != 'original_code'}
                    profiles.append(profile_summary)
            except Exception as e:
                logger.warning(f'[EA_PARSER] Could not load profile {profile_file}: {e}')
        
        return profiles

    def _load_ea_profile_by_id(self, profile_id):
        """Load specific EA profile by ID"""
        import json
        from pathlib import Path
        
        profile_file = Path("ea_profiles") / f"{profile_id}.json"
        if not profile_file.exists():
            return None
        
        try:
            with open(profile_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f'[EA_PARSER] Error loading profile {profile_id}: {e}')
            return None

    def _load_parsed_ea_profile_by_name(self, ea_name):
        """Load parsed EA profile by EA name"""
        import json
        from pathlib import Path
        
        try:
            profiles_dir = Path("ea_profiles")
            if not profiles_dir.exists():
                return None
            
            # Search through all profile files to find matching EA name
            for profile_file in profiles_dir.glob("*.json"):
                try:
                    with open(profile_file, 'r', encoding='utf-8') as f:
                        profile_data = json.load(f)
                    
                    # Check if this profile matches the EA name
                    if profile_data.get('name') == ea_name:
                        logger.info(f"Found parsed profile for {ea_name} in {profile_file}")
                        return profile_data
                        
                except Exception as e:
                    logger.warning(f'Could not load profile {profile_file}: {e}')
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading parsed EA profile for {ea_name}: {e}")
            return None

    def _update_ea_profile_status(self, profile_id, status, execution_id=None):
        """Update EA profile status"""
        profile_data = self._load_ea_profile_by_id(profile_id)
        if profile_data:
            profile_data['status'] = status
            if execution_id:
                profile_data['execution_id'] = execution_id
            elif 'execution_id' in profile_data:
                del profile_data['execution_id']
            
            # Save back to storage
            import json
            from pathlib import Path
            
            profile_file = Path("ea_profiles") / f"{profile_id}.json"
            with open(profile_file, 'w', encoding='utf-8') as f:
                json.dump(profile_data, f, indent=2)

    def _reconstruct_ea_profile_from_data(self, profile_data):
        """Reconstruct EAProfile object from stored data"""
        from qnti_ea_parser import EAProfile, EAParameter, TradingRule
        from datetime import datetime
        
        # Reconstruct parameters
        parameters = []
        for param_data in profile_data.get('profile', {}).get('parameters', []):
            parameters.append(EAParameter(
                name=param_data['name'],
                type=param_data['type'],
                default_value=param_data['default_value'],
                description=param_data.get('description', ''),
                min_value=param_data.get('min_value'),
                max_value=param_data.get('max_value'),
                step=param_data.get('step')
            ))
        
        # Reconstruct trading rules
        trading_rules = []
        for rule_data in profile_data.get('profile', {}).get('trading_rules', []):
            trading_rules.append(TradingRule(
                type=rule_data['type'],
                direction=rule_data['direction'],
                conditions=rule_data.get('conditions', []),
                actions=rule_data.get('actions', []),
                indicators_used=rule_data.get('indicators_used', []),
                line_number=rule_data.get('line_number', 0)
            ))
        
        # Create and return EA profile
        return EAProfile(
            id=profile_data.get('id', 'unknown'),
            name=profile_data['name'],
            description=profile_data.get('description', ''),
            parameters=parameters,
            trading_rules=trading_rules,
            indicators=profile_data.get('profile', {}).get('indicators', []),
            symbols=profile_data['symbols'],
            timeframes=profile_data['timeframes'],
            magic_numbers=profile_data.get('magic_numbers', []),
            created_at=profile_data.get('created_at', datetime.now().isoformat()),
            source_code=profile_data.get('original_code', ''),
            performance_stats=profile_data.get('performance_stats', {})
        )
    
    def _get_total_trades_count(self):
        """Get total trades count from trade manager"""
        try:
            if hasattr(self.main_system, 'trade_manager') and self.main_system.trade_manager:
                return len(self.main_system.trade_manager.trades)
            return 0
        except Exception as e:
            logger.warning(f"Error getting total trades count: {e}")
            return 0
    
    def _get_daily_profit(self):
        """Get today's profit from trade manager"""
        try:
            if hasattr(self.main_system, 'trade_manager') and self.main_system.trade_manager:
                today = datetime.now().date()
                daily_profit = 0.0
                
                for trade in self.main_system.trade_manager.trades.values():
                    if trade.close_time and trade.close_time.date() == today:
                        daily_profit += trade.profit or 0.0
                    elif not trade.close_time and trade.open_time and trade.open_time.date() == today:
                        daily_profit += trade.profit or 0.0  # Running profit for open trades
                
                return daily_profit
            return 0.0
        except Exception as e:
            logger.warning(f"Error getting daily profit: {e}")
            return 0.0
    
    def _get_win_rate(self):
        """Calculate overall win rate"""
        try:
            if hasattr(self.main_system, 'trade_manager') and self.main_system.trade_manager:
                closed_trades = [t for t in self.main_system.trade_manager.trades.values() 
                               if t.close_time is not None]
                
                if not closed_trades:
                    return 0.0
                
                winning_trades = [t for t in closed_trades if t.profit > 0]
                return (len(winning_trades) / len(closed_trades)) * 100
            return 0.0
        except Exception as e:
            logger.warning(f"Error calculating win rate: {e}")
            return 0.0
    
    def _get_best_trade(self):
        """Get best trade profit"""
        try:
            if hasattr(self.main_system, 'trade_manager') and self.main_system.trade_manager:
                closed_trades = [t for t in self.main_system.trade_manager.trades.values() 
                               if t.close_time is not None]
                
                if not closed_trades:
                    return 0.0
                
                return max(trade.profit for trade in closed_trades)
            return 0.0
        except Exception as e:
            logger.warning(f"Error getting best trade: {e}")
            return 0.0
    
    def _get_worst_trade(self):
        """Get worst trade profit"""
        try:
            if hasattr(self.main_system, 'trade_manager') and self.main_system.trade_manager:
                closed_trades = [t for t in self.main_system.trade_manager.trades.values() 
                               if t.close_time is not None]
                
                if not closed_trades:
                    return 0.0
                
                return min(trade.profit for trade in closed_trades)
            return 0.0
        except Exception as e:
            logger.warning(f"Error getting worst trade: {e}")
            return 0.0
    
    def _get_avg_trade(self):
        """Get average trade profit"""
        try:
            if hasattr(self.main_system, 'trade_manager') and self.main_system.trade_manager:
                closed_trades = [t for t in self.main_system.trade_manager.trades.values() 
                               if t.close_time is not None]
                
                if not closed_trades:
                    return 0.0
                
                total_profit = sum(trade.profit for trade in closed_trades)
                return total_profit / len(closed_trades)
            return 0.0
        except Exception as e:
            logger.warning(f"Error getting average trade: {e}")
            return 0.0
    
    def _get_sharpe_ratio(self):
        """Calculate Sharpe ratio"""
        try:
            if hasattr(self.main_system, 'trade_manager') and self.main_system.trade_manager:
                closed_trades = [t for t in self.main_system.trade_manager.trades.values() 
                               if t.close_time is not None]
                
                if len(closed_trades) < 2:
                    return 0.0
                
                profits = [trade.profit for trade in closed_trades]
                avg_profit = sum(profits) / len(profits)
                
                # Calculate standard deviation
                variance = sum((p - avg_profit) ** 2 for p in profits) / len(profits)
                std_dev = variance ** 0.5
                
                if std_dev == 0:
                    return 0.0
                
                # Simplified Sharpe ratio (assuming risk-free rate = 0)
                return avg_profit / std_dev
            return 0.0
        except Exception as e:
            logger.warning(f"Error calculating Sharpe ratio: {e}")
            return 0.0
