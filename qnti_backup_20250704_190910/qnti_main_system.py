#!/usr/bin/env python3
"""
Quantum Nexus Trading Intelligence (QNTI) - Main Integration System (FIXED)
Unified orchestration of all QNTI modules with Flask API and WebSocket support
"""

import asyncio
import threading
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import signal
import sys
import os

# Flask and WebSocket imports
from flask import Flask, jsonify, request, render_template_string
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# Import QNTI modules
from qnti_core_system import QNTITradeManager, Trade, TradeSource, TradeStatus
from qnti_mt5_integration import QNTIMT5Bridge
from qnti_vision_analysis import QNTIVisionAnalyzer

# Configure logging with safe Unicode handling
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qnti_main.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('QNTI_MAIN')

class QNTIMainSystem:
    """Quantum Nexus Trading Intelligence - Main Orchestration System"""
    
    def __init__(self, config_file: str = "qnti_config.json"):
        self.config_file = config_file
        self.config = {}
        self.running = False
        
        # Core components
        self.trade_manager: Optional[QNTITradeManager] = None
        self.mt5_bridge: Optional[QNTIMT5Bridge] = None
        self.vision_analyzer: Optional[QNTIVisionAnalyzer] = None
        
        # Flask app and SocketIO
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'qnti_secret_key_2024'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        CORS(self.app)
        
        # Control flags
        self.auto_trading_enabled = False
        self.vision_auto_analysis = False
        self.ea_monitoring_enabled = True
        
        # Performance tracking
        self.performance_metrics = {
            'system_start_time': datetime.now(),
            'total_analyses': 0,
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'api_calls': 0,
            'errors': 0
        }
        
        # Load configuration
        self._load_config()
        
        # Initialize components
        self._initialize_components()
        
        # Setup Flask routes
        self._setup_routes()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("QNTI Main System initialized successfully")
    
    def _load_config(self):
        """Load main system configuration with proper defaults"""
        try:
            if Path(self.config_file).exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            else:
                # Create comprehensive default configuration
                self.config = {
                    "system": {
                        "auto_trading": False,
                        "vision_auto_analysis": True,
                        "ea_monitoring": True,
                        "api_port": 5000,
                        "debug_mode": True,
                        "max_concurrent_trades": 10,
                        "risk_management": {
                            "max_daily_loss": 1000,
                            "max_drawdown": 0.20,
                            "position_size_limit": 1.0,
                            "emergency_close_drawdown": 0.20  # FIXED: Added missing parameter
                        }
                    },
                    "integration": {
                        "mt5_enabled": True,
                        "vision_enabled": True,
                        "dashboard_enabled": True,
                        "webhook_enabled": False,
                        "telegram_notifications": False
                    },
                    "ea_monitoring": {  # FIXED: Added missing ea_monitoring section
                        "check_interval": 30,
                        "log_directory": "MQL5/Files/EA_Logs",
                        "enable_file_monitoring": True
                    },
                    "scheduling": {
                        "vision_analysis_interval": 300,  # seconds
                        "health_check_interval": 60,
                        "performance_update_interval": 30,
                        "backup_interval": 3600
                    },
                    "alerts": {
                        "email_alerts": False,
                        "telegram_alerts": False,
                        "webhook_alerts": False,
                        "log_alerts": True
                    },
                    "vision": {  # FIXED: Added vision config section
                        "primary_symbols": ["EURUSD", "GBPUSD", "USDJPY"],
                        "timeframes": ["H1", "H4"]
                    }
                }
                
                self._save_config()
                logger.info("Created default configuration file")
        
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self.config = {}
    
    def _save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def _initialize_components(self):
        """Initialize all QNTI components with proper error handling"""
        try:
            # Initialize core trade manager
            self.trade_manager = QNTITradeManager()
            logger.info("Trade Manager initialized")
            
            # Initialize MT5 bridge if enabled
            if self.config.get("integration", {}).get("mt5_enabled", True):
                try:
                    self.mt5_bridge = QNTIMT5Bridge(self.trade_manager)
                    logger.info("MT5 Bridge initialized")
                except Exception as e:
                    logger.warning(f"MT5 Bridge initialization failed: {e}")
                    self.mt5_bridge = None
            
            # Initialize vision analyzer if enabled
            if self.config.get("integration", {}).get("vision_enabled", True):
                try:
                    self.vision_analyzer = QNTIVisionAnalyzer(self.trade_manager)
                    logger.info("Vision Analyzer initialized")
                except Exception as e:
                    logger.warning(f"Vision Analyzer initialization failed: {e}")
                    self.vision_analyzer = None
            
            # Set configuration flags
            self.auto_trading_enabled = self.config.get("system", {}).get("auto_trading", False)
            self.vision_auto_analysis = self.config.get("system", {}).get("vision_auto_analysis", True)
            self.ea_monitoring_enabled = self.config.get("system", {}).get("ea_monitoring", True)
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            # Don't raise - continue with available components
    
    def _setup_routes(self):
        """Setup Flask API routes"""
        
        # Dashboard route
        @self.app.route('/')
        def dashboard():
            """Serve the main dashboard"""
            try:
                # Read the dashboard HTML file with proper encoding
                dashboard_file = Path("qnti_dashboard.html")
                if dashboard_file.exists():
                    with open(dashboard_file, 'r', encoding='utf-8') as f:
                        return f.read()
                else:
                    # Return basic dashboard if file not found
                    return self._get_basic_dashboard()
            except Exception as e:
                logger.error(f"Error loading dashboard: {e}")
                return f"Error loading dashboard: {e}", 500
        
        # API Routes
        @self.app.route('/api/health')
        def health_check():
            """System health check endpoint"""
            try:
                health_data = self.get_system_health()
                return jsonify(health_data)
            except Exception as e:
                logger.error(f"Health check error: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/trades')
        def get_trades():
            """Get all trades"""
            try:
                trades_data = []
                for trade_id, trade in self.trade_manager.trades.items():
                    trade_dict = {
                        'id': trade.trade_id,
                        'symbol': trade.symbol,
                        'type': trade.trade_type,
                        'size': trade.lot_size,
                        'open_price': trade.open_price,
                        'current_price': trade.close_price,
                        'profit': trade.profit,
                        'status': trade.status.value,
                        'source': trade.source.value,
                        'open_time': trade.open_time.isoformat(),
                        'ea_name': trade.ea_name
                    }
                    trades_data.append(trade_dict)
                
                return jsonify(trades_data)
            except Exception as e:
                logger.error(f"Get trades error: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/trades', methods=['POST'])
        def create_manual_trade():
            """Create a manual trade"""
            try:
                data = request.json
                
                trade = Trade(
                    trade_id=f"MANUAL_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    magic_number=data.get('magic_number', 999999),
                    symbol=data['symbol'],
                    trade_type=data['type'],
                    lot_size=data['lot_size'],
                    open_price=data['price'],
                    stop_loss=data.get('stop_loss'),
                    take_profit=data.get('take_profit'),
                    source=TradeSource.MANUAL,
                    notes=data.get('notes', 'Manual trade from dashboard')
                )
                
                # Execute trade if MT5 bridge available and auto-trading enabled
                if self.mt5_bridge and self.auto_trading_enabled:
                    success, message = self.mt5_bridge.execute_trade(trade)
                    if success:
                        self.performance_metrics['successful_trades'] += 1
                        self._emit_trade_update(trade)
                        return jsonify({"success": True, "message": message, "trade_id": trade.trade_id})
                    else:
                        self.performance_metrics['failed_trades'] += 1
                        return jsonify({"success": False, "message": message}), 400
                else:
                    # Add to trade manager only (simulation mode)
                    self.trade_manager.add_trade(trade)
                    self._emit_trade_update(trade)
                    return jsonify({"success": True, "message": "Trade added (simulation mode)", "trade_id": trade.trade_id})
                
            except Exception as e:
                logger.error(f"Create trade error: {e}")
                self.performance_metrics['failed_trades'] += 1
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/trades/<trade_id>', methods=['DELETE'])
        def close_trade(trade_id):
            """Close a specific trade"""
            try:
                if self.mt5_bridge:
                    success, message = self.mt5_bridge.close_trade(trade_id)
                    if success:
                        self._emit_trade_update(None, action="closed")
                        return jsonify({"success": True, "message": message})
                    else:
                        return jsonify({"success": False, "message": message}), 400
                else:
                    # Simulation mode - just mark as closed
                    if trade_id in self.trade_manager.trades:
                        trade = self.trade_manager.trades[trade_id]
                        # Simulate close price
                        close_price = trade.open_price * (1 + (0.01 if trade.trade_type == "BUY" else -0.01))
                        self.trade_manager.close_trade(trade_id, close_price)
                        self._emit_trade_update(None, action="closed")
                        return jsonify({"success": True, "message": "Trade closed (simulation)"})
                    else:
                        return jsonify({"success": False, "message": "Trade not found"}), 404
                
            except Exception as e:
                logger.error(f"Close trade error: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/eas')
        def get_eas():
            """Get EA performance data"""
            try:
                ea_data = []
                for ea_name, performance in self.trade_manager.ea_performances.items():
                    ea_dict = {
                        'name': ea_name,
                        'status': performance.status.value,
                        'total_trades': performance.total_trades,
                        'winning_trades': performance.winning_trades,
                        'win_rate': performance.win_rate,
                        'total_profit': performance.total_profit,
                        'total_loss': performance.total_loss,
                        'profit_factor': performance.profit_factor,
                        'risk_score': performance.risk_score,
                        'last_trade_time': performance.last_trade_time.isoformat() if performance.last_trade_time else None
                    }
                    ea_data.append(ea_dict)
                
                return jsonify(ea_data)
            except Exception as e:
                logger.error(f"Get EAs error: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/eas/<ea_name>/control', methods=['POST'])
        def control_ea(ea_name):
            """Control EA (pause, resume, block, etc.)"""
            try:
                data = request.json
                action = data.get('action')
                parameters = data.get('parameters', {})
                
                success = self.trade_manager.control_ea(ea_name, action, parameters)
                
                if success:
                    self._emit_ea_update(ea_name, action)
                    return jsonify({"success": True, "message": f"EA {ea_name} {action} successful"})
                else:
                    return jsonify({"success": False, "message": f"Failed to {action} EA {ea_name}"}), 400
                
            except Exception as e:
                logger.error(f"Control EA error: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/vision/analyze', methods=['POST'])
        def trigger_vision_analysis():
            """Trigger manual vision analysis"""
            try:
                data = request.json
                symbol = data.get('symbol', 'EURUSD')
                timeframe = data.get('timeframe', 'H1')
                
                if not self.vision_analyzer:
                    return jsonify({"error": "Vision analyzer not available"}), 400
                
                # Run analysis in background thread
                def analyze():
                    try:
                        analysis = self.vision_analyzer.manual_analysis(symbol, timeframe)
                        if analysis:
                            self.performance_metrics['total_analyses'] += 1
                            self._emit_vision_update(analysis)
                            
                            # Generate trade if conditions met
                            if self.auto_trading_enabled:
                                trade = self.vision_analyzer.generate_trade_from_analysis(analysis)
                                if trade and self.mt5_bridge:
                                    success, message = self.mt5_bridge.execute_trade(trade)
                                    if success:
                                        self.performance_metrics['successful_trades'] += 1
                                        self._emit_trade_update(trade)
                    except Exception as e:
                        logger.error(f"Vision analysis background error: {e}")
                
                threading.Thread(target=analyze, daemon=True).start()
                
                return jsonify({"success": True, "message": "Vision analysis started"})
                
            except Exception as e:
                logger.error(f"Vision analysis error: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/vision/status')
        def get_vision_status():
            """Get vision analysis status and recent results"""
            try:
                if not self.vision_analyzer:
                    return jsonify({"error": "Vision analyzer not available"}), 400
                
                # FIXED: Safe method calls with error handling
                try:
                    status = self.vision_analyzer.get_vision_status()
                except AttributeError:
                    status = {"error": "get_vision_status method not available"}
                
                try:
                    summary = self.vision_analyzer.get_analysis_summary(hours=24)
                except AttributeError:
                    summary = {"error": "get_analysis_summary method not available"}
                
                return jsonify({
                    "status": status,
                    "summary": summary
                })
                
            except Exception as e:
                logger.error(f"Get vision status error: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/system/config', methods=['GET', 'POST'])
        def system_config():
            """Get or update system configuration"""
            try:
                if request.method == 'GET':
                    return jsonify(self.config)
                else:
                    # Update configuration
                    new_config = request.json
                    self.config.update(new_config)
                    self._save_config()
                    
                    # Apply configuration changes
                    self._apply_config_changes()
                    
                    return jsonify({"success": True, "message": "Configuration updated"})
                    
            except Exception as e:
                logger.error(f"System config error: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/system/emergency-stop', methods=['POST'])
        def emergency_stop():
            """Emergency stop all trading activities"""
            try:
                logger.critical("EMERGENCY STOP TRIGGERED")
                
                # Close all trades
                if self.mt5_bridge:
                    open_trades = [t for t in self.trade_manager.trades.values() if t.status == TradeStatus.OPEN]
                    for trade in open_trades:
                        try:
                            self.mt5_bridge.close_trade(trade.trade_id)
                        except Exception as e:
                            logger.error(f"Error closing trade {trade.trade_id}: {e}")
                
                # Pause all EAs
                for ea_name in self.trade_manager.ea_performances.keys():
                    try:
                        self.trade_manager.control_ea(ea_name, 'pause')
                    except Exception as e:
                        logger.error(f"Error pausing EA {ea_name}: {e}")
                
                # Disable auto trading
                self.auto_trading_enabled = False
                self.vision_auto_analysis = False
                
                self._emit_system_alert("EMERGENCY STOP ACTIVATED", "critical")
                
                return jsonify({"success": True, "message": "Emergency stop executed"})
                
            except Exception as e:
                logger.error(f"Emergency stop error: {e}")
                return jsonify({"error": str(e)}), 500
        
        # WebSocket event handlers
        @self.socketio.on('connect')
        def handle_connect():
            logger.info(f"Client connected: {request.sid}")
            # Send initial data to newly connected client
            try:
                emit('initial_data', self.get_dashboard_data())
            except Exception as e:
                logger.error(f"Error sending initial data: {e}")
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('request_update')
        def handle_update_request():
            try:
                emit('dashboard_update', self.get_dashboard_data())
            except Exception as e:
                logger.error(f"Error sending dashboard update: {e}")
    
    def _get_basic_dashboard(self):
        """Return basic HTML dashboard if main file not found"""
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>QNTI Dashboard</title>
            <meta charset="UTF-8">
        </head>
        <body>
            <h1>Quantum Nexus Trading Intelligence</h1>
            <p>Dashboard loading...</p>
            <p>Main dashboard file not found. Please ensure qnti_dashboard.html exists.</p>
            <script>
                // Auto-refresh to check for dashboard file
                setTimeout(() => location.reload(), 5000);
            </script>
        </body>
        </html>
        '''
    
    def _apply_config_changes(self):
        """Apply configuration changes to running system"""
        try:
            # Update system flags
            self.auto_trading_enabled = self.config.get("system", {}).get("auto_trading", False)
            self.vision_auto_analysis = self.config.get("system", {}).get("vision_auto_analysis", True)
            self.ea_monitoring_enabled = self.config.get("system", {}).get("ea_monitoring", True)
            
            logger.info("Configuration changes applied")
            
        except Exception as e:
            logger.error(f"Error applying config changes: {e}")
    
    def get_system_health(self) -> Dict:
        """Get comprehensive system health status"""
        try:
            health = self.trade_manager.get_system_health()
            
            # Add component status
            mt5_status = None
            if self.mt5_bridge:
                try:
                    mt5_status = self.mt5_bridge.get_mt5_status()
                except Exception as e:
                    mt5_status = {"error": str(e)}
            
            vision_status = None
            if self.vision_analyzer:
                try:
                    vision_status = self.vision_analyzer.get_vision_status()
                except Exception as e:
                    vision_status = {"error": str(e)}
            
            health.update({
                'mt5_status': mt5_status,
                'vision_status': vision_status,
                'auto_trading_enabled': self.auto_trading_enabled,
                'vision_auto_analysis': self.vision_auto_analysis,
                'ea_monitoring_enabled': self.ea_monitoring_enabled,
                'performance_metrics': self.performance_metrics,
                'uptime_seconds': (datetime.now() - self.performance_metrics['system_start_time']).total_seconds()
            })
            
            return health
            
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {"error": str(e)}
    
    def get_dashboard_data(self) -> Dict:
        """Get all dashboard data for WebSocket updates"""
        try:
            data = {
                'system_health': self.get_system_health(),
                'trades': [],
                'eas': [],
                'vision_analyses': [],
                'alerts': [],
                'timestamp': datetime.now().isoformat()
            }
            
            # Get trades data
            for trade_id, trade in self.trade_manager.trades.items():
                trade_data = {
                    'id': trade.trade_id,
                    'symbol': trade.symbol,
                    'type': trade.trade_type,
                    'size': trade.lot_size,
                    'profit': trade.profit,
                    'status': trade.status.value,
                    'source': trade.source.value,
                    'open_time': trade.open_time.isoformat(),
                    'ea_name': trade.ea_name
                }
                data['trades'].append(trade_data)
            
            # Get EA data
            for ea_name, performance in self.trade_manager.ea_performances.items():
                ea_data = {
                    'name': ea_name,
                    'status': performance.status.value,
                    'profit': performance.total_profit - performance.total_loss,
                    'trades': performance.total_trades,
                    'win_rate': performance.win_rate,
                    'risk_score': performance.risk_score
                }
                data['eas'].append(ea_data)
            
            # Get vision analyses
            if self.vision_analyzer and hasattr(self.vision_analyzer, 'analysis_history') and self.vision_analyzer.analysis_history:
                recent_analyses = self.vision_analyzer.analysis_history[-5:]  # Last 5
                for analysis in recent_analyses:
                    analysis_data = {
                        'symbol': analysis.symbol,
                        'signal': analysis.entry_signal,
                        'confidence': analysis.confidence_score,
                        'timestamp': analysis.timestamp.isoformat()
                    }
                    data['vision_analyses'].append(analysis_data)
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {"error": str(e)}
    
    def _emit_trade_update(self, trade: Trade = None, action: str = "created"):
        """Emit trade update via WebSocket"""
        try:
            if trade:
                trade_data = {
                    'id': trade.trade_id,
                    'symbol': trade.symbol,
                    'type': trade.trade_type,
                    'size': trade.lot_size,
                    'profit': trade.profit,
                    'status': trade.status.value,
                    'action': action
                }
            else:
                trade_data = {'action': action}
            
            self.socketio.emit('trade_update', trade_data)
            
        except Exception as e:
            logger.error(f"Error emitting trade update: {e}")
    
    def _emit_ea_update(self, ea_name: str, action: str):
        """Emit EA update via WebSocket"""
        try:
            self.socketio.emit('ea_update', {
                'ea_name': ea_name,
                'action': action,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error emitting EA update: {e}")
    
    def _emit_vision_update(self, analysis):
        """Emit vision analysis update via WebSocket"""
        try:
            analysis_data = {
                'symbol': analysis.symbol,
                'signal': analysis.entry_signal,
                'confidence': analysis.confidence_score,
                'timestamp': analysis.timestamp.isoformat(),
                'reasoning': analysis.reasoning[:200] if analysis.reasoning else ""
            }
            
            self.socketio.emit('vision_update', analysis_data)
            
        except Exception as e:
            logger.error(f"Error emitting vision update: {e}")
    
    def _emit_system_alert(self, message: str, level: str = "info"):
        """Emit system alert via WebSocket"""
        try:
            alert_data = {
                'message': message,
                'level': level,
                'timestamp': datetime.now().isoformat()
            }
            
            self.socketio.emit('system_alert', alert_data)
            
        except Exception as e:
            logger.error(f"Error emitting system alert: {e}")
    
    def start_background_tasks(self):
        """Start background monitoring and analysis tasks"""
        try:
            # Start MT5 monitoring
            if self.mt5_bridge and self.ea_monitoring_enabled:
                try:
                    self.mt5_bridge.start_monitoring()
                    logger.info("MT5 monitoring started")
                except Exception as e:
                    logger.error(f"Error starting MT5 monitoring: {e}")
            
            # Start vision auto-analysis
            if self.vision_analyzer and self.vision_auto_analysis:
                try:
                    symbols = self.config.get("vision", {}).get("primary_symbols", ["EURUSD", "GBPUSD", "USDJPY"])
                    timeframes = self.config.get("vision", {}).get("timeframes", ["H1", "H4"])
                    self.vision_analyzer.start_automated_analysis(symbols, timeframes)
                    logger.info("Vision auto-analysis started")
                except Exception as e:
                    logger.error(f"Error starting vision analysis: {e}")
            
            # Start periodic tasks
            self._start_periodic_tasks()
            
        except Exception as e:
            logger.error(f"Error starting background tasks: {e}")
    
    def _start_periodic_tasks(self):
        """Start periodic maintenance and monitoring tasks"""
        def periodic_health_check():
            while self.running:
                try:
                    # Update performance metrics
                    self.performance_metrics['api_calls'] += 1
                    
                    # Emit health update via WebSocket
                    health_data = self.get_system_health()
                    self.socketio.emit('health_update', health_data)
                    
                    # Check for recommendations
                    if self.trade_manager:
                        recommendations = self.trade_manager.get_ea_recommendations()
                        if recommendations:
                            for rec in recommendations:
                                self._emit_system_alert(
                                    f"EA Recommendation: {rec['action']} for {rec['ea_name']} - {rec['reason']}", 
                                    "info"
                                )
                    
                except Exception as e:
                    logger.error(f"Periodic health check error: {e}")
                    self.performance_metrics['errors'] += 1
                
                time.sleep(self.config.get("scheduling", {}).get("health_check_interval", 60))
        
        def periodic_backup():
            while self.running:
                try:
                    # Backup trade data and logs
                    backup_dir = Path("qnti_backups")
                    backup_dir.mkdir(exist_ok=True)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # Save current state
                    backup_data = {
                        'trades': {k: v.__dict__ for k, v in self.trade_manager.trades.items()},
                        'ea_performances': {k: v.__dict__ for k, v in self.trade_manager.ea_performances.items()},
                        'performance_metrics': self.performance_metrics,
                        'config': self.config,
                        'timestamp': timestamp
                    }
                    
                    backup_file = backup_dir / f"qnti_backup_{timestamp}.json"
                    with open(backup_file, 'w', encoding='utf-8') as f:
                        json.dump(backup_data, f, indent=2, default=str, ensure_ascii=False)
                    
                    logger.info(f"Backup created: {backup_file}")
                    
                except Exception as e:
                    logger.error(f"Backup error: {e}")
                
                time.sleep(self.config.get("scheduling", {}).get("backup_interval", 3600))
        
        # Start background threads
        threading.Thread(target=periodic_health_check, daemon=True).start()
        threading.Thread(target=periodic_backup, daemon=True).start()
        
        logger.info("Periodic tasks started")
    
    def start(self, host: str = "0.0.0.0", port: int = None, debug: bool = None):
        """Start the QNTI main system with safe Unicode logging"""
        try:
            self.running = True
            
            # Get configuration
            port = port or self.config.get("system", {}).get("api_port", 5000)
            debug = debug if debug is not None else self.config.get("system", {}).get("debug_mode", False)
            
            # Start background tasks
            self.start_background_tasks()
            
            logger.info(f"Starting QNTI Main System on {host}:{port}")
            logger.info(f"Dashboard URL: http://{host}:{port}")
            logger.info("=== QUANTUM NEXUS TRADING INTELLIGENCE ===")
            logger.info("Components Status:")
            logger.info(f"  * Trade Manager: Active")
            logger.info(f"  * MT5 Bridge: {'Active' if self.mt5_bridge else 'Disabled'}")
            logger.info(f"  * Vision Analyzer: {'Active' if self.vision_analyzer else 'Disabled'}")
            logger.info(f"  * Auto Trading: {'Enabled' if self.auto_trading_enabled else 'Disabled'}")
            logger.info(f"  * Vision Auto-Analysis: {'Enabled' if self.vision_auto_analysis else 'Disabled'}")
            logger.info("==========================================")
            
            # Start Flask-SocketIO server
            self.socketio.run(
                self.app,
                host=host,
                port=port,
                debug=debug,
                use_reloader=False  # Disable reloader to prevent issues with threading
            )
            
        except Exception as e:
            logger.error(f"Error starting QNTI system: {e}")
            self.shutdown()
            raise
    
    def shutdown(self):
        """Gracefully shutdown the QNTI system"""
        try:
            logger.info("Shutting down QNTI Main System...")
            
            self.running = False
            
            # Stop background processes
            if self.mt5_bridge:
                try:
                    self.mt5_bridge.stop_monitoring()
                    self.mt5_bridge.shutdown()
                except Exception as e:
                    logger.error(f"Error shutting down MT5 bridge: {e}")
            
            if self.vision_analyzer:
                try:
                    self.vision_analyzer.stop_automated_analysis()
                except Exception as e:
                    logger.error(f"Error shutting down vision analyzer: {e}")
            
            # Save final state
            self._save_config()
            
            logger.info("QNTI shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown()
        sys.exit(0)

# CLI and main execution
def main():
    """Main entry point with improved error handling"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quantum Nexus Trading Intelligence (QNTI)")
    parser.add_argument('--config', default='qnti_config.json', help='Configuration file path')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no-auto-trading', action='store_true', help='Disable auto trading')
    parser.add_argument('--no-vision', action='store_true', help='Disable vision analysis')
    
    args = parser.parse_args()
    
    try:
        # Set console encoding for Windows
        if os.name == 'nt':  # Windows
            try:
                os.system('chcp 65001 >nul 2>&1')  # Set UTF-8 encoding
            except:
                pass
        
        # Initialize QNTI system
        qnti = QNTIMainSystem(config_file=args.config)
        
        # Override configuration with CLI arguments
        if args.no_auto_trading:
            qnti.auto_trading_enabled = False
        if args.no_vision:
            qnti.vision_auto_analysis = False
        
        # Start the system
        qnti.start(host=args.host, port=args.port, debug=args.debug)
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()