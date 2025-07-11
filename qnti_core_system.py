#!/usr/bin/env python3
"""
Quantum Nexus Trading Intelligence (QNTI) - Core System
Universal Trade Tracker & EA Management Module
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path
import asyncio
import websockets
import sqlite3
from threading import Lock
import threading
import time
import uuid
import csv
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qnti_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('QNTI')

class TradeSource(Enum):
    VISION_AI = "vision_ai"
    EXPERT_ADVISOR = "ea"
    MANUAL = "manual"
    HYBRID = "hybrid"

class TradeStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    PENDING = "pending"
    CANCELLED = "cancelled"

class EAStatus(Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    BLOCKED = "blocked"
    ERROR = "error"
    STOPPED = "stopped"

@dataclass
class Trade:
    """Universal trade structure for all trade sources"""
    trade_id: str
    magic_number: int
    symbol: str
    trade_type: str  # BUY, SELL, BUY_LIMIT, etc.
    lot_size: float
    open_price: float
    close_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    open_time: Optional[datetime] = None
    close_time: Optional[datetime] = None
    profit: Optional[float] = None
    commission: Optional[float] = None
    swap: Optional[float] = None
    source: TradeSource = TradeSource.MANUAL
    status: TradeStatus = TradeStatus.OPEN
    ea_name: Optional[str] = None
    ai_confidence: Optional[float] = None
    strategy_tags: Optional[List[str]] = None
    notes: Optional[str] = None
    
    def __post_init__(self):
        if self.open_time is None:
            self.open_time = datetime.now()
        if self.strategy_tags is None:
            self.strategy_tags = []

@dataclass
class EAPerformance:
    """Expert Advisor performance metrics"""
    ea_name: str
    magic_number: int
    symbol: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit: float = 0.0
    total_loss: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    avg_trade_duration: timedelta = timedelta(0)
    last_trade_time: Optional[datetime] = None
    status: EAStatus = EAStatus.ACTIVE
    risk_score: float = 0.0
    confidence_level: float = 0.0
    
    @property
    def net_profit(self) -> float:
        """Calculate net profit (total_profit - total_loss)"""
        return self.total_profit - self.total_loss
    
    def update_metrics(self, trades: List[Trade]):
        """Update performance metrics based on trade list"""
        ea_trades = [t for t in trades if t.ea_name == self.ea_name]
        
        if not ea_trades:
            return
            
        self.total_trades = len(ea_trades)
        self.winning_trades = len([t for t in ea_trades if t.profit and t.profit > 0])
        self.losing_trades = len([t for t in ea_trades if t.profit and t.profit < 0])
        
        self.total_profit = sum(t.profit for t in ea_trades if t.profit and t.profit > 0)
        self.total_loss = abs(sum(t.profit for t in ea_trades if t.profit and t.profit < 0))
        
        self.win_rate = (self.winning_trades / self.total_trades) * 100 if self.total_trades > 0 else 0
        self.profit_factor = self.total_profit / self.total_loss if self.total_loss > 0 else float('inf')
        
        # Calculate average trade duration
        closed_trades = [t for t in ea_trades if t.close_time and t.open_time]
        if closed_trades:
            durations = [(t.close_time - t.open_time).total_seconds() for t in closed_trades if t.close_time and t.open_time]
            self.avg_trade_duration = timedelta(seconds=float(np.mean(durations)))
        
        # Get last trade time, filtering out None values
        trade_times = [t.open_time for t in ea_trades if t.open_time is not None]
        self.last_trade_time = max(trade_times) if trade_times else None
        
        # Calculate risk score (higher = more risky)
        self.risk_score = self._calculate_risk_score(ea_trades)
        
    def _calculate_risk_score(self, trades: List[Trade]) -> float:
        """Calculate risk score based on trade patterns"""
        if not trades:
            return 0.0
            
        # Factors: drawdown, volatility, consecutive losses
        profits = [t.profit for t in trades if t.profit is not None]
        if not profits:
            return 0.0
            
        # Drawdown component
        cumulative = np.cumsum(profits)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (running_max - cumulative) / (running_max + 1e-10)
        max_drawdown = np.max(drawdown)
        
        # Volatility component
        volatility = np.std(profits) / (np.mean(profits) + 1e-10)
        
        # Consecutive loss component
        consecutive_losses = self._max_consecutive_losses(profits)
        
        risk_score = (max_drawdown * 0.4 + 
                     min(volatility, 5) * 0.4 + 
                     consecutive_losses * 0.2)
        
        return float(min(risk_score, 10.0))  # Cap at 10
    
    def _max_consecutive_losses(self, profits: List[float]) -> int:
        """Calculate maximum consecutive losses"""
        max_consecutive = 0
        current_consecutive = 0
        
        for profit in profits:
            if profit < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
                
        return max_consecutive

class QNTITradeManager:
    """Enhanced QNTI Trade Manager with comprehensive EA performance tracking"""
    
    def __init__(self, db_path: str = "qnti_data/qnti.db"):
        self.db_path = db_path
        self.trades: Dict[str, Trade] = {}
        self.ea_performances: Dict[str, EAPerformance] = {}
        self._lock = threading.Lock()
        
        # Add caching for EA performance
        self._ea_cache: Dict[str, Dict] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_duration = timedelta(seconds=30)  # Cache for 30 seconds
        
        # CRITICAL: Always ensure paths are initialized first
        self._ensure_paths_initialized()
        
        # Initialize database
        self._init_database()
        
        # Load existing data
        self._load_data()
        
        logger.info(f"Loaded {len(self.trades)} trades and {len(self.ea_performances)} EA profiles")
        logger.info("QNTI Trade Manager initialized successfully")
    
    def _ensure_paths_initialized(self):
        """Ensure all required file paths and attributes are initialized"""
        try:
            # Define file paths for data persistence
            if not hasattr(self, 'data_dir'):
                self.data_dir = Path("qnti_data")
                self.data_dir.mkdir(exist_ok=True)
            
            if not hasattr(self, 'trades_file'):
                self.trades_file = self.data_dir / "open_trades.json"
            
            if not hasattr(self, 'ea_performance_file'):
                self.ea_performance_file = self.data_dir / "ea_performance.json"
            
            if not hasattr(self, 'ea_controls_file'):
                self.ea_controls_file = self.data_dir / "ea_controls.json"
            
            if not hasattr(self, 'trade_log_file'):
                self.trade_log_file = self.data_dir / "trade_log.csv"
            
            # Initialize EA controls
            if not hasattr(self, 'ea_controls'):
                self.ea_controls = {}
                
            logger.info("File paths and attributes reinitialized")
            
        except Exception as e:
            logger.error(f"Error reinitializing paths: {e}")
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                magic_number INTEGER,
                symbol TEXT,
                trade_type TEXT,
                lot_size REAL,
                open_price REAL,
                close_price REAL,
                stop_loss REAL,
                take_profit REAL,
                open_time TIMESTAMP,
                close_time TIMESTAMP,
                profit REAL,
                commission REAL,
                swap REAL,
                source TEXT,
                status TEXT,
                ea_name TEXT,
                ai_confidence REAL,
                strategy_tags TEXT,
                notes TEXT
            )
        ''')
        
        # EA Performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ea_performance (
                ea_name TEXT PRIMARY KEY,
                magic_number INTEGER,
                symbol TEXT,
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                total_profit REAL,
                total_loss REAL,
                win_rate REAL,
                profit_factor REAL,
                max_drawdown REAL,
                avg_trade_duration REAL,
                last_trade_time TIMESTAMP,
                status TEXT,
                risk_score REAL,
                confidence_level REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_data(self):
        """Load existing trade and EA data"""
        try:
            # Ensure file paths are initialized - if not, reinitialize them
            if not hasattr(self, 'trades_file') or not hasattr(self, 'ea_controls'):
                logger.warning("File paths not initialized, initializing them now")
                self._ensure_paths_initialized()
                # Continue with loading after initialization
            
            # Load trades from JSON
            if self.trades_file.exists():
                with open(self.trades_file, 'r') as f:
                    trades_data = json.load(f)
                    for trade_id, trade_dict in trades_data.items():
                        # Convert datetime strings back to datetime objects
                        if 'open_time' in trade_dict:
                            trade_dict['open_time'] = datetime.fromisoformat(trade_dict['open_time'])
                        if 'close_time' in trade_dict and trade_dict['close_time']:
                            trade_dict['close_time'] = datetime.fromisoformat(trade_dict['close_time'])
                        
                        # Convert enums
                        trade_dict['source'] = TradeSource(trade_dict['source'])
                        trade_dict['status'] = TradeStatus(trade_dict['status'])
                        
                        self.trades[trade_id] = Trade(**trade_dict)
            
            # Load EA performance data
            if self.ea_performance_file.exists():
                with open(self.ea_performance_file, 'r') as f:
                    ea_data = json.load(f)
                    for ea_name, ea_dict in ea_data.items():
                        # Convert datetime and timedelta
                        if 'last_trade_time' in ea_dict and ea_dict['last_trade_time']:
                            ea_dict['last_trade_time'] = datetime.fromisoformat(ea_dict['last_trade_time'])
                        if 'avg_trade_duration' in ea_dict:
                            ea_dict['avg_trade_duration'] = timedelta(seconds=ea_dict['avg_trade_duration'])
                        
                        ea_dict['status'] = EAStatus(ea_dict['status'])
                        self.ea_performances[ea_name] = EAPerformance(**ea_dict)
            
            # Load EA controls
            if self.ea_controls_file.exists():
                with open(self.ea_controls_file, 'r') as f:
                    self.ea_controls = json.load(f)
                    
            logger.info(f"Loaded {len(self.trades)} trades and {len(self.ea_performances)} EA profiles")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
    
    def add_trade(self, trade: Trade) -> bool:
        """Add a new trade to the system"""
        try:
            # Ensure file paths are initialized
            if not hasattr(self, 'trades_file'):
                self._ensure_paths_initialized()
            
            with self._lock:
                self.trades[trade.trade_id] = trade
                
                # Update EA performance if it's an EA trade
                if trade.source == TradeSource.EXPERT_ADVISOR and trade.ea_name:
                    self._update_ea_performance(trade.ea_name, trade.symbol, trade.magic_number)
                
                self._save_trades()
                self._log_trade_to_csv(trade)
                
                logger.info(f"Added trade {trade.trade_id} from {trade.source.value}")
                return True
                
        except Exception as e:
            logger.error(f"Error adding trade: {e}")
            return False
    
    def close_trade(self, trade_id: str, close_price: float, close_time: Optional[datetime] = None) -> bool:
        """Close an existing trade"""
        try:
            with self._lock:
                if trade_id not in self.trades:
                    logger.warning(f"Trade {trade_id} not found")
                    return False
                
                trade = self.trades[trade_id]
                trade.close_price = close_price
                trade.close_time = close_time or datetime.now()
                trade.status = TradeStatus.CLOSED
                
                # Calculate profit
                if trade.trade_type in ['BUY', 'BUY_LIMIT']:
                    trade.profit = (close_price - trade.open_price) * trade.lot_size * 100000  # Simplified
                else:
                    trade.profit = (trade.open_price - close_price) * trade.lot_size * 100000
                
                # Update EA performance
                if trade.source == TradeSource.EXPERT_ADVISOR and trade.ea_name:
                    self._update_ea_performance(trade.ea_name, trade.symbol, trade.magic_number)
                
                self._save_trades()
                self._log_trade_to_csv(trade)
                
                logger.info(f"Closed trade {trade_id} with profit {trade.profit}")
                return True
                
        except Exception as e:
            logger.error(f"Error closing trade: {e}")
            return False
    
    def _update_ea_performance(self, ea_name: str, symbol: str, magic_number: int):
        """Update EA performance metrics"""
        if ea_name not in self.ea_performances:
            self.ea_performances[ea_name] = EAPerformance(
                ea_name=ea_name,
                magic_number=magic_number,
                symbol=symbol
            )
        
        # Update metrics based on current trades
        ea_trades = [t for t in self.trades.values() if t.ea_name == ea_name]
        self.ea_performances[ea_name].update_metrics(ea_trades)
        
        # Save updated performance
        self._save_ea_performance()
    
    def control_ea(self, ea_name: str, action: str, parameters: Optional[Dict] = None) -> bool:
        """Control EA behavior (pause, resume, block, modify parameters)"""
        try:
            with self._lock:
                if ea_name not in self.ea_controls:
                    self.ea_controls[ea_name] = {}
                
                self.ea_controls[ea_name]['action'] = action
                self.ea_controls[ea_name]['timestamp'] = datetime.now().isoformat()
                self.ea_controls[ea_name]['parameters'] = parameters or {}
                
                # Update EA status
                if ea_name in self.ea_performances:
                    if action == 'pause':
                        self.ea_performances[ea_name].status = EAStatus.PAUSED
                    elif action == 'resume':
                        self.ea_performances[ea_name].status = EAStatus.ACTIVE
                    elif action == 'block':
                        self.ea_performances[ea_name].status = EAStatus.BLOCKED
                    elif action == 'stop':
                        self.ea_performances[ea_name].status = EAStatus.STOPPED
                
                self._save_ea_controls()
                logger.info(f"Applied control action '{action}' to EA {ea_name}")
                return True
                
        except Exception as e:
            logger.error(f"Error controlling EA {ea_name}: {e}")
            return False
    
    def get_ea_recommendations(self) -> List[Dict]:
        """Get AI recommendations for EA management"""
        recommendations = []
        
        with self._lock:
            for ea_name, performance in self.ea_performances.items():
                # Risk-based recommendations
                if performance.risk_score > 7:
                    recommendations.append({
                        'ea_name': ea_name,
                        'action': 'reduce_risk',
                        'reason': f'High risk score: {performance.risk_score:.2f}',
                        'suggested_params': {'lot_multiplier': 0.5}
                    })
                
                # Performance-based recommendations
                if performance.win_rate < 30 and performance.total_trades > 10:
                    recommendations.append({
                        'ea_name': ea_name,
                        'action': 'pause',
                        'reason': f'Low win rate: {performance.win_rate:.1f}%',
                        'suggested_params': {}
                    })
                
                # Promote high performers
                if performance.profit_factor > 2 and performance.win_rate > 60:
                    recommendations.append({
                        'ea_name': ea_name,
                        'action': 'promote',
                        'reason': f'Excellent performance: PF={performance.profit_factor:.2f}, WR={performance.win_rate:.1f}%',
                        'suggested_params': {'lot_multiplier': 1.5}
                    })
        
        return recommendations
    
    def get_system_health(self) -> Dict:
        """Get overall system health metrics"""
        with self._lock:
            total_trades = len(self.trades)
            open_trades = len([t for t in self.trades.values() if t.status == TradeStatus.OPEN])
            active_eas = len([ea for ea in self.ea_performances.values() if ea.status == EAStatus.ACTIVE])
            
            # Calculate overall P&L
            total_profit = sum(t.profit for t in self.trades.values() if t.profit is not None)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'total_trades': total_trades,
                'open_trades': open_trades,
                'active_eas': active_eas,
                'total_profit': total_profit,
                'system_status': 'healthy',
                'mt5_connected': True,  # Placeholder
                'ai_api_status': 'connected',  # Placeholder
                'last_update': datetime.now().isoformat()
            }
    
    def _save_trades(self):
        """Save trades to JSON file"""
        try:
            # FIXED: Ensure file paths are initialized
            if not hasattr(self, 'trades_file'):
                self._ensure_paths_initialized()
            
            trades_dict = {}
            for trade_id, trade in self.trades.items():
                trade_dict = asdict(trade)
                # Convert datetime objects to ISO format
                if trade_dict['open_time']:
                    trade_dict['open_time'] = trade_dict['open_time'].isoformat()
                if trade_dict['close_time']:
                    trade_dict['close_time'] = trade_dict['close_time'].isoformat()
                # Convert enums to string
                trade_dict['source'] = trade_dict['source'].value
                trade_dict['status'] = trade_dict['status'].value
                trades_dict[trade_id] = trade_dict
            
            with open(self.trades_file, 'w') as f:
                json.dump(trades_dict, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving trades: {e}")
    

    def _save_ea_performance(self):
        """Save EA performance data"""
        ea_dict = {}
        for ea_name, performance in self.ea_performances.items():
            perf_dict = asdict(performance)
            # Convert datetime and timedelta
            if perf_dict['last_trade_time']:
                perf_dict['last_trade_time'] = perf_dict['last_trade_time'].isoformat()
            perf_dict['avg_trade_duration'] = perf_dict['avg_trade_duration'].total_seconds()
            perf_dict['status'] = perf_dict['status'].value
            ea_dict[ea_name] = perf_dict
        
        with open(self.ea_performance_file, 'w') as f:
            json.dump(ea_dict, f, indent=2)
    
    def _save_ea_controls(self):
        """Save EA control settings"""
        with open(self.ea_controls_file, 'w') as f:
            json.dump(self.ea_controls, f, indent=2)
    
    def _log_trade_to_csv(self, trade: Trade):
        """Log trade to CSV file"""
        try:
            # Ensure file paths are initialized
            if not hasattr(self, 'trade_log_file'):
                self._ensure_paths_initialized()
            
            trade_dict = asdict(trade)
            # Convert datetime objects
            if trade_dict['open_time']:
                trade_dict['open_time'] = trade_dict['open_time'].isoformat()
            if trade_dict['close_time']:
                trade_dict['close_time'] = trade_dict['close_time'].isoformat()
            # Convert enums
            trade_dict['source'] = trade_dict['source'].value
            trade_dict['status'] = trade_dict['status'].value
            # Convert list to string
            trade_dict['strategy_tags'] = ','.join(trade_dict['strategy_tags'])
            
            df = pd.DataFrame([trade_dict])
            
            # Append to CSV
            if self.trade_log_file.exists():
                df.to_csv(self.trade_log_file, mode='a', header=False, index=False)
            else:
                df.to_csv(self.trade_log_file, mode='w', header=True, index=False)
        except Exception as e:
            logger.error(f"Error logging trade to CSV: {e}")
    
    def recalculate_all_ea_performance(self):
        """Recalculate performance metrics for all EAs from existing trades"""
        try:
            with self._lock:
                updated_count = 0
                
                # Group trades by EA name
                ea_trade_groups = {}
                for trade in self.trades.values():
                    if trade.ea_name and trade.source == TradeSource.EXPERT_ADVISOR:
                        if trade.ea_name not in ea_trade_groups:
                            ea_trade_groups[trade.ea_name] = []
                        ea_trade_groups[trade.ea_name].append(trade)
                
                # Update performance for each EA
                for ea_name, ea_trades in ea_trade_groups.items():
                    if ea_name in self.ea_performances:
                        # Update metrics
                        self.ea_performances[ea_name].update_metrics(ea_trades)
                        updated_count += 1
                        logger.info(f"Recalculated performance for {ea_name}: "
                                  f"{len(ea_trades)} trades, {self.ea_performances[ea_name].win_rate:.1f}% win rate")
                
                # Save updated performance
                if updated_count > 0:
                    self._save_ea_performance()
                    logger.info(f"Recalculated performance metrics for {updated_count} EAs")
                
                return updated_count
                
        except Exception as e:
            logger.error(f"Error recalculating EA performance: {e}")
            return 0
    
    def associate_trades_with_eas(self):
        """Associate existing MT5 trades with EAs based on magic numbers"""
        try:
            with self._lock:
                updated_count = 0
                
                for trade in self.trades.values():
                    # Skip trades that already have EA names
                    if trade.ea_name or trade.source != TradeSource.EXPERT_ADVISOR:
                        continue
                    
                    # Find EA with matching magic number
                    for ea_name, performance in self.ea_performances.items():
                        if performance.magic_number == trade.magic_number:
                            trade.ea_name = ea_name
                            updated_count += 1
                            logger.debug(f"Associated trade {trade.trade_id} with EA {ea_name}")
                            break
                
                # Save updated trades
                if updated_count > 0:
                    self._save_trades()
                    logger.info(f"Associated {updated_count} trades with EAs")
                
                return updated_count
                
        except Exception as e:
            logger.error(f"Error associating trades with EAs: {e}")
            return 0

    def get_ea_performance_cached(self, ea_name: str) -> Optional[Dict]:
        """Get cached EA performance data"""
        try:
            with self._lock:
                # Check if cache is still valid
                if (self._cache_timestamp and 
                    datetime.now() - self._cache_timestamp < self._cache_duration):
                    return self._ea_cache.get(ea_name)
                
                # Cache expired or doesn't exist, return None to trigger refresh
                return None
        except Exception as e:
            logger.error(f"Error getting cached EA performance: {e}")
            return None
    
    def update_ea_performance_cache(self):
        """Update the EA performance cache"""
        try:
            with self._lock:
                self._ea_cache = {}
                
                for ea_name, performance in self.ea_performances.items():
                    self._ea_cache[ea_name] = {
                        "total_trades": performance.total_trades,
                        "win_rate": performance.win_rate,
                        "total_profit": performance.total_profit,
                        "profit_factor": performance.profit_factor,
                        "max_drawdown": performance.max_drawdown,
                        "avg_trade_duration": performance.avg_trade_duration.total_seconds(),
                        "avg_trade": round(performance.net_profit / performance.total_trades, 2) if performance.total_trades else 0.0,
                        "last_trade_time": performance.last_trade_time.isoformat() if performance.last_trade_time else None,
                        "status": performance.status,
                        "risk_score": performance.risk_score
                    }
                
                self._cache_timestamp = datetime.now()
                logger.debug(f"Updated EA performance cache for {len(self._ea_cache)} EAs")
                
        except Exception as e:
            logger.error(f"Error updating EA performance cache: {e}")

    def get_all_ea_performances_cached(self) -> Dict[str, Dict]:
        """Get all EA performances with caching"""
        try:
            with self._lock:
                # Check if cache is still valid
                if (self._cache_timestamp and 
                    datetime.now() - self._cache_timestamp < self._cache_duration):
                    return self._ea_cache.copy()
                
                # Cache expired, update it
                self.update_ea_performance_cache()
                return self._ea_cache.copy()
                
        except Exception as e:
            logger.error(f"Error getting cached EA performances: {e}")
            return {}

    def calculate_statistics(self) -> Dict[str, Any]:
        """Calculate high-level trade statistics used by dashboard analytics."""
        with self._lock:
            stats: Dict[str, Any] = {}
            total_trades = len(self.trades)
            closed_trades = [t for t in self.trades.values() if t.status == TradeStatus.CLOSED and t.profit is not None]
            winning_trades = [t for t in closed_trades if (t.profit or 0) > 0]
            losing_trades = [t for t in closed_trades if (t.profit or 0) < 0]
            
            # Aggregate P&L
            total_profit = sum([float(t.profit or 0.0) for t in winning_trades])
            total_loss = abs(sum([float(t.profit or 0.0) for t in losing_trades]))
            total_pnl = total_profit - total_loss
            
            win_rate = (len(winning_trades) / len(closed_trades) * 100) if closed_trades else 0.0
            profit_factor = (total_profit / total_loss) if total_loss > 0 else float('inf')
            
            if profit_factor == float('inf'):
                profit_factor_out = 999.99
            else:
                profit_factor_out = round(profit_factor, 2)
            
            stats.update({
                "total_trades": total_trades,
                "closed_trades": len(closed_trades),
                "winning_trades": len(winning_trades),
                "losing_trades": len(losing_trades),
                "total_pnl": round(total_pnl, 2),
                "win_rate": round(win_rate, 2),
                "profit_factor": profit_factor_out
            })
            return stats

    def get_ea_performance(self) -> Dict[str, Any]:
        """Return a plain dict representation of all EA performance objects for JSON serialization."""
        with self._lock:
            result: Dict[str, Any] = {}
            for ea_name, perf in self.ea_performances.items():
                # Ensure metrics are up-to-date
                perf.update_metrics(list(self.trades.values()))
                result[ea_name] = {
                    "name": perf.ea_name,  # Alias for front-end compatibility
                    "ea_name": perf.ea_name,
                    "magic_number": perf.magic_number,
                    "symbol": perf.symbol,
                    "total_trades": perf.total_trades,
                    "winning_trades": perf.winning_trades,
                    "losing_trades": perf.losing_trades,
                    "total_profit": round(perf.total_profit, 2),
                    "total_loss": round(perf.total_loss, 2),
                    "win_rate": round(perf.win_rate, 2),
                    "profit_factor": round(perf.profit_factor, 2) if perf.profit_factor != float('inf') else float('inf'),
                    "max_drawdown": round(perf.max_drawdown, 2),
                    "avg_trade_duration": perf.avg_trade_duration.total_seconds(),
                    "avg_trade": round(perf.net_profit / perf.total_trades, 2) if perf.total_trades else 0.0,
                    "last_trade_time": perf.last_trade_time.isoformat() if perf.last_trade_time else None,
                    "status": perf.status.value,
                    "risk_score": perf.risk_score,
                    "confidence_level": perf.confidence_level
                }
            return result

# Example usage and testing
if __name__ == "__main__":
    # Initialize the system
    qnti = QNTITradeManager()
    
    # Create sample trades
    trade1 = Trade(
        trade_id="T001",
        magic_number=12345,
        symbol="EURUSD",
        trade_type="BUY",
        lot_size=0.1,
        open_price=1.0500,
        source=TradeSource.EXPERT_ADVISOR,
        ea_name="TrendFollower_EA",
        ai_confidence=0.85,
        strategy_tags=["trend", "breakout"]
    )
    
    trade2 = Trade(
        trade_id="T002",
        magic_number=67890,
        symbol="GBPUSD",
        trade_type="SELL",
        lot_size=0.2,
        open_price=1.2800,
        source=TradeSource.VISION_AI,
        ai_confidence=0.92,
        strategy_tags=["reversal", "support_resistance"]
    )
    
    # Add trades
    qnti.add_trade(trade1)
    qnti.add_trade(trade2)
    
    # Close a trade
    qnti.close_trade("T001", 1.0550)
    
    # Get recommendations
    recommendations = qnti.get_ea_recommendations()
    print("EA Recommendations:", recommendations)
    
    # Get system health
    health = qnti.get_system_health()
    print("System Health:", health)
    
    # Control an EA
    qnti.control_ea("TrendFollower_EA", "pause", {"reason": "High drawdown detected"})
    
    print("QNTI Core System test completed successfully!")