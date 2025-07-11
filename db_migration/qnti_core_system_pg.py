#!/usr/bin/env python3
"""
Quantum Nexus Trading Intelligence (QNTI) - Core System PostgreSQL Version
Universal Trade Tracker & EA Management Module with PostgreSQL backend
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
import threading
import time
import uuid
import csv
import traceback

# Import PostgreSQL database manager
from database_config import get_database_manager, execute_query, execute_command, get_cursor

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

class QNTITradeManagerPG:
    """Enhanced QNTI Trade Manager with PostgreSQL backend"""
    
    def __init__(self):
        self.trades: Dict[str, Trade] = {}
        self.ea_performances: Dict[str, EAPerformance] = {}
        self._lock = threading.Lock()
        
        # Database manager
        self.db_manager = get_database_manager()
        
        # Add caching for EA performance
        self._ea_cache: Dict[str, Dict] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_duration = timedelta(seconds=30)  # Cache for 30 seconds
        
        # EA controls
        self.ea_controls: Dict[str, Dict] = {}
        
        # Load existing data
        self._load_data()
        
        logger.info(f"Loaded {len(self.trades)} trades and {len(self.ea_performances)} EA profiles")
        logger.info("QNTI Trade Manager (PostgreSQL) initialized successfully")
    
    def _load_data(self):
        """Load existing trade and EA data from PostgreSQL"""
        try:
            # Load trades
            trades_query = "SELECT * FROM trades ORDER BY open_time DESC LIMIT 1000"
            trades_data = execute_query(trades_query)
            
            for row in trades_data:
                trade = Trade(
                    trade_id=row['trade_id'],
                    magic_number=int(row['magic_number']) if row['magic_number'] else 0,
                    symbol=row['symbol'],
                    trade_type=row['trade_type'],
                    lot_size=float(row['lot_size']),
                    open_price=float(row['open_price']),
                    close_price=float(row['close_price']) if row['close_price'] else None,
                    stop_loss=float(row['stop_loss']) if row['stop_loss'] else None,
                    take_profit=float(row['take_profit']) if row['take_profit'] else None,
                    open_time=row['open_time'],
                    close_time=row['close_time'],
                    profit=float(row['profit']) if row['profit'] else None,
                    commission=float(row['commission']) if row['commission'] else None,
                    swap=float(row['swap']) if row['swap'] else None,
                    source=TradeSource(row['source']),
                    status=TradeStatus(row['status']),
                    ea_name=row['ea_name'],
                    ai_confidence=float(row['ai_confidence']) if row['ai_confidence'] else None,
                    strategy_tags=row['strategy_tags'] if row['strategy_tags'] else [],
                    notes=row['notes']
                )
                self.trades[trade.trade_id] = trade
            
            # Load EA performance data
            ea_perf_query = "SELECT * FROM ea_performance"
            ea_perf_data = execute_query(ea_perf_query)
            
            for row in ea_perf_data:
                ea_perf = EAPerformance(
                    ea_name=row['ea_name'],
                    magic_number=int(row['magic_number']) if row['magic_number'] else 0,
                    symbol=row['symbol'] or '',
                    total_trades=int(row['total_trades']),
                    winning_trades=int(row['winning_trades']),
                    losing_trades=int(row['losing_trades']),
                    total_profit=float(row['total_profit']),
                    total_loss=float(row['total_loss']),
                    win_rate=float(row['win_rate']),
                    profit_factor=float(row['profit_factor']),
                    max_drawdown=float(row['max_drawdown']),
                    avg_trade_duration=row['avg_trade_duration'] or timedelta(0),
                    last_trade_time=row['last_trade_time'],
                    status=EAStatus(row['status']),
                    risk_score=float(row['risk_score']),
                    confidence_level=float(row['confidence_level'])
                )
                self.ea_performances[ea_perf.ea_name] = ea_perf
            
            logger.info(f"Loaded {len(self.trades)} trades and {len(self.ea_performances)} EA profiles")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
    
    def add_trade(self, trade: Trade) -> bool:
        """Add a new trade to the system"""
        try:
            with self._lock:
                self.trades[trade.trade_id] = trade
                
                # Update EA performance if it's an EA trade
                if trade.source == TradeSource.EXPERT_ADVISOR and trade.ea_name:
                    self._update_ea_performance(trade.ea_name, trade.symbol, trade.magic_number)
                
                self._save_trade(trade)
                
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
                
                self._save_trade(trade)
                
                logger.info(f"Closed trade {trade_id} with profit {trade.profit}")
                return True
                
        except Exception as e:
            logger.error(f"Error closing trade: {e}")
            return False
    
    def _update_ea_performance(self, ea_name: str, symbol: str, magic_number: int):
        """Update EA performance metrics using PostgreSQL queries"""
        try:
            # Get or create EA performance record
            if ea_name not in self.ea_performances:
                self.ea_performances[ea_name] = EAPerformance(
                    ea_name=ea_name,
                    magic_number=magic_number,
                    symbol=symbol
                )
            
            # Calculate metrics using PostgreSQL queries
            metrics_query = """
            SELECT 
                COUNT(*) as total_trades,
                COUNT(CASE WHEN profit > 0 THEN 1 END) as winning_trades,
                COUNT(CASE WHEN profit < 0 THEN 1 END) as losing_trades,
                COALESCE(SUM(CASE WHEN profit > 0 THEN profit ELSE 0 END), 0) as total_profit,
                COALESCE(ABS(SUM(CASE WHEN profit < 0 THEN profit ELSE 0 END)), 0) as total_loss,
                MAX(open_time) as last_trade_time,
                AVG(CASE WHEN close_time IS NOT NULL AND open_time IS NOT NULL 
                         THEN EXTRACT(EPOCH FROM (close_time - open_time)) 
                         ELSE NULL END) as avg_duration_seconds
            FROM trades 
            WHERE ea_name = %s AND status = 'closed' AND profit IS NOT NULL
            """
            
            result = execute_query(metrics_query, (ea_name,), fetch_all=False)
            
            if result:
                ea_perf = self.ea_performances[ea_name]
                ea_perf.total_trades = int(result['total_trades'])
                ea_perf.winning_trades = int(result['winning_trades'])
                ea_perf.losing_trades = int(result['losing_trades'])
                ea_perf.total_profit = float(result['total_profit'])
                ea_perf.total_loss = float(result['total_loss'])
                ea_perf.last_trade_time = result['last_trade_time']
                
                # Calculate derived metrics
                ea_perf.win_rate = (ea_perf.winning_trades / ea_perf.total_trades) * 100 if ea_perf.total_trades > 0 else 0
                ea_perf.profit_factor = ea_perf.total_profit / ea_perf.total_loss if ea_perf.total_loss > 0 else float('inf')
                
                # Average trade duration
                if result['avg_duration_seconds']:
                    ea_perf.avg_trade_duration = timedelta(seconds=float(result['avg_duration_seconds']))
                
                # Calculate risk score using PostgreSQL function
                try:
                    risk_query = """
                    SELECT 
                        COALESCE(STDDEV(profit), 0) as volatility,
                        COUNT(*) as trade_count
                    FROM trades 
                    WHERE ea_name = %s AND status = 'closed' AND profit IS NOT NULL
                    """
                    
                    risk_result = execute_query(risk_query, (ea_name,), fetch_all=False)
                    if risk_result and risk_result['trade_count'] > 0:
                        volatility = float(risk_result['volatility'])
                        mean_profit = ea_perf.net_profit / ea_perf.total_trades if ea_perf.total_trades > 0 else 0
                        normalized_volatility = volatility / (abs(mean_profit) + 1e-10) if mean_profit != 0 else 0
                        ea_perf.risk_score = min(normalized_volatility, 10.0)
                    
                except Exception as e:
                    logger.warning(f"Error calculating risk score for {ea_name}: {e}")
                    ea_perf.risk_score = 0.0
                
                # Save updated performance
                self._save_ea_performance(ea_perf)
            
        except Exception as e:
            logger.error(f"Error updating EA performance for {ea_name}: {e}")
    
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
                    
                    # Update in database
                    self._save_ea_performance(self.ea_performances[ea_name])
                
                # Log control action
                self._log_system_event('ea_control', f"Applied '{action}' to EA {ea_name}", {
                    'ea_name': ea_name,
                    'action': action,
                    'parameters': parameters
                })
                
                logger.info(f"Applied control action '{action}' to EA {ea_name}")
                return True
                
        except Exception as e:
            logger.error(f"Error controlling EA {ea_name}: {e}")
            return False
    
    def get_ea_recommendations(self) -> List[Dict]:
        """Get AI recommendations for EA management using PostgreSQL analytics"""
        recommendations = []
        
        try:
            with self._lock:
                # Get EA performance analytics from PostgreSQL
                analytics_query = """
                SELECT 
                    ep.ea_name,
                    ep.win_rate,
                    ep.profit_factor,
                    ep.total_trades,
                    ep.risk_score,
                    ep.last_trade_time,
                    COUNT(t.id) as recent_trades,
                    AVG(t.profit) as avg_recent_profit
                FROM ea_performance ep
                LEFT JOIN trades t ON ep.ea_name = t.ea_name 
                    AND t.open_time >= (CURRENT_TIMESTAMP - INTERVAL '7 days')
                WHERE ep.total_trades > 0
                GROUP BY ep.ea_name, ep.win_rate, ep.profit_factor, ep.total_trades, 
                         ep.risk_score, ep.last_trade_time
                ORDER BY ep.total_trades DESC
                """
                
                ea_analytics = execute_query(analytics_query)
                
                for row in ea_analytics:
                    ea_name = row['ea_name']
                    win_rate = float(row['win_rate'])
                    profit_factor = float(row['profit_factor'])
                    total_trades = int(row['total_trades'])
                    risk_score = float(row['risk_score'])
                    recent_trades = int(row['recent_trades']) if row['recent_trades'] else 0
                    avg_recent_profit = float(row['avg_recent_profit']) if row['avg_recent_profit'] else 0
                    
                    # Risk-based recommendations
                    if risk_score > 7:
                        recommendations.append({
                            'ea_name': ea_name,
                            'action': 'reduce_risk',
                            'reason': f'High risk score: {risk_score:.2f}',
                            'suggested_params': {'lot_multiplier': 0.5},
                            'priority': 'high'
                        })
                    
                    # Performance-based recommendations
                    if win_rate < 30 and total_trades > 10:
                        recommendations.append({
                            'ea_name': ea_name,
                            'action': 'pause',
                            'reason': f'Low win rate: {win_rate:.1f}%',
                            'suggested_params': {},
                            'priority': 'medium'
                        })
                    
                    # Promote high performers
                    if profit_factor > 2 and win_rate > 60:
                        recommendations.append({
                            'ea_name': ea_name,
                            'action': 'promote',
                            'reason': f'Excellent performance: PF={profit_factor:.2f}, WR={win_rate:.1f}%',
                            'suggested_params': {'lot_multiplier': 1.5},
                            'priority': 'low'
                        })
                    
                    # Inactivity check
                    if recent_trades == 0 and row['last_trade_time']:
                        days_since_last_trade = (datetime.now() - row['last_trade_time']).days
                        if days_since_last_trade > 7:
                            recommendations.append({
                                'ea_name': ea_name,
                                'action': 'check_activity',
                                'reason': f'No trades for {days_since_last_trade} days',
                                'suggested_params': {},
                                'priority': 'medium'
                            })
                
        except Exception as e:
            logger.error(f"Error generating EA recommendations: {e}")
        
        return recommendations
    
    def get_system_health(self) -> Dict:
        """Get overall system health metrics using PostgreSQL queries"""
        try:
            with self._lock:
                # Get system metrics from PostgreSQL
                health_query = """
                SELECT 
                    (SELECT COUNT(*) FROM trades) as total_trades,
                    (SELECT COUNT(*) FROM trades WHERE status = 'open') as open_trades,
                    (SELECT COUNT(*) FROM ea_performance WHERE status = 'active') as active_eas,
                    (SELECT COALESCE(SUM(profit), 0) FROM trades WHERE profit IS NOT NULL) as total_profit,
                    (SELECT COUNT(*) FROM trades WHERE open_time >= (CURRENT_TIMESTAMP - INTERVAL '24 hours')) as trades_24h,
                    (SELECT COUNT(*) FROM ea_performance WHERE last_trade_time >= (CURRENT_TIMESTAMP - INTERVAL '24 hours')) as active_eas_24h
                """
                
                health_data = execute_query(health_query, fetch_all=False)
                
                # Get pool statistics
                pool_stats = self.db_manager.get_pool_stats()
                
                return {
                    'timestamp': datetime.now().isoformat(),
                    'total_trades': int(health_data['total_trades']) if health_data else 0,
                    'open_trades': int(health_data['open_trades']) if health_data else 0,
                    'active_eas': int(health_data['active_eas']) if health_data else 0,
                    'total_profit': float(health_data['total_profit']) if health_data else 0.0,
                    'trades_24h': int(health_data['trades_24h']) if health_data else 0,
                    'active_eas_24h': int(health_data['active_eas_24h']) if health_data else 0,
                    'system_status': 'healthy',
                    'database_type': 'PostgreSQL',
                    'database_pool': pool_stats,
                    'last_update': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'system_status': 'error',
                'error': str(e),
                'database_type': 'PostgreSQL'
            }
    
    def _save_trade(self, trade: Trade):
        """Save trade to PostgreSQL database"""
        try:
            upsert_query = """
            INSERT INTO trades (
                trade_id, magic_number, symbol, trade_type, lot_size,
                open_price, close_price, stop_loss, take_profit,
                open_time, close_time, profit, commission, swap,
                source, status, ea_name, ai_confidence, strategy_tags, notes
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (trade_id) DO UPDATE SET
                magic_number = EXCLUDED.magic_number,
                symbol = EXCLUDED.symbol,
                trade_type = EXCLUDED.trade_type,
                lot_size = EXCLUDED.lot_size,
                open_price = EXCLUDED.open_price,
                close_price = EXCLUDED.close_price,
                stop_loss = EXCLUDED.stop_loss,
                take_profit = EXCLUDED.take_profit,
                open_time = EXCLUDED.open_time,
                close_time = EXCLUDED.close_time,
                profit = EXCLUDED.profit,
                commission = EXCLUDED.commission,
                swap = EXCLUDED.swap,
                source = EXCLUDED.source,
                status = EXCLUDED.status,
                ea_name = EXCLUDED.ea_name,
                ai_confidence = EXCLUDED.ai_confidence,
                strategy_tags = EXCLUDED.strategy_tags,
                notes = EXCLUDED.notes
            """
            
            params = (
                trade.trade_id, trade.magic_number, trade.symbol, trade.trade_type,
                trade.lot_size, trade.open_price, trade.close_price,
                trade.stop_loss, trade.take_profit, trade.open_time, trade.close_time,
                trade.profit, trade.commission, trade.swap, trade.source.value,
                trade.status.value, trade.ea_name, trade.ai_confidence,
                json.dumps(trade.strategy_tags), trade.notes
            )
            
            execute_command(upsert_query, params)
            
        except Exception as e:
            logger.error(f"Error saving trade: {e}")
    
    def _save_ea_performance(self, ea_performance: EAPerformance):
        """Save EA performance to PostgreSQL database"""
        try:
            upsert_query = """
            INSERT INTO ea_performance (
                ea_name, magic_number, symbol, total_trades, winning_trades,
                losing_trades, total_profit, total_loss, win_rate,
                profit_factor, max_drawdown, avg_trade_duration,
                last_trade_time, status, risk_score, confidence_level
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (ea_name) DO UPDATE SET
                magic_number = EXCLUDED.magic_number,
                symbol = EXCLUDED.symbol,
                total_trades = EXCLUDED.total_trades,
                winning_trades = EXCLUDED.winning_trades,
                losing_trades = EXCLUDED.losing_trades,
                total_profit = EXCLUDED.total_profit,
                total_loss = EXCLUDED.total_loss,
                win_rate = EXCLUDED.win_rate,
                profit_factor = EXCLUDED.profit_factor,
                max_drawdown = EXCLUDED.max_drawdown,
                avg_trade_duration = EXCLUDED.avg_trade_duration,
                last_trade_time = EXCLUDED.last_trade_time,
                status = EXCLUDED.status,
                risk_score = EXCLUDED.risk_score,
                confidence_level = EXCLUDED.confidence_level,
                last_updated = CURRENT_TIMESTAMP
            """
            
            params = (
                ea_performance.ea_name, ea_performance.magic_number, ea_performance.symbol,
                ea_performance.total_trades, ea_performance.winning_trades,
                ea_performance.losing_trades, ea_performance.total_profit,
                ea_performance.total_loss, ea_performance.win_rate,
                ea_performance.profit_factor, ea_performance.max_drawdown,
                ea_performance.avg_trade_duration, ea_performance.last_trade_time,
                ea_performance.status.value, ea_performance.risk_score,
                ea_performance.confidence_level
            )
            
            execute_command(upsert_query, params)
            
        except Exception as e:
            logger.error(f"Error saving EA performance: {e}")
    
    def _log_system_event(self, module: str, message: str, details: Dict = None):
        """Log system event to PostgreSQL"""
        try:
            log_query = """
            INSERT INTO system_logs (level, module, message, details)
            VALUES (%s, %s, %s, %s)
            """
            
            params = ('INFO', module, message, json.dumps(details or {}))
            execute_command(log_query, params)
            
        except Exception as e:
            logger.error(f"Error logging system event: {e}")
    
    def recalculate_all_ea_performance(self):
        """Recalculate performance metrics for all EAs from existing trades"""
        try:
            with self._lock:
                # Get all EA names from trades
                ea_names_query = """
                SELECT DISTINCT ea_name 
                FROM trades 
                WHERE ea_name IS NOT NULL AND source = 'ea'
                """
                
                ea_names_data = execute_query(ea_names_query)
                updated_count = 0
                
                for row in ea_names_data:
                    ea_name = row['ea_name']
                    if ea_name:
                        # Get EA details
                        ea_details_query = """
                        SELECT magic_number, symbol 
                        FROM trades 
                        WHERE ea_name = %s 
                        ORDER BY open_time DESC 
                        LIMIT 1
                        """
                        
                        ea_details = execute_query(ea_details_query, (ea_name,), fetch_all=False)
                        
                        if ea_details:
                            magic_number = int(ea_details['magic_number']) if ea_details['magic_number'] else 0
                            symbol = ea_details['symbol'] or ''
                            
                            # Update performance
                            self._update_ea_performance(ea_name, symbol, magic_number)
                            updated_count += 1
                            
                            logger.info(f"Recalculated performance for {ea_name}")
                
                logger.info(f"Recalculated performance metrics for {updated_count} EAs")
                return updated_count
                
        except Exception as e:
            logger.error(f"Error recalculating EA performance: {e}")
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
                        "status": performance.status.value,
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
        """Calculate high-level trade statistics using PostgreSQL queries"""
        try:
            with self._lock:
                stats_query = """
                SELECT 
                    COUNT(*) as total_trades,
                    COUNT(CASE WHEN status = 'closed' AND profit IS NOT NULL THEN 1 END) as closed_trades,
                    COUNT(CASE WHEN status = 'closed' AND profit > 0 THEN 1 END) as winning_trades,
                    COUNT(CASE WHEN status = 'closed' AND profit < 0 THEN 1 END) as losing_trades,
                    COALESCE(SUM(CASE WHEN profit > 0 THEN profit ELSE 0 END), 0) as total_profit,
                    COALESCE(ABS(SUM(CASE WHEN profit < 0 THEN profit ELSE 0 END)), 0) as total_loss
                FROM trades
                """
                
                result = execute_query(stats_query, fetch_all=False)
                
                if result:
                    total_trades = int(result['total_trades'])
                    closed_trades = int(result['closed_trades'])
                    winning_trades = int(result['winning_trades'])
                    losing_trades = int(result['losing_trades'])
                    total_profit = float(result['total_profit'])
                    total_loss = float(result['total_loss'])
                    
                    total_pnl = total_profit - total_loss
                    win_rate = (winning_trades / closed_trades * 100) if closed_trades > 0 else 0.0
                    profit_factor = (total_profit / total_loss) if total_loss > 0 else float('inf')
                    
                    if profit_factor == float('inf'):
                        profit_factor_out = 999.99
                    else:
                        profit_factor_out = round(profit_factor, 2)
                    
                    return {
                        "total_trades": total_trades,
                        "closed_trades": closed_trades,
                        "winning_trades": winning_trades,
                        "losing_trades": losing_trades,
                        "total_pnl": round(total_pnl, 2),
                        "win_rate": round(win_rate, 2),
                        "profit_factor": profit_factor_out
                    }
                else:
                    return {
                        "total_trades": 0,
                        "closed_trades": 0,
                        "winning_trades": 0,
                        "losing_trades": 0,
                        "total_pnl": 0.0,
                        "win_rate": 0.0,
                        "profit_factor": 0.0
                    }
                    
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            return {
                "total_trades": 0,
                "closed_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "total_pnl": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "error": str(e)
            }
    
    def get_ea_performance(self) -> Dict[str, Any]:
        """Return EA performance data optimized for JSON serialization"""
        try:
            with self._lock:
                # Get EA performance data from PostgreSQL with recent trade counts
                ea_performance_query = """
                SELECT 
                    ep.*,
                    COUNT(t.id) as recent_trades_count
                FROM ea_performance ep
                LEFT JOIN trades t ON ep.ea_name = t.ea_name 
                    AND t.open_time >= (CURRENT_TIMESTAMP - INTERVAL '7 days')
                GROUP BY ep.ea_name, ep.magic_number, ep.symbol, ep.total_trades, 
                         ep.winning_trades, ep.losing_trades, ep.total_profit, 
                         ep.total_loss, ep.win_rate, ep.profit_factor, ep.max_drawdown, 
                         ep.avg_trade_duration, ep.last_trade_time, ep.status, 
                         ep.risk_score, ep.confidence_level
                ORDER BY ep.total_trades DESC
                """
                
                ea_data = execute_query(ea_performance_query)
                result = {}
                
                for row in ea_data:
                    ea_name = row['ea_name']
                    avg_trade_duration = row['avg_trade_duration']
                    avg_duration_seconds = avg_trade_duration.total_seconds() if avg_trade_duration else 0
                    
                    result[ea_name] = {
                        "name": ea_name,
                        "ea_name": ea_name,
                        "magic_number": int(row['magic_number']) if row['magic_number'] else 0,
                        "symbol": row['symbol'] or '',
                        "total_trades": int(row['total_trades']),
                        "winning_trades": int(row['winning_trades']),
                        "losing_trades": int(row['losing_trades']),
                        "total_profit": round(float(row['total_profit']), 2),
                        "total_loss": round(float(row['total_loss']), 2),
                        "win_rate": round(float(row['win_rate']), 2),
                        "profit_factor": round(float(row['profit_factor']), 2) if row['profit_factor'] != float('inf') else 999.99,
                        "max_drawdown": round(float(row['max_drawdown']), 2),
                        "avg_trade_duration": avg_duration_seconds,
                        "avg_trade": round((float(row['total_profit']) - float(row['total_loss'])) / int(row['total_trades']), 2) if int(row['total_trades']) > 0 else 0.0,
                        "last_trade_time": row['last_trade_time'].isoformat() if row['last_trade_time'] else None,
                        "status": row['status'],
                        "risk_score": float(row['risk_score']),
                        "confidence_level": float(row['confidence_level']),
                        "recent_trades_count": int(row['recent_trades_count']) if row['recent_trades_count'] else 0
                    }
                
                return result
                
        except Exception as e:
            logger.error(f"Error getting EA performance: {e}")
            return {}

# Example usage and testing
if __name__ == "__main__":
    # Initialize the system
    qnti = QNTITradeManagerPG()
    
    # Test system health
    health = qnti.get_system_health()
    print("System Health:", json.dumps(health, indent=2))
    
    # Test statistics
    stats = qnti.calculate_statistics()
    print("\nTrading Statistics:", json.dumps(stats, indent=2))
    
    # Test EA performance
    ea_performance = qnti.get_ea_performance()
    print(f"\nEA Performance: {len(ea_performance)} EAs loaded")
    
    # Test recommendations
    recommendations = qnti.get_ea_recommendations()
    print(f"\nEA Recommendations: {len(recommendations)} recommendations")
    
    # Create sample trade
    trade1 = Trade(
        trade_id=f"T_{uuid.uuid4().hex[:8]}",
        magic_number=12345,
        symbol="EURUSD",
        trade_type="BUY",
        lot_size=0.1,
        open_price=1.0500,
        source=TradeSource.EXPERT_ADVISOR,
        ea_name="TestEA_PG",
        ai_confidence=0.85,
        strategy_tags=["trend", "breakout"]
    )
    
    # Add trade
    if qnti.add_trade(trade1):
        print(f"\nAdded test trade: {trade1.trade_id}")
        
        # Close trade
        if qnti.close_trade(trade1.trade_id, 1.0550):
            print(f"Closed test trade with profit")
    
    print("\nQNTI PostgreSQL Core System test completed successfully!")
