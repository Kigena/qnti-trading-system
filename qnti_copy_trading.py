#!/usr/bin/env python3
"""
QNTI Copy Trading System
Implements leader/follower copy trading functionality with real-time synchronization
"""

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set
from uuid import uuid4

logger = logging.getLogger('QNTI_COPY_TRADING')

class TraderRole(Enum):
    """Trader role enumeration"""
    LEADER = "leader"
    FOLLOWER = "follower"
    INDEPENDENT = "independent"

class CopyMode(Enum):
    """Copy trading mode enumeration"""
    FULL_COPY = "full_copy"  # Copy all trades
    SELECTIVE_COPY = "selective_copy"  # Copy only selected symbols
    PROPORTIONAL_COPY = "proportional_copy"  # Copy with proportion adjustment
    MIRROR_COPY = "mirror_copy"  # Mirror trades exactly
    REVERSE_COPY = "reverse_copy"  # Reverse signal copy

class CopyStatus(Enum):
    """Copy relationship status"""
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    PENDING = "pending"

class RiskLevel(Enum):
    """Risk level enumeration"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"

@dataclass
class TraderProfile:
    """Trader profile information"""
    id: str
    name: str
    email: str
    role: TraderRole
    
    # Performance metrics
    total_trades: int = 0
    win_rate: float = 0.0
    average_profit: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    
    # Account information
    account_balance: float = 0.0
    account_equity: float = 0.0
    leverage: float = 1.0
    
    # Trading preferences
    preferred_symbols: List[str] = None
    max_position_size: float = 1.0
    risk_level: RiskLevel = RiskLevel.MODERATE
    
    # Leader specific
    followers_count: int = 0
    subscription_fee: float = 0.0  # Monthly fee
    performance_fee: float = 0.0  # Percentage of profits
    
    # Follower specific
    following_leaders: List[str] = None
    copy_settings: Dict = None
    
    # Timestamps
    created_at: datetime = None
    updated_at: datetime = None
    last_active: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.last_active is None:
            self.last_active = datetime.now()
        if self.preferred_symbols is None:
            self.preferred_symbols = []
        if self.following_leaders is None:
            self.following_leaders = []
        if self.copy_settings is None:
            self.copy_settings = {}

@dataclass
class CopyRelationship:
    """Copy trading relationship"""
    id: str
    leader_id: str
    follower_id: str
    copy_mode: CopyMode
    status: CopyStatus
    
    # Copy settings
    copy_ratio: float = 1.0  # Position size multiplier
    max_copy_amount: float = 1000.0  # Max amount to copy per trade
    allowed_symbols: List[str] = None
    blocked_symbols: List[str] = None
    
    # Risk management
    max_daily_loss: float = 100.0
    max_concurrent_trades: int = 5
    stop_loss_buffer: float = 0.0  # Additional SL buffer
    take_profit_buffer: float = 0.0  # Additional TP buffer
    
    # Performance tracking
    copied_trades: int = 0
    successful_copies: int = 0
    total_profit: float = 0.0
    
    # Timestamps
    created_at: datetime = None
    updated_at: datetime = None
    last_copy_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.allowed_symbols is None:
            self.allowed_symbols = []
        if self.blocked_symbols is None:
            self.blocked_symbols = []

@dataclass
class CopyTrade:
    """Copy trade record"""
    id: str
    relationship_id: str
    leader_trade_id: str
    follower_trade_id: str
    
    # Trade details
    symbol: str
    side: str
    volume: float
    open_price: float
    close_price: Optional[float] = None
    
    # Copy specific
    copy_ratio: float = 1.0
    leader_volume: float = 0.0
    
    # Performance
    profit: float = 0.0
    commission: float = 0.0
    
    # Timestamps
    copied_at: datetime = None
    closed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.copied_at is None:
            self.copied_at = datetime.now()

@dataclass
class LeaderboardEntry:
    """Leaderboard entry"""
    trader_id: str
    trader_name: str
    rank: int
    
    # Performance metrics
    profit_percentage: float
    win_rate: float
    total_trades: int
    followers_count: int
    risk_score: float
    
    # Recent performance
    monthly_return: float
    weekly_return: float
    daily_return: float
    
    # Risk metrics
    max_drawdown: float
    sharpe_ratio: float
    volatility: float

class QNTICopyTrading:
    """QNTI Copy Trading System"""
    
    def __init__(self, trade_manager, mt5_bridge=None, advanced_trading=None):
        self.trade_manager = trade_manager
        self.mt5_bridge = mt5_bridge
        self.advanced_trading = advanced_trading
        
        # Data storage
        self.traders: Dict[str, TraderProfile] = {}
        self.relationships: Dict[str, CopyRelationship] = {}
        self.copy_trades: List[CopyTrade] = []
        
        # Active monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        self.monitoring_interval = 1.0  # 1 second monitoring
        
        # Performance tracking
        self.performance_stats = {
            'total_relationships': 0,
            'active_relationships': 0,
            'total_copy_trades': 0,
            'successful_copies': 0,
            'total_profit': 0.0,
            'average_copy_delay': 0.0
        }
        
        # Event callbacks
        self.callbacks: Dict[str, List[Callable]] = {
            'trade_copied': [],
            'relationship_created': [],
            'relationship_updated': [],
            'leader_trade_opened': [],
            'leader_trade_closed': [],
            'copy_failed': []
        }
        
        # Risk management
        self.risk_settings = {
            'max_copy_delay': 5.0,  # seconds
            'max_slippage': 0.0005,  # 0.5 pips
            'min_leader_balance': 1000.0,
            'max_follower_relationships': 10,
            'max_leader_followers': 100
        }
        
        # Leader tracking
        self.leader_trades: Dict[str, List[str]] = {}  # leader_id -> [trade_ids]
        self.trade_listeners: Dict[str, Set[str]] = {}  # trade_id -> {relationship_ids}
        
        # Load existing data
        self._load_data()
        
        # Start monitoring
        self.start_monitoring()
        
        logger.info("Copy Trading System initialized")
    
    def start_monitoring(self):
        """Start copy trading monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logger.info("Copy trading monitoring started")
    
    def stop_monitoring(self):
        """Stop copy trading monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        logger.info("Copy trading monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self._monitor_leader_trades()
                self._update_trader_performance()
                self._check_risk_limits()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in copy trading monitoring loop: {e}")
                time.sleep(5.0)
    
    def _monitor_leader_trades(self):
        """Monitor leader trades for copying"""
        try:
            # Get current trades from trade manager
            if not self.trade_manager:
                return
            
            current_trades = getattr(self.trade_manager, 'trades', {})
            
            # Check for new leader trades
            for trade_id, trade in current_trades.items():
                if not hasattr(trade, 'ea_name'):
                    continue
                
                # Find leader by EA name or magic number
                leader_id = self._find_leader_by_trade(trade)
                if not leader_id:
                    continue
                
                # Check if this trade is already being monitored
                if trade_id in self.trade_listeners:
                    continue
                
                # Add trade to monitoring
                self.trade_listeners[trade_id] = set()
                
                # Find followers of this leader
                followers = self._get_leader_followers(leader_id)
                
                for follower_id in followers:
                    relationship = self._get_relationship(leader_id, follower_id)
                    if relationship and relationship.status == CopyStatus.ACTIVE:
                        # Check if trade should be copied
                        if self._should_copy_trade(trade, relationship):
                            self._copy_trade(trade, relationship)
                            self.trade_listeners[trade_id].add(relationship.id)
                            
        except Exception as e:
            logger.error(f"Error monitoring leader trades: {e}")
    
    def _find_leader_by_trade(self, trade) -> Optional[str]:
        """Find leader ID by trade properties"""
        try:
            # Check by EA name
            ea_name = getattr(trade, 'ea_name', None)
            if ea_name:
                for trader_id, trader in self.traders.items():
                    if trader.role == TraderRole.LEADER and ea_name in trader.preferred_symbols:
                        return trader_id
            
            # Check by magic number
            magic_number = getattr(trade, 'magic_number', None)
            if magic_number:
                for trader_id, trader in self.traders.items():
                    if (trader.role == TraderRole.LEADER and 
                        trader.copy_settings.get('magic_number') == magic_number):
                        return trader_id
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding leader by trade: {e}")
            return None
    
    def _get_leader_followers(self, leader_id: str) -> List[str]:
        """Get all followers of a leader"""
        try:
            followers = []
            for relationship in self.relationships.values():
                if (relationship.leader_id == leader_id and 
                    relationship.status == CopyStatus.ACTIVE):
                    followers.append(relationship.follower_id)
            return followers
        except Exception as e:
            logger.error(f"Error getting leader followers: {e}")
            return []
    
    def _get_relationship(self, leader_id: str, follower_id: str) -> Optional[CopyRelationship]:
        """Get relationship between leader and follower"""
        try:
            for relationship in self.relationships.values():
                if (relationship.leader_id == leader_id and 
                    relationship.follower_id == follower_id):
                    return relationship
            return None
        except Exception as e:
            logger.error(f"Error getting relationship: {e}")
            return None
    
    def _should_copy_trade(self, trade, relationship: CopyRelationship) -> bool:
        """Check if trade should be copied"""
        try:
            # Check symbol filters
            if relationship.allowed_symbols and trade.symbol not in relationship.allowed_symbols:
                return False
            
            if relationship.blocked_symbols and trade.symbol in relationship.blocked_symbols:
                return False
            
            # Check concurrent trades limit
            follower_active_trades = self._get_follower_active_trades(relationship.follower_id)
            if len(follower_active_trades) >= relationship.max_concurrent_trades:
                return False
            
            # Check daily loss limit
            today_loss = self._get_follower_daily_loss(relationship.follower_id)
            if today_loss >= relationship.max_daily_loss:
                return False
            
            # Check copy amount limit
            trade_amount = trade.volume * trade.open_price
            if trade_amount > relationship.max_copy_amount:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking if trade should be copied: {e}")
            return False
    
    def _copy_trade(self, leader_trade, relationship: CopyRelationship):
        """Copy a trade from leader to follower"""
        try:
            copy_start_time = time.time()
            
            # Calculate copy volume
            copy_volume = self._calculate_copy_volume(leader_trade, relationship)
            if copy_volume <= 0:
                return
            
            # Get current market price
            current_price = self._get_current_price(leader_trade.symbol)
            if current_price is None:
                return
            
            # Calculate adjusted stop loss and take profit
            adjusted_sl = self._calculate_adjusted_sl(leader_trade, relationship)
            adjusted_tp = self._calculate_adjusted_tp(leader_trade, relationship)
            
            # Create copy trade
            copy_trade_id = str(uuid4())
            
            # Execute copy trade
            success = False
            
            if self.advanced_trading:
                # Use advanced trading system
                copy_order_id = self.advanced_trading.place_bracket_order(
                    symbol=leader_trade.symbol,
                    side=leader_trade.side,
                    quantity=copy_volume,
                    price=current_price,
                    take_profit=adjusted_tp,
                    stop_loss=adjusted_sl,
                    ea_name=f"Copy_{relationship.leader_id}",
                    magic_number=relationship.follower_id.hash() % 100000,
                    comment=f"Copy of {leader_trade.id}"
                )
                success = copy_order_id is not None
                
            elif self.mt5_bridge:
                # Use MT5 bridge
                result = self.mt5_bridge.place_order(
                    symbol=leader_trade.symbol,
                    volume=copy_volume,
                    type=leader_trade.side,
                    price=current_price,
                    sl=adjusted_sl,
                    tp=adjusted_tp,
                    comment=f"Copy of {leader_trade.id}",
                    magic=relationship.follower_id.hash() % 100000
                )
                success = result.get('success', False)
                
            else:
                # Use trade manager
                from qnti_core_system import Trade, TradeSource, TradeStatus
                
                copy_trade = Trade(
                    id=copy_trade_id,
                    symbol=leader_trade.symbol,
                    side=leader_trade.side,
                    volume=copy_volume,
                    open_price=current_price,
                    stop_loss=adjusted_sl,
                    take_profit=adjusted_tp,
                    source=TradeSource.COPY_TRADING,
                    status=TradeStatus.ACTIVE,
                    ea_name=f"Copy_{relationship.leader_id}",
                    magic_number=relationship.follower_id.hash() % 100000
                )
                
                self.trade_manager.add_trade(copy_trade)
                success = True
            
            if success:
                # Record copy trade
                copy_trade_record = CopyTrade(
                    id=str(uuid4()),
                    relationship_id=relationship.id,
                    leader_trade_id=leader_trade.id,
                    follower_trade_id=copy_trade_id,
                    symbol=leader_trade.symbol,
                    side=leader_trade.side,
                    volume=copy_volume,
                    open_price=current_price,
                    copy_ratio=relationship.copy_ratio,
                    leader_volume=leader_trade.volume
                )
                
                self.copy_trades.append(copy_trade_record)
                
                # Update relationship stats
                relationship.copied_trades += 1
                relationship.last_copy_at = datetime.now()
                relationship.updated_at = datetime.now()
                
                # Update performance stats
                self.performance_stats['total_copy_trades'] += 1
                self.performance_stats['successful_copies'] += 1
                
                copy_delay = time.time() - copy_start_time
                self._update_average_copy_delay(copy_delay)
                
                # Trigger callbacks
                self._trigger_callback('trade_copied', {
                    'leader_trade': leader_trade,
                    'copy_trade': copy_trade_record,
                    'relationship': relationship
                })
                
                logger.info(f"Trade copied successfully: {leader_trade.id} -> {copy_trade_id}")
                
            else:
                # Record copy failure
                self._trigger_callback('copy_failed', {
                    'leader_trade': leader_trade,
                    'relationship': relationship,
                    'reason': 'Execution failed'
                })
                
                logger.warning(f"Failed to copy trade: {leader_trade.id}")
                
        except Exception as e:
            logger.error(f"Error copying trade: {e}")
            self._trigger_callback('copy_failed', {
                'leader_trade': leader_trade,
                'relationship': relationship,
                'reason': str(e)
            })
    
    def _calculate_copy_volume(self, leader_trade, relationship: CopyRelationship) -> float:
        """Calculate copy volume based on relationship settings"""
        try:
            base_volume = leader_trade.volume * relationship.copy_ratio
            
            # Apply volume limits
            follower = self.traders.get(relationship.follower_id)
            if follower and base_volume > follower.max_position_size:
                base_volume = follower.max_position_size
            
            # Apply copy amount limit
            trade_amount = base_volume * leader_trade.open_price
            if trade_amount > relationship.max_copy_amount:
                base_volume = relationship.max_copy_amount / leader_trade.open_price
            
            return max(0.01, base_volume)  # Minimum 0.01 lots
            
        except Exception as e:
            logger.error(f"Error calculating copy volume: {e}")
            return 0.0
    
    def _calculate_adjusted_sl(self, leader_trade, relationship: CopyRelationship) -> Optional[float]:
        """Calculate adjusted stop loss with buffer"""
        try:
            if not leader_trade.stop_loss:
                return None
            
            buffer = relationship.stop_loss_buffer
            if leader_trade.side == 'BUY':
                return leader_trade.stop_loss - buffer
            else:
                return leader_trade.stop_loss + buffer
                
        except Exception as e:
            logger.error(f"Error calculating adjusted SL: {e}")
            return leader_trade.stop_loss
    
    def _calculate_adjusted_tp(self, leader_trade, relationship: CopyRelationship) -> Optional[float]:
        """Calculate adjusted take profit with buffer"""
        try:
            if not leader_trade.take_profit:
                return None
            
            buffer = relationship.take_profit_buffer
            if leader_trade.side == 'BUY':
                return leader_trade.take_profit + buffer
            else:
                return leader_trade.take_profit - buffer
                
        except Exception as e:
            logger.error(f"Error calculating adjusted TP: {e}")
            return leader_trade.take_profit
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price"""
        try:
            if self.mt5_bridge:
                symbol_info = self.mt5_bridge.get_symbol_info(symbol)
                if symbol_info:
                    return symbol_info.get('bid', 0.0)
            return None
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    def _get_follower_active_trades(self, follower_id: str) -> List:
        """Get active trades for a follower"""
        try:
            active_trades = []
            for copy_trade in self.copy_trades:
                if (copy_trade.relationship_id in self.relationships and
                    self.relationships[copy_trade.relationship_id].follower_id == follower_id and
                    copy_trade.closed_at is None):
                    active_trades.append(copy_trade)
            return active_trades
        except Exception as e:
            logger.error(f"Error getting follower active trades: {e}")
            return []
    
    def _get_follower_daily_loss(self, follower_id: str) -> float:
        """Get daily loss for a follower"""
        try:
            today = datetime.now().date()
            daily_loss = 0.0
            
            for copy_trade in self.copy_trades:
                if (copy_trade.relationship_id in self.relationships and
                    self.relationships[copy_trade.relationship_id].follower_id == follower_id and
                    copy_trade.copied_at.date() == today and
                    copy_trade.profit < 0):
                    daily_loss += abs(copy_trade.profit)
            
            return daily_loss
        except Exception as e:
            logger.error(f"Error getting follower daily loss: {e}")
            return 0.0
    
    def _update_trader_performance(self):
        """Update trader performance metrics"""
        try:
            for trader_id, trader in self.traders.items():
                if trader.role == TraderRole.LEADER:
                    self._update_leader_performance(trader)
                elif trader.role == TraderRole.FOLLOWER:
                    self._update_follower_performance(trader)
        except Exception as e:
            logger.error(f"Error updating trader performance: {e}")
    
    def _update_leader_performance(self, leader: TraderProfile):
        """Update leader performance metrics"""
        try:
            # Get leader's copy trades
            leader_copy_trades = [
                ct for ct in self.copy_trades
                if ct.relationship_id in self.relationships and
                self.relationships[ct.relationship_id].leader_id == leader.id
            ]
            
            if not leader_copy_trades:
                return
            
            # Calculate metrics
            total_trades = len(leader_copy_trades)
            winning_trades = sum(1 for ct in leader_copy_trades if ct.profit > 0)
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            total_profit = sum(ct.profit for ct in leader_copy_trades)
            average_profit = total_profit / total_trades if total_trades > 0 else 0
            
            # Update leader profile
            leader.total_trades = total_trades
            leader.win_rate = win_rate
            leader.average_profit = average_profit
            leader.updated_at = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating leader performance: {e}")
    
    def _update_follower_performance(self, follower: TraderProfile):
        """Update follower performance metrics"""
        try:
            # Get follower's copy trades
            follower_copy_trades = [
                ct for ct in self.copy_trades
                if ct.relationship_id in self.relationships and
                self.relationships[ct.relationship_id].follower_id == follower.id
            ]
            
            if not follower_copy_trades:
                return
            
            # Calculate metrics
            total_trades = len(follower_copy_trades)
            winning_trades = sum(1 for ct in follower_copy_trades if ct.profit > 0)
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            total_profit = sum(ct.profit for ct in follower_copy_trades)
            average_profit = total_profit / total_trades if total_trades > 0 else 0
            
            # Update follower profile
            follower.total_trades = total_trades
            follower.win_rate = win_rate
            follower.average_profit = average_profit
            follower.updated_at = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating follower performance: {e}")
    
    def _check_risk_limits(self):
        """Check risk limits for all relationships"""
        try:
            for relationship in self.relationships.values():
                if relationship.status != CopyStatus.ACTIVE:
                    continue
                
                # Check daily loss limit
                daily_loss = self._get_follower_daily_loss(relationship.follower_id)
                if daily_loss >= relationship.max_daily_loss:
                    self.pause_relationship(relationship.id, "Daily loss limit exceeded")
                
                # Check follower account balance
                follower = self.traders.get(relationship.follower_id)
                if follower and follower.account_balance < self.risk_settings['min_leader_balance']:
                    self.pause_relationship(relationship.id, "Insufficient account balance")
                    
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
    
    def _update_average_copy_delay(self, delay: float):
        """Update average copy delay statistic"""
        try:
            current_avg = self.performance_stats['average_copy_delay']
            total_copies = self.performance_stats['successful_copies']
            
            if total_copies == 1:
                self.performance_stats['average_copy_delay'] = delay
            else:
                self.performance_stats['average_copy_delay'] = (
                    (current_avg * (total_copies - 1) + delay) / total_copies
                )
        except Exception as e:
            logger.error(f"Error updating average copy delay: {e}")
    
    def _trigger_callback(self, event_type: str, data: Any):
        """Trigger event callbacks"""
        try:
            if event_type in self.callbacks:
                for callback in self.callbacks[event_type]:
                    try:
                        callback(data)
                    except Exception as e:
                        logger.error(f"Error in callback {callback}: {e}")
        except Exception as e:
            logger.error(f"Error triggering callback {event_type}: {e}")
    
    def _load_data(self):
        """Load existing data from storage"""
        try:
            # Load from JSON files if they exist
            import os
            
            if os.path.exists('copy_trading_data.json'):
                with open('copy_trading_data.json', 'r') as f:
                    data = json.load(f)
                    
                    # Load traders
                    for trader_data in data.get('traders', []):
                        trader = TraderProfile(**trader_data)
                        trader.created_at = datetime.fromisoformat(trader_data['created_at'])
                        trader.updated_at = datetime.fromisoformat(trader_data['updated_at'])
                        trader.last_active = datetime.fromisoformat(trader_data['last_active'])
                        self.traders[trader.id] = trader
                    
                    # Load relationships
                    for rel_data in data.get('relationships', []):
                        relationship = CopyRelationship(**rel_data)
                        relationship.created_at = datetime.fromisoformat(rel_data['created_at'])
                        relationship.updated_at = datetime.fromisoformat(rel_data['updated_at'])
                        if rel_data.get('last_copy_at'):
                            relationship.last_copy_at = datetime.fromisoformat(rel_data['last_copy_at'])
                        self.relationships[relationship.id] = relationship
                        
                logger.info("Copy trading data loaded successfully")
                
        except Exception as e:
            logger.error(f"Error loading copy trading data: {e}")
    
    def _save_data(self):
        """Save data to storage"""
        try:
            data = {
                'traders': [asdict(trader) for trader in self.traders.values()],
                'relationships': [asdict(rel) for rel in self.relationships.values()]
            }
            
            # Convert datetime objects to ISO format
            for trader_data in data['traders']:
                trader_data['created_at'] = trader_data['created_at'].isoformat()
                trader_data['updated_at'] = trader_data['updated_at'].isoformat()
                trader_data['last_active'] = trader_data['last_active'].isoformat()
            
            for rel_data in data['relationships']:
                rel_data['created_at'] = rel_data['created_at'].isoformat()
                rel_data['updated_at'] = rel_data['updated_at'].isoformat()
                if rel_data.get('last_copy_at'):
                    rel_data['last_copy_at'] = rel_data['last_copy_at'].isoformat()
            
            with open('copy_trading_data.json', 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving copy trading data: {e}")
    
    # Public API methods
    
    def create_trader_profile(self, name: str, email: str, role: TraderRole, **kwargs) -> str:
        """Create a new trader profile"""
        try:
            trader_id = str(uuid4())
            trader = TraderProfile(
                id=trader_id,
                name=name,
                email=email,
                role=role,
                **kwargs
            )
            
            self.traders[trader_id] = trader
            self._save_data()
            
            logger.info(f"Trader profile created: {trader_id} ({name})")
            return trader_id
            
        except Exception as e:
            logger.error(f"Error creating trader profile: {e}")
            return None
    
    def create_copy_relationship(self, leader_id: str, follower_id: str, 
                               copy_mode: CopyMode, **kwargs) -> str:
        """Create a copy trading relationship"""
        try:
            # Validate participants
            if leader_id not in self.traders or follower_id not in self.traders:
                raise ValueError("Invalid leader or follower ID")
            
            leader = self.traders[leader_id]
            follower = self.traders[follower_id]
            
            if leader.role != TraderRole.LEADER:
                raise ValueError("Leader must have LEADER role")
            
            if follower.role != TraderRole.FOLLOWER:
                raise ValueError("Follower must have FOLLOWER role")
            
            # Check limits
            if leader.followers_count >= self.risk_settings['max_leader_followers']:
                raise ValueError("Leader has too many followers")
            
            follower_relationships = sum(1 for r in self.relationships.values() 
                                       if r.follower_id == follower_id)
            if follower_relationships >= self.risk_settings['max_follower_relationships']:
                raise ValueError("Follower has too many relationships")
            
            # Create relationship
            relationship_id = str(uuid4())
            relationship = CopyRelationship(
                id=relationship_id,
                leader_id=leader_id,
                follower_id=follower_id,
                copy_mode=copy_mode,
                status=CopyStatus.ACTIVE,
                **kwargs
            )
            
            self.relationships[relationship_id] = relationship
            
            # Update counters
            leader.followers_count += 1
            follower.following_leaders.append(leader_id)
            
            # Update performance stats
            self.performance_stats['total_relationships'] += 1
            self.performance_stats['active_relationships'] += 1
            
            self._save_data()
            
            # Trigger callback
            self._trigger_callback('relationship_created', relationship)
            
            logger.info(f"Copy relationship created: {relationship_id}")
            return relationship_id
            
        except Exception as e:
            logger.error(f"Error creating copy relationship: {e}")
            return None
    
    def pause_relationship(self, relationship_id: str, reason: str = "") -> bool:
        """Pause a copy relationship"""
        try:
            if relationship_id not in self.relationships:
                return False
            
            relationship = self.relationships[relationship_id]
            relationship.status = CopyStatus.PAUSED
            relationship.updated_at = datetime.now()
            
            if reason:
                relationship.copy_settings['pause_reason'] = reason
            
            self.performance_stats['active_relationships'] -= 1
            
            self._save_data()
            
            # Trigger callback
            self._trigger_callback('relationship_updated', relationship)
            
            logger.info(f"Relationship paused: {relationship_id} - {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Error pausing relationship: {e}")
            return False
    
    def resume_relationship(self, relationship_id: str) -> bool:
        """Resume a paused copy relationship"""
        try:
            if relationship_id not in self.relationships:
                return False
            
            relationship = self.relationships[relationship_id]
            relationship.status = CopyStatus.ACTIVE
            relationship.updated_at = datetime.now()
            
            if 'pause_reason' in relationship.copy_settings:
                del relationship.copy_settings['pause_reason']
            
            self.performance_stats['active_relationships'] += 1
            
            self._save_data()
            
            # Trigger callback
            self._trigger_callback('relationship_updated', relationship)
            
            logger.info(f"Relationship resumed: {relationship_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error resuming relationship: {e}")
            return False
    
    def stop_relationship(self, relationship_id: str) -> bool:
        """Stop a copy relationship"""
        try:
            if relationship_id not in self.relationships:
                return False
            
            relationship = self.relationships[relationship_id]
            relationship.status = CopyStatus.STOPPED
            relationship.updated_at = datetime.now()
            
            # Update counters
            leader = self.traders.get(relationship.leader_id)
            if leader:
                leader.followers_count -= 1
            
            follower = self.traders.get(relationship.follower_id)
            if follower and relationship.leader_id in follower.following_leaders:
                follower.following_leaders.remove(relationship.leader_id)
            
            self.performance_stats['active_relationships'] -= 1
            
            self._save_data()
            
            # Trigger callback
            self._trigger_callback('relationship_updated', relationship)
            
            logger.info(f"Relationship stopped: {relationship_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping relationship: {e}")
            return False
    
    def get_trader_profile(self, trader_id: str) -> Optional[Dict]:
        """Get trader profile"""
        try:
            if trader_id in self.traders:
                trader = self.traders[trader_id]
                return {
                    'id': trader.id,
                    'name': trader.name,
                    'email': trader.email,
                    'role': trader.role.value,
                    'total_trades': trader.total_trades,
                    'win_rate': trader.win_rate,
                    'average_profit': trader.average_profit,
                    'max_drawdown': trader.max_drawdown,
                    'sharpe_ratio': trader.sharpe_ratio,
                    'account_balance': trader.account_balance,
                    'account_equity': trader.account_equity,
                    'followers_count': trader.followers_count,
                    'following_leaders': trader.following_leaders,
                    'created_at': trader.created_at.isoformat(),
                    'updated_at': trader.updated_at.isoformat(),
                    'last_active': trader.last_active.isoformat()
                }
            return None
        except Exception as e:
            logger.error(f"Error getting trader profile: {e}")
            return None
    
    def get_leaderboard(self, limit: int = 20) -> List[LeaderboardEntry]:
        """Get trader leaderboard"""
        try:
            leaders = [t for t in self.traders.values() if t.role == TraderRole.LEADER]
            
            # Sort by profit percentage (descending)
            leaders.sort(key=lambda x: x.average_profit, reverse=True)
            
            leaderboard = []
            for rank, leader in enumerate(leaders[:limit], 1):
                entry = LeaderboardEntry(
                    trader_id=leader.id,
                    trader_name=leader.name,
                    rank=rank,
                    profit_percentage=leader.average_profit,
                    win_rate=leader.win_rate,
                    total_trades=leader.total_trades,
                    followers_count=leader.followers_count,
                    risk_score=self._calculate_risk_score(leader),
                    monthly_return=self._calculate_monthly_return(leader.id),
                    weekly_return=self._calculate_weekly_return(leader.id),
                    daily_return=self._calculate_daily_return(leader.id),
                    max_drawdown=leader.max_drawdown,
                    sharpe_ratio=leader.sharpe_ratio,
                    volatility=self._calculate_volatility(leader.id)
                )
                leaderboard.append(entry)
            
            return leaderboard
            
        except Exception as e:
            logger.error(f"Error getting leaderboard: {e}")
            return []
    
    def get_copy_relationships(self, trader_id: str) -> List[Dict]:
        """Get copy relationships for a trader"""
        try:
            relationships = []
            for rel in self.relationships.values():
                if rel.leader_id == trader_id or rel.follower_id == trader_id:
                    relationships.append({
                        'id': rel.id,
                        'leader_id': rel.leader_id,
                        'follower_id': rel.follower_id,
                        'copy_mode': rel.copy_mode.value,
                        'status': rel.status.value,
                        'copy_ratio': rel.copy_ratio,
                        'copied_trades': rel.copied_trades,
                        'successful_copies': rel.successful_copies,
                        'total_profit': rel.total_profit,
                        'created_at': rel.created_at.isoformat(),
                        'updated_at': rel.updated_at.isoformat()
                    })
            return relationships
        except Exception as e:
            logger.error(f"Error getting copy relationships: {e}")
            return []
    
    def get_copy_history(self, relationship_id: str, limit: int = 100) -> List[Dict]:
        """Get copy trade history for a relationship"""
        try:
            history = []
            for copy_trade in self.copy_trades:
                if copy_trade.relationship_id == relationship_id:
                    history.append({
                        'id': copy_trade.id,
                        'leader_trade_id': copy_trade.leader_trade_id,
                        'follower_trade_id': copy_trade.follower_trade_id,
                        'symbol': copy_trade.symbol,
                        'side': copy_trade.side,
                        'volume': copy_trade.volume,
                        'open_price': copy_trade.open_price,
                        'close_price': copy_trade.close_price,
                        'profit': copy_trade.profit,
                        'copied_at': copy_trade.copied_at.isoformat(),
                        'closed_at': copy_trade.closed_at.isoformat() if copy_trade.closed_at else None
                    })
            
            # Sort by copied_at descending
            history.sort(key=lambda x: x['copied_at'], reverse=True)
            return history[:limit]
            
        except Exception as e:
            logger.error(f"Error getting copy history: {e}")
            return []
    
    def get_performance_stats(self) -> Dict:
        """Get copy trading performance statistics"""
        try:
            stats = self.performance_stats.copy()
            stats['total_traders'] = len(self.traders)
            stats['total_leaders'] = sum(1 for t in self.traders.values() if t.role == TraderRole.LEADER)
            stats['total_followers'] = sum(1 for t in self.traders.values() if t.role == TraderRole.FOLLOWER)
            return stats
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {}
    
    def _calculate_risk_score(self, trader: TraderProfile) -> float:
        """Calculate risk score for a trader"""
        try:
            # Simple risk score based on drawdown and volatility
            risk_score = 50.0  # Base score
            
            if trader.max_drawdown > 0.2:
                risk_score += 30.0
            elif trader.max_drawdown > 0.1:
                risk_score += 15.0
            
            if trader.win_rate < 40:
                risk_score += 20.0
            elif trader.win_rate > 70:
                risk_score -= 10.0
            
            return min(100.0, max(0.0, risk_score))
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 50.0
    
    def _calculate_monthly_return(self, trader_id: str) -> float:
        """Calculate monthly return for a trader"""
        try:
            # Simplified calculation
            trader = self.traders.get(trader_id)
            if not trader:
                return 0.0
            
            return trader.average_profit * 30  # Approximate monthly return
            
        except Exception as e:
            logger.error(f"Error calculating monthly return: {e}")
            return 0.0
    
    def _calculate_weekly_return(self, trader_id: str) -> float:
        """Calculate weekly return for a trader"""
        try:
            # Simplified calculation
            trader = self.traders.get(trader_id)
            if not trader:
                return 0.0
            
            return trader.average_profit * 7  # Approximate weekly return
            
        except Exception as e:
            logger.error(f"Error calculating weekly return: {e}")
            return 0.0
    
    def _calculate_daily_return(self, trader_id: str) -> float:
        """Calculate daily return for a trader"""
        try:
            # Simplified calculation
            trader = self.traders.get(trader_id)
            if not trader:
                return 0.0
            
            return trader.average_profit  # Daily return
            
        except Exception as e:
            logger.error(f"Error calculating daily return: {e}")
            return 0.0
    
    def _calculate_volatility(self, trader_id: str) -> float:
        """Calculate volatility for a trader"""
        try:
            # Simplified volatility calculation
            trader = self.traders.get(trader_id)
            if not trader:
                return 0.0
            
            return abs(trader.max_drawdown) * 100  # Approximate volatility
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.0
    
    def add_callback(self, event_type: str, callback: Callable):
        """Add event callback"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
    
    def remove_callback(self, event_type: str, callback: Callable):
        """Remove event callback"""
        if event_type in self.callbacks and callback in self.callbacks[event_type]:
            self.callbacks[event_type].remove(callback)
    
    def update_risk_settings(self, settings: Dict):
        """Update risk management settings"""
        try:
            self.risk_settings.update(settings)
            logger.info("Copy trading risk settings updated")
        except Exception as e:
            logger.error(f"Error updating risk settings: {e}")
    
    def shutdown(self):
        """Shutdown the copy trading system"""
        try:
            self.stop_monitoring()
            self._save_data()
            logger.info("Copy trading system shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")