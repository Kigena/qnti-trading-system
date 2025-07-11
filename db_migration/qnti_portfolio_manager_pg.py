#!/usr/bin/env python3
"""
QNTI Portfolio Manager - PostgreSQL Version
Advanced portfolio management system with PostgreSQL backend
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import uuid

# Import PostgreSQL database manager
from database_config import get_database_manager, execute_query, execute_command

logger = logging.getLogger(__name__)

class AccountType(Enum):
    """Account types"""
    DEMO = "demo"
    LIVE = "live"
    PROP_FIRM = "prop_firm"
    PERSONAL = "personal"

class AllocationMethod(Enum):
    """Portfolio allocation methods"""
    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"
    MARKET_CAP = "market_cap"
    CUSTOM = "custom"

class PortfolioStatus(Enum):
    """Portfolio status"""
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"

@dataclass
class Account:
    """Account data structure"""
    id: str
    name: str
    account_type: AccountType
    broker: str
    login: str
    server: str
    currency: str
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float
    profit: float
    leverage: int
    is_active: bool
    created_at: datetime
    last_updated: datetime
    metadata: Dict[str, Any]

@dataclass
class Position:
    """Position data structure"""
    id: str
    account_id: str
    symbol: str
    side: str  # buy/sell
    volume: float
    open_price: float
    current_price: float
    profit_loss: float
    swap: float
    commission: float
    open_time: datetime
    magic_number: int
    comment: str

@dataclass
class Portfolio:
    """Portfolio data structure"""
    id: str
    name: str
    description: str
    account_ids: List[str]
    allocation_method: AllocationMethod
    allocation_weights: Dict[str, float]
    total_value: float
    total_profit: float
    daily_pnl: float
    status: PortfolioStatus
    created_at: datetime
    last_updated: datetime
    performance_metrics: Dict[str, Any]

@dataclass
class PortfolioSnapshot:
    """Portfolio performance snapshot"""
    portfolio_id: str
    timestamp: datetime
    total_value: float
    total_profit: float
    daily_pnl: float
    account_values: Dict[str, float]
    positions_count: int
    metrics: Dict[str, Any]

class QNTIPortfolioManagerPG:
    """PostgreSQL-based portfolio management system for QNTI"""
    
    def __init__(self):
        self.accounts: Dict[str, Account] = {}
        self.portfolios: Dict[str, Portfolio] = {}
        self.positions: Dict[str, Position] = {}
        self.running = False
        self.update_thread = None
        
        # Performance tracking
        self.performance_history: Dict[str, List[PortfolioSnapshot]] = {}
        
        # Database manager
        self.db_manager = get_database_manager()
        
        # Load existing data
        self._load_data()
        
        logger.info("QNTI Portfolio Manager (PostgreSQL) initialized")
    
    def _load_data(self):
        """Load existing data from PostgreSQL database"""
        try:
            # Load accounts
            accounts_query = "SELECT * FROM accounts WHERE is_active = true"
            accounts_data = execute_query(accounts_query)
            
            for row in accounts_data:
                account = Account(
                    id=row['id'],
                    name=row['name'],
                    account_type=AccountType(row['account_type']),
                    broker=row['broker'],
                    login=row['login'],
                    server=row['server'],
                    currency=row['currency'],
                    balance=float(row['balance']),
                    equity=float(row['equity']),
                    margin=float(row['margin']),
                    free_margin=float(row['free_margin']),
                    margin_level=float(row['margin_level']),
                    profit=float(row['profit']),
                    leverage=int(row['leverage']),
                    is_active=bool(row['is_active']),
                    created_at=row['created_at'],
                    last_updated=row['last_updated'],
                    metadata=row['metadata'] or {}
                )
                self.accounts[account.id] = account
            
            # Load portfolios
            portfolios_query = "SELECT * FROM portfolios WHERE status = 'active'"
            portfolios_data = execute_query(portfolios_query)
            
            for row in portfolios_data:
                portfolio = Portfolio(
                    id=row['id'],
                    name=row['name'],
                    description=row['description'],
                    account_ids=row['account_ids'],
                    allocation_method=AllocationMethod(row['allocation_method']),
                    allocation_weights=row['allocation_weights'],
                    total_value=float(row['total_value']),
                    total_profit=float(row['total_profit']),
                    daily_pnl=float(row['daily_pnl']),
                    status=PortfolioStatus(row['status']),
                    created_at=row['created_at'],
                    last_updated=row['last_updated'],
                    performance_metrics=row['performance_metrics'] or {}
                )
                self.portfolios[portfolio.id] = portfolio
            
            # Load positions
            positions_query = """
            SELECT p.*, a.id as account_id_str
            FROM positions p
            JOIN accounts a ON p.account_id = a.id
            """
            positions_data = execute_query(positions_query)
            
            for row in positions_data:
                position = Position(
                    id=row['id'],
                    account_id=row['account_id'],
                    symbol=row['symbol'],
                    side=row['side'],
                    volume=float(row['volume']),
                    open_price=float(row['open_price']),
                    current_price=float(row['current_price']),
                    profit_loss=float(row['profit_loss']),
                    swap=float(row['swap']),
                    commission=float(row['commission']),
                    open_time=row['open_time'],
                    magic_number=int(row['magic_number']) if row['magic_number'] else 0,
                    comment=row['comment'] or ""
                )
                self.positions[position.id] = position
            
            logger.info(f"Loaded {len(self.accounts)} accounts, {len(self.portfolios)} portfolios, {len(self.positions)} positions")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
    
    def add_account(self, name: str, account_type: AccountType, broker: str, 
                   login: str, server: str, currency: str = "USD", 
                   leverage: int = 100, metadata: Dict = None) -> str:
        """Add a new trading account"""
        try:
            account_id = str(uuid.uuid4())
            
            account = Account(
                id=account_id,
                name=name,
                account_type=account_type,
                broker=broker,
                login=login,
                server=server,
                currency=currency,
                balance=0.0,
                equity=0.0,
                margin=0.0,
                free_margin=0.0,
                margin_level=0.0,
                profit=0.0,
                leverage=leverage,
                is_active=True,
                created_at=datetime.now(),
                last_updated=datetime.now(),
                metadata=metadata or {}
            )
            
            self.accounts[account_id] = account
            self._save_account(account)
            
            logger.info(f"Added account: {name} ({account_id})")
            return account_id
            
        except Exception as e:
            logger.error(f"Error adding account: {e}")
            raise
    
    def create_portfolio(self, name: str, description: str, account_ids: List[str],
                        allocation_method: AllocationMethod = AllocationMethod.EQUAL_WEIGHT,
                        custom_weights: Dict[str, float] = None) -> str:
        """Create a new portfolio"""
        try:
            # Validate account IDs
            for account_id in account_ids:
                if account_id not in self.accounts:
                    raise ValueError(f"Account {account_id} not found")
            
            portfolio_id = str(uuid.uuid4())
            
            # Calculate allocation weights
            if allocation_method == AllocationMethod.EQUAL_WEIGHT:
                weight = 1.0 / len(account_ids)
                allocation_weights = {acc_id: weight for acc_id in account_ids}
            elif allocation_method == AllocationMethod.CUSTOM and custom_weights:
                allocation_weights = custom_weights
            else:
                allocation_weights = {acc_id: 1.0 / len(account_ids) for acc_id in account_ids}
            
            portfolio = Portfolio(
                id=portfolio_id,
                name=name,
                description=description,
                account_ids=account_ids,
                allocation_method=allocation_method,
                allocation_weights=allocation_weights,
                total_value=0.0,
                total_profit=0.0,
                daily_pnl=0.0,
                status=PortfolioStatus.ACTIVE,
                created_at=datetime.now(),
                last_updated=datetime.now(),
                performance_metrics={}
            )
            
            self.portfolios[portfolio_id] = portfolio
            self._save_portfolio(portfolio)
            
            logger.info(f"Created portfolio: {name} ({portfolio_id})")
            return portfolio_id
            
        except Exception as e:
            logger.error(f"Error creating portfolio: {e}")
            raise
    
    def update_account_data(self, account_id: str, account_data: Dict):
        """Update account data from MT5 or other sources"""
        try:
            if account_id not in self.accounts:
                logger.warning(f"Account {account_id} not found")
                return
            
            account = self.accounts[account_id]
            
            # Update account fields
            account.balance = account_data.get('balance', account.balance)
            account.equity = account_data.get('equity', account.equity)
            account.margin = account_data.get('margin', account.margin)
            account.free_margin = account_data.get('free_margin', account.free_margin)
            account.margin_level = account_data.get('margin_level', account.margin_level)
            account.profit = account_data.get('profit', account.profit)
            account.last_updated = datetime.now()
            
            self._save_account(account)
            
        except Exception as e:
            logger.error(f"Error updating account data: {e}")
    
    def update_positions(self, account_id: str, positions_data: List[Dict]):
        """Update positions for an account"""
        try:
            # Remove existing positions for this account
            account_positions = [pos_id for pos_id, pos in self.positions.items() 
                               if pos.account_id == account_id]
            
            # Delete from database
            if account_positions:
                delete_query = "DELETE FROM positions WHERE account_id = %s"
                execute_command(delete_query, (account_id,))
                
                # Remove from memory
                for pos_id in account_positions:
                    del self.positions[pos_id]
            
            # Add new positions
            for pos_data in positions_data:
                position_id = str(uuid.uuid4())
                
                position = Position(
                    id=position_id,
                    account_id=account_id,
                    symbol=pos_data.get('symbol', ''),
                    side=pos_data.get('type', 'buy'),
                    volume=pos_data.get('volume', 0.0),
                    open_price=pos_data.get('price_open', 0.0),
                    current_price=pos_data.get('price_current', 0.0),
                    profit_loss=pos_data.get('profit', 0.0),
                    swap=pos_data.get('swap', 0.0),
                    commission=pos_data.get('commission', 0.0),
                    open_time=datetime.fromisoformat(pos_data.get('time', datetime.now().isoformat())),
                    magic_number=pos_data.get('magic', 0),
                    comment=pos_data.get('comment', '')
                )
                
                self.positions[position_id] = position
                self._save_position(position)
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    def calculate_portfolio_metrics(self, portfolio_id: str) -> Dict[str, Any]:
        """Calculate comprehensive portfolio metrics using PostgreSQL functions"""
        try:
            if portfolio_id not in self.portfolios:
                raise ValueError(f"Portfolio {portfolio_id} not found")
            
            portfolio = self.portfolios[portfolio_id]
            
            # Use PostgreSQL function to calculate total value
            total_value_query = "SELECT calculate_portfolio_total_value(%s) as total_value"
            result = execute_query(total_value_query, (portfolio_id,), fetch_all=False)
            calculated_total_value = float(result['total_value']) if result else 0.0
            
            # Get account data with optimized query
            account_values_query = """
            SELECT a.id, a.equity, aw.weight
            FROM accounts a
            JOIN (
                SELECT unnest(account_ids) as account_id,
                       unnest(array(SELECT value FROM jsonb_each_text(allocation_weights))) as weight
                FROM portfolios p
                WHERE p.id = %s
            ) aw ON a.id::text = aw.account_id
            WHERE a.is_active = true
            """
            
            account_data = execute_query(account_values_query, (portfolio_id,))
            
            account_values = {}
            total_value = 0.0
            total_profit = 0.0
            
            for row in account_data:
                account_id = row['id']
                equity = float(row['equity'])
                weight = float(row['weight']) if row['weight'] else 0.0
                
                account_values[account_id] = equity
                total_value += equity * weight
                
                # Get profit from account
                if account_id in self.accounts:
                    total_profit += self.accounts[account_id].profit * weight
            
            # Count positions
            positions_count_query = """
            SELECT COUNT(*) as count
            FROM positions p
            WHERE p.account_id = ANY(%s)
            """
            
            positions_result = execute_query(positions_count_query, (portfolio.account_ids,), fetch_all=False)
            positions_count = int(positions_result['count']) if positions_result else 0
            
            # Calculate performance metrics
            metrics = {
                'total_accounts': len(portfolio.account_ids),
                'active_accounts': len([acc_id for acc_id in portfolio.account_ids 
                                      if acc_id in self.accounts and self.accounts[acc_id].is_active]),
                'total_positions': positions_count,
                'account_values': account_values,
                'largest_account_value': max(account_values.values()) if account_values else 0.0,
                'smallest_account_value': min(account_values.values()) if account_values else 0.0,
                'value_distribution': self._calculate_value_distribution(account_values),
                'risk_metrics': self._calculate_risk_metrics(portfolio_id)
            }
            
            # Update portfolio
            portfolio.total_value = total_value
            portfolio.total_profit = total_profit
            portfolio.performance_metrics = metrics
            portfolio.last_updated = datetime.now()
            
            self._save_portfolio(portfolio)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return {}
    
    def _calculate_value_distribution(self, account_values: Dict[str, float]) -> Dict[str, float]:
        """Calculate value distribution across accounts"""
        if not account_values:
            return {}
        
        total_value = sum(account_values.values())
        if total_value == 0:
            return {}
        
        distribution = {}
        for account_id, value in account_values.items():
            distribution[account_id] = value / total_value
        
        return distribution
    
    def _calculate_risk_metrics(self, portfolio_id: str) -> Dict[str, Any]:
        """Calculate risk metrics for portfolio using PostgreSQL queries"""
        try:
            # Get recent snapshots for risk calculations
            snapshots_query = """
            SELECT total_value, timestamp
            FROM portfolio_snapshots
            WHERE portfolio_id = %s
            ORDER BY timestamp DESC
            LIMIT 30
            """
            
            snapshots_data = execute_query(snapshots_query, (portfolio_id,))
            
            if len(snapshots_data) < 2:
                return {'insufficient_data': True}
            
            # Calculate daily returns
            values = [float(row['total_value']) for row in reversed(snapshots_data)]
            returns = []
            for i in range(1, len(values)):
                if values[i-1] != 0:
                    returns.append((values[i] - values[i-1]) / values[i-1])
            
            if not returns:
                return {'insufficient_data': True}
            
            # Risk metrics
            volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
            max_drawdown = self._calculate_max_drawdown(values)
            var_95 = np.percentile(returns, 5) if returns else 0.0  # Value at Risk (95%)
            
            return {
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'value_at_risk_95': var_95,
                'daily_returns_count': len(returns),
                'avg_daily_return': np.mean(returns) if returns else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """Calculate maximum drawdown"""
        if len(values) < 2:
            return 0.0
        
        peak = values[0]
        max_dd = 0.0
        
        for value in values[1:]:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def take_portfolio_snapshot(self, portfolio_id: str):
        """Take a performance snapshot of the portfolio"""
        try:
            if portfolio_id not in self.portfolios:
                return
            
            metrics = self.calculate_portfolio_metrics(portfolio_id)
            portfolio = self.portfolios[portfolio_id]
            
            snapshot = PortfolioSnapshot(
                portfolio_id=portfolio_id,
                timestamp=datetime.now(),
                total_value=portfolio.total_value,
                total_profit=portfolio.total_profit,
                daily_pnl=portfolio.daily_pnl,
                account_values=metrics.get('account_values', {}),
                positions_count=metrics.get('total_positions', 0),
                metrics=metrics
            )
            
            # Store in memory
            if portfolio_id not in self.performance_history:
                self.performance_history[portfolio_id] = []
            
            self.performance_history[portfolio_id].append(snapshot)
            
            # Keep only last 100 snapshots in memory
            if len(self.performance_history[portfolio_id]) > 100:
                self.performance_history[portfolio_id] = self.performance_history[portfolio_id][-100:]
            
            # Save to database
            self._save_snapshot(snapshot)
            
            # Cleanup old snapshots in database
            self._cleanup_old_snapshots()
            
        except Exception as e:
            logger.error(f"Error taking portfolio snapshot: {e}")
    
    def _cleanup_old_snapshots(self):
        """Cleanup old snapshots using PostgreSQL function"""
        try:
            cleanup_query = "SELECT cleanup_old_snapshots()"
            execute_query(cleanup_query, fetch_all=False)
        except Exception as e:
            logger.error(f"Error cleaning up old snapshots: {e}")
    
    def get_portfolio_performance(self, portfolio_id: str, days: int = 30) -> Dict[str, Any]:
        """Get portfolio performance over specified period using PostgreSQL queries"""
        try:
            if portfolio_id not in self.portfolios:
                return {"error": "Portfolio not found"}
            
            portfolio = self.portfolios[portfolio_id]
            
            # Get recent snapshots from database
            snapshots_query = """
            SELECT *
            FROM portfolio_snapshots
            WHERE portfolio_id = %s
              AND timestamp >= %s
            ORDER BY timestamp ASC
            """
            
            cutoff_date = datetime.now() - timedelta(days=days)
            snapshots_data = execute_query(snapshots_query, (portfolio_id, cutoff_date))
            
            if not snapshots_data:
                return {"error": "No performance data available"}
            
            # Calculate performance metrics
            start_value = float(snapshots_data[0]['total_value'])
            end_value = float(snapshots_data[-1]['total_value'])
            total_return = (end_value - start_value) / start_value if start_value > 0 else 0.0
            
            daily_returns = []
            for i in range(1, len(snapshots_data)):
                prev_value = float(snapshots_data[i-1]['total_value'])
                curr_value = float(snapshots_data[i]['total_value'])
                if prev_value > 0:
                    daily_returns.append((curr_value - prev_value) / prev_value)
            
            values = [float(row['total_value']) for row in snapshots_data]
            
            return {
                'portfolio_name': portfolio.name,
                'period_days': days,
                'start_value': start_value,
                'end_value': end_value,
                'total_return': total_return,
                'daily_returns': daily_returns,
                'volatility': np.std(daily_returns) if daily_returns else 0.0,
                'max_drawdown': self._calculate_max_drawdown(values),
                'snapshots_count': len(snapshots_data),
                'current_metrics': portfolio.performance_metrics
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio performance: {e}")
            return {"error": str(e)}
    
    def get_account_summary(self, account_id: str) -> Dict[str, Any]:
        """Get detailed account summary using PostgreSQL queries"""
        try:
            if account_id not in self.accounts:
                return {"error": "Account not found"}
            
            account = self.accounts[account_id]
            
            # Get positions for this account using optimized query
            positions_query = """
            SELECT 
                COUNT(*) as total_positions,
                COUNT(CASE WHEN side = 'buy' THEN 1 END) as long_positions,
                COUNT(CASE WHEN side = 'sell' THEN 1 END) as short_positions,
                COALESCE(SUM(profit_loss), 0) as total_profit_loss,
                COALESCE(SUM(volume), 0) as total_volume
            FROM positions
            WHERE account_id = %s
            """
            
            positions_stats = execute_query(positions_query, (account_id,), fetch_all=False)
            
            # Get recent positions
            recent_positions_query = """
            SELECT *
            FROM positions
            WHERE account_id = %s
            ORDER BY open_time DESC
            LIMIT 10
            """
            
            recent_positions_data = execute_query(recent_positions_query, (account_id,))
            recent_positions = [dict(row) for row in recent_positions_data]
            
            return {
                'account_info': asdict(account),
                'positions_summary': {
                    'total_positions': int(positions_stats['total_positions']) if positions_stats else 0,
                    'long_positions': int(positions_stats['long_positions']) if positions_stats else 0,
                    'short_positions': int(positions_stats['short_positions']) if positions_stats else 0,
                    'total_profit_loss': float(positions_stats['total_profit_loss']) if positions_stats else 0.0,
                    'total_volume': float(positions_stats['total_volume']) if positions_stats else 0.0
                },
                'recent_positions': recent_positions
            }
            
        except Exception as e:
            logger.error(f"Error getting account summary: {e}")
            return {"error": str(e)}
    
    def start_monitoring(self):
        """Start portfolio monitoring"""
        if not self.running:
            self.running = True
            self.update_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.update_thread.start()
            logger.info("Portfolio monitoring started")
    
    def stop_monitoring(self):
        """Stop portfolio monitoring"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
        logger.info("Portfolio monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.running:
            try:
                # Take snapshots for all active portfolios
                for portfolio_id, portfolio in self.portfolios.items():
                    if portfolio.status == PortfolioStatus.ACTIVE:
                        self.take_portfolio_snapshot(portfolio_id)
                
                # Sleep for 5 minutes
                time.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait a minute before retrying
    
    def _save_account(self, account: Account):
        """Save account to PostgreSQL database"""
        try:
            upsert_query = """
            INSERT INTO accounts (
                id, name, account_type, broker, login, server, currency,
                balance, equity, margin, free_margin, margin_level, profit,
                leverage, is_active, created_at, last_updated, metadata
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (id) DO UPDATE SET
                name = EXCLUDED.name,
                account_type = EXCLUDED.account_type,
                broker = EXCLUDED.broker,
                login = EXCLUDED.login,
                server = EXCLUDED.server,
                currency = EXCLUDED.currency,
                balance = EXCLUDED.balance,
                equity = EXCLUDED.equity,
                margin = EXCLUDED.margin,
                free_margin = EXCLUDED.free_margin,
                margin_level = EXCLUDED.margin_level,
                profit = EXCLUDED.profit,
                leverage = EXCLUDED.leverage,
                is_active = EXCLUDED.is_active,
                last_updated = EXCLUDED.last_updated,
                metadata = EXCLUDED.metadata
            """
            
            params = (
                account.id, account.name, account.account_type.value, account.broker,
                account.login, account.server, account.currency, account.balance,
                account.equity, account.margin, account.free_margin, account.margin_level,
                account.profit, account.leverage, account.is_active,
                account.created_at, account.last_updated, json.dumps(account.metadata)
            )
            
            execute_command(upsert_query, params)
            
        except Exception as e:
            logger.error(f"Error saving account: {e}")
    
    def _save_portfolio(self, portfolio: Portfolio):
        """Save portfolio to PostgreSQL database"""
        try:
            upsert_query = """
            INSERT INTO portfolios (
                id, name, description, account_ids, allocation_method,
                allocation_weights, total_value, total_profit, daily_pnl,
                status, created_at, last_updated, performance_metrics
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (id) DO UPDATE SET
                name = EXCLUDED.name,
                description = EXCLUDED.description,
                account_ids = EXCLUDED.account_ids,
                allocation_method = EXCLUDED.allocation_method,
                allocation_weights = EXCLUDED.allocation_weights,
                total_value = EXCLUDED.total_value,
                total_profit = EXCLUDED.total_profit,
                daily_pnl = EXCLUDED.daily_pnl,
                status = EXCLUDED.status,
                last_updated = EXCLUDED.last_updated,
                performance_metrics = EXCLUDED.performance_metrics
            """
            
            params = (
                portfolio.id, portfolio.name, portfolio.description,
                json.dumps(portfolio.account_ids), portfolio.allocation_method.value,
                json.dumps(portfolio.allocation_weights), portfolio.total_value,
                portfolio.total_profit, portfolio.daily_pnl, portfolio.status.value,
                portfolio.created_at, portfolio.last_updated,
                json.dumps(portfolio.performance_metrics)
            )
            
            execute_command(upsert_query, params)
            
        except Exception as e:
            logger.error(f"Error saving portfolio: {e}")
    
    def _save_position(self, position: Position):
        """Save position to PostgreSQL database"""
        try:
            upsert_query = """
            INSERT INTO positions (
                id, account_id, symbol, side, volume, open_price,
                current_price, profit_loss, swap, commission,
                open_time, magic_number, comment
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (id) DO UPDATE SET
                symbol = EXCLUDED.symbol,
                side = EXCLUDED.side,
                volume = EXCLUDED.volume,
                open_price = EXCLUDED.open_price,
                current_price = EXCLUDED.current_price,
                profit_loss = EXCLUDED.profit_loss,
                swap = EXCLUDED.swap,
                commission = EXCLUDED.commission,
                open_time = EXCLUDED.open_time,
                magic_number = EXCLUDED.magic_number,
                comment = EXCLUDED.comment
            """
            
            params = (
                position.id, position.account_id, position.symbol, position.side,
                position.volume, position.open_price, position.current_price,
                position.profit_loss, position.swap, position.commission,
                position.open_time, position.magic_number, position.comment
            )
            
            execute_command(upsert_query, params)
            
        except Exception as e:
            logger.error(f"Error saving position: {e}")
    
    def _save_snapshot(self, snapshot: PortfolioSnapshot):
        """Save portfolio snapshot to PostgreSQL database"""
        try:
            insert_query = """
            INSERT INTO portfolio_snapshots (
                portfolio_id, timestamp, total_value, total_profit, daily_pnl,
                account_values, positions_count, metrics
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s
            )
            """
            
            params = (
                snapshot.portfolio_id, snapshot.timestamp,
                snapshot.total_value, snapshot.total_profit, snapshot.daily_pnl,
                json.dumps(snapshot.account_values), snapshot.positions_count,
                json.dumps(snapshot.metrics)
            )
            
            execute_command(insert_query, params)
            
        except Exception as e:
            logger.error(f"Error saving snapshot: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get portfolio manager status"""
        try:
            # Get counts from database
            counts_query = """
            SELECT 
                (SELECT COUNT(*) FROM accounts) as total_accounts,
                (SELECT COUNT(*) FROM accounts WHERE is_active = true) as active_accounts,
                (SELECT COUNT(*) FROM portfolios) as total_portfolios,
                (SELECT COUNT(*) FROM portfolios WHERE status = 'active') as active_portfolios,
                (SELECT COUNT(*) FROM positions) as total_positions
            """
            
            counts = execute_query(counts_query, fetch_all=False)
            
            return {
                'running': self.running,
                'total_accounts': int(counts['total_accounts']) if counts else 0,
                'active_accounts': int(counts['active_accounts']) if counts else 0,
                'total_portfolios': int(counts['total_portfolios']) if counts else 0,
                'active_portfolios': int(counts['active_portfolios']) if counts else 0,
                'total_positions': int(counts['total_positions']) if counts else 0,
                'database_type': 'PostgreSQL',
                'pool_stats': self.db_manager.get_pool_stats()
            }
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {
                'running': self.running,
                'error': str(e),
                'database_type': 'PostgreSQL'
            }

# Example usage and testing
if __name__ == "__main__":
    # Create portfolio manager
    portfolio_manager = QNTIPortfolioManagerPG()
    
    # Test connection
    status = portfolio_manager.get_status()
    print(f"Portfolio manager status: {json.dumps(status, indent=2)}")
    
    # Add demo accounts
    try:
        acc1_id = portfolio_manager.add_account(
            name="Demo Account 1",
            account_type=AccountType.DEMO,
            broker="Demo Broker",
            login="12345",
            server="demo-server"
        )
        
        acc2_id = portfolio_manager.add_account(
            name="Demo Account 2", 
            account_type=AccountType.DEMO,
            broker="Demo Broker",
            login="67890",
            server="demo-server"
        )
        
        # Create portfolio
        port_id = portfolio_manager.create_portfolio(
            name="Demo Portfolio",
            description="Test portfolio with multiple accounts",
            account_ids=[acc1_id, acc2_id]
        )
        
        # Update account data
        portfolio_manager.update_account_data(acc1_id, {
            'balance': 10000.0,
            'equity': 10500.0,
            'profit': 500.0
        })
        
        portfolio_manager.update_account_data(acc2_id, {
            'balance': 20000.0,
            'equity': 19800.0,
            'profit': -200.0
        })
        
        # Calculate metrics
        metrics = portfolio_manager.calculate_portfolio_metrics(port_id)
        print(f"Portfolio metrics: {json.dumps(metrics, indent=2, default=str)}")
        
        # Get account summary
        account_summary = portfolio_manager.get_account_summary(acc1_id)
        print(f"Account summary: {json.dumps(account_summary, indent=2, default=str)}")
        
        print("PostgreSQL Portfolio Manager test completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
    
    # Stop monitoring
    portfolio_manager.stop_monitoring()
