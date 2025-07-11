#!/usr/bin/env python3
"""
QNTI Portfolio Manager - Multi-Account Portfolio Management
Advanced portfolio management system with multi-account tracking, allocation, and performance analysis
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
import time

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

class QNTIPortfolioManager:
    """Advanced portfolio management system for QNTI"""
    
    def __init__(self, db_path: str = "qnti_portfolio.db"):
        self.db_path = db_path
        self.accounts: Dict[str, Account] = {}
        self.portfolios: Dict[str, Portfolio] = {}
        self.positions: Dict[str, Position] = {}
        self.running = False
        self.update_thread = None
        
        # Performance tracking
        self.performance_history: Dict[str, List[PortfolioSnapshot]] = {}
        
        # Initialize database
        self._init_database()
        
        # Load existing data
        self._load_data()
        
        logger.info("QNTI Portfolio Manager initialized")
    
    def _init_database(self):
        """Initialize SQLite database for portfolio data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Accounts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS accounts (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    account_type TEXT NOT NULL,
                    broker TEXT NOT NULL,
                    login TEXT NOT NULL,
                    server TEXT NOT NULL,
                    currency TEXT NOT NULL,
                    balance REAL NOT NULL,
                    equity REAL NOT NULL,
                    margin REAL NOT NULL,
                    free_margin REAL NOT NULL,
                    margin_level REAL NOT NULL,
                    profit REAL NOT NULL,
                    leverage INTEGER NOT NULL,
                    is_active BOOLEAN NOT NULL,
                    created_at TEXT NOT NULL,
                    last_updated TEXT NOT NULL,
                    metadata TEXT
                )
            ''')
            
            # Portfolios table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolios (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    account_ids TEXT NOT NULL,
                    allocation_method TEXT NOT NULL,
                    allocation_weights TEXT NOT NULL,
                    total_value REAL NOT NULL,
                    total_profit REAL NOT NULL,
                    daily_pnl REAL NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_updated TEXT NOT NULL,
                    performance_metrics TEXT
                )
            ''')
            
            # Positions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    id TEXT PRIMARY KEY,
                    account_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    volume REAL NOT NULL,
                    open_price REAL NOT NULL,
                    current_price REAL NOT NULL,
                    profit_loss REAL NOT NULL,
                    swap REAL NOT NULL,
                    commission REAL NOT NULL,
                    open_time TEXT NOT NULL,
                    magic_number INTEGER,
                    comment TEXT
                )
            ''')
            
            # Portfolio snapshots table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    portfolio_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    total_value REAL NOT NULL,
                    total_profit REAL NOT NULL,
                    daily_pnl REAL NOT NULL,
                    account_values TEXT NOT NULL,
                    positions_count INTEGER NOT NULL,
                    metrics TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def _load_data(self):
        """Load existing data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Load accounts
            cursor.execute("SELECT * FROM accounts")
            for row in cursor.fetchall():
                account = Account(
                    id=row[0],
                    name=row[1],
                    account_type=AccountType(row[2]),
                    broker=row[3],
                    login=row[4],
                    server=row[5],
                    currency=row[6],
                    balance=row[7],
                    equity=row[8],
                    margin=row[9],
                    free_margin=row[10],
                    margin_level=row[11],
                    profit=row[12],
                    leverage=row[13],
                    is_active=bool(row[14]),
                    created_at=datetime.fromisoformat(row[15]),
                    last_updated=datetime.fromisoformat(row[16]),
                    metadata=json.loads(row[17]) if row[17] else {}
                )
                self.accounts[account.id] = account
            
            # Load portfolios
            cursor.execute("SELECT * FROM portfolios")
            for row in cursor.fetchall():
                portfolio = Portfolio(
                    id=row[0],
                    name=row[1],
                    description=row[2],
                    account_ids=json.loads(row[3]),
                    allocation_method=AllocationMethod(row[4]),
                    allocation_weights=json.loads(row[5]),
                    total_value=row[6],
                    total_profit=row[7],
                    daily_pnl=row[8],
                    status=PortfolioStatus(row[9]),
                    created_at=datetime.fromisoformat(row[10]),
                    last_updated=datetime.fromisoformat(row[11]),
                    performance_metrics=json.loads(row[12]) if row[12] else {}
                )
                self.portfolios[portfolio.id] = portfolio
            
            # Load positions
            cursor.execute("SELECT * FROM positions")
            for row in cursor.fetchall():
                position = Position(
                    id=row[0],
                    account_id=row[1],
                    symbol=row[2],
                    side=row[3],
                    volume=row[4],
                    open_price=row[5],
                    current_price=row[6],
                    profit_loss=row[7],
                    swap=row[8],
                    commission=row[9],
                    open_time=datetime.fromisoformat(row[10]),
                    magic_number=row[11],
                    comment=row[12] or ""
                )
                self.positions[position.id] = position
            
            conn.close()
            logger.info(f"Loaded {len(self.accounts)} accounts, {len(self.portfolios)} portfolios, {len(self.positions)} positions")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
    
    def add_account(self, name: str, account_type: AccountType, broker: str, 
                   login: str, server: str, currency: str = "USD", 
                   leverage: int = 100, metadata: Dict = None) -> str:
        """Add a new trading account"""
        try:
            account_id = f"acc_{int(datetime.now().timestamp())}_{len(self.accounts)}"
            
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
            
            portfolio_id = f"port_{int(datetime.now().timestamp())}_{len(self.portfolios)}"
            
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
            for pos_id in account_positions:
                del self.positions[pos_id]
            
            # Add new positions
            for pos_data in positions_data:
                position_id = f"pos_{account_id}_{pos_data.get('ticket', int(datetime.now().timestamp()))}"
                
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
        """Calculate comprehensive portfolio metrics"""
        try:
            if portfolio_id not in self.portfolios:
                raise ValueError(f"Portfolio {portfolio_id} not found")
            
            portfolio = self.portfolios[portfolio_id]
            
            # Get account data
            account_values = {}
            total_value = 0.0
            total_profit = 0.0
            
            for account_id in portfolio.account_ids:
                if account_id in self.accounts:
                    account = self.accounts[account_id]
                    account_values[account_id] = account.equity
                    total_value += account.equity * portfolio.allocation_weights.get(account_id, 0.0)
                    total_profit += account.profit * portfolio.allocation_weights.get(account_id, 0.0)
            
            # Count positions
            positions_count = len([pos for pos in self.positions.values() 
                                 if pos.account_id in portfolio.account_ids])
            
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
        """Calculate risk metrics for portfolio"""
        try:
            portfolio = self.portfolios[portfolio_id]
            
            # Get historical snapshots for risk calculations
            snapshots = self.performance_history.get(portfolio_id, [])
            if len(snapshots) < 2:
                return {'insufficient_data': True}
            
            # Calculate daily returns
            values = [snapshot.total_value for snapshot in snapshots[-30:]]  # Last 30 snapshots
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
            
            # Keep only last 1000 snapshots
            if len(self.performance_history[portfolio_id]) > 1000:
                self.performance_history[portfolio_id] = self.performance_history[portfolio_id][-1000:]
            
            # Save to database
            self._save_snapshot(snapshot)
            
        except Exception as e:
            logger.error(f"Error taking portfolio snapshot: {e}")
    
    def get_portfolio_performance(self, portfolio_id: str, days: int = 30) -> Dict[str, Any]:
        """Get portfolio performance over specified period"""
        try:
            if portfolio_id not in self.portfolios:
                return {"error": "Portfolio not found"}
            
            portfolio = self.portfolios[portfolio_id]
            snapshots = self.performance_history.get(portfolio_id, [])
            
            # Filter snapshots by date
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_snapshots = [s for s in snapshots if s.timestamp >= cutoff_date]
            
            if not recent_snapshots:
                return {"error": "No performance data available"}
            
            # Calculate performance metrics
            start_value = recent_snapshots[0].total_value
            end_value = recent_snapshots[-1].total_value
            total_return = (end_value - start_value) / start_value if start_value > 0 else 0.0
            
            daily_returns = []
            for i in range(1, len(recent_snapshots)):
                prev_value = recent_snapshots[i-1].total_value
                curr_value = recent_snapshots[i].total_value
                if prev_value > 0:
                    daily_returns.append((curr_value - prev_value) / prev_value)
            
            return {
                'portfolio_name': portfolio.name,
                'period_days': days,
                'start_value': start_value,
                'end_value': end_value,
                'total_return': total_return,
                'daily_returns': daily_returns,
                'volatility': np.std(daily_returns) if daily_returns else 0.0,
                'max_drawdown': self._calculate_max_drawdown([s.total_value for s in recent_snapshots]),
                'snapshots_count': len(recent_snapshots),
                'current_metrics': portfolio.performance_metrics
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio performance: {e}")
            return {"error": str(e)}
    
    def get_account_summary(self, account_id: str) -> Dict[str, Any]:
        """Get detailed account summary"""
        try:
            if account_id not in self.accounts:
                return {"error": "Account not found"}
            
            account = self.accounts[account_id]
            
            # Get positions for this account
            account_positions = [pos for pos in self.positions.values() 
                               if pos.account_id == account_id]
            
            # Calculate position metrics
            total_profit_loss = sum(pos.profit_loss for pos in account_positions)
            long_positions = [pos for pos in account_positions if pos.side.lower() == 'buy']
            short_positions = [pos for pos in account_positions if pos.side.lower() == 'sell']
            
            return {
                'account_info': asdict(account),
                'positions_summary': {
                    'total_positions': len(account_positions),
                    'long_positions': len(long_positions),
                    'short_positions': len(short_positions),
                    'total_profit_loss': total_profit_loss,
                    'total_volume': sum(pos.volume for pos in account_positions)
                },
                'recent_positions': [asdict(pos) for pos in account_positions[-10:]]  # Last 10 positions
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
        """Save account to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO accounts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                account.id, account.name, account.account_type.value, account.broker,
                account.login, account.server, account.currency, account.balance,
                account.equity, account.margin, account.free_margin, account.margin_level,
                account.profit, account.leverage, account.is_active,
                account.created_at.isoformat(), account.last_updated.isoformat(),
                json.dumps(account.metadata)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving account: {e}")
    
    def _save_portfolio(self, portfolio: Portfolio):
        """Save portfolio to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO portfolios VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                portfolio.id, portfolio.name, portfolio.description,
                json.dumps(portfolio.account_ids), portfolio.allocation_method.value,
                json.dumps(portfolio.allocation_weights), portfolio.total_value,
                portfolio.total_profit, portfolio.daily_pnl, portfolio.status.value,
                portfolio.created_at.isoformat(), portfolio.last_updated.isoformat(),
                json.dumps(portfolio.performance_metrics)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving portfolio: {e}")
    
    def _save_position(self, position: Position):
        """Save position to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO positions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                position.id, position.account_id, position.symbol, position.side,
                position.volume, position.open_price, position.current_price,
                position.profit_loss, position.swap, position.commission,
                position.open_time.isoformat(), position.magic_number, position.comment
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving position: {e}")
    
    def _save_snapshot(self, snapshot: PortfolioSnapshot):
        """Save portfolio snapshot to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO portfolio_snapshots 
                (portfolio_id, timestamp, total_value, total_profit, daily_pnl, account_values, positions_count, metrics)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                snapshot.portfolio_id, snapshot.timestamp.isoformat(),
                snapshot.total_value, snapshot.total_profit, snapshot.daily_pnl,
                json.dumps(snapshot.account_values), snapshot.positions_count,
                json.dumps(snapshot.metrics)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving snapshot: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get portfolio manager status"""
        return {
            'running': self.running,
            'total_accounts': len(self.accounts),
            'active_accounts': len([acc for acc in self.accounts.values() if acc.is_active]),
            'total_portfolios': len(self.portfolios),
            'active_portfolios': len([port for port in self.portfolios.values() if port.status == PortfolioStatus.ACTIVE]),
            'total_positions': len(self.positions),
            'database_path': self.db_path
        }

# Example usage and testing
if __name__ == "__main__":
    # Create portfolio manager
    portfolio_manager = QNTIPortfolioManager()
    
    # Add demo accounts
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
    print(f"Portfolio metrics: {json.dumps(metrics, indent=2)}")
    
    # Start monitoring
    portfolio_manager.start_monitoring()
    
    # Show status
    status = portfolio_manager.get_status()
    print(f"Portfolio manager status: {json.dumps(status, indent=2)}")
    
    # Stop monitoring
    portfolio_manager.stop_monitoring()