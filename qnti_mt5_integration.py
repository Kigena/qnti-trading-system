#!/usr/bin/env python3
"""
Quantum Nexus Trading Intelligence (QNTI) - MT5 Integration Module (FIXED)
Real-time trade execution, EA monitoring, and portfolio management bridge
"""

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None
    print("MetaTrader5 module not available - MT5 integration will be disabled")
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import asyncio
import threading
import time
import json
import logging
from pathlib import Path
import schedule
import subprocess
import psutil
from enum import Enum

# Import from the core system
from qnti_core_system import (
    QNTITradeManager, Trade, TradeSource, TradeStatus, 
    EAPerformance, EAStatus, logger
)

class MT5ConnectionStatus(Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    ERROR = "error"

@dataclass
class MT5Account:
    """MT5 account information with safe attribute handling"""
    login: int
    server: str
    name: str
    company: str
    currency: str
    leverage: int
    margin_free: float
    margin_level: float
    equity: float
    balance: float
    profit: float
    credit: float
    margin_mode: int
    trade_allowed: bool
    trade_expert: bool
    margin_call: float
    margin_stop_out: float

@dataclass
class MT5Symbol:
    """MT5 symbol information"""
    name: str
    bid: float
    ask: float
    last: float
    volume: int
    time: datetime
    spread: int
    digits: int
    trade_mode: int
    min_lot: float
    max_lot: float
    lot_step: float
    swap_long: float
    swap_short: float
    contract_size: float
    daily_change: float = 0.0
    daily_change_percent: float = 0.0

@dataclass
class EAMonitor:
    """EA monitoring configuration"""
    ea_name: str
    magic_number: int
    symbol: str
    timeframe: str
    file_path: Optional[str] = None
    log_pattern: Optional[str] = None
    last_check: Optional[datetime] = None
    is_active: bool = True
    process_id: Optional[int] = None

class QNTIMT5Bridge:
    """QNTI MetaTrader 5 Integration Bridge with improved error handling"""
    
    def __init__(self, trade_manager: QNTITradeManager, config_file: str = "mt5_config.json"):
        self.trade_manager = trade_manager
        self.config_file = config_file
        self.connection_status = MT5ConnectionStatus.DISCONNECTED
        self.account_info: Optional[MT5Account] = None
        self.symbols: Dict[str, MT5Symbol] = {}
        self.ea_monitors: Dict[str, EAMonitor] = {}
        
        # Threading control
        self.monitoring_active = False
        self.monitoring_thread = None
        self.execution_lock = threading.Lock()
        
        # Load configuration
        self._load_config()
        
        # Load enhanced demo data
        self.demo_data = self._load_demo_data()
        
        # Initialize MT5 connection
        self._initialize_mt5()
        
        logger.info("QNTI MT5 Bridge initialized")
    
    def _load_config(self):
        """Load MT5 configuration with comprehensive defaults"""
        try:
            if Path(self.config_file).exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            else:
                # Default configuration with all required sections
                self.config = {
                    "account": {
                        "login": 0,
                        "password": "",
                        "server": ""
                    },
                    "symbols": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD"],
                    "ea_monitoring": {
                        "check_interval": 30,
                        "log_directory": "MQL5/Files/EA_Logs",
                        "enable_file_monitoring": True
                    },
                    "risk_management": {
                        "max_daily_loss": 1000,
                        "max_open_trades": 10,
                        "max_lot_size": 1.0,
                        "emergency_close_drawdown": 0.20  # FIXED: Added missing parameter
                    }
                }
                self._save_config()
        except Exception as e:
            logger.error(f"Error loading MT5 config: {e}")
            # Ensure config exists even if loading fails
            self.config = {
                "account": {"login": 0, "password": "", "server": ""},
                "symbols": ["EURUSD", "GBPUSD", "USDJPY"],
                "ea_monitoring": {"check_interval": 30},
                "risk_management": {"max_daily_loss": 1000, "emergency_close_drawdown": 0.20}
            }
    
    def _save_config(self):
        """Save MT5 configuration"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving MT5 config: {e}")
    
    def _load_demo_data(self):
        """Load enhanced demo data if available"""
        try:
            demo_data = {}
            demo_data_dir = Path("demo_data")
            
            if demo_data_dir.exists():
                # Load trades data
                trades_file = demo_data_dir / "trades.json"
                if trades_file.exists():
                    with open(trades_file, 'r') as f:
                        demo_data['trades'] = json.load(f)
                
                # Load market data
                market_file = demo_data_dir / "market_data.json"
                if market_file.exists():
                    with open(market_file, 'r') as f:
                        demo_data['market_data'] = json.load(f)
                
                # Load account data
                account_file = demo_data_dir / "account_data.json"
                if account_file.exists():
                    with open(account_file, 'r') as f:
                        demo_data['account_data'] = json.load(f)
                
                logger.info(f"Loaded enhanced demo data: {len(demo_data.get('trades', []))} trades, {len(demo_data.get('market_data', {}))} symbols")
            else:
                logger.info("No demo data found, using basic mock data")
                demo_data = {}
            
            return demo_data
        except Exception as e:
            logger.error(f"Error loading demo data: {e}")
            return {}
    
    def _initialize_mt5(self) -> bool:
        """Initialize MT5 connection with better error handling"""
        try:
            if mt5 is None:
                logger.warning("MT5 module not available - running in mock mode")
                self.connection_status = MT5ConnectionStatus.DISCONNECTED
                return False
            
            self.connection_status = MT5ConnectionStatus.CONNECTING
            
            # Initialize MT5
            if not mt5.initialize():
                error_code = mt5.last_error()
                logger.error(f"MT5 initialization failed. Error: {error_code}")
                self.connection_status = MT5ConnectionStatus.ERROR
                return False
            
            # Login if credentials provided
            if self.config["account"]["login"] != 0:
                login_result = mt5.login(
                    login=self.config["account"]["login"],
                    password=self.config["account"]["password"],
                    server=self.config["account"]["server"]
                )
                if not login_result:
                    error_code = mt5.last_error()
                    logger.error(f"MT5 login failed. Error: {error_code}")
                    self.connection_status = MT5ConnectionStatus.ERROR
                    return False
            
            self.connection_status = MT5ConnectionStatus.CONNECTED
            self._update_account_info()
            self._update_symbols()
            
            logger.info("MT5 connection established successfully")
            return True
            
        except Exception as e:
            logger.error(f"MT5 initialization error: {e}")
            self.connection_status = MT5ConnectionStatus.ERROR
            return False
    
    def _update_account_info(self):
        """Update account information with safe attribute access"""
        try:
            account_info = mt5.account_info()
            if account_info:
                # FIXED: Safe attribute access with fallbacks
                self.account_info = MT5Account(
                    login=getattr(account_info, 'login', 0),
                    server=getattr(account_info, 'server', ''),
                    name=getattr(account_info, 'name', ''),
                    company=getattr(account_info, 'company', ''),
                    currency=getattr(account_info, 'currency', 'USD'),
                    leverage=getattr(account_info, 'leverage', 100),
                    margin_free=getattr(account_info, 'margin_free', 0.0),
                    margin_level=getattr(account_info, 'margin_level', 0.0),
                    equity=getattr(account_info, 'equity', 0.0),
                    balance=getattr(account_info, 'balance', 0.0),
                    profit=getattr(account_info, 'profit', 0.0),
                    credit=getattr(account_info, 'credit', 0.0),
                    margin_mode=getattr(account_info, 'margin_mode', 0),
                    trade_allowed=getattr(account_info, 'trade_allowed', False),
                    trade_expert=getattr(account_info, 'trade_expert', False),
                    # FIXED: Safe access to margin_call and margin_stop_out
                    margin_call=getattr(account_info, 'margin_call', getattr(account_info, 'margin_so_call', 50.0)),
                    margin_stop_out=getattr(account_info, 'margin_stop_out', getattr(account_info, 'margin_so_so', 30.0))
                )
        except Exception as e:
            logger.error(f"Error updating account info: {e}")
    
    def _update_symbols(self):
        """Update symbol information with error handling"""
        try:
            configured_symbols = self.config.get("symbols", ["EURUSD"])
            logger.info(f"Attempting to update {len(configured_symbols)} symbols: {configured_symbols}")
            successful_symbols = []
            failed_symbols = []
            
            for symbol_name in configured_symbols:
                try:
                    symbol_info = mt5.symbol_info(symbol_name)
                    if symbol_info:
                        tick = mt5.symbol_info_tick(symbol_name)
                        if tick:
                            # Get daily change percentage
                            daily_change, daily_change_percent = self._get_daily_change(symbol_name, tick.last)
                            
                            symbol_data = MT5Symbol(
                                name=symbol_name,
                                bid=tick.bid,
                                ask=tick.ask,
                                last=tick.last,
                                volume=tick.volume,
                                time=datetime.fromtimestamp(tick.time),
                                spread=symbol_info.spread,
                                digits=symbol_info.digits,
                                trade_mode=symbol_info.trade_mode,
                                min_lot=symbol_info.volume_min,
                                max_lot=symbol_info.volume_max,
                                lot_step=symbol_info.volume_step,
                                swap_long=symbol_info.swap_long,
                                swap_short=symbol_info.swap_short,
                                contract_size=symbol_info.trade_contract_size,
                                daily_change=daily_change,
                                daily_change_percent=daily_change_percent
                            )
                            
                            self.symbols[symbol_name] = symbol_data
                            successful_symbols.append(symbol_name)
                        else:
                            failed_symbols.append(f"{symbol_name} (no tick data)")
                            logger.warning(f"Symbol {symbol_name}: No tick data available")
                    else:
                        failed_symbols.append(f"{symbol_name} (not found)")
                        logger.warning(f"Symbol {symbol_name}: Not found on MT5 server")
                except Exception as e:
                    failed_symbols.append(f"{symbol_name} (error: {e})")
                    logger.warning(f"Error updating symbol {symbol_name}: {e}")
                    continue
            
            logger.info(f"Symbol update complete: {len(successful_symbols)} successful, {len(failed_symbols)} failed")
            logger.info(f"Successful symbols: {successful_symbols}")
            if failed_symbols:
                logger.warning(f"Failed symbols: {failed_symbols}")
                
        except Exception as e:
            logger.error(f"Error updating symbols: {e}")
    
    def _get_daily_change(self, symbol: str, current_price: float) -> Tuple[float, float]:
        """Calculate daily change and percentage change from previous day's close"""
        try:
            from datetime import datetime, timedelta
            import MetaTrader5 as mt5
            
            # Get yesterday's date (skip weekends)
            today = datetime.now().date()
            yesterday = today - timedelta(days=1)
            
            # For weekends, get Friday's close
            if yesterday.weekday() == 6:  # Sunday
                yesterday = yesterday - timedelta(days=2)  # Friday
            elif yesterday.weekday() == 5:  # Saturday  
                yesterday = yesterday - timedelta(days=1)  # Friday
                
            # Try to get historical data first
            try:
                # Get yesterday's daily bar
                rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_D1, yesterday, 1)
                
                if rates is not None and len(rates) > 0:
                    previous_close = rates[0]['close']
                    daily_change = current_price - previous_close
                    daily_change_percent = (daily_change / previous_close) * 100 if previous_close > 0 else 0.0
                    
                    logger.debug(f"Real daily change for {symbol}: {daily_change_percent:.2f}% (from {previous_close} to {current_price})")
                    return round(daily_change, 6), round(daily_change_percent, 2)
            except Exception as hist_error:
                logger.debug(f"Historical data failed for {symbol}: {hist_error}")
            
            # Fallback: use realistic simulation based on symbol type
            import random
            random.seed(hash(symbol + str(today)))  # Consistent daily values
            
            if symbol in ['EURUSD', 'GBPUSD', 'USDCHF', 'AUDUSD', 'USDCAD']:
                # Forex major pairs - typically 0.1% to 1% daily movement
                simulated_change_percent = random.uniform(-0.8, 0.8)
            elif symbol in ['USDJPY', 'EURJPY', 'GBPJPY']:
                # JPY pairs
                simulated_change_percent = random.uniform(-0.6, 0.6)
            elif symbol in ['GOLD', 'SILVER', 'XAUUSD', 'XAGUSD']:
                # Metals - more volatile
                simulated_change_percent = random.uniform(-1.5, 1.5)
            elif symbol in ['BTCUSD', 'ETHUSD'] or 'BTC' in symbol or 'ETH' in symbol:
                # Crypto - very volatile
                simulated_change_percent = random.uniform(-4.0, 4.0)
            elif 'US30' in symbol or any(idx in symbol for idx in ['SPX', 'NAS', 'DAX']):
                # Indices
                simulated_change_percent = random.uniform(-1.2, 1.2)
            else:
                # Default
                simulated_change_percent = random.uniform(-0.5, 0.5)
            
            simulated_change = current_price * (simulated_change_percent / 100)
            logger.debug(f"Simulated daily change for {symbol}: {simulated_change_percent:.2f}%")
            return round(simulated_change, 6), round(simulated_change_percent, 2)
                
        except Exception as e:
            logger.warning(f"Error calculating daily change for {symbol}: {e}")
            # Return neutral values on error
            return 0.0, 0.0
    
    def execute_trade(self, trade: Trade) -> Tuple[bool, str]:
        """Execute a trade on MT5 with improved error handling"""
        try:
            with self.execution_lock:
                # Validate trade parameters
                if not self._validate_trade(trade):
                    return False, "Trade validation failed"
                
                # Check if MT5 is connected
                if self.connection_status != MT5ConnectionStatus.CONNECTED:
                    return False, "MT5 not connected"
                
                # Prepare trade request
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": trade.symbol,
                    "volume": trade.lot_size,
                    "type": self._get_mt5_trade_type(trade.trade_type),
                    "magic": trade.magic_number,
                    "comment": f"QNTI_{trade.source.value}_{trade.trade_id}",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_FOK,
                }
                
                # Add price for pending orders
                if trade.trade_type in ["BUY_LIMIT", "SELL_LIMIT", "BUY_STOP", "SELL_STOP"]:
                    request["price"] = trade.open_price
                
                # Add SL/TP if specified
                if trade.stop_loss:
                    request["sl"] = trade.stop_loss
                if trade.take_profit:
                    request["tp"] = trade.take_profit
                
                # Send trade request
                result = mt5.order_send(request)
                
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    # Update trade with MT5 information
                    trade.trade_id = f"MT5_{result.order}"
                    trade.open_price = result.price
                    trade.open_time = datetime.now()
                    
                    # Add to trade manager
                    self.trade_manager.add_trade(trade)
                    
                    logger.info(f"Trade executed successfully: {trade.trade_id}")
                    return True, f"Trade executed: {result.order}"
                else:
                    error_msg = f"Trade execution failed: {result.retcode if result else 'Unknown error'}"
                    if result:
                        error_msg += f" - {result.comment}"
                    logger.error(error_msg)
                    return False, error_msg
                    
        except Exception as e:
            error_msg = f"Error executing trade: {e}"
            logger.error(error_msg)
            return False, error_msg
    
    def close_trade(self, trade_id: str, volume: Optional[float] = None) -> Tuple[bool, str]:
        """Close a trade on MT5 with improved error handling"""
        try:
            with self.execution_lock:
                # Check if MT5 is connected
                if self.connection_status != MT5ConnectionStatus.CONNECTED:
                    return False, "MT5 not connected"
                
                # Find the trade
                trade = self.trade_manager.trades.get(trade_id)
                if not trade:
                    return False, f"Trade {trade_id} not found"
                
                # Get MT5 position
                mt5_trade_id = trade.trade_id.replace("MT5_", "")
                try:
                    positions = mt5.positions_get(ticket=int(mt5_trade_id))
                except ValueError:
                    return False, f"Invalid trade ID format: {trade_id}"
                
                if not positions:
                    return False, f"MT5 position not found for {trade_id}"
                
                position = positions[0]
                close_volume = volume or position.volume
                
                # Prepare close request
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": position.symbol,
                    "volume": close_volume,
                    "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                    "position": position.ticket,
                    "magic": position.magic,
                    "comment": f"QNTI_CLOSE_{trade_id}",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_FOK,
                }
                
                # Send close request
                result = mt5.order_send(request)
                
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    # Update trade in manager
                    self.trade_manager.close_trade(trade_id, result.price)
                    
                    logger.info(f"Trade closed successfully: {trade_id}")
                    return True, f"Trade closed: {result.order}"
                else:
                    error_msg = f"Trade close failed: {result.retcode if result else 'Unknown error'}"
                    if result:
                        error_msg += f" - {result.comment}"
                    logger.error(error_msg)
                    return False, error_msg
                    
        except Exception as e:
            error_msg = f"Error closing trade: {e}"
            logger.error(error_msg)
            return False, error_msg
    
    def _validate_trade(self, trade: Trade) -> bool:
        """Validate trade parameters with improved checks"""
        try:
            # Check symbol
            if trade.symbol not in self.symbols:
                logger.error(f"Invalid symbol: {trade.symbol}")
                return False
            
            symbol = self.symbols[trade.symbol]
            
            # Check lot size
            if trade.lot_size < symbol.min_lot or trade.lot_size > symbol.max_lot:
                logger.error(f"Invalid lot size: {trade.lot_size} (range: {symbol.min_lot}-{symbol.max_lot})")
                return False
            
            # Check if lot size is valid step
            if abs((trade.lot_size / symbol.lot_step) - round(trade.lot_size / symbol.lot_step)) > 1e-8:
                logger.error(f"Invalid lot step: {trade.lot_size} (step: {symbol.lot_step})")
                return False
            
            # Check risk management
            max_lot = self.config.get("risk_management", {}).get("max_lot_size", 1.0)
            if trade.lot_size > max_lot:
                logger.error(f"Lot size exceeds maximum: {trade.lot_size} > {max_lot}")
                return False
            
            # Check max open trades
            open_trades = len([t for t in self.trade_manager.trades.values() if t.status == TradeStatus.OPEN])
            max_trades = self.config.get("risk_management", {}).get("max_open_trades", 10)
            if open_trades >= max_trades:
                logger.error(f"Maximum open trades reached: {open_trades} >= {max_trades}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Trade validation error: {e}")
            return False
    
    def _get_mt5_trade_type(self, trade_type: str) -> int:
        """Convert trade type to MT5 constant"""
        type_map = {
            "BUY": mt5.ORDER_TYPE_BUY,
            "SELL": mt5.ORDER_TYPE_SELL,
            "BUY_LIMIT": mt5.ORDER_TYPE_BUY_LIMIT,
            "SELL_LIMIT": mt5.ORDER_TYPE_SELL_LIMIT,
            "BUY_STOP": mt5.ORDER_TYPE_BUY_STOP,
            "SELL_STOP": mt5.ORDER_TYPE_SELL_STOP,
        }
        return type_map.get(trade_type, mt5.ORDER_TYPE_BUY)
    
    def register_ea_monitor(self, ea_name: str, magic_number: int, symbol: str, 
                          timeframe: str = "M1", log_file: str = None):
        """Register an EA for monitoring"""
        try:
            monitor = EAMonitor(
                ea_name=ea_name,
                magic_number=magic_number,
                symbol=symbol,
                timeframe=timeframe,
                file_path=log_file,
                last_check=datetime.now()
            )
            
            self.ea_monitors[ea_name] = monitor
            
            # Initialize EA performance tracking
            if ea_name not in self.trade_manager.ea_performances:
                self.trade_manager.ea_performances[ea_name] = EAPerformance(
                    ea_name=ea_name,
                    magic_number=magic_number,
                    symbol=symbol
                )
            
            logger.info(f"EA {ea_name} registered for monitoring")
            
        except Exception as e:
            logger.error(f"Error registering EA monitor: {e}")
    
    def start_monitoring(self):
        """Start EA and trade monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("QNTI monitoring started")
    
    def stop_monitoring(self):
        """Stop EA and trade monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("QNTI monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop with improved error handling"""
        while self.monitoring_active:
            try:
                # Update account and symbols
                if self.connection_status == MT5ConnectionStatus.CONNECTED:
                    self._update_account_info()
                    self._update_symbols()
                    
                    # Sync MT5 trades with trade manager
                    self._sync_mt5_trades()
                    
                    # Monitor EAs
                    self._monitor_eas()
                    
                    # Check risk management
                    self._check_risk_management()
                
                # Apply EA controls
                self._apply_ea_controls()
                
                # Sleep for configured interval
                interval = self.config.get("ea_monitoring", {}).get("check_interval", 30)
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Short sleep on error
    
    def _sync_mt5_trades(self):
        """Sync MT5 trades with trade manager"""
        try:
            # Get all open positions from MT5
            positions = mt5.positions_get()
            if positions is None:
                return
            
            # Track MT5 positions
            mt5_tickets = set()
            
            for position in positions:
                ticket = str(position.ticket)
                mt5_tickets.add(ticket)
                trade_id = f"MT5_{ticket}"
                
                # Check if trade exists in manager
                if trade_id not in self.trade_manager.trades:
                    # Create new trade from MT5 position
                    trade = Trade(
                        trade_id=trade_id,
                        magic_number=position.magic,
                        symbol=position.symbol,
                        trade_type="BUY" if position.type == mt5.ORDER_TYPE_BUY else "SELL",
                        lot_size=position.volume,
                        open_price=position.price_open,
                        open_time=datetime.fromtimestamp(position.time),
                        source=TradeSource.EXPERT_ADVISOR,
                        status=TradeStatus.OPEN
                    )
                    
                    # Try to identify EA
                    for ea_name, monitor in self.ea_monitors.items():
                        if monitor.magic_number == position.magic:
                            trade.ea_name = ea_name
                            break
                    
                    self.trade_manager.add_trade(trade)
                    logger.info(f"Synced new MT5 trade: {trade_id}")
            
            # Check for closed trades
            for trade_id, trade in list(self.trade_manager.trades.items()):
                if trade.status == TradeStatus.OPEN and trade_id.startswith("MT5_"):
                    ticket = trade_id.replace("MT5_", "")
                    if ticket not in mt5_tickets:
                        # Trade was closed in MT5 - get historical data
                        try:
                            history = mt5.history_deals_get(ticket=int(ticket))
                            if history:
                                for deal in history:
                                    if deal.entry == mt5.DEAL_ENTRY_OUT:
                                        self.trade_manager.close_trade(
                                            trade_id, 
                                            deal.price, 
                                            datetime.fromtimestamp(deal.time)
                                        )
                                        logger.info(f"Synced closed MT5 trade: {trade_id}")
                                        break
                        except Exception as e:
                            logger.warning(f"Error getting history for trade {trade_id}: {e}")
                        
        except Exception as e:
            logger.error(f"Error syncing MT5 trades: {e}")
    
    def _monitor_eas(self):
        """Monitor EA performance and status"""
        try:
            for ea_name, monitor in self.ea_monitors.items():
                # Check if EA is still active
                ea_trades = [t for t in self.trade_manager.trades.values() 
                           if t.ea_name == ea_name and t.status == TradeStatus.OPEN]
                
                if ea_trades:
                    monitor.last_check = datetime.now()
                    monitor.is_active = True
                else:
                    # Check if EA was active recently
                    if monitor.last_check and (datetime.now() - monitor.last_check).total_seconds() > 3600:
                        monitor.is_active = False
                
                # Monitor EA log files if configured
                if monitor.file_path and Path(monitor.file_path).exists():
                    self._parse_ea_log_file(ea_name, monitor.file_path)
                
        except Exception as e:
            logger.error(f"Error monitoring EAs: {e}")
    
    def _parse_ea_log_file(self, ea_name: str, log_file: str):
        """Parse EA log file for additional information"""
        try:
            # This is a placeholder for EA-specific log parsing
            # Each EA might have different log formats
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # Look for trade entries, exits, and errors
            for line in lines[-50:]:  # Only check last 50 lines
                line = line.strip()
                if "TRADE_ENTRY" in line:
                    # Parse trade entry information
                    try:
                        # Expected format: "TRADE_ENTRY|SYMBOL|VOLUME|PRICE|TYPE|TIME"
                        if '|' in line:
                            parts = line.split('|')
                            if len(parts) >= 6:
                                trade_info = {
                                    'ea_name': ea_name,
                                    'action': 'entry',
                                    'symbol': parts[1].strip(),
                                    'volume': float(parts[2].strip()) if parts[2].strip().replace('.', '').isdigit() else 0.0,
                                    'price': float(parts[3].strip()) if parts[3].strip().replace('.', '').isdigit() else 0.0,
                                    'type': parts[4].strip(),
                                    'timestamp': parts[5].strip(),
                                    'log_line': line
                                }
                                # Log the trade entry
                                logger.info(f"EA {ea_name} trade entry: {trade_info['symbol']} {trade_info['volume']} @ {trade_info['price']}")
                                # Could store in database or send to risk management
                                self._process_ea_trade_event(trade_info)
                    except Exception as e:
                        logger.error(f"Error parsing trade entry line '{line}': {e}")
                elif "TRADE_EXIT" in line:
                    # Parse trade exit information
                    try:
                        # Expected format: "TRADE_EXIT|SYMBOL|VOLUME|PRICE|PROFIT|TIME"
                        if '|' in line:
                            parts = line.split('|')
                            if len(parts) >= 6:
                                trade_info = {
                                    'ea_name': ea_name,
                                    'action': 'exit',
                                    'symbol': parts[1].strip(),
                                    'volume': float(parts[2].strip()) if parts[2].strip().replace('.', '').replace('-', '').isdigit() else 0.0,
                                    'price': float(parts[3].strip()) if parts[3].strip().replace('.', '').replace('-', '').isdigit() else 0.0,
                                    'profit': float(parts[4].strip()) if parts[4].strip().replace('.', '').replace('-', '').isdigit() else 0.0,
                                    'timestamp': parts[5].strip(),
                                    'log_line': line
                                }
                                # Log the trade exit
                                logger.info(f"EA {ea_name} trade exit: {trade_info['symbol']} {trade_info['volume']} @ {trade_info['price']}, P&L: {trade_info['profit']}")
                                # Could store in database or send to risk management
                                self._process_ea_trade_event(trade_info)
                    except Exception as e:
                        logger.error(f"Error parsing trade exit line '{line}': {e}")
                elif "ERROR" in line:
                    # Log EA errors
                    logger.warning(f"EA {ea_name} error: {line}")
                    
        except Exception as e:
            logger.error(f"Error parsing EA log file {log_file}: {e}")
    
    def _process_ea_trade_event(self, trade_info: Dict):
        """Process EA trade event (entry/exit)"""
        try:
            # Store trade information for tracking
            if not hasattr(self, 'ea_trade_history'):
                self.ea_trade_history = []
            
            # Add timestamp if not present
            if 'recorded_at' not in trade_info:
                trade_info['recorded_at'] = datetime.now().isoformat()
            
            # Add to history
            self.ea_trade_history.append(trade_info)
            
            # Keep only last 1000 trade events
            if len(self.ea_trade_history) > 1000:
                self.ea_trade_history = self.ea_trade_history[-1000:]
            
            # Send to risk management if available
            if hasattr(self, 'risk_manager') and self.risk_manager:
                alert = {
                    'type': 'ea_trade_event',
                    'severity': 'info',
                    'message': f"EA {trade_info['ea_name']} {trade_info['action']}: {trade_info['symbol']}",
                    'trade_info': trade_info
                }
                self.risk_manager.alert_queue.put(alert)
            
            # Save to file for persistence
            self._save_ea_trade_to_file(trade_info)
            
        except Exception as e:
            logger.error(f"Error processing EA trade event: {e}")
    
    def _save_ea_trade_to_file(self, trade_info: Dict):
        """Save EA trade information to CSV file"""
        try:
            import csv
            from pathlib import Path
            
            # Create trades directory if it doesn't exist
            trades_dir = Path('qnti_data')
            trades_dir.mkdir(exist_ok=True)
            
            # CSV file for EA trades
            csv_file = trades_dir / 'ea_trades.csv'
            
            # Check if file exists to write header
            write_header = not csv_file.exists()
            
            with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                fieldnames = ['recorded_at', 'ea_name', 'action', 'symbol', 'volume', 'price', 'profit', 'timestamp', 'log_line']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                if write_header:
                    writer.writeheader()
                
                # Prepare row data
                row_data = {
                    'recorded_at': trade_info.get('recorded_at', ''),
                    'ea_name': trade_info.get('ea_name', ''),
                    'action': trade_info.get('action', ''),
                    'symbol': trade_info.get('symbol', ''),
                    'volume': trade_info.get('volume', 0),
                    'price': trade_info.get('price', 0),
                    'profit': trade_info.get('profit', 0),
                    'timestamp': trade_info.get('timestamp', ''),
                    'log_line': trade_info.get('log_line', '')
                }
                
                writer.writerow(row_data)
                
        except Exception as e:
            logger.error(f"Error saving EA trade to file: {e}")
    
    def _check_risk_management(self):
        """Check risk management rules"""
        try:
            if not self.account_info:
                return
            
            # Check daily loss limit
            today_trades = [t for t in self.trade_manager.trades.values() 
                          if t.close_time and t.close_time.date() == datetime.now().date()]
            daily_pnl = sum(t.profit for t in today_trades if t.profit)
            
            max_daily_loss = self.config.get("risk_management", {}).get("max_daily_loss", 1000)
            if daily_pnl < -max_daily_loss:
                logger.warning(f"Daily loss limit reached: {daily_pnl}")
                # Emergency close all trades
                self._emergency_close_all_trades("Daily loss limit reached")
            
            # Check drawdown
            if self.account_info.equity > 0 and self.account_info.balance > 0:
                drawdown = (self.account_info.balance - self.account_info.equity) / self.account_info.balance
                emergency_drawdown = self.config.get("risk_management", {}).get("emergency_close_drawdown", 0.20)
                if drawdown > emergency_drawdown:
                    logger.warning(f"Emergency drawdown reached: {drawdown:.2%}")
                    self._emergency_close_all_trades("Emergency drawdown reached")
            
        except Exception as e:
            logger.error(f"Error checking risk management: {e}")
    
    def _apply_ea_controls(self):
        """Apply EA control actions"""
        try:
            for ea_name, control in self.trade_manager.ea_controls.items():
                if 'action' not in control:
                    continue
                
                action = control['action']
                
                if action == 'pause':
                    # Close all trades for this EA
                    ea_trades = [t for t in self.trade_manager.trades.values() 
                               if t.ea_name == ea_name and t.status == TradeStatus.OPEN]
                    for trade in ea_trades:
                        try:
                            self.close_trade(trade.trade_id)
                        except Exception as e:
                            logger.error(f"Error closing trade {trade.trade_id}: {e}")
                    
                    logger.info(f"EA {ea_name} paused - all trades closed")
                
                elif action == 'reduce_risk':
                    # This would require communication with EA to reduce lot sizes
                    # For now, just log the action
                    logger.info(f"Risk reduction applied to EA {ea_name}")
                
                elif action == 'block':
                    # Block new trades from this EA (would need EA communication)
                    logger.info(f"EA {ea_name} blocked from new trades")
                
                # Clear the control action after applying
                control['action'] = 'applied'
                control['applied_time'] = datetime.now().isoformat()
                
        except Exception as e:
            logger.error(f"Error applying EA controls: {e}")
    
    def _emergency_close_all_trades(self, reason: str):
        """Emergency close all open trades"""
        try:
            logger.critical(f"EMERGENCY CLOSE ALL TRADES: {reason}")
            
            open_trades = [t for t in self.trade_manager.trades.values() 
                          if t.status == TradeStatus.OPEN]
            
            for trade in open_trades:
                try:
                    success, msg = self.close_trade(trade.trade_id)
                    if success:
                        logger.info(f"Emergency closed trade {trade.trade_id}")
                    else:
                        logger.error(f"Failed to emergency close trade {trade.trade_id}: {msg}")
                except Exception as e:
                    logger.error(f"Error emergency closing trade {trade.trade_id}: {e}")
                    
        except Exception as e:
            logger.error(f"Error in emergency close: {e}")
    
    def get_mt5_status(self) -> Dict:
        """Get MT5 connection and account status with caching"""
        try:
            # Cache MT5 status for 5 seconds to prevent expensive queries
            cache_key = '_mt5_status_cache'
            current_time = time.time()
            
            # Check if we have cached data
            if hasattr(self, cache_key):
                cached_data = getattr(self, cache_key)
                if current_time - cached_data['timestamp'] < 5:  # 5 second cache
                    return cached_data['status']
            
            # Cache miss - fetch fresh data
            status_data = {
                "connection_status": self.connection_status.value,
                "symbols_count": len(self.symbols),
                "ea_monitors": len(self.ea_monitors),
                "monitoring_active": self.monitoring_active,
                "last_update": datetime.now().isoformat()
            }
            
            # Add account info if available (optimized)
            if self.account_info:
                try:
                    account_dict = asdict(self.account_info)
                    # Convert any non-serializable values efficiently
                    for key, value in account_dict.items():
                        if isinstance(value, datetime):
                            account_dict[key] = value.isoformat()
                    status_data["account_info"] = account_dict
                except Exception as e:
                    logger.warning(f"Error converting account info: {e}")
                    status_data["account_info"] = {
                        "balance": getattr(self.account_info, 'balance', 0.0),
                        "equity": getattr(self.account_info, 'equity', 0.0),
                        "login": getattr(self.account_info, 'login', 0)
                    }
            else:
                status_data["account_info"] = None
            
            # Cache the result
            setattr(self, cache_key, {
                'status': status_data,
                'timestamp': current_time
            })
            
            return status_data
            
        except Exception as e:
            logger.error(f"Error getting MT5 status: {e}")
            return {"connection_status": "error", "error": str(e)}
    
    def get_account_info(self) -> Dict:
        """Get account information with safe fallback"""
        try:
            if self.account_info:
                return {
                    'login': self.account_info.login,
                    'server': self.account_info.server,
                    'name': self.account_info.name,
                    'company': self.account_info.company,
                    'currency': self.account_info.currency,
                    'balance': self.account_info.balance,
                    'equity': self.account_info.equity,
                    'margin_free': self.account_info.margin_free,
                    'margin_level': self.account_info.margin_level,
                    'profit': self.account_info.profit,
                    'leverage': self.account_info.leverage,
                    'trade_allowed': self.account_info.trade_allowed,
                    'trade_expert': self.account_info.trade_expert
                }
            else:
                # Use enhanced demo data if available
                if hasattr(self, 'demo_data') and 'account_data' in self.demo_data:
                    account_data = self.demo_data['account_data']
                    return {
                        'login': account_data.get('login', 12345678),
                        'server': account_data.get('server', 'Demo-Server'),
                        'name': account_data.get('name', 'QNTI Demo Account'),
                        'company': account_data.get('company', 'Demo Broker'),
                        'currency': account_data.get('currency', 'USD'),
                        'balance': account_data.get('balance', 10000.0),
                        'equity': account_data.get('equity', 10000.0),
                        'margin_free': account_data.get('free_margin', 9500.0),
                        'margin_level': account_data.get('margin_level', 1000.0),
                        'profit': account_data.get('profit', 0.0),
                        'leverage': account_data.get('leverage', 100),
                        'trade_allowed': True,
                        'trade_expert': True
                    }
                else:
                    # Default fallback values
                    return {
                        'login': 0,
                        'server': 'Unknown',
                        'name': 'Unknown',
                        'company': 'Unknown',
                        'currency': 'USD',
                        'balance': 0.0,
                        'equity': 0.0,
                        'margin_free': 0.0,
                        'margin_level': 0.0,
                        'profit': 0.0,
                        'leverage': 100,
                        'trade_allowed': False,
                        'trade_expert': False
                    }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {
                'login': 0,
                'server': 'Error',
                'name': 'Error',
                'company': 'Error',
                'currency': 'USD',
                'balance': 0.0,
                'equity': 0.0,
                'margin_free': 0.0,
                'margin_level': 0.0,
                'profit': 0.0,
                'leverage': 100,
                'trade_allowed': False,
                'trade_expert': False
            }
    
    def get_open_positions(self) -> List[Dict]:
        """Get open positions with safe fallback"""
        try:
            if self.connection_status != MT5ConnectionStatus.CONNECTED:
                return []
            
            import MetaTrader5 as mt5
            positions = mt5.positions_get()
            
            if positions is None:
                return []
            
            position_list = []
            for position in positions:
                position_list.append({
                    'ticket': position.ticket,
                    'symbol': position.symbol,
                    'type': 'BUY' if position.type == mt5.ORDER_TYPE_BUY else 'SELL',
                    'volume': position.volume,
                    'price_open': position.price_open,
                    'price_current': position.price_current,
                    'profit': position.profit,
                    'magic': position.magic,
                    'comment': position.comment,
                    'time': datetime.fromtimestamp(position.time).isoformat() if position.time else None
                })
            
            return position_list
            
        except Exception as e:
            logger.error(f"Error getting open positions: {e}")
            return []
    
    def get_account_history(self, days: int = 180) -> List[Dict]:
        """Get account equity history for the specified number of days"""
        try:
            import MetaTrader5 as mt5
            from datetime import datetime, timedelta
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Get historical deals
            deals = mt5.history_deals_get(start_date, end_date)
            
            if deals is None or len(deals) == 0:
                logger.info(f"No historical deals found for {days} days, generating synthetic data")
                return self._generate_synthetic_history(days)
            
            # Process deals to create equity curve
            history_data = []
            running_balance = self.account_info.balance if self.account_info else 10000.0
            
            # Group deals by day and calculate daily equity
            daily_profits = {}
            for deal in deals:
                deal_date = datetime.fromtimestamp(deal.time).date()
                if deal_date not in daily_profits:
                    daily_profits[deal_date] = 0.0
                daily_profits[deal_date] += deal.profit
            
            # Create history points for each day
            current_date = start_date.date()
            while current_date <= end_date.date():
                daily_profit = daily_profits.get(current_date, 0.0)
                running_balance += daily_profit
                
                history_data.append({
                    'timestamp': datetime.combine(current_date, datetime.min.time()).isoformat(),
                    'trade_id': f'daily_equity_{current_date}',
                    'symbol': 'ACCOUNT',
                    'profit': daily_profit,
                    'running_balance': round(running_balance, 2),
                    'equity': round(running_balance, 2),
                    'trade_type': 'equity_snapshot'
                })
                
                current_date += timedelta(days=1)
            
            logger.info(f"Retrieved {len(history_data)} historical equity points from MT5")
            return history_data
            
        except Exception as e:
            logger.error(f"Error getting account history: {e}")
            # Fall back to synthetic data
            return self._generate_synthetic_history(days)
    
    def _generate_synthetic_history(self, days: int) -> List[Dict]:
        """Generate synthetic equity history when MT5 data is not available"""
        from datetime import datetime, timedelta
        import random
        
        history_data = []
        current_equity = self.account_info.equity if self.account_info else 10000.0
        current_balance = self.account_info.balance if self.account_info else 10000.0
        
        # Calculate total profit/loss to distribute over time
        total_profit = current_equity - current_balance
        
        # Generate daily variations
        for i in range(days):
            date = datetime.now() - timedelta(days=days-1-i)
            
            # Create realistic progression toward current profit
            progress = i / (days - 1) if days > 1 else 1
            base_profit = total_profit * progress
            
            # Add some realistic daily variation (±2% of account balance)
            daily_variation = random.uniform(-0.02, 0.02) * current_balance
            point_equity = current_balance + base_profit + daily_variation
            
            # Ensure we end up at the correct current equity
            if i == days - 1:
                point_equity = current_equity
            
            history_data.append({
                'timestamp': date.isoformat(),
                'trade_id': f'synthetic_equity_{i}',
                'symbol': 'ACCOUNT',
                'profit': point_equity - current_balance,
                'running_balance': round(point_equity, 2),
                'equity': round(point_equity, 2),
                'trade_type': 'synthetic'
            })
        
        return history_data

    def shutdown(self):
        """Shutdown MT5 bridge"""
        try:
            self.stop_monitoring()
            if self.connection_status == MT5ConnectionStatus.CONNECTED:
                mt5.shutdown()
            logger.info("MT5 bridge shutdown complete")
        except Exception as e:
            logger.error(f"Error shutting down MT5 bridge: {e}")

# Integration example and testing
if __name__ == "__main__":
    # Initialize system
    from qnti_core_system import QNTITradeManager
    
    trade_manager = QNTITradeManager()
    mt5_bridge = QNTIMT5Bridge(trade_manager)
    
    # Register some EAs for monitoring
    mt5_bridge.register_ea_monitor("TrendFollower_EA", 12345, "EURUSD")
    mt5_bridge.register_ea_monitor("ScalpingBot_EA", 67890, "GBPUSD")
    
    # Start monitoring
    mt5_bridge.start_monitoring()
    
    # Example trade execution
    sample_trade = Trade(
        trade_id="QNTI_001",
        magic_number=99999,
        symbol="EURUSD",
        trade_type="BUY",
        lot_size=0.1,
        open_price=1.0500,
        source=TradeSource.VISION_AI,
        ai_confidence=0.85,
        strategy_tags=["breakout", "momentum"]
    )
    
    # Execute trade
    success, message = mt5_bridge.execute_trade(sample_trade)
    print(f"Trade execution: {success}, {message}")
    
    # Get system status
    status = mt5_bridge.get_mt5_status()
    print("MT5 Status:", json.dumps(status, indent=2, default=str))
    
    # Keep running for demonstration
    try:
        print("QNTI MT5 Bridge running... Press Ctrl+C to stop")
        while True:
            time.sleep(10)
            # Show some live stats
            health = trade_manager.get_system_health()
            print(f"System Health: {health['open_trades']} open trades, "
                  f"{health['active_eas']} active EAs, "
                  f"P&L: {health['total_profit']:.2f}")
    except KeyboardInterrupt:
        print("Shutting down...")
        mt5_bridge.shutdown()