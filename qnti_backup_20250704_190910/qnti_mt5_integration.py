#!/usr/bin/env python3
"""
Quantum Nexus Trading Intelligence (QNTI) - MT5 Integration Module (FIXED)
Real-time trade execution, EA monitoring, and portfolio management bridge
"""

import MetaTrader5 as mt5
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
    
    def _initialize_mt5(self) -> bool:
        """Initialize MT5 connection with better error handling"""
        try:
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
            for symbol_name in self.config.get("symbols", ["EURUSD"]):
                try:
                    symbol_info = mt5.symbol_info(symbol_name)
                    if symbol_info:
                        tick = mt5.symbol_info_tick(symbol_name)
                        if tick:
                            self.symbols[symbol_name] = MT5Symbol(
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
                                contract_size=symbol_info.trade_contract_size
                            )
                except Exception as e:
                    logger.warning(f"Error updating symbol {symbol_name}: {e}")
                    continue
        except Exception as e:
            logger.error(f"Error updating symbols: {e}")
    
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
    
    def close_trade(self, trade_id: str, volume: float = None) -> Tuple[bool, str]:
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
                    pass
                elif "TRADE_EXIT" in line:
                    # Parse trade exit information
                    pass
                elif "ERROR" in line:
                    # Log EA errors
                    logger.warning(f"EA {ea_name} error: {line}")
                    
        except Exception as e:
            logger.error(f"Error parsing EA log file {log_file}: {e}")
    
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
        """Get MT5 connection and account status"""
        try:
            status_data = {
                "connection_status": self.connection_status.value,
                "symbols_count": len(self.symbols),
                "ea_monitors": len(self.ea_monitors),
                "monitoring_active": self.monitoring_active,
                "last_update": datetime.now().isoformat()
            }
            
            # Add account info if available
            if self.account_info:
                account_dict = asdict(self.account_info)
                # Convert any non-serializable values
                for key, value in account_dict.items():
                    if isinstance(value, datetime):
                        account_dict[key] = value.isoformat()
                status_data["account_info"] = account_dict
            else:
                status_data["account_info"] = None
            
            return status_data
            
        except Exception as e:
            logger.error(f"Error getting MT5 status: {e}")
            return {"connection_status": "error", "error": str(e)}
    
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