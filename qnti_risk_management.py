#!/usr/bin/env python3
"""
QNTI Risk Management System - Automated Risk Controls and Position Sizing
Provides comprehensive risk management, position sizing, and automated controls
"""

import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import queue

logger = logging.getLogger('QNTI_RISK')

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class PositionSizeMethod(Enum):
    FIXED = "fixed"
    PERCENTAGE = "percentage"
    RISK_BASED = "risk_based"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    KELLY_CRITERION = "kelly_criterion"

@dataclass
class RiskParameters:
    """Risk management parameters"""
    max_risk_per_trade: float = 2.0  # % of account
    max_daily_loss: float = 5.0  # % of account
    max_drawdown: float = 15.0  # % of account
    max_correlation: float = 0.8  # Max correlation between positions
    max_exposure_per_symbol: float = 10.0  # % of account
    max_exposure_per_currency: float = 25.0  # % of account
    max_open_positions: int = 10
    position_size_method: PositionSizeMethod = PositionSizeMethod.RISK_BASED
    base_position_size: float = 0.01  # Base lot size
    leverage_limit: float = 10.0  # Maximum leverage
    margin_call_threshold: float = 50.0  # Margin level %
    
@dataclass
class RiskAlert:
    """Risk alert data structure"""
    alert_id: str
    alert_type: str
    severity: RiskLevel
    message: str
    symbol: Optional[str] = None
    value: Optional[float] = None
    threshold: Optional[float] = None
    timestamp: datetime = None
    acknowledged: bool = False
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class PositionSizeResult:
    """Position sizing calculation result"""
    recommended_size: float
    max_size: float
    risk_amount: float
    risk_percentage: float
    method_used: str
    warnings: List[str]
    calculations: Dict

class QNTIRiskManager:
    """Comprehensive risk management system"""
    
    def __init__(self, trade_manager=None, mt5_bridge=None):
        self.trade_manager = trade_manager
        self.mt5_bridge = mt5_bridge
        self.risk_params = RiskParameters()
        self.alerts = []
        self.risk_metrics = {}
        self.position_limits = {}
        self.currency_exposure = {}
        self.correlation_matrix = {}
        self.volatility_cache = {}
        self.running = False
        self.risk_thread = None
        self.alert_queue = queue.Queue()
        
        # Load risk parameters from config if available
        self._load_risk_config()
        
        logger.info("QNTI Risk Manager initialized")
    
    def start_monitoring(self):
        """Start risk monitoring thread"""
        if not self.running:
            self.running = True
            self.risk_thread = threading.Thread(target=self._risk_monitoring_loop, daemon=True)
            self.risk_thread.start()
            logger.info("Risk monitoring started")
    
    def stop_monitoring(self):
        """Stop risk monitoring"""
        self.running = False
        if self.risk_thread:
            self.risk_thread.join(timeout=5)
        logger.info("Risk monitoring stopped")
    
    def _risk_monitoring_loop(self):
        """Main risk monitoring loop"""
        while self.running:
            try:
                # Update risk metrics
                self._update_risk_metrics()
                
                # Check for risk violations
                self._check_risk_violations()
                
                # Update position limits
                self._update_position_limits()
                
                # Process alerts
                self._process_alerts()
                
                # Sleep for monitoring interval
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in risk monitoring loop: {e}")
                time.sleep(10)
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                              stop_loss: float, account_balance: float,
                              account_equity: float = None) -> PositionSizeResult:
        """Calculate optimal position size based on risk parameters"""
        
        if account_equity is None:
            account_equity = account_balance
        
        warnings = []
        calculations = {}
        
        # Calculate risk amount
        risk_amount = account_equity * (self.risk_params.max_risk_per_trade / 100)
        calculations['risk_amount'] = risk_amount
        
        # Calculate pip value and stop loss distance
        pip_distance = abs(entry_price - stop_loss)
        calculations['pip_distance'] = pip_distance
        
        if pip_distance == 0:
            warnings.append("Invalid stop loss - same as entry price")
            return PositionSizeResult(
                recommended_size=0.0,
                max_size=0.0,
                risk_amount=0.0,
                risk_percentage=0.0,
                method_used="error",
                warnings=warnings,
                calculations=calculations
            )
        
        # Get symbol specifications
        symbol_info = self._get_symbol_info(symbol)
        pip_value = symbol_info.get('pip_value', 10.0)  # Default for major pairs
        calculations['pip_value'] = pip_value
        
        # Calculate base position size
        base_size = risk_amount / (pip_distance * pip_value)
        calculations['base_size'] = base_size
        
        # Apply position sizing method
        if self.risk_params.position_size_method == PositionSizeMethod.FIXED:
            recommended_size = self.risk_params.base_position_size
            method_used = "fixed"
            
        elif self.risk_params.position_size_method == PositionSizeMethod.PERCENTAGE:
            recommended_size = account_equity * (self.risk_params.base_position_size / 100)
            method_used = "percentage"
            
        elif self.risk_params.position_size_method == PositionSizeMethod.RISK_BASED:
            recommended_size = base_size
            method_used = "risk_based"
            
        elif self.risk_params.position_size_method == PositionSizeMethod.VOLATILITY_ADJUSTED:
            volatility = self._get_symbol_volatility(symbol)
            volatility_multiplier = 1.0 / max(volatility, 0.01)  # Reduce size for high volatility
            recommended_size = base_size * volatility_multiplier
            method_used = "volatility_adjusted"
            calculations['volatility'] = volatility
            calculations['volatility_multiplier'] = volatility_multiplier
            
        elif self.risk_params.position_size_method == PositionSizeMethod.KELLY_CRITERION:
            win_rate = self._get_symbol_win_rate(symbol)
            avg_win = self._get_symbol_avg_win(symbol)
            avg_loss = self._get_symbol_avg_loss(symbol)
            
            if avg_loss > 0:
                kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
                recommended_size = base_size * kelly_fraction
                method_used = "kelly_criterion"
                calculations.update({
                    'win_rate': win_rate,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'kelly_fraction': kelly_fraction
                })
            else:
                recommended_size = base_size
                method_used = "risk_based"
                warnings.append("Insufficient data for Kelly criterion, using risk-based")
        
        else:
            recommended_size = base_size
            method_used = "risk_based"
        
        # Apply position limits
        max_size = self._calculate_max_position_size(symbol, account_equity)
        calculations['max_size'] = max_size
        
        if recommended_size > max_size:
            warnings.append(f"Position size reduced from {recommended_size:.4f} to {max_size:.4f} due to limits")
            recommended_size = max_size
        
        # Calculate final risk
        final_risk_amount = recommended_size * pip_distance * pip_value
        final_risk_percentage = (final_risk_amount / account_equity) * 100
        
        # Check for warnings
        if final_risk_percentage > self.risk_params.max_risk_per_trade:
            warnings.append(f"Risk {final_risk_percentage:.2f}% exceeds maximum {self.risk_params.max_risk_per_trade}%")
        
        return PositionSizeResult(
            recommended_size=round(recommended_size, 4),
            max_size=round(max_size, 4),
            risk_amount=round(final_risk_amount, 2),
            risk_percentage=round(final_risk_percentage, 2),
            method_used=method_used,
            warnings=warnings,
            calculations=calculations
        )
    
    def check_trade_risk(self, symbol: str, volume: float, entry_price: float,
                        stop_loss: float = None, take_profit: float = None) -> Dict:
        """Check if a trade meets risk criteria"""
        
        risk_check = {
            "approved": True,
            "warnings": [],
            "violations": [],
            "risk_metrics": {}
        }
        
        # Get current account info
        account_info = self._get_account_info()
        account_equity = account_info.get('equity', 10000)
        
        # Check position size limits
        if volume > self.risk_params.base_position_size * 10:  # 10x base size limit
            risk_check["violations"].append(f"Position size {volume} exceeds maximum allowed")
            risk_check["approved"] = False
        
        # Check symbol exposure
        current_exposure = self._get_symbol_exposure(symbol)
        new_exposure = current_exposure + (volume * entry_price)
        max_symbol_exposure = account_equity * (self.risk_params.max_exposure_per_symbol / 100)
        
        if new_exposure > max_symbol_exposure:
            risk_check["violations"].append(f"Symbol exposure would exceed limit")
            risk_check["approved"] = False
        
        # Check currency exposure
        base_currency = symbol[:3]
        quote_currency = symbol[3:6]
        
        for currency in [base_currency, quote_currency]:
            currency_exposure = self._get_currency_exposure(currency)
            # Add this trade's exposure
            if currency == base_currency:
                currency_exposure += volume * entry_price
            else:
                currency_exposure += volume
            
            max_currency_exposure = account_equity * (self.risk_params.max_exposure_per_currency / 100)
            if currency_exposure > max_currency_exposure:
                risk_check["warnings"].append(f"{currency} exposure approaching limit")
        
        # Check correlation limits
        correlations = self._check_correlation_limits(symbol)
        if correlations:
            for corr_symbol, corr_value in correlations.items():
                if abs(corr_value) > self.risk_params.max_correlation:
                    risk_check["warnings"].append(f"High correlation with {corr_symbol}: {corr_value:.2f}")
        
        # Check daily loss limit
        daily_loss = self._get_daily_loss()
        max_daily_loss = account_equity * (self.risk_params.max_daily_loss / 100)
        if daily_loss > max_daily_loss:
            risk_check["violations"].append("Daily loss limit exceeded")
            risk_check["approved"] = False
        
        # Check drawdown limit
        current_drawdown = self._get_current_drawdown()
        if current_drawdown > self.risk_params.max_drawdown:
            risk_check["violations"].append("Maximum drawdown exceeded")
            risk_check["approved"] = False
        
        # Check position count limit
        open_positions = self._get_open_positions_count()
        if open_positions >= self.risk_params.max_open_positions:
            risk_check["violations"].append("Maximum open positions reached")
            risk_check["approved"] = False
        
        # Calculate risk metrics
        if stop_loss:
            risk_amount = volume * abs(entry_price - stop_loss) * self._get_pip_value(symbol)
            risk_percentage = (risk_amount / account_equity) * 100
            
            risk_check["risk_metrics"] = {
                "risk_amount": round(risk_amount, 2),
                "risk_percentage": round(risk_percentage, 2),
                "reward_risk_ratio": self._calculate_reward_risk_ratio(entry_price, stop_loss, take_profit) if take_profit else None
            }
            
            if risk_percentage > self.risk_params.max_risk_per_trade:
                risk_check["violations"].append(f"Trade risk {risk_percentage:.2f}% exceeds maximum {self.risk_params.max_risk_per_trade}%")
                risk_check["approved"] = False
        
        return risk_check
    
    def get_risk_dashboard(self) -> Dict:
        """Get comprehensive risk dashboard data"""
        
        account_info = self._get_account_info()
        
        return {
            "account_metrics": {
                "equity": account_info.get('equity', 0),
                "balance": account_info.get('balance', 0),
                "margin_used": account_info.get('margin', 0),
                "margin_free": account_info.get('free_margin', 0),
                "margin_level": account_info.get('margin_level', 0),
                "leverage_used": self._calculate_leverage_used(account_info)
            },
            "risk_metrics": {
                "daily_pnl": self._get_daily_pnl(),
                "daily_loss": self._get_daily_loss(),
                "current_drawdown": self._get_current_drawdown(),
                "open_positions": self._get_open_positions_count(),
                "total_exposure": self._get_total_exposure(),
                "risk_score": self._calculate_risk_score()
            },
            "exposure_analysis": {
                "symbol_exposure": self._get_all_symbol_exposure(),
                "currency_exposure": self._get_all_currency_exposure(),
                "correlation_risks": self._get_correlation_risks()
            },
            "position_limits": self.position_limits,
            "active_alerts": [asdict(alert) for alert in self.alerts if not alert.acknowledged],
            "risk_parameters": asdict(self.risk_params),
            "recommendations": self._get_risk_recommendations()
        }
    
    def update_risk_parameters(self, new_params: Dict) -> bool:
        """Update risk management parameters"""
        try:
            for key, value in new_params.items():
                if hasattr(self.risk_params, key):
                    setattr(self.risk_params, key, value)
            
            # Save to configuration
            self._save_risk_config()
            
            logger.info(f"Risk parameters updated: {new_params}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating risk parameters: {e}")
            return False
    
    def create_alert(self, alert_type: str, severity: RiskLevel, message: str,
                    symbol: str = None, value: float = None, threshold: float = None) -> str:
        """Create a risk alert"""
        
        alert_id = f"RISK_{int(time.time())}_{len(self.alerts)}"
        
        alert = RiskAlert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            message=message,
            symbol=symbol,
            value=value,
            threshold=threshold
        )
        
        self.alerts.append(alert)
        self.alert_queue.put(alert)
        
        logger.warning(f"Risk alert created: {alert_type} - {message}")
        return alert_id
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge a risk alert"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                logger.info(f"Risk alert acknowledged: {alert_id}")
                return True
        return False
    
    # Helper methods
    def _get_symbol_info(self, symbol: str) -> Dict:
        """Get symbol information"""
        if self.mt5_bridge:
            return self.mt5_bridge.get_symbol_info(symbol) or {}
        return {"pip_value": 10.0, "digits": 5}
    
    def _get_symbol_volatility(self, symbol: str) -> float:
        """Get symbol volatility (cached)"""
        if symbol in self.volatility_cache:
            return self.volatility_cache[symbol]
        
        # Mock volatility data
        volatility = 0.015  # 1.5% daily volatility
        self.volatility_cache[symbol] = volatility
        return volatility
    
    def _get_symbol_win_rate(self, symbol: str) -> float:
        """Get historical win rate for symbol"""
        return 0.55  # 55% win rate
    
    def _get_symbol_avg_win(self, symbol: str) -> float:
        """Get average win for symbol"""
        return 150.0  # $150 average win
    
    def _get_symbol_avg_loss(self, symbol: str) -> float:
        """Get average loss for symbol"""
        return 100.0  # $100 average loss
    
    def _calculate_max_position_size(self, symbol: str, account_equity: float) -> float:
        """Calculate maximum allowed position size"""
        # Based on symbol exposure limit
        max_symbol_exposure = account_equity * (self.risk_params.max_exposure_per_symbol / 100)
        symbol_price = self._get_symbol_price(symbol)
        
        if symbol_price > 0:
            return max_symbol_exposure / symbol_price
        return self.risk_params.base_position_size
    
    def _get_symbol_price(self, symbol: str) -> float:
        """Get current symbol price"""
        if self.mt5_bridge:
            info = self.mt5_bridge.get_symbol_info(symbol)
            return info.get('bid', 1.0) if info else 1.0
        return 1.0
    
    def _get_account_info(self) -> Dict:
        """Get account information"""
        if self.mt5_bridge:
            return self.mt5_bridge.get_account_info() or {}
        return {"equity": 10000, "balance": 10000, "margin": 0, "free_margin": 10000, "margin_level": 1000}
    
    def _get_symbol_exposure(self, symbol: str) -> float:
        """Get current exposure for symbol"""
        # Mock implementation
        return 0.0
    
    def _get_currency_exposure(self, currency: str) -> float:
        """Get current exposure for currency"""
        return self.currency_exposure.get(currency, 0.0)
    
    def _check_correlation_limits(self, symbol: str) -> Dict:
        """Check correlation with existing positions"""
        # Mock correlation data
        correlations = {
            "EURUSD": {"GBPUSD": 0.75, "USDCHF": -0.85},
            "GBPUSD": {"EURUSD": 0.75, "USDJPY": -0.58},
            "USDJPY": {"EURUSD": -0.65, "GBPUSD": -0.58}
        }
        return correlations.get(symbol, {})
    
    def _get_daily_loss(self) -> float:
        """Get today's realized loss"""
        return 0.0  # Mock
    
    def _get_current_drawdown(self) -> float:
        """Get current drawdown percentage"""
        return 0.0  # Mock
    
    def _get_open_positions_count(self) -> int:
        """Get number of open positions"""
        if self.trade_manager:
            return len([t for t in self.trade_manager.trades.values() if t.status.value == 'open'])
        return 0
    
    def _get_pip_value(self, symbol: str) -> float:
        """Get pip value for symbol"""
        return 10.0  # Mock
    
    def _calculate_reward_risk_ratio(self, entry: float, stop_loss: float, take_profit: float) -> float:
        """Calculate reward to risk ratio"""
        if not take_profit:
            return None
        
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        
        return reward / risk if risk > 0 else None
    
    def _update_risk_metrics(self):
        """Update risk metrics"""
        try:
            # Get current account info
            if self.mt5_bridge:
                account_info = self.mt5_bridge.get_account_info()
                if account_info:
                    # Update basic metrics
                    self.risk_metrics['account_equity'] = account_info.get('equity', 0)
                    self.risk_metrics['account_balance'] = account_info.get('balance', 0)
                    self.risk_metrics['account_margin'] = account_info.get('margin', 0)
                    self.risk_metrics['margin_level'] = account_info.get('margin_level', 0)
                    self.risk_metrics['free_margin'] = account_info.get('margin_free', 0)
                    
                    # Calculate leverage used
                    self.risk_metrics['leverage_used'] = self._calculate_leverage_used(account_info)
                    
                    # Get current positions
                    positions = self.mt5_bridge.get_positions()
                    if positions:
                        # Calculate total exposure
                        total_exposure = sum(pos.get('volume', 0) * pos.get('price_current', 0) for pos in positions)
                        self.risk_metrics['total_exposure'] = total_exposure
                        
                        # Calculate daily P&L
                        daily_pnl = sum(pos.get('profit', 0) for pos in positions)
                        self.risk_metrics['daily_pnl'] = daily_pnl
                        
                        # Update currency exposure
                        self._update_currency_exposure(positions)
                        
                        # Calculate drawdown
                        equity = self.risk_metrics['account_equity']
                        balance = self.risk_metrics['account_balance']
                        if balance > 0:
                            current_drawdown = max(0, (balance - equity) / balance * 100)
                            self.risk_metrics['current_drawdown'] = current_drawdown
                    
                    # Update timestamp
                    self.risk_metrics['last_update'] = datetime.now().isoformat()
                    
            logger.debug(f"Risk metrics updated: {self.risk_metrics}")
            
        except Exception as e:
            logger.error(f"Error updating risk metrics: {e}")
            # Set safe defaults
            self.risk_metrics['last_update'] = datetime.now().isoformat()
            self.risk_metrics['status'] = 'error'
    
    def _check_risk_violations(self):
        """Check for risk violations"""
        try:
            violations = []
            
            # Check drawdown limit
            current_drawdown = self.risk_metrics.get('current_drawdown', 0)
            if current_drawdown > self.risk_params.max_drawdown:
                violations.append({
                    'type': 'drawdown_exceeded',
                    'severity': 'critical',
                    'message': f'Drawdown {current_drawdown:.2f}% exceeds limit {self.risk_params.max_drawdown}%',
                    'value': current_drawdown,
                    'limit': self.risk_params.max_drawdown
                })
            
            # Check daily loss limit
            daily_pnl = self.risk_metrics.get('daily_pnl', 0)
            account_equity = self.risk_metrics.get('account_equity', 1)
            if daily_pnl < 0 and account_equity > 0:
                daily_loss_pct = abs(daily_pnl) / account_equity * 100
                if daily_loss_pct > self.risk_params.max_daily_loss:
                    violations.append({
                        'type': 'daily_loss_exceeded',
                        'severity': 'high',
                        'message': f'Daily loss {daily_loss_pct:.2f}% exceeds limit {self.risk_params.max_daily_loss}%',
                        'value': daily_loss_pct,
                        'limit': self.risk_params.max_daily_loss
                    })
            
            # Check margin level
            margin_level = self.risk_metrics.get('margin_level', 100)
            if margin_level < self.risk_params.margin_call_threshold:
                violations.append({
                    'type': 'margin_call_risk',
                    'severity': 'critical',
                    'message': f'Margin level {margin_level:.2f}% below threshold {self.risk_params.margin_call_threshold}%',
                    'value': margin_level,
                    'limit': self.risk_params.margin_call_threshold
                })
            
            # Check leverage limit
            leverage_used = self.risk_metrics.get('leverage_used', 0)
            if leverage_used > self.risk_params.leverage_limit:
                violations.append({
                    'type': 'leverage_exceeded',
                    'severity': 'high',
                    'message': f'Leverage {leverage_used:.2f}x exceeds limit {self.risk_params.leverage_limit}x',
                    'value': leverage_used,
                    'limit': self.risk_params.leverage_limit
                })
            
            # Check currency exposure limits
            for currency, exposure in self.currency_exposure.items():
                if exposure > self.risk_params.max_exposure_per_currency:
                    violations.append({
                        'type': 'currency_exposure_exceeded',
                        'severity': 'medium',
                        'message': f'{currency} exposure {exposure:.2f}% exceeds limit {self.risk_params.max_exposure_per_currency}%',
                        'value': exposure,
                        'limit': self.risk_params.max_exposure_per_currency,
                        'currency': currency
                    })
            
            # Process violations
            for violation in violations:
                self._handle_risk_violation(violation)
                
            # Update violation count
            self.risk_metrics['violation_count'] = len(violations)
            
            if violations:
                logger.warning(f"Risk violations detected: {len(violations)}")
                
        except Exception as e:
            logger.error(f"Error checking risk violations: {e}")
    
    def _update_position_limits(self):
        """Update position limits based on current conditions"""
        try:
            # Get current account equity
            account_equity = self.risk_metrics.get('account_equity', 0)
            if account_equity <= 0:
                logger.warning("Invalid account equity for position limits calculation")
                return
            
            # Update maximum position size per symbol
            max_risk_per_symbol = account_equity * (self.risk_params.max_exposure_per_symbol / 100)
            
            # Get current positions to calculate used limits
            if self.mt5_bridge:
                positions = self.mt5_bridge.get_positions()
                symbol_exposure = {}
                
                if positions:
                    for pos in positions:
                        symbol = pos.get('symbol', '')
                        volume = pos.get('volume', 0)
                        price = pos.get('price_current', 0)
                        exposure = volume * price
                        
                        if symbol in symbol_exposure:
                            symbol_exposure[symbol] += exposure
                        else:
                            symbol_exposure[symbol] = exposure
                
                # Calculate available limits for each symbol
                for symbol in symbol_exposure:
                    used_exposure = symbol_exposure[symbol]
                    available_exposure = max_risk_per_symbol - used_exposure
                    
                    self.position_limits[symbol] = {
                        'max_exposure': max_risk_per_symbol,
                        'used_exposure': used_exposure,
                        'available_exposure': max(0, available_exposure),
                        'exposure_percentage': (used_exposure / account_equity) * 100
                    }
            
            # Update global position limits
            self.position_limits['global'] = {
                'max_positions': self.risk_params.max_open_positions,
                'max_risk_per_trade': account_equity * (self.risk_params.max_risk_per_trade / 100),
                'max_daily_loss': account_equity * (self.risk_params.max_daily_loss / 100),
                'account_equity': account_equity,
                'last_update': datetime.now().isoformat()
            }
            
            logger.debug(f"Position limits updated for {len(self.position_limits)} symbols")
            
        except Exception as e:
            logger.error(f"Error updating position limits: {e}")
    
    def _process_alerts(self):
        """Process pending alerts"""
        try:
            # Process alerts from queue
            while not self.alert_queue.empty():
                try:
                    alert = self.alert_queue.get_nowait()
                    
                    # Add timestamp if not present
                    if 'timestamp' not in alert:
                        alert['timestamp'] = datetime.now().isoformat()
                    
                    # Add to alerts list
                    self.alerts.append(alert)
                    
                    # Log alert based on severity
                    severity = alert.get('severity', 'info')
                    message = alert.get('message', 'Unknown alert')
                    
                    if severity == 'critical':
                        logger.critical(f"RISK ALERT: {message}")
                    elif severity == 'high':
                        logger.error(f"RISK ALERT: {message}")
                    elif severity == 'medium':
                        logger.warning(f"RISK ALERT: {message}")
                    else:
                        logger.info(f"RISK ALERT: {message}")
                    
                    # Send notification if configured
                    if hasattr(self, 'notification_system') and self.notification_system:
                        self.notification_system.send_risk_alert(alert)
                    
                    # Take automated action if required
                    self._handle_alert_action(alert)
                    
                    # Mark as processed
                    self.alert_queue.task_done()
                    
                except queue.Empty:
                    break
                except Exception as e:
                    logger.error(f"Error processing alert: {e}")
            
            # Clean up old alerts (keep last 100)
            if len(self.alerts) > 100:
                self.alerts = self.alerts[-100:]
                
        except Exception as e:
            logger.error(f"Error in alert processing: {e}")
    
    def _update_currency_exposure(self, positions: List[Dict]):
        """Update currency exposure based on current positions"""
        try:
            currency_exposure = {}
            account_equity = self.risk_metrics.get('account_equity', 1)
            
            for pos in positions:
                symbol = pos.get('symbol', '')
                volume = pos.get('volume', 0)
                price = pos.get('price_current', 0)
                exposure = volume * price
                
                # Extract base and quote currencies from symbol
                if len(symbol) >= 6:
                    base_currency = symbol[:3]
                    quote_currency = symbol[3:6]
                    
                    # Add to base currency exposure
                    if base_currency in currency_exposure:
                        currency_exposure[base_currency] += exposure
                    else:
                        currency_exposure[base_currency] = exposure
                    
                    # Subtract from quote currency exposure
                    if quote_currency in currency_exposure:
                        currency_exposure[quote_currency] -= exposure
                    else:
                        currency_exposure[quote_currency] = -exposure
            
            # Convert to percentages
            for currency in currency_exposure:
                if account_equity > 0:
                    currency_exposure[currency] = abs(currency_exposure[currency]) / account_equity * 100
            
            self.currency_exposure = currency_exposure
            
        except Exception as e:
            logger.error(f"Error updating currency exposure: {e}")
    
    def _handle_risk_violation(self, violation: Dict):
        """Handle a specific risk violation"""
        try:
            # Add to alert queue
            alert = {
                'type': 'risk_violation',
                'severity': violation.get('severity', 'medium'),
                'message': violation.get('message', 'Risk violation detected'),
                'violation_type': violation.get('type'),
                'value': violation.get('value'),
                'limit': violation.get('limit'),
                'timestamp': datetime.now().isoformat()
            }
            
            self.alert_queue.put(alert)
            
            # Take immediate action for critical violations
            if violation.get('severity') == 'critical':
                self._handle_critical_violation(violation)
                
        except Exception as e:
            logger.error(f"Error handling risk violation: {e}")
    
    def _handle_critical_violation(self, violation: Dict):
        """Handle critical risk violations with immediate action"""
        try:
            violation_type = violation.get('type')
            
            if violation_type == 'drawdown_exceeded':
                logger.critical("EMERGENCY: Maximum drawdown exceeded - consider closing positions")
                # Could implement auto-close here if configured
                
            elif violation_type == 'margin_call_risk':
                logger.critical("EMERGENCY: Margin call risk - immediate action required")
                # Could implement emergency position closure
                
        except Exception as e:
            logger.error(f"Error handling critical violation: {e}")
    
    def _handle_alert_action(self, alert: Dict):
        """Handle automated actions based on alert type"""
        try:
            alert_type = alert.get('type')
            severity = alert.get('severity')
            
            # Log the alert action
            logger.info(f"Processing alert action: {alert_type} - {severity}")
            
            # Implement specific actions based on alert type
            # This is where you could add automated trading halts,
            # position adjustments, or notifications
            
        except Exception as e:
            logger.error(f"Error handling alert action: {e}")
    
    def _calculate_leverage_used(self, account_info: Dict) -> float:
        """Calculate leverage currently used"""
        equity = account_info.get('equity', 1)
        margin = account_info.get('margin', 0)
        return margin / equity if equity > 0 else 0
    
    def _get_daily_pnl(self) -> float:
        """Get daily P&L"""
        try:
            if self.mt5_bridge:
                positions = self.mt5_bridge.get_positions()
                if positions:
                    return sum(pos.get('profit', 0) for pos in positions)
            return self.risk_metrics.get('daily_pnl', 0.0)
        except Exception as e:
            logger.error(f"Error getting daily P&L: {e}")
            return 0.0
    
    def _get_total_exposure(self) -> float:
        """Get total exposure"""
        try:
            if self.mt5_bridge:
                positions = self.mt5_bridge.get_positions()
                if positions:
                    return sum(pos.get('volume', 0) * pos.get('price_current', 0) for pos in positions)
            return self.risk_metrics.get('total_exposure', 0.0)
        except Exception as e:
            logger.error(f"Error getting total exposure: {e}")
            return 0.0
    
    def _calculate_risk_score(self) -> float:
        """Calculate overall risk score (0-100)"""
        try:
            score = 0.0
            
            # Drawdown component (0-30 points)
            drawdown = self.risk_metrics.get('current_drawdown', 0)
            if drawdown > 0:
                drawdown_score = min(30, (drawdown / self.risk_params.max_drawdown) * 30)
                score += drawdown_score
            
            # Leverage component (0-25 points)
            leverage = self.risk_metrics.get('leverage_used', 0)
            if leverage > 0:
                leverage_score = min(25, (leverage / self.risk_params.leverage_limit) * 25)
                score += leverage_score
            
            # Exposure component (0-20 points)
            total_exposure = self._get_total_exposure()
            account_equity = self.risk_metrics.get('account_equity', 1)
            if account_equity > 0:
                exposure_ratio = total_exposure / account_equity
                exposure_score = min(20, exposure_ratio * 20)
                score += exposure_score
            
            # Margin level component (0-15 points)
            margin_level = self.risk_metrics.get('margin_level', 1000)
            if margin_level < 1000:
                margin_score = max(0, 15 - (margin_level / 1000) * 15)
                score += margin_score
            
            # Violation component (0-10 points)
            violations = self.risk_metrics.get('violation_count', 0)
            violation_score = min(10, violations * 2)
            score += violation_score
            
            return min(100, max(0, score))
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 50.0  # Default moderate risk
    
    def _get_all_symbol_exposure(self) -> Dict:
        """Get exposure for all symbols"""
        try:
            symbol_exposure = {}
            if self.mt5_bridge:
                positions = self.mt5_bridge.get_positions()
                if positions:
                    for pos in positions:
                        symbol = pos.get('symbol', '')
                        volume = pos.get('volume', 0)
                        price = pos.get('price_current', 0)
                        exposure = volume * price
                        profit = pos.get('profit', 0)
                        
                        if symbol in symbol_exposure:
                            symbol_exposure[symbol]['exposure'] += exposure
                            symbol_exposure[symbol]['volume'] += volume
                            symbol_exposure[symbol]['profit'] += profit
                        else:
                            symbol_exposure[symbol] = {
                                'exposure': exposure,
                                'volume': volume,
                                'profit': profit,
                                'positions': 1
                            }
            return symbol_exposure
        except Exception as e:
            logger.error(f"Error getting symbol exposure: {e}")
            return {}
    
    def _get_all_currency_exposure(self) -> Dict:
        """Get exposure for all currencies"""
        return self.currency_exposure
    
    def _get_correlation_risks(self) -> List[Dict]:
        """Get correlation risks"""
        return []  # Mock
    
    def _get_risk_recommendations(self) -> List[str]:
        """Get risk management recommendations"""
        return [
            "Consider reducing position sizes due to high market volatility",
            "Monitor USD exposure - approaching limit",
            "Review correlation between EUR positions"
        ]
    
    def _load_risk_config(self):
        """Load risk configuration from file"""
        try:
            with open('risk_config.json', 'r') as f:
                config = json.load(f)
                for key, value in config.items():
                    if hasattr(self.risk_params, key):
                        setattr(self.risk_params, key, value)
            logger.info("Risk configuration loaded")
        except FileNotFoundError:
            logger.info("No risk configuration file found, using defaults")
        except Exception as e:
            logger.error(f"Error loading risk configuration: {e}")
    
    def _save_risk_config(self):
        """Save risk configuration to file"""
        try:
            with open('risk_config.json', 'w') as f:
                json.dump(asdict(self.risk_params), f, indent=2)
            logger.info("Risk configuration saved")
        except Exception as e:
            logger.error(f"Error saving risk configuration: {e}")