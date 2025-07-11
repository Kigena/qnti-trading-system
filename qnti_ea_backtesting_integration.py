#!/usr/bin/env python3
"""
QNTI EA Backtesting Integration - Seamless Strategy Testing
Integration layer between EA Generation Engine and backtesting systems
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
import threading
import time
import json
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import from main systems
from qnti_ea_generation_engine import EATemplate, EAGenerationEngine, IndicatorLibrary
from qnti_ea_optimization_engine import BacktestMetrics
from qnti_backtesting_engine import QNTIBacktestingEngine  # Assuming this exists

logger = logging.getLogger('QNTI_EA_BACKTESTING')

@dataclass
class BacktestConfiguration:
    """Configuration for EA backtesting"""
    start_date: datetime
    end_date: datetime
    symbol: str
    timeframe: str = "H1"
    initial_balance: float = 10000.0
    commission: float = 0.0002  # 0.02% commission
    slippage: float = 0.0001   # 0.01% slippage
    spread: float = 0.0002     # 0.02% spread
    
    # Advanced settings
    max_positions: int = 1
    margin_requirement: float = 1.0  # 1:1 leverage
    swap_rates: Dict[str, float] = field(default_factory=lambda: {"long": 0.0, "short": 0.0})
    
    # Data settings
    warmup_periods: int = 200  # Periods for indicator initialization
    tick_data: bool = False    # Use tick data if available
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result['start_date'] = self.start_date.isoformat()
        result['end_date'] = self.end_date.isoformat()
        return result

@dataclass
class Trade:
    """Individual trade record"""
    id: str
    symbol: str
    direction: str  # "long" or "short"
    entry_time: datetime
    entry_price: float
    volume: float
    
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    commission: float = 0.0
    swap: float = 0.0
    
    # Trade metadata
    entry_reason: str = ""
    exit_reason: str = ""
    indicators_at_entry: Dict[str, float] = field(default_factory=dict)
    
    @property
    def is_open(self) -> bool:
        """Check if trade is still open"""
        return self.exit_time is None
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Get trade duration"""
        if self.exit_time:
            return self.exit_time - self.entry_time
        return None

@dataclass
class PortfolioState:
    """Current portfolio state during backtesting"""
    balance: float
    equity: float
    margin_used: float
    free_margin: float
    open_trades: List[Trade]
    closed_trades: List[Trade]
    
    def add_trade(self, trade: Trade):
        """Add a new trade"""
        if trade.is_open:
            self.open_trades.append(trade)
        else:
            self.closed_trades.append(trade)
    
    def close_trade(self, trade_id: str, exit_time: datetime, 
                   exit_price: float, exit_reason: str = ""):
        """Close an open trade"""
        for trade in self.open_trades:
            if trade.id == trade_id:
                trade.exit_time = exit_time
                trade.exit_price = exit_price
                trade.exit_reason = exit_reason
                
                # Calculate P&L
                if trade.direction == "long":
                    trade.pnl = (exit_price - trade.entry_price) * trade.volume
                else:
                    trade.pnl = (trade.entry_price - exit_price) * trade.volume
                
                # Move to closed trades
                self.open_trades.remove(trade)
                self.closed_trades.append(trade)
                break

class EASignalGenerator:
    """Generate trading signals from EA template"""
    
    def __init__(self, ea_template: EATemplate, indicator_library: IndicatorLibrary):
        self.ea_template = ea_template
        self.indicator_library = indicator_library
        self.indicator_cache = {}
        
    def calculate_indicators(self, data: pd.DataFrame, 
                           parameters: Dict[str, Any]) -> Dict[str, pd.Series]:
        """Calculate all required indicators"""
        results = {}
        
        for indicator_name in self.ea_template.indicators:
            # Extract parameters for this indicator
            indicator_params = self._extract_indicator_params(indicator_name, parameters)
            
            try:
                # Calculate indicator
                indicator_results = self.indicator_library.calculate_indicator(
                    indicator_name, data, indicator_params
                )
                
                # Add to results with prefixed names
                for output_name, series in indicator_results.items():
                    full_name = f"{indicator_name}_{output_name}"
                    results[full_name] = series
                    
            except Exception as e:
                logger.error(f"Error calculating indicator {indicator_name}: {e}")
                # Create empty series as fallback
                results[f"{indicator_name}_error"] = pd.Series(0.0, index=data.index)
        
        return results
    
    def _extract_indicator_params(self, indicator_name: str, 
                                 all_params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract parameters for specific indicator"""
        indicator_params = {}
        prefix = f"{indicator_name}_"
        
        for param_name, value in all_params.items():
            if param_name.startswith(prefix):
                clean_name = param_name[len(prefix):]
                indicator_params[clean_name] = value
        
        return indicator_params
    
    def generate_signals(self, data: pd.DataFrame, 
                        indicators: Dict[str, pd.Series],
                        parameters: Dict[str, Any]) -> pd.DataFrame:
        """Generate entry/exit signals based on trading rules"""
        
        signals = pd.DataFrame(index=data.index)
        signals['entry_long'] = False
        signals['entry_short'] = False
        signals['exit_long'] = False
        signals['exit_short'] = False
        signals['entry_confidence'] = 0.0
        signals['exit_confidence'] = 0.0
        
        # Evaluate trading rules for each bar
        for i, timestamp in enumerate(data.index):
            if i < 1:  # Skip first bar (no previous data)
                continue
            
            # Create data context for condition evaluation
            context = self._create_context(data, indicators, i)
            
            # Evaluate entry conditions
            for rule in self.ea_template.trading_rules:
                try:
                    entry_signal = rule.entry_conditions.evaluate(context)
                    if entry_signal:
                        # Determine direction (simplified logic)
                        if self._should_go_long(context, rule):
                            signals.loc[timestamp, 'entry_long'] = True
                            signals.loc[timestamp, 'entry_confidence'] = 0.8
                        else:
                            signals.loc[timestamp, 'entry_short'] = True
                            signals.loc[timestamp, 'entry_confidence'] = 0.8
                    
                    # Evaluate exit conditions
                    exit_signal = rule.exit_conditions.evaluate(context)
                    if exit_signal:
                        signals.loc[timestamp, 'exit_long'] = True
                        signals.loc[timestamp, 'exit_short'] = True
                        signals.loc[timestamp, 'exit_confidence'] = 0.9
                        
                except Exception as e:
                    logger.error(f"Error evaluating rule at {timestamp}: {e}")
        
        return signals
    
    def _create_context(self, data: pd.DataFrame, 
                       indicators: Dict[str, pd.Series], i: int) -> Dict[str, Any]:
        """Create evaluation context for current bar"""
        context = {}
        
        # Add price data
        context['open'] = data.iloc[i]['open']
        context['high'] = data.iloc[i]['high']
        context['low'] = data.iloc[i]['low']
        context['close'] = data.iloc[i]['close']
        context['volume'] = data.iloc[i].get('volume', 0)
        
        # Add previous price data
        if i > 0:
            context['open_prev'] = data.iloc[i-1]['open']
            context['high_prev'] = data.iloc[i-1]['high']
            context['low_prev'] = data.iloc[i-1]['low']
            context['close_prev'] = data.iloc[i-1]['close']
        
        # Add indicator values
        for name, series in indicators.items():
            if i < len(series) and not pd.isna(series.iloc[i]):
                context[name] = series.iloc[i]
                # Add previous values for crossing detection
                if i > 0 and not pd.isna(series.iloc[i-1]):
                    context[f"{name}_prev"] = series.iloc[i-1]
        
        return context
    
    def _should_go_long(self, context: Dict[str, Any], rule) -> bool:
        """Determine if signal should be long (simplified logic)"""
        # This is a simplified implementation
        # In practice, this would analyze the specific conditions
        return True  # Default to long

class EABacktester:
    """Main EA backtesting engine"""
    
    def __init__(self, mt5_bridge=None):
        self.mt5_bridge = mt5_bridge
        self.indicator_library = IndicatorLibrary()
        
    def run_backtest(self, ea_template: EATemplate,
                    config: BacktestConfiguration,
                    parameters: Dict[str, Any],
                    data: pd.DataFrame = None) -> BacktestMetrics:
        """Run complete backtest for EA"""
        
        logger.info(f"Starting backtest for EA: {ea_template.name}")
        
        # Get market data if not provided
        if data is None:
            data = self._get_market_data(config)
        
        # Initialize signal generator
        signal_generator = EASignalGenerator(ea_template, self.indicator_library)
        
        # Calculate indicators
        indicators = signal_generator.calculate_indicators(data, parameters)
        
        # Generate signals
        signals = signal_generator.generate_signals(data, indicators, parameters)
        
        # Run simulation
        portfolio_states = self._run_simulation(data, signals, config)
        
        # Calculate performance metrics
        metrics = self._calculate_metrics(portfolio_states, config)
        
        logger.info(f"Backtest completed. Return: {metrics.annual_return:.2%}")
        
        return metrics
    
    def _get_market_data(self, config: BacktestConfiguration) -> pd.DataFrame:
        """Get market data for backtesting"""
        if self.mt5_bridge:
            # Get data from MT5
            try:
                data = self.mt5_bridge.get_historical_data(
                    symbol=config.symbol,
                    timeframe=config.timeframe,
                    start_date=config.start_date,
                    end_date=config.end_date
                )
                return data
            except Exception as e:
                logger.error(f"Error getting MT5 data: {e}")
        
        # Generate synthetic data as fallback
        return self._generate_synthetic_data(config)
    
    def _generate_synthetic_data(self, config: BacktestConfiguration) -> pd.DataFrame:
        """Generate synthetic market data for testing"""
        logger.warning("Using synthetic data for backtesting")
        
        # Create date range
        dates = pd.date_range(
            start=config.start_date,
            end=config.end_date,
            freq='H' if config.timeframe == 'H1' else 'D'
        )
        
        # Generate realistic price data with trends and volatility
        n_periods = len(dates)
        base_price = 1.1000  # EURUSD-like
        
        # Generate returns with some autocorrelation
        returns = np.random.normal(0, 0.001, n_periods)  # 0.1% daily volatility
        for i in range(1, len(returns)):
            returns[i] += 0.1 * returns[i-1]  # Add some persistence
        
        # Calculate prices
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Create OHLC data
        data = pd.DataFrame(index=dates)
        data['close'] = prices
        
        # Generate open, high, low based on close
        data['open'] = data['close'].shift(1).fillna(data['close'].iloc[0])
        
        # Add intrabar volatility
        volatility = np.abs(np.random.normal(0, 0.0005, n_periods))
        data['high'] = data[['open', 'close']].max(axis=1) + volatility
        data['low'] = data[['open', 'close']].min(axis=1) - volatility
        
        # Add volume
        data['volume'] = np.random.exponential(1000, n_periods)
        
        return data
    
    def _run_simulation(self, data: pd.DataFrame, signals: pd.DataFrame,
                       config: BacktestConfiguration) -> List[PortfolioState]:
        """Run trading simulation"""
        
        portfolio = PortfolioState(
            balance=config.initial_balance,
            equity=config.initial_balance,
            margin_used=0.0,
            free_margin=config.initial_balance,
            open_trades=[],
            closed_trades=[]
        )
        
        portfolio_history = []
        trade_id_counter = 0
        
        for i, (timestamp, row) in enumerate(data.iterrows()):
            if i < config.warmup_periods:
                continue  # Skip warmup period
            
            current_price = row['close']
            
            # Update open positions
            self._update_open_positions(portfolio, current_price)
            
            # Process exit signals first
            signal_row = signals.loc[timestamp]
            if signal_row['exit_long'] or signal_row['exit_short']:
                self._process_exit_signals(portfolio, timestamp, current_price, signal_row)
            
            # Process entry signals
            if (signal_row['entry_long'] or signal_row['entry_short']) and len(portfolio.open_trades) < config.max_positions:
                trade_id_counter += 1
                self._process_entry_signals(
                    portfolio, timestamp, current_price, signal_row, 
                    trade_id_counter, config
                )
            
            # Update portfolio equity
            portfolio.equity = self._calculate_equity(portfolio, current_price)
            
            # Store portfolio state
            portfolio_state_copy = PortfolioState(
                balance=portfolio.balance,
                equity=portfolio.equity,
                margin_used=portfolio.margin_used,
                free_margin=portfolio.free_margin,
                open_trades=portfolio.open_trades.copy(),
                closed_trades=portfolio.closed_trades.copy()
            )
            portfolio_history.append(portfolio_state_copy)
        
        # Close any remaining open trades
        final_price = data['close'].iloc[-1]
        final_timestamp = data.index[-1]
        for trade in portfolio.open_trades.copy():
            portfolio.close_trade(trade.id, final_timestamp, final_price, "End of backtest")
        
        return portfolio_history
    
    def _update_open_positions(self, portfolio: PortfolioState, current_price: float):
        """Update unrealized P&L for open positions"""
        for trade in portfolio.open_trades:
            if trade.direction == "long":
                unrealized_pnl = (current_price - trade.entry_price) * trade.volume
            else:
                unrealized_pnl = (trade.entry_price - current_price) * trade.volume
            
            trade.pnl = unrealized_pnl
    
    def _process_exit_signals(self, portfolio: PortfolioState, 
                             timestamp: datetime, price: float, signals):
        """Process exit signals"""
        for trade in portfolio.open_trades.copy():
            should_exit = False
            
            if trade.direction == "long" and signals['exit_long']:
                should_exit = True
            elif trade.direction == "short" and signals['exit_short']:
                should_exit = True
            
            if should_exit:
                portfolio.close_trade(trade.id, timestamp, price, "Signal exit")
                portfolio.balance += trade.pnl
    
    def _process_entry_signals(self, portfolio: PortfolioState,
                              timestamp: datetime, price: float, signals,
                              trade_id: int, config: BacktestConfiguration):
        """Process entry signals"""
        
        # Calculate position size (simplified)
        position_size = self._calculate_position_size(portfolio, config)
        
        if signals['entry_long']:
            trade = Trade(
                id=f"trade_{trade_id}",
                symbol=config.symbol,
                direction="long",
                entry_time=timestamp,
                entry_price=price * (1 + config.slippage),  # Add slippage
                volume=position_size,
                commission=position_size * price * config.commission,
                entry_reason="Signal entry long"
            )
            portfolio.add_trade(trade)
            portfolio.balance -= trade.commission
            
        elif signals['entry_short']:
            trade = Trade(
                id=f"trade_{trade_id}",
                symbol=config.symbol,
                direction="short",
                entry_time=timestamp,
                entry_price=price * (1 - config.slippage),  # Add slippage
                volume=position_size,
                commission=position_size * price * config.commission,
                entry_reason="Signal entry short"
            )
            portfolio.add_trade(trade)
            portfolio.balance -= trade.commission
    
    def _calculate_position_size(self, portfolio: PortfolioState,
                                config: BacktestConfiguration) -> float:
        """Calculate position size based on risk management"""
        # Simplified: fixed percentage of equity
        risk_percent = 0.02  # 2% risk per trade
        return portfolio.equity * risk_percent / 1000  # Simplified calculation
    
    def _calculate_equity(self, portfolio: PortfolioState, current_price: float) -> float:
        """Calculate current portfolio equity"""
        equity = portfolio.balance
        
        # Add unrealized P&L from open trades
        for trade in portfolio.open_trades:
            equity += trade.pnl or 0
        
        return equity
    
    def _calculate_metrics(self, portfolio_history: List[PortfolioState],
                          config: BacktestConfiguration) -> BacktestMetrics:
        """Calculate comprehensive performance metrics"""
        
        if not portfolio_history:
            return BacktestMetrics()
        
        # Extract equity curve
        equity_curve = [state.equity for state in portfolio_history]
        returns = pd.Series(equity_curve).pct_change().dropna()
        
        # Get all closed trades
        all_trades = []
        for state in portfolio_history:
            all_trades.extend(state.closed_trades)
        
        # Remove duplicates (keep unique trade IDs)
        unique_trades = {}
        for trade in all_trades:
            if trade.id not in unique_trades:
                unique_trades[trade.id] = trade
        
        closed_trades = list(unique_trades.values())
        
        # Calculate basic metrics
        final_equity = equity_curve[-1]
        initial_equity = config.initial_balance
        total_return = (final_equity - initial_equity) / initial_equity
        
        # Annualized return
        days = (config.end_date - config.start_date).days
        annual_return = (1 + total_return) ** (365.25 / days) - 1
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252 * 24)  # Assuming hourly data
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Drawdown calculation
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (np.array(equity_curve) - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Trade-based metrics
        trade_returns = [trade.pnl for trade in closed_trades if trade.pnl is not None]
        
        if trade_returns:
            winning_trades = [r for r in trade_returns if r > 0]
            losing_trades = [r for r in trade_returns if r < 0]
            
            win_rate = len(winning_trades) / len(trade_returns)
            avg_win = np.mean(winning_trades) if winning_trades else 0
            avg_loss = np.mean(losing_trades) if losing_trades else 0
            profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else float('inf')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        # Sortino ratio
        negative_returns = returns[returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252 * 24)
        sortino_ratio = annual_return / downside_deviation if downside_deviation > 0 else 0
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        return BacktestMetrics(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            total_trades=len(closed_trades),
            winning_trades=len([r for r in trade_returns if r > 0]),
            losing_trades=len([r for r in trade_returns if r < 0]),
            volatility=volatility
        )

class EABacktestingIntegration:
    """Main integration class connecting EA generation with backtesting"""
    
    def __init__(self, mt5_bridge=None):
        self.ea_generation_engine = EAGenerationEngine(mt5_bridge)
        self.backtester = EABacktester(mt5_bridge)
        
    def backtest_ea(self, ea_template: EATemplate, parameters: Dict[str, Any],
                   symbol: str, start_date: datetime, end_date: datetime,
                   **kwargs) -> BacktestMetrics:
        """Convenient method to backtest an EA"""
        
        config = BacktestConfiguration(
            start_date=start_date,
            end_date=end_date,
            symbol=symbol,
            **kwargs
        )
        
        return self.backtester.run_backtest(ea_template, config, parameters)
    
    def batch_backtest(self, ea_templates: List[EATemplate],
                      parameter_sets: List[Dict[str, Any]],
                      config: BacktestConfiguration) -> List[Tuple[str, BacktestMetrics]]:
        """Run backtests for multiple EAs or parameter sets"""
        
        results = []
        
        for i, (ea_template, params) in enumerate(zip(ea_templates, parameter_sets)):
            try:
                logger.info(f"Running backtest {i+1}/{len(ea_templates)}: {ea_template.name}")
                metrics = self.backtester.run_backtest(ea_template, config, params)
                results.append((ea_template.id, metrics))
                
            except Exception as e:
                logger.error(f"Error in backtest {i+1}: {e}")
                results.append((ea_template.id, BacktestMetrics()))  # Empty metrics
        
        return results

# Example usage
if __name__ == "__main__":
    # Create EA integration
    integration = EABacktestingIntegration()
    
    # Create test EA
    ea = integration.ea_generation_engine.create_ea_template(
        name="Test Strategy",
        description="Simple test strategy for backtesting",
        author="QNTI"
    )
    
    # Add indicators
    integration.ea_generation_engine.add_indicator_to_ea(ea.id, "SMA", {"period": 20})
    integration.ea_generation_engine.add_indicator_to_ea(ea.id, "RSI", {"period": 14})
    
    # Test parameters
    test_params = {
        "SMA_period": 20,
        "RSI_period": 14,
        "RSI_overbought": 70,
        "RSI_oversold": 30
    }
    
    # Run backtest
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    metrics = integration.backtest_ea(
        ea, test_params, "EURUSD", start_date, end_date
    )
    
    print(f"Backtest Results:")
    print(f"Annual Return: {metrics.annual_return:.2%}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"Win Rate: {metrics.win_rate:.2%}")
    print(f"Total Trades: {metrics.total_trades}")