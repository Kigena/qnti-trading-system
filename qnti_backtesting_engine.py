#!/usr/bin/env python3
"""
QNTI Backtesting Engine - Advanced Strategy Testing & Analysis
Comprehensive backtesting framework for trading strategies with performance analytics
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import sqlite3
import yfinance as yf
import ta
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Order types for backtesting"""
    BUY = "buy"
    SELL = "sell"

class BacktestMode(Enum):
    """Backtesting modes"""
    HISTORICAL = "historical"
    MONTE_CARLO = "monte_carlo"
    WALK_FORWARD = "walk_forward"

@dataclass
class BacktestOrder:
    """Backtest order data structure"""
    id: str
    symbol: str
    order_type: OrderType
    quantity: float
    entry_price: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    commission: float = 0.0
    slippage: float = 0.0
    is_open: bool = True
    profit_loss: float = 0.0
    holding_period: Optional[timedelta] = None

@dataclass
class BacktestResult:
    """Backtest results data structure"""
    strategy_name: str
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    profit_factor: float
    trades: List[BacktestOrder]
    equity_curve: List[Tuple[datetime, float]]
    daily_returns: List[float]
    performance_metrics: Dict[str, Any]

class TradingStrategy:
    """Base class for trading strategies"""
    
    def __init__(self, name: str, parameters: Dict = None):
        self.name = name
        self.parameters = parameters or {}
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement generate_signals method")
    
    def should_buy(self, data: pd.DataFrame, index: int) -> bool:
        """Check if should buy at given index"""
        return False
    
    def should_sell(self, data: pd.DataFrame, index: int) -> bool:
        """Check if should sell at given index"""
        return False
    
    def calculate_position_size(self, capital: float, price: float, risk_per_trade: float = 0.02) -> float:
        """Calculate position size based on risk management"""
        return (capital * risk_per_trade) / price

class MovingAverageCrossStrategy(TradingStrategy):
    """Simple moving average crossover strategy"""
    
    def __init__(self, short_window: int = 10, long_window: int = 30):
        super().__init__("MA_Cross", {"short_window": short_window, "long_window": long_window})
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate MA crossover signals"""
        signals = data.copy()
        signals['MA_short'] = signals['Close'].rolling(window=self.short_window).mean()
        signals['MA_long'] = signals['Close'].rolling(window=self.long_window).mean()
        
        # Generate signals
        signals['signal'] = 0
        signals['signal'][self.short_window:] = np.where(
            signals['MA_short'][self.short_window:] > signals['MA_long'][self.short_window:], 1, 0
        )
        signals['positions'] = signals['signal'].diff()
        
        return signals

class RSIStrategy(TradingStrategy):
    """RSI-based mean reversion strategy"""
    
    def __init__(self, rsi_period: int = 14, oversold: float = 30, overbought: float = 70):
        super().__init__("RSI_Strategy", {
            "rsi_period": rsi_period, 
            "oversold": oversold, 
            "overbought": overbought
        })
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate RSI signals"""
        signals = data.copy()
        signals['RSI'] = ta.momentum.RSIIndicator(
            close=signals['Close'], 
            window=self.rsi_period
        ).rsi()
        
        # Generate signals
        signals['signal'] = 0
        signals.loc[signals['RSI'] < self.oversold, 'signal'] = 1  # Buy signal
        signals.loc[signals['RSI'] > self.overbought, 'signal'] = -1  # Sell signal
        
        return signals

class MACDStrategy(TradingStrategy):
    """MACD trend following strategy"""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__("MACD_Strategy", {
            "fast_period": fast_period,
            "slow_period": slow_period, 
            "signal_period": signal_period
        })
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate MACD signals"""
        signals = data.copy()
        macd_indicator = ta.trend.MACD(
            close=signals['Close'],
            window_fast=self.fast_period,
            window_slow=self.slow_period,
            window_sign=self.signal_period
        )
        
        signals['MACD'] = macd_indicator.macd()
        signals['MACD_signal'] = macd_indicator.macd_signal()
        signals['MACD_hist'] = macd_indicator.macd_diff()
        
        # Generate trading signals
        signals['signal'] = 0
        signals.loc[(signals['MACD'] > signals['MACD_signal']) & 
                   (signals['MACD'].shift(1) <= signals['MACD_signal'].shift(1)), 'signal'] = 1
        signals.loc[(signals['MACD'] < signals['MACD_signal']) & 
                   (signals['MACD'].shift(1) >= signals['MACD_signal'].shift(1)), 'signal'] = -1
        
        return signals

class QNTIBacktestingEngine:
    """Advanced backtesting engine for QNTI trading strategies"""
    
    def __init__(self, data_source: str = "yfinance"):
        self.data_source = data_source
        self.strategies = {}
        self.results_cache = {}
        self.data_cache = {}
        
        # Backtesting parameters
        self.commission = 0.001  # 0.1% commission
        self.slippage = 0.0005   # 0.05% slippage
        self.initial_capital = 10000.0
        self.risk_per_trade = 0.02  # 2% risk per trade
        
        # Register built-in strategies
        self._register_builtin_strategies()
        
        logger.info("QNTI Backtesting Engine initialized")
    
    def _register_builtin_strategies(self):
        """Register built-in trading strategies"""
        self.strategies['ma_cross'] = MovingAverageCrossStrategy
        self.strategies['rsi'] = RSIStrategy
        self.strategies['macd'] = MACDStrategy
    
    def register_strategy(self, name: str, strategy_class: type):
        """Register a custom trading strategy"""
        self.strategies[name] = strategy_class
        logger.info(f"Registered strategy: {name}")
    
    def get_market_data(self, symbol: str, start_date: datetime, end_date: datetime, 
                       timeframe: str = "1d") -> pd.DataFrame:
        """Get market data for backtesting"""
        cache_key = f"{symbol}_{start_date}_{end_date}_{timeframe}"
        
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        try:
            if self.data_source == "yfinance":
                ticker = yf.Ticker(symbol)
                data = ticker.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval=timeframe
                )
                
                # Standardize column names
                if not data.empty:
                    data = data.rename(columns={
                        'Open': 'Open',
                        'High': 'High', 
                        'Low': 'Low',
                        'Close': 'Close',
                        'Volume': 'Volume'
                    })
                    
                    # Cache the data
                    self.data_cache[cache_key] = data
                    return data
                else:
                    logger.warning(f"No data found for {symbol}")
                    return pd.DataFrame()
            
            else:
                # Placeholder for other data sources
                logger.error(f"Data source {self.data_source} not implemented")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def run_backtest(self, strategy_name: str, symbol: str, start_date: datetime, 
                    end_date: datetime, strategy_params: Dict = None,
                    initial_capital: float = None) -> BacktestResult:
        """Run a backtest for a given strategy"""
        try:
            # Get strategy class
            if strategy_name not in self.strategies:
                raise ValueError(f"Strategy {strategy_name} not registered")
            
            strategy_class = self.strategies[strategy_name]
            
            # Initialize strategy with parameters
            if strategy_params:
                strategy = strategy_class(**strategy_params)
            else:
                strategy = strategy_class()
            
            # Get market data
            data = self.get_market_data(symbol, start_date, end_date)
            if data.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # Generate trading signals
            signals = strategy.generate_signals(data)
            
            # Run simulation
            capital = initial_capital or self.initial_capital
            return self._simulate_trading(strategy, signals, symbol, capital, start_date, end_date)
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            raise
    
    def _simulate_trading(self, strategy: TradingStrategy, signals: pd.DataFrame, 
                         symbol: str, initial_capital: float, start_date: datetime, 
                         end_date: datetime) -> BacktestResult:
        """Simulate trading based on signals"""
        trades = []
        equity_curve = []
        capital = initial_capital
        position = None
        trade_id = 0
        
        for i, (timestamp, row) in enumerate(signals.iterrows()):
            current_price = row['Close']
            equity_curve.append((timestamp, capital))
            
            # Check for buy signal
            if 'signal' in row and row['signal'] == 1 and position is None:
                # Open long position
                quantity = strategy.calculate_position_size(capital, current_price, self.risk_per_trade)
                
                position = BacktestOrder(
                    id=f"trade_{trade_id}",
                    symbol=symbol,
                    order_type=OrderType.BUY,
                    quantity=quantity,
                    entry_price=current_price * (1 + self.slippage),  # Apply slippage
                    entry_time=timestamp,
                    commission=current_price * quantity * self.commission
                )
                
                capital -= position.entry_price * position.quantity + position.commission
                trade_id += 1
            
            # Check for sell signal or exit conditions
            elif 'signal' in row and row['signal'] == -1 and position is not None:
                # Close position
                exit_price = current_price * (1 - self.slippage)  # Apply slippage
                position.exit_price = exit_price
                position.exit_time = timestamp
                position.is_open = False
                position.holding_period = timestamp - position.entry_time
                
                # Calculate P&L
                if position.order_type == OrderType.BUY:
                    position.profit_loss = (exit_price - position.entry_price) * position.quantity
                else:
                    position.profit_loss = (position.entry_price - exit_price) * position.quantity
                
                position.profit_loss -= position.commission * 2  # Entry and exit commission
                
                capital += exit_price * position.quantity - position.commission
                trades.append(position)
                position = None
        
        # Close any remaining open positions
        if position is not None:
            final_price = signals.iloc[-1]['Close']
            position.exit_price = final_price * (1 - self.slippage)
            position.exit_time = signals.index[-1]
            position.is_open = False
            position.holding_period = position.exit_time - position.entry_time
            
            if position.order_type == OrderType.BUY:
                position.profit_loss = (position.exit_price - position.entry_price) * position.quantity
            else:
                position.profit_loss = (position.entry_price - position.exit_price) * position.quantity
            
            position.profit_loss -= position.commission * 2
            capital += position.exit_price * position.quantity - position.commission
            trades.append(position)
        
        # Calculate performance metrics
        final_capital = capital
        performance_metrics = self._calculate_performance_metrics(
            trades, equity_curve, initial_capital, final_capital
        )
        
        return BacktestResult(
            strategy_name=strategy.name,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_return=(final_capital - initial_capital) / initial_capital,
            total_trades=len(trades),
            winning_trades=len([t for t in trades if t.profit_loss > 0]),
            losing_trades=len([t for t in trades if t.profit_loss < 0]),
            win_rate=len([t for t in trades if t.profit_loss > 0]) / len(trades) if trades else 0,
            avg_win=np.mean([t.profit_loss for t in trades if t.profit_loss > 0]) if trades else 0,
            avg_loss=np.mean([t.profit_loss for t in trades if t.profit_loss < 0]) if trades else 0,
            max_drawdown=performance_metrics['max_drawdown'],
            sharpe_ratio=performance_metrics['sharpe_ratio'],
            sortino_ratio=performance_metrics['sortino_ratio'],
            calmar_ratio=performance_metrics['calmar_ratio'],
            profit_factor=performance_metrics['profit_factor'],
            trades=trades,
            equity_curve=equity_curve,
            daily_returns=performance_metrics['daily_returns'],
            performance_metrics=performance_metrics
        )
    
    def _calculate_performance_metrics(self, trades: List[BacktestOrder], 
                                     equity_curve: List[Tuple[datetime, float]], 
                                     initial_capital: float, final_capital: float) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        metrics = {}
        
        if not trades:
            return {
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0,
                'profit_factor': 0.0,
                'daily_returns': []
            }
        
        # Convert equity curve to pandas series
        equity_df = pd.DataFrame(equity_curve, columns=['Date', 'Equity'])
        equity_df.set_index('Date', inplace=True)
        
        # Calculate daily returns
        equity_df['Daily_Return'] = equity_df['Equity'].pct_change().fillna(0)
        daily_returns = equity_df['Daily_Return'].tolist()
        metrics['daily_returns'] = daily_returns
        
        # Maximum Drawdown
        rolling_max = equity_df['Equity'].expanding().max()
        drawdown = (equity_df['Equity'] - rolling_max) / rolling_max
        metrics['max_drawdown'] = abs(drawdown.min())
        
        # Sharpe Ratio (assuming risk-free rate = 0)
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            metrics['sharpe_ratio'] = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        else:
            metrics['sharpe_ratio'] = 0.0
        
        # Sortino Ratio
        negative_returns = [r for r in daily_returns if r < 0]
        if negative_returns and np.std(negative_returns) > 0:
            metrics['sortino_ratio'] = np.mean(daily_returns) / np.std(negative_returns) * np.sqrt(252)
        else:
            metrics['sortino_ratio'] = 0.0
        
        # Calmar Ratio
        annual_return = (final_capital / initial_capital) ** (252 / len(equity_curve)) - 1
        if metrics['max_drawdown'] > 0:
            metrics['calmar_ratio'] = annual_return / metrics['max_drawdown']
        else:
            metrics['calmar_ratio'] = 0.0
        
        # Profit Factor
        gross_profit = sum(t.profit_loss for t in trades if t.profit_loss > 0)
        gross_loss = abs(sum(t.profit_loss for t in trades if t.profit_loss < 0))
        if gross_loss > 0:
            metrics['profit_factor'] = gross_profit / gross_loss
        else:
            metrics['profit_factor'] = float('inf') if gross_profit > 0 else 0.0
        
        return metrics
    
    def run_strategy_comparison(self, strategies: List[str], symbol: str, 
                              start_date: datetime, end_date: datetime,
                              strategy_params: Dict = None) -> Dict[str, BacktestResult]:
        """Compare multiple strategies on the same data"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            
            for strategy_name in strategies:
                params = strategy_params.get(strategy_name, {}) if strategy_params else {}
                future = executor.submit(
                    self.run_backtest, strategy_name, symbol, start_date, end_date, params
                )
                futures[future] = strategy_name
            
            for future in as_completed(futures):
                strategy_name = futures[future]
                try:
                    results[strategy_name] = future.result()
                    logger.info(f"Completed backtest for {strategy_name}")
                except Exception as e:
                    logger.error(f"Error running backtest for {strategy_name}: {e}")
        
        return results
    
    def generate_report(self, result: BacktestResult, output_dir: str = "backtest_reports") -> str:
        """Generate comprehensive backtest report"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Create report filename
            report_file = output_path / f"{result.strategy_name}_{result.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            
            # Generate HTML report
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>QNTI Backtest Report - {result.strategy_name}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background: #1e293b; color: white; padding: 20px; border-radius: 8px; }}
                    .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f1f5f9; border-radius: 4px; }}
                    .positive {{ color: #059669; }}
                    .negative {{ color: #dc2626; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>QNTI Backtest Report</h1>
                    <h2>{result.strategy_name} - {result.symbol}</h2>
                    <p>Period: {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}</p>
                </div>
                
                <h2>Performance Summary</h2>
                <div class="metric">Initial Capital: ${result.initial_capital:,.2f}</div>
                <div class="metric">Final Capital: ${result.final_capital:,.2f}</div>
                <div class="metric">Total Return: <span class="{'positive' if result.total_return > 0 else 'negative'}">{result.total_return:.2%}</span></div>
                <div class="metric">Total Trades: {result.total_trades}</div>
                <div class="metric">Win Rate: {result.win_rate:.2%}</div>
                <div class="metric">Max Drawdown: <span class="negative">{result.max_drawdown:.2%}</span></div>
                <div class="metric">Sharpe Ratio: {result.sharpe_ratio:.2f}</div>
                <div class="metric">Sortino Ratio: {result.sortino_ratio:.2f}</div>
                <div class="metric">Profit Factor: {result.profit_factor:.2f}</div>
                
                <h2>Trade History</h2>
                <table>
                    <tr>
                        <th>Trade ID</th>
                        <th>Entry Time</th>
                        <th>Exit Time</th>
                        <th>Type</th>
                        <th>Entry Price</th>
                        <th>Exit Price</th>
                        <th>Quantity</th>
                        <th>P&L</th>
                        <th>Holding Period</th>
                    </tr>
            """
            
            for trade in result.trades:
                pnl_class = "positive" if trade.profit_loss > 0 else "negative"
                html_content += f"""
                    <tr>
                        <td>{trade.id}</td>
                        <td>{trade.entry_time.strftime('%Y-%m-%d %H:%M')}</td>
                        <td>{trade.exit_time.strftime('%Y-%m-%d %H:%M') if trade.exit_time else 'Open'}</td>
                        <td>{trade.order_type.value.upper()}</td>
                        <td>${trade.entry_price:.2f}</td>
                        <td>${trade.exit_price:.2f if trade.exit_price else 0:.2f}</td>
                        <td>{trade.quantity:.2f}</td>
                        <td><span class="{pnl_class}">${trade.profit_loss:.2f}</span></td>
                        <td>{str(trade.holding_period).split('.')[0] if trade.holding_period else 'N/A'}</td>
                    </tr>
                """
            
            html_content += """
                </table>
                
                <h2>Additional Metrics</h2>
                <ul>
            """
            
            for key, value in result.performance_metrics.items():
                if key != 'daily_returns':
                    html_content += f"<li>{key.replace('_', ' ').title()}: {value}</li>"
            
            html_content += """
                </ul>
                
                <footer>
                    <p>Report generated by QNTI Backtesting Engine on {}</p>
                </footer>
            </body>
            </html>
            """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            
            # Write report to file
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Backtest report generated: {report_file}")
            return str(report_file)
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise
    
    def save_results(self, result: BacktestResult, filename: str = None):
        """Save backtest results to JSON file"""
        try:
            if not filename:
                filename = f"backtest_{result.strategy_name}_{result.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Convert result to dict (handling non-serializable types)
            result_dict = asdict(result)
            
            # Convert datetime objects to strings
            result_dict['start_date'] = result.start_date.isoformat()
            result_dict['end_date'] = result.end_date.isoformat()
            
            # Convert trades
            for trade in result_dict['trades']:
                trade['entry_time'] = trade['entry_time'].isoformat()
                if trade['exit_time']:
                    trade['exit_time'] = trade['exit_time'].isoformat()
                if trade['holding_period']:
                    trade['holding_period'] = str(trade['holding_period'])
                trade['order_type'] = trade['order_type'].value
            
            # Convert equity curve
            result_dict['equity_curve'] = [
                (timestamp.isoformat(), value) for timestamp, value in result.equity_curve
            ]
            
            with open(filename, 'w') as f:
                json.dump(result_dict, f, indent=2)
            
            logger.info(f"Backtest results saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise

# Example usage and testing
if __name__ == "__main__":
    # Create backtesting engine
    backtest_engine = QNTIBacktestingEngine()
    
    # Test with multiple strategies
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 1, 1)
    symbol = "AAPL"
    
    try:
        # Run individual backtest
        result = backtest_engine.run_backtest(
            strategy_name="ma_cross",
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            strategy_params={"short_window": 10, "long_window": 30}
        )
        
        print(f"\nBacktest Results for {result.strategy_name}:")
        print(f"Symbol: {result.symbol}")
        print(f"Period: {result.start_date} to {result.end_date}")
        print(f"Total Return: {result.total_return:.2%}")
        print(f"Total Trades: {result.total_trades}")
        print(f"Win Rate: {result.win_rate:.2%}")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {result.max_drawdown:.2%}")
        
        # Generate report
        report_path = backtest_engine.generate_report(result)
        print(f"Report generated: {report_path}")
        
        # Save results
        backtest_engine.save_results(result)
        
    except Exception as e:
        print(f"Error running backtest: {e}")