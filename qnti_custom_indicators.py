#!/usr/bin/env python3
"""
QNTI Custom Indicators System
Advanced indicator builder and manager with real-time calculations and visual components
"""

import asyncio
import json
import logging
import math
import numpy as np
import pandas as pd
import talib
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
from uuid import uuid4

logger = logging.getLogger('QNTI_CUSTOM_INDICATORS')

class IndicatorType(Enum):
    """Indicator types"""
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    CUSTOM = "custom"
    COMPOSITE = "composite"

class DataType(Enum):
    """Data types for indicator inputs"""
    PRICE = "price"
    VOLUME = "volume"
    INDICATOR = "indicator"
    CUSTOM = "custom"

class VisualizationType(Enum):
    """Visualization types"""
    LINE = "line"
    HISTOGRAM = "histogram"
    AREA = "area"
    CANDLESTICK = "candlestick"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    OVERLAY = "overlay"
    SUBPLOT = "subplot"

class SignalType(Enum):
    """Signal types"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    NEUTRAL = "neutral"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"

@dataclass
class IndicatorParameter:
    """Parameter definition for indicators"""
    name: str
    type: str  # int, float, string, bool
    default_value: Any
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    description: str = ""
    options: Optional[List] = None  # For dropdown/select parameters

@dataclass
class IndicatorInput:
    """Input definition for indicators"""
    name: str
    data_type: DataType
    source: str  # e.g., "close", "high", "volume", "indicator_id"
    description: str = ""
    required: bool = True

@dataclass
class IndicatorOutput:
    """Output definition for indicators"""
    name: str
    display_name: str
    visualization: VisualizationType
    color: str = "#3498db"
    style: Dict = None
    
    def __post_init__(self):
        if self.style is None:
            self.style = {}

@dataclass
class SignalRule:
    """Signal generation rule"""
    id: str
    name: str
    condition: str  # Python expression
    signal_type: SignalType
    confidence: float = 0.5
    description: str = ""
    
    # Alert settings
    alert_enabled: bool = False
    alert_message: str = ""

@dataclass
class IndicatorDefinition:
    """Custom indicator definition"""
    id: str
    name: str
    display_name: str
    indicator_type: IndicatorType
    description: str
    
    # Code and logic
    calculation_code: str
    initialization_code: str = ""
    
    # Parameters, inputs, and outputs
    parameters: List[IndicatorParameter] = None
    inputs: List[IndicatorInput] = None
    outputs: List[IndicatorOutput] = None
    
    # Signal generation
    signal_rules: List[SignalRule] = None
    
    # Metadata
    author: str = ""
    version: str = "1.0"
    tags: List[str] = None
    
    # Performance
    min_periods: int = 1
    lookback_period: int = 100
    
    # Timestamps
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.parameters is None:
            self.parameters = []
        if self.inputs is None:
            self.inputs = []
        if self.outputs is None:
            self.outputs = []
        if self.signal_rules is None:
            self.signal_rules = []
        if self.tags is None:
            self.tags = []

@dataclass
class IndicatorInstance:
    """Instance of an indicator with specific parameters"""
    id: str
    definition_id: str
    name: str
    symbol: str
    timeframe: str
    
    # Parameter values
    parameter_values: Dict = None
    
    # Runtime data
    data: Dict = None
    last_update: datetime = None
    
    # Status
    is_active: bool = True
    error_message: str = ""
    
    # Timestamps
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.parameter_values is None:
            self.parameter_values = {}
        if self.data is None:
            self.data = {}

@dataclass
class IndicatorValue:
    """Single indicator value with timestamp"""
    timestamp: datetime
    values: Dict[str, float]  # output_name -> value
    signals: List[Dict] = None
    
    def __post_init__(self):
        if self.signals is None:
            self.signals = []

@dataclass
class IndicatorTemplate:
    """Pre-built indicator template"""
    id: str
    name: str
    category: str
    description: str
    definition: IndicatorDefinition
    
    # Usage statistics
    usage_count: int = 0
    rating: float = 0.0
    
    # Template metadata
    is_featured: bool = False
    is_verified: bool = False
    
    def __post_init__(self):
        """Post-initialization for IndicatorTemplate"""
        try:
            # Validate template ID format
            if not self.id or not isinstance(self.id, str):
                raise ValueError("Template ID must be a non-empty string")
            
            # Ensure name is not empty
            if not self.name or not self.name.strip():
                raise ValueError("Template name cannot be empty")
            
            # Validate category
            valid_categories = ['trend', 'momentum', 'volatility', 'volume', 'oscillator', 'custom']
            if self.category not in valid_categories:
                logger.warning(f"Template {self.id}: Unknown category '{self.category}', defaulting to 'custom'")
                self.category = 'custom'
            
            # Validate definition
            if not isinstance(self.definition, IndicatorDefinition):
                raise ValueError("Template definition must be an IndicatorDefinition instance")
            
            # Initialize rating bounds
            if self.rating < 0.0:
                self.rating = 0.0
            elif self.rating > 5.0:
                self.rating = 5.0
            
            # Initialize usage count
            if self.usage_count < 0:
                self.usage_count = 0
            
            # Set verification status based on rating and usage
            if self.rating >= 4.0 and self.usage_count >= 100:
                self.is_verified = True
            
            # Set featured status for highly rated templates
            if self.rating >= 4.5 and self.usage_count >= 50:
                self.is_featured = True
            
            # Validate and complete the indicator definition
            self._validate_definition()
            
            logger.debug(f"Template {self.id} initialized successfully")
            
        except Exception as e:
            logger.error(f"Error in IndicatorTemplate post-init: {e}")
            raise
    
    def _validate_definition(self):
        """Validate the indicator definition"""
        try:
            # Check if definition has required components
            if not self.definition.formula:
                raise ValueError(f"Template {self.id}: Formula cannot be empty")
            
            # Validate parameters
            if not self.definition.parameters:
                logger.warning(f"Template {self.id}: No parameters defined")
            
            # Validate outputs
            if not self.definition.outputs:
                raise ValueError(f"Template {self.id}: At least one output must be defined")
            
            # Check formula syntax (basic validation)
            formula = self.definition.formula.strip()
            if not formula:
                raise ValueError(f"Template {self.id}: Formula is empty")
            
            # Ensure all parameter references in formula exist
            for param in self.definition.parameters:
                param_ref = f"${param.name}"
                if param_ref not in formula and param.required:
                    logger.warning(f"Template {self.id}: Required parameter '{param.name}' not used in formula")
            
        except Exception as e:
            logger.error(f"Error validating template definition: {e}")
            raise

class QNTICustomIndicators:
    """QNTI Custom Indicators System"""
    
    def __init__(self, trade_manager, mt5_bridge=None):
        self.trade_manager = trade_manager
        self.mt5_bridge = mt5_bridge
        
        # Data storage
        self.indicator_definitions: Dict[str, IndicatorDefinition] = {}
        self.indicator_instances: Dict[str, IndicatorInstance] = {}
        self.indicator_templates: Dict[str, IndicatorTemplate] = {}
        self.indicator_values: Dict[str, List[IndicatorValue]] = {}
        
        # Runtime management
        self.calculation_threads: Dict[str, threading.Thread] = {}
        self.active_calculations: Dict[str, bool] = {}
        
        # Market data cache
        self.market_data_cache: Dict[str, pd.DataFrame] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        self.cache_duration = 300  # 5 minutes
        
        # Built-in indicators
        self.builtin_indicators = {}
        
        # Event callbacks
        self.callbacks: Dict[str, List[Callable]] = {
            'value_updated': [],
            'signal_generated': [],
            'error_occurred': [],
            'indicator_created': [],
            'indicator_updated': []
        }
        
        # Performance tracking
        self.performance_stats = {
            'total_indicators': 0,
            'active_indicators': 0,
            'total_calculations': 0,
            'calculation_errors': 0,
            'average_calculation_time': 0.0
        }
        
        # Load existing data
        self._load_data()
        
        # Initialize built-in indicators
        self._initialize_builtin_indicators()
        
        # Load templates
        self._load_templates()
        
        logger.info("Custom Indicators System initialized")
    
    def _load_data(self):
        """Load existing indicator data"""
        try:
            import os
            
            if os.path.exists('custom_indicators_data.json'):
                with open('custom_indicators_data.json', 'r') as f:
                    data = json.load(f)
                    
                    # Load indicator definitions
                    for def_data in data.get('indicator_definitions', []):
                        definition = IndicatorDefinition(**def_data)
                        definition.created_at = datetime.fromisoformat(def_data['created_at'])
                        definition.updated_at = datetime.fromisoformat(def_data['updated_at'])
                        self.indicator_definitions[definition.id] = definition
                    
                    # Load indicator instances
                    for inst_data in data.get('indicator_instances', []):
                        instance = IndicatorInstance(**inst_data)
                        instance.created_at = datetime.fromisoformat(inst_data['created_at'])
                        instance.updated_at = datetime.fromisoformat(inst_data['updated_at'])
                        if inst_data.get('last_update'):
                            instance.last_update = datetime.fromisoformat(inst_data['last_update'])
                        self.indicator_instances[instance.id] = instance
                        
                logger.info("Custom indicators data loaded successfully")
                
        except Exception as e:
            logger.error(f"Error loading custom indicators data: {e}")
    
    def _save_data(self):
        """Save indicator data"""
        try:
            data = {
                'indicator_definitions': [],
                'indicator_instances': []
            }
            
            # Save indicator definitions
            for definition in self.indicator_definitions.values():
                def_data = asdict(definition)
                def_data['created_at'] = definition.created_at.isoformat()
                def_data['updated_at'] = definition.updated_at.isoformat()
                data['indicator_definitions'].append(def_data)
            
            # Save indicator instances
            for instance in self.indicator_instances.values():
                inst_data = asdict(instance)
                inst_data['created_at'] = instance.created_at.isoformat()
                inst_data['updated_at'] = instance.updated_at.isoformat()
                if instance.last_update:
                    inst_data['last_update'] = instance.last_update.isoformat()
                data['indicator_instances'].append(inst_data)
            
            with open('custom_indicators_data.json', 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving custom indicators data: {e}")
    
    def _initialize_builtin_indicators(self):
        """Initialize built-in indicators"""
        try:
            # Simple Moving Average
            sma_definition = IndicatorDefinition(
                id="builtin_sma",
                name="Simple Moving Average",
                display_name="SMA",
                indicator_type=IndicatorType.TREND,
                description="Simple Moving Average calculated over a specified period",
                calculation_code="""
def calculate(data, period=20):
    close_prices = data['close']
    if len(close_prices) < period:
        return {'sma': np.nan}
    
    sma = close_prices.rolling(window=period).mean()
    return {'sma': sma.iloc[-1]}
""",
                parameters=[
                    IndicatorParameter("period", "int", 20, 1, 1000, "Period for moving average")
                ],
                inputs=[
                    IndicatorInput("close", DataType.PRICE, "close", "Close price")
                ],
                outputs=[
                    IndicatorOutput("sma", "SMA", VisualizationType.LINE, "#3498db")
                ]
            )
            self.builtin_indicators['sma'] = sma_definition
            
            # Relative Strength Index
            rsi_definition = IndicatorDefinition(
                id="builtin_rsi",
                name="Relative Strength Index",
                display_name="RSI",
                indicator_type=IndicatorType.MOMENTUM,
                description="Relative Strength Index momentum oscillator",
                calculation_code="""
def calculate(data, period=14):
    close_prices = data['close']
    if len(close_prices) < period + 1:
        return {'rsi': np.nan}
    
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return {'rsi': rsi.iloc[-1]}
""",
                parameters=[
                    IndicatorParameter("period", "int", 14, 1, 100, "Period for RSI calculation")
                ],
                inputs=[
                    IndicatorInput("close", DataType.PRICE, "close", "Close price")
                ],
                outputs=[
                    IndicatorOutput("rsi", "RSI", VisualizationType.LINE, "#e74c3c")
                ],
                signal_rules=[
                    SignalRule("rsi_oversold", "RSI Oversold", "rsi < 30", SignalType.BUY, 0.7),
                    SignalRule("rsi_overbought", "RSI Overbought", "rsi > 70", SignalType.SELL, 0.7)
                ]
            )
            self.builtin_indicators['rsi'] = rsi_definition
            
            # MACD
            macd_definition = IndicatorDefinition(
                id="builtin_macd",
                name="Moving Average Convergence Divergence",
                display_name="MACD",
                indicator_type=IndicatorType.MOMENTUM,
                description="MACD with signal line and histogram",
                calculation_code="""
def calculate(data, fast_period=12, slow_period=26, signal_period=9):
    close_prices = data['close']
    if len(close_prices) < slow_period + signal_period:
        return {'macd': np.nan, 'signal': np.nan, 'histogram': np.nan}
    
    ema_fast = close_prices.ewm(span=fast_period).mean()
    ema_slow = close_prices.ewm(span=slow_period).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period).mean()
    histogram = macd - signal
    
    return {
        'macd': macd.iloc[-1],
        'signal': signal.iloc[-1],
        'histogram': histogram.iloc[-1]
    }
""",
                parameters=[
                    IndicatorParameter("fast_period", "int", 12, 1, 50, "Fast EMA period"),
                    IndicatorParameter("slow_period", "int", 26, 1, 100, "Slow EMA period"),
                    IndicatorParameter("signal_period", "int", 9, 1, 50, "Signal line period")
                ],
                inputs=[
                    IndicatorInput("close", DataType.PRICE, "close", "Close price")
                ],
                outputs=[
                    IndicatorOutput("macd", "MACD", VisualizationType.LINE, "#3498db"),
                    IndicatorOutput("signal", "Signal", VisualizationType.LINE, "#e74c3c"),
                    IndicatorOutput("histogram", "Histogram", VisualizationType.HISTOGRAM, "#95a5a6")
                ]
            )
            self.builtin_indicators['macd'] = macd_definition
            
            # Bollinger Bands
            bb_definition = IndicatorDefinition(
                id="builtin_bollinger",
                name="Bollinger Bands",
                display_name="BB",
                indicator_type=IndicatorType.VOLATILITY,
                description="Bollinger Bands with upper, middle, and lower bands",
                calculation_code="""
def calculate(data, period=20, std_dev=2):
    close_prices = data['close']
    if len(close_prices) < period:
        return {'upper': np.nan, 'middle': np.nan, 'lower': np.nan}
    
    middle = close_prices.rolling(window=period).mean()
    std = close_prices.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    
    return {
        'upper': upper.iloc[-1],
        'middle': middle.iloc[-1],
        'lower': lower.iloc[-1]
    }
""",
                parameters=[
                    IndicatorParameter("period", "int", 20, 1, 100, "Period for moving average"),
                    IndicatorParameter("std_dev", "float", 2.0, 0.1, 5.0, "Standard deviation multiplier")
                ],
                inputs=[
                    IndicatorInput("close", DataType.PRICE, "close", "Close price")
                ],
                outputs=[
                    IndicatorOutput("upper", "Upper Band", VisualizationType.LINE, "#e74c3c"),
                    IndicatorOutput("middle", "Middle Band", VisualizationType.LINE, "#3498db"),
                    IndicatorOutput("lower", "Lower Band", VisualizationType.LINE, "#27ae60")
                ]
            )
            self.builtin_indicators['bollinger'] = bb_definition
            
            # Add built-in indicators to definitions
            for indicator in self.builtin_indicators.values():
                self.indicator_definitions[indicator.id] = indicator
                
        except Exception as e:
            logger.error(f"Error initializing built-in indicators: {e}")
    
    def _load_templates(self):
        """Load indicator templates"""
        try:
            # Create templates from built-in indicators
            for indicator_id, definition in self.builtin_indicators.items():
                template = IndicatorTemplate(
                    id=f"template_{indicator_id}",
                    name=definition.display_name,
                    category=definition.indicator_type.value,
                    description=definition.description,
                    definition=definition,
                    is_featured=True,
                    is_verified=True
                )
                self.indicator_templates[template.id] = template
                
        except Exception as e:
            logger.error(f"Error loading templates: {e}")
    
    def _get_market_data(self, symbol: str, timeframe: str, lookback: int = 1000) -> pd.DataFrame:
        """Get market data for calculations"""
        try:
            cache_key = f"{symbol}_{timeframe}"
            current_time = datetime.now()
            
            # Check cache
            if (cache_key in self.market_data_cache and 
                cache_key in self.cache_timestamps and
                (current_time - self.cache_timestamps[cache_key]).total_seconds() < self.cache_duration):
                return self.market_data_cache[cache_key]
            
            # Get data from MT5 bridge
            if self.mt5_bridge:
                try:
                    # Convert timeframe to MT5 format
                    mt5_timeframe = self._convert_timeframe(timeframe)
                    
                    # Get OHLCV data
                    rates = self.mt5_bridge.get_rates(symbol, mt5_timeframe, lookback)
                    
                    if rates is not None and len(rates) > 0:
                        # Convert to pandas DataFrame
                        df = pd.DataFrame(rates)
                        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
                        df.set_index('timestamp', inplace=True)
                        
                        # Cache the data
                        self.market_data_cache[cache_key] = df
                        self.cache_timestamps[cache_key] = current_time
                        
                        return df
                except Exception as e:
                    logger.error(f"Error getting MT5 data: {e}")
            
            # Fallback to demo data
            return self._generate_demo_data(symbol, lookback)
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return pd.DataFrame()
    
    def _convert_timeframe(self, timeframe: str) -> int:
        """Convert timeframe string to MT5 timeframe constant"""
        timeframe_map = {
            'M1': 1,
            'M5': 5,
            'M15': 15,
            'M30': 30,
            'H1': 60,
            'H4': 240,
            'D1': 1440,
            'W1': 10080,
            'MN1': 43200
        }
        return timeframe_map.get(timeframe, 60)  # Default to H1
    
    def _generate_demo_data(self, symbol: str, periods: int) -> pd.DataFrame:
        """Generate demo market data for testing"""
        try:
            # Generate realistic price data
            np.random.seed(42)  # For reproducible demo data
            
            # Starting values
            open_price = 1.1000
            dates = pd.date_range(end=datetime.now(), periods=periods, freq='H')
            
            # Generate price movements
            returns = np.random.normal(0, 0.001, periods)  # Small random returns
            prices = [open_price]
            
            for i in range(1, periods):
                prices.append(prices[-1] * (1 + returns[i]))
            
            # Generate OHLCV data
            data = []
            for i in range(periods):
                base_price = prices[i]
                high = base_price * (1 + abs(np.random.normal(0, 0.0005)))
                low = base_price * (1 - abs(np.random.normal(0, 0.0005)))
                close = base_price * (1 + np.random.normal(0, 0.0002))
                volume = np.random.randint(1000, 10000)
                
                data.append({
                    'open': base_price,
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': volume
                })
            
            df = pd.DataFrame(data, index=dates)
            return df
            
        except Exception as e:
            logger.error(f"Error generating demo data: {e}")
            return pd.DataFrame()
    
    def _calculate_indicator(self, instance: IndicatorInstance) -> Optional[IndicatorValue]:
        """Calculate indicator value for an instance"""
        try:
            start_time = time.time()
            
            # Get indicator definition
            definition = self.indicator_definitions.get(instance.definition_id)
            if not definition:
                raise ValueError(f"Indicator definition not found: {instance.definition_id}")
            
            # Get market data
            market_data = self._get_market_data(instance.symbol, instance.timeframe, definition.lookback_period)
            
            if market_data.empty:
                raise ValueError("No market data available")
            
            # Prepare calculation environment
            calc_env = {
                'np': np,
                'pd': pd,
                'talib': talib,
                'math': math,
                'data': market_data,
                **instance.parameter_values
            }
            
            # Execute initialization code if present
            if definition.initialization_code:
                exec(definition.initialization_code, calc_env)
            
            # Execute calculation code
            exec(definition.calculation_code, calc_env)
            
            # Get the calculate function
            if 'calculate' not in calc_env:
                raise ValueError("No calculate function found in indicator code")
            
            calculate_func = calc_env['calculate']
            
            # Call the calculate function
            result = calculate_func(market_data, **instance.parameter_values)
            
            if not isinstance(result, dict):
                raise ValueError("Calculate function must return a dictionary")
            
            # Create indicator value
            indicator_value = IndicatorValue(
                timestamp=datetime.now(),
                values=result
            )
            
            # Check for signals
            signals = self._check_signals(definition, result, calc_env)
            indicator_value.signals = signals
            
            # Update performance stats
            calculation_time = time.time() - start_time
            self.performance_stats['total_calculations'] += 1
            self._update_average_calculation_time(calculation_time)
            
            return indicator_value
            
        except Exception as e:
            logger.error(f"Error calculating indicator {instance.id}: {e}")
            self.performance_stats['calculation_errors'] += 1
            
            # Update instance error status
            instance.error_message = str(e)
            instance.is_active = False
            
            return None
    
    def _check_signals(self, definition: IndicatorDefinition, values: Dict, calc_env: Dict) -> List[Dict]:
        """Check signal rules for an indicator"""
        try:
            signals = []
            
            for rule in definition.signal_rules:
                try:
                    # Prepare signal environment
                    signal_env = {**calc_env, **values}
                    
                    # Evaluate condition
                    condition_result = eval(rule.condition, signal_env)
                    
                    if condition_result:
                        signal = {
                            'rule_id': rule.id,
                            'name': rule.name,
                            'type': rule.signal_type.value,
                            'confidence': rule.confidence,
                            'description': rule.description,
                            'timestamp': datetime.now().isoformat()
                        }
                        signals.append(signal)
                        
                        # Trigger callback
                        self._trigger_callback('signal_generated', {
                            'signal': signal,
                            'values': values,
                            'definition': definition
                        })
                        
                        # Send alert if enabled
                        if rule.alert_enabled:
                            self._send_alert(rule, signal, values)
                        
                except Exception as e:
                    logger.error(f"Error evaluating signal rule {rule.id}: {e}")
            
            return signals
            
        except Exception as e:
            logger.error(f"Error checking signals: {e}")
            return []
    
    def _send_alert(self, rule: SignalRule, signal: Dict, values: Dict):
        """Send signal alert"""
        try:
            # This would integrate with notification system
            alert_message = rule.alert_message or f"Signal triggered: {rule.name}"
            
            # Format message with values
            formatted_message = alert_message.format(**values)
            
            logger.info(f"Alert: {formatted_message}")
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
    
    def _update_average_calculation_time(self, calculation_time: float):
        """Update average calculation time statistic"""
        try:
            current_avg = self.performance_stats['average_calculation_time']
            total_calculations = self.performance_stats['total_calculations']
            
            if total_calculations == 1:
                self.performance_stats['average_calculation_time'] = calculation_time
            else:
                self.performance_stats['average_calculation_time'] = (
                    (current_avg * (total_calculations - 1) + calculation_time) / total_calculations
                )
        except Exception as e:
            logger.error(f"Error updating average calculation time: {e}")
    
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
    
    def _start_calculation_thread(self, instance_id: str):
        """Start calculation thread for an indicator instance"""
        try:
            if instance_id in self.calculation_threads:
                return  # Already running
            
            def calculation_loop():
                while self.active_calculations.get(instance_id, False):
                    try:
                        instance = self.indicator_instances.get(instance_id)
                        if not instance or not instance.is_active:
                            break
                        
                        # Calculate indicator value
                        indicator_value = self._calculate_indicator(instance)
                        
                        if indicator_value:
                            # Store value
                            if instance_id not in self.indicator_values:
                                self.indicator_values[instance_id] = []
                            
                            self.indicator_values[instance_id].append(indicator_value)
                            
                            # Keep only last 1000 values
                            if len(self.indicator_values[instance_id]) > 1000:
                                self.indicator_values[instance_id] = self.indicator_values[instance_id][-1000:]
                            
                            # Update instance
                            instance.last_update = datetime.now()
                            instance.error_message = ""
                            instance.is_active = True
                            
                            # Trigger callback
                            self._trigger_callback('value_updated', {
                                'instance_id': instance_id,
                                'value': indicator_value
                            })
                        
                        # Wait before next calculation
                        time.sleep(1)  # Calculate every second
                        
                    except Exception as e:
                        logger.error(f"Error in calculation loop for {instance_id}: {e}")
                        time.sleep(5)  # Wait longer on error
            
            # Start thread
            self.active_calculations[instance_id] = True
            thread = threading.Thread(target=calculation_loop, daemon=True)
            thread.start()
            self.calculation_threads[instance_id] = thread
            
        except Exception as e:
            logger.error(f"Error starting calculation thread: {e}")
    
    def _stop_calculation_thread(self, instance_id: str):
        """Stop calculation thread for an indicator instance"""
        try:
            if instance_id in self.active_calculations:
                self.active_calculations[instance_id] = False
            
            if instance_id in self.calculation_threads:
                thread = self.calculation_threads[instance_id]
                if thread.is_alive():
                    thread.join(timeout=1.0)
                del self.calculation_threads[instance_id]
                
        except Exception as e:
            logger.error(f"Error stopping calculation thread: {e}")
    
    # Public API methods
    
    def create_indicator_definition(self, name: str, display_name: str, 
                                  indicator_type: IndicatorType, description: str,
                                  calculation_code: str, **kwargs) -> str:
        """Create a new custom indicator definition"""
        try:
            definition_id = str(uuid4())
            definition = IndicatorDefinition(
                id=definition_id,
                name=name,
                display_name=display_name,
                indicator_type=indicator_type,
                description=description,
                calculation_code=calculation_code,
                **kwargs
            )
            
            # Validate calculation code
            if not self._validate_calculation_code(calculation_code):
                raise ValueError("Invalid calculation code")
            
            self.indicator_definitions[definition_id] = definition
            self.performance_stats['total_indicators'] += 1
            
            self._save_data()
            
            # Trigger callback
            self._trigger_callback('indicator_created', definition)
            
            logger.info(f"Indicator definition created: {definition_id}")
            return definition_id
            
        except Exception as e:
            logger.error(f"Error creating indicator definition: {e}")
            return None
    
    def update_indicator_definition(self, definition_id: str, **kwargs) -> bool:
        """Update an existing indicator definition"""
        try:
            if definition_id not in self.indicator_definitions:
                return False
            
            definition = self.indicator_definitions[definition_id]
            
            # Update fields
            for key, value in kwargs.items():
                if hasattr(definition, key):
                    setattr(definition, key, value)
            
            # Validate calculation code if updated
            if 'calculation_code' in kwargs:
                if not self._validate_calculation_code(kwargs['calculation_code']):
                    raise ValueError("Invalid calculation code")
            
            definition.updated_at = datetime.now()
            
            self._save_data()
            
            # Trigger callback
            self._trigger_callback('indicator_updated', definition)
            
            logger.info(f"Indicator definition updated: {definition_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating indicator definition: {e}")
            return False
    
    def _validate_calculation_code(self, code: str) -> bool:
        """Validate calculation code for security and syntax"""
        try:
            # Basic security checks
            forbidden_keywords = ['import', 'exec', 'eval', 'open', 'file', '__import__']
            code_lower = code.lower()
            
            for keyword in forbidden_keywords:
                if keyword in code_lower:
                    return False
            
            # Check for calculate function
            if 'def calculate(' not in code:
                return False
            
            # Try to compile the code
            compile(code, '<string>', 'exec')
            
            return True
            
        except Exception as e:
            logger.error(f"Code validation error: {e}")
            return False
    
    def create_indicator_instance(self, definition_id: str, name: str, 
                                symbol: str, timeframe: str, **kwargs) -> str:
        """Create a new indicator instance"""
        try:
            if definition_id not in self.indicator_definitions:
                raise ValueError("Indicator definition not found")
            
            instance_id = str(uuid4())
            instance = IndicatorInstance(
                id=instance_id,
                definition_id=definition_id,
                name=name,
                symbol=symbol,
                timeframe=timeframe,
                **kwargs
            )
            
            # Set default parameter values
            definition = self.indicator_definitions[definition_id]
            for param in definition.parameters:
                if param.name not in instance.parameter_values:
                    instance.parameter_values[param.name] = param.default_value
            
            self.indicator_instances[instance_id] = instance
            self.performance_stats['active_indicators'] += 1
            
            # Start calculation thread
            self._start_calculation_thread(instance_id)
            
            self._save_data()
            
            logger.info(f"Indicator instance created: {instance_id}")
            return instance_id
            
        except Exception as e:
            logger.error(f"Error creating indicator instance: {e}")
            return None
    
    def update_indicator_instance(self, instance_id: str, **kwargs) -> bool:
        """Update an indicator instance"""
        try:
            if instance_id not in self.indicator_instances:
                return False
            
            instance = self.indicator_instances[instance_id]
            
            # Update fields
            for key, value in kwargs.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)
            
            instance.updated_at = datetime.now()
            
            self._save_data()
            
            logger.info(f"Indicator instance updated: {instance_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating indicator instance: {e}")
            return False
    
    def delete_indicator_instance(self, instance_id: str) -> bool:
        """Delete an indicator instance"""
        try:
            if instance_id not in self.indicator_instances:
                return False
            
            # Stop calculation thread
            self._stop_calculation_thread(instance_id)
            
            # Remove instance
            del self.indicator_instances[instance_id]
            
            # Remove values
            if instance_id in self.indicator_values:
                del self.indicator_values[instance_id]
            
            self.performance_stats['active_indicators'] -= 1
            
            self._save_data()
            
            logger.info(f"Indicator instance deleted: {instance_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting indicator instance: {e}")
            return False
    
    def get_indicator_definitions(self, indicator_type: IndicatorType = None) -> List[Dict]:
        """Get all indicator definitions"""
        try:
            definitions = list(self.indicator_definitions.values())
            
            # Filter by type if specified
            if indicator_type:
                definitions = [d for d in definitions if d.indicator_type == indicator_type]
            
            # Convert to dict format
            result = []
            for definition in definitions:
                def_dict = asdict(definition)
                def_dict['created_at'] = definition.created_at.isoformat()
                def_dict['updated_at'] = definition.updated_at.isoformat()
                result.append(def_dict)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting indicator definitions: {e}")
            return []
    
    def get_indicator_instances(self, symbol: str = None, timeframe: str = None) -> List[Dict]:
        """Get indicator instances"""
        try:
            instances = list(self.indicator_instances.values())
            
            # Filter by symbol and timeframe if specified
            if symbol:
                instances = [i for i in instances if i.symbol == symbol]
            if timeframe:
                instances = [i for i in instances if i.timeframe == timeframe]
            
            # Convert to dict format
            result = []
            for instance in instances:
                inst_dict = asdict(instance)
                inst_dict['created_at'] = instance.created_at.isoformat()
                inst_dict['updated_at'] = instance.updated_at.isoformat()
                if instance.last_update:
                    inst_dict['last_update'] = instance.last_update.isoformat()
                result.append(inst_dict)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting indicator instances: {e}")
            return []
    
    def get_indicator_values(self, instance_id: str, limit: int = 100) -> List[Dict]:
        """Get indicator values for an instance"""
        try:
            if instance_id not in self.indicator_values:
                return []
            
            values = self.indicator_values[instance_id]
            
            # Get latest values
            latest_values = values[-limit:] if limit > 0 else values
            
            # Convert to dict format
            result = []
            for value in latest_values:
                value_dict = {
                    'timestamp': value.timestamp.isoformat(),
                    'values': value.values,
                    'signals': value.signals
                }
                result.append(value_dict)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting indicator values: {e}")
            return []
    
    def get_indicator_templates(self, category: str = None) -> List[Dict]:
        """Get indicator templates"""
        try:
            templates = list(self.indicator_templates.values())
            
            # Filter by category if specified
            if category:
                templates = [t for t in templates if t.category == category]
            
            # Sort by rating and usage
            templates.sort(key=lambda x: (x.rating, x.usage_count), reverse=True)
            
            # Convert to dict format
            result = []
            for template in templates:
                template_dict = asdict(template)
                # Include definition details
                template_dict['definition'] = asdict(template.definition)
                result.append(template_dict)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting indicator templates: {e}")
            return []
    
    def create_instance_from_template(self, template_id: str, name: str, 
                                    symbol: str, timeframe: str, **kwargs) -> str:
        """Create indicator instance from template"""
        try:
            if template_id not in self.indicator_templates:
                raise ValueError("Template not found")
            
            template = self.indicator_templates[template_id]
            
            # Create instance using template definition
            instance_id = self.create_indicator_instance(
                template.definition.id,
                name,
                symbol,
                timeframe,
                **kwargs
            )
            
            # Update template usage
            template.usage_count += 1
            
            return instance_id
            
        except Exception as e:
            logger.error(f"Error creating instance from template: {e}")
            return None
    
    def calculate_indicator_now(self, instance_id: str) -> Optional[Dict]:
        """Calculate indicator value immediately"""
        try:
            if instance_id not in self.indicator_instances:
                return None
            
            instance = self.indicator_instances[instance_id]
            indicator_value = self._calculate_indicator(instance)
            
            if indicator_value:
                return {
                    'timestamp': indicator_value.timestamp.isoformat(),
                    'values': indicator_value.values,
                    'signals': indicator_value.signals
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error calculating indicator now: {e}")
            return None
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        try:
            stats = self.performance_stats.copy()
            stats['active_threads'] = len(self.calculation_threads)
            stats['cached_symbols'] = len(self.market_data_cache)
            return stats
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {}
    
    def export_indicator_definition(self, definition_id: str) -> str:
        """Export indicator definition as JSON"""
        try:
            if definition_id not in self.indicator_definitions:
                return ""
            
            definition = self.indicator_definitions[definition_id]
            def_dict = asdict(definition)
            def_dict['created_at'] = definition.created_at.isoformat()
            def_dict['updated_at'] = definition.updated_at.isoformat()
            
            return json.dumps(def_dict, indent=2)
            
        except Exception as e:
            logger.error(f"Error exporting indicator definition: {e}")
            return ""
    
    def import_indicator_definition(self, json_data: str) -> str:
        """Import indicator definition from JSON"""
        try:
            def_dict = json.loads(json_data)
            
            # Create new definition
            definition = IndicatorDefinition(**def_dict)
            definition.id = str(uuid4())  # Generate new ID
            definition.created_at = datetime.now()
            definition.updated_at = datetime.now()
            
            # Validate
            if not self._validate_calculation_code(definition.calculation_code):
                raise ValueError("Invalid calculation code")
            
            self.indicator_definitions[definition.id] = definition
            self._save_data()
            
            logger.info(f"Indicator definition imported: {definition.id}")
            return definition.id
            
        except Exception as e:
            logger.error(f"Error importing indicator definition: {e}")
            return None
    
    def add_callback(self, event_type: str, callback: Callable):
        """Add event callback"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
    
    def remove_callback(self, event_type: str, callback: Callable):
        """Remove event callback"""
        if event_type in self.callbacks and callback in self.callbacks[event_type]:
            self.callbacks[event_type].remove(callback)
    
    def clear_cache(self):
        """Clear market data cache"""
        try:
            self.market_data_cache.clear()
            self.cache_timestamps.clear()
            logger.info("Market data cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def shutdown(self):
        """Shutdown the custom indicators system"""
        try:
            # Stop all calculation threads
            for instance_id in list(self.active_calculations.keys()):
                self._stop_calculation_thread(instance_id)
            
            # Clear cache
            self.clear_cache()
            
            # Save data
            self._save_data()
            
            logger.info("Custom indicators system shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")