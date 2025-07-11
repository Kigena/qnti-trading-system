#!/usr/bin/env python3
"""
QNTI EA Generation Engine - Advanced Algorithmic Trading Strategy Factory
Comprehensive EA construction, optimization, and robustness testing platform
"""

import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
import threading
import time
import uuid
import copy
from pathlib import Path
import pickle
import random
from abc import ABC, abstractmethod

logger = logging.getLogger('QNTI_EA_ENGINE')

class ParameterType(Enum):
    """Parameter types for EA construction"""
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    STRING = "string"
    CHOICE = "choice"
    TIMEFRAME = "timeframe"
    SYMBOL = "symbol"

class LogicOperator(Enum):
    """Logic operators for condition building"""
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    XOR = "XOR"

class ComparisonOperator(Enum):
    """Comparison operators for conditions"""
    GT = ">"           # Greater than
    GTE = ">="         # Greater than or equal
    LT = "<"           # Less than
    LTE = "<="         # Less than or equal
    EQ = "=="          # Equal
    NEQ = "!="         # Not equal
    CROSS_ABOVE = "X>"  # Crosses above
    CROSS_BELOW = "X<"  # Crosses below
    INSIDE = "IN"      # Inside range
    OUTSIDE = "OUT"    # Outside range

class OptimizationMethod(Enum):
    """Optimization algorithms"""
    GRID_SEARCH = "grid_search"
    GENETIC_ALGORITHM = "genetic_algorithm"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    PARTICLE_SWARM = "particle_swarm"

class RobustnessTest(Enum):
    """Robustness testing methods"""
    WALK_FORWARD = "walk_forward"
    MONTE_CARLO = "monte_carlo"
    PARAMETER_SENSITIVITY = "parameter_sensitivity"
    RANDOM_START_DATES = "random_start_dates"
    STRESS_TESTING = "stress_testing"
    OUT_OF_SAMPLE = "out_of_sample"

@dataclass
class Parameter:
    """EA parameter definition"""
    name: str
    param_type: ParameterType
    min_value: Union[int, float] = None
    max_value: Union[int, float] = None
    default_value: Any = None
    choices: List[Any] = None
    step: Union[int, float] = None
    description: str = ""
    category: str = "general"
    
    def __post_init__(self):
        if self.param_type == ParameterType.CHOICE and not self.choices:
            raise ValueError(f"Parameter {self.name} of type CHOICE must have choices defined")
        
        if self.param_type in [ParameterType.INTEGER, ParameterType.FLOAT]:
            if self.min_value is None or self.max_value is None:
                raise ValueError(f"Parameter {self.name} must have min_value and max_value")

@dataclass
class Condition:
    """Single trading condition"""
    id: str
    left_operand: str  # Indicator or value
    operator: ComparisonOperator
    right_operand: str  # Indicator, value, or threshold
    description: str = ""
    weight: float = 1.0
    
    def evaluate(self, data: Dict[str, Any]) -> bool:
        """Evaluate condition against market data"""
        try:
            left_val = self._get_value(self.left_operand, data)
            right_val = self._get_value(self.right_operand, data)
            
            if self.operator == ComparisonOperator.GT:
                return left_val > right_val
            elif self.operator == ComparisonOperator.GTE:
                return left_val >= right_val
            elif self.operator == ComparisonOperator.LT:
                return left_val < right_val
            elif self.operator == ComparisonOperator.LTE:
                return left_val <= right_val
            elif self.operator == ComparisonOperator.EQ:
                return abs(left_val - right_val) < 1e-10
            elif self.operator == ComparisonOperator.NEQ:
                return abs(left_val - right_val) >= 1e-10
            elif self.operator == ComparisonOperator.CROSS_ABOVE:
                return self._check_cross_above(left_val, right_val, data)
            elif self.operator == ComparisonOperator.CROSS_BELOW:
                return self._check_cross_below(left_val, right_val, data)
            elif self.operator == ComparisonOperator.INSIDE:
                return self._check_inside_range(left_val, right_val, data)
            elif self.operator == ComparisonOperator.OUTSIDE:
                return self._check_outside_range(left_val, right_val, data)
            
            return False
        except Exception as e:
            logger.error(f"Error evaluating condition {self.id}: {e}")
            return False
    
    def _get_value(self, operand: str, data: Dict[str, Any]) -> float:
        """Get value from operand (indicator or literal)"""
        try:
            # Try to parse as float first (literal value)
            return float(operand)
        except ValueError:
            # Must be an indicator reference
            return data.get(operand, 0.0)
    
    def _check_cross_above(self, left_val: float, right_val: float, data: Dict[str, Any]) -> bool:
        """Check if left crosses above right"""
        # This would need historical data to determine crossing
        # For now, simplified implementation
        prev_left = data.get(f"{self.left_operand}_prev", left_val)
        prev_right = data.get(f"{self.right_operand}_prev", right_val)
        return prev_left <= prev_right and left_val > right_val
    
    def _check_cross_below(self, left_val: float, right_val: float, data: Dict[str, Any]) -> bool:
        """Check if left crosses below right"""
        prev_left = data.get(f"{self.left_operand}_prev", left_val)
        prev_right = data.get(f"{self.right_operand}_prev", right_val)
        return prev_left >= prev_right and left_val < right_val
    
    def _check_inside_range(self, left_val: float, right_val: float, data: Dict[str, Any]) -> bool:
        """Check if value is inside range"""
        # Assume right_val is center, look for range in data
        range_size = data.get(f"{self.right_operand}_range", 0.1)
        return (right_val - range_size) <= left_val <= (right_val + range_size)
    
    def _check_outside_range(self, left_val: float, right_val: float, data: Dict[str, Any]) -> bool:
        """Check if value is outside range"""
        return not self._check_inside_range(left_val, right_val, data)

@dataclass
class ConditionGroup:
    """Group of conditions with logic operators"""
    id: str
    conditions: List[Union[Condition, 'ConditionGroup']]
    operator: LogicOperator
    description: str = ""
    
    def evaluate(self, data: Dict[str, Any]) -> bool:
        """Evaluate condition group"""
        try:
            results = [cond.evaluate(data) for cond in self.conditions]
            
            if self.operator == LogicOperator.AND:
                return all(results)
            elif self.operator == LogicOperator.OR:
                return any(results)
            elif self.operator == LogicOperator.NOT:
                return not results[0] if results else False
            elif self.operator == LogicOperator.XOR:
                return sum(results) == 1
            
            return False
        except Exception as e:
            logger.error(f"Error evaluating condition group {self.id}: {e}")
            return False

@dataclass
class TradingRule:
    """Complete trading rule with entry/exit conditions"""
    id: str
    name: str
    entry_conditions: ConditionGroup
    exit_conditions: ConditionGroup
    stop_loss_conditions: Optional[ConditionGroup] = None
    take_profit_conditions: Optional[ConditionGroup] = None
    position_sizing: Dict[str, Any] = None
    risk_management: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.position_sizing is None:
            self.position_sizing = {"method": "fixed", "size": 0.01}
        if self.risk_management is None:
            self.risk_management = {"max_risk_percent": 2.0}

@dataclass
class EATemplate:
    """Expert Advisor template"""
    id: str
    name: str
    description: str
    version: str
    author: str
    created_at: datetime
    
    # Core components
    parameters: List[Parameter]
    indicators: List[str]  # Indicator names used
    trading_rules: List[TradingRule]
    
    # Metadata
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    optimization_targets: List[str] = field(default_factory=lambda: ["profit", "sharpe_ratio"])
    
    # Performance tracking
    backtest_results: Dict[str, Any] = field(default_factory=dict)
    optimization_history: List[Dict] = field(default_factory=list)
    robustness_scores: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.now()

@dataclass
class OptimizationResult:
    """Optimization result container"""
    ea_id: str
    method: OptimizationMethod
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    optimization_time: float
    iterations: int
    convergence_achieved: bool
    timestamp: datetime
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now()

@dataclass
class RobustnessTestResult:
    """Robustness testing result"""
    ea_id: str
    test_type: RobustnessTest
    passed: bool
    score: float
    details: Dict[str, Any]
    timestamp: datetime
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now()

class IndicatorLibrary:
    """Comprehensive technical indicator library"""
    
    def __init__(self):
        self.indicators = {}
        self._initialize_indicators()
    
    def _initialize_indicators(self):
        """Initialize 80+ technical indicators"""
        
        # Moving Averages (10 indicators)
        self.indicators.update({
            'SMA': self._create_sma_definition(),
            'EMA': self._create_ema_definition(),
            'WMA': self._create_wma_definition(),
            'LWMA': self._create_lwma_definition(),
            'DEMA': self._create_dema_definition(),
            'TEMA': self._create_tema_definition(),
            'TRIMA': self._create_trima_definition(),
            'KAMA': self._create_kama_definition(),
            'MAMA': self._create_mama_definition(),
            'VWMA': self._create_vwma_definition(),
        })
        
        # Momentum Indicators (15 indicators)
        self.indicators.update({
            'RSI': self._create_rsi_definition(),
            'MACD': self._create_macd_definition(),
            'Stochastic': self._create_stochastic_definition(),
            'Williams_R': self._create_williams_r_definition(),
            'ROC': self._create_roc_definition(),
            'Momentum': self._create_momentum_definition(),
            'CCI': self._create_cci_definition(),
            'CMO': self._create_cmo_definition(),
            'MFI': self._create_mfi_definition(),
            'TSI': self._create_tsi_definition(),
            'UO': self._create_ultimate_oscillator_definition(),
            'Aroon': self._create_aroon_definition(),
            'PPO': self._create_ppo_definition(),
            'DPO': self._create_dpo_definition(),
            'TRIX': self._create_trix_definition(),
        })
        
        # Volatility Indicators (12 indicators)
        self.indicators.update({
            'ATR': self._create_atr_definition(),
            'Bollinger_Bands': self._create_bollinger_definition(),
            'Keltner_Channel': self._create_keltner_definition(),
            'Donchian_Channel': self._create_donchian_definition(),
            'Standard_Deviation': self._create_stddev_definition(),
            'Variance': self._create_variance_definition(),
            'NATR': self._create_natr_definition(),
            'Volatility_Ratio': self._create_volatility_ratio_definition(),
            'VIX_Style': self._create_vix_style_definition(),
            'GARCH_Vol': self._create_garch_vol_definition(),
            'Parkinson_Vol': self._create_parkinson_vol_definition(),
            'Rogers_Satchell_Vol': self._create_rogers_satchell_vol_definition(),
        })
        
        # Volume Indicators (10 indicators)
        self.indicators.update({
            'OBV': self._create_obv_definition(),
            'Volume_SMA': self._create_volume_sma_definition(),
            'Volume_RSI': self._create_volume_rsi_definition(),
            'AD_Line': self._create_ad_line_definition(),
            'Chaikin_MF': self._create_chaikin_mf_definition(),
            'Force_Index': self._create_force_index_definition(),
            'Ease_of_Movement': self._create_eom_definition(),
            'VWAP': self._create_vwap_definition(),
            'Volume_Profile': self._create_volume_profile_definition(),
            'Klinger_Oscillator': self._create_klinger_definition(),
        })
        
        # Trend Indicators (15 indicators)
        self.indicators.update({
            'ADX': self._create_adx_definition(),
            'DI_Plus': self._create_di_plus_definition(),
            'DI_Minus': self._create_di_minus_definition(),
            'Parabolic_SAR': self._create_psar_definition(),
            'Supertrend': self._create_supertrend_definition(),
            'Ichimoku_Tenkan': self._create_ichimoku_tenkan_definition(),
            'Ichimoku_Kijun': self._create_ichimoku_kijun_definition(),
            'Ichimoku_Senkou_A': self._create_ichimoku_senkou_a_definition(),
            'Ichimoku_Senkou_B': self._create_ichimoku_senkou_b_definition(),
            'Zigzag': self._create_zigzag_definition(),
            'Linear_Regression': self._create_linear_regression_definition(),
            'Time_Series_Forecast': self._create_tsf_definition(),
            'Trend_Intensity': self._create_trend_intensity_definition(),
            'McGinley_Dynamic': self._create_mcginley_definition(),
            'HMA': self._create_hma_definition(),
        })
        
        # Cycle Indicators (8 indicators)
        self.indicators.update({
            'Sine_Wave': self._create_sine_wave_definition(),
            'Dominant_Cycle': self._create_dominant_cycle_definition(),
            'Hilbert_Transform': self._create_hilbert_transform_definition(),
            'MESA_Phase': self._create_mesa_phase_definition(),
            'Cycle_Period': self._create_cycle_period_definition(),
            'Instantaneous_Trendline': self._create_inst_trendline_definition(),
            'Market_Facilitation': self._create_market_facilitation_definition(),
            'Ehlers_Fisher': self._create_ehlers_fisher_definition(),
        })
        
        # Price Pattern Indicators (10 indicators)
        self.indicators.update({
            'Pivot_Points': self._create_pivot_points_definition(),
            'Support_Resistance': self._create_support_resistance_definition(),
            'Fractals': self._create_fractals_definition(),
            'High_Low_Index': self._create_high_low_index_definition(),
            'Price_Channel': self._create_price_channel_definition(),
            'Envelope': self._create_envelope_definition(),
            'Price_Oscillator': self._create_price_oscillator_definition(),
            'Detrended_Price': self._create_detrended_price_definition(),
            'Median_Price': self._create_median_price_definition(),
            'Typical_Price': self._create_typical_price_definition(),
        })
        
        logger.info(f"Initialized {len(self.indicators)} technical indicators")
    
    def _create_sma_definition(self) -> Dict[str, Any]:
        """Simple Moving Average indicator definition"""
        return {
            'name': 'Simple Moving Average',
            'category': 'Moving Average',
            'description': 'Arithmetic mean of prices over specified period',
            'parameters': [
                Parameter('period', ParameterType.INTEGER, 1, 200, 14, description='Number of periods'),
                Parameter('source', ParameterType.CHOICE, choices=['close', 'open', 'high', 'low', 'hl2', 'hlc3', 'ohlc4'], default_value='close')
            ],
            'outputs': ['sma'],
            'formula': 'sum(source, period) / period',
            'calculation_method': self._calculate_sma
        }
    
    def _create_ema_definition(self) -> Dict[str, Any]:
        """Exponential Moving Average indicator definition"""
        return {
            'name': 'Exponential Moving Average',
            'category': 'Moving Average',
            'description': 'Exponentially weighted moving average giving more weight to recent prices',
            'parameters': [
                Parameter('period', ParameterType.INTEGER, 1, 200, 14, description='Number of periods'),
                Parameter('source', ParameterType.CHOICE, choices=['close', 'open', 'high', 'low', 'hl2', 'hlc3', 'ohlc4'], default_value='close'),
                Parameter('alpha', ParameterType.FLOAT, 0.01, 0.99, None, description='Smoothing factor (auto-calculated if None)')
            ],
            'outputs': ['ema'],
            'formula': 'ema = alpha * source + (1 - alpha) * ema_prev',
            'calculation_method': self._calculate_ema
        }
    
    def _create_rsi_definition(self) -> Dict[str, Any]:
        """RSI indicator definition"""
        return {
            'name': 'Relative Strength Index',
            'category': 'Momentum',
            'description': 'Momentum oscillator measuring speed and magnitude of price changes',
            'parameters': [
                Parameter('period', ParameterType.INTEGER, 2, 100, 14, description='Number of periods'),
                Parameter('source', ParameterType.CHOICE, choices=['close', 'open', 'high', 'low'], default_value='close'),
                Parameter('overbought', ParameterType.FLOAT, 70, 90, 70, description='Overbought threshold'),
                Parameter('oversold', ParameterType.FLOAT, 10, 30, 30, description='Oversold threshold')
            ],
            'outputs': ['rsi', 'overbought_signal', 'oversold_signal'],
            'formula': 'rsi = 100 - (100 / (1 + rs)) where rs = avg_gain / avg_loss',
            'calculation_method': self._calculate_rsi
        }
    
    def _create_macd_definition(self) -> Dict[str, Any]:
        """MACD indicator definition"""
        return {
            'name': 'Moving Average Convergence Divergence',
            'category': 'Momentum',
            'description': 'Trend-following momentum indicator',
            'parameters': [
                Parameter('fast_period', ParameterType.INTEGER, 5, 50, 12, description='Fast EMA period'),
                Parameter('slow_period', ParameterType.INTEGER, 10, 100, 26, description='Slow EMA period'),
                Parameter('signal_period', ParameterType.INTEGER, 3, 30, 9, description='Signal line EMA period'),
                Parameter('source', ParameterType.CHOICE, choices=['close', 'open', 'high', 'low'], default_value='close')
            ],
            'outputs': ['macd', 'signal', 'histogram'],
            'formula': 'macd = ema(fast) - ema(slow), signal = ema(macd, signal_period)',
            'calculation_method': self._calculate_macd
        }
    
    def _create_bollinger_definition(self) -> Dict[str, Any]:
        """Bollinger Bands indicator definition"""
        return {
            'name': 'Bollinger Bands',
            'category': 'Volatility',
            'description': 'Volatility bands based on standard deviation',
            'parameters': [
                Parameter('period', ParameterType.INTEGER, 5, 100, 20, description='Moving average period'),
                Parameter('deviation', ParameterType.FLOAT, 0.5, 5.0, 2.0, description='Standard deviation multiplier'),
                Parameter('source', ParameterType.CHOICE, choices=['close', 'open', 'high', 'low'], default_value='close')
            ],
            'outputs': ['bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_percent'],
            'formula': 'upper = sma + (deviation * stddev), lower = sma - (deviation * stddev)',
            'calculation_method': self._calculate_bollinger
        }
    
    # Add more indicator definitions...
    # (Continuing with the remaining 60+ indicators)
    
    def _calculate_sma(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, pd.Series]:
        """Calculate Simple Moving Average"""
        period = params['period']
        source = params['source']
        
        if source in data.columns:
            sma = data[source].rolling(window=period).mean()
            return {'sma': sma}
        else:
            logger.error(f"Source column '{source}' not found in data")
            return {'sma': pd.Series(index=data.index, dtype=float)}
    
    def _calculate_ema(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, pd.Series]:
        """Calculate Exponential Moving Average"""
        period = params['period']
        source = params['source']
        alpha = params.get('alpha', 2.0 / (period + 1))
        
        if source in data.columns:
            ema = data[source].ewm(alpha=alpha).mean()
            return {'ema': ema}
        else:
            logger.error(f"Source column '{source}' not found in data")
            return {'ema': pd.Series(index=data.index, dtype=float)}
    
    def _calculate_rsi(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, pd.Series]:
        """Calculate RSI"""
        period = params['period']
        source = params['source']
        overbought = params['overbought']
        oversold = params['oversold']
        
        if source in data.columns:
            delta = data[source].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            overbought_signal = (rsi > overbought).astype(int)
            oversold_signal = (rsi < oversold).astype(int)
            
            return {
                'rsi': rsi,
                'overbought_signal': overbought_signal,
                'oversold_signal': oversold_signal
            }
        else:
            logger.error(f"Source column '{source}' not found in data")
            return {
                'rsi': pd.Series(index=data.index, dtype=float),
                'overbought_signal': pd.Series(index=data.index, dtype=int),
                'oversold_signal': pd.Series(index=data.index, dtype=int)
            }
    
    def _calculate_macd(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, pd.Series]:
        """Calculate MACD"""
        fast_period = params['fast_period']
        slow_period = params['slow_period']
        signal_period = params['signal_period']
        source = params['source']
        
        if source in data.columns:
            fast_ema = data[source].ewm(span=fast_period).mean()
            slow_ema = data[source].ewm(span=slow_period).mean()
            macd = fast_ema - slow_ema
            signal = macd.ewm(span=signal_period).mean()
            histogram = macd - signal
            
            return {
                'macd': macd,
                'signal': signal,
                'histogram': histogram
            }
        else:
            logger.error(f"Source column '{source}' not found in data")
            return {
                'macd': pd.Series(index=data.index, dtype=float),
                'signal': pd.Series(index=data.index, dtype=float),
                'histogram': pd.Series(index=data.index, dtype=float)
            }
    
    def _calculate_bollinger(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        period = params['period']
        deviation = params['deviation']
        source = params['source']
        
        if source in data.columns:
            sma = data[source].rolling(window=period).mean()
            std = data[source].rolling(window=period).std()
            
            bb_upper = sma + (deviation * std)
            bb_lower = sma - (deviation * std)
            bb_width = bb_upper - bb_lower
            bb_percent = (data[source] - bb_lower) / (bb_upper - bb_lower)
            
            return {
                'bb_upper': bb_upper,
                'bb_middle': sma,
                'bb_lower': bb_lower,
                'bb_width': bb_width,
                'bb_percent': bb_percent
            }
        else:
            logger.error(f"Source column '{source}' not found in data")
            return {col: pd.Series(index=data.index, dtype=float) 
                    for col in ['bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_percent']}
    
    # Placeholder methods for remaining indicators (to be implemented)
    def _create_wma_definition(self): return {"name": "WMA", "category": "Moving Average"}
    def _create_lwma_definition(self): return {"name": "LWMA", "category": "Moving Average"}
    def _create_dema_definition(self): return {"name": "DEMA", "category": "Moving Average"}
    def _create_tema_definition(self): return {"name": "TEMA", "category": "Moving Average"}
    def _create_trima_definition(self): return {"name": "TRIMA", "category": "Moving Average"}
    def _create_kama_definition(self): return {"name": "KAMA", "category": "Moving Average"}
    def _create_mama_definition(self): return {"name": "MAMA", "category": "Moving Average"}
    def _create_vwma_definition(self): return {"name": "VWMA", "category": "Moving Average"}
    
    # Continue with remaining indicator definitions...
    # (This is a foundation - the remaining 60+ indicators would follow the same pattern)
    
    def get_indicator(self, name: str) -> Optional[Dict[str, Any]]:
        """Get indicator definition by name"""
        return self.indicators.get(name)
    
    def list_indicators(self, category: str = None) -> List[str]:
        """List available indicators, optionally filtered by category"""
        if category:
            return [name for name, defn in self.indicators.items() 
                   if defn.get('category', '').lower() == category.lower()]
        return list(self.indicators.keys())
    
    def calculate_indicator(self, name: str, data: pd.DataFrame, 
                          params: Dict[str, Any]) -> Dict[str, pd.Series]:
        """Calculate indicator values"""
        indicator_def = self.get_indicator(name)
        if not indicator_def:
            raise ValueError(f"Unknown indicator: {name}")
        
        calculation_method = indicator_def.get('calculation_method')
        if not calculation_method:
            raise ValueError(f"No calculation method defined for indicator: {name}")
        
        return calculation_method(data, params)

class EAGenerationEngine:
    """Main EA Generation Engine"""
    
    def __init__(self, mt5_bridge=None, backtesting_engine=None):
        self.mt5_bridge = mt5_bridge
        self.backtesting_engine = backtesting_engine
        self.indicator_library = IndicatorLibrary()
        
        # Storage
        self.ea_templates = {}
        self.optimization_results = {}
        self.robustness_results = {}
        
        # Configuration
        self.data_path = Path("qnti_data/ea_generation")
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("QNTI EA Generation Engine initialized")
    
    def create_ea_template(self, name: str, description: str, author: str = "QNTI") -> EATemplate:
        """Create a new EA template"""
        template = EATemplate(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            version="1.0.0",
            author=author,
            created_at=datetime.now(),
            parameters=[],
            indicators=[],
            trading_rules=[]
        )
        
        self.ea_templates[template.id] = template
        logger.info(f"Created EA template: {name} ({template.id})")
        return template
    
    def add_indicator_to_ea(self, ea_id: str, indicator_name: str, 
                           params: Dict[str, Any]) -> bool:
        """Add an indicator to an EA template"""
        try:
            ea_template = self.ea_templates.get(ea_id)
            if not ea_template:
                logger.error(f"EA template not found: {ea_id}")
                return False
            
            indicator_def = self.indicator_library.get_indicator(indicator_name)
            if not indicator_def:
                logger.error(f"Indicator not found: {indicator_name}")
                return False
            
            # Add indicator to EA
            if indicator_name not in ea_template.indicators:
                ea_template.indicators.append(indicator_name)
            
            # Add parameters
            for param_def in indicator_def.get('parameters', []):
                param_name = f"{indicator_name}_{param_def.name}"
                if param_name not in [p.name for p in ea_template.parameters]:
                    param = Parameter(
                        name=param_name,
                        param_type=param_def.param_type,
                        min_value=param_def.min_value,
                        max_value=param_def.max_value,
                        default_value=params.get(param_def.name, param_def.default_value),
                        choices=param_def.choices,
                        step=param_def.step,
                        description=f"{indicator_name} - {param_def.description}",
                        category=f"indicator_{indicator_name.lower()}"
                    )
                    ea_template.parameters.append(param)
            
            logger.info(f"Added indicator {indicator_name} to EA {ea_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding indicator to EA: {e}")
            return False
    
    def save_ea_template(self, ea_id: str) -> bool:
        """Save EA template to disk"""
        try:
            ea_template = self.ea_templates.get(ea_id)
            if not ea_template:
                return False
            
            file_path = self.data_path / f"ea_template_{ea_id}.json"
            with open(file_path, 'w') as f:
                json.dump(asdict(ea_template), f, indent=2, default=str)
            
            logger.info(f"Saved EA template: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving EA template: {e}")
            return False
    
    def load_ea_template(self, ea_id: str) -> Optional[EATemplate]:
        """Load EA template from disk"""
        try:
            file_path = self.data_path / f"ea_template_{ea_id}.json"
            if not file_path.exists():
                return None
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Reconstruct EATemplate object
            # This is a simplified version - full implementation would handle nested objects
            template = EATemplate(**data)
            self.ea_templates[ea_id] = template
            
            logger.info(f"Loaded EA template: {ea_id}")
            return template
            
        except Exception as e:
            logger.error(f"Error loading EA template: {e}")
            return None

# Additional utility functions and classes would continue here...

if __name__ == "__main__":
    # Example usage
    engine = EAGenerationEngine()
    
    # Create a simple EA template
    ea = engine.create_ea_template(
        name="RSI Bollinger Strategy",
        description="Strategy using RSI and Bollinger Bands",
        author="QNTI System"
    )
    
    # Add indicators
    engine.add_indicator_to_ea(ea.id, "RSI", {"period": 14, "overbought": 70, "oversold": 30})
    engine.add_indicator_to_ea(ea.id, "Bollinger_Bands", {"period": 20, "deviation": 2.0})
    
    # Save template
    engine.save_ea_template(ea.id)
    
    print(f"Created EA template with {len(ea.indicators)} indicators and {len(ea.parameters)} parameters")