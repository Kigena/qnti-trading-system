"""
QNTI Expert Advisor Profiling System

This module provides comprehensive EA analysis and profiling capabilities,
allowing the AI to understand each EA's trading logic, indicators, and parameters
for intelligent optimization and management decisions.
"""

import json
import logging
import re
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class StrategyType(Enum):
    """EA Strategy Categories"""
    TREND_FOLLOWING = "trend_following"
    SCALPING = "scalping"
    GRID_TRADING = "grid_trading"
    HEDGING = "hedging"
    ARBITRAGE = "arbitrage"
    BREAKOUT = "breakout"
    MEAN_REVERSION = "mean_reversion"
    NEWS_TRADING = "news_trading"
    CORRELATION = "correlation"
    MARTINGALE = "martingale"
    COUNTER_TREND = "counter_trend"
    UNKNOWN = "unknown"

class IndicatorType(Enum):
    """Technical Indicator Categories"""
    # Trend Indicators
    MA = "moving_average"
    EMA = "exponential_ma"
    SMA = "simple_ma"
    WMA = "weighted_ma"
    BOLLINGER = "bollinger_bands"
    ICHIMOKU = "ichimoku"
    PARABOLIC_SAR = "parabolic_sar"
    
    # Momentum Indicators
    RSI = "rsi"
    MACD = "macd"
    STOCHASTIC = "stochastic"
    CCI = "cci"
    WILLIAMS_R = "williams_r"
    MOMENTUM = "momentum"
    
    # Volume Indicators
    VOLUME = "volume"
    OBV = "on_balance_volume"
    MFI = "money_flow_index"
    
    # Volatility Indicators
    ATR = "average_true_range"
    STANDARD_DEV = "standard_deviation"
    
    # Custom Indicators
    CUSTOM = "custom"
    UNKNOWN = "unknown"

class RiskLevel(Enum):
    """Risk Level Classifications"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    EXTREME = "extreme"

@dataclass
class IndicatorConfig:
    """Configuration for a technical indicator"""
    name: str
    type: IndicatorType
    parameters: Dict[str, Any]
    timeframe: str
    weight: float = 1.0  # Importance in decision making
    description: Optional[str] = None

@dataclass
class EntryCondition:
    """EA Entry condition logic"""
    indicator: str
    condition: str  # e.g., "RSI < 30", "MACD > Signal", "Price > EMA"
    logic_operator: str = "AND"  # AND, OR, XOR
    weight: float = 1.0
    description: Optional[str] = None

@dataclass
class ExitCondition:
    """EA Exit condition logic"""
    type: str  # "stop_loss", "take_profit", "trailing_stop", "indicator_signal"
    value: Optional[float] = None
    indicator: Optional[str] = None
    condition: Optional[str] = None
    is_dynamic: bool = False
    description: Optional[str] = None

@dataclass
class RiskManagement:
    """EA Risk management configuration"""
    lot_sizing_method: str  # "fixed", "percent_balance", "percent_equity", "atr_based"
    lot_size: float = 0.01
    max_risk_per_trade: float = 0.02  # 2% default
    max_open_trades: int = 1
    max_drawdown_limit: float = 0.20  # 20% default
    correlation_filter: bool = False
    time_filter: Dict[str, Any] = None  # Trading hours, days
    news_filter: bool = False

@dataclass
class EAProfile:
    """Comprehensive EA Profile with trading logic and characteristics"""
    ea_name: str
    magic_number: int
    symbol: str
    strategy_type: StrategyType
    description: str
    
    # Trading Logic
    indicators: List[IndicatorConfig]
    entry_conditions: List[EntryCondition]
    exit_conditions: List[ExitCondition]
    risk_management: RiskManagement
    
    # Market Analysis
    timeframe: str
    multi_timeframe: bool = False
    supported_symbols: List[str] = None
    market_sessions: List[str] = None  # "london", "new_york", "tokyo", "sydney"
    
    # EA Characteristics
    is_grid_based: bool = False
    is_martingale: bool = False
    uses_hedging: bool = False
    is_news_sensitive: bool = False
    requires_low_spread: bool = False
    
    # Performance Optimization
    optimization_parameters: Dict[str, Dict] = None  # Parameter ranges for optimization
    backtest_results: Dict[str, Any] = None
    forward_test_results: Dict[str, Any] = None
    
    # AI Understanding
    strengths: List[str] = None
    weaknesses: List[str] = None
    best_market_conditions: List[str] = None
    worst_market_conditions: List[str] = None
    recommended_pairs: List[str] = None
    
    # Metadata
    version: str = "1.0"
    author: str = "Unknown"
    created_date: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    confidence_score: float = 0.5  # AI confidence in profile accuracy

class QNTIEAProfiler:
    """Advanced EA Profiling and Analysis System"""
    
    def __init__(self, data_dir: str = "qnti_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.profiles_file = self.data_dir / "ea_profiles.json"
        self.db_path = self.data_dir / "ea_profiling.db"
        
        # Load existing profiles
        self.profiles: Dict[str, EAProfile] = {}
        self._init_database()
        self._load_profiles()
        
        logger.info(f"EA Profiler initialized with {len(self.profiles)} profiles")

    def _init_database(self):
        """Initialize SQLite database for EA profiling"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # EA Profiles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ea_profiles (
                ea_name TEXT PRIMARY KEY,
                magic_number INTEGER,
                symbol TEXT,
                strategy_type TEXT,
                description TEXT,
                profile_data TEXT,
                confidence_score REAL,
                created_date TIMESTAMP,
                last_updated TIMESTAMP
            )
        ''')
        
        # Indicators table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ea_indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ea_name TEXT,
                indicator_name TEXT,
                indicator_type TEXT,
                parameters TEXT,
                timeframe TEXT,
                weight REAL,
                FOREIGN KEY (ea_name) REFERENCES ea_profiles (ea_name)
            )
        ''')
        
        # Performance Analysis table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ea_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ea_name TEXT,
                analysis_date TIMESTAMP,
                market_condition TEXT,
                performance_score REAL,
                recommendations TEXT,
                FOREIGN KEY (ea_name) REFERENCES ea_profiles (ea_name)
            )
        ''')
        
        conn.commit()
        conn.close()

    def _load_profiles(self):
        """Load EA profiles from file"""
        try:
            if self.profiles_file.exists():
                with open(self.profiles_file, 'r') as f:
                    data = json.load(f)
                    
                for ea_name, profile_dict in data.items():
                    # Convert dates
                    if profile_dict.get('created_date'):
                        profile_dict['created_date'] = datetime.fromisoformat(profile_dict['created_date'])
                    if profile_dict.get('last_updated'):
                        profile_dict['last_updated'] = datetime.fromisoformat(profile_dict['last_updated'])
                    
                    # Convert enums
                    profile_dict['strategy_type'] = StrategyType(profile_dict.get('strategy_type', 'unknown'))
                    
                    # Convert indicators
                    indicators = []
                    for ind_dict in profile_dict.get('indicators', []):
                        ind_dict['type'] = IndicatorType(ind_dict.get('type', 'unknown'))
                        indicators.append(IndicatorConfig(**ind_dict))
                    profile_dict['indicators'] = indicators
                    
                    # Convert entry conditions
                    entry_conditions = []
                    for cond_dict in profile_dict.get('entry_conditions', []):
                        entry_conditions.append(EntryCondition(**cond_dict))
                    profile_dict['entry_conditions'] = entry_conditions
                    
                    # Convert exit conditions
                    exit_conditions = []
                    for cond_dict in profile_dict.get('exit_conditions', []):
                        exit_conditions.append(ExitCondition(**cond_dict))
                    profile_dict['exit_conditions'] = exit_conditions
                    
                    # Convert risk management
                    if 'risk_management' in profile_dict:
                        profile_dict['risk_management'] = RiskManagement(**profile_dict['risk_management'])
                    else:
                        profile_dict['risk_management'] = RiskManagement(lot_sizing_method="fixed")
                    
                    self.profiles[ea_name] = EAProfile(**profile_dict)
                    
        except Exception as e:
            logger.error(f"Error loading EA profiles: {e}")

    def _save_profiles(self):
        """Save EA profiles to file"""
        try:
            data = {}
            for ea_name, profile in self.profiles.items():
                profile_dict = asdict(profile)
                
                # Convert dates to ISO format
                if profile_dict.get('created_date'):
                    profile_dict['created_date'] = profile_dict['created_date'].isoformat()
                if profile_dict.get('last_updated'):
                    profile_dict['last_updated'] = profile_dict['last_updated'].isoformat()
                
                # Convert enums to strings
                profile_dict['strategy_type'] = profile_dict['strategy_type'].value
                
                # Convert indicator types
                for ind in profile_dict.get('indicators', []):
                    ind['type'] = ind['type'].value
                
                data[ea_name] = profile_dict
            
            with open(self.profiles_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving EA profiles: {e}")

    def create_profile(self, ea_name: str, magic_number: int, symbol: str, 
                      strategy_type: str = "unknown", description: str = "") -> EAProfile:
        """Create a new EA profile"""
        try:
            profile = EAProfile(
                ea_name=ea_name,
                magic_number=magic_number,
                symbol=symbol,
                strategy_type=StrategyType(strategy_type),
                description=description,
                indicators=[],
                entry_conditions=[],
                exit_conditions=[],
                risk_management=RiskManagement(lot_sizing_method="fixed"),
                timeframe="M15",
                created_date=datetime.now(),
                last_updated=datetime.now()
            )
            
            self.profiles[ea_name] = profile
            self._save_profiles()
            
            logger.info(f"Created new EA profile for {ea_name}")
            return profile
            
        except Exception as e:
            logger.error(f"Error creating EA profile for {ea_name}: {e}")
            return None

    def analyze_ea_from_mql_code(self, ea_name: str, mql_code: str) -> EAProfile:
        """Analyze MQL4/MQL5 code to extract EA logic and create profile"""
        try:
            # Initialize profile
            profile = self.profiles.get(ea_name)
            if not profile:
                # Extract magic number from code
                magic_match = re.search(r'MagicNumber\s*=\s*(\d+)', mql_code, re.IGNORECASE)
                magic_number = int(magic_match.group(1)) if magic_match else 0
                
                profile = self.create_profile(ea_name, magic_number, "UNKNOWN")
            
            # Analyze strategy type
            strategy_type = self._detect_strategy_type(mql_code)
            profile.strategy_type = strategy_type
            
            # Extract indicators
            indicators = self._extract_indicators(mql_code)
            profile.indicators = indicators
            
            # Extract entry/exit logic
            entry_conditions = self._extract_entry_conditions(mql_code)
            exit_conditions = self._extract_exit_conditions(mql_code)
            profile.entry_conditions = entry_conditions
            profile.exit_conditions = exit_conditions
            
            # Extract risk management
            risk_management = self._extract_risk_management(mql_code)
            profile.risk_management = risk_management
            
            # Analyze EA characteristics
            profile.is_grid_based = self._is_grid_based(mql_code)
            profile.is_martingale = self._is_martingale(mql_code)
            profile.uses_hedging = self._uses_hedging(mql_code)
            profile.is_news_sensitive = self._is_news_sensitive(mql_code)
            
            # Update metadata
            profile.last_updated = datetime.now()
            profile.confidence_score = 0.8  # High confidence from code analysis
            
            self.profiles[ea_name] = profile
            self._save_profiles()
            
            logger.info(f"Analyzed and updated profile for {ea_name} from MQL code")
            return profile
            
        except Exception as e:
            logger.error(f"Error analyzing MQL code for {ea_name}: {e}")
            return None

    def _detect_strategy_type(self, mql_code: str) -> StrategyType:
        """Detect EA strategy type from MQL code"""
        code_lower = mql_code.lower()
        
        # Strategy detection patterns
        if any(keyword in code_lower for keyword in ['grid', 'pending', 'buystop', 'sellstop']):
            return StrategyType.GRID_TRADING
        elif any(keyword in code_lower for keyword in ['scalp', 'quick', 'fast', 'short']):
            return StrategyType.SCALPING
        elif any(keyword in code_lower for keyword in ['trend', 'ema', 'sma', 'moving']):
            return StrategyType.TREND_FOLLOWING
        elif any(keyword in code_lower for keyword in ['hedge', 'opposite', 'reverse']):
            return StrategyType.HEDGING
        elif any(keyword in code_lower for keyword in ['breakout', 'break', 'support', 'resistance']):
            return StrategyType.BREAKOUT
        elif any(keyword in code_lower for keyword in ['martingale', 'doubling', 'recovery']):
            return StrategyType.MARTINGALE
        elif any(keyword in code_lower for keyword in ['mean', 'reversion', 'bollinger']):
            return StrategyType.MEAN_REVERSION
        elif any(keyword in code_lower for keyword in ['news', 'economic', 'calendar']):
            return StrategyType.NEWS_TRADING
        elif any(keyword in code_lower for keyword in ['correlation', 'pair', 'basket']):
            return StrategyType.CORRELATION
        elif any(keyword in code_lower for keyword in ['arbitrage', 'spread', 'difference']):
            return StrategyType.ARBITRAGE
        else:
            return StrategyType.UNKNOWN

    def _extract_indicators(self, mql_code: str) -> List[IndicatorConfig]:
        """Extract technical indicators from MQL code"""
        indicators = []
        
        # Common indicator patterns
        indicator_patterns = {
            IndicatorType.MA: r'iMA\s*\(\s*[^,]+,\s*(\d+),\s*(\d+)',
            IndicatorType.EMA: r'iMA\s*\([^,]+,\s*(\d+),\s*(\d+),\s*[^,]+,\s*MODE_EMA',
            IndicatorType.RSI: r'iRSI\s*\(\s*[^,]+,\s*(\d+),\s*(\d+)',
            IndicatorType.MACD: r'iMACD\s*\(\s*[^,]+,\s*(\d+),\s*(\d+),\s*(\d+)',
            IndicatorType.BOLLINGER: r'iBands\s*\(\s*[^,]+,\s*(\d+),\s*(\d+)',
            IndicatorType.STOCHASTIC: r'iStochastic\s*\(\s*[^,]+,\s*(\d+),\s*(\d+),\s*(\d+)',
            IndicatorType.ATR: r'iATR\s*\(\s*[^,]+,\s*(\d+),\s*(\d+)',
        }
        
        for indicator_type, pattern in indicator_patterns.items():
            matches = re.finditer(pattern, mql_code)
            for match in matches:
                params = {}
                if indicator_type == IndicatorType.MA:
                    params = {'timeframe': match.group(1), 'period': int(match.group(2))}
                elif indicator_type == IndicatorType.RSI:
                    params = {'timeframe': match.group(1), 'period': int(match.group(2))}
                elif indicator_type == IndicatorType.MACD:
                    params = {'timeframe': match.group(1), 'fast_ema': int(match.group(2)), 'slow_ema': int(match.group(3))}
                
                indicator = IndicatorConfig(
                    name=f"{indicator_type.value}_{len(indicators)}",
                    type=indicator_type,
                    parameters=params,
                    timeframe=params.get('timeframe', 'M15')
                )
                indicators.append(indicator)
        
        return indicators

    def _extract_entry_conditions(self, mql_code: str) -> List[EntryCondition]:
        """Extract entry conditions from MQL code"""
        conditions = []
        
        # Look for buy/sell conditions in the code
        buy_patterns = [
            r'OrderSend.*OP_BUY.*if\s*\((.*?)\)',
            r'if\s*\((.*?)\).*OrderSend.*OP_BUY',
        ]
        
        sell_patterns = [
            r'OrderSend.*OP_SELL.*if\s*\((.*?)\)',
            r'if\s*\((.*?)\).*OrderSend.*OP_SELL',
        ]
        
        for pattern in buy_patterns + sell_patterns:
            matches = re.finditer(pattern, mql_code, re.DOTALL)
            for match in matches:
                condition_text = match.group(1).strip()
                
                condition = EntryCondition(
                    indicator="detected_from_code",
                    condition=condition_text[:100],  # Limit length
                    description=f"Auto-detected: {condition_text[:50]}..."
                )
                conditions.append(condition)
        
        return conditions

    def _extract_exit_conditions(self, mql_code: str) -> List[ExitCondition]:
        """Extract exit conditions from MQL code"""
        conditions = []
        
        # Look for stop loss and take profit
        sl_pattern = r'StopLoss\s*=\s*(\d+(?:\.\d+)?)'
        tp_pattern = r'TakeProfit\s*=\s*(\d+(?:\.\d+)?)'
        
        sl_match = re.search(sl_pattern, mql_code)
        if sl_match:
            conditions.append(ExitCondition(
                type="stop_loss",
                value=float(sl_match.group(1)),
                description="Auto-detected stop loss"
            ))
        
        tp_match = re.search(tp_pattern, mql_code)
        if tp_match:
            conditions.append(ExitCondition(
                type="take_profit",
                value=float(tp_match.group(1)),
                description="Auto-detected take profit"
            ))
        
        # Look for trailing stop
        if 'trailing' in mql_code.lower():
            conditions.append(ExitCondition(
                type="trailing_stop",
                is_dynamic=True,
                description="Auto-detected trailing stop"
            ))
        
        return conditions

    def _extract_risk_management(self, mql_code: str) -> RiskManagement:
        """Extract risk management settings from MQL code"""
        # Default risk management
        risk_mgmt = RiskManagement(lot_sizing_method="fixed")
        
        # Look for lot size logic
        if 'accountfreeMargin' in mql_code.lower() or 'accountequity' in mql_code.lower():
            risk_mgmt.lot_sizing_method = "percent_equity"
        elif 'accountbalance' in mql_code.lower():
            risk_mgmt.lot_sizing_method = "percent_balance"
        elif 'atr' in mql_code.lower() and 'lot' in mql_code.lower():
            risk_mgmt.lot_sizing_method = "atr_based"
        
        # Extract max trades
        max_trades_pattern = r'MaxTrades\s*=\s*(\d+)'
        max_trades_match = re.search(max_trades_pattern, mql_code)
        if max_trades_match:
            risk_mgmt.max_open_trades = int(max_trades_match.group(1))
        
        # Extract risk percentage
        risk_pattern = r'Risk\s*=\s*(\d+(?:\.\d+)?)'
        risk_match = re.search(risk_pattern, mql_code)
        if risk_match:
            risk_mgmt.max_risk_per_trade = float(risk_match.group(1)) / 100
        
        return risk_mgmt

    def _is_grid_based(self, mql_code: str) -> bool:
        """Check if EA uses grid trading strategy"""
        grid_keywords = ['grid', 'pending', 'distance', 'step', 'buystop', 'sellstop']
        return any(keyword in mql_code.lower() for keyword in grid_keywords)

    def _is_martingale(self, mql_code: str) -> bool:
        """Check if EA uses martingale strategy"""
        martingale_keywords = ['martingale', 'doubling', 'multiply', 'recovery', 'factor']
        return any(keyword in mql_code.lower() for keyword in martingale_keywords)

    def _uses_hedging(self, mql_code: str) -> bool:
        """Check if EA uses hedging"""
        hedging_keywords = ['hedge', 'opposite', 'reverse', 'balance']
        return any(keyword in mql_code.lower() for keyword in hedging_keywords)

    def _is_news_sensitive(self, mql_code: str) -> bool:
        """Check if EA is sensitive to news"""
        news_keywords = ['news', 'economic', 'calendar', 'event', 'announcement']
        return any(keyword in mql_code.lower() for keyword in news_keywords)

    def get_ai_optimization_recommendations(self, ea_name: str, 
                                         market_condition: str = "normal") -> List[Dict]:
        """Generate AI-powered optimization recommendations for specific EA"""
        try:
            profile = self.profiles.get(ea_name)
            if not profile:
                return []
            
            recommendations = []
            
            # Strategy-specific recommendations
            if profile.strategy_type == StrategyType.SCALPING:
                if market_condition == "high_volatility":
                    recommendations.append({
                        "type": "parameter_adjustment",
                        "parameter": "take_profit",
                        "action": "increase",
                        "value": 1.5,
                        "reason": "Increase TP in high volatility for scalping EAs"
                    })
                elif market_condition == "low_volatility":
                    recommendations.append({
                        "type": "status_change",
                        "action": "pause",
                        "reason": "Scalping EAs perform poorly in low volatility"
                    })
            
            elif profile.strategy_type == StrategyType.TREND_FOLLOWING:
                if market_condition == "trending":
                    recommendations.append({
                        "type": "parameter_adjustment",
                        "parameter": "lot_size",
                        "action": "increase",
                        "value": 1.2,
                        "reason": "Increase lot size during strong trends"
                    })
                elif market_condition == "ranging":
                    recommendations.append({
                        "type": "parameter_adjustment",
                        "parameter": "sensitivity",
                        "action": "decrease",
                        "value": 0.8,
                        "reason": "Reduce sensitivity in ranging markets"
                    })
            
            elif profile.strategy_type == StrategyType.GRID_TRADING:
                if market_condition == "trending":
                    recommendations.append({
                        "type": "status_change",
                        "action": "pause",
                        "reason": "Grid EAs can be dangerous in strong trends"
                    })
                elif market_condition == "ranging":
                    recommendations.append({
                        "type": "parameter_adjustment",
                        "parameter": "grid_step",
                        "action": "optimize",
                        "reason": "Optimize grid step for current range"
                    })
            
            # Risk-based recommendations
            if profile.risk_management.max_risk_per_trade > 0.05:  # 5%
                recommendations.append({
                    "type": "risk_reduction",
                    "parameter": "risk_per_trade",
                    "action": "reduce",
                    "value": 0.02,
                    "reason": "Risk per trade too high, reduce to 2%"
                })
            
            # Indicator-based recommendations
            for indicator in profile.indicators:
                if indicator.type == IndicatorType.RSI:
                    recommendations.append({
                        "type": "indicator_optimization",
                        "indicator": indicator.name,
                        "suggestion": "Consider RSI divergence signals",
                        "reason": "RSI can be enhanced with divergence analysis"
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations for {ea_name}: {e}")
            return []

    def get_profile(self, ea_name: str) -> Optional[EAProfile]:
        """Get EA profile by name"""
        return self.profiles.get(ea_name)

    def update_profile(self, ea_name: str, updates: Dict[str, Any]) -> bool:
        """Update an existing EA profile"""
        try:
            if ea_name not in self.profiles:
                return False
            
            profile = self.profiles[ea_name]
            
            # Update fields
            for key, value in updates.items():
                if hasattr(profile, key):
                    setattr(profile, key, value)
            
            profile.last_updated = datetime.now()
            self._save_profiles()
            
            logger.info(f"Updated profile for {ea_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating profile for {ea_name}: {e}")
            return False

    def get_all_profiles(self) -> Dict[str, EAProfile]:
        """Get all EA profiles"""
        return self.profiles

    def analyze_market_compatibility(self, ea_name: str, 
                                   current_market: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how compatible an EA is with current market conditions"""
        try:
            profile = self.profiles.get(ea_name)
            if not profile:
                return {"compatibility": 0.0, "reason": "Profile not found"}
            
            compatibility_score = 0.5  # Base score
            reasons = []
            
            # Market condition analysis
            volatility = current_market.get('volatility', 'normal')
            trend_strength = current_market.get('trend_strength', 0.5)
            
            # Strategy-specific compatibility
            if profile.strategy_type == StrategyType.SCALPING:
                if volatility == 'high':
                    compatibility_score += 0.3
                    reasons.append("High volatility favors scalping")
                elif volatility == 'low':
                    compatibility_score -= 0.4
                    reasons.append("Low volatility unfavorable for scalping")
            
            elif profile.strategy_type == StrategyType.TREND_FOLLOWING:
                if trend_strength > 0.7:
                    compatibility_score += 0.4
                    reasons.append("Strong trend favors trend-following")
                elif trend_strength < 0.3:
                    compatibility_score -= 0.3
                    reasons.append("Weak trend unfavorable for trend-following")
            
            elif profile.strategy_type == StrategyType.GRID_TRADING:
                if trend_strength < 0.4:  # Ranging market
                    compatibility_score += 0.3
                    reasons.append("Ranging market favors grid trading")
                else:
                    compatibility_score -= 0.4
                    reasons.append("Trending market dangerous for grid trading")
            
            # Risk assessment
            if profile.risk_management.max_risk_per_trade > 0.05:
                compatibility_score -= 0.2
                reasons.append("High risk settings reduce compatibility")
            
            # Clamp score between 0 and 1
            compatibility_score = max(0.0, min(1.0, compatibility_score))
            
            return {
                "compatibility": round(compatibility_score, 2),
                "reasons": reasons,
                "recommendation": "suitable" if compatibility_score > 0.6 else 
                                "caution" if compatibility_score > 0.3 else "avoid"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market compatibility for {ea_name}: {e}")
            return {"compatibility": 0.0, "reason": f"Analysis error: {str(e)}"}

    def export_profile_summary(self, ea_name: str) -> Dict[str, Any]:
        """Export a comprehensive profile summary for AI analysis"""
        try:
            profile = self.profiles.get(ea_name)
            if not profile:
                return {}
            
            return {
                "name": profile.ea_name,
                "strategy_type": profile.strategy_type.value,
                "description": profile.description,
                "key_indicators": [
                    {
                        "name": ind.name,
                        "type": ind.type.value,
                        "parameters": ind.parameters
                    } for ind in profile.indicators
                ],
                "trading_logic": {
                    "entry_conditions": len(profile.entry_conditions),
                    "exit_conditions": len(profile.exit_conditions),
                    "risk_management": {
                        "method": profile.risk_management.lot_sizing_method,
                        "max_risk": profile.risk_management.max_risk_per_trade,
                        "max_trades": profile.risk_management.max_open_trades
                    }
                },
                "characteristics": {
                    "is_grid_based": profile.is_grid_based,
                    "is_martingale": profile.is_martingale,
                    "uses_hedging": profile.uses_hedging,
                    "is_news_sensitive": profile.is_news_sensitive,
                    "requires_low_spread": profile.requires_low_spread
                },
                "optimization_hints": {
                    "strengths": profile.strengths or [],
                    "weaknesses": profile.weaknesses or [],
                    "best_conditions": profile.best_market_conditions or [],
                    "worst_conditions": profile.worst_market_conditions or []
                },
                "confidence_score": profile.confidence_score
            }
            
        except Exception as e:
            logger.error(f"Error exporting profile summary for {ea_name}: {e}")
            return {} 