#!/usr/bin/env python3
"""
QNTI Vision Analysis - Data Models and Structures
Contains all enums, dataclasses, and data structures for the vision analysis system
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

class AnalysisConfidence(Enum):
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95

class SignalStrength(Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

class MarketBias(Enum):
    STRONGLY_BULLISH = "strongly_bullish"
    BULLISH = "bullish"
    NEUTRAL_BULLISH = "neutral_bullish"
    NEUTRAL = "neutral"
    NEUTRAL_BEARISH = "neutral_bearish"
    BEARISH = "bearish"
    STRONGLY_BEARISH = "strongly_bearish"

@dataclass
class TechnicalIndicator:
    """Technical indicator reading from chart"""
    name: str
    value: Optional[float] = None
    signal: Optional[str] = None  # bullish, bearish, neutral
    strength: Optional[str] = None
    notes: Optional[str] = None

@dataclass
class PriceLevel:
    """Price level with context"""
    price: float
    level_type: str  # support, resistance, pivot, fib_level
    strength: str  # weak, moderate, strong
    context: str  # description of why this level is significant
    tested_count: int = 0
    last_test: Optional[str] = None

@dataclass
class TradeScenario:
    """Complete trade scenario with all details"""
    scenario_name: str
    trade_type: str  # BUY, SELL, HOLD
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: Optional[float] = None
    take_profit_3: Optional[float] = None
    lot_size_recommendation: float = 0.1
    risk_reward_ratio: float = 0.0
    probability_success: float = 0.0  # 0.0 to 1.0
    entry_conditions: List[str] = field(default_factory=list)
    invalidation_conditions: List[str] = field(default_factory=list)
    time_frame_validity: str = "4H"
    notes: str = ""
    
    # Add this property for backward compatibility
    @property
    def take_profit(self):
        return self.take_profit_1

@dataclass
class ComprehensiveChartAnalysis:
    """Comprehensive chart analysis with all components"""
    # Core required fields
    analysis_id: str
    timestamp: datetime
    symbol: str
    timeframe: str
    overall_trend: str  # bullish, bearish, sideways
    trend_strength: SignalStrength
    market_bias: MarketBias
    market_structure_notes: str
    current_price: float
    support_levels: List[PriceLevel]
    resistance_levels: List[PriceLevel]
    key_levels: List[PriceLevel]
    indicators: List[TechnicalIndicator]
    patterns_detected: List[str]
    pattern_completion: float  # 0-1
    pattern_reliability: str  # low, medium, high
    pattern_notes: str
    primary_scenario: TradeScenario
    overall_confidence: float  # 0.0 to 1.0 - CRITICAL FOR FRONTEND
    
    # Optional fields
    alternative_scenario: Optional[TradeScenario] = None
    risk_factors: List[str] = field(default_factory=list)
    confluence_factors: List[str] = field(default_factory=list)
    market_context: Dict[str, Any] = field(default_factory=dict)
    session_analysis: str = ""
    volatility_assessment: str = ""
    news_considerations: str = ""
    reasoning: str = ""
    chart_quality: str = "good"
    analysis_notes: str = ""

@dataclass
class ImageUpload:
    """Image upload metadata"""
    filename: str
    upload_time: datetime
    file_size: int
    image_format: str
    dimensions: Tuple[int, int]
    base64_data: str
    analysis_id: Optional[str] = None 