#!/usr/bin/env python3
"""
QNTI Trade Journal System
Comprehensive trade journaling with analytics, performance tracking, and insights
"""

import json
import logging
import math
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from uuid import uuid4
import pandas as pd
import numpy as np

logger = logging.getLogger('QNTI_TRADE_JOURNAL')

class JournalEntryType(Enum):
    """Journal entry types"""
    TRADE_ENTRY = "trade_entry"
    MARKET_OBSERVATION = "market_observation"
    STRATEGY_NOTE = "strategy_note"
    LESSON_LEARNED = "lesson_learned"
    GOAL_SETTING = "goal_setting"
    PERFORMANCE_REVIEW = "performance_review"
    PSYCHOLOGY_NOTE = "psychology_note"
    NEWS_IMPACT = "news_impact"

class EmotionType(Enum):
    """Trading emotion types"""
    CONFIDENT = "confident"
    ANXIOUS = "anxious"
    EXCITED = "excited"
    FEARFUL = "fearful"
    GREEDY = "greedy"
    PATIENT = "patient"
    FRUSTRATED = "frustrated"
    CALM = "calm"
    STRESSED = "stressed"
    OPTIMISTIC = "optimistic"

class TradingSetup(Enum):
    """Trading setup types"""
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    TREND_CONTINUATION = "trend_continuation"
    SUPPORT_RESISTANCE = "support_resistance"
    FIBONACCI_RETRACEMENT = "fibonacci_retracement"
    MOVING_AVERAGE_CROSSOVER = "moving_average_crossover"
    DIVERGENCE = "divergence"
    PATTERN_RECOGNITION = "pattern_recognition"
    NEWS_DRIVEN = "news_driven"
    ALGORITHMIC = "algorithmic"

class TradeOutcome(Enum):
    """Trade outcome classification"""
    BIG_WIN = "big_win"
    SMALL_WIN = "small_win"
    BREAKEVEN = "breakeven"
    SMALL_LOSS = "small_loss"
    BIG_LOSS = "big_loss"
    STOPPED_OUT = "stopped_out"

@dataclass
class TradingGoal:
    """Trading goal data structure"""
    id: str
    title: str
    description: str
    target_value: float
    current_value: float
    unit: str  # percentage, dollars, trades, etc.
    target_date: datetime
    is_achieved: bool = False
    
    # Progress tracking
    progress_percentage: float = 0.0
    milestones: List[Dict] = None
    
    # Timestamps
    created_at: datetime = None
    updated_at: datetime = None
    achieved_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.milestones is None:
            self.milestones = []
        
        # Calculate progress
        if self.target_value != 0:
            self.progress_percentage = min(100, (self.current_value / self.target_value) * 100)

@dataclass
class JournalEntry:
    """Journal entry data structure"""
    id: str
    entry_type: JournalEntryType
    title: str
    content: str
    
    # Trade-specific fields
    trade_id: Optional[str] = None
    symbol: Optional[str] = None
    setup_type: Optional[TradingSetup] = None
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    position_size: Optional[float] = None
    profit_loss: Optional[float] = None
    
    # Psychological fields
    emotions_before: List[EmotionType] = None
    emotions_during: List[EmotionType] = None
    emotions_after: List[EmotionType] = None
    confidence_level: Optional[int] = None  # 1-10
    
    # Analysis fields
    what_worked: str = ""
    what_didnt_work: str = ""
    lessons_learned: str = ""
    improvements_needed: str = ""
    
    # Market conditions
    market_conditions: Dict = None
    news_events: List[str] = None
    
    # Media attachments
    screenshots: List[str] = None
    charts: List[str] = None
    
    # Tags and categorization
    tags: List[str] = None
    category: str = ""
    
    # Ratings (1-10)
    execution_quality: Optional[int] = None
    risk_management: Optional[int] = None
    patience_level: Optional[int] = None
    
    # Timestamps
    created_at: datetime = None
    updated_at: datetime = None
    trade_date: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.emotions_before is None:
            self.emotions_before = []
        if self.emotions_during is None:
            self.emotions_during = []
        if self.emotions_after is None:
            self.emotions_after = []
        if self.market_conditions is None:
            self.market_conditions = {}
        if self.news_events is None:
            self.news_events = []
        if self.screenshots is None:
            self.screenshots = []
        if self.charts is None:
            self.charts = []
        if self.tags is None:
            self.tags = []

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    # Basic metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # Profit/Loss metrics
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_profit: float = 0.0
    profit_factor: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_percentage: float = 0.0
    current_drawdown: float = 0.0
    
    # Advanced metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Trade quality metrics
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    # Consistency metrics
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    
    # Time-based metrics
    average_trade_duration: float = 0.0  # in hours
    trades_per_day: float = 0.0
    
    # Risk-adjusted metrics
    return_on_investment: float = 0.0
    risk_reward_ratio: float = 0.0
    
    # Psychological metrics
    average_confidence: float = 0.0
    emotional_stability: float = 0.0

@dataclass
class TradingInsight:
    """Trading insight generated from journal analysis"""
    id: str
    title: str
    description: str
    insight_type: str
    confidence_score: float
    
    # Supporting data
    supporting_trades: List[str] = None
    metrics: Dict = None
    
    # Recommendations
    recommendations: List[str] = None
    
    # Timestamps
    generated_at: datetime = None
    
    def __post_init__(self):
        if self.generated_at is None:
            self.generated_at = datetime.now()
        if self.supporting_trades is None:
            self.supporting_trades = []
        if self.metrics is None:
            self.metrics = {}
        if self.recommendations is None:
            self.recommendations = []

class QNTITradeJournal:
    """QNTI Trade Journal System"""
    
    def __init__(self, trade_manager, mt5_bridge=None):
        self.trade_manager = trade_manager
        self.mt5_bridge = mt5_bridge
        
        # Data storage
        self.journal_entries: Dict[str, JournalEntry] = {}
        self.trading_goals: Dict[str, TradingGoal] = {}
        self.insights: Dict[str, TradingInsight] = {}
        
        # Performance tracking
        self.performance_metrics = PerformanceMetrics()
        self.daily_metrics: Dict[str, PerformanceMetrics] = {}
        self.monthly_metrics: Dict[str, PerformanceMetrics] = {}
        
        # Analysis settings
        self.analysis_settings = {
            'min_trades_for_insight': 10,
            'confidence_threshold': 0.7,
            'update_interval': 3600,  # 1 hour
            'auto_generate_insights': True
        }
        
        # Load existing data
        self._load_data()
        
        # Calculate initial metrics
        self._calculate_all_metrics()
        
        logger.info("Trade Journal System initialized")
    
    def _load_data(self):
        """Load existing journal data"""
        try:
            import os
            
            if os.path.exists('trade_journal_data.json'):
                with open('trade_journal_data.json', 'r') as f:
                    data = json.load(f)
                    
                    # Load journal entries
                    for entry_data in data.get('journal_entries', []):
                        entry = JournalEntry(**entry_data)
                        entry.created_at = datetime.fromisoformat(entry_data['created_at'])
                        entry.updated_at = datetime.fromisoformat(entry_data['updated_at'])
                        if entry_data.get('trade_date'):
                            entry.trade_date = datetime.fromisoformat(entry_data['trade_date'])
                        self.journal_entries[entry.id] = entry
                    
                    # Load trading goals
                    for goal_data in data.get('trading_goals', []):
                        goal = TradingGoal(**goal_data)
                        goal.created_at = datetime.fromisoformat(goal_data['created_at'])
                        goal.updated_at = datetime.fromisoformat(goal_data['updated_at'])
                        goal.target_date = datetime.fromisoformat(goal_data['target_date'])
                        if goal_data.get('achieved_at'):
                            goal.achieved_at = datetime.fromisoformat(goal_data['achieved_at'])
                        self.trading_goals[goal.id] = goal
                    
                    # Load insights
                    for insight_data in data.get('insights', []):
                        insight = TradingInsight(**insight_data)
                        insight.generated_at = datetime.fromisoformat(insight_data['generated_at'])
                        self.insights[insight.id] = insight
                        
                logger.info("Trade journal data loaded successfully")
                
        except Exception as e:
            logger.error(f"Error loading trade journal data: {e}")
    
    def _save_data(self):
        """Save journal data"""
        try:
            data = {
                'journal_entries': [],
                'trading_goals': [],
                'insights': []
            }
            
            # Save journal entries
            for entry in self.journal_entries.values():
                entry_data = asdict(entry)
                entry_data['created_at'] = entry.created_at.isoformat()
                entry_data['updated_at'] = entry.updated_at.isoformat()
                if entry.trade_date:
                    entry_data['trade_date'] = entry.trade_date.isoformat()
                data['journal_entries'].append(entry_data)
            
            # Save trading goals
            for goal in self.trading_goals.values():
                goal_data = asdict(goal)
                goal_data['created_at'] = goal.created_at.isoformat()
                goal_data['updated_at'] = goal.updated_at.isoformat()
                goal_data['target_date'] = goal.target_date.isoformat()
                if goal.achieved_at:
                    goal_data['achieved_at'] = goal.achieved_at.isoformat()
                data['trading_goals'].append(goal_data)
            
            # Save insights
            for insight in self.insights.values():
                insight_data = asdict(insight)
                insight_data['generated_at'] = insight.generated_at.isoformat()
                data['insights'].append(insight_data)
            
            with open('trade_journal_data.json', 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving trade journal data: {e}")
    
    def _calculate_all_metrics(self):
        """Calculate all performance metrics"""
        try:
            # Get all trade entries
            trade_entries = [entry for entry in self.journal_entries.values() 
                           if entry.entry_type == JournalEntryType.TRADE_ENTRY 
                           and entry.profit_loss is not None]
            
            if not trade_entries:
                return
            
            # Calculate basic metrics
            self.performance_metrics.total_trades = len(trade_entries)
            self.performance_metrics.winning_trades = sum(1 for entry in trade_entries 
                                                        if entry.profit_loss > 0)
            self.performance_metrics.losing_trades = sum(1 for entry in trade_entries 
                                                       if entry.profit_loss < 0)
            
            if self.performance_metrics.total_trades > 0:
                self.performance_metrics.win_rate = (
                    self.performance_metrics.winning_trades / self.performance_metrics.total_trades
                ) * 100
            
            # Calculate profit/loss metrics
            profits = [entry.profit_loss for entry in trade_entries if entry.profit_loss > 0]
            losses = [entry.profit_loss for entry in trade_entries if entry.profit_loss < 0]
            
            self.performance_metrics.gross_profit = sum(profits) if profits else 0
            self.performance_metrics.gross_loss = sum(losses) if losses else 0
            self.performance_metrics.net_profit = sum(entry.profit_loss for entry in trade_entries)
            
            if self.performance_metrics.gross_loss != 0:
                self.performance_metrics.profit_factor = abs(
                    self.performance_metrics.gross_profit / self.performance_metrics.gross_loss
                )
            
            # Calculate advanced metrics
            self._calculate_drawdown_metrics(trade_entries)
            self._calculate_risk_metrics(trade_entries)
            self._calculate_consistency_metrics(trade_entries)
            self._calculate_psychological_metrics(trade_entries)
            
            # Calculate daily and monthly metrics
            self._calculate_period_metrics(trade_entries)
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
    
    def _calculate_drawdown_metrics(self, trade_entries: List[JournalEntry]):
        """Calculate drawdown metrics"""
        try:
            if not trade_entries:
                return
            
            # Sort by date
            sorted_entries = sorted(trade_entries, key=lambda x: x.created_at)
            
            # Calculate running balance
            running_balance = 0
            peak_balance = 0
            max_drawdown = 0
            current_drawdown = 0
            
            for entry in sorted_entries:
                running_balance += entry.profit_loss
                
                if running_balance > peak_balance:
                    peak_balance = running_balance
                    current_drawdown = 0
                else:
                    current_drawdown = peak_balance - running_balance
                    max_drawdown = max(max_drawdown, current_drawdown)
            
            self.performance_metrics.max_drawdown = max_drawdown
            self.performance_metrics.current_drawdown = current_drawdown
            
            if peak_balance != 0:
                self.performance_metrics.max_drawdown_percentage = (max_drawdown / peak_balance) * 100
                
        except Exception as e:
            logger.error(f"Error calculating drawdown metrics: {e}")
    
    def _calculate_risk_metrics(self, trade_entries: List[JournalEntry]):
        """Calculate risk-adjusted metrics"""
        try:
            if not trade_entries:
                return
            
            returns = [entry.profit_loss for entry in trade_entries]
            
            if len(returns) < 2:
                return
            
            # Calculate Sharpe ratio
            avg_return = statistics.mean(returns)
            std_return = statistics.stdev(returns)
            
            if std_return != 0:
                self.performance_metrics.sharpe_ratio = avg_return / std_return
            
            # Calculate Sortino ratio (downside deviation)
            negative_returns = [r for r in returns if r < 0]
            if negative_returns:
                downside_deviation = statistics.stdev(negative_returns)
                if downside_deviation != 0:
                    self.performance_metrics.sortino_ratio = avg_return / downside_deviation
            
            # Calculate other metrics
            profits = [entry.profit_loss for entry in trade_entries if entry.profit_loss > 0]
            losses = [entry.profit_loss for entry in trade_entries if entry.profit_loss < 0]
            
            if profits:
                self.performance_metrics.average_win = statistics.mean(profits)
                self.performance_metrics.largest_win = max(profits)
            
            if losses:
                self.performance_metrics.average_loss = statistics.mean(losses)
                self.performance_metrics.largest_loss = min(losses)
            
            # Risk-reward ratio
            if self.performance_metrics.average_loss != 0:
                self.performance_metrics.risk_reward_ratio = abs(
                    self.performance_metrics.average_win / self.performance_metrics.average_loss
                )
                
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
    
    def _calculate_consistency_metrics(self, trade_entries: List[JournalEntry]):
        """Calculate consistency metrics"""
        try:
            if not trade_entries:
                return
            
            # Sort by date
            sorted_entries = sorted(trade_entries, key=lambda x: x.created_at)
            
            # Calculate consecutive wins/losses
            current_wins = 0
            current_losses = 0
            max_wins = 0
            max_losses = 0
            
            for entry in sorted_entries:
                if entry.profit_loss > 0:
                    current_wins += 1
                    current_losses = 0
                    max_wins = max(max_wins, current_wins)
                elif entry.profit_loss < 0:
                    current_losses += 1
                    current_wins = 0
                    max_losses = max(max_losses, current_losses)
            
            self.performance_metrics.max_consecutive_wins = max_wins
            self.performance_metrics.max_consecutive_losses = max_losses
            
            # Calculate trades per day
            if len(sorted_entries) > 1:
                first_trade = sorted_entries[0].created_at
                last_trade = sorted_entries[-1].created_at
                days_diff = (last_trade - first_trade).days + 1
                self.performance_metrics.trades_per_day = len(sorted_entries) / days_diff
                
        except Exception as e:
            logger.error(f"Error calculating consistency metrics: {e}")
    
    def _calculate_psychological_metrics(self, trade_entries: List[JournalEntry]):
        """Calculate psychological metrics"""
        try:
            if not trade_entries:
                return
            
            # Calculate average confidence
            confidence_scores = [entry.confidence_level for entry in trade_entries 
                               if entry.confidence_level is not None]
            
            if confidence_scores:
                self.performance_metrics.average_confidence = statistics.mean(confidence_scores)
            
            # Calculate emotional stability (simplified)
            # This would be more complex in a real implementation
            execution_scores = [entry.execution_quality for entry in trade_entries 
                              if entry.execution_quality is not None]
            
            if execution_scores:
                self.performance_metrics.emotional_stability = statistics.mean(execution_scores)
                
        except Exception as e:
            logger.error(f"Error calculating psychological metrics: {e}")
    
    def _calculate_period_metrics(self, trade_entries: List[JournalEntry]):
        """Calculate daily and monthly metrics"""
        try:
            # Group by day
            daily_trades = {}
            monthly_trades = {}
            
            for entry in trade_entries:
                date_key = entry.created_at.date().isoformat()
                month_key = entry.created_at.strftime('%Y-%m')
                
                if date_key not in daily_trades:
                    daily_trades[date_key] = []
                daily_trades[date_key].append(entry)
                
                if month_key not in monthly_trades:
                    monthly_trades[month_key] = []
                monthly_trades[month_key].append(entry)
            
            # Calculate daily metrics
            for date_key, entries in daily_trades.items():
                metrics = self._calculate_metrics_for_entries(entries)
                self.daily_metrics[date_key] = metrics
            
            # Calculate monthly metrics
            for month_key, entries in monthly_trades.items():
                metrics = self._calculate_metrics_for_entries(entries)
                self.monthly_metrics[month_key] = metrics
                
        except Exception as e:
            logger.error(f"Error calculating period metrics: {e}")
    
    def _calculate_metrics_for_entries(self, entries: List[JournalEntry]) -> PerformanceMetrics:
        """Calculate metrics for a specific set of entries"""
        try:
            metrics = PerformanceMetrics()
            
            if not entries:
                return metrics
            
            metrics.total_trades = len(entries)
            metrics.winning_trades = sum(1 for entry in entries if entry.profit_loss > 0)
            metrics.losing_trades = sum(1 for entry in entries if entry.profit_loss < 0)
            
            if metrics.total_trades > 0:
                metrics.win_rate = (metrics.winning_trades / metrics.total_trades) * 100
            
            profits = [entry.profit_loss for entry in entries if entry.profit_loss > 0]
            losses = [entry.profit_loss for entry in entries if entry.profit_loss < 0]
            
            metrics.gross_profit = sum(profits) if profits else 0
            metrics.gross_loss = sum(losses) if losses else 0
            metrics.net_profit = sum(entry.profit_loss for entry in entries)
            
            if metrics.gross_loss != 0:
                metrics.profit_factor = abs(metrics.gross_profit / metrics.gross_loss)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics for entries: {e}")
            return PerformanceMetrics()
    
    def _generate_insights(self):
        """Generate trading insights from journal data"""
        try:
            if len(self.journal_entries) < self.analysis_settings['min_trades_for_insight']:
                return
            
            # Clear old insights
            self.insights.clear()
            
            # Generate different types of insights
            self._generate_performance_insights()
            self._generate_psychological_insights()
            self._generate_setup_insights()
            self._generate_timing_insights()
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
    
    def _generate_performance_insights(self):
        """Generate performance-based insights"""
        try:
            # Win rate analysis
            if self.performance_metrics.win_rate < 40:
                insight = TradingInsight(
                    id=str(uuid4()),
                    title="Low Win Rate Alert",
                    description=f"Your win rate is {self.performance_metrics.win_rate:.1f}%, which is below optimal levels.",
                    insight_type="performance",
                    confidence_score=0.9,
                    recommendations=[
                        "Review your entry criteria",
                        "Consider tightening your setup requirements",
                        "Analyze your most profitable setups"
                    ]
                )
                self.insights[insight.id] = insight
            
            # Risk-reward analysis
            if self.performance_metrics.risk_reward_ratio < 1.5:
                insight = TradingInsight(
                    id=str(uuid4()),
                    title="Poor Risk-Reward Ratio",
                    description=f"Your risk-reward ratio is {self.performance_metrics.risk_reward_ratio:.2f}, indicating insufficient reward for risk taken.",
                    insight_type="risk_management",
                    confidence_score=0.8,
                    recommendations=[
                        "Increase your profit targets",
                        "Tighten your stop losses",
                        "Look for higher probability setups"
                    ]
                )
                self.insights[insight.id] = insight
                
        except Exception as e:
            logger.error(f"Error generating performance insights: {e}")
    
    def _generate_psychological_insights(self):
        """Generate psychology-based insights"""
        try:
            # Confidence analysis
            if self.performance_metrics.average_confidence < 6:
                insight = TradingInsight(
                    id=str(uuid4()),
                    title="Low Confidence Levels",
                    description=f"Your average confidence level is {self.performance_metrics.average_confidence:.1f}/10, which may indicate uncertainty in your strategy.",
                    insight_type="psychology",
                    confidence_score=0.7,
                    recommendations=[
                        "Backtest your strategy more thoroughly",
                        "Start with smaller position sizes",
                        "Focus on high-probability setups only"
                    ]
                )
                self.insights[insight.id] = insight
            
            # Emotional stability analysis
            trade_entries = [entry for entry in self.journal_entries.values() 
                           if entry.entry_type == JournalEntryType.TRADE_ENTRY]
            
            stress_trades = sum(1 for entry in trade_entries 
                              if EmotionType.STRESSED in entry.emotions_before or
                                 EmotionType.ANXIOUS in entry.emotions_before)
            
            if stress_trades > len(trade_entries) * 0.3:
                insight = TradingInsight(
                    id=str(uuid4()),
                    title="High Stress Levels Detected",
                    description=f"You show stress or anxiety in {stress_trades} trades ({stress_trades/len(trade_entries)*100:.1f}%).",
                    insight_type="psychology",
                    confidence_score=0.8,
                    recommendations=[
                        "Reduce position sizes",
                        "Take breaks between trades",
                        "Consider meditation or stress management techniques"
                    ]
                )
                self.insights[insight.id] = insight
                
        except Exception as e:
            logger.error(f"Error generating psychological insights: {e}")
    
    def _generate_setup_insights(self):
        """Generate setup-based insights"""
        try:
            trade_entries = [entry for entry in self.journal_entries.values() 
                           if entry.entry_type == JournalEntryType.TRADE_ENTRY 
                           and entry.setup_type is not None]
            
            if not trade_entries:
                return
            
            # Analyze setup performance
            setup_performance = {}
            
            for entry in trade_entries:
                setup = entry.setup_type
                if setup not in setup_performance:
                    setup_performance[setup] = {'trades': [], 'profit': 0, 'count': 0}
                
                setup_performance[setup]['trades'].append(entry)
                setup_performance[setup]['profit'] += entry.profit_loss or 0
                setup_performance[setup]['count'] += 1
            
            # Find best and worst setups
            best_setup = None
            worst_setup = None
            best_profit = float('-inf')
            worst_profit = float('inf')
            
            for setup, data in setup_performance.items():
                if data['count'] >= 3:  # Minimum trades for analysis
                    avg_profit = data['profit'] / data['count']
                    if avg_profit > best_profit:
                        best_profit = avg_profit
                        best_setup = setup
                    if avg_profit < worst_profit:
                        worst_profit = avg_profit
                        worst_setup = setup
            
            if best_setup:
                insight = TradingInsight(
                    id=str(uuid4()),
                    title="Best Performing Setup",
                    description=f"Your {best_setup.value} setup has the highest average profit of ${best_profit:.2f}.",
                    insight_type="setup_analysis",
                    confidence_score=0.8,
                    recommendations=[
                        f"Focus more on {best_setup.value} setups",
                        "Analyze what makes these setups successful",
                        "Consider increasing position size for this setup"
                    ]
                )
                self.insights[insight.id] = insight
            
            if worst_setup and worst_profit < 0:
                insight = TradingInsight(
                    id=str(uuid4()),
                    title="Underperforming Setup",
                    description=f"Your {worst_setup.value} setup has a negative average profit of ${worst_profit:.2f}.",
                    insight_type="setup_analysis",
                    confidence_score=0.8,
                    recommendations=[
                        f"Avoid or refine {worst_setup.value} setups",
                        "Analyze why these setups are failing",
                        "Consider removing this setup from your strategy"
                    ]
                )
                self.insights[insight.id] = insight
                
        except Exception as e:
            logger.error(f"Error generating setup insights: {e}")
    
    def _generate_timing_insights(self):
        """Generate timing-based insights"""
        try:
            trade_entries = [entry for entry in self.journal_entries.values() 
                           if entry.entry_type == JournalEntryType.TRADE_ENTRY]
            
            if not trade_entries:
                return
            
            # Analyze performance by hour
            hour_performance = {}
            
            for entry in trade_entries:
                hour = entry.created_at.hour
                if hour not in hour_performance:
                    hour_performance[hour] = {'profit': 0, 'count': 0}
                
                hour_performance[hour]['profit'] += entry.profit_loss or 0
                hour_performance[hour]['count'] += 1
            
            # Find best and worst trading hours
            best_hour = None
            worst_hour = None
            best_profit = float('-inf')
            worst_profit = float('inf')
            
            for hour, data in hour_performance.items():
                if data['count'] >= 3:  # Minimum trades for analysis
                    avg_profit = data['profit'] / data['count']
                    if avg_profit > best_profit:
                        best_profit = avg_profit
                        best_hour = hour
                    if avg_profit < worst_profit:
                        worst_profit = avg_profit
                        worst_hour = hour
            
            if best_hour is not None:
                insight = TradingInsight(
                    id=str(uuid4()),
                    title="Best Trading Hour",
                    description=f"Your best trading performance occurs at {best_hour:02d}:00 with average profit of ${best_profit:.2f}.",
                    insight_type="timing_analysis",
                    confidence_score=0.7,
                    recommendations=[
                        f"Focus more trading activity around {best_hour:02d}:00",
                        "Analyze market conditions during this hour",
                        "Consider your mental state during this time"
                    ]
                )
                self.insights[insight.id] = insight
            
            if worst_hour is not None and worst_profit < 0:
                insight = TradingInsight(
                    id=str(uuid4()),
                    title="Worst Trading Hour",
                    description=f"Your worst trading performance occurs at {worst_hour:02d}:00 with average loss of ${worst_profit:.2f}.",
                    insight_type="timing_analysis",
                    confidence_score=0.7,
                    recommendations=[
                        f"Avoid trading around {worst_hour:02d}:00",
                        "Analyze why this time period is problematic",
                        "Consider taking breaks during this hour"
                    ]
                )
                self.insights[insight.id] = insight
                
        except Exception as e:
            logger.error(f"Error generating timing insights: {e}")
    
    # Public API methods
    
    def create_journal_entry(self, entry_type: JournalEntryType, title: str, 
                           content: str, **kwargs) -> str:
        """Create a new journal entry"""
        try:
            entry_id = str(uuid4())
            entry = JournalEntry(
                id=entry_id,
                entry_type=entry_type,
                title=title,
                content=content,
                **kwargs
            )
            
            self.journal_entries[entry_id] = entry
            
            # Update metrics if it's a trade entry
            if entry_type == JournalEntryType.TRADE_ENTRY:
                self._calculate_all_metrics()
                
                # Generate insights if enabled
                if self.analysis_settings['auto_generate_insights']:
                    self._generate_insights()
            
            self._save_data()
            
            logger.info(f"Journal entry created: {entry_id}")
            return entry_id
            
        except Exception as e:
            logger.error(f"Error creating journal entry: {e}")
            return None
    
    def update_journal_entry(self, entry_id: str, **kwargs) -> bool:
        """Update an existing journal entry"""
        try:
            if entry_id not in self.journal_entries:
                return False
            
            entry = self.journal_entries[entry_id]
            
            # Update fields
            for key, value in kwargs.items():
                if hasattr(entry, key):
                    setattr(entry, key, value)
            
            entry.updated_at = datetime.now()
            
            # Recalculate metrics if it's a trade entry
            if entry.entry_type == JournalEntryType.TRADE_ENTRY:
                self._calculate_all_metrics()
            
            self._save_data()
            
            logger.info(f"Journal entry updated: {entry_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating journal entry: {e}")
            return False
    
    def delete_journal_entry(self, entry_id: str) -> bool:
        """Delete a journal entry"""
        try:
            if entry_id not in self.journal_entries:
                return False
            
            entry = self.journal_entries[entry_id]
            del self.journal_entries[entry_id]
            
            # Recalculate metrics if it was a trade entry
            if entry.entry_type == JournalEntryType.TRADE_ENTRY:
                self._calculate_all_metrics()
            
            self._save_data()
            
            logger.info(f"Journal entry deleted: {entry_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting journal entry: {e}")
            return False
    
    def create_trading_goal(self, title: str, description: str, target_value: float,
                          unit: str, target_date: datetime, **kwargs) -> str:
        """Create a new trading goal"""
        try:
            goal_id = str(uuid4())
            goal = TradingGoal(
                id=goal_id,
                title=title,
                description=description,
                target_value=target_value,
                current_value=0.0,
                unit=unit,
                target_date=target_date,
                **kwargs
            )
            
            self.trading_goals[goal_id] = goal
            self._save_data()
            
            logger.info(f"Trading goal created: {goal_id}")
            return goal_id
            
        except Exception as e:
            logger.error(f"Error creating trading goal: {e}")
            return None
    
    def update_goal_progress(self, goal_id: str, current_value: float) -> bool:
        """Update progress on a trading goal"""
        try:
            if goal_id not in self.trading_goals:
                return False
            
            goal = self.trading_goals[goal_id]
            goal.current_value = current_value
            goal.updated_at = datetime.now()
            
            # Calculate progress percentage
            if goal.target_value != 0:
                goal.progress_percentage = min(100, (current_value / goal.target_value) * 100)
            
            # Check if goal is achieved
            if current_value >= goal.target_value and not goal.is_achieved:
                goal.is_achieved = True
                goal.achieved_at = datetime.now()
                logger.info(f"Goal achieved: {goal.title}")
            
            self._save_data()
            return True
            
        except Exception as e:
            logger.error(f"Error updating goal progress: {e}")
            return False
    
    def get_journal_entries(self, entry_type: JournalEntryType = None, 
                          limit: int = 100, offset: int = 0) -> List[Dict]:
        """Get journal entries"""
        try:
            entries = list(self.journal_entries.values())
            
            # Filter by type if specified
            if entry_type:
                entries = [e for e in entries if e.entry_type == entry_type]
            
            # Sort by creation date (newest first)
            entries.sort(key=lambda x: x.created_at, reverse=True)
            
            # Apply pagination
            entries = entries[offset:offset + limit]
            
            # Convert to dict format
            result = []
            for entry in entries:
                entry_dict = asdict(entry)
                entry_dict['created_at'] = entry.created_at.isoformat()
                entry_dict['updated_at'] = entry.updated_at.isoformat()
                if entry.trade_date:
                    entry_dict['trade_date'] = entry.trade_date.isoformat()
                result.append(entry_dict)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting journal entries: {e}")
            return []
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        try:
            return asdict(self.performance_metrics)
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    def get_daily_performance(self, days: int = 30) -> Dict[str, Dict]:
        """Get daily performance for the last N days"""
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            
            result = {}
            current_date = start_date
            
            while current_date <= end_date:
                date_key = current_date.isoformat()
                
                if date_key in self.daily_metrics:
                    result[date_key] = asdict(self.daily_metrics[date_key])
                else:
                    result[date_key] = asdict(PerformanceMetrics())
                
                current_date += timedelta(days=1)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting daily performance: {e}")
            return {}
    
    def get_monthly_performance(self, months: int = 12) -> Dict[str, Dict]:
        """Get monthly performance for the last N months"""
        try:
            result = {}
            current_date = datetime.now().date()
            
            for i in range(months):
                # Calculate month key
                month_date = current_date.replace(day=1) - timedelta(days=i*30)
                month_key = month_date.strftime('%Y-%m')
                
                if month_key in self.monthly_metrics:
                    result[month_key] = asdict(self.monthly_metrics[month_key])
                else:
                    result[month_key] = asdict(PerformanceMetrics())
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting monthly performance: {e}")
            return {}
    
    def get_trading_insights(self, limit: int = 10) -> List[Dict]:
        """Get trading insights"""
        try:
            insights = list(self.insights.values())
            
            # Sort by confidence score (highest first)
            insights.sort(key=lambda x: x.confidence_score, reverse=True)
            
            # Apply limit
            insights = insights[:limit]
            
            # Convert to dict format
            result = []
            for insight in insights:
                insight_dict = asdict(insight)
                insight_dict['generated_at'] = insight.generated_at.isoformat()
                result.append(insight_dict)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting trading insights: {e}")
            return []
    
    def get_trading_goals(self, include_achieved: bool = True) -> List[Dict]:
        """Get trading goals"""
        try:
            goals = list(self.trading_goals.values())
            
            # Filter by achievement status if specified
            if not include_achieved:
                goals = [g for g in goals if not g.is_achieved]
            
            # Sort by target date
            goals.sort(key=lambda x: x.target_date)
            
            # Convert to dict format
            result = []
            for goal in goals:
                goal_dict = asdict(goal)
                goal_dict['created_at'] = goal.created_at.isoformat()
                goal_dict['updated_at'] = goal.updated_at.isoformat()
                goal_dict['target_date'] = goal.target_date.isoformat()
                if goal.achieved_at:
                    goal_dict['achieved_at'] = goal.achieved_at.isoformat()
                result.append(goal_dict)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting trading goals: {e}")
            return []
    
    def search_journal_entries(self, query: str, entry_type: JournalEntryType = None) -> List[Dict]:
        """Search journal entries"""
        try:
            query_lower = query.lower()
            results = []
            
            for entry in self.journal_entries.values():
                # Filter by type if specified
                if entry_type and entry.entry_type != entry_type:
                    continue
                
                # Search in title and content
                if (query_lower in entry.title.lower() or 
                    query_lower in entry.content.lower() or
                    query_lower in entry.category.lower() or
                    any(query_lower in tag.lower() for tag in entry.tags)):
                    
                    entry_dict = asdict(entry)
                    entry_dict['created_at'] = entry.created_at.isoformat()
                    entry_dict['updated_at'] = entry.updated_at.isoformat()
                    if entry.trade_date:
                        entry_dict['trade_date'] = entry.trade_date.isoformat()
                    results.append(entry_dict)
            
            # Sort by relevance (for now, just by date)
            results.sort(key=lambda x: x['created_at'], reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching journal entries: {e}")
            return []
    
    def generate_performance_report(self, period: str = 'all') -> Dict:
        """Generate comprehensive performance report"""
        try:
            report = {
                'period': period,
                'generated_at': datetime.now().isoformat(),
                'overall_metrics': asdict(self.performance_metrics),
                'insights': [asdict(insight) for insight in self.insights.values()],
                'goals_progress': []
            }
            
            # Add goals progress
            for goal in self.trading_goals.values():
                goal_progress = {
                    'title': goal.title,
                    'progress_percentage': goal.progress_percentage,
                    'is_achieved': goal.is_achieved,
                    'target_date': goal.target_date.isoformat(),
                    'days_remaining': (goal.target_date - datetime.now()).days
                }
                report['goals_progress'].append(goal_progress)
            
            # Add period-specific metrics
            if period == 'daily':
                report['daily_metrics'] = self.get_daily_performance(30)
            elif period == 'monthly':
                report['monthly_metrics'] = self.get_monthly_performance(12)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {}
    
    def export_journal_data(self, format: str = 'json') -> str:
        """Export journal data in specified format"""
        try:
            if format == 'json':
                return json.dumps({
                    'journal_entries': [asdict(entry) for entry in self.journal_entries.values()],
                    'trading_goals': [asdict(goal) for goal in self.trading_goals.values()],
                    'performance_metrics': asdict(self.performance_metrics),
                    'insights': [asdict(insight) for insight in self.insights.values()]
                }, indent=2, default=str)
            
            elif format == 'csv':
                # Create CSV export for trade entries
                trade_entries = [entry for entry in self.journal_entries.values() 
                               if entry.entry_type == JournalEntryType.TRADE_ENTRY]
                
                if not trade_entries:
                    return ""
                
                # Convert to DataFrame for CSV export
                data = []
                for entry in trade_entries:
                    data.append({
                        'date': entry.created_at.isoformat(),
                        'symbol': entry.symbol,
                        'setup_type': entry.setup_type.value if entry.setup_type else '',
                        'entry_price': entry.entry_price,
                        'exit_price': entry.exit_price,
                        'profit_loss': entry.profit_loss,
                        'confidence': entry.confidence_level,
                        'execution_quality': entry.execution_quality
                    })
                
                df = pd.DataFrame(data)
                return df.to_csv(index=False)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting journal data: {e}")
            return ""
    
    def manual_generate_insights(self):
        """Manually trigger insight generation"""
        try:
            self._generate_insights()
            self._save_data()
            logger.info("Manual insight generation completed")
            return True
        except Exception as e:
            logger.error(f"Error in manual insight generation: {e}")
            return False
    
    def shutdown(self):
        """Shutdown the trade journal system"""
        try:
            self._save_data()
            logger.info("Trade journal system shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")