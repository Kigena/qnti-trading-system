"""
QNTI Intelligent EA Management System

This module integrates EA profiling with real-time management to provide
AI-powered optimization recommendations and automated EA control based on
market conditions and EA characteristics.
"""

import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from qnti_ea_profiling_system import (
    QNTIEAProfiler, StrategyType, IndicatorType, RiskLevel,
    EAProfile, EntryCondition, ExitCondition, RiskManagement
)

logger = logging.getLogger(__name__)

class MarketCondition:
    """Current market condition analysis"""
    def __init__(self):
        self.volatility = "normal"  # low, normal, high, extreme
        self.trend_strength = 0.5  # 0-1
        self.trend_direction = "sideways"  # up, down, sideways
        self.session = "overlap"  # london, new_york, tokyo, sydney, overlap
        self.spread_level = "normal"  # tight, normal, wide
        self.news_proximity = False  # Is major news event within 1 hour
        self.correlation_strength = 0.5  # Cross-pair correlation
        
class QNTIIntelligentEAManager:
    """Intelligent EA Management with AI-powered optimization"""
    
    def __init__(self, trade_manager, mt5_bridge, data_dir: str = "qnti_data"):
        self.trade_manager = trade_manager
        self.mt5_bridge = mt5_bridge
        self.data_dir = Path(data_dir)
        
        # Initialize EA profiler
        self.profiler = QNTIEAProfiler(data_dir)
        
        # Current market analysis
        self.market_condition = MarketCondition()
        
        # Optimization settings
        self.optimization_file = self.data_dir / "ea_optimizations.json"
        self.optimization_history: Dict[str, List] = {}
        
        # Load optimization history
        self._load_optimization_history()
        
        # Auto-profiling settings
        self.auto_profiling_enabled = True
        self.mql_files_dir = Path("MQL5/Files")  # Default MQL files directory
        
        logger.info("Intelligent EA Manager initialized")

    def _load_optimization_history(self):
        """Load optimization history from file"""
        try:
            if self.optimization_file.exists():
                with open(self.optimization_file, 'r') as f:
                    self.optimization_history = json.load(f)
        except Exception as e:
            logger.error(f"Error loading optimization history: {e}")

    def _save_optimization_history(self):
        """Save optimization history to file"""
        try:
            with open(self.optimization_file, 'w') as f:
                json.dump(self.optimization_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving optimization history: {e}")

    def analyze_current_market(self, symbol: str = "EURUSD") -> MarketCondition:
        """Analyze current market conditions"""
        try:
            # Get market data from MT5 bridge
            if not self.mt5_bridge or not hasattr(self.mt5_bridge, 'symbols'):
                return self.market_condition
            
            symbol_data = None
            for sym in self.mt5_bridge.symbols.values():
                if sym.symbol == symbol:
                    symbol_data = sym
                    break
            
            if not symbol_data:
                return self.market_condition
            
            # Analyze volatility (using spread and price movement)
            current_spread = symbol_data.spread
            if current_spread < 15:
                self.market_condition.spread_level = "tight"
            elif current_spread > 30:
                self.market_condition.spread_level = "wide"
            else:
                self.market_condition.spread_level = "normal"
            
            # Simple volatility analysis based on price movement
            price_change = abs((symbol_data.bid - symbol_data.ask) / symbol_data.bid)
            if price_change > 0.001:  # 0.1%
                self.market_condition.volatility = "high"
            elif price_change < 0.0003:  # 0.03%
                self.market_condition.volatility = "low"
            else:
                self.market_condition.volatility = "normal"
            
            # Determine trading session
            current_hour = datetime.now().hour
            if 8 <= current_hour < 16:
                self.market_condition.session = "london"
            elif 13 <= current_hour < 22:
                self.market_condition.session = "new_york"
            elif 0 <= current_hour < 9:
                self.market_condition.session = "tokyo"
            else:
                self.market_condition.session = "overlap"
            
            logger.debug(f"Market analysis: {self.market_condition.volatility} volatility, "
                        f"{self.market_condition.session} session")
            
            return self.market_condition
            
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {e}")
            return self.market_condition

    def auto_profile_eas(self):
        """Automatically profile EAs from MQL files and existing EA data"""
        try:
            profiled_count = 0
            
            # Profile from existing EA performance data
            if hasattr(self.trade_manager, 'ea_performances'):
                for ea_name, performance in self.trade_manager.ea_performances.items():
                    if ea_name not in self.profiler.profiles:
                        # Create basic profile from performance data
                        profile = self.profiler.create_profile(
                            ea_name=ea_name,
                            magic_number=performance.magic_number,
                            symbol=performance.symbol,
                            description=f"Auto-profiled from performance data"
                        )
                        
                        # Try to deduce strategy type from name
                        ea_name_lower = ea_name.lower()
                        if 'scalp' in ea_name_lower:
                            profile.strategy_type = StrategyType.SCALPING
                        elif 'trend' in ea_name_lower:
                            profile.strategy_type = StrategyType.TREND_FOLLOWING
                        elif 'grid' in ea_name_lower:
                            profile.strategy_type = StrategyType.GRID_TRADING
                        elif 'hedge' in ea_name_lower:
                            profile.strategy_type = StrategyType.HEDGING
                        elif 'martingale' in ea_name_lower or 'recovery' in ea_name_lower:
                            profile.strategy_type = StrategyType.MARTINGALE
                        
                        # Add basic risk management based on performance
                        if performance.risk_score > 7:
                            profile.risk_management.max_risk_per_trade = 0.01  # Conservative
                        elif performance.risk_score < 3:
                            profile.risk_management.max_risk_per_trade = 0.03  # More aggressive
                        
                        # Set confidence based on data availability
                        profile.confidence_score = 0.4  # Low confidence from limited data
                        
                        profiled_count += 1
                        logger.info(f"Auto-profiled EA: {ea_name}")
            
            # Look for MQL files to analyze
            if self.mql_files_dir.exists():
                for mql_file in self.mql_files_dir.glob("*.mq*"):
                    try:
                        with open(mql_file, 'r', encoding='utf-8', errors='ignore') as f:
                            mql_code = f.read()
                        
                        ea_name = mql_file.stem
                        if ea_name not in self.profiler.profiles:
                            profile = self.profiler.analyze_ea_from_mql_code(ea_name, mql_code)
                            if profile:
                                profiled_count += 1
                                logger.info(f"Profiled EA from MQL: {ea_name}")
                    except Exception as e:
                        logger.warning(f"Could not analyze MQL file {mql_file}: {e}")
            
            if profiled_count > 0:
                logger.info(f"Auto-profiled {profiled_count} EAs")
            
            return profiled_count
            
        except Exception as e:
            logger.error(f"Error in auto-profiling EAs: {e}")
            return 0

    def get_ai_recommendations(self, ea_name: str = None) -> List[Dict[str, Any]]:
        """Get comprehensive AI recommendations for EA management"""
        try:
            recommendations = []
            
            # Update market analysis
            self.analyze_current_market()
            
            # Get recommendations for specific EA or all EAs
            target_eas = [ea_name] if ea_name else list(self.profiler.profiles.keys())
            
            for ea in target_eas:
                profile = self.profiler.get_profile(ea)
                if not profile:
                    continue
                
                # Get market compatibility
                market_data = {
                    'volatility': self.market_condition.volatility,
                    'trend_strength': self.market_condition.trend_strength,
                    'session': self.market_condition.session,
                    'spread_level': self.market_condition.spread_level
                }
                
                compatibility = self.profiler.analyze_market_compatibility(ea, market_data)
                
                # Get profile-based recommendations
                profile_recommendations = self.profiler.get_ai_optimization_recommendations(
                    ea, self.market_condition.volatility
                )
                
                # Combine with performance-based recommendations
                performance = self.trade_manager.ea_performances.get(ea)
                if performance:
                    perf_recommendations = self._get_performance_recommendations(ea, performance)
                    profile_recommendations.extend(perf_recommendations)
                
                # Add market compatibility assessment
                for rec in profile_recommendations:
                    rec['ea_name'] = ea
                    rec['market_compatibility'] = compatibility['compatibility']
                    rec['market_reasoning'] = compatibility['reasons']
                    rec['urgency'] = self._calculate_urgency(rec, compatibility, performance)
                
                recommendations.extend(profile_recommendations)
            
            # Sort by urgency
            recommendations.sort(key=lambda x: x.get('urgency', 0.5), reverse=True)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting AI recommendations: {e}")
            return []

    def _get_performance_recommendations(self, ea_name: str, performance) -> List[Dict]:
        """Generate recommendations based on EA performance"""
        recommendations = []
        
        try:
            # Poor performance recommendations
            if performance.win_rate < 30 and performance.total_trades > 10:
                recommendations.append({
                    "type": "status_change",
                    "action": "pause",
                    "reason": f"Poor win rate: {performance.win_rate:.1f}%",
                    "urgency": 0.9
                })
            
            # High risk recommendations
            if performance.risk_score > 8:
                recommendations.append({
                    "type": "risk_reduction",
                    "action": "reduce_lot_size",
                    "value": 0.5,
                    "reason": f"High risk score: {performance.risk_score:.1f}",
                    "urgency": 0.8
                })
            
            # Excellent performance recommendations
            if performance.profit_factor > 2.0 and performance.win_rate > 60:
                recommendations.append({
                    "type": "performance_enhancement",
                    "action": "increase_allocation",
                    "value": 1.3,
                    "reason": f"Excellent performance: PF={performance.profit_factor:.2f}",
                    "urgency": 0.6
                })
            
            # Drawdown recommendations
            if performance.max_drawdown > 0.15:  # 15%
                recommendations.append({
                    "type": "risk_management",
                    "action": "implement_drawdown_limit",
                    "value": 0.10,
                    "reason": f"High drawdown: {performance.max_drawdown:.1%}",
                    "urgency": 0.7
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating performance recommendations: {e}")
            return []

    def _calculate_urgency(self, recommendation: Dict, compatibility: Dict, performance) -> float:
        """Calculate urgency score for recommendation"""
        try:
            urgency = 0.5  # Base urgency
            
            # High urgency for safety recommendations
            if recommendation.get('type') in ['status_change', 'risk_reduction']:
                urgency += 0.3
            
            # Market compatibility impact
            if compatibility['compatibility'] < 0.3:
                urgency += 0.2
            
            # Performance impact
            if performance:
                if performance.risk_score > 7:
                    urgency += 0.2
                if performance.total_profit < -100:  # Significant losses
                    urgency += 0.3
            
            return min(1.0, urgency)
            
        except Exception as e:
            logger.error(f"Error calculating urgency: {e}")
            return 0.5

    def apply_recommendation(self, recommendation: Dict) -> Dict[str, Any]:
        """Apply an AI recommendation to an EA"""
        try:
            ea_name = recommendation.get('ea_name')
            action = recommendation.get('action')
            
            if not ea_name or not action:
                return {"success": False, "error": "Missing EA name or action"}
            
            result = {"success": False, "applied_action": action, "ea_name": ea_name}
            
            # Apply different types of recommendations
            if action == "pause":
                success = self.trade_manager.control_ea(ea_name, "pause")
                result["success"] = success
                result["message"] = f"EA {ea_name} paused due to: {recommendation.get('reason', 'AI recommendation')}"
            
            elif action == "reduce_lot_size":
                multiplier = recommendation.get('value', 0.7)
                success = self.trade_manager.control_ea(ea_name, "optimize", {
                    "lot_multiplier": multiplier,
                    "optimization_type": "risk_reduction"
                })
                result["success"] = success
                result["message"] = f"Reduced lot size for {ea_name} by {multiplier}x"
            
            elif action == "increase_allocation":
                multiplier = recommendation.get('value', 1.2)
                success = self.trade_manager.control_ea(ea_name, "optimize", {
                    "lot_multiplier": multiplier,
                    "optimization_type": "performance_enhancement"
                })
                result["success"] = success
                result["message"] = f"Increased allocation for {ea_name} by {multiplier}x"
            
            elif action == "optimize_parameters":
                optimization_type = recommendation.get('optimization_type', 'general')
                success = self.trade_manager.control_ea(ea_name, "optimize", {
                    "optimization_type": optimization_type,
                    "auto_adjust": True
                })
                result["success"] = success
                result["message"] = f"Applied {optimization_type} optimization to {ea_name}"
            
            else:
                # Generic optimization
                success = self.trade_manager.control_ea(ea_name, action, 
                                                      recommendation.get('parameters', {}))
                result["success"] = success
                result["message"] = f"Applied {action} to {ea_name}"
            
            # Record optimization in history
            if result["success"]:
                self._record_optimization(ea_name, recommendation, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error applying recommendation: {e}")
            return {"success": False, "error": str(e)}

    def _record_optimization(self, ea_name: str, recommendation: Dict, result: Dict):
        """Record optimization action in history"""
        try:
            if ea_name not in self.optimization_history:
                self.optimization_history[ea_name] = []
            
            record = {
                "timestamp": datetime.now().isoformat(),
                "recommendation": recommendation,
                "result": result,
                "market_condition": {
                    "volatility": self.market_condition.volatility,
                    "session": self.market_condition.session,
                    "spread_level": self.market_condition.spread_level
                }
            }
            
            self.optimization_history[ea_name].append(record)
            
            # Keep only last 50 records per EA
            if len(self.optimization_history[ea_name]) > 50:
                self.optimization_history[ea_name] = self.optimization_history[ea_name][-50:]
            
            self._save_optimization_history()
            
        except Exception as e:
            logger.error(f"Error recording optimization: {e}")

    def bulk_apply_recommendations(self, recommendations: List[Dict], 
                                 max_applications: int = 5) -> List[Dict]:
        """Apply multiple recommendations with safety limits"""
        try:
            results = []
            applications = 0
            
            # Sort by urgency and apply most urgent first
            sorted_recs = sorted(recommendations, 
                               key=lambda x: x.get('urgency', 0.5), reverse=True)
            
            for rec in sorted_recs:
                if applications >= max_applications:
                    break
                
                # Skip if EA is already being optimized recently
                ea_name = rec.get('ea_name')
                if self._was_recently_optimized(ea_name):
                    continue
                
                result = self.apply_recommendation(rec)
                results.append(result)
                
                if result.get('success'):
                    applications += 1
            
            return results
            
        except Exception as e:
            logger.error(f"Error in bulk apply recommendations: {e}")
            return []

    def _was_recently_optimized(self, ea_name: str, hours: int = 2) -> bool:
        """Check if EA was optimized recently"""
        try:
            if ea_name not in self.optimization_history:
                return False
            
            recent_limit = datetime.now() - timedelta(hours=hours)
            
            for record in self.optimization_history[ea_name]:
                record_time = datetime.fromisoformat(record['timestamp'])
                if record_time > recent_limit:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking recent optimization: {e}")
            return False

    def create_ea_profile_manually(self, ea_data: Dict[str, Any]) -> bool:
        """Create EA profile manually from provided data"""
        try:
            ea_name = ea_data.get('name')
            if not ea_name:
                return False
            
            # Create basic profile
            profile = self.profiler.create_profile(
                ea_name=ea_name,
                magic_number=ea_data.get('magic_number', 0),
                symbol=ea_data.get('symbol', 'UNKNOWN'),
                strategy_type=ea_data.get('strategy_type', 'unknown'),
                description=ea_data.get('description', '')
            )
            
            # Add indicators if provided
            for ind_data in ea_data.get('indicators', []):
                from qnti_ea_profiling_system import IndicatorConfig
                
                indicator = IndicatorConfig(
                    name=ind_data.get('name', 'unknown'),
                    type=IndicatorType(ind_data.get('type', 'unknown')),
                    parameters=ind_data.get('parameters', {}),
                    timeframe=ind_data.get('timeframe', 'M15'),
                    weight=ind_data.get('weight', 1.0),
                    description=ind_data.get('description')
                )
                profile.indicators.append(indicator)
            
            # Add risk management
            if 'risk_management' in ea_data:
                rm_data = ea_data['risk_management']
                profile.risk_management = RiskManagement(
                    lot_sizing_method=rm_data.get('lot_sizing_method', 'fixed'),
                    lot_size=rm_data.get('lot_size', 0.01),
                    max_risk_per_trade=rm_data.get('max_risk_per_trade', 0.02),
                    max_open_trades=rm_data.get('max_open_trades', 1),
                    max_drawdown_limit=rm_data.get('max_drawdown_limit', 0.20)
                )
            
            # Set characteristics
            profile.is_grid_based = ea_data.get('is_grid_based', False)
            profile.is_martingale = ea_data.get('is_martingale', False)
            profile.uses_hedging = ea_data.get('uses_hedging', False)
            profile.is_news_sensitive = ea_data.get('is_news_sensitive', False)
            
            # Set AI understanding fields
            profile.strengths = ea_data.get('strengths', [])
            profile.weaknesses = ea_data.get('weaknesses', [])
            profile.best_market_conditions = ea_data.get('best_market_conditions', [])
            profile.worst_market_conditions = ea_data.get('worst_market_conditions', [])
            
            profile.confidence_score = ea_data.get('confidence_score', 0.8)
            
            self.profiler.profiles[ea_name] = profile
            self.profiler._save_profiles()
            
            logger.info(f"Created manual profile for {ea_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating manual EA profile: {e}")
            return False

    def get_ea_intelligence_summary(self) -> Dict[str, Any]:
        """Get comprehensive intelligence summary of all EAs"""
        try:
            summary = {
                "total_eas": len(self.profiler.profiles),
                "profiled_eas": len([p for p in self.profiler.profiles.values() if p.confidence_score > 0.5]),
                "strategy_distribution": {},
                "risk_distribution": {},
                "market_readiness": {},
                "optimization_opportunities": []
            }
            
            # Strategy distribution
            for profile in self.profiler.profiles.values():
                strategy = profile.strategy_type.value
                summary["strategy_distribution"][strategy] = summary["strategy_distribution"].get(strategy, 0) + 1
            
            # Risk distribution
            for profile in self.profiler.profiles.values():
                risk_level = "low" if profile.risk_management.max_risk_per_trade < 0.02 else \
                           "medium" if profile.risk_management.max_risk_per_trade < 0.05 else "high"
                summary["risk_distribution"][risk_level] = summary["risk_distribution"].get(risk_level, 0) + 1
            
            # Market readiness (compatibility with current market)
            current_market = {
                'volatility': self.market_condition.volatility,
                'trend_strength': self.market_condition.trend_strength
            }
            
            for profile in self.profiler.profiles.values():
                compatibility = self.profiler.analyze_market_compatibility(profile.ea_name, current_market)
                readiness = "ready" if compatibility['compatibility'] > 0.6 else \
                          "caution" if compatibility['compatibility'] > 0.3 else "avoid"
                summary["market_readiness"][readiness] = summary["market_readiness"].get(readiness, 0) + 1
            
            # Get top optimization opportunities
            recommendations = self.get_ai_recommendations()
            summary["optimization_opportunities"] = recommendations[:10]  # Top 10
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating intelligence summary: {e}")
            return {}

    def update_ea_profile_from_performance(self, ea_name: str) -> bool:
        """Update EA profile based on recent performance data"""
        try:
            performance = self.trade_manager.ea_performances.get(ea_name)
            if not performance:
                return False
            
            profile = self.profiler.get_profile(ea_name)
            if not profile:
                # Create new profile if doesn't exist
                profile = self.profiler.create_profile(
                    ea_name=ea_name,
                    magic_number=performance.magic_number,
                    symbol=performance.symbol
                )
            
            # Update characteristics based on performance patterns
            if performance.risk_score > 7:
                if "High risk trading pattern" not in (profile.weaknesses or []):
                    profile.weaknesses = (profile.weaknesses or []) + ["High risk trading pattern"]
            
            if performance.win_rate > 70:
                if "High win rate" not in (profile.strengths or []):
                    profile.strengths = (profile.strengths or []) + ["High win rate"]
            
            if performance.profit_factor > 2.0:
                if "Excellent profit factor" not in (profile.strengths or []):
                    profile.strengths = (profile.strengths or []) + ["Excellent profit factor"]
            
            # Update confidence based on trade count
            if performance.total_trades > 50:
                profile.confidence_score = min(1.0, profile.confidence_score + 0.2)
            
            profile.last_updated = datetime.now()
            self.profiler._save_profiles()
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating profile from performance: {e}")
            return False 