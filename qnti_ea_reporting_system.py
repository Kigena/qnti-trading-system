#!/usr/bin/env python3
"""
QNTI EA Reporting System - Comprehensive Performance Monitoring and Analytics
Advanced reporting, visualization, and monitoring for EA generation and performance
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
import json
from pathlib import Path
import base64
from io import BytesIO
import threading
import time

# Plotting imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    plt.style.use('default')
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logging.warning("Matplotlib not available - plotting disabled")

# Import from EA systems
from qnti_ea_generation_engine import EATemplate
from qnti_ea_optimization_engine import BacktestMetrics, OptimizationResult
from qnti_ea_robustness_testing import RobustnessReport

logger = logging.getLogger('QNTI_EA_REPORTING')

@dataclass
class ReportConfiguration:
    """Configuration for report generation"""
    include_charts: bool = True
    include_detailed_analysis: bool = True
    include_recommendations: bool = True
    chart_style: str = "professional"  # professional, minimal, colorful
    chart_dpi: int = 300
    report_format: str = "html"  # html, pdf, json
    
    # Chart specifications
    figure_width: float = 12.0
    figure_height: float = 8.0
    color_palette: str = "viridis"
    
    # Content sections
    sections: List[str] = field(default_factory=lambda: [
        "executive_summary",
        "optimization_results", 
        "robustness_analysis",
        "performance_metrics",
        "risk_analysis",
        "recommendations"
    ])

@dataclass
class EAPerformanceReport:
    """Comprehensive EA performance report"""
    ea_id: str
    ea_name: str
    report_timestamp: datetime
    
    # Core data
    optimization_result: Optional[OptimizationResult] = None
    robustness_report: Optional[RobustnessReport] = None
    backtest_metrics: Optional[BacktestMetrics] = None
    
    # Analysis results
    performance_score: float = 0.0
    risk_score: float = 0.0
    overall_rating: str = "Unknown"
    
    # Report sections
    executive_summary: str = ""
    detailed_analysis: Dict[str, Any] = field(default_factory=dict)
    charts: Dict[str, str] = field(default_factory=dict)  # Chart name -> base64 image
    recommendations: List[str] = field(default_factory=list)
    
    # Metadata
    generation_time: float = 0.0
    report_version: str = "1.0"

class PerformanceAnalyzer:
    """Advanced performance analysis for EAs"""
    
    def __init__(self):
        self.analysis_methods = {
            'return_analysis': self._analyze_returns,
            'risk_analysis': self._analyze_risk,
            'drawdown_analysis': self._analyze_drawdowns,
            'trade_analysis': self._analyze_trade_patterns,
            'stability_analysis': self._analyze_stability,
            'market_regime_analysis': self._analyze_market_regimes
        }
    
    def analyze_performance(self, backtest_metrics: BacktestMetrics,
                          robustness_report: RobustnessReport = None) -> Dict[str, Any]:
        """Comprehensive performance analysis"""
        
        analysis = {}
        
        for method_name, method in self.analysis_methods.items():
            try:
                result = method(backtest_metrics, robustness_report)
                analysis[method_name] = result
            except Exception as e:
                logger.error(f"Error in {method_name}: {e}")
                analysis[method_name] = {"error": str(e)}
        
        # Calculate composite scores
        analysis['composite_scores'] = self._calculate_composite_scores(analysis)
        
        return analysis
    
    def _analyze_returns(self, metrics: BacktestMetrics, robustness: RobustnessReport = None) -> Dict[str, Any]:
        """Analyze return characteristics"""
        return {
            'annual_return': metrics.annual_return,
            'total_return': metrics.total_return,
            'return_consistency': self._calculate_return_consistency(metrics, robustness),
            'return_quality_score': self._score_return_quality(metrics),
            'benchmark_comparison': self._compare_to_benchmark(metrics)
        }
    
    def _analyze_risk(self, metrics: BacktestMetrics, robustness: RobustnessReport = None) -> Dict[str, Any]:
        """Analyze risk characteristics"""
        
        # Basic risk metrics
        risk_analysis = {
            'volatility': metrics.volatility,
            'sharpe_ratio': metrics.sharpe_ratio,
            'sortino_ratio': metrics.sortino_ratio,
            'max_drawdown': metrics.max_drawdown,
            'calmar_ratio': metrics.calmar_ratio
        }
        
        # Advanced risk metrics
        if robustness:
            risk_analysis.update({
                'worst_case_scenario': robustness.worst_case_drawdown,
                'var_99': robustness.value_at_risk_99,
                'tail_risk_score': self._calculate_tail_risk_score(robustness),
                'stress_test_resilience': self._calculate_stress_resilience(robustness)
            })
        
        # Risk score
        risk_analysis['composite_risk_score'] = self._calculate_risk_score(risk_analysis)
        
        return risk_analysis
    
    def _analyze_drawdowns(self, metrics: BacktestMetrics, robustness: RobustnessReport = None) -> Dict[str, Any]:
        """Analyze drawdown patterns"""
        return {
            'max_drawdown': metrics.max_drawdown,
            'average_drawdown': metrics.max_drawdown * 0.4,  # Estimated
            'drawdown_frequency': self._estimate_drawdown_frequency(metrics),
            'recovery_analysis': self._analyze_recovery_patterns(metrics),
            'drawdown_severity_score': self._score_drawdown_severity(metrics)
        }
    
    def _analyze_trade_patterns(self, metrics: BacktestMetrics, robustness: RobustnessReport = None) -> Dict[str, Any]:
        """Analyze trading pattern characteristics"""
        return {
            'win_rate': metrics.win_rate,
            'profit_factor': metrics.profit_factor,
            'avg_win_loss_ratio': abs(metrics.avg_win / metrics.avg_loss) if metrics.avg_loss != 0 else 0,
            'trade_frequency': self._calculate_trade_frequency(metrics),
            'trade_consistency': self._analyze_trade_consistency(metrics),
            'trade_quality_score': self._score_trade_quality(metrics)
        }
    
    def _analyze_stability(self, metrics: BacktestMetrics, robustness: RobustnessReport = None) -> Dict[str, Any]:
        """Analyze performance stability"""
        stability_score = 75.0  # Default
        
        if robustness:
            # Use robustness test results
            wf_stability = robustness.walk_forward_results.get('stability_score', 75)
            mc_consistency = robustness.monte_carlo_results.get('consistency_score', 75)
            param_sensitivity = 100 - robustness.parameter_sensitivity_score
            
            stability_score = np.mean([wf_stability, mc_consistency, param_sensitivity])
        
        return {
            'overall_stability_score': stability_score,
            'temporal_stability': robustness.walk_forward_results.get('stability_score', 75) if robustness else 75,
            'parameter_robustness': 100 - robustness.parameter_sensitivity_score if robustness else 75,
            'noise_resistance': robustness.monte_carlo_results.get('consistency_score', 75) if robustness else 75
        }
    
    def _analyze_market_regimes(self, metrics: BacktestMetrics, robustness: RobustnessReport = None) -> Dict[str, Any]:
        """Analyze performance across market regimes"""
        
        regime_analysis = {
            'trending_performance': 80.0,  # Mock values
            'sideways_performance': 60.0,
            'volatile_performance': 70.0,
            'regime_adaptability_score': 70.0
        }
        
        if robustness and robustness.stress_test_results:
            # Extract regime performance from stress tests
            stress_results = robustness.stress_test_results.get('scenario_results', {})
            
            regime_analysis['trending_performance'] = stress_results.get('trending_market', {}).get('score', 70)
            regime_analysis['sideways_performance'] = stress_results.get('sideways_market', {}).get('score', 70)
            regime_analysis['volatile_performance'] = stress_results.get('high_volatility', {}).get('score', 70)
            
            regime_scores = [regime_analysis['trending_performance'], 
                           regime_analysis['sideways_performance'],
                           regime_analysis['volatile_performance']]
            regime_analysis['regime_adaptability_score'] = np.mean(regime_scores)
        
        return regime_analysis
    
    def _calculate_composite_scores(self, analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate composite performance scores"""
        
        # Extract key metrics
        annual_return = analysis.get('return_analysis', {}).get('annual_return', 0)
        sharpe_ratio = analysis.get('risk_analysis', {}).get('sharpe_ratio', 0)
        max_drawdown = analysis.get('risk_analysis', {}).get('max_drawdown', 0)
        stability_score = analysis.get('stability_analysis', {}).get('overall_stability_score', 0)
        
        # Performance score (0-100)
        return_score = min(100, max(0, annual_return * 100 + 50))
        risk_adj_score = min(100, max(0, sharpe_ratio * 20 + 50))
        drawdown_score = min(100, max(0, 100 + max_drawdown * 200))
        
        performance_score = np.mean([return_score, risk_adj_score, drawdown_score])
        
        # Risk score (0-100, lower is better risk)
        volatility_score = max(0, 100 - abs(annual_return / max(0.01, abs(max_drawdown))) * 10)
        tail_risk_score = analysis.get('risk_analysis', {}).get('tail_risk_score', 50)
        
        risk_score = np.mean([volatility_score, tail_risk_score])
        
        return {
            'performance_score': performance_score,
            'risk_score': risk_score,
            'stability_score': stability_score,
            'overall_score': np.mean([performance_score, 100 - risk_score, stability_score])
        }
    
    # Helper methods for calculations
    def _calculate_return_consistency(self, metrics: BacktestMetrics, robustness: RobustnessReport = None) -> float:
        if robustness and robustness.walk_forward_results:
            return robustness.walk_forward_results.get('stability_score', 50)
        return 50.0  # Default moderate consistency
    
    def _score_return_quality(self, metrics: BacktestMetrics) -> float:
        # Quality based on risk-adjusted returns
        if metrics.volatility > 0:
            return min(100, metrics.sharpe_ratio * 20 + 50)
        return 50.0
    
    def _compare_to_benchmark(self, metrics: BacktestMetrics) -> Dict[str, float]:
        # Simplified benchmark comparison (assuming 5% annual benchmark)
        benchmark_return = 0.05
        outperformance = metrics.annual_return - benchmark_return
        return {
            'benchmark_return': benchmark_return,
            'outperformance': outperformance,
            'relative_sharpe': metrics.sharpe_ratio / 1.0  # Assuming benchmark Sharpe of 1.0
        }
    
    def _calculate_tail_risk_score(self, robustness: RobustnessReport) -> float:
        # Score based on worst-case scenarios
        if robustness.value_at_risk_99 != 0:
            return max(0, 100 + robustness.value_at_risk_99 * 500)  # Convert VaR to score
        return 50.0
    
    def _calculate_stress_resilience(self, robustness: RobustnessReport) -> float:
        if robustness.stress_test_results:
            return robustness.stress_test_results.get('stress_resistance_score', 50)
        return 50.0
    
    def _calculate_risk_score(self, risk_analysis: Dict[str, Any]) -> float:
        # Composite risk score
        sharpe = risk_analysis.get('sharpe_ratio', 0)
        drawdown = risk_analysis.get('max_drawdown', 0)
        
        sharpe_score = min(100, max(0, sharpe * 25 + 50))
        drawdown_score = min(100, max(0, 100 + drawdown * 200))
        
        return np.mean([sharpe_score, drawdown_score])
    
    def _estimate_drawdown_frequency(self, metrics: BacktestMetrics) -> float:
        # Estimate based on volatility and max drawdown
        if metrics.volatility > 0:
            return abs(metrics.max_drawdown) / metrics.volatility
        return 1.0
    
    def _analyze_recovery_patterns(self, metrics: BacktestMetrics) -> Dict[str, Any]:
        # Simplified recovery analysis
        return {
            'estimated_recovery_time': abs(metrics.max_drawdown) / max(0.01, metrics.annual_return) if metrics.annual_return > 0 else float('inf'),
            'recovery_strength': min(100, metrics.annual_return / abs(metrics.max_drawdown) * 100) if metrics.max_drawdown != 0 else 100
        }
    
    def _score_drawdown_severity(self, metrics: BacktestMetrics) -> float:
        # Score drawdown severity (0-100, higher is better)
        return max(0, 100 + metrics.max_drawdown * 200)
    
    def _calculate_trade_frequency(self, metrics: BacktestMetrics) -> float:
        # Estimate trades per month
        return metrics.total_trades / 12.0 if metrics.total_trades > 0 else 0
    
    def _analyze_trade_consistency(self, metrics: BacktestMetrics) -> Dict[str, float]:
        return {
            'win_loss_balance': abs(metrics.win_rate - 0.5) * 2,  # How far from 50/50
            'profit_consistency': metrics.profit_factor if metrics.profit_factor > 0 else 0
        }
    
    def _score_trade_quality(self, metrics: BacktestMetrics) -> float:
        # Overall trade quality score
        win_rate_score = metrics.win_rate * 100
        profit_factor_score = min(100, metrics.profit_factor * 25)
        
        return np.mean([win_rate_score, profit_factor_score])

class ChartGenerator:
    """Generate visualizations for EA performance"""
    
    def __init__(self, config: ReportConfiguration):
        self.config = config
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting not available - charts will be disabled")
    
    def generate_all_charts(self, metrics: BacktestMetrics, 
                           analysis: Dict[str, Any],
                           robustness: RobustnessReport = None) -> Dict[str, str]:
        """Generate all charts for the report"""
        
        if not PLOTTING_AVAILABLE:
            return {}
        
        charts = {}
        
        try:
            # Performance overview chart
            charts['performance_overview'] = self._create_performance_overview(metrics, analysis)
            
            # Risk-return scatter
            charts['risk_return'] = self._create_risk_return_chart(metrics, analysis)
            
            # Drawdown analysis
            charts['drawdown_analysis'] = self._create_drawdown_chart(metrics)
            
            # Performance breakdown
            charts['performance_breakdown'] = self._create_performance_breakdown(analysis)
            
            # Robustness testing results
            if robustness:
                charts['robustness_summary'] = self._create_robustness_chart(robustness)
                charts['sensitivity_analysis'] = self._create_sensitivity_chart(robustness)
            
        except Exception as e:
            logger.error(f"Error generating charts: {e}")
        
        return charts
    
    def _create_performance_overview(self, metrics: BacktestMetrics, analysis: Dict[str, Any]) -> str:
        """Create performance overview chart"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(self.config.figure_width, self.config.figure_height))
        
        # Performance metrics bar chart
        perf_metrics = ['Annual Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate']
        perf_values = [metrics.annual_return * 100, metrics.sharpe_ratio * 10, 
                      metrics.max_drawdown * 100, metrics.win_rate * 100]
        
        bars = ax1.bar(perf_metrics, perf_values, color=['green', 'blue', 'red', 'orange'])
        ax1.set_title('Key Performance Metrics')
        ax1.set_ylabel('Value (%)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Risk metrics radar chart (simplified as bar chart)
        risk_metrics = ['Volatility', 'Drawdown', 'Tail Risk', 'Stability']
        risk_scores = [
            100 - min(100, metrics.volatility * 1000),  # Invert volatility
            100 + metrics.max_drawdown * 100,  # Invert drawdown
            analysis.get('risk_analysis', {}).get('tail_risk_score', 50),
            analysis.get('stability_analysis', {}).get('overall_stability_score', 50)
        ]
        
        ax2.bar(risk_metrics, risk_scores, color='lightcoral')
        ax2.set_title('Risk Assessment')
        ax2.set_ylabel('Score (0-100)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Trade analysis
        trade_data = ['Total Trades', 'Winning Trades', 'Losing Trades']
        trade_counts = [metrics.total_trades, metrics.winning_trades, metrics.losing_trades]
        
        ax3.bar(trade_data, trade_counts, color=['blue', 'green', 'red'])
        ax3.set_title('Trade Analysis')
        ax3.set_ylabel('Count')
        
        # Return distribution (simplified)
        ax4.hist([metrics.annual_return], bins=1, alpha=0.7, color='green', edgecolor='black')
        ax4.axvline(x=0, color='red', linestyle='--', label='Break-even')
        ax4.set_title('Return Distribution')
        ax4.set_xlabel('Annual Return')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_risk_return_chart(self, metrics: BacktestMetrics, analysis: Dict[str, Any]) -> str:
        """Create risk-return scatter plot"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot EA point
        ax.scatter(metrics.volatility * 100, metrics.annual_return * 100, 
                  s=200, c='red', alpha=0.7, label='EA Performance')
        
        # Add benchmark points (for reference)
        benchmarks = {
            'Conservative': (5, 3),
            'Moderate': (10, 7),
            'Aggressive': (20, 12),
            'Market': (15, 8)
        }
        
        for name, (vol, ret) in benchmarks.items():
            ax.scatter(vol, ret, s=100, alpha=0.6, label=name)
        
        # Add efficient frontier line (simplified)
        vol_range = np.linspace(0, 25, 100)
        efficient_return = np.sqrt(vol_range) * 2  # Simplified efficient frontier
        ax.plot(vol_range, efficient_return, '--', alpha=0.5, label='Efficient Frontier')
        
        ax.set_xlabel('Volatility (%)')
        ax.set_ylabel('Annual Return (%)')
        ax.set_title('Risk-Return Profile')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return self._fig_to_base64(fig)
    
    def _create_drawdown_chart(self, metrics: BacktestMetrics) -> str:
        """Create drawdown analysis chart"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Simulated equity curve and drawdowns
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        
        # Generate synthetic equity curve
        np.random.seed(42)  # For reproducibility
        daily_returns = np.random.normal(metrics.annual_return/252, metrics.volatility/np.sqrt(252), 252)
        equity_curve = 10000 * np.cumprod(1 + daily_returns)
        
        # Calculate drawdowns
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak * 100
        
        # Plot equity curve
        ax1.plot(dates, equity_curve, color='blue', linewidth=2)
        ax1.fill_between(dates, equity_curve, peak, alpha=0.3, color='red', 
                        where=(equity_curve < peak), label='Drawdown Periods')
        ax1.set_title('Equity Curve')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot drawdown
        ax2.fill_between(dates, drawdown, 0, alpha=0.6, color='red')
        ax2.axhline(y=metrics.max_drawdown * 100, color='darkred', linestyle='--', 
                   label=f'Max Drawdown: {metrics.max_drawdown:.1%}')
        ax2.set_title('Drawdown Analysis')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_performance_breakdown(self, analysis: Dict[str, Any]) -> str:
        """Create performance breakdown chart"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Return analysis
        return_data = analysis.get('return_analysis', {})
        return_metrics = ['Annual Return', 'Return Quality', 'Consistency']
        return_values = [
            return_data.get('annual_return', 0) * 100,
            return_data.get('return_quality_score', 0),
            return_data.get('return_consistency', 0)
        ]
        
        ax1.bar(return_metrics, return_values, color='green', alpha=0.7)
        ax1.set_title('Return Analysis')
        ax1.set_ylabel('Score/Value')
        ax1.tick_params(axis='x', rotation=45)
        
        # Risk breakdown
        risk_data = analysis.get('risk_analysis', {})
        risk_metrics = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio']
        risk_values = [
            risk_data.get('sharpe_ratio', 0),
            risk_data.get('sortino_ratio', 0),
            risk_data.get('calmar_ratio', 0)
        ]
        
        ax2.bar(risk_metrics, risk_values, color='orange', alpha=0.7)
        ax2.set_title('Risk-Adjusted Metrics')
        ax2.set_ylabel('Ratio')
        ax2.tick_params(axis='x', rotation=45)
        
        # Trade analysis
        trade_data = analysis.get('trade_analysis', {})
        ax3.pie([trade_data.get('win_rate', 0.5), 1 - trade_data.get('win_rate', 0.5)], 
               labels=['Winning Trades', 'Losing Trades'], 
               colors=['green', 'red'], autopct='%1.1f%%')
        ax3.set_title('Win/Loss Distribution')
        
        # Stability scores
        stability_data = analysis.get('stability_analysis', {})
        stability_metrics = ['Temporal', 'Parameter', 'Noise Resistance']
        stability_scores = [
            stability_data.get('temporal_stability', 50),
            stability_data.get('parameter_robustness', 50),
            stability_data.get('noise_resistance', 50)
        ]
        
        ax4.bar(stability_metrics, stability_scores, color='blue', alpha=0.7)
        ax4.set_title('Stability Analysis')
        ax4.set_ylabel('Score (0-100)')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_robustness_chart(self, robustness: RobustnessReport) -> str:
        """Create robustness testing summary chart"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Overall robustness scores
        test_names = ['Walk Forward', 'Monte Carlo', 'Sensitivity', 'Stress Tests']
        test_scores = [
            robustness.walk_forward_results.get('stability_score', 0),
            robustness.monte_carlo_results.get('consistency_score', 0),
            100 - robustness.parameter_sensitivity_score,
            robustness.stress_test_results.get('average_score', 0)
        ]
        
        bars = ax1.bar(test_names, test_scores, color=['blue', 'green', 'orange', 'red'])
        ax1.set_title('Robustness Test Results')
        ax1.set_ylabel('Score (0-100)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add score labels on bars
        for bar, score in zip(bars, test_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{score:.1f}', ha='center', va='bottom')
        
        # Stress test scenarios
        if robustness.stress_test_results and 'scenario_results' in robustness.stress_test_results:
            scenarios = robustness.stress_test_results['scenario_results']
            scenario_names = list(scenarios.keys())
            scenario_scores = [scenarios[name].get('score', 0) for name in scenario_names]
            
            ax2.barh(scenario_names, scenario_scores, color='lightcoral')
            ax2.set_title('Stress Test Scenarios')
            ax2.set_xlabel('Score (0-100)')
        
        # Parameter sensitivity (top sensitive parameters)
        if robustness.sensitivity_results and 'sensitivity_scores' in robustness.sensitivity_results:
            sens_scores = robustness.sensitivity_results['sensitivity_scores']
            top_sensitive = sorted(sens_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            
            if top_sensitive:
                param_names = [item[0] for item in top_sensitive]
                param_scores = [item[1] for item in top_sensitive]
                
                ax3.barh(param_names, param_scores, color='yellow')
                ax3.set_title('Most Sensitive Parameters')
                ax3.set_xlabel('Sensitivity Score')
        
        # Overall robustness radar-style chart (as bar chart)
        robustness_aspects = ['Stability', 'Consistency', 'Robustness', 'Adaptability']
        robustness_scores = [
            robustness.performance_stability,
            robustness.monte_carlo_results.get('consistency_score', 50),
            robustness.overall_robustness_score,
            robustness.market_regime_adaptability
        ]
        
        ax4.bar(robustness_aspects, robustness_scores, color='purple', alpha=0.7)
        ax4.set_title('Robustness Profile')
        ax4.set_ylabel('Score (0-100)')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_sensitivity_chart(self, robustness: RobustnessReport) -> str:
        """Create parameter sensitivity analysis chart"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        if robustness.sensitivity_results and 'sensitivity_scores' in robustness.sensitivity_results:
            sens_scores = robustness.sensitivity_results['sensitivity_scores']
            
            # Parameter sensitivity distribution
            sensitivity_values = list(sens_scores.values())
            ax1.hist(sensitivity_values, bins=min(10, len(sensitivity_values)), 
                    alpha=0.7, color='orange', edgecolor='black')
            ax1.axvline(x=np.mean(sensitivity_values), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(sensitivity_values):.1f}')
            ax1.set_title('Parameter Sensitivity Distribution')
            ax1.set_xlabel('Sensitivity Score')
            ax1.set_ylabel('Frequency')
            ax1.legend()
            
            # Top 10 most sensitive parameters
            top_sensitive = sorted(sens_scores.items(), key=lambda x: x[1], reverse=True)[:10]
            
            if top_sensitive:
                param_names = [item[0] for item in top_sensitive]
                param_scores = [item[1] for item in top_sensitive]
                
                y_pos = np.arange(len(param_names))
                ax2.barh(y_pos, param_scores, color='red', alpha=0.6)
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels(param_names)
                ax2.set_title('Most Sensitive Parameters')
                ax2.set_xlabel('Sensitivity Score')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=self.config.chart_dpi, bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close(fig)
        
        graphic = base64.b64encode(image_png)
        return graphic.decode('utf-8')

class ReportGenerator:
    """Generate comprehensive EA performance reports"""
    
    def __init__(self, config: ReportConfiguration = None):
        self.config = config or ReportConfiguration()
        self.analyzer = PerformanceAnalyzer()
        self.chart_generator = ChartGenerator(self.config)
    
    def generate_report(self, ea_template: EATemplate,
                       backtest_metrics: BacktestMetrics,
                       optimization_result: OptimizationResult = None,
                       robustness_report: RobustnessReport = None) -> EAPerformanceReport:
        """Generate comprehensive performance report"""
        
        start_time = time.time()
        logger.info(f"Generating performance report for EA: {ea_template.name}")
        
        # Analyze performance
        analysis = self.analyzer.analyze_performance(backtest_metrics, robustness_report)
        
        # Generate charts
        charts = {}
        if self.config.include_charts:
            charts = self.chart_generator.generate_all_charts(
                backtest_metrics, analysis, robustness_report
            )
        
        # Create report
        report = EAPerformanceReport(
            ea_id=ea_template.id,
            ea_name=ea_template.name,
            report_timestamp=datetime.now(),
            optimization_result=optimization_result,
            robustness_report=robustness_report,
            backtest_metrics=backtest_metrics,
            charts=charts,
            detailed_analysis=analysis
        )
        
        # Calculate scores and rating
        composite_scores = analysis.get('composite_scores', {})
        report.performance_score = composite_scores.get('performance_score', 0)
        report.risk_score = composite_scores.get('risk_score', 0)
        report.overall_rating = self._calculate_rating(composite_scores.get('overall_score', 0))
        
        # Generate executive summary
        if self.config.include_detailed_analysis:
            report.executive_summary = self._generate_executive_summary(report, analysis)
        
        # Generate recommendations
        if self.config.include_recommendations:
            report.recommendations = self._generate_recommendations(report, analysis)
        
        report.generation_time = time.time() - start_time
        
        logger.info(f"Report generated in {report.generation_time:.2f} seconds")
        return report
    
    def _calculate_rating(self, overall_score: float) -> str:
        """Calculate letter rating based on overall score"""
        if overall_score >= 85:
            return "A+ (Excellent)"
        elif overall_score >= 75:
            return "A (Very Good)"
        elif overall_score >= 65:
            return "B+ (Good)"
        elif overall_score >= 55:
            return "B (Acceptable)"
        elif overall_score >= 45:
            return "C+ (Below Average)"
        elif overall_score >= 35:
            return "C (Poor)"
        else:
            return "D (Very Poor)"
    
    def _generate_executive_summary(self, report: EAPerformanceReport, analysis: Dict[str, Any]) -> str:
        """Generate executive summary"""
        
        metrics = report.backtest_metrics
        
        summary = f"""
# Executive Summary - {report.ea_name}

## Overall Assessment
- **Rating**: {report.overall_rating}
- **Performance Score**: {report.performance_score:.1f}/100
- **Risk Score**: {report.risk_score:.1f}/100

## Key Performance Metrics
- **Annual Return**: {metrics.annual_return:.2%}
- **Sharpe Ratio**: {metrics.sharpe_ratio:.2f}
- **Maximum Drawdown**: {metrics.max_drawdown:.2%}
- **Win Rate**: {metrics.win_rate:.2%}
- **Profit Factor**: {metrics.profit_factor:.2f}

## Trading Statistics
- **Total Trades**: {metrics.total_trades}
- **Winning Trades**: {metrics.winning_trades}
- **Losing Trades**: {metrics.losing_trades}
- **Average Win**: ${metrics.avg_win:.2f}
- **Average Loss**: ${metrics.avg_loss:.2f}

## Risk Assessment
The strategy demonstrates {'high' if report.risk_score < 30 else 'moderate' if report.risk_score < 60 else 'low'} risk characteristics 
with a maximum drawdown of {metrics.max_drawdown:.2%} and volatility-adjusted returns 
{'above' if metrics.sharpe_ratio > 1.0 else 'below'} market expectations.

## Robustness Evaluation
{'Comprehensive robustness testing shows ' + self._summarize_robustness(report.robustness_report) if report.robustness_report else 'Robustness testing not available.'}

## Bottom Line
{self._generate_bottom_line_assessment(report, analysis)}
        """.strip()
        
        return summary
    
    def _summarize_robustness(self, robustness: RobustnessReport) -> str:
        """Summarize robustness test results"""
        if not robustness:
            return "no robustness data available."
        
        score = robustness.overall_robustness_score
        
        if score >= 70:
            return f"excellent robustness (score: {score:.1f}/100) across multiple market conditions and parameter variations."
        elif score >= 50:
            return f"moderate robustness (score: {score:.1f}/100) with some sensitivity to market conditions."
        else:
            return f"limited robustness (score: {score:.1f}/100) indicating potential overfitting or instability."
    
    def _generate_bottom_line_assessment(self, report: EAPerformanceReport, analysis: Dict[str, Any]) -> str:
        """Generate bottom line assessment"""
        
        performance_score = report.performance_score
        risk_score = report.risk_score
        
        if performance_score >= 70 and risk_score >= 50:
            return "This strategy shows strong potential for live trading with appropriate risk management."
        elif performance_score >= 50:
            return "The strategy shows promise but requires further optimization and risk management enhancement."
        else:
            return "The strategy requires significant improvement before considering live deployment."
    
    def _generate_recommendations(self, report: EAPerformanceReport, analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Performance-based recommendations
        if report.backtest_metrics.annual_return < 0.05:
            recommendations.append(
                "Consider enhancing entry/exit logic to improve return generation"
            )
        
        if report.backtest_metrics.sharpe_ratio < 1.0:
            recommendations.append(
                "Improve risk-adjusted returns by optimizing position sizing or reducing volatility"
            )
        
        if report.backtest_metrics.max_drawdown < -0.2:
            recommendations.append(
                "Implement stricter drawdown controls - current maximum drawdown exceeds acceptable levels"
            )
        
        if report.backtest_metrics.win_rate < 0.4:
            recommendations.append(
                "Review entry criteria to improve win rate - current level may indicate poor signal quality"
            )
        
        # Robustness-based recommendations
        if report.robustness_report:
            if report.robustness_report.overall_robustness_score < 50:
                recommendations.append(
                    "Strategy shows poor robustness - consider parameter stabilization or logic simplification"
                )
            
            if report.robustness_report.parameter_sensitivity_score > 70:
                recommendations.append(
                    "High parameter sensitivity detected - fix sensitive parameters or add robustness constraints"
                )
        
        # Risk management recommendations
        if report.risk_score < 40:
            recommendations.append(
                "Implement enhanced risk management controls including stop-losses and position sizing limits"
            )
        
        # General recommendations
        if len(recommendations) == 0:
            recommendations.append(
                "Strategy meets basic performance criteria - monitor closely during initial live deployment"
            )
        
        return recommendations
    
    def save_report(self, report: EAPerformanceReport, filepath: str):
        """Save report to file"""
        
        if self.config.report_format == "json":
            self._save_json_report(report, filepath)
        elif self.config.report_format == "html":
            self._save_html_report(report, filepath)
        else:
            logger.warning(f"Unsupported report format: {self.config.report_format}")
    
    def _save_json_report(self, report: EAPerformanceReport, filepath: str):
        """Save report as JSON"""
        report_dict = asdict(report)
        
        # Convert datetime objects to strings
        report_dict['report_timestamp'] = report.report_timestamp.isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
    
    def _save_html_report(self, report: EAPerformanceReport, filepath: str):
        """Save report as HTML"""
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>EA Performance Report - {report.ea_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .chart {{ text-align: center; margin: 20px 0; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }}
        .metric-box {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; text-align: center; }}
        .recommendations {{ background-color: #fffacd; padding: 15px; border-radius: 5px; }}
        img {{ max-width: 100%; height: auto; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>EA Performance Report</h1>
        <h2>{report.ea_name}</h2>
        <p>Generated on: {report.report_timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Overall Rating: <strong>{report.overall_rating}</strong></p>
    </div>
    
    <div class="section">
        <h3>Executive Summary</h3>
        <pre>{report.executive_summary}</pre>
    </div>
    
    <div class="section">
        <h3>Key Metrics</h3>
        <div class="metrics">
            <div class="metric-box">
                <h4>Performance Score</h4>
                <p><strong>{report.performance_score:.1f}/100</strong></p>
            </div>
            <div class="metric-box">
                <h4>Risk Score</h4>
                <p><strong>{report.risk_score:.1f}/100</strong></p>
            </div>
            <div class="metric-box">
                <h4>Annual Return</h4>
                <p><strong>{report.backtest_metrics.annual_return:.2%}</strong></p>
            </div>
            <div class="metric-box">
                <h4>Sharpe Ratio</h4>
                <p><strong>{report.backtest_metrics.sharpe_ratio:.2f}</strong></p>
            </div>
        </div>
    </div>
        """
        
        # Add charts
        for chart_name, chart_data in report.charts.items():
            html_content += f"""
    <div class="section">
        <h3>{chart_name.replace('_', ' ').title()}</h3>
        <div class="chart">
            <img src="data:image/png;base64,{chart_data}" alt="{chart_name}">
        </div>
    </div>
            """
        
        # Add recommendations
        if report.recommendations:
            html_content += """
    <div class="section recommendations">
        <h3>Recommendations</h3>
        <ul>
            """
            for rec in report.recommendations:
                html_content += f"<li>{rec}</li>"
            
            html_content += """
        </ul>
    </div>
            """
        
        html_content += """
</body>
</html>
        """
        
        with open(filepath, 'w') as f:
            f.write(html_content)

# Example usage
if __name__ == "__main__":
    from qnti_ea_generation_engine import EAGenerationEngine
    
    # Create sample data
    engine = EAGenerationEngine()
    ea = engine.create_ea_template(
        name="Sample Strategy",
        description="Test strategy for reporting",
        author="QNTI"
    )
    
    # Sample metrics
    metrics = BacktestMetrics(
        annual_return=0.15,
        sharpe_ratio=1.2,
        max_drawdown=-0.08,
        win_rate=0.65,
        profit_factor=1.8,
        total_trades=150,
        winning_trades=98,
        losing_trades=52
    )
    
    # Generate report
    config = ReportConfiguration(include_charts=True)
    generator = ReportGenerator(config)
    
    report = generator.generate_report(ea, metrics)
    
    # Save report
    generator.save_report(report, "sample_ea_report.html")
    
    print(f"Report generated with rating: {report.overall_rating}")
    print(f"Performance score: {report.performance_score:.1f}")
    print(f"Generated {len(report.charts)} charts")