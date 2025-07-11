#!/usr/bin/env python3
"""
QNTI EA Robustness Testing Framework - Advanced Strategy Validation
Comprehensive robustness testing with walk-forward analysis, Monte Carlo simulation,
parameter sensitivity analysis, and stress testing
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
import threading
import time
import random
import copy
from abc import ABC, abstractmethod
import json
from pathlib import Path
import concurrent.futures
import warnings
from collections import defaultdict

# Import from main engine
from qnti_ea_generation_engine import EATemplate, Parameter, ParameterType, RobustnessTest, RobustnessTestResult
from qnti_ea_optimization_engine import BacktestMetrics

logger = logging.getLogger('QNTI_ROBUSTNESS')

@dataclass
class RobustnessConfig:
    """Configuration for robustness testing"""
    # Walk-forward analysis
    wf_window_size: int = 252  # Trading days (1 year)
    wf_step_size: int = 63     # Quarter step
    wf_min_samples: int = 100  # Minimum samples for training
    
    # Monte Carlo simulation
    mc_simulations: int = 1000
    mc_noise_level: float = 0.01  # 1% noise
    mc_randomize_order: bool = True
    mc_randomize_slippage: bool = True
    mc_slippage_range: Tuple[float, float] = (0.0001, 0.001)  # 0.1-1 pip
    
    # Parameter sensitivity
    sensitivity_steps: int = 10
    sensitivity_range: float = 0.2  # 20% variation around optimal
    
    # Stress testing
    stress_scenarios: List[str] = field(default_factory=lambda: [
        'high_volatility', 'low_volatility', 'trending_market', 
        'sideways_market', 'flash_crash', 'gap_scenarios'
    ])
    
    # General settings
    confidence_level: float = 0.95
    min_sample_size: int = 30
    parallel_processing: bool = True
    max_workers: int = 4

@dataclass
class RobustnessReport:
    """Comprehensive robustness testing report"""
    ea_id: str
    ea_name: str
    test_timestamp: datetime
    
    # Overall scores
    overall_robustness_score: float = 0.0
    risk_adjusted_score: float = 0.0
    
    # Test results
    walk_forward_results: Dict[str, Any] = field(default_factory=dict)
    monte_carlo_results: Dict[str, Any] = field(default_factory=dict)
    sensitivity_results: Dict[str, Any] = field(default_factory=dict)
    stress_test_results: Dict[str, Any] = field(default_factory=dict)
    out_of_sample_results: Dict[str, Any] = field(default_factory=dict)
    
    # Statistical measures
    performance_stability: float = 0.0
    parameter_sensitivity_score: float = 0.0
    market_regime_adaptability: float = 0.0
    
    # Risk metrics
    worst_case_drawdown: float = 0.0
    value_at_risk_99: float = 0.0
    expected_shortfall: float = 0.0
    
    # Recommendations
    passed_tests: List[str] = field(default_factory=list)
    failed_tests: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def calculate_overall_score(self) -> float:
        """Calculate overall robustness score (0-100)"""
        scores = []
        weights = []
        
        # Walk-forward score
        if self.walk_forward_results:
            scores.append(self.walk_forward_results.get('stability_score', 0))
            weights.append(0.3)
        
        # Monte Carlo score
        if self.monte_carlo_results:
            scores.append(self.monte_carlo_results.get('consistency_score', 0))
            weights.append(0.25)
        
        # Sensitivity score
        if self.sensitivity_results:
            scores.append(self.parameter_sensitivity_score)
            weights.append(0.2)
        
        # Stress test score
        if self.stress_test_results:
            scores.append(self.stress_test_results.get('average_score', 0))
            weights.append(0.15)
        
        # Out-of-sample score
        if self.out_of_sample_results:
            scores.append(self.out_of_sample_results.get('performance_ratio', 0) * 100)
            weights.append(0.1)
        
        if scores and weights:
            self.overall_robustness_score = np.average(scores, weights=weights)
        
        return self.overall_robustness_score

class BaseRobustnessTest(ABC):
    """Base class for robustness tests"""
    
    def __init__(self, config: RobustnessConfig):
        self.config = config
        self.logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
    
    @abstractmethod
    def run_test(self, ea_template: EATemplate, 
                data: pd.DataFrame,
                backtest_function: Callable,
                optimal_params: Dict[str, Any]) -> Dict[str, Any]:
        """Run the robustness test"""
        pass
    
    def _prepare_data(self, data: pd.DataFrame, 
                     start_date: datetime = None, 
                     end_date: datetime = None) -> pd.DataFrame:
        """Prepare data for testing"""
        if start_date or end_date:
            mask = pd.Series(True, index=data.index)
            if start_date:
                mask &= (data.index >= start_date)
            if end_date:
                mask &= (data.index <= end_date)
            return data[mask].copy()
        return data.copy()

class WalkForwardAnalysis(BaseRobustnessTest):
    """Walk-forward analysis for temporal robustness"""
    
    def run_test(self, ea_template: EATemplate, 
                data: pd.DataFrame,
                backtest_function: Callable,
                optimal_params: Dict[str, Any]) -> Dict[str, Any]:
        """Run walk-forward analysis"""
        
        self.logger.info("Starting walk-forward analysis")
        
        results = {
            'periods': [],
            'in_sample_metrics': [],
            'out_sample_metrics': [],
            'stability_score': 0.0,
            'degradation_factor': 0.0
        }
        
        total_days = len(data)
        if total_days < self.config.wf_window_size + self.config.wf_step_size:
            self.logger.warning("Insufficient data for walk-forward analysis")
            return results
        
        # Calculate walk-forward periods
        start_idx = 0
        period_count = 0
        
        while start_idx + self.config.wf_window_size + self.config.wf_step_size <= total_days:
            # Define in-sample and out-of-sample periods
            in_sample_end = start_idx + self.config.wf_window_size
            out_sample_end = min(in_sample_end + self.config.wf_step_size, total_days)
            
            in_sample_data = data.iloc[start_idx:in_sample_end]
            out_sample_data = data.iloc[in_sample_end:out_sample_end]
            
            period_info = {
                'period': period_count,
                'in_sample_start': data.index[start_idx],
                'in_sample_end': data.index[in_sample_end - 1],
                'out_sample_start': data.index[in_sample_end],
                'out_sample_end': data.index[out_sample_end - 1]
            }
            
            # Run backtests on both periods
            try:
                in_sample_metrics = backtest_function(in_sample_data, optimal_params)
                out_sample_metrics = backtest_function(out_sample_data, optimal_params)
                
                results['periods'].append(period_info)
                results['in_sample_metrics'].append(in_sample_metrics)
                results['out_sample_metrics'].append(out_sample_metrics)
                
                self.logger.info(f"Completed walk-forward period {period_count}")
                
            except Exception as e:
                self.logger.error(f"Error in walk-forward period {period_count}: {e}")
            
            start_idx += self.config.wf_step_size
            period_count += 1
        
        # Calculate stability metrics
        if results['out_sample_metrics']:
            results['stability_score'] = self._calculate_stability_score(results)
            results['degradation_factor'] = self._calculate_degradation_factor(results)
        
        self.logger.info(f"Walk-forward analysis completed: {period_count} periods")
        return results
    
    def _calculate_stability_score(self, results: Dict[str, Any]) -> float:
        """Calculate performance stability score"""
        out_sample_returns = [m.annual_return for m in results['out_sample_metrics'] 
                             if hasattr(m, 'annual_return')]
        
        if len(out_sample_returns) < 2:
            return 0.0
        
        # Calculate coefficient of variation (lower is more stable)
        mean_return = np.mean(out_sample_returns)
        std_return = np.std(out_sample_returns)
        
        if mean_return <= 0:
            return 0.0
        
        cv = std_return / abs(mean_return)
        
        # Convert to score (0-100, where 100 is most stable)
        stability_score = max(0, 100 - (cv * 100))
        return stability_score
    
    def _calculate_degradation_factor(self, results: Dict[str, Any]) -> float:
        """Calculate performance degradation from in-sample to out-of-sample"""
        if not results['in_sample_metrics'] or not results['out_sample_metrics']:
            return 0.0
        
        in_sample_avg = np.mean([m.annual_return for m in results['in_sample_metrics'] 
                                if hasattr(m, 'annual_return')])
        out_sample_avg = np.mean([m.annual_return for m in results['out_sample_metrics'] 
                                 if hasattr(m, 'annual_return')])
        
        if in_sample_avg <= 0:
            return 0.0
        
        degradation = (in_sample_avg - out_sample_avg) / in_sample_avg
        return max(0, degradation)

class MonteCarloSimulation(BaseRobustnessTest):
    """Monte Carlo simulation for statistical robustness"""
    
    def run_test(self, ea_template: EATemplate, 
                data: pd.DataFrame,
                backtest_function: Callable,
                optimal_params: Dict[str, Any]) -> Dict[str, Any]:
        """Run Monte Carlo simulation"""
        
        self.logger.info(f"Starting Monte Carlo simulation with {self.config.mc_simulations} runs")
        
        results = {
            'simulation_results': [],
            'baseline_metrics': None,
            'consistency_score': 0.0,
            'worst_case_metrics': None,
            'best_case_metrics': None,
            'confidence_intervals': {}
        }
        
        # Run baseline backtest
        try:
            baseline_metrics = backtest_function(data, optimal_params)
            results['baseline_metrics'] = baseline_metrics
        except Exception as e:
            self.logger.error(f"Error in baseline backtest: {e}")
            return results
        
        # Run Monte Carlo simulations
        simulation_metrics = []
        
        for sim in range(self.config.mc_simulations):
            try:
                # Apply randomization to data
                perturbed_data = self._apply_monte_carlo_perturbation(data)
                
                # Run backtest on perturbed data
                sim_metrics = backtest_function(perturbed_data, optimal_params)
                simulation_metrics.append(sim_metrics)
                
                if sim % 100 == 0:
                    self.logger.info(f"Completed Monte Carlo simulation {sim}/{self.config.mc_simulations}")
                
            except Exception as e:
                self.logger.error(f"Error in Monte Carlo simulation {sim}: {e}")
                continue
        
        if simulation_metrics:
            results['simulation_results'] = simulation_metrics
            results['consistency_score'] = self._calculate_consistency_score(
                baseline_metrics, simulation_metrics)
            results['confidence_intervals'] = self._calculate_confidence_intervals(simulation_metrics)
            
            # Find worst and best case scenarios
            returns = [m.annual_return for m in simulation_metrics if hasattr(m, 'annual_return')]
            if returns:
                worst_idx = np.argmin(returns)
                best_idx = np.argmax(returns)
                results['worst_case_metrics'] = simulation_metrics[worst_idx]
                results['best_case_metrics'] = simulation_metrics[best_idx]
        
        self.logger.info(f"Monte Carlo simulation completed: {len(simulation_metrics)} valid runs")
        return results
    
    def _apply_monte_carlo_perturbation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply Monte Carlo perturbations to market data"""
        perturbed_data = data.copy()
        
        # Add noise to price data
        if self.config.mc_noise_level > 0:
            for col in ['open', 'high', 'low', 'close']:
                if col in perturbed_data.columns:
                    noise = np.random.normal(0, self.config.mc_noise_level, len(perturbed_data))
                    perturbed_data[col] *= (1 + noise)
        
        # Randomize data order (bootstrap resampling)
        if self.config.mc_randomize_order:
            perturbed_data = perturbed_data.sample(frac=1.0, replace=True).sort_index()
        
        # Add random slippage
        if self.config.mc_randomize_slippage:
            min_slip, max_slip = self.config.mc_slippage_range
            slippage = np.random.uniform(min_slip, max_slip, len(perturbed_data))
            perturbed_data['slippage'] = slippage
        
        return perturbed_data
    
    def _calculate_consistency_score(self, baseline_metrics, simulation_metrics) -> float:
        """Calculate consistency score based on simulation spread"""
        if not hasattr(baseline_metrics, 'annual_return'):
            return 0.0
        
        sim_returns = [m.annual_return for m in simulation_metrics if hasattr(m, 'annual_return')]
        if len(sim_returns) < 10:
            return 0.0
        
        baseline_return = baseline_metrics.annual_return
        
        # Calculate percentage of simulations within reasonable range of baseline
        tolerance = 0.5  # 50% tolerance
        within_range = sum(1 for r in sim_returns 
                          if abs(r - baseline_return) <= tolerance * abs(baseline_return))
        
        consistency_score = (within_range / len(sim_returns)) * 100
        return consistency_score
    
    def _calculate_confidence_intervals(self, simulation_metrics) -> Dict[str, Dict[str, float]]:
        """Calculate confidence intervals for key metrics"""
        intervals = {}
        confidence = self.config.confidence_level
        
        metrics_to_analyze = ['annual_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        
        for metric in metrics_to_analyze:
            values = [getattr(m, metric, 0) for m in simulation_metrics if hasattr(m, metric)]
            if values:
                lower_percentile = (1 - confidence) / 2 * 100
                upper_percentile = (1 + confidence) / 2 * 100
                
                intervals[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'lower': np.percentile(values, lower_percentile),
                    'upper': np.percentile(values, upper_percentile),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return intervals

class ParameterSensitivityAnalysis(BaseRobustnessTest):
    """Parameter sensitivity analysis"""
    
    def run_test(self, ea_template: EATemplate, 
                data: pd.DataFrame,
                backtest_function: Callable,
                optimal_params: Dict[str, Any]) -> Dict[str, Any]:
        """Run parameter sensitivity analysis"""
        
        self.logger.info("Starting parameter sensitivity analysis")
        
        results = {
            'parameter_sensitivity': {},
            'sensitivity_scores': {},
            'robust_parameters': [],
            'sensitive_parameters': [],
            'overall_sensitivity': 0.0
        }
        
        for param in ea_template.parameters:
            param_name = param.name
            optimal_value = optimal_params.get(param_name)
            
            if optimal_value is None:
                continue
            
            self.logger.info(f"Analyzing sensitivity for parameter: {param_name}")
            
            param_results = self._analyze_parameter_sensitivity(
                param, optimal_value, data, backtest_function, optimal_params)
            
            results['parameter_sensitivity'][param_name] = param_results
            
            # Calculate sensitivity score
            sensitivity_score = self._calculate_parameter_sensitivity_score(param_results)
            results['sensitivity_scores'][param_name] = sensitivity_score
            
            # Classify parameter
            if sensitivity_score < 30:  # Low sensitivity threshold
                results['robust_parameters'].append(param_name)
            elif sensitivity_score > 70:  # High sensitivity threshold
                results['sensitive_parameters'].append(param_name)
        
        # Calculate overall sensitivity
        if results['sensitivity_scores']:
            results['overall_sensitivity'] = np.mean(list(results['sensitivity_scores'].values()))
        
        self.logger.info(f"Parameter sensitivity analysis completed")
        return results
    
    def _analyze_parameter_sensitivity(self, param: Parameter, optimal_value: Any,
                                     data: pd.DataFrame, backtest_function: Callable,
                                     base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sensitivity for a single parameter"""
        
        test_values = self._generate_test_values(param, optimal_value)
        sensitivity_results = []
        
        for test_value in test_values:
            # Create modified parameters
            modified_params = base_params.copy()
            modified_params[param.name] = test_value
            
            try:
                # Run backtest with modified parameter
                metrics = backtest_function(data, modified_params)
                
                sensitivity_results.append({
                    'parameter_value': test_value,
                    'metrics': metrics,
                    'return': getattr(metrics, 'annual_return', 0),
                    'sharpe': getattr(metrics, 'sharpe_ratio', 0),
                    'drawdown': getattr(metrics, 'max_drawdown', 0)
                })
                
            except Exception as e:
                self.logger.error(f"Error testing {param.name}={test_value}: {e}")
        
        return {
            'optimal_value': optimal_value,
            'test_values': test_values,
            'results': sensitivity_results
        }
    
    def _generate_test_values(self, param: Parameter, optimal_value: Any) -> List[Any]:
        """Generate test values around optimal value"""
        test_values = []
        
        if param.param_type == ParameterType.INTEGER:
            # Generate integer values around optimal
            range_size = max(1, int((param.max_value - param.min_value) * self.config.sensitivity_range))
            for offset in range(-range_size, range_size + 1, max(1, range_size // self.config.sensitivity_steps)):
                test_val = optimal_value + offset
                if param.min_value <= test_val <= param.max_value:
                    test_values.append(test_val)
        
        elif param.param_type == ParameterType.FLOAT:
            # Generate float values around optimal
            range_size = (param.max_value - param.min_value) * self.config.sensitivity_range
            for i in range(-self.config.sensitivity_steps, self.config.sensitivity_steps + 1):
                offset = (i / self.config.sensitivity_steps) * range_size
                test_val = optimal_value + offset
                if param.min_value <= test_val <= param.max_value:
                    test_values.append(test_val)
        
        elif param.param_type == ParameterType.BOOLEAN:
            test_values = [True, False]
        
        elif param.param_type == ParameterType.CHOICE:
            test_values = param.choices.copy()
        
        return test_values
    
    def _calculate_parameter_sensitivity_score(self, param_results: Dict[str, Any]) -> float:
        """Calculate sensitivity score for a parameter (0-100, higher = more sensitive)"""
        results = param_results.get('results', [])
        if len(results) < 2:
            return 0.0
        
        returns = [r['return'] for r in results]
        return_range = max(returns) - min(returns)
        mean_return = np.mean(returns)
        
        if mean_return == 0:
            return 100.0  # Highly sensitive if mean is zero
        
        # Sensitivity as coefficient of variation
        cv = np.std(returns) / abs(mean_return)
        sensitivity_score = min(100, cv * 100)
        
        return sensitivity_score

class StressTesting(BaseRobustnessTest):
    """Stress testing under adverse market conditions"""
    
    def run_test(self, ea_template: EATemplate, 
                data: pd.DataFrame,
                backtest_function: Callable,
                optimal_params: Dict[str, Any]) -> Dict[str, Any]:
        """Run stress tests"""
        
        self.logger.info("Starting stress testing")
        
        results = {
            'scenario_results': {},
            'average_score': 0.0,
            'worst_scenario': None,
            'stress_resistance_score': 0.0
        }
        
        scenario_scores = []
        
        for scenario in self.config.stress_scenarios:
            self.logger.info(f"Running stress test scenario: {scenario}")
            
            try:
                # Generate stressed data
                stressed_data = self._generate_stress_scenario(data, scenario)
                
                # Run backtest on stressed data
                stressed_metrics = backtest_function(stressed_data, optimal_params)
                
                # Calculate scenario score
                scenario_score = self._calculate_scenario_score(stressed_metrics)
                
                results['scenario_results'][scenario] = {
                    'metrics': stressed_metrics,
                    'score': scenario_score,
                    'description': self._get_scenario_description(scenario)
                }
                
                scenario_scores.append(scenario_score)
                
            except Exception as e:
                self.logger.error(f"Error in stress test scenario {scenario}: {e}")
                scenario_scores.append(0.0)
        
        # Calculate aggregate scores
        if scenario_scores:
            results['average_score'] = np.mean(scenario_scores)
            results['stress_resistance_score'] = min(scenario_scores)  # Worst-case performance
            
            # Find worst scenario
            worst_idx = np.argmin(scenario_scores)
            worst_scenario_name = self.config.stress_scenarios[worst_idx]
            results['worst_scenario'] = worst_scenario_name
        
        self.logger.info("Stress testing completed")
        return results
    
    def _generate_stress_scenario(self, data: pd.DataFrame, scenario: str) -> pd.DataFrame:
        """Generate stressed market data for scenario"""
        stressed_data = data.copy()
        
        if scenario == 'high_volatility':
            # Increase volatility by 300%
            returns = stressed_data['close'].pct_change()
            enhanced_returns = returns * 3.0
            stressed_data['close'] = stressed_data['close'].iloc[0] * (1 + enhanced_returns).cumprod()
            
        elif scenario == 'low_volatility':
            # Reduce volatility by 80%
            returns = stressed_data['close'].pct_change()
            dampened_returns = returns * 0.2
            stressed_data['close'] = stressed_data['close'].iloc[0] * (1 + dampened_returns).cumprod()
            
        elif scenario == 'trending_market':
            # Add strong trend
            trend = np.linspace(0, 0.5, len(stressed_data))  # 50% trend over period
            stressed_data['close'] *= (1 + trend)
            
        elif scenario == 'sideways_market':
            # Create sideways market (detrend)
            ma = stressed_data['close'].rolling(20).mean()
            stressed_data['close'] = ma + (stressed_data['close'] - ma) * 0.1
            
        elif scenario == 'flash_crash':
            # Simulate flash crash events
            crash_points = np.random.choice(len(stressed_data), size=3, replace=False)
            for point in crash_points:
                crash_size = np.random.uniform(0.05, 0.15)  # 5-15% crash
                stressed_data['close'].iloc[point:] *= (1 - crash_size)
                
        elif scenario == 'gap_scenarios':
            # Simulate overnight gaps
            gap_points = np.random.choice(len(stressed_data), size=5, replace=False)
            for point in gap_points:
                gap_size = np.random.uniform(-0.03, 0.03)  # ±3% gaps
                stressed_data['open'].iloc[point:] *= (1 + gap_size)
                stressed_data['close'].iloc[point:] *= (1 + gap_size)
        
        # Recalculate OHLC consistency
        stressed_data = self._ensure_ohlc_consistency(stressed_data)
        
        return stressed_data
    
    def _ensure_ohlc_consistency(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure OHLC data remains consistent after stress modifications"""
        for i in range(len(data)):
            open_val = data['open'].iloc[i]
            close_val = data['close'].iloc[i]
            
            # Adjust high/low to maintain consistency
            data['high'].iloc[i] = max(data['high'].iloc[i], open_val, close_val)
            data['low'].iloc[i] = min(data['low'].iloc[i], open_val, close_val)
        
        return data
    
    def _calculate_scenario_score(self, metrics) -> float:
        """Calculate score for stress scenario (0-100)"""
        if not hasattr(metrics, 'annual_return'):
            return 0.0
        
        # Score based on return and risk-adjusted metrics
        return_score = max(0, min(100, metrics.annual_return * 100 + 50))
        drawdown_score = max(0, 100 + metrics.max_drawdown * 200)  # Less penalty for drawdown
        
        # Weight the components
        scenario_score = 0.6 * return_score + 0.4 * drawdown_score
        return scenario_score
    
    def _get_scenario_description(self, scenario: str) -> str:
        """Get description for stress scenario"""
        descriptions = {
            'high_volatility': 'Market volatility increased by 300%',
            'low_volatility': 'Market volatility reduced by 80%',
            'trending_market': 'Strong trending market conditions',
            'sideways_market': 'Detrended sideways market',
            'flash_crash': 'Multiple flash crash events',
            'gap_scenarios': 'Frequent overnight gaps'
        }
        return descriptions.get(scenario, f'Stress scenario: {scenario}')

class RobustnessTestingEngine:
    """Main robustness testing engine"""
    
    def __init__(self, config: RobustnessConfig = None):
        self.config = config or RobustnessConfig()
        self.test_classes = {
            RobustnessTest.WALK_FORWARD: WalkForwardAnalysis,
            RobustnessTest.MONTE_CARLO: MonteCarloSimulation,
            RobustnessTest.PARAMETER_SENSITIVITY: ParameterSensitivityAnalysis,
            RobustnessTest.STRESS_TESTING: StressTesting,
        }
        
    def run_comprehensive_test(self, ea_template: EATemplate,
                             data: pd.DataFrame,
                             backtest_function: Callable,
                             optimal_params: Dict[str, Any],
                             tests_to_run: List[RobustnessTest] = None) -> RobustnessReport:
        """Run comprehensive robustness testing"""
        
        if tests_to_run is None:
            tests_to_run = list(self.test_classes.keys())
        
        logger.info(f"Starting comprehensive robustness testing for EA: {ea_template.name}")
        
        report = RobustnessReport(
            ea_id=ea_template.id,
            ea_name=ea_template.name,
            test_timestamp=datetime.now()
        )
        
        # Run each test
        for test_type in tests_to_run:
            test_class = self.test_classes.get(test_type)
            if not test_class:
                logger.warning(f"Unknown test type: {test_type}")
                continue
            
            try:
                logger.info(f"Running {test_type.value} test")
                test_instance = test_class(self.config)
                test_results = test_instance.run_test(ea_template, data, backtest_function, optimal_params)
                
                # Store results in report
                if test_type == RobustnessTest.WALK_FORWARD:
                    report.walk_forward_results = test_results
                elif test_type == RobustnessTest.MONTE_CARLO:
                    report.monte_carlo_results = test_results
                elif test_type == RobustnessTest.PARAMETER_SENSITIVITY:
                    report.sensitivity_results = test_results
                    report.parameter_sensitivity_score = test_results.get('overall_sensitivity', 0)
                elif test_type == RobustnessTest.STRESS_TESTING:
                    report.stress_test_results = test_results
                
                report.passed_tests.append(test_type.value)
                
            except Exception as e:
                logger.error(f"Error running {test_type.value} test: {e}")
                report.failed_tests.append(test_type.value)
                report.warnings.append(f"{test_type.value} test failed: {str(e)}")
        
        # Calculate overall scores and generate recommendations
        report.calculate_overall_score()
        self._generate_recommendations(report)
        
        logger.info(f"Robustness testing completed. Overall score: {report.overall_robustness_score:.2f}")
        
        return report
    
    def _generate_recommendations(self, report: RobustnessReport):
        """Generate recommendations based on test results"""
        
        # Performance stability recommendations
        if report.walk_forward_results:
            stability_score = report.walk_forward_results.get('stability_score', 0)
            if stability_score < 50:
                report.recommendations.append(
                    "Strategy shows low temporal stability - consider parameter re-optimization"
                )
        
        # Monte Carlo recommendations
        if report.monte_carlo_results:
            consistency_score = report.monte_carlo_results.get('consistency_score', 0)
            if consistency_score < 70:
                report.recommendations.append(
                    "Strategy shows inconsistent performance under market noise - review entry/exit logic"
                )
        
        # Parameter sensitivity recommendations
        if report.sensitivity_results:
            sensitive_params = report.sensitivity_results.get('sensitive_parameters', [])
            if sensitive_params:
                report.recommendations.append(
                    f"High sensitivity detected in parameters: {', '.join(sensitive_params)} - "
                    "consider robustifying or fixing these parameters"
                )
        
        # Stress testing recommendations
        if report.stress_test_results:
            stress_score = report.stress_test_results.get('stress_resistance_score', 0)
            if stress_score < 30:
                report.recommendations.append(
                    "Strategy performs poorly under stress conditions - "
                    "implement additional risk management controls"
                )
        
        # Overall score recommendations
        if report.overall_robustness_score < 50:
            report.recommendations.append(
                "Overall robustness score is low - strategy requires significant improvement before live trading"
            )
        elif report.overall_robustness_score < 70:
            report.recommendations.append(
                "Moderate robustness detected - proceed with caution and enhanced monitoring"
            )
        else:
            report.recommendations.append(
                "Good robustness characteristics - strategy suitable for live trading with proper risk management"
            )

# Example usage
if __name__ == "__main__":
    from qnti_ea_generation_engine import EAGenerationEngine, Parameter, ParameterType
    
    # Create sample EA and data
    engine = EAGenerationEngine()
    ea = engine.create_ea_template(
        name="Test Robust EA",
        description="EA for robustness testing",
        author="QNTI"
    )
    
    # Mock data and backtest function
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    mock_data = pd.DataFrame({
        'open': np.random.random(len(dates)) * 100 + 100,
        'high': np.random.random(len(dates)) * 100 + 100,
        'low': np.random.random(len(dates)) * 100 + 100,
        'close': np.random.random(len(dates)) * 100 + 100,
        'volume': np.random.random(len(dates)) * 1000000
    }, index=dates)
    
    def mock_backtest(data, params):
        return BacktestMetrics(
            annual_return=random.uniform(-0.3, 0.8),
            sharpe_ratio=random.uniform(-1.0, 2.5),
            max_drawdown=random.uniform(-0.4, -0.02),
            win_rate=random.uniform(0.3, 0.7)
        )
    
    # Run robustness testing
    config = RobustnessConfig(
        mc_simulations=100,  # Reduced for demo
        wf_window_size=252,
        sensitivity_steps=5
    )
    
    testing_engine = RobustnessTestingEngine(config)
    optimal_params = {"test_param": 14}
    
    report = testing_engine.run_comprehensive_test(
        ea, mock_data, mock_backtest, optimal_params
    )
    
    print(f"Robustness testing completed:")
    print(f"Overall score: {report.overall_robustness_score:.2f}")
    print(f"Passed tests: {report.passed_tests}")
    print(f"Failed tests: {report.failed_tests}")
    print(f"Recommendations: {len(report.recommendations)}")